"""Persistent State Filter: task-agnostic cached-hidden state tracking."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

TAPS=(.25,.50,.75,1.0)

def frozen_text(text: str, dim: int=4096) -> torch.Tensor:
    """Deterministic smoke-only fallback when no cached LM text feature exists."""
    seed=int(hashlib.sha256(text.encode()).hexdigest()[:16],16)%(2**32)
    x=np.random.default_rng(seed).standard_normal(dim).astype('float32')
    return torch.from_numpy(x/np.linalg.norm(x))

def tap_layers(x: np.ndarray) -> tuple[np.ndarray,bool]:
    if x.ndim==1: return np.repeat(x[None],4,axis=0),True
    ids=[min(x.shape[0]-1,max(0,round((x.shape[0]-1)*p))) for p in TAPS]
    return x[ids],False

@dataclass
class Trajectory:
    model_name:str; task_name:str; trajectory_id:str; hidden:torch.Tensor
    target_text:str; candidate_texts:list[str]; initial_state_id:int
    state_ids:torch.Tensor; native_ids:torch.Tensor|None=None; schema_fallback:bool=False

class TextFeatureStore:
    """Read-only store for features produced by a frozen LM/VLM text encoder.

    The NPZ uses SHA256(text) as keys.  A deterministic fallback is deliberately
    available for unit/smoke tests, but strict mode makes formal runs fail fast.
    """
    def __init__(self,path: str|Path|None=None,dim:int=4096,strict:bool=False):
        self.path=Path(path) if path else None; self.dim=dim; self.strict=strict
        self.cache=np.load(self.path) if self.path and self.path.exists() else None
        if strict and self.cache is None:
            raise FileNotFoundError(f'Frozen text embedding cache required: {self.path}')
        self.fallback_count=0
    @staticmethod
    def key(text): return hashlib.sha256(text.encode()).hexdigest()
    def get(self,text):
        key=self.key(text)
        if self.cache is not None and key in self.cache:
            x=np.asarray(self.cache[key],dtype='float32').reshape(-1)
            if x.size != self.dim: raise ValueError(f'text feature {key} has dim {x.size}, expected {self.dim}')
            return torch.from_numpy(x)
        if self.strict: raise KeyError(f'Missing frozen text feature for: {text!r}')
        self.fallback_count+=1; return frozen_text(text,self.dim)

class ShellAdapter:
    states=['Left','Middle','Right']
    texts=['the ball is at the left position','the ball is at the middle position','the ball is at the right position']
    def __init__(self,model='qwen',hidden_path=None,behavior_path=None):
        self.model=model
        self.hidden_path=Path(hidden_path or ('outputs/vetbench/hidden_state_probe/hidden_states.npz' if model=='qwen' else 'outputs/vetbench/llava_next_video_7b_replication_v1/hidden_states.npz'))
        self.behavior_path=Path(behavior_path or ('outputs/vetbench/behavior_audit_v2/mechanism_candidates.csv' if model=='qwen' else 'outputs/vetbench/llava_next_video_7b_replication_v1/behavior.csv'))
    def load(self,max_trajectories=0):
        d=pd.read_csv(self.behavior_path); z=np.load(self.hidden_path); have=set(z.files); out=[]
        for tid,g in d.sort_values(['trajectory_id','t']).groupby('trajectory_id'):
            rows=[r for r in g.itertuples() if f'{tid}_t{int(r.t)}' in have]
            if not rows: continue
            hs=[]; fb=False
            for r in rows: q,b=tap_layers(z[f'{tid}_t{int(r.t)}']); hs.append(q); fb|=b
            ids=torch.tensor([self.states.index(str(r.gt_state).title()) for r in rows])
            native=torch.tensor([self.states.index(str(r.state_pred).title()) if str(r.state_pred).title() in self.states else -1 for r in rows])
            out.append(Trajectory(self.model,'shell',str(tid),torch.tensor(np.stack(hs)), 'the tracked ball',self.texts,self.states.index(str(rows[0].initial_state).title()),ids,native,fb))
            if max_trajectories and len(out)>=max_trajectories: break
        return out

class ChessAdapter:
    squares=[f'{f}{r}' for r in range(1,9) for f in 'abcdefgh']+['captured']
    texts=[f'the tracked knight is on {s}' for s in squares[:-1]]+['the tracked knight is captured']
    def __init__(self,model='qwen',root=None):
        self.model=model; self.root=Path(root or ('outputs/metbench_chess/single_piece_tracking_pilot_qwen_v1' if model=='qwen' else 'outputs/metbench_chess/llava_compact_event_replication_v1'))
    def load(self,max_trajectories=0,split='discovery'):
        m=pd.read_csv('outputs/metbench_chess/single_piece_tracking_pilot_qwen_v1/pilot_manifest.csv'); m=m[m.protocol_split.eq(split)]
        name=f'hidden_qwen_{split}.npz' if self.model=='qwen' else f'hidden_{split}.npz'; z=np.load(self.root/name); have=set(z.files); out=[]
        for tid,g in m.sort_values(['game_id','t']).groupby('game_id'):
            rows=[r for r in g.itertuples() if f'{tid}_t{int(r.t)}' in have]
            if not rows: continue
            hs=[]; fb=False
            for r in rows: q,b=tap_layers(z[f'{tid}_t{int(r.t)}']); hs.append(q); fb|=b
            ids=torch.tensor([self.squares.index(str(r.current_state)) for r in rows])
            out.append(Trajectory(self.model,'chess',str(tid),torch.tensor(np.stack(hs)), 'the white knight initially on g1',self.texts,self.squares.index(str(rows[0].initial_square)),ids,None,fb))
            if max_trajectories and len(out)>=max_trajectories: break
        return out

class Projector(nn.Module):
    def __init__(self,hidden_dim=4096,rank=8,d=64,kind='rank8_old'):
        super().__init__(); self.kind=kind
        if kind=='full_linear': self.net=nn.Linear(hidden_dim,d)
        elif kind in ('rank32','rank8_no_norm','rank8_old'): self.net=nn.Sequential(nn.Linear(hidden_dim,32 if kind=='rank32' else 8,bias=False),nn.Linear(32 if kind=='rank32' else 8,d))
        else: raise ValueError(f'unknown projector {kind}')
    def forward(self,x):
        y=self.net(x)
        return F.normalize(y,dim=-1) if self.kind=='rank8_old' else y

class PSF(nn.Module):
    def __init__(self,backbones=('qwen',),hidden_dim=4096,d=64,rank=8,tau=.1,ablation='full',text_stores=None,projector_kind='rank8_old'):
        super().__init__(); self.d=d; self.tau=tau; self.ablation=ablation
        self.projector_kind=projector_kind
        self.projectors=nn.ModuleDict({b:Projector(hidden_dim,rank,d,projector_kind) for b in backbones})
        # Candidate-state text features have a separate adapter; visual and
        # text geometry must not be forced through one learned projection.
        self.state_text_projectors=nn.ModuleDict({b:Projector(hidden_dim,rank,d,'rank8_no_norm') for b in backbones})
        self.change_weights=nn.Parameter(torch.zeros(4)); self.obs_weights=nn.Parameter(torch.zeros(4))
        self.change=nn.Sequential(nn.Linear(d*2,d),nn.Tanh()); self.obs=nn.Sequential(nn.Linear(d*2,d),nn.Tanh())
        self.gru=nn.GRUCell(d,d); self.initial=nn.Linear(d,d); self.change_head=nn.Linear(d,1)
        self.gate=nn.Sequential(nn.Linear(d*3,16),nn.ReLU(),nn.Linear(16,1))
        self.text_stores=text_stores or {b:TextFeatureStore(dim=hidden_dim) for b in backbones}
    def mix(self,x,w):
        # Accept [layers,d] for one step or [batch,layers,d].
        layer_dim=x.ndim-2
        if self.ablation=='single_layer': return x.select(layer_dim,x.shape[layer_dim]-1)
        shape=[1]*x.ndim; shape[layer_dim]=len(w)
        return (x*torch.softmax(w,0).view(shape)).sum(layer_dim)
    def encode_texts(self,texts,model,device):
        store=self.text_stores[model]
        return self.state_text_projectors[model](torch.stack([store.get(x).to(device) for x in texts]))
    def forward_trajectory(self,tr:Trajectory):
        dev=next(self.parameters()).device; x=tr.hidden.to(dev); proj=self.projectors[tr.model_name](x)
        target=self.encode_texts([tr.target_text],tr.model_name,dev)[0]; states=self.encode_texts(tr.candidate_texts,tr.model_name,dev)
        m=F.normalize(self.initial(states[tr.initial_state_id]),dim=-1); prev=None; logits=[]; memories=[]; changes=[]; gates=[]
        for xt in proj:
            ch=self.mix(xt,self.change_weights); ob=self.mix(xt,self.obs_weights)
            delta=torch.zeros_like(ch) if prev is None else ch-prev; e=self.change(torch.cat([delta,target])); pred=self.gru(e,m)
            o=F.normalize(self.obs(torch.cat([ob,target])),dim=-1)
            lam=torch.sigmoid(self.gate(torch.cat([pred,o,(pred-o).abs()])))
            if self.ablation=='memory_only': lam=lam*0
            elif self.ablation=='observation_only': lam=lam*0+1
            elif self.ablation in ('fixed_fusion','fixed_gate'): lam=lam*0+.5
            m=F.normalize((1-lam)*pred+lam*o,dim=-1); logits.append(m@states.T/self.tau); memories.append(m); changes.append(self.change_head(e).squeeze()); gates.append(lam.squeeze()); prev=ch
        return {'logits':torch.stack(logits),'memory':torch.stack(memories),'change_logits':torch.stack(changes),'gate':torch.stack(gates)}
    def parameter_counts(self):
        proj={k:sum(p.numel() for p in v.parameters()) for k,v in self.projectors.items()}; txt={k:sum(p.numel() for p in v.parameters()) for k,v in self.state_text_projectors.items()}; total=sum(p.numel() for p in self.parameters()); return {'shared_core':total-sum(proj.values())-sum(txt.values()),'projectors':proj,'state_text_projectors':txt,'total_trainable':total}

    def text_feature_stats(self):
        return {k:{'cache':str(v.path) if v.path else None,'strict':v.strict,'fallback_lookups':v.fallback_count} for k,v in self.text_stores.items()}


class SemanticStateDeltaPSF(nn.Module):
    """PSF variant whose update is supervised by adjacent semantic states.

    No event vocabulary is represented here: both the binary change target and
    the vector target are derived solely from consecutive state ids.  Visual
    projections are backbone-specific, while the change encoder, delta updater
    and correction gate are deliberately shared across tasks/backbones.
    """
    def __init__(self, backbones=('qwen',), hidden_dim=4096, d=64, rank=8,
                 tau=.1, text_stores=None, updater='additive', projector_kind='rank8_no_norm'):
        super().__init__()
        if updater not in ('additive', 'residual'):
            raise ValueError('updater must be additive or residual')
        self.d, self.tau, self.updater, self.projector_kind = d, tau, updater, projector_kind
        self.projectors=nn.ModuleDict({b:Projector(hidden_dim,rank,d,projector_kind) for b in backbones})
        self.state_text_projectors=nn.ModuleDict({b:Projector(hidden_dim,rank,d,'rank8_no_norm') for b in backbones})
        # The MLP sees all taps for current, previous, and visual delta. This
        # is a learned mixer rather than a fixed choice of representation.
        self.change_encoder=nn.Sequential(nn.Linear(13*d, 2*d), nn.GELU(), nn.Linear(2*d,d))
        self.observation_mixer=nn.Sequential(nn.Linear(5*d,d), nn.GELU(), nn.Linear(d,d))
        self.residual_updater=nn.Sequential(nn.Linear(2*d,2*d),nn.GELU(),nn.Linear(2*d,d))
        self.initial=nn.Linear(d,d)
        self.change_head=nn.Linear(d,1)
        self.gate=nn.Sequential(nn.Linear(d*3,32),nn.ReLU(),nn.Linear(32,1))
        self.text_stores=text_stores or {b:TextFeatureStore(dim=hidden_dim) for b in backbones}

    def encode_texts(self,texts,model,device):
        store=self.text_stores[model]
        return self.state_text_projectors[model](torch.stack([store.get(x).to(device) for x in texts]))

    def forward_trajectory(self,tr:Trajectory):
        dev=next(self.parameters()).device
        # [time,tap,d]; retain each tap for the change mixer.
        h=self.projectors[tr.model_name](tr.hidden.to(dev))
        target=self.encode_texts([tr.target_text],tr.model_name,dev)[0]
        states=self.encode_texts(tr.candidate_texts,tr.model_name,dev)
        m=F.normalize(self.initial(states[tr.initial_state_id]),dim=-1)
        logits=[]; memories=[]; deltas=[]; changes=[]; gates=[]; predictions=[]; observations=[]
        prev_h=None
        for t,ht in enumerate(h):
            # No pre-action cache exists for step one.  Reusing H_t makes the
            # delta channel zero there but preserves H_t and entity context.
            hp=ht if prev_h is None else prev_h
            visual_delta=ht-hp
            inp=torch.cat([ht.flatten(),hp.flatten(),visual_delta.flatten(),target])
            delta=self.change_encoder(inp)
            m_pred=m+delta if self.updater=='additive' else m+self.residual_updater(torch.cat([m,delta]))
            o=F.normalize(self.observation_mixer(torch.cat([ht.flatten(),target])),dim=-1)
            lam=torch.sigmoid(self.gate(torch.cat([m_pred,o,(m_pred-o).abs()])))
            m=F.normalize((1-lam)*m_pred+lam*o,dim=-1)
            logits.append(m@states.T/self.tau); memories.append(m); deltas.append(delta)
            changes.append(self.change_head(delta).squeeze()); gates.append(lam.squeeze())
            predictions.append(m_pred); observations.append(o); prev_h=ht
        return {'logits':torch.stack(logits),'memory':torch.stack(memories),
                'delta_pred':torch.stack(deltas),'change_logits':torch.stack(changes),
                'gate':torch.stack(gates),'m_pred':torch.stack(predictions),
                'observation':torch.stack(observations),'state_embeddings':states}

    def parameter_counts(self):
        proj={k:sum(p.numel() for p in v.parameters()) for k,v in self.projectors.items()}
        txt={k:sum(p.numel() for p in v.parameters()) for k,v in self.state_text_projectors.items()}
        total=sum(p.numel() for p in self.parameters())
        return {'shared_core':total-sum(proj.values())-sum(txt.values()),'projectors':proj,
                'state_text_projectors':txt,'total_trainable':total}


def state_delta_loss(model, tr, delta_kind='cosine', w_delta=1., w_change=.2,
                     w_persist=.05, changed_weight=3.):
    """CE + semantic delta + derived change BCE + persistence, without events."""
    o=model.forward_trajectory(tr); dev=o['logits'].device; y=tr.state_ids.to(dev)
    prev_ids=torch.cat([torch.tensor([tr.initial_state_id],device=dev),y[:-1]])
    changed=y.ne(prev_ids); weights=1+(changed_weight-1)*changed.float()
    ls=(F.cross_entropy(o['logits'],y,reduction='none')*weights).mean()
    z=o['state_embeddings']; delta_gt=z[y]-z[prev_ids]
    pred=o['delta_pred']; unchanged=~changed
    if changed.any():
        if delta_kind=='cosine':
            changed_loss=(1-F.cosine_similarity(pred[changed],delta_gt[changed],dim=-1)).mean()
        elif delta_kind=='smoothl1':
            # Normalization compares direction/relative geometry and avoids
            # text-projector scale becoming a degenerate loss shortcut.
            changed_loss=F.smooth_l1_loss(F.normalize(pred[changed],dim=-1),F.normalize(delta_gt[changed],dim=-1))
        else: raise ValueError('delta_kind must be cosine or smoothl1')
    else: changed_loss=pred.sum()*0
    unchanged_loss=pred[unchanged].norm(dim=-1).mean() if unchanged.any() else pred.sum()*0
    ld=changed_loss+unchanged_loss
    pos=float(unchanged.sum())/max(float(changed.sum()),1.)
    lc=F.binary_cross_entropy_with_logits(o['change_logits'],changed.float(),pos_weight=torch.tensor(pos,device=dev))
    same=unchanged[1:]
    lp=((o['memory'][1:]-o['memory'][:-1]).square().sum(-1)[same].mean() if same.any() else ls*0)
    loss=ls+w_delta*ld+w_change*lc+w_persist*lp
    return loss,{'state':float(ls.detach()),'delta':float(ld.detach()),'delta_changed':float(changed_loss.detach()),
                 'delta_unchanged':float(unchanged_loss.detach()),'change':float(lc.detach()),'persist':float(lp.detach())}


class KeepSetPSF(nn.Module):
    """Task-agnostic KEEP/SET tracker trained from adjacent state ids only."""
    def __init__(self, backbones=('qwen',), hidden_dim=4096, d=64, tau=.1,
                 text_stores=None, fusion='fixed'):
        super().__init__()
        if fusion not in ('fixed','learned'): raise ValueError('fusion must be fixed or learned')
        self.d,self.tau,self.fusion=d,tau,fusion
        # Deliberately full, unnormalised 4096 -> 64 adapters for this v5 path.
        self.projectors=nn.ModuleDict({b:nn.Linear(hidden_dim,d) for b in backbones})
        self.state_text_projectors=nn.ModuleDict({b:Projector(hidden_dim,d,d,'rank8_no_norm') for b in backbones})
        # Shared modules never receive a task id or task-specific candidate count.
        # A 48-wide shared core keeps even a two-backbone instantiation under
        # the 0.01% of 7B parameter budget while retaining full input access.
        core_width=48
        self.update_detector=nn.Sequential(nn.Linear(13*d,core_width),nn.GELU(),nn.Linear(core_width,1))
        self.set_selector=nn.Sequential(nn.Linear(14*d,core_width),nn.GELU(),nn.Linear(core_width,d))
        self.observation_encoder=nn.Sequential(nn.Linear(5*d,core_width),nn.GELU(),nn.Linear(core_width,d))
        # entropy(track/direct), max(track/direct), and distribution agreement
        self.fusion_gate=nn.Sequential(nn.Linear(5,16),nn.ReLU(),nn.Linear(16,1))
        self.text_stores=text_stores or {b:TextFeatureStore(dim=hidden_dim) for b in backbones}

    def encode_texts(self,texts,model,device):
        return self.state_text_projectors[model](torch.stack([self.text_stores[model].get(x).to(device) for x in texts]))

    def _scores(self,q,states): return q@states.T/(self.d**.5*self.tau)
    @staticmethod
    def _entropy(p): return -(p.clamp_min(1e-8)*p.clamp_min(1e-8).log()).sum(-1)

    def forward_trajectory(self,tr:Trajectory, teacher_prob:float=0.0):
        """Run one trajectory.

        ``teacher_prob`` is a training-only exposure-bias control: at each
        step the recurrent belief is mixed with the ground-truth previous
        state before KEEP/SET is applied.  The default remains fully free
        rollout, so existing evaluation callers are unchanged.
        """
        dev=next(self.parameters()).device; h=self.projectors[tr.model_name](tr.hidden.to(dev))
        target=self.encode_texts([tr.target_text],tr.model_name,dev)[0]
        states=self.encode_texts(tr.candidate_texts,tr.model_name,dev); n=len(states)
        track=F.one_hot(torch.tensor(tr.initial_state_id,device=dev),n).float()
        y=tr.state_ids.to(dev)
        prev_ids=torch.cat([torch.tensor([tr.initial_state_id],device=dev),y[:-1]])
        outputs={k:[] for k in ('update_logits','set_logits','direct_logits','track_probs','final_probs','gate')}
        prev_h=None
        for t,ht in enumerate(h):
            hp=ht if prev_h is None else prev_h; hd=ht-hp
            common=torch.cat([ht.flatten(),hp.flatten(),hd.flatten(),target])
            update_logit=self.update_detector(common).squeeze()
            # Ground-truth belief is used only when explicitly requested by
            # the training curriculum; validation always uses the default 0.
            if teacher_prob > 0:
                teacher=F.one_hot(prev_ids[t],n).float()
                use_teacher=torch.rand((),device=dev) < teacher_prob
                track_in=torch.where(use_teacher,teacher,track)
            else:
                track_in=track
            belief_emb=track_in@states
            q=self.set_selector(torch.cat([common,belief_emb]))
            set_logits=self._scores(q,states); new=F.softmax(set_logits,dim=-1)
            p_update=torch.sigmoid(update_logit); track=(1-p_update)*track_in+p_update*new
            direct_logits=self._scores(self.observation_encoder(torch.cat([ht.flatten(),target])),states)
            direct=F.softmax(direct_logits,dim=-1)
            if self.fusion=='fixed': alpha=track.new_tensor(.5)
            else:
                stats=torch.stack([self._entropy(track),self._entropy(direct),track.max(),direct.max(),(track*direct).sum()])
                alpha=torch.sigmoid(self.fusion_gate(stats)).squeeze()
            final=alpha*track+(1-alpha)*direct
            for key,val in [('update_logits',update_logit),('set_logits',set_logits),('direct_logits',direct_logits),
                            ('track_probs',track),('final_probs',final),('gate',alpha)]: outputs[key].append(val)
            prev_h=ht
        outputs={k:torch.stack(v) for k,v in outputs.items()}; outputs['state_embeddings']=states
        return outputs

    def parameter_counts(self):
        p={k:sum(x.numel() for x in v.parameters()) for k,v in self.projectors.items()}
        t={k:sum(x.numel() for x in v.parameters()) for k,v in self.state_text_projectors.items()}
        total=sum(x.numel() for x in self.parameters())
        return {'per_backbone_projector':p,'state_text_projector':t,
                'shared_core':total-sum(p.values())-sum(t.values()),'total_trainable':total}


def keep_set_loss(model,tr,teacher_prob:float=0.0,pos_weight:float|None=None):
    """Free-rollout losses: no event field or GT previous belief is consumed."""
    o=model.forward_trajectory(tr,teacher_prob=teacher_prob); dev=o['final_probs'].device; y=tr.state_ids.to(dev)
    prev=torch.cat([torch.tensor([tr.initial_state_id],device=dev),y[:-1]]); changed=y.ne(prev)
    lt=F.nll_loss(o['final_probs'].clamp_min(1e-8).log(),y)
    ld=F.cross_entropy(o['direct_logits'],y)
    pos=float((~changed).sum())/max(float(changed.sum()),1.) if pos_weight is None else float(pos_weight)
    lu=F.binary_cross_entropy_with_logits(o['update_logits'],changed.float(),pos_weight=torch.tensor(pos,device=dev))
    ls=F.cross_entropy(o['set_logits'][changed],y[changed]) if changed.any() else o['set_logits'].sum()*0
    return lt+lu+ls+.5*ld,{'track':float(lt.detach()),'update':float(lu.detach()),'set':float(ls.detach()),'direct':float(ld.detach())}

def loss_for(model,tr,w_change=.2,w_persist=.05,changed_lambda=1.0):
    o=model.forward_trajectory(tr); y=tr.state_ids.to(o['logits'].device); ls=F.cross_entropy(o['logits'],y)
    prev=torch.cat([torch.tensor([tr.initial_state_id],device=y.device),y[:-1]]); change=y.ne(prev).float(); lc=F.binary_cross_entropy_with_logits(o['change_logits'],change)
    changed=y.ne(prev); weights=1.0+(changed_lambda-1.0)*changed.float()
    ls=(F.cross_entropy(o['logits'],y,reduction='none')*weights).mean()
    pos=float((~changed).sum())/max(float(changed.sum()),1.0)
    lc=F.binary_cross_entropy_with_logits(o['change_logits'],change, pos_weight=torch.tensor(pos,device=y.device))
    same=~changed; lp=((o['memory'][1:]-o['memory'][:-1])**2).sum(-1)[same[1:]].mean() if same[1:].any() else ls*0
    if model.ablation=='no_change_loss': w_change=0
    if model.ablation=='no_persistence_loss': w_persist=0
    return ls+w_change*lc+w_persist*lp,{'state':float(ls.detach()),'change':float(lc.detach()),'persist':float(lp.detach())}

def evaluate(model,data):
    rows=[]
    model.eval()
    with torch.no_grad():
        for tr in data:
            o=model.forward_trajectory(tr); pred=o['logits'].argmax(-1).cpu(); y=tr.state_ids
            for t,(p,g) in enumerate(zip(pred,y),1): rows.append({'task':tr.task_name,'model':tr.model_name,'trajectory_id':tr.trajectory_id,'t':t,'gt':int(g),'psf_pred':int(p),'correct':int(p==g),'gate':float(o['gate'][t-1])})
    return pd.DataFrame(rows)

def evaluate_ablations(model,data):
    """Uniform baseline table using one frozen checkpoint (no retraining)."""
    rows=[]
    for name in ('full','memory_only','observation_only','fixed_fusion'):
        clone=PSF(tuple(model.projectors.keys()),d=model.d,ablation=name,
                  text_stores=model.text_stores,projector_kind=model.projector_kind).to(next(model.parameters()).device)
        clone.load_state_dict(model.state_dict(),strict=True); rows.append(evaluate(clone,data).assign(method=name))
    return pd.concat(rows,ignore_index=True)

def split_grouped(data,seed=17,dev_frac=.2):
    rng=np.random.default_rng(seed); train=[]; dev=[]
    # Stratify by task/backbone while keeping entire trajectory groups intact.
    for task,model in sorted({(x.task_name,x.model_name) for x in data}):
        q=[x for x in data if x.task_name==task and x.model_name==model]; rng.shuffle(q)
        n=max(1,round(len(q)*dev_frac)) if len(q)>1 else 1
        dev+=q[:n]; train+=q[n:]
    return train,dev

def train(model,train_data,dev_data,epochs=20,lr=1e-3,seed=17,changed_lambda=1.0):
    torch.manual_seed(seed); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); history=[]; best=None; bad=0
    tasks=sorted({x.task_name for x in train_data}); by={t:[x for x in train_data if x.task_name==t] for t in tasks}; rng=np.random.default_rng(seed)
    for ep in range(epochs):
        model.train(); seq=[]; n=max(map(len,by.values()))
        for i in range(n):
            order=tasks.copy(); rng.shuffle(order)
            seq += [by[t][i%len(by[t])] for t in order]
        losses=[]
        for tr in seq:
            opt.zero_grad(); loss,parts=loss_for(model,tr,changed_lambda=changed_lambda); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach()))
        dv=evaluate(model,dev_data); metric=float(dv.correct.mean()); history.append({'epoch':ep+1,'train_loss':float(np.mean(losses)),'dev_accuracy':metric})
        if best is None or metric>best[0]: best=(metric,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}); bad=0
        else: bad+=1
        if bad>=5: break
    model.load_state_dict(best[1]); return pd.DataFrame(history)

def oracle_transition_rows(data):
    """Explicit-event oracle upper bound, isolated from PSF training.

    Events are reconstructed from adjacent state annotations solely for this
    reported upper bound. No event field is exposed to or consumed by PSF.
    """
    rows=[]
    for tr in data:
        prev=tr.initial_state_id
        for t,gt in enumerate(tr.state_ids.tolist(),1):
            event='unchanged' if gt==prev else ('captured' if gt==len(tr.candidate_texts)-1 and tr.task_name=='chess' else 'moved')
            pred=prev if event=='unchanged' else gt
            rows.append({'task':tr.task_name,'model':tr.model_name,'trajectory_id':tr.trajectory_id,'t':t,'gt':gt,'prediction':pred,'correct':int(pred==gt),'method':'explicit_event_oracle'})
            prev=pred
    return pd.DataFrame(rows)

def native_state_rows(data):
    rows=[]
    for tr in data:
        if tr.native_ids is None: continue
        for t,(gt,pred) in enumerate(zip(tr.state_ids.tolist(),tr.native_ids.tolist()),1):
            rows.append({'task':tr.task_name,'model':tr.model_name,'trajectory_id':tr.trajectory_id,
                         't':t,'gt':gt,'prediction':pred,'correct':int(pred==gt),'parse_success':int(pred>=0),'method':'native_vlm'})
    return pd.DataFrame(rows,columns=['task','model','trajectory_id','t','gt','prediction','correct','parse_success','method'])
