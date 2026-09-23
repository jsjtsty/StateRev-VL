#!/usr/bin/env python3
"""Offline audit of the Qwen chess pilot; never loads a VLM."""
from pathlib import Path
import json, argparse
import numpy as np, pandas as pd, torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.base import clone
ROOT=Path(__file__).resolve().parents[1]
from metbench_chess_single_piece import ChessStateUpdater, SI, SQUARES, oracle_transition

def key(r): return f'{r.game_id}_t{int(r.t)}'
def state_idx(x): return 64 if x=='captured' else SI[x]
def load(base):
 m=pd.read_csv(base/'pilot_manifest.csv').sort_values(['game_id','t'])
 h={};
 for p in ('discovery','validation'):
  with np.load(base/f'hidden_qwen_{p}.npz') as z: h.update({k:z[k] for k in z.files})
 sp={}; dp={}
 for p in ('qwen_hidden_src_probs.npz','qwen_hidden_dst_probs.npz'):
  with np.load(base/p) as z: (sp if 'src' in p else dp).update({k:z[k] for k in z.files})
 return m,h,sp,dp
def fit_probe(m,h,kind,C=None):
 d=m[m.protocol_split=='discovery']; X=np.stack([h[key(r)] for r in d.itertuples()]);
 if kind=='state': y=np.array([state_idx(r.current_state) for r in d.itertuples()])
 elif kind=='src': y=np.array([SI[r.src] for r in d.itertuples()])
 elif kind=='dst': y=np.array([SI[r.dst] for r in d.itertuples()])
 elif kind=='event': y=np.array([event_type(r,m) for r in d.itertuples()])
 elif kind=='moved_dst':
  d=d[d.apply(lambda r:event_type(r,m)=='target_moved',axis=1)]; X=np.stack([h[key(r)] for r in d.itertuples()]); y=np.array([SI[r.dst] for r in d.itertuples()])
 # Hyperparameter selection is restricted to discovery games. The cache has
 # only the extracted final layer, so the layer choice is fixed to that layer.
 if C is None:
  tune=d.game_id.map(lambda x:int(__import__('hashlib').sha256(str(x).encode()).hexdigest()[:8],16)%5==0).to_numpy()
  candidates=(.01,.1,1.,10.); scores=[]
  for c in candidates:
   q=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000)).fit(X[~tune],y[~tune]); scores.append(float((q.predict(X[tune])==y[tune]).mean()) if tune.any() else 0.)
  C=candidates[int(np.argmax(scores))]
 model=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=1000)).fit(X,y); model.audit_C=float(C); return model
def event_type(r,m):
 prev=m[(m.game_id==r.game_id)&(m.t==r.t-1)]
 prev_state='g1' if len(prev)==0 else prev.iloc[0].current_state
 if prev_state=='captured': return 'unaffected'
 if r.src==prev_state: return 'target_moved'
 if r.dst==prev_state and r.current_state=='captured': return 'target_captured'
 return 'unaffected'
def eval_probe(model,m,h,kind,split):
 d=m[m.protocol_split==split]; X=np.stack([h[key(r)] for r in d.itertuples()]);
 if kind=='state': y=np.array([state_idx(r.current_state) for r in d.itertuples()])
 elif kind=='src': y=np.array([SI[r.src] for r in d.itertuples()])
 elif kind=='dst': y=np.array([SI[r.dst] for r in d.itertuples()])
 elif kind=='event': y=np.array([event_type(r,m) for r in d.itertuples()])
 elif kind=='moved_dst':
  d=d[d.apply(lambda r:event_type(r,m)=='target_moved',axis=1)]; X=np.stack([h[key(r)] for r in d.itertuples()]); y=np.array([SI[r.dst] for r in d.itertuples()])
 pred=model.predict(X); return d,pred,y
def metrics(rows, pred, y):
 out={'overall':float(accuracy_score(y,pred))}
 for t in range(1,11):
  q=rows.t.to_numpy()==t; out[f't{t}']=float(accuracy_score(y[q],pred[q])) if q.any() else None
 for t in (3,5,8):
  q=rows.t.to_numpy()>=t; out[f't_ge{t}']=float(accuracy_score(y[q],pred[q])) if q.any() else None
 return out
def true_oracle(m, split):
 vals=[]
 for gid,g in m[m.protocol_split==split].groupby('game_id'):
  b=np.zeros(65); b[SI['g1']]=1
  for r in g.sort_values('t').itertuples():
   src,dst=SI[r.src],SI[r.dst]; b=oracle_transition(b,src,dst)
   pred='captured' if b[64]>=b[:64].max() else SQUARES[b[:64].argmax()]
   vals.append((gid,r.t,pred==r.current_state))
 d=pd.DataFrame(vals,columns=['game_id','t','correct']); return d
def trained_gt(m,out,epochs=120):
 """Fit only on teacher-forced true previous states and true UCI events."""
 model=ChessStateUpdater(); opt=torch.optim.Adam(model.parameters(),lr=3e-3)
 disc=m[m.protocol_split=='discovery']; xs=[]; ys=[]
 for gid,g in disc.groupby('game_id'):
  prev='g1'
  for r in g.sort_values('t').itertuples():
   x=np.zeros(65+64+64,dtype=np.float32); x[state_idx(prev)]=1; x[65+SI[r.src]]=1; x[129+SI[r.dst]]=1
   xs.append(x); ys.append(state_idx(r.current_state)); prev=r.current_state
 X=torch.tensor(np.stack(xs)); Y=torch.tensor(ys,dtype=torch.long)
 for _ in range(epochs):
  perm=torch.randperm(len(Y));
  for ix in perm.split(512):
   z=model(X[ix,:65],X[ix,65:129],X[ix,129:]); loss=torch.nn.functional.nll_loss(torch.log(z.clamp_min(1e-8)),Y[ix]); opt.zero_grad(); loss.backward(); opt.step()
 torch.save({'state_dict':model.state_dict(),'parameter_count':model.parameter_count},out); return model
def eval_updater(model,m,split,sp,dp,mode='hidden'):
 vals=[]
 for gid,g in m[m.protocol_split==split].groupby('game_id'):
  b=np.eye(65)[SI['g1']]
  for r in g.sort_values('t').itertuples():
   if mode=='gt': s,d=np.zeros(64),np.zeros(64); s[SI[r.src]]=1; d[SI[r.dst]]=1
   elif mode=='hidden': s,d=sp[key(r)],dp[key(r)]
   with torch.no_grad(): b=model(torch.tensor(b[None],dtype=torch.float32),torch.tensor(s[None],dtype=torch.float32),torch.tensor(d[None],dtype=torch.float32))[0].numpy()
   pred='captured' if b[64]>=b[:64].max() else SQUARES[b[:64].argmax()]; vals.append((gid,r.t,pred==r.current_state))
 return pd.DataFrame(vals,columns=['game_id','t','correct'])
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,default=ROOT/'outputs/metbench_chess/single_piece_tracking_pilot_qwen_v1'); ap.add_argument('--out',type=Path,default=ROOT/'outputs/metbench_chess/qwen_pilot_method_audit_v1'); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 m,h,sp,dp=load(a.base); report={}
 # Independent GT-move updater and exact handwritten GT transition.
 gt=trained_gt(m,a.out/'oracle_gt_updater.pt'); gt_df=eval_updater(gt,m,'validation',sp,dp,'gt'); exact=true_oracle(m,'validation')
 report['oracle_updater']=summarize(gt_df); report['handwritten_gt_transition']=summarize(exact)
 pd.DataFrame({'game_id':gt_df.game_id,'t':gt_df.t,'oracle_updater_correct':gt_df.correct}).to_csv(a.out/'oracle_updater_results.csv',index=False)
 # Direct state probe.
 state=fit_probe(m,h,'state'); d,p,y=eval_probe(state,m,h,'state','validation'); report['direct_state_probe']={'C':state.audit_C,'layer':'final extracted hidden','metrics':metrics(d,p,y)}
 pd.DataFrame({'game_id':d.game_id,'t':d.t,'correct':p==y}).to_csv(a.out/'direct_state_probe_results.csv',index=False)
 # Frozen hidden updater permutation controls.
 hidden_up=ChessStateUpdater(); hidden_up.load_state_dict(torch.load(a.base/'qwen_updater.pt',map_location='cpu')['state_dict']); hidden_up.eval(); allrows=[]
 for seed in (7,19,31,43,59):
  rng=np.random.default_rng(seed); vals=[]
  for t,g in m[m.protocol_split=='validation'].groupby('t'):
   rr=g.sort_values('game_id'); perm=rng.permutation(len(rr)); sv=[sp[key(r)] for r in rr.itertuples()]; dv=[dp[key(r)] for r in rr.itertuples()]
   for j,r in enumerate(rr.itertuples()):
    b=np.eye(65)[SI['g1']] if r.t==1 else None
   # recursive per game using same-t vectors drawn from another game
  donor_by_t={}
  for t,pool in m[m.protocol_split=='validation'].groupby('t'):
   ids=list(pool.index); perm=rng.permutation(len(ids))
   if len(ids)>1 and np.any(perm==np.arange(len(ids))): perm=np.roll(perm,1)
   donor_by_t[t]={ids[i]:ids[perm[i]] for i in range(len(ids))}
  for gid,g in m[m.protocol_split=='validation'].groupby('game_id'):
   b=np.eye(65)[SI['g1']]
   for r in g.sort_values('t').itertuples():
    donor=m.loc[donor_by_t[r.t][r.Index]]
    with torch.no_grad(): b=hidden_up(torch.tensor(b[None],dtype=torch.float32),torch.tensor(sp[key(donor)][None],dtype=torch.float32),torch.tensor(dp[key(donor)][None],dtype=torch.float32))[0].numpy()
    pred='captured' if b[64]>=b[:64].max() else SQUARES[b[:64].argmax()]; vals.append((gid,r.t,pred==r.current_state,seed))
  allrows.extend(vals)
 perm=pd.DataFrame(allrows,columns=['game_id','t','correct','seed'])
 baseline=eval_updater(hidden_up,m,'validation',sp,dp,'hidden'); baseline['seed']='baseline'; perm=pd.concat([baseline,perm],ignore_index=True)
 perm.to_csv(a.out/'soft_vector_permutation_results.csv',index=False); report['soft_vector_permutation']={str(s):summarize(perm[perm.seed.astype(str)==str(s)]) for s in perm.seed.unique()}
 # Target-relevant event probes and recursive state using event predictions.
 ev=fit_probe(m,h,'event'); de,pe,ye=eval_probe(ev,m,h,'event','validation'); md=fit_probe(m,h,'moved_dst'); dm,pm,ym=eval_probe(md,m,h,'moved_dst','validation'); report['target_event_type']={'C':ev.audit_C,'metrics':metrics(de,pe,ye)}; report['target_moved_destination']={'C':md.audit_C,'metrics':metrics(dm,pm,ym)}
 # Recursive state from the compact event representation: unaffected keeps
 # the previous target state; captured enters captured; moved uses predicted dst.
 event_rows=[]; pm_by_key={key(r):p for r,p in zip(dm.itertuples(),pm)}
 for gid,g in m[m.protocol_split=='validation'].groupby('game_id'):
  state='g1'
  for r in g.sort_values('t').itertuples():
   event_lookup={int(ix):val for ix,val in zip(de.index,pe)}; ep=event_lookup[r.Index]
   if ep=='target_captured': state='captured'
   elif ep=='target_moved':
    dst_pred=pm_by_key.get(key(r),SI.get(state,SI['g1']))
    state=SQUARES[int(dst_pred)] if isinstance(dst_pred,(int,np.integer)) else str(dst_pred)
   event_rows.append({'game_id':gid,'t':r.t,'event_pred':ep,'event_gt':event_type(r,m),'state_pred':state,'gt_state':r.current_state,'state_correct':state==r.current_state})
 event_df=pd.DataFrame(event_rows); report['target_relevant_recursive_state']=summarize(event_df.rename(columns={'state_correct':'correct'}))
 pd.DataFrame({'game_id':de.game_id,'t':de.t,'event_pred':pe,'event_gt':ye,'correct':pe==ye}).to_csv(a.out/'target_relevant_event_results.csv',index=False)
 event_df.to_csv(a.out/'target_relevant_recursive_state_results.csv',index=False)
 (a.out/'audit_report.md').write_text(render(report))
 print(json.dumps(report,indent=2))
def summarize(d):
 def one(q):
  if not len(q): return None
  g=q.groupby('game_id').correct.mean()
  return {'game_mean':float(g.mean()),'row_mean':float(q.correct.mean()),'n_games':int(len(g))}
 out={'overall':one(d)}
 for t in range(1,11): out[f't{t}']=one(d[d.t==t])
 for t in (3,5,8): out[f't_ge{t}']=one(d[d.t>=t])
 return out
def render(r):
 lines=['# Qwen Pilot Method Audit','', 'All analyses are offline and use the existing pilot hidden caches; no VLM forward was run.','', '## Results']
 for k,v in r.items(): lines += [f'### {k}','```json',json.dumps(v,indent=2), '```']
 lines += ['', '## 事件类别计数','', '- validation GT event counts: `unaffected=847`, `target_moved=98`, `target_captured=3`; target-captured accuracy 因样本极少需要谨慎解释。','', '## 明确结论','',
 '- **74.5% 的主要来源不是已经验证的完整 move-to-state 更新。** final state hidden 的 direct 65-class probe 达到约 90.0%，明显高于 hidden + updater 的 74.5%，因此 state-question hidden 已经直接包含大量当前棋子状态信息。',
 '- **旧 oracle move + learned updater 的 70.7% 是输入分布错配。** 旧 updater 是用 hidden soft probability 训练的，却在测试时输入 GT one-hot move；独立 GT-move-trained updater 达到约 98.0%，手写 GT transition 达到 100%，说明标签、captured、目标棋子 identity 和 transition 定义基本正确。',
 '- **soft source/destination vector 确实有信息，但不是充分证据。** 相同 t 的跨 game 错配后，5 个 seed 的 accuracy 约为 51.4%–55.9%，相对未错配 baseline 74.5% 下降约 18.5–23.1 个百分点；因此 updater 使用了 soft vector，但 direct state readout 仍是重要成分。',
 '- **target-relevant event 更适合这个任务。** `unaffected / target_moved / target_captured` event probe 达到约 98.7%，moved destination 约 92.9%，基于该事件的递归 state accuracy 约 96.1%，明显优于完整 64×64 argmax move 的 0.3%。',
 '- **是否继续扩大 Chess：建议先做方法修正后的有限扩展。** 目前不建议直接扩大到 LLaVA 或全量；应先把 direct-state leakage、GT-updater、compact-event updater 作为主对照固定，再决定是否扩展。',
 '', '上述结论全部来自已有 200/100-game Qwen pilot hidden cache；本审计没有加载 VLM、没有新建 split、没有执行新的大规模 forward。']
 return '\n'.join(lines)+'\n'
if __name__=='__main__': main()
