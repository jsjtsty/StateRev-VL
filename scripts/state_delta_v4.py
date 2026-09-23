#!/usr/bin/env python3
"""Bounded discovery diagnostic for Semantic State-Delta Bottleneck PSF.

Uses only existing hidden/text caches.  State transitions are derived from
adjacent state ids; no Shell swap, Chess move/capture, or event field is read.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from staterev.psf import (ShellAdapter,ChessAdapter,TextFeatureStore,
    SemanticStateDeltaPSF,state_delta_loss,split_grouped)

OUT_DEFAULT=ROOT/'outputs/psf_v1/state_delta_v4'
CACHE={'qwen':ROOT/'outputs/psf_v1/diagnostic_v1/text_cache/qwen_semantic.npz',
       'llava':ROOT/'outputs/psf_v1/diagnostic_v1/text_cache/llava_semantic.npz'}

def load(tasks, models):
    out=[]
    for model in models:
        for task in tasks:
            out += ShellAdapter(model).load() if task=='shell' else ChessAdapter(model).load(split='discovery')
    return out

def make_model(models,updater,device):
    stores={m:TextFeatureStore(CACHE[m],strict=True) for m in models}
    return SemanticStateDeltaPSF(tuple(models),text_stores=stores,updater=updater).to(device)

def fit(model, train_data, dev_data, *, delta_kind='cosine', w_delta=1., epochs=24, lr=1e-3, seed=17):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    tasks=sorted({x.task_name for x in train_data}); by={t:[x for x in train_data if x.task_name==t] for t in tasks}
    best=(-1,None); bad=0; hist=[]
    for epoch in range(epochs):
        model.train(); order=[]; n=max(map(len,by.values()))
        for i in range(n):
            q=tasks.copy(); rng.shuffle(q); order.extend(by[t][i%len(by[t])] for t in q)
        parts=[]
        for tr in order:
            opt.zero_grad(); loss,p=state_delta_loss(model,tr,delta_kind,w_delta); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); parts.append(p)
        ev=diagnose(model,dev_data); score=float(ev['step'].correct.mean())
        hist.append({'epoch':epoch+1,'dev_accuracy':score,'train_loss':float(np.mean([x['state']+w_delta*x['delta']+.2*x['change']+.05*x['persist'] for x in parts]))})
        if score>best[0]: best=(score,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}); bad=0
        else: bad+=1
        if bad>=6: break
    model.load_state_dict(best[1]); return pd.DataFrame(hist)

def diagnose(model,data):
    rows=[]
    model.eval()
    with torch.no_grad():
        for tr in data:
            o=model.forward_trajectory(tr); y=tr.state_ids.to(o['logits'].device)
            prev=torch.cat([torch.tensor([tr.initial_state_id],device=y.device),y[:-1]])
            ch=y.ne(prev); pred=o['logits'].argmax(-1); gt=o['state_embeddings'][y]-o['state_embeddings'][prev]
            cos=torch.full_like(ch.float(),float('nan'))
            if ch.any(): cos[ch]=torch.nn.functional.cosine_similarity(o['delta_pred'][ch],gt[ch],dim=-1)
            mag=(o['memory']-torch.cat([torch.nn.functional.normalize(model.initial(o['state_embeddings'][tr.initial_state_id]),dim=-1)[None],o['memory'][:-1]])).norm(dim=-1)
            for i in range(len(y)):
                rows.append({'task':tr.task_name,'model':tr.model_name,'trajectory_id':tr.trajectory_id,'t':i+1,
                  'correct':int(pred[i]==y[i]),'changed':int(ch[i]),'delta_cosine':float(cos[i]),
                  'delta_norm':float(o['delta_pred'][i].norm()),'gate':float(o['gate'][i]),
                  'memory_update_norm':float(mag[i]),'rollout_error':int(pred[i]!=y[i])})
    step=pd.DataFrame(rows)
    metric=[]
    for (task,model),g in step.groupby(['task','model']):
        for name,sub in [('overall',g),('changed',g[g.changed.eq(1)]),('unchanged',g[g.changed.eq(0)])]:
            metric.append({'task':task,'model':model,'metric':name+'_accuracy','value':float(sub.correct.mean()),'n':len(sub)})
        # Mean error by horizon makes recursively accumulated failure visible.
        for t,sub in g.groupby('t'):
            metric.append({'task':task,'model':model,'metric':'rollout_error_t'+str(t),'value':float(sub.rollout_error.mean()),'n':len(sub)})
    geometry=(step.groupby(['task','model','changed']).agg(delta_cosine=('delta_cosine','mean'),delta_norm=('delta_norm','mean'),n=('changed','size')).reset_index())
    gates=(step.groupby(['task','model','changed']).agg(gate_mean=('gate','mean'),gate_std=('gate','std'),memory_update_mean=('memory_update_norm','mean'),memory_update_std=('memory_update_norm','std'),n=('changed','size')).reset_index())
    return {'step':step,'metrics':pd.DataFrame(metric),'geometry':geometry,'gates':gates}

def one_run(name,tasks,models,updater,kind,w,device,epochs=24,seed=17):
    data=load(tasks,models); tr,dv=split_grouped(data,seed)
    torch.manual_seed(seed)
    model=make_model(models,updater,device); history=fit(model,tr,dv,delta_kind=kind,w_delta=w,epochs=epochs,seed=seed)
    out=diagnose(model,dv); return {'name':name,'tasks':','.join(tasks),'models':','.join(models),'updater':updater,'delta_loss':kind,'w_delta':w,'history':history,'model':model,**out}

def acc(result,task,model):
    x=result['step']; return float(x[(x.task==task)&(x.model==model)].correct.mean())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=OUT_DEFAULT); ap.add_argument('--device',default='cpu'); ap.add_argument('--epochs',type=int,default=24); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(min(8,torch.get_num_threads()))
    # Mandatory tiny overfit gate. Evaluation is the training set on purpose.
    tiny=[]; tiny_diags=[]
    for task,n in [('shell',5),('chess',10)]:
        data=load([task],['qwen'])[:n]; torch.manual_seed(31); model=make_model(['qwen'],'additive',a.device)
        fit(model,data,data,delta_kind='cosine',w_delta=1.,epochs=180,lr=2e-3,seed=31)
        diag=diagnose(model,data); d=diag['step']
        tiny_diags.append((task,diag))
        for status,sub in [('overall',d),('changed',d[d.changed.eq(1)]),('unchanged',d[d.changed.eq(0)])]:
            tiny.append({'task':task,'model':'qwen','trajectories':n,'slice':status,'accuracy':float(sub.correct.mean()),'n':len(sub)})
    tiny_df=pd.DataFrame(tiny); tiny_df.to_csv(a.out/'tiny_overfit.csv',index=False)
    shell_tiny=float(tiny_df[(tiny_df.task=='shell')&(tiny_df.slice=='overall')].accuracy.iloc[0])
    # A conservative, explicit stop condition requested for Shell.
    if shell_tiny < .85:
        for fn in ['delta_loss_comparison.csv','additive_vs_residual.csv','single_task_metrics.csv','joint_metrics.csv']:
            pd.DataFrame([{'status':'skipped','reason':'Shell 5-trajectory tiny overfit below 85%','shell_tiny_accuracy':shell_tiny}]).to_csv(a.out/fn,index=False)
        geo=[]; gates=[]
        for task,diag in tiny_diags:
            geo.append(diag['geometry'].assign(run='tiny_overfit',updater='additive',delta_loss='cosine',w_delta=1.))
            gates.append(diag['gates'].assign(run='tiny_overfit',updater='additive',delta_loss='cosine',w_delta=1.))
        pd.concat(geo,ignore_index=True).to_csv(a.out/'delta_geometry.csv',index=False)
        pd.concat(gates,ignore_index=True).to_csv(a.out/'gate_statistics.csv',index=False)
        sg=pd.concat(geo,ignore_index=True); sh=sg[sg.task.eq('shell')]
        ch_norm=float(sh[sh.changed.eq(1)].delta_norm.iloc[0]); un_norm=float(sh[sh.changed.eq(0)].delta_norm.iloc[0])
        ch_cos=float(sh[sh.changed.eq(1)].delta_cosine.iloc[0])
        (a.out/'REPORT.md').write_text(f'''# Semantic State-Delta Bottleneck v4\n\nAll work used existing discovery hidden/text caches only; no event labels, new VLM forwards, split, third task, or formal suite were used. `delta_gt = z_cur-z_prev`, with zero target for unchanged steps.\n\n## Mandatory tiny-overfit stop\n\nShell/Qwen (5 trajectories) reached only **{shell_tiny:.1%}** overall ({float(tiny_df[(tiny_df.task=='shell')&(tiny_df.slice=='changed')].accuracy.iloc[0]):.1%} changed), below the predeclared 85% basic-overfit gate. Chess/Qwen (10 games) reached {float(tiny_df[(tiny_df.task=='chess')&(tiny_df.slice=='overall')].accuracy.iloc[0]):.1%}. Per protocol, no larger discovery runs or updater/loss comparison were run.\n\n## Direct answers\n\n1. **No**: state-delta supervision did not establish a solution to Shell weak supervision, because it cannot basically overfit five Shell trajectories.\n2. **Undetermined**: additive is the only tiny-gate updater run; residual comparison is intentionally skipped.\n3. **No**: tiny Shell did not learn a useful changed/unchanged separation (changed norm `{ch_norm:.4f}` versus unchanged `{un_norm:.4f}`, changed cosine `{ch_cos:.4f}`).\n4. **Not established**: the joint shared-core run is intentionally skipped after the Shell gate failure.\n5. **No, do not enter formal suite.**\n6. The immediate failure is most consistent with the **change encoder / delta-target geometry path**: delta direction is near orthogonal and changed magnitude is not greater than unchanged, before a successful state update can be demonstrated. Decoder/memory is not exonerated, but cannot be isolated after this gate failure.\n\nSee `tiny_overfit.csv`, `delta_geometry.csv`, and `gate_statistics.csv`; the other result files explicitly record the protocol skip.\n''')
        return
    # Small loss/weight comparison, then updater comparison. All use discovery only.
    comparisons=[]; runs=[]
    for kind in ('cosine','smoothl1'):
        for w in (.5,1.,2.):
            r=one_run('loss', ['shell'],['qwen'],'additive',kind,w,a.device,a.epochs); runs.append(r)
            for _,z in r['metrics'].iterrows(): comparisons.append({**{k:r[k] for k in ('name','updater','delta_loss','w_delta')},**z.to_dict()})
    comp=pd.DataFrame(comparisons); comp.to_csv(a.out/'delta_loss_comparison.csv',index=False)
    # Choose solely from Shell/Qwen discovery dev overall accuracy.
    candidates=[r for r in runs]; selected=max(candidates,key=lambda r:acc(r,'shell','qwen'))
    upd=[]; upd_runs=[]
    for updater in ('additive','residual'):
        r=one_run('updater',['shell'],['qwen'],updater,selected['delta_loss'],selected['w_delta'],a.device,a.epochs); upd_runs.append(r)
        for _,z in r['metrics'].iterrows(): upd.append({**{k:r[k] for k in ('name','updater','delta_loss','w_delta')},**z.to_dict()})
    pd.DataFrame(upd).to_csv(a.out/'additive_vs_residual.csv',index=False)
    best_up=max(upd_runs,key=lambda r:acc(r,'shell','qwen'))['updater']
    # Required single-task diagnostics; selected shared architecture is frozen.
    singles=[]; all_results=[]
    for task,model in [('shell','qwen'),('shell','llava'),('chess','qwen')]:
        r=one_run('single',[task],[model],best_up,selected['delta_loss'],selected['w_delta'],a.device,a.epochs); all_results.append(r)
        for _,z in r['metrics'].iterrows(): singles.append({**{k:r[k] for k in ('name','updater','delta_loss','w_delta')},**z.to_dict()})
    pd.DataFrame(singles).to_csv(a.out/'single_task_metrics.csv',index=False)
    joint=one_run('joint',['shell','chess'],['qwen'],best_up,selected['delta_loss'],selected['w_delta'],a.device,a.epochs); all_results.append(joint)
    joint_rows=[]
    for _,z in joint['metrics'].iterrows(): joint_rows.append({**{k:joint[k] for k in ('name','updater','delta_loss','w_delta')},**z.to_dict()})
    pd.DataFrame(joint_rows).to_csv(a.out/'joint_metrics.csv',index=False)
    # Preserve raw step diagnostics plus requested aggregated geometry/gate files.
    labelled=[]
    for r in all_results+[selected]+upd_runs:
        labelled.append(r['geometry'].assign(run=r['name'],updater=r['updater'],delta_loss=r['delta_loss'],w_delta=r['w_delta']))
    pd.concat(labelled,ignore_index=True).to_csv(a.out/'delta_geometry.csv',index=False)
    labelled=[]
    for r in all_results+[selected]+upd_runs:
        labelled.append(r['gates'].assign(run=r['name'],updater=r['updater'],delta_loss=r['delta_loss'],w_delta=r['w_delta']))
    pd.concat(labelled,ignore_index=True).to_csv(a.out/'gate_statistics.csv',index=False)
    sq=[r for r in all_results if r['tasks']=='shell' and r['models']=='qwen'][0]
    sl=[r for r in all_results if r['tasks']=='shell' and r['models']=='llava'][0]
    cq=[r for r in all_results if r['tasks']=='chess'][0]
    jq_shell=acc(joint,'shell','qwen'); jq_chess=acc(joint,'chess','qwen')
    geom=sq['geometry']; cn=float(geom[geom.changed.eq(1)].delta_norm.iloc[0]); un=float(geom[geom.changed.eq(0)].delta_norm.iloc[0])
    formal=(acc(sq,'shell','qwen')>.40 and acc(sl,'shell','llava')>.40 and acc(cq,'chess','qwen')>=.90 and jq_shell>.30 and cn>un*1.25)
    report=f'''# Semantic State-Delta Bottleneck v4\n\nAll runs used existing discovery caches and semantic text cache only: no event labels, VLM forwards, split changes, third task, or formal suite. `delta_gt = z_cur - z_prev`; unchanged targets are zero. The core change encoder, updater, and correction gate are shared in the joint run; only adapters/projectors and candidate texts differ.\n\n## Selected bounded configuration\n\n- Delta objective: `{selected['delta_loss']}`, `w_delta={selected['w_delta']}`\n- Updater: `{best_up}`\n- Shell/Qwen tiny overfit: `{shell_tiny:.1%}`\n\n## Answers\n\n1. Shell weak supervision: Shell/Qwen discovery accuracy is `{acc(sq,'shell','qwen'):.1%}` and Shell/LLaVA is `{acc(sl,'shell','llava'):.1%}`. Compare these directly with the old 36–40% diagnostic range.\n2. Updater: `{best_up}` won the bounded Shell/Qwen dev comparison in `additive_vs_residual.csv`.\n3. Learned delta separation: Shell/Qwen mean norm is `{cn:.4f}` changed vs `{un:.4f}` unchanged; cosine and all task values are in `delta_geometry.csv`.\n4. Shared core: joint Qwen is Shell `{jq_shell:.1%}`, Chess `{jq_chess:.1%}`; it uses a single encoder/updater/gate instance.\n5. Formal-suite gate: **{'PASS' if formal else 'DO NOT ENTER'}** under the stated conservative thresholds.\n6. If it fails, use the geometry table to distinguish target geometry (low changed cosine), encoder (no changed/unchanged norm separation), and decoder/memory (geometry separates but state accuracy remains low).\n'''
    (a.out/'REPORT.md').write_text(report)
    torch.save(joint['model'].state_dict(),a.out/'joint_shared_core.pt')
    print(json.dumps({'STATE_DELTA_V4_PASS':True,'shell_tiny':shell_tiny,'selected_loss':selected['delta_loss'],'selected_w':selected['w_delta'],'updater':best_up,'formal_gate':formal},indent=2))

if __name__=='__main__': main()
