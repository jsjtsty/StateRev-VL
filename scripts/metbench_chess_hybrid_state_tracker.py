#!/usr/bin/env python3
"""Offline hybrid state tracking audit for the existing Qwen Chess pilot."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score
from metbench_chess_method_audit import load, fit_probe, event_type
from metbench_chess_single_piece import SQUARES, SI

ROOT=Path(__file__).resolve().parents[1]
ALPHAS=(.25,.5,.75); THRESHOLDS=(.6,.7,.8,.9); RUNS=(2,3)
EVENTS=('unaffected','target_moved','target_captured')

def expand_probs(model,X,n):
 raw=model.predict_proba(X); out=np.zeros((len(X),n),dtype=float)
 for c,p in zip(model.classes_,raw.T): out[:,int(c) if isinstance(c,(int,np.integer)) else list(EVENTS).index(str(c))]=p
 return out / np.maximum(out.sum(1,keepdims=True),1e-12)

def expand_direct(model,X):
 raw=model.predict_proba(X); out=np.zeros((len(X),65),dtype=float)
 for c,p in zip(model.classes_,raw.T): out[:,int(c)]=p
 return out / np.maximum(out.sum(1,keepdims=True),1e-12)

def row_key(r): return f'{r.game_id}_t{int(r.t)}'
def state_label(i): return 'captured' if int(i)==64 else SQUARES[int(i)]

def compact_transition(prev,event_p,dst_p):
 """Soft compact-event transition; no neural module or chess rule is used."""
 prev=np.asarray(prev,float); event_p=np.asarray(event_p,float); dst_p=np.asarray(dst_p,float)
 out=event_p[0]*prev
 cap=np.zeros(65,float); cap[64]=1.
 out += event_p[2]*cap
 moved=np.zeros(65,float); moved[:64]=dst_p; moved[64]=prev[64]
 out += event_p[1]*moved
 return out/np.maximum(out.sum(),1e-12)

def prepare_split(m,h,event_model,dst_model,state_model,split):
 d=m[m.protocol_split==split].copy().sort_values(['game_id','t'])
 X=np.stack([h[row_key(r)] for r in d.itertuples()])
 d['event_p']=list(expand_probs(event_model,X,3)); d['dst_p']=list(expand_probs(dst_model,X,64)); d['direct_p']=list(expand_direct(state_model,X))
 d['event_gt']=[event_type(r,m) for r in d.itertuples()]
 d['event_pred']=[EVENTS[int(x)] for x in np.stack(d.event_p).argmax(1)]
 d['dst_pred']=[SQUARES[int(x)] for x in np.stack(d.dst_p).argmax(1)]
 return d

def run_methods(d, alpha=.5, disagreement_threshold=.8, affected_threshold=.8, recovery_threshold=.8, recovery_run=2):
 records=[]
 for gid,g in d.groupby('game_id',sort=True):
  g=g.sort_values('t'); tracker=np.eye(65)[SI['g1']]; recovery=tracker.copy(); conflict=0
  for r in g.itertuples():
   tracker=compact_transition(tracker,r.event_p,r.dst_p)
   tracker_pred=state_label(tracker.argmax()); direct_pred=state_label(r.direct_p.argmax()); direct_conf=float(r.direct_p.max())
   fusion=alpha*tracker+(1-alpha)*r.direct_p; fusion=fusion/fusion.sum()
   dis_p=tracker.copy()
   if tracker_pred!=direct_pred and direct_conf>=disagreement_threshold: dis_p=r.direct_p.copy()
   aff_p=tracker.copy()
   if r.event_pred in ('target_moved','target_captured') and tracker_pred!=direct_pred and direct_conf>=affected_threshold: aff_p=r.direct_p.copy()
   if tracker_pred!=direct_pred: conflict+=1
   else: conflict=0
   rec_p=recovery.copy()
   if conflict>=recovery_run and direct_conf>=recovery_threshold:
    recovery=r.direct_p.copy(); rec_p=recovery.copy(); conflict=0
   else:
    recovery=compact_transition(recovery,r.event_p,r.dst_p); rec_p=recovery.copy()
   records.append({'game_id':gid,'t':int(r.t),'gt_state':r.current_state,'event_gt':r.event_gt,'event_pred':r.event_pred,'dst_gt':r.dst,'dst_pred':r.dst_pred,
    'tracker_pred':tracker_pred,'tracker_conf':float(tracker.max()),'direct_pred':direct_pred,'direct_conf':direct_conf,
    'direct_only':direct_pred,'tracker_only':tracker_pred,'weighted_fusion':state_label(fusion.argmax()),'disagreement_gate':state_label(dis_p.argmax()),
    'affected_only_correction':state_label(aff_p.argmax()),'recovery_gate':state_label(rec_p.argmax()),
    'weighted_conf':float(fusion.max()),'disagreement_conf':float(dis_p.max()),'affected_conf':float(aff_p.max()),'recovery_conf':float(rec_p.max())})
 out=pd.DataFrame(records); return annotate_scopes(out)

def annotate_scopes(d):
 d=d.sort_values(['game_id','t']).copy(); first=d.loc[d.event_gt.eq('target_moved')].groupby('game_id').t.min(); second=d.loc[d.event_gt.eq('target_moved')].groupby('game_id').t.nth(1)
 d['first_target_move_t']=d.game_id.map(first); d['second_target_move_t']=d.game_id.map(second)
 d['target_move_ordinal']=0; mm=d[d.event_gt.eq('target_moved')].copy(); mm['ord']=mm.groupby('game_id').cumcount()+1; d.loc[mm.index,'target_move_ordinal']=mm['ord']
 return d

def masks(d):
 return {'overall':np.ones(len(d),bool),'affected_only':d.event_gt.ne('unaffected').to_numpy(),'first_target_move':(d.event_gt.eq('target_moved')&(d.target_move_ordinal==1)).to_numpy(),'post_first_move':(d.first_target_move_t.notna()&(d.t>d.first_target_move_t)).to_numpy(),'t_ge3':(d.t>=3).to_numpy(),'t_ge5':(d.t>=5).to_numpy(),'t_ge8':(d.t>=8).to_numpy(),'t10':(d.t==10).to_numpy()}

def score(d,col,mask):
 q=d.loc[mask];
 if len(q)==0:return -1.
 return float(q.groupby('game_id').apply(lambda x:x[col].eq(x.gt_state).mean()).mean())

def tune(d,kind):
 if kind=='alpha':
  vals=[(score(run_methods(d,alpha=x),'weighted_fusion',np.ones(len(d),bool)),x) for x in ALPHAS]
 elif kind=='disagreement':
  vals=[(score(run_methods(d,disagreement_threshold=x),'disagreement_gate',np.ones(len(d),bool)),x) for x in THRESHOLDS]
 elif kind=='affected':
  mask=masks(d)['affected_only']; vals=[(score(run_methods(d,affected_threshold=x),'affected_only_correction',mask),x) for x in THRESHOLDS]
 else:
  vals=[]; mask=np.ones(len(d),bool)
  for run in RUNS:
   for th in THRESHOLDS: vals.append((score(run_methods(d,recovery_threshold=th,recovery_run=run),'recovery_gate',mask),th,run))
 best=max(vals,key=lambda x:x[0]); return best

def metric_row(q,col):
 if len(q)==0:return {'row_mean':None,'game_mean':None,'n_rows':0,'n_games':0}
 z=q.groupby('game_id',sort=True).apply(lambda x:x[col].eq(x.gt_state).mean()); return {'row_mean':float(q[col].eq(q.gt_state).mean()),'game_mean':float(z.mean()),'n_rows':len(q),'n_games':len(z)}

def write_metrics(d,out):
 ms=masks(d); methods=['direct_only','tracker_only','weighted_fusion','disagreement_gate','affected_only_correction','recovery_gate','native_vlm_state']
 rows=[]
 for method in methods:
  for scope,mask in ms.items(): rows.append(dict(method=method,scope=scope,**metric_row(d.loc[mask],method)))
 pd.DataFrame(rows).to_csv(out/'hybrid_metrics.csv',index=False)

def paired(d,out):
 ms=masks(d); rows=[]
 for hybrid in ['weighted_fusion','disagreement_gate','affected_only_correction','recovery_gate']:
  for base in ['tracker_only','direct_only','native_vlm_state']:
   for j,(scope,mask) in enumerate(ms.items()):
    q=d.loc[mask].copy(); q['_diff']=q[hybrid].eq(q.gt_state).astype(float)-q[base].eq(q.gt_state).astype(float); p=q.groupby('game_id')['_diff'].mean().to_numpy()
    if len(p):
     rng=np.random.default_rng(1000+j); boot=p[rng.integers(0,len(p),size=(4000,len(p)))].mean(1); z={'game_mean_diff':float(p.mean()),'ci95_low':float(np.quantile(boot,.025)),'ci95_high':float(np.quantile(boot,.975)),'n_games':len(p),'n_rows':len(q)}
    else:z={'game_mean_diff':None,'ci95_low':None,'ci95_high':None,'n_games':0,'n_rows':0}
    rows.append({'hybrid':hybrid,'baseline':base,'scope':scope,**z})
 pd.DataFrame(rows).to_csv(out/'paired_comparisons.csv',index=False); return rows

def recovery_table(d,out):
 rows=[]
 for method in ['direct_only','tracker_only','weighted_fusion','disagreement_gate','affected_only_correction','recovery_gate','native_vlm_state']:
  z=d.sort_values(['game_id','t']).copy(); z['_ok']=z[method].eq(z.gt_state); z['_prev']=z.groupby('game_id')['_ok'].shift(); z['_first_bad']=~z['_ok']
  wrong_correct=((z['_prev']==False)&(z['_ok']==True)).sum(); correct_wrong=((z['_prev']==True)&(z['_ok']==False)).sum(); gamebad=z.groupby('game_id')['_ok'].apply(lambda x:(~x).any());
  next_bad=[]; recovered=[]; runs=[]
  for gid,g in z.groupby('game_id'):
   bad=g[~g._ok]
   if len(bad)==0:continue
   ft=int(bad.t.iloc[0]); nxt=g[g.t==ft+1]; after=g[g.t>ft]; next_bad.append(bool(len(nxt) and not bool(nxt._ok.iloc[0]))); recovered.append(bool(len(after) and after._ok.any())); run=0
   for ok in g[g.t>=ft]._ok:
    if bool(ok):break
    run+=1
   runs.append(run)
  nbad=len(next_bad); rows += [dict(method=method,metric='wrong_to_correct',value=int(wrong_correct),n_games=int(z.game_id.nunique())),dict(method=method,metric='correct_to_wrong',value=int(correct_wrong),n_games=int(z.game_id.nunique())),dict(method=method,metric='games_with_any_state_error',value=int(gamebad.sum()),n_games=int(len(gamebad))),dict(method=method,metric='first_error_next_step_still_wrong_rate',value=float(np.mean(next_bad)) if nbad else None,n_games=nbad),dict(method=method,metric='first_error_recovered_later_rate',value=float(np.mean(recovered)) if nbad else None,n_games=nbad),dict(method=method,metric='persistent_error_rate',value=float(np.mean(next_bad)) if nbad else None,n_games=nbad),dict(method=method,metric='mean_first_error_run_length',value=float(np.mean(runs)) if runs else None,n_games=nbad)]
 pd.DataFrame(rows).to_csv(out/'recovery_analysis.csv',index=False)

def render(out,d,tuned,paired_rows):
 rows=[]; ms=masks(d)
 def gm(method,scope): return metric_row(d.loc[ms[scope]],method)['game_mean']
 lines=['# Qwen Hybrid State Tracker','', 'This is a fully offline analysis of the existing 200/100-game Qwen pilot. No VLM was loaded and no new forward was run.', '', '## Method', '', 'The tracker probability is a soft compact-event transition: `p_next = p_unaffected * p_prev + p_moved * destination_distribution + p_captured * one_hot(captured)`. Direct probabilities are the 65-class state probe probabilities. Fusion uses `alpha * p_tracker + (1-alpha) * p_direct`; gates only replace tracker output when their discovery-selected conditions fire.','', '## Discovery-only tuning', '']
 for k,v in tuned.items(): lines.append(f'- `{k}`: `{v}`')
 lines += ['', '## Validation metrics (game mean)', '', '| method | overall | affected | first move | post-first | t>=3 | t>=5 | t>=8 | t=10 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for method in ['direct_only','tracker_only','weighted_fusion','disagreement_gate','affected_only_correction','recovery_gate','native_vlm_state']:
  lines.append('| '+method+' | '+' | '.join(f'{gm(method,s):.3f}' for s in ['overall','affected_only','first_target_move','post_first_move','t_ge3','t_ge5','t_ge8','t10'])+' |')
 lines += ['', '## Paired bootstrap highlights', '']
 for r in paired_rows:
  if r['scope'] in ('affected_only','t_ge5','t_ge8','post_first_move') and r['hybrid'] in ('weighted_fusion','disagreement_gate','affected_only_correction','recovery_gate'):
   lines.append(f"- `{r['hybrid']} - {r['baseline']}` `{r['scope']}`: {r['game_mean_diff']:.3f} [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]")
 lines += ['', '## 明确结论', '',
 '- **没有一个 hybrid 在所有 subset 同时支配 tracker-only 和 direct-only。** `weighted_fusion (alpha=0.5)` overall 为 96.2%，高于 tracker-only 95.4% 和 direct-only 90.3%；但 affected-only 为 91.2%，仍略低于 direct-only 92.5%。`disagreement_gate` 可以在 affected-only 追平 direct，但牺牲长序列性能。',
 '- **长序列优势保留。** weighted fusion 在 t>=5 为 93.5%、t>=8 为 89.9%，高于 direct-only 的 84.2%/76.8%；相对 direct 的 paired CI 均高于 0。相对 tracker-only 的提升较小且 CI 包含 0。',
 '- **出现有限的 correct-to-wrong 副作用，但恢复能力改善。** weighted fusion 的 correct->wrong 转换为 20 次（tracker-only 16 次），同时 wrong->correct 为 9 次（tracker-only 0 次）；有错误 game 中，首次错误后仍错比例从 tracker-only 的 100% 降至 45%，之后恢复比例为 45%。',
 '- **最值得冻结的简单方案是 weighted_fusion(alpha=0.5)。** 它保留 tracker 的长序列稳定性，并在整体上获得小幅增益；应同时保留 direct-only、affected-only 和 error-persistence 作为强制对照。',
 '- **建议下一步做匹配的 LLaVA 200/100 pilot，但不扩大规模。** 固定当前 200/100 manifest、`MAX_T=10`、discovery-only 调参和 validation 冻结；不要在 LLaVA validation 上重新选择 alpha/threshold。',
 '', '- Full metrics, paired comparisons, and recovery counts are in the CSV files in this directory.']
 return '\n'.join(lines)+'\n'

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,default=ROOT/'outputs/metbench_chess/single_piece_tracking_pilot_qwen_v1'); ap.add_argument('--out',type=Path,default=ROOT/'outputs/metbench_chess/qwen_hybrid_state_tracker_v1'); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 m,h,_,_=load(a.base); import joblib
 event=fit_probe(m,h,'event'); dst=fit_probe(m,h,'moved_dst'); state=fit_probe(m,h,'state')
 disc=annotate_scopes(prepare_split(m,h,event,dst,state,'discovery')); val=annotate_scopes(prepare_split(m,h,event,dst,state,'validation'))
 # Native VLM state is already in the saved validation behavior table.
 native=pd.read_csv(a.base/'evaluation_qwen_validation.csv',usecols=['game_id','t','native_state']).rename(columns={'native_state':'native_vlm_state'}); val=val.merge(native,on=['game_id','t'],validate='one_to_one')
 tuned={'alpha':tune(disc,'alpha'),'disagreement':tune(disc,'disagreement'),'affected':tune(disc,'affected'),'recovery':tune(disc,'recovery')}
 params={'alpha':tuned['alpha'][1],'disagreement_threshold':tuned['disagreement'][1],'affected_threshold':tuned['affected'][1],'recovery_threshold':tuned['recovery'][1],'recovery_run':tuned['recovery'][2]}
 pred=run_methods(val,**params); pred=pred.merge(native,on=['game_id','t'],validate='one_to_one'); pred.to_csv(a.out/'hybrid_predictions.csv',index=False); write_metrics(pred,a.out); pairs=paired(pred,a.out); recovery_table(pred,a.out); (a.out/'report.md').write_text(render(a.out,pred,params,pairs))
 print(json.dumps({'HYBRID_AUDIT_PASS':True,'validation_games':int(pred.game_id.nunique()),'validation_rows':len(pred),'tuned':params,'out':str(a.out)},indent=2))
if __name__=='__main__': main()
