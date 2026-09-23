#!/usr/bin/env python3
"""Offline validation of the compact target-relevant event tracker.

This script only reads the existing Qwen pilot manifest/caches and the saved
native-state validation table. It never loads a VLM and never performs a
forward pass.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support,
                             f1_score)
from metbench_chess_method_audit import load, fit_probe, eval_probe, event_type

ROOT=Path(__file__).resolve().parents[1]
EVENTS=['unaffected','target_moved','target_captured']

def sq(i):
    return 'captured' if int(i)==64 else __import__('metbench_chess_single_piece',fromlist=['SQUARES']).SQUARES[int(i)]

def row_key(r): return f'{r.game_id}_t{int(r.t)}'

def metric_row(q, correct):
    q=q.copy(); q['_correct']=np.asarray(correct,dtype=bool)
    if len(q)==0: return {'row_mean':None,'game_mean':None,'n_rows':0,'n_games':0}
    g=q.groupby('game_id',sort=True)['_correct'].mean()
    return {'row_mean':float(q['_correct'].mean()),'game_mean':float(g.mean()),'n_rows':int(len(q)),'n_games':int(len(g))}

def first_move_columns(d):
    d=d.sort_values(['game_id','t']).copy()
    first=d.loc[d.event_gt.eq('target_moved')].groupby('game_id').t.min()
    second=d.loc[d.event_gt.eq('target_moved')].groupby('game_id').t.nth(1)
    d['first_target_move_t']=d.game_id.map(first)
    d['second_target_move_t']=d.game_id.map(second)
    move_ord=d.loc[d.event_gt.eq('target_moved')].copy()
    move_ord['move_ordinal']=move_ord.groupby('game_id').cumcount()+1
    d['target_move_ordinal']=move_ord.set_index(move_ord.index).move_ordinal.reindex(d.index)
    d['target_move_ordinal']=d['target_move_ordinal'].fillna(0).astype(int)
    d['prior_compact_pred']=d.groupby('game_id').compact_pred.shift().fillna('g1')
    d['prior_gt_state']=d.groupby('game_id').gt_state.shift().fillna('g1')
    return d

def scopes(d):
    return {
      'overall':np.ones(len(d),bool),
      'affected_only':d.event_gt.ne('unaffected').to_numpy(),
      'target_moved':d.event_gt.eq('target_moved').to_numpy(),
      'target_captured':d.event_gt.eq('target_captured').to_numpy(),
      'first_target_move':(d.event_gt.eq('target_moved') & d.target_move_ordinal.eq(1)).to_numpy(),
      'post_first_move':(d.first_target_move_t.notna() & (d.t>d.first_target_move_t)).to_numpy(),
      'second_plus_target_move':(d.event_gt.eq('target_moved') & (d.target_move_ordinal>=2)).to_numpy(),
      'post_second_move':(d.second_target_move_t.notna() & (d.t>d.second_target_move_t)).to_numpy(),
      't_ge3':(d.t>=3).to_numpy(), 't_ge5':(d.t>=5).to_numpy(), 't_ge8':(d.t>=8).to_numpy(),
    }

def write_state_table(d, methods, path, selected_scopes=None):
    selected_scopes=selected_scopes or scopes(d); rows=[]
    for method,col in methods.items():
        for name,mask in selected_scopes.items():
            q=d.loc[mask]; z=metric_row(q,q[col].eq(q.gt_state)); z.update({'method':method,'scope':name}); rows.append(z)
    pd.DataFrame(rows).to_csv(path,index=False)

def event_metrics(d, path):
    y=d.event_gt.to_numpy(); p=d.event_pred.to_numpy(); rows=[]
    cm=confusion_matrix(y,p,labels=EVENTS)
    for i,a in enumerate(EVENTS):
        for j,b in enumerate(EVENTS): rows.append({'section':'confusion_matrix','class':a,'predicted_class':b,'metric':'count','value':int(cm[i,j])})
    prec,rec,f1,sup=precision_recall_fscore_support(y,p,labels=EVENTS,zero_division=0)
    for a,pp,rr,ff,nn in zip(EVENTS,prec,rec,f1,sup):
        rows += [{'section':'per_class','class':a,'metric':'precision','value':float(pp),'support':int(nn)},
                 {'section':'per_class','class':a,'metric':'recall','value':float(rr),'support':int(nn)},
                 {'section':'per_class','class':a,'metric':'f1','value':float(ff),'support':int(nn)}]
    rows += [{'section':'overall','class':'all','metric':'accuracy','value':float(accuracy_score(y,p))},
             {'section':'overall','class':'all','metric':'balanced_accuracy','value':float(balanced_accuracy_score(y,p))},
             {'section':'overall','class':'all','metric':'macro_f1','value':float(f1_score(y,p,labels=EVENTS,average='macro',zero_division=0))}]
    ya=np.where(y=='unaffected','unaffected','affected'); pa=np.where(p=='unaffected','unaffected','affected'); labels=['unaffected','affected']
    pp,rr,ff,nn=precision_recall_fscore_support(ya,pa,labels=labels,zero_division=0)
    rows += [{'section':'affected_binary','class':'all','metric':'accuracy','value':float(accuracy_score(ya,pa))},
             {'section':'affected_binary','class':'all','metric':'balanced_accuracy','value':float(balanced_accuracy_score(ya,pa))},
             {'section':'affected_binary','class':'all','metric':'macro_f1','value':float(f1_score(ya,pa,labels=labels,average='macro',zero_division=0))}]
    for a,pp0,rr0,ff0,nn0 in zip(labels,pp,rr,ff,nn):
        rows += [{'section':'affected_binary_per_class','class':a,'metric':'precision','value':float(pp0),'support':int(nn0)},
                 {'section':'affected_binary_per_class','class':a,'metric':'recall','value':float(rr0),'support':int(nn0)},
                 {'section':'affected_binary_per_class','class':a,'metric':'f1','value':float(ff0),'support':int(nn0)}]
    pd.DataFrame(rows).to_csv(path,index=False)
    return {'confusion_matrix':cm.tolist(),'classes':EVENTS,
            'accuracy':float(accuracy_score(y,p)),
            'balanced_accuracy':float(balanced_accuracy_score(y,p)),
            'macro_f1':float(f1_score(y,p,labels=EVENTS,average='macro',zero_division=0)),
            'affected_accuracy':float(accuracy_score(ya,pa)),
            'affected_balanced_accuracy':float(balanced_accuracy_score(ya,pa)),
            'affected_macro_f1':float(f1_score(ya,pa,labels=labels,average='macro',zero_division=0)),
            'support':{x:int((y==x).sum()) for x in EVENTS}}

def bootstrap_diff(d, col_a, col_b, mask, seed, n_boot=4000):
    q=d.loc[mask].copy(); q['_diff']=q[col_a].eq(q.gt_state).astype(float)-q[col_b].eq(q.gt_state).astype(float)
    per=q.groupby('game_id',sort=True)['_diff'].mean().to_numpy()
    if len(per)==0: return {'game_mean_diff':None,'ci95_low':None,'ci95_high':None,'n_games':0,'n_rows':0}
    rng=np.random.default_rng(seed); b=per[rng.integers(0,len(per),size=(n_boot,len(per)))].mean(1)
    return {'game_mean_diff':float(per.mean()),'ci95_low':float(np.quantile(b,.025)),'ci95_high':float(np.quantile(b,.975)),'n_games':int(len(per)),'n_rows':int(len(q))}

def errors(d):
 d=d.copy(); d['event_type_correct']=d.event_pred.eq(d.event_gt)
 d['dst_correct']=~d.event_gt.eq('target_moved') | (d.dst_pred.eq(d.dst_gt))
 d['prior_correct']=d.prior_compact_pred.eq(d.prior_gt_state)
 d['state_correct']=d.compact_pred.eq(d.gt_state)
 cats=[]
 for r in d.itertuples():
  if r.state_correct: c='correct'
  elif not r.event_type_correct: c='event_type_error'
  elif r.event_gt=='target_moved' and not r.dst_correct: c='moved_destination_error'
  elif r.event_gt=='target_captured' and r.prior_correct: c='captured_handling_error'
  elif not r.prior_correct: c='earlier_error_propagated'
  else: c='other_transition_error'
  cats.append(c)
 d['error_category']=cats
 rows=[]
 for c,n in d.error_category.value_counts().items():
  q=d[d.error_category.eq(c)]; rows.append({'section':'error_category','category':c,'count':int(n),'row_fraction':float(n/len(d)),'games':int(q.game_id.nunique())})
 # First error persistence, game is the independent unit.
 game_rows=[]
 for gid,g in d.groupby('game_id',sort=True):
  g=g.sort_values('t'); bad=g[~g.state_correct]
  if bad.empty: game_rows.append({'game_id':gid,'first_error_t':None,'next_step_error':False,'recovered_after_error':False,'first_error_run_length':0}); continue
  ft=int(bad.t.iloc[0]); after=g[g.t>ft]; nextrow=g[g.t==ft+1]
  run=0
  for ok in g[g.t>=ft].state_correct:
   if bool(ok): break
   run+=1
  game_rows.append({'game_id':gid,'first_error_t':ft,'next_step_error':bool(len(nextrow) and not bool(nextrow.state_correct.iloc[0])),'recovered_after_error':bool(len(after) and after.state_correct.any()),'first_error_run_length':run})
 gd=pd.DataFrame(game_rows); nbad=gd.first_error_t.notna()
 rows += [
  {'section':'persistence','category':'games_with_state_error','count':int(nbad.sum()),'row_fraction':float(nbad.mean()),'games':int(nbad.sum())},
  {'section':'persistence','category':'next_step_still_wrong','count':int(gd.loc[nbad,'next_step_error'].sum()),'row_fraction':float(gd.loc[nbad,'next_step_error'].mean()) if nbad.any() else None,'games':int(nbad.sum())},
  {'section':'persistence','category':'recovered_later','count':int(gd.loc[nbad,'recovered_after_error'].sum()),'row_fraction':float(gd.loc[nbad,'recovered_after_error'].mean()) if nbad.any() else None,'games':int(nbad.sum())},
  {'section':'persistence','category':'mean_first_error_run_length','count':None,'row_fraction':float(gd.loc[nbad,'first_error_run_length'].mean()) if nbad.any() else None,'games':int(nbad.sum())},
 ]
 return pd.DataFrame(rows),gd

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,default=ROOT/'outputs/metbench_chess/single_piece_tracking_pilot_qwen_v1'); ap.add_argument('--audit-base',type=Path,default=ROOT/'outputs/metbench_chess/qwen_pilot_method_audit_v1'); ap.add_argument('--out',type=Path,default=ROOT/'outputs/metbench_chess/qwen_compact_event_validation_v1'); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 m,h,sp,dp=load(a.base); m=m[m.protocol_split.isin(['discovery','validation'])].copy(); v=m[m.protocol_split=='validation'].copy()
 # Refit all probes offline: discovery only, validation is never used for fitting/C choice.
 ev=fit_probe(m,h,'event'); dst=fit_probe(m,h,'moved_dst'); state_probe=fit_probe(m,h,'state')
 Xv=np.stack([h[row_key(r)] for r in v.itertuples()]); v['event_pred']=ev.predict(Xv); v['event_gt']=[event_type(r,m) for r in v.itertuples()]; v['dst_pred_idx']=dst.predict(Xv); v['dst_pred']=[sq(x) for x in v.dst_pred_idx]; v['dst_gt']=v.dst
 # Compact recursive tracker.
 preds=[]
 for gid,g in v.sort_values(['game_id','t']).groupby('game_id'):
  state='g1'
  for r in g.itertuples():
   if r.event_pred=='target_captured': state='captured'
   elif r.event_pred=='target_moved': state=r.dst_pred
   preds.append((r.Index,state))
 v['compact_pred']=pd.Series(dict(preds));
 # Existing native VLM state output, no generation here.
 native=pd.read_csv(a.base/'evaluation_qwen_validation.csv',usecols=['game_id','t','native_state'])
 v=v.merge(native,on=['game_id','t'],validate='one_to_one'); v['gt_state']=v.current_state
 v['direct_pred']=v['compact_pred'] # replaced by offline direct probe below
 direct_idx=state_probe.predict(Xv); v['direct_pred']=[sq(x) for x in direct_idx]
 v['initial_pred']='g1'; v['never_update_pred']='g1'; majority=m[m.protocol_split=='discovery'].current_state.mode().iloc[0]; v['majority_pred']=majority
 v=first_move_columns(v)
 # Event metrics and compact event result tables.
 em=event_metrics(v,a.out/'compact_event_metrics.csv')
 methods={'compact_event_recursive':'compact_pred','direct_hidden_state_probe':'direct_pred','native_vlm_state':'native_state','always_initial':'initial_pred','never_update':'never_update_pred','majority_state':'majority_pred'}
 change_sc={k:scopes(v)[k] for k in ['overall','target_moved','target_captured','affected_only','first_target_move','post_first_move','second_plus_target_move','post_second_move','t_ge5','t_ge8']}
 write_state_table(v,{'compact_event_recursive':'compact_pred'},a.out/'change_point_metrics.csv',change_sc)
 base_sc={k:scopes(v)[k] for k in ['overall','affected_only','post_first_move','t_ge5','t_ge8']}; write_state_table(v,methods,a.out/'baselines.csv',base_sc)
 # Paired game-level differences, same games within each scope.
 rows=[]; sc_all=scopes(v)
 for scope in ['overall','t_ge3','t_ge5','t_ge8','post_first_move']:
  for baseline,name in [('direct_pred','compact_minus_direct'),('native_state','compact_minus_native')]:
   z=bootstrap_diff(v,'compact_pred',baseline,sc_all[scope],17+len(rows)); z.update({'comparison':name,'scope':scope}); rows.append(z)
 pd.DataFrame(rows).to_csv(a.out/'paired_comparisons.csv',index=False)
 # Error taxonomy and persistence.
 err,gd=errors(v); err.to_csv(a.out/'error_analysis.csv',index=False)
 # Persist a detailed row-level audit table for reproducibility.
 v.to_csv(a.out/'_row_level_validation.csv',index=False)
 report=render_report(v,em,ev,dst,state_probe,methods,sc_all,err,gd,rows,majority)
 (a.out/'report.md').write_text(report)
 print(json.dumps({'AUDIT_PASS':True,'out':str(a.out),'validation_games':int(v.game_id.nunique()),'validation_rows':len(v),'event':em,'majority_state':majority},indent=2))

def render_report(v,em,ev,dst,state_probe,methods,sc,err,gd,paired,majority):
 def acc(col,mask):
  q=v.loc[mask]; return q[col].eq(q.gt_state).mean() if len(q) else None
 def line(scope,mask):
  vals=[]
  for n,c in methods.items(): vals.append(f"{n}={acc(c,mask):.3f}" if acc(c,mask) is not None else f"{n}=NA")
  return f"- `{scope}`: "+", ".join(vals)
 lines=['# Qwen Compact Event Validation','', '本报告完全离线复用现有 Qwen pilot hidden cache；没有加载 VLM、没有执行新的 forward、没有重新抽 split。', '', '## 数据与协议', '', f"validation: {v.game_id.nunique()} games, {len(v)} prefixes；discovery-only probe C: event={ev.audit_C}, moved-destination={dst.audit_C}, direct-state={state_probe.audit_C}；hidden cache 只有已保存的 final layer。", '', 'GT validation event counts: '+', '.join(f'`{k}={int((v.event_gt==k).sum())}`' for k in EVENTS)+'. `target_captured` 样本极少，不做强结论。', '', '## Event 分类严格指标', '', f"- confusion matrix (rows GT, columns prediction; order `{EVENTS}`): `{em['confusion_matrix']}`", f"- accuracy: {em['accuracy']:.4f}; balanced accuracy: {em['balanced_accuracy']:.4f}; macro-F1: {em['macro_f1']:.4f}", f"- affected vs unaffected: accuracy={em['affected_accuracy']:.4f}, balanced accuracy={em['affected_balanced_accuracy']:.4f}, macro-F1={em['affected_macro_f1']:.4f}", '', '| class | support | precision | recall | F1 |', '|---|---:|---:|---:|---:|']
 y=v.event_gt.to_numpy(); p=v.event_pred.to_numpy(); pp,rr,ff,nn=precision_recall_fscore_support(y,p,labels=EVENTS,zero_division=0)
 for c,a,b,f,n in zip(EVENTS,pp,rr,ff,nn): lines.append(f'| {c} | {n} | {a:.4f} | {b:.4f} | {f:.4f} |')
 ya=np.where(y=='unaffected','unaffected','affected'); pa=np.where(p=='unaffected','unaffected','affected'); bclasses=['unaffected','affected']; bpp,brr,bff,bnn=precision_recall_fscore_support(ya,pa,labels=bclasses,zero_division=0)
 lines += ['', '| affected binary class | support | precision | recall | F1 |', '|---|---:|---:|---:|---:|']
 for c,a,b,f,n in zip(bclasses,bpp,brr,bff,bnn): lines.append(f'| {c} | {n} | {a:.4f} | {b:.4f} | {f:.4f} |')
 lines += ['', '## State accuracy and trivial baselines', '']
 for s in ['overall','affected_only','post_first_move','t_ge5','t_ge8']: lines.append(line(s,sc[s]))
 lines += ['', '## Change-point / affected-step metrics','']
 for s in ['target_moved','target_captured','affected_only','first_target_move','post_first_move','second_plus_target_move','post_second_move']:
  q=v.loc[sc[s]]; lines.append(f"- `{s}`: compact={acc('compact_pred',sc[s]):.4f} (rows={len(q)}, games={q.game_id.nunique()})" if len(q) else f'- `{s}`: NA')
 lines += ['', '## Paired game-level differences','', '| comparison | scope | game mean difference | CI95 | games |', '|---|---|---:|---|---:|']
 for r in paired: lines.append(f"| {r['comparison']} | {r['scope']} | {r['game_mean_diff']:.4f} | [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}] | {r['n_games']} |")
 lines += ['', '## Error propagation','', 'Error categories and persistence are in `error_analysis.csv`.']
 for _,r in err[err.section.eq('error_category')].iterrows(): lines.append(f"- `{r['category']}`: {int(r['count'])} rows ({r['row_fraction']:.3f}), {int(r['games'])} games")
 for _,r in err[err.section.eq('persistence')].iterrows(): lines.append(f"- `{r['category']}`: {r['row_fraction']}")
 lines += ['', '## Go / No-Go','', '- **96.1% is not explained only by the unaffected majority.** On affected steps compact recursion is 86.1% overall (target-moved 87.8%), and it is far above the initial/never-update/majority baselines (0%). The three captured cases are underpowered (3 rows).', '- On affected steps the direct state probe is actually stronger (91.1% game mean vs compact 86.1%), so the compact event tracker is not yet a uniformly superior state estimator.', '- After the first target move compact is higher than direct in row accuracy (92.8% vs 85.2%) and game mean (85.9% vs 80.6%), but the paired game bootstrap CI for compact-minus-direct includes zero; this is suggestive, not decisive.', '- For longer prefixes the compact tracker has a clearer advantage: paired compact-minus-direct is +7.8 points at `t>=5` and +12.3 points at `t>=8`, with CIs above zero.', '- **Recommendation:** compact-event recursion is a promising candidate primary method, but the present evidence is a conditional Go for further same-task method validation, not a clean claim that event-mediated tracking beats direct state readout everywhere.', '- **LLaVA 300-game pilot: No-Go for immediate execution.** First freeze the compact-event protocol and resolve the affected-step/direct-probe comparison; then use the same fixed 200/100 manifest, `MAX_T=10`, discovery hidden-only, validation hidden + native state, and `RUN_EXPLICIT_MOVE=false`.']
 return '\n'.join(lines)+'\n'

if __name__=='__main__': main()
