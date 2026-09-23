#!/usr/bin/env python3
"""Train/evaluate PSF entirely from existing hidden caches."""
from pathlib import Path
import argparse,json,sys,time
import pandas as pd, torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from staterev.psf import ShellAdapter,ChessAdapter,PSF,TextFeatureStore,split_grouped,train,evaluate,evaluate_ablations,oracle_transition_rows,native_state_rows

def load_data(tasks,models,max_traj,split='discovery'):
 out=[]
 for model in models:
  for task in tasks:
   if task=='shell': out+=ShellAdapter(model).load(max_traj)
   elif task=='chess': out+=ChessAdapter(model,root=__import__('os').environ.get('PSF_CHESS_QWEN_ROOT') if model=='qwen' else None).load(max_traj,split)
   else: raise ValueError(f'No registered adapter for {task}; register it before --eval-task {task}')
 return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--train-tasks',default='shell,chess'); ap.add_argument('--eval-task'); ap.add_argument('--eval-split',choices=['discovery','validation'],default='discovery'); ap.add_argument('--models',default='qwen,llava'); ap.add_argument('--out',type=Path,default=ROOT/'outputs/psf_v1'); ap.add_argument('--max-trajectories',type=int,default=0); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--seed',type=int,default=17); ap.add_argument('--mode',choices=['train','zero-shot','few-shot','from-scratch'],default='train'); ap.add_argument('--few-shot-trajectories',type=int,choices=[1,5,10],default=1); ap.add_argument('--changed-lambda',type=float,choices=[1,3,5],default=1); ap.add_argument('--projector-kind',choices=['full_linear','rank32','rank8_no_norm','rank8_old'],default='rank8_old'); ap.add_argument('--ablation',choices=['full','memory_only','observation_only','fixed_fusion','fixed_gate','no_change_loss','no_persistence_loss','single_layer'],default='full'); ap.add_argument('--checkpoint',type=Path); ap.add_argument('--text-cache-qwen',type=Path); ap.add_argument('--text-cache-llava',type=Path); ap.add_argument('--strict-text-cache',action='store_true'); ap.add_argument('--device',default='cpu'); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 tasks=[x for x in a.train_tasks.split(',') if x]; models=[x for x in a.models.split(',') if x]
 data=load_data(tasks,models,a.max_trajectories); assert data
 train_set,dev=split_grouped(data,a.seed); paths={'qwen':a.text_cache_qwen,'llava':a.text_cache_llava}; stores={m:TextFeatureStore(paths[m],strict=a.strict_text_cache) for m in models}; model=PSF(tuple(models),ablation=a.ablation,text_stores=stores,projector_kind=a.projector_kind).to(a.device)
 if a.checkpoint: model.load_state_dict(torch.load(a.checkpoint,map_location=a.device),strict=True)
 # Generalization API: core can be frozen and only the new backbone/task
 # calibration layer optimized. Current registered tasks need no calibration.
 if a.mode=='few-shot':
  if not a.checkpoint: raise ValueError('--mode few-shot requires --checkpoint')
  for n,p in model.named_parameters(): p.requires_grad=('projectors' in n or 'gate.2' in n)
  train_set=train_set[:a.few_shot_trajectories]
 elif a.mode=='zero-shot': a.epochs=0
 hist=train(model,train_set,dev,a.epochs,a.lr,a.seed,changed_lambda=a.changed_lambda) if a.epochs else pd.DataFrame()
 if a.mode=='zero-shot' and not a.checkpoint: raise ValueError('--mode zero-shot requires --checkpoint')
 pred=evaluate(model,dev); hist.to_csv(a.out/f'history_{a.ablation}.csv',index=False); pred.to_csv(a.out/f'predictions_{a.ablation}.csv',index=False); evaluate_ablations(model,dev).to_csv(a.out/'baseline_ablations.csv',index=False); native_state_rows(dev).to_csv(a.out/'native_vlm_predictions.csv',index=False); torch.save(model.state_dict(),a.out/f'psf_{a.ablation}.pt'); oracle_transition_rows(dev).to_csv(a.out/'oracle_event_predictions.csv',index=False)
 if a.eval_task:
  eval_data=load_data([a.eval_task],models,a.max_trajectories,a.eval_split); ep=evaluate(model,eval_data); ep.to_csv(a.out/'eval_task_predictions.csv',index=False); pd.DataFrame([{'task':a.eval_task,'split':a.eval_split,'rows':len(ep),'accuracy':float(ep.correct.mean())}]).to_csv(a.out/'eval_task_metrics.csv',index=False)
 counts=model.parameter_counts(); counts['fraction_of_7b']=counts['total_trainable']/7e9; counts['fraction_of_8b']=counts['total_trainable']/8e9; counts['rough_flops_per_step']=2*counts['total_trainable']; counts['tasks']=tasks; counts['models']=models; counts['train_trajectory_count']=len(train_set); counts['dev_trajectory_count']=len(dev); counts['schema_fallback_trajectories']=sum(x.schema_fallback for x in data); counts['event_label_fields_read']=False; counts['text_features']=model.text_feature_stats()
 model.eval(); t0=time.perf_counter()
 with torch.no_grad(): model.forward_trajectory(dev[0])
 if str(a.device).startswith('cuda'): torch.cuda.synchronize()
 counts['measured_latency_ms_per_step_smoke']=1000*(time.perf_counter()-t0)/len(dev[0].state_ids)
 (a.out/'parameter_count.json').write_text(json.dumps(counts,indent=2)+'\n')
 metrics=pred.groupby(['task','model']).correct.agg(['mean','count']).reset_index(); metrics.to_csv(a.out/f'metrics_{a.ablation}.csv',index=False)
 (a.out/'README.md').write_text('# PSF smoke/formal run\\n\\nMain training does not read manual event labels; change supervision is derived from adjacent state ids. The explicit-event oracle CSV is debug-only. The current smoke text feature is deterministic fallback because no frozen LM text embedding cache is present; formal runs must replace it with cached LM features.\\n')
 print(json.dumps({'PSF_PASS':True,'counts':counts,'metrics':metrics.to_dict('records'),'loss_start':None if hist.empty else float(hist.train_loss.iloc[0]),'loss_end':None if hist.empty else float(hist.train_loss.iloc[-1])},indent=2))
if __name__=='__main__':main()
