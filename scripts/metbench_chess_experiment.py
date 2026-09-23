#!/usr/bin/env python3
"""Runnable MET-Bench Chess experiment pipeline.

Stages are deliberately resumable: prepare (manifest), extract (frozen model
native answers + state-question hidden vectors), fit (discovery probes and
learned updater), evaluate (validation metrics). ``--max-games`` is intended
for smoke tests; without it the full manifest is processed.
"""
from __future__ import annotations
import argparse, hashlib, json, re, os
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
SQUARES=[f"{f}{r}" for r in range(1,9) for f in "abcdefgh"]
SI={s:i for i,s in enumerate(SQUARES)}

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def load_ref(ref, cache):
 m=re.match(r"parquet://(.+)#row=(\d+)&action=(\d+)",str(ref))
 if not m: raise ValueError(f"invalid image reference: {ref}")
 key=(m.group(1),int(m.group(2)))
 if key not in cache:
  tab=pq.read_table(ROOT/m.group(1),columns=['image_actions']).slice(key[1],1)
  cache[key]=tab['image_actions'][0].as_py()
 return cache[key][int(m.group(3))]['bytes']

def model_load(kind, model_dir):
 from transformers import AutoProcessor
 if kind=='qwen':
  from transformers import Qwen3VLForConditionalGeneration as Model
 else:
  from transformers import LlavaNextVideoForConditionalGeneration as Model
 model=Model.from_pretrained(model_dir,device_map='auto',torch_dtype='auto').eval()
 return model,AutoProcessor.from_pretrained(model_dir)

def prompt(row, question):
 return [{"role":"system","content":"Answer the chess question concisely."},{"role":"user","content":[{"type":"video","video":question[0]},{"type":"text","text":question[1]}]}]

def forward_one(model,processor,kind,frames,text,generate=False,layer_ids=None):
 clip=np.stack(frames)
 if len(clip)<2: clip=np.concatenate([clip,clip],axis=0)
 msgs=prompt(None,(clip,text))
 try: inp=processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
 except Exception: inp=processor(text=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True),videos=clip,return_tensors='pt')
 if 'pixel_values_videos' not in inp: inp=processor(text=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True),videos=clip,return_tensors='pt')
 dev=next(model.parameters()).device; inp={k:(v.to(dev) if hasattr(v,'to') else v) for k,v in inp.items()}
 with torch.inference_mode():
  if generate:
   out=model.generate(**inp,max_new_tokens=16,do_sample=False); txt=processor.batch_decode(out[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip(); return txt,None
  base=model.model(**inp,output_hidden_states=True,return_dict=True) if kind=='qwen' else model(**inp,output_hidden_states=True,return_dict=True,logits_to_keep=1)
  pos=inp['input_ids'].shape[1]-1; hs=base.hidden_states
  ids=list(range(len(hs))) if layer_ids is None else [i for i in layer_ids if i < len(hs)]
  return None,np.stack([hs[i][0,pos].float().cpu().numpy() for i in ids])

def pilot_manifest(args):
 rng=np.random.default_rng(args.seed); df=pd.read_csv(args.manifest)
 games={s:sorted(df.loc[df.protocol_split.eq(s),'game_id'].astype(str).unique()) for s in ('discovery','validation')}
 chosen={s:rng.choice(v,size=min(len(v),getattr(args,'max_'+s+'_games')),replace=False).tolist() for s,v in games.items()}
 out=df[df.game_id.astype(str).isin(set(chosen['discovery'])|set(chosen['validation']))].copy()
 out=out[out.t.le(args.max_t)].sort_values(['game_id','t']); out.to_csv(args.out,index=False)
 meta={'seed':args.seed,'max_discovery_games':args.max_discovery_games,'max_validation_games':args.max_validation_games,'max_t':args.max_t,'games':{k:len(v) for k,v in chosen.items()},'prefixes':int(len(out)),'discovery_games':chosen['discovery'],'validation_games':chosen['validation']}
 args.out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta))

def extract(args):
 out=args.out; phase=args.phase; tag=f'_{phase}' if phase!='all' else ''; marker=out/f'extract_{args.model}{tag}_{args.shard_index}.complete.json'; cache={}; rows=[]; hidden={}
 if marker.exists() and not args.overwrite:
  print(json.dumps({'EXTRACT_ALREADY_COMPLETE':True,'marker':str(marker)})); return
 manifest=pd.read_csv(args.manifest,usecols=['game_id','t','protocol_split','initial_state','current_state','image_path','src','dst','initial_square'],chunksize=100000)
 model,processor=model_load(args.model,args.model_dir)
 wanted=[]
 for c in manifest:
  for gid in c.game_id.astype(str).unique():
   if int(hashlib.sha256(gid.encode()).hexdigest()[:8],16)%args.num_shards==args.shard_index and (args.max_games<=0 or len(wanted)<args.max_games): wanted.append(gid)
  if args.max_games>0 and len(wanted)>=args.max_games: break
 wanted=set(wanted)
 manifest=pd.read_csv(args.manifest,usecols=['game_id','t','protocol_split','initial_state','current_state','image_path','src','dst','initial_square'])
 for gid,g in manifest[manifest.game_id.astype(str).isin(wanted)].groupby('game_id',sort=True):
  if phase!='all' and str(g.protocol_split.iloc[0]) != phase: continue
  g=g.sort_values('t');
  if args.max_t>0: g=g.head(args.max_t)
  frames=[]
  for r in g.itertuples(index=False):
   raw=load_ref(r.image_path,cache); frames.append(np.asarray(Image.open(__import__('io').BytesIO(raw)).convert('RGB')))
   text=f"Initial FEN: {r.initial_state}. Track the white knight initially on g1 through move {r.t}. Where is it now? Answer one square or captured."
   _,h=forward_one(model,processor,args.model,frames,text,False,args.layer_ids); hidden[f'{gid}_t{int(r.t)}']=h
   state_text,_=(forward_one(model,processor,args.model,frames,text,True) if args.run_native_state else ('',None))
   move_text,_=(forward_one(model,processor,args.model,frames,f"Initial FEN: {r.initial_state}. What is the latest move? Answer source destination.",True) if args.run_explicit_move else ('',None))
   rows.append({'game_id':gid,'t':int(r.t),'protocol_split':r.protocol_split,'gt_state':r.current_state,
                'gt_src':r.src,'gt_dst':r.dst,'initial_square':r.initial_square,
                'native_state_text':state_text,'native_move_text':move_text,
                'question_type_state':'state','question_type_move':'move'})
 np.savez_compressed(out/f'hidden_{args.model}{tag}_{args.shard_index}.npz',**hidden); pd.DataFrame(rows).to_csv(out/f'behavior_{args.model}{tag}_{args.shard_index}.csv',index=False)
 (out/f'hidden_{args.model}{tag}_{args.shard_index}.json').write_text(json.dumps({'question_type':'state','feature_source':'state-question forward hidden states','model':args.model,'phase':phase,'rows':len(rows),'run_native_state':args.run_native_state,'run_explicit_move':args.run_explicit_move},indent=2)+'\n')
 marker.write_text(json.dumps({'status':'complete','model':args.model,'phase':phase,'shard_index':args.shard_index,'num_shards':args.num_shards,'games':len(wanted),'rows':len(rows),'hidden_keys':len(hidden),'layer_ids':args.layer_ids,'manifest_sha256':sha(args.manifest)},indent=2)+'\n')
 print(json.dumps({'EXTRACT_PASS':True,'games':len(wanted),'rows':len(rows),'marker':str(marker)}))

def merge(args):
 out=args.out; phase=args.phase; tag=f'_{phase}' if phase!='all' else ''; frames=[]; hidden={}
 layer_ids=None
 for i in range(args.num_shards):
  m=out/f'extract_{args.model}{tag}_{i}.complete.json';
  if not m.exists(): raise FileNotFoundError(m)
  meta=json.loads(m.read_text()); layer_ids=meta.get('layer_ids') if layer_ids is None else layer_ids
  if layer_ids != meta.get('layer_ids'): raise AssertionError('layer mismatch across shards')
  frames.append(pd.read_csv(out/f'behavior_{args.model}{tag}_{i}.csv'))
  with np.load(out/f'hidden_{args.model}{tag}_{i}.npz') as z: hidden.update({k:z[k] for k in z.files})
 pd.concat(frames,ignore_index=True).to_csv(out/f'behavior_{args.model}{tag}.csv',index=False); np.savez_compressed(out/f'hidden_{args.model}{tag}.npz',**hidden)
 (out/f'hidden_{args.model}{tag}.json').write_text(json.dumps({'question_type':'state','layer_ids':layer_ids,'rows':len(hidden),'hidden_shape':list(next(iter(hidden.values())).shape)},indent=2)+'\n')
 (out/f'merge_{args.model}{tag}.complete.json').write_text(json.dumps({'status':'complete','phase':phase,'rows':sum(map(len,frames)),'hidden_keys':len(hidden),'num_shards':args.num_shards,'layer_ids':layer_ids},indent=2)+'\n'); print(json.dumps({'MERGE_PASS':True,'phase':phase,'rows':sum(map(len,frames)),'hidden_keys':len(hidden),'layer_ids':layer_ids}))

def _sq(x):
 s=str(x).lower()
 if any(w in s for w in ('captured','taken','off the board','not on the board')): return 'captured'
 m=re.search(r'\b([a-h][1-8])\b',s); return m.group(1) if m else None
def _move(x):
 s=str(x).lower().replace('−','-').replace('–','-')
 m=re.search(r'\b([a-h][1-8])\s*(?:[-–>]|to|:)\s*([a-h][1-8])\b',s)
 if not m: m=re.search(r'\b([a-h][1-8])([a-h][1-8])\b',s)
 return (m.group(1),m.group(2)) if m else (None,None)

def _onehot_move(move):
 p=np.zeros(64,dtype=np.float32)
 if move and move[0] in SI and move[1] in SI: p[SI[move[0]]]=1.0
 else: p.fill(1.0/64.0)
 q=np.zeros(64,dtype=np.float32)
 if move and move[0] in SI and move[1] in SI: q[SI[move[1]]]=1.0
 else: q.fill(1.0/64.0)
 return p,q

def _expand64(p):
 p=np.asarray(p,dtype=float).ravel(); out=np.zeros(64,dtype=float)
 if len(p)==64: out=p
 elif len(p)>0: out[:min(64,len(p))]=p[:64]
 else: out.fill(1/64)
 if out.sum()<=0: out.fill(1/64)
 return out/out.sum()

def _cluster_metric(values, games, seed=7, n_boot=1000):
 values=np.asarray(values,dtype=float); games=np.asarray(games,dtype=str)
 if len(values)==0: return {'n_rows':0,'n_games':0,'mean':None,'ci95':[None,None]}
 per=pd.DataFrame({'value':values,'game_id':games}).groupby('game_id',sort=True).value.mean().to_numpy()
 rng=np.random.default_rng(seed); boot=per[rng.integers(0,len(per),size=(n_boot,len(per)))].mean(1)
 return {'n_rows':int(len(values)),'n_games':int(len(per)),'mean':float(per.mean()),'ci95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}

def fit(args):
 from metbench_chess_single_piece import fit_probe, train_state_updater
 import joblib
 hidden=args.out/f'hidden_{args.model}_discovery.npz'; artifact=args.out/f'{args.model}_square_probes.joblib'
 fit_probe(args.manifest,hidden,artifact,args.candidate_layers,args.candidate_c)
 probe=joblib.load(artifact); z_phase={p:np.load(args.out/f'hidden_{args.model}_{p}.npz') for p in ('discovery','validation')}; keys={p:set(z_phase[p].files) for p in z_phase}; srcp={}; dstp={}
 for phase in ('discovery','validation'):
  beh=pd.read_csv(args.out/f'behavior_{args.model}_{phase}.csv')
  for r in beh.itertuples(index=False):
   k=f'{r.game_id}_t{int(r.t)}'
   phase=str(r.protocol_split)
   if phase not in z_phase or k not in keys[phase]: continue
   x=np.asarray(z_phase[phase][k]); x=x[probe['layer']] if x.ndim==2 else x
   src_raw=probe['src_probe'].predict_proba(x[None])[0]; dst_raw=probe['dst_probe'].predict_proba(x[None])[0]
   src_full=np.zeros(64,dtype=np.float32); dst_full=np.zeros(64,dtype=np.float32)
   for cls,pv in zip(probe['src_probe'].classes_,src_raw): src_full[int(cls)]=pv
   for cls,pv in zip(probe['dst_probe'].classes_,dst_raw): dst_full[int(cls)]=pv
   srcp[k]=_expand64(src_full); dstp[k]=_expand64(dst_full)
 np.savez_compressed(args.out/f'{args.model}_hidden_src_probs.npz',**srcp); np.savez_compressed(args.out/f'{args.model}_hidden_dst_probs.npz',**dstp)
 # Fit the small updater on discovery games; optionally cap for a practical smoke run.
 if args.max_fit_games>0:
  mf=pd.read_csv(args.manifest); gs=mf.loc[mf.protocol_split.eq('discovery'),'game_id'].drop_duplicates().head(args.max_fit_games); mf=mf[mf.game_id.isin(set(gs))]; tmp=args.out/f'._fit_{args.model}.csv'; mf.to_csv(tmp,index=False); fit_manifest=tmp
 else: fit_manifest=args.manifest
 model=train_state_updater(fit_manifest,srcp,dstp,args.out/f'{args.model}_updater.pt',epochs=args.epochs)
 print(json.dumps({'FIT_PASS':True,'probe':str(artifact),'updater':str(args.out/f'{args.model}_updater.pt'),'parameter_count':model.parameter_count,'probe_rows':len(srcp)}))

def evaluate(args):
 from metbench_chess_single_piece import ChessStateUpdater, oracle_transition, SI, SQUARES
 import joblib
 beh=pd.read_csv(args.out/f'behavior_{args.model}_validation.csv')
 mf=pd.read_csv(args.manifest,usecols=['game_id','t','protocol_split','current_state','src','dst','initial_square'])
 beh=beh.merge(mf,on=['game_id','t'],how='inner',suffixes=('','_manifest'))
 probe=joblib.load(args.out/f'{args.model}_square_probes.joblib')
 zs=np.load(args.out/f'{args.model}_hidden_src_probs.npz'); zd=np.load(args.out/f'{args.model}_hidden_dst_probs.npz')
 up=ChessStateUpdater(); up.load_state_dict(torch.load(args.out/f'{args.model}_updater.pt',map_location='cpu')['state_dict']); up.eval()
 rows=[]; fit_games=set(probe.get('fit_games',[]))
 for gid,g in beh.sort_values('t').groupby('game_id',sort=True):
  if str(g.protocol_split.iloc[0]) != args.eval_split: continue
  # All prefixes of a game must be present; silently dropping a partial game
  # would bias trajectory statistics, so mark it and skip the whole trajectory.
  keys=[f'{gid}_t{int(t)}' for t in g.t]
  if any(k not in zs or k not in zd for k in keys): continue
  init=np.zeros(65,dtype=float); init[SI[str(g.initial_square.iloc[0])]]=1.0
  b_hidden=init.copy(); b_explicit=init.copy(); b_oracle=init.copy(); b_hidden_oracle=init.copy()
  for r in g.itertuples(index=False):
   k=f'{gid}_t{int(r.t)}'; hs=_expand64(zs[k]); hd=_expand64(zd[k]);
   em=_move(r.native_move_text); es=_onehot_move(em); truth=_onehot_move((str(r.src),str(r.dst)))
   with torch.no_grad():
    b_hidden=up(torch.tensor(b_hidden[None],dtype=torch.float32),torch.tensor(hs[None],dtype=torch.float32),torch.tensor(hd[None],dtype=torch.float32))[0].numpy()
    b_explicit=up(torch.tensor(b_explicit[None],dtype=torch.float32),torch.tensor(es[0][None],dtype=torch.float32),torch.tensor(es[1][None],dtype=torch.float32))[0].numpy()
    b_oracle=up(torch.tensor(b_oracle[None],dtype=torch.float32),torch.tensor(truth[0][None],dtype=torch.float32),torch.tensor(truth[1][None],dtype=torch.float32))[0].numpy()
   hb=oracle_transition(b_hidden_oracle,int(hs.argmax()),int(hd.argmax())); b_hidden_oracle=hb
   native=_sq(r.native_state_text) or 'unparsed'
   hidden_pred='captured' if b_hidden[64]>=b_hidden[:64].max() else SQUARES[int(b_hidden[:64].argmax())]
   explicit_pred='captured' if b_explicit[64]>=b_explicit[:64].max() else SQUARES[int(b_explicit[:64].argmax())]
   oracle_pred='captured' if b_oracle[64]>=b_oracle[:64].max() else SQUARES[int(b_oracle[:64].argmax())]
   hidden_oracle_pred='captured' if b_hidden_oracle[64]>=b_hidden_oracle[:64].max() else SQUARES[int(b_hidden_oracle[:64].argmax())]
   rows.append({'game_id':str(gid),'t':int(r.t),'protocol_split':r.protocol_split,'gt_state':r.current_state,
    'gt_src':r.src,'gt_dst':r.dst,'native_state':native,'explicit_src':em[0] or 'unparsed','explicit_dst':em[1] or 'unparsed',
    'hidden_src':SQUARES[int(hs.argmax())],'hidden_dst':SQUARES[int(hd.argmax())],
    'native_state_correct':native==r.current_state,'explicit_move_correct':em==(str(r.src),str(r.dst)),
    'hidden_move_correct':(SQUARES[int(hs.argmax())]==str(r.src) and SQUARES[int(hd.argmax())]==str(r.dst)),
    'hidden_learned':hidden_pred,'explicit_learned':explicit_pred,'oracle_learned':oracle_pred,'hidden_oracle':hidden_oracle_pred})
 out=pd.DataFrame(rows); metrics={}
 if len(out):
  out['hidden_learned_correct']=out.hidden_learned.eq(out.gt_state); out['explicit_learned_correct']=out.explicit_learned.eq(out.gt_state); out['oracle_learned_correct']=out.oracle_learned.eq(out.gt_state); out['hidden_oracle_correct']=out.hidden_oracle.eq(out.gt_state)
  for col in ['native_state_correct','explicit_move_correct','hidden_move_correct','explicit_learned_correct','hidden_learned_correct','oracle_learned_correct','hidden_oracle_correct']:
   metrics[col]={}
   for name,mask in [('overall',np.ones(len(out),dtype=bool)),*[(f't{t}',out.t.eq(t).to_numpy()) for t in range(1,11)],*[(f't_ge{t}',(out.t>=t).to_numpy()) for t in (3,5,8)]]:
    x=out.loc[mask]; metrics[col][name]=_cluster_metric(x[col].to_numpy(float),x.game_id.to_numpy(),seed=17+len(metrics[col]))
 out.to_csv(args.out/f'evaluation_{args.model}_{args.eval_split}.csv',index=False)
 payload={'schema_version':1,'model':args.model,'eval_split':args.eval_split,'rows':len(out),'games':int(out.game_id.nunique()) if len(out) else 0,'probe_fit_games':len(fit_games),'probe_eval_overlap':len(fit_games & set(out.game_id.astype(str))) if len(out) else 0,'metrics':metrics}
 (args.out/f'evaluation_{args.model}_{args.eval_split}.json').write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps({'EVALUATE_PASS':True,**payload}))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=ROOT/'outputs/metbench_chess/single_piece_tracking_v1'); ap.add_argument('--manifest',type=Path); sub=ap.add_subparsers(dest='cmd',required=True)
 p=sub.add_parser('pilot-manifest'); p.add_argument('--source-manifest',dest='manifest',type=Path,required=True); p.add_argument('--max-discovery-games',type=int,default=200); p.add_argument('--max-validation-games',type=int,default=100); p.add_argument('--max-t',type=int,default=10); p.add_argument('--seed',type=int,default=20260918); p.add_argument('--pilot-out',type=Path,required=True)
 e=sub.add_parser('extract'); e.add_argument('--model',choices=['qwen','llava'],required=True); e.add_argument('--model-dir',type=Path,required=True); e.add_argument('--shard-index',type=int,required=True); e.add_argument('--num-shards',type=int,required=True); e.add_argument('--max-games',type=int,default=0); e.add_argument('--max-t',type=int,default=10); e.add_argument('--phase',choices=['all','discovery','validation'],default='all'); e.add_argument('--layer-ids',default='9,18,27,36'); e.add_argument('--run-native-state',action='store_true'); e.add_argument('--run-explicit-move',action='store_true'); e.add_argument('--overwrite',action='store_true')
 m=sub.add_parser('merge'); m.add_argument('--model',choices=['qwen','llava'],required=True); m.add_argument('--phase',choices=['all','discovery','validation'],default='all'); m.add_argument('--num-shards',type=int,required=True)
 f=sub.add_parser('fit'); f.add_argument('--model',choices=['qwen','llava'],required=True); f.add_argument('--candidate-layers',default='0'); f.add_argument('--candidate-c',default='1.0'); f.add_argument('--epochs',type=int,default=3); f.add_argument('--max-fit-games',type=int,default=0)
 v=sub.add_parser('evaluate'); v.add_argument('--model',choices=['qwen','llava'],required=True); v.add_argument('--eval-split',choices=['validation','holdout_test'],default='validation')
 a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True); a.manifest=a.manifest or a.out/'manifest.csv'
 if a.cmd=='pilot-manifest': a.out=a.pilot_out; a.out.parent.mkdir(parents=True,exist_ok=True); pilot_manifest(a)
 elif a.cmd=='extract':
  a.layer_ids=[int(x) for x in a.layer_ids.split(',') if x.strip()]
  # Backward-compatible full-run mode keeps the original native baselines.
  # Pilot phases explicitly opt into the cheaper state-only discovery path.
  if a.phase=='all': a.run_native_state=True; a.run_explicit_move=True
  extract(a)
 elif a.cmd=='merge': merge(a)
 elif a.cmd=='fit': fit(a)
 else: evaluate(a)
if __name__=='__main__': main()
