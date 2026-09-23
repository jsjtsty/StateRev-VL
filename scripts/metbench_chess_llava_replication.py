#!/usr/bin/env python3
"""Strict, fixed-manifest LLaVA Chess replication.

Extraction is the only stage that touches the VLM.  Discovery stores only
state-question hidden vectors (selected candidate layers); validation stores
state hidden vectors plus native state generation.  No move-question and no
64x64 move probe is used.
"""
from __future__ import annotations
import argparse, hashlib, io, json, re
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
import pyarrow.parquet as pq
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support, f1_score

ROOT=Path(__file__).resolve().parents[1]
SQUARES=[f"{f}{r}" for r in range(1,9) for f in "abcdefgh"]
SI={s:i for i,s in enumerate(SQUARES)}
EVENTS=['unaffected','target_moved','target_captured']

def row_key(r): return f'{r.game_id}_t{int(r.t)}'
def sha256_file(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def load_ref(ref, cache):
 m=re.match(r'parquet://(.+)#row=(\d+)&action=(\d+)',str(ref))
 if not m: raise ValueError(f'bad image ref {ref}')
 key=(m.group(1),int(m.group(2)))
 if key not in cache:
  tab=pq.read_table(ROOT/m.group(1),columns=['image_actions']).slice(key[1],1)
  cache[key]=tab['image_actions'][0].as_py()
 return cache[key][int(m.group(3))]['bytes']

def load_model(model_dir):
 from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration
 model=LlavaNextVideoForConditionalGeneration.from_pretrained(model_dir,device_map='auto',torch_dtype='auto').eval()
 return model,AutoProcessor.from_pretrained(model_dir)

def inputs(processor,clip,text):
 if len(clip)<2: clip=np.concatenate([clip,clip],axis=0)
 msgs=[{'role':'system','content':'Answer the chess question concisely.'},{'role':'user','content':[{'type':'video','video':clip},{'type':'text','text':text}]}]
 try: x=processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
 except Exception:
  prompt=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True); x=processor(text=prompt,videos=clip,return_tensors='pt')
 if 'pixel_values_videos' not in x:
  prompt=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True); x=processor(text=prompt,videos=clip,return_tensors='pt')
 return x

def forward(model,processor,clip,text,need_hidden=True,generate=False,layer_ids=None):
 x=inputs(processor,clip,text); dev=next(model.parameters()).device; x={k:(v.to(dev) if hasattr(v,'to') else v) for k,v in x.items()}
 with torch.inference_mode():
  if generate:
   out=model.generate(**x,max_new_tokens=16,do_sample=False)
   return processor.batch_decode(out[:,x['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip(),None
  o=model(**x,output_hidden_states=True,return_dict=True,logits_to_keep=1)
  pos=x['input_ids'].shape[1]-1; hs=o.hidden_states
  ids=list(range(len(hs))) if layer_ids is None else [i for i in layer_ids if i<len(hs)]
  return None,np.stack([hs[i][0,pos].float().cpu().numpy() for i in ids])

def validate_manifest(path, strict=True):
 d=pd.read_csv(path); assert d.protocol_split.value_counts().to_dict()=={'discovery':1904,'validation':948}, d.protocol_split.value_counts()
 assert d[d.protocol_split=='discovery'].game_id.nunique()==200
 assert d[d.protocol_split=='validation'].game_id.nunique()==100
 assert int(d.t.max())<=10
 assert not set(d[d.protocol_split=='discovery'].game_id)&set(d[d.protocol_split=='validation'].game_id)
 return d

def extract(a):
 marker=a.out/f'extract_llava_{a.phase}_{a.shard_index}.complete.json'
 if marker.exists() and not a.overwrite: print(json.dumps({'EXTRACT_ALREADY_COMPLETE':True,'marker':str(marker)})); return
 manifest=validate_manifest(a.manifest); layer_ids=[int(x) for x in a.candidate_layers.split(',') if x.strip()]
 model,processor=load_model(a.model_dir); cache={};
 phase_games=manifest.loc[manifest.protocol_split.eq(a.phase),'game_id'].astype(str).unique()
 wanted=[g for g in phase_games if int(hashlib.sha256(g.encode()).hexdigest()[:8],16)%a.num_shards==a.shard_index]
 if a.max_games>0: wanted=wanted[:a.max_games]
 wanted=set(wanted)
 rows=[]; hidden={}
 for gid,g in manifest[manifest.game_id.astype(str).isin(wanted)].groupby('game_id',sort=True):
  if str(g.protocol_split.iloc[0])!=a.phase: continue
  frames=[]
  for r in g.sort_values('t').itertuples(index=False):
   frames.append(np.asarray(Image.open(io.BytesIO(load_ref(r.image_path,cache))).convert('RGB')))
   q=f'Initial FEN: {r.initial_state}. Track the white knight initially on g1 through move {int(r.t)}. Where is it now? Answer one square or captured.'
   text,h=forward(model,processor,np.stack(frames),q,need_hidden=True,layer_ids=layer_ids); hidden[row_key(r)]=h
   native=''
   if a.phase=='validation' and a.run_native_state:
    native,_=forward(model,processor,np.stack(frames),q,need_hidden=False,generate=True)
   rows.append({'game_id':str(gid),'t':int(r.t),'protocol_split':str(r.protocol_split),'gt_state':str(r.current_state),'gt_src':str(r.src),'gt_dst':str(r.dst),'initial_square':str(r.initial_square),'native_state_text':native,'question_type_state':'state','question_type_move':'disabled'})
 tag=f'llava_{a.phase}_{a.shard_index}'
 np.savez_compressed(a.out/f'hidden_{tag}.npz',**hidden); pd.DataFrame(rows).to_csv(a.out/f'behavior_{tag}.csv',index=False)
 meta={'status':'complete','model':'llava','phase':a.phase,'shard_index':a.shard_index,'num_shards':a.num_shards,'games':len(wanted),'rows':len(rows),'hidden_keys':len(hidden),'layer_ids':layer_ids,'run_native_state':bool(a.run_native_state),'run_explicit_move':False,'manifest_sha256':sha256_file(a.manifest)}
 if rows:
  actual_layers=int(next(iter(hidden.values())).shape[0]); assert actual_layers==len(layer_ids), (actual_layers,layer_ids)
 (a.out/f'hidden_{tag}.json').write_text(json.dumps({'question_type':'state','layer_ids':layer_ids,'rows':len(rows)},indent=2)+'\n'); marker.write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps({'EXTRACT_PASS':True,**meta}))

def merge(a):
 validate_manifest(a.manifest); hidden={}; frames=[]; layer_ids=None
 for i in range(a.num_shards):
  marker=a.out/f'extract_llava_{a.phase}_{i}.complete.json'; assert marker.exists(),marker
  meta=json.loads(marker.read_text()); assert meta['manifest_sha256']==sha256_file(a.manifest); layer_ids=meta['layer_ids'] if layer_ids is None else layer_ids; assert layer_ids==meta['layer_ids']
  frames.append(pd.read_csv(a.out/f'behavior_llava_{a.phase}_{i}.csv'))
  with np.load(a.out/f'hidden_llava_{a.phase}_{i}.npz') as z: hidden.update({k:z[k] for k in z.files})
 pd.concat(frames,ignore_index=True).sort_values(['game_id','t']).to_csv(a.out/f'behavior_{a.phase}.csv',index=False); np.savez_compressed(a.out/f'hidden_{a.phase}.npz',**hidden)
 (a.out/f'hidden_{a.phase}.json').write_text(json.dumps({'question_type':'state','layer_ids':layer_ids,'rows':len(hidden)},indent=2)+'\n'); (a.out/f'merge_{a.phase}.complete.json').write_text(json.dumps({'status':'complete','phase':a.phase,'rows':len(hidden),'games':len(set(pd.concat(frames).game_id)),'num_shards':a.num_shards,'manifest_sha256':sha256_file(a.manifest)},indent=2)+'\n'); print(json.dumps({'MERGE_PASS':True,'phase':a.phase,'rows':len(hidden),'games':len(set(pd.concat(frames).game_id))}))

def state_idx(s): return 64 if s=='captured' else SI[s]
def event_type(r,manifest):
 prev=manifest[(manifest.game_id==r.game_id)&(manifest.t==r.t-1)]
 prev_state='g1' if len(prev)==0 else prev.iloc[0].current_state
 if prev_state=='captured': return 'unaffected'
 if r.src==prev_state: return 'target_moved'
 if r.dst==prev_state and r.current_state=='captured': return 'target_captured'
 return 'unaffected'
def fit_probe(manifest,hidden,kind,layer_ids):
 d=manifest[manifest.protocol_split=='discovery'].copy(); tune=d.game_id.map(lambda x:int(hashlib.sha256(str(x).encode()).hexdigest()[:8],16)%5==0).to_numpy()
 candidates=[.01,.1,1.,10.]; best=None
 for li,layer in enumerate(layer_ids):
  X=np.stack([hidden[row_key(r)][li] for r in d.itertuples()])
  if kind=='state': y=np.array([state_idx(r.current_state) for r in d.itertuples()])
  elif kind=='event': y=np.array([EVENTS.index(event_type(r,manifest)) for r in d.itertuples()])
  else:
   d2=d[d.apply(lambda r:event_type(r,manifest)=='target_moved',axis=1)]; X=np.stack([hidden[row_key(r)][li] for r in d2.itertuples()]); y=np.array([SI[r.dst] for r in d2.itertuples()]); tune=d2.game_id.map(lambda x:int(hashlib.sha256(str(x).encode()).hexdigest()[:8],16)%5==0).to_numpy()
  for c in candidates:
   q=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000)).fit(X[~tune],y[~tune]); score=float((q.predict(X[tune])==y[tune]).mean()) if tune.any() else 0
   if best is None or score>best[0]: best=(score,layer,li,c,q)
 score,layer,li,c,_=best; X=np.stack([hidden[row_key(r)][li] for r in d.itertuples()])
 if kind=='state': y=np.array([state_idx(r.current_state) for r in d.itertuples()])
 elif kind=='event': y=np.array([EVENTS.index(event_type(r,manifest)) for r in d.itertuples()])
 else:
  d=d[d.apply(lambda r:event_type(r,manifest)=='target_moved',axis=1)]; X=np.stack([hidden[row_key(r)][li] for r in d.itertuples()]); y=np.array([SI[r.dst] for r in d.itertuples()])
 model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000)).fit(X,y); model.layer=layer; model.layer_index=li; model.C_selected=c; model.selection_score=score; return model
def probs(model,hidden,rows,nclasses):
 X=np.stack([hidden[row_key(r)][model.layer_index] for r in rows.itertuples()]); raw=model.predict_proba(X); out=np.zeros((len(rows),nclasses))
 for c,p in zip(model.classes_,raw.T): out[:,int(c)]=p
 return out/np.maximum(out.sum(1,keepdims=True),1e-12)
def parse_state(x):
 s=str(x).lower();
 if any(w in s for w in ['captured','taken','off the board','not on the board']): return 'captured'
 m=re.search(r'\b([a-h][1-8])\b',s); return m.group(1) if m else 'unparsed'
def trans(prev,event_p,dst_p):
 out=event_p[0]*prev; c=np.zeros(65); c[64]=1; out+=event_p[2]*c; moved=np.zeros(65); moved[:64]=dst_p; moved[64]=prev[64]; out+=event_p[1]*moved; return out/np.maximum(out.sum(),1e-12)
def summarize(d,col,mask=None):
 q=d if mask is None else d.loc[mask];
 if not len(q): return {'row_mean':None,'game_mean':None,'n_rows':0,'n_games':0}
 g=q.groupby('game_id').apply(lambda x:x[col].eq(x.gt_state).mean()); return {'row_mean':float(q[col].eq(q.gt_state).mean()),'game_mean':float(g.mean()),'n_rows':len(q),'n_games':len(g)}
def scopes(d):
 return {'overall':np.ones(len(d),bool),'affected_only':d.event_gt.ne('unaffected').to_numpy(),'post_first_move':(d.first_move_t.notna()&(d.t>d.first_move_t)).to_numpy(),'t_ge3':(d.t>=3).to_numpy(),'t_ge5':(d.t>=5).to_numpy(),'t_ge8':(d.t>=8).to_numpy(),'t10':(d.t==10).to_numpy()}

def evaluate(a):
 manifest=validate_manifest(a.manifest); hidden={}
 for phase in ('discovery','validation'):
  with np.load(a.out/f'hidden_{phase}.npz') as z: hidden.update({k:z[k] for k in z.files})
 meta=json.loads((a.out/'hidden_discovery.json').read_text()); layer_ids=meta['layer_ids']; event=fit_probe(manifest,hidden,'event',layer_ids); dst=fit_probe(manifest,hidden,'dst',layer_ids); state=fit_probe(manifest,hidden,'state',layer_ids)
 discovery=manifest[manifest.protocol_split=='discovery'].copy(); val=manifest[manifest.protocol_split=='validation'].copy().sort_values(['game_id','t']); X=np.stack([hidden[row_key(r)][state.layer_index] for r in val.itertuples()]); direct_p=probs(state,hidden,val,65); event_p=probs(event,hidden,val,3); dst_p=probs(dst,hidden,val,64)
 beh=pd.read_csv(a.out/'behavior_validation.csv',usecols=['game_id','t','native_state_text'])
 val=val.merge(beh,on=['game_id','t'],validate='one_to_one').sort_values(['game_id','t']).reset_index(drop=True)
 val['gt_state']=val.current_state; val['event_gt']=[event_type(r,manifest) for r in val.itertuples()]; val['event_pred']=[EVENTS[i] for i in event_p.argmax(1)]; val['dst_pred']=[SQUARES[i] for i in dst_p.argmax(1)]; val['direct_p']=list(direct_p); val['event_p']=list(event_p); val['dst_p']=list(dst_p); val['native_state']=val['native_state_text'].map(parse_state)
 # Recursive methods per game: hidden compact tracker, GT-event oracle tracker.
 rec=[]
 for gid,g in val.groupby('game_id',sort=True):
  b=np.eye(65)[SI['g1']]; oracle=np.eye(65)[SI['g1']]
  for r in g.sort_values('t').itertuples():
   b=trans(b,r.event_p,r.dst_p); ep=np.zeros(3); ep[EVENTS.index(r.event_gt)]=1; dp=np.zeros(64); dp[SI[r.dst]]=1; oracle=trans(oracle,ep,dp)
   rec.append((r.Index,state_label(b.argmax()),state_label(oracle.argmax())))
 rec_map={i:(x,y) for i,x,y in rec}; val['tracker_state']=[rec_map[int(i)][0] for i in val.index]; val['oracle_tracker_state']=[rec_map[int(i)][1] for i in val.index]
 val['direct_only']=val.direct_p.map(lambda x:state_label(np.argmax(x))); val['tracker_only']=val.tracker_state; val['oracle_tracker']=val.oracle_tracker_state; val['gt_state']=val.current_state
 val=annotate(val)
 val['tracker_correct']=val.tracker_only.eq(val.gt_state)
 val['tracker_error_after_first']=False
 for gid,ix in val.groupby('game_id',sort=False).groups.items():
  arr=list(ix); bad=[j for j in arr if not bool(val.loc[j,'tracker_correct'])]
  if bad:
   first=bad[0]; tail=arr[arr.index(first):]
   val.loc[tail,'tracker_error_after_first']=~val.loc[tail,'tracker_correct']
 val.to_csv(a.out/'behavior_validation.csv',index=False)
 # Event metrics table.
 cm=confusion_matrix(val.event_gt,val.event_pred,labels=EVENTS); ep,er,ef,en=precision_recall_fscore_support(val.event_gt,val.event_pred,labels=EVENTS,zero_division=0)
 emrows=[]
 for i,c in enumerate(EVENTS):
  for j,p in enumerate(EVENTS): emrows.append({'section':'confusion','class':c,'predicted':p,'metric':'count','value':int(cm[i,j])})
 for c,p,r,f,n in zip(EVENTS,ep,er,ef,en): emrows += [{'section':'class','class':c,'metric':'precision','value':p,'support':n},{'section':'class','class':c,'metric':'recall','value':r,'support':n},{'section':'class','class':c,'metric':'f1','value':f,'support':n}]
 y=val.event_gt.to_numpy(); p=val.event_pred.to_numpy(); ya=np.where(y=='unaffected','unaffected','affected'); pa=np.where(p=='unaffected','unaffected','affected'); moved=val.event_gt.eq('target_moved'); moved_acc=float(val.loc[moved,'dst_pred'].eq(val.loc[moved,'dst']).mean()) if moved.any() else None; emrows += [{'section':'moved_destination','class':'target_moved','metric':'accuracy','value':moved_acc,'support':int(moved.sum())}] + [{'section':'overall','class':'all','metric':'accuracy','value':accuracy_score(y,p)},{'section':'overall','class':'all','metric':'balanced_accuracy','value':balanced_accuracy_score(y,p)},{'section':'overall','class':'all','metric':'macro_f1','value':f1_score(y,p,labels=EVENTS,average='macro',zero_division=0)},{'section':'affected_binary','class':'all','metric':'balanced_accuracy','value':balanced_accuracy_score(ya,pa)},{'section':'affected_binary','class':'all','metric':'macro_f1','value':f1_score(ya,pa,labels=['unaffected','affected'],average='macro',zero_division=0)}]
 pd.DataFrame(emrows).to_csv(a.out/'compact_event_metrics.csv',index=False)
 methods=['native_state','direct_only','tracker_only','oracle_tracker']; ms=scopes(val); rows=[]
 for meth in methods:
  for scope,mask in ms.items(): rows.append({'method':meth,'scope':scope,**summarize(val,meth,mask)})
 rows.append({'method':'tracker_only','scope':'persistent_error_rate','row_mean':float(val.tracker_error_after_first.mean()),'game_mean':float(val.groupby('game_id').tracker_error_after_first.any().mean()),'n_rows':len(val),'n_games':int(val.game_id.nunique())}); pd.DataFrame(rows).to_csv(a.out/'tracker_metrics.csv',index=False); pd.DataFrame([{'method':'direct_only','layer':state.layer,'C':state.C_selected,'selection_score':state.selection_score},{'method':'compact_event','layer':event.layer,'C':event.C_selected,'selection_score':event.selection_score},{'method':'moved_destination','layer':dst.layer,'C':dst.C_selected,'selection_score':dst.selection_score}]).to_csv(a.out/'direct_state_probe_metrics.csv',index=False)
 # Tune alpha on discovery by running a compact vector tracker with discovery probabilities.
 disc=manifest[manifest.protocol_split=='discovery'].copy().sort_values(['game_id','t']).reset_index(drop=True); d_event=probs(event,hidden,disc,3); d_dst=probs(dst,hidden,disc,64); d_direct=probs(state,hidden,disc,65); tuned=[]
 for alpha in (.25,.5,.75):
  z=run_fusion(disc,d_event,d_dst,d_direct,alpha); tuned.append((score(z,'ok'),alpha))
 alpha_l=max(tuned)[1]; val_event=np.stack(val.event_p); val_dst=np.stack(val.dst_p); val_direct=np.stack(val.direct_p); 
 # Fuse the recursively maintained tracker probability with direct-state
 # probability. This is the only hybrid operation evaluated on validation.
 val['tracker_p']=recursive_probs(val); val['LLaVA-selected']=[state_label((alpha_l*b+(1-alpha_l)*d).argmax()) for b,d in zip(val.tracker_p,val.direct_p)]; val['Qwen-transferred']=[state_label((.5*b+.5*d).argmax()) for b,d in zip(val.tracker_p,val.direct_p)]
 val.to_csv(a.out/'behavior_validation.csv',index=False); frows=[]
 for meth in ['LLaVA-selected','Qwen-transferred']:
  for scope,mask in ms.items(): frows.append({'method':meth,'scope':scope,**summarize(val,meth,mask)})
 pd.DataFrame(frows).to_csv(a.out/'fusion_metrics.csv',index=False)
 # Paired fusion comparisons.
 prows=[]
 for meth in ['LLaVA-selected','Qwen-transferred']:
  for base in ['direct_only','tracker_only','native_state']:
   for scope,mask in ms.items():
    q=val.loc[mask].copy(); q['_d']=q[meth].eq(q.gt_state).astype(float)-q[base].eq(q.gt_state).astype(float); g=q.groupby('game_id')._d.mean().to_numpy(); rng=np.random.default_rng(77+len(prows)); boot=g[rng.integers(0,len(g),size=(4000,len(g)))].mean(1); prows.append({'method':meth,'baseline':base,'scope':scope,'game_mean_diff':g.mean(),'ci95_low':np.quantile(boot,.025),'ci95_high':np.quantile(boot,.975),'n_games':len(g),'n_rows':len(q)})
 pd.DataFrame(prows).to_csv(a.out/'paired_comparisons.csv',index=False); (a.out/'report.md').write_text(report(val,alpha_l,emrows,prows,state,event,dst))
 print(json.dumps({'REPLICATION_PASS':True,'validation_games':int(val.game_id.nunique()),'validation_rows':len(val),'alpha_llava':alpha_l,'layers':layer_ids},indent=2))

def state_label(i): return 'captured' if int(i)==64 else SQUARES[int(i)]
def annotate(d):
 first=d[d.event_gt.eq('target_moved')].groupby('game_id').t.min(); d=d.copy(); d['first_move_t']=d.game_id.map(first); return d
def recursive_probs(d):
 out={}
 for gid,g in d.groupby('game_id',sort=True):
  b=np.eye(65)[SI['g1']]
  for r in g.sort_values('t').itertuples(): b=trans(b,r.event_p,r.dst_p); out[int(r.Index)]=b.copy()
 return [out[int(i)] for i in d.index]
def run_fusion(d,ep,dp,sp,alpha):
 # lightweight discovery score: per-game recursive compact state + direct fusion.
 rows=[]; k=0
 for gid,g in d.groupby('game_id',sort=True):
  b=np.eye(65)[SI['g1']]
  for r in g.sort_values('t').itertuples():
   b=trans(b,ep[k],dp[k]); f=alpha*b+(1-alpha)*sp[k]; rows.append({'game_id':gid,'ok':state_label(f.argmax())==r.current_state}); k+=1
 return pd.DataFrame(rows)
def score(d,col): return d.groupby('game_id')[col].mean().mean()
def report(v,alpha,emrows,prows,state,event,dst):
 lines=['# LLaVA-NeXT-Video-7B Chess Compact-Event Replication','', '严格复用 Qwen 的 200 discovery / 100 validation game pilot manifest，`MAX_T=10`。本报告只在 extraction 阶段使用 LLaVA；没有 explicit move generation、完整 64x64 move probe 或新 split。', '', f'Discovery-only selected: direct layer={state.layer}, C={state.C_selected}; event layer={event.layer}, C={event.C_selected}; moved-destination layer={dst.layer}, C={dst.C_selected}; LLaVA-selected alpha={alpha}; Qwen-transferred alpha=0.5.', '', '## Validation game-mean metrics', '', '| method | overall | affected | post-first | t>=3 | t>=5 | t>=8 | t=10 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
 for meth in ['native_state','direct_only','tracker_only','oracle_tracker','LLaVA-selected','Qwen-transferred']:
  vals=[]
  for s in ['overall','affected_only','post_first_move','t_ge3','t_ge5','t_ge8','t10']:
   q=v if s=='overall' else v.loc[({'overall':np.ones(len(v),bool),'affected_only':v.event_gt.ne('unaffected').to_numpy(),'post_first_move':(v.first_move_t.notna()&(v.t>v.first_move_t)).to_numpy(),'t_ge3':(v.t>=3).to_numpy(),'t_ge5':(v.t>=5).to_numpy(),'t_ge8':(v.t>=8).to_numpy(),'t10':(v.t==10).to_numpy()})[s]]; vals.append(v if False else q.groupby('game_id').apply(lambda x:x[meth].eq(x.gt_state).mean()).mean())
  lines.append('| '+meth+' | '+' | '.join(f'{x:.3f}' for x in vals)+' |')
 lines += ['', '## Event probe', '', 'See `compact_event_metrics.csv` for confusion matrix, per-class metrics, balanced accuracy, macro-F1 and affected-vs-unaffected metrics.', '', '## Fusion paired comparisons', '', '| method | baseline | scope | diff | CI95 |', '|---|---|---|---:|---|']
 for r in prows:
  if r['scope'] in ('affected_only','post_first_move','t_ge5','t_ge8'): lines.append(f"| {r['method']} | {r['baseline']} | {r['scope']} | {r['game_mean_diff']:.3f} | [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}] |")
 lines += ['', '## Cross-model comparison', '', 'Qwen pilot baseline (from `qwen_hybrid_state_tracker_v1`) versus this fixed-manifest LLaVA result should be read from the corresponding metrics CSVs; both use the same game split and scopes.', '', '## Interpretation', '', '- Check whether native state is weak while compact tracker/direct probe are stronger, and whether tracker retains the t>=5/t>=8 advantage.', '- `LLaVA-selected` may use only discovery tuning; `Qwen-transferred` fixes alpha=0.5 without validation retuning.', '- A common local-event/readout/tracking framework is supported only if both models show the same ordering after affected-step and long-prefix controls.', '']
 return '\n'.join(lines)+'\n'

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=ROOT/'outputs/metbench_chess/llava_compact_event_replication_v1'); ap.add_argument('--manifest',type=Path,required=True); sub=ap.add_subparsers(dest='cmd',required=True)
 e=sub.add_parser('extract'); e.add_argument('--model-dir',type=Path,required=True); e.add_argument('--phase',choices=['discovery','validation'],required=True); e.add_argument('--shard-index',type=int,required=True); e.add_argument('--num-shards',type=int,default=4); e.add_argument('--candidate-layers',default='0,8,16,24,32'); e.add_argument('--max-games',type=int,default=0); e.add_argument('--run-native-state',action='store_true'); e.add_argument('--overwrite',action='store_true')
 m=sub.add_parser('merge'); m.add_argument('--phase',choices=['discovery','validation'],required=True); m.add_argument('--num-shards',type=int,default=4)
 sub.add_parser('evaluate')
 a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 if a.cmd=='extract': extract(a)
 elif a.cmd=='merge': merge(a)
 else: evaluate(a)
if __name__=='__main__': main()
