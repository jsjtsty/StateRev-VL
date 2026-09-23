#!/usr/bin/env python3
"""Corrected native-state generation on the fixed LLaVA validation split."""
from pathlib import Path
import argparse, hashlib, io, json
import sys
import numpy as np, pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.metbench_chess_llava_replication import validate_manifest, load_ref, load_model, inputs

PROMPT='Initial FEN: {fen}. Track the white knight initially on g1 through move {t}. Where is it now? Answer one square or captured.'

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--model-dir',type=Path,required=True); ap.add_argument('--shard-index',type=int,required=True); ap.add_argument('--num-shards',type=int,default=4); ap.add_argument('--max-new-tokens',type=int,default=8); ap.add_argument('--max-games',type=int,default=0); ap.add_argument('--overwrite',action='store_true'); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 marker=a.out/f'corrected_native_{a.shard_index}.complete.json'
 if marker.exists() and not a.overwrite: print(json.dumps({'ALREADY_COMPLETE':True})); return
 m=validate_manifest(a.manifest); m=m[m.protocol_split.eq('validation')].copy(); games=m.game_id.astype(str).unique(); games=[g for g in games if int(hashlib.sha256(g.encode()).hexdigest()[:8],16)%a.num_shards==a.shard_index]; games=set(games)
 if a.max_games>0: games=set(sorted(games)[:a.max_games])
 model,processor=load_model(a.model_dir); cache={}; rows=[]
 for gid,g in m[m.game_id.astype(str).isin(games)].groupby('game_id',sort=True):
  frames=[]
  for r in g.sort_values('t').itertuples(index=False):
   frames.append(np.asarray(Image.open(io.BytesIO(load_ref(r.image_path,cache))).convert('RGB')))
   q=PROMPT.format(fen=r.initial_state,t=int(r.t)); x=inputs(processor,np.stack(frames),q); dev=next(model.parameters()).device; x={k:(v.to(dev) if hasattr(v,'to') else v) for k,v in x.items()}
   import torch
   with torch.inference_mode(): out=model.generate(**x,max_new_tokens=a.max_new_tokens,do_sample=False)
   txt=processor.batch_decode(out[:,x['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
   rows.append({'game_id':str(gid),'t':int(r.t),'corrected_native_text':txt,'prompt':q})
 pd.DataFrame(rows).to_csv(a.out/f'corrected_native_{a.shard_index}.csv',index=False)
 marker.write_text(json.dumps({'status':'complete','shard_index':a.shard_index,'num_shards':a.num_shards,'games':len(games),'rows':len(rows),'max_new_tokens':a.max_new_tokens,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest()},indent=2)+'\n')
 print(json.dumps({'CORRECTED_NATIVE_PASS':True,'shard_index':a.shard_index,'games':len(games),'rows':len(rows)}))
if __name__=='__main__': main()
