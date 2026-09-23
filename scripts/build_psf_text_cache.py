#!/usr/bin/env python3
"""Build frozen candidate-state text features for PSF (no gradient/update)."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[1]
SHELL=['the ball is at the left position','the ball is at the middle position','the ball is at the right position']
CHESS=[f'the tracked knight is on {f}{r}' for r in range(1,9) for f in 'abcdefgh']+['the tracked knight is captured']

def sha(s): return hashlib.sha256(s if isinstance(s,bytes) else s.encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','llava'],required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(a.model_dir)
 if a.model=='qwen':
  from transformers import Qwen3VLForConditionalGeneration as Model
 else:
  from transformers import LlavaNextVideoForConditionalGeneration as Model
 model=Model.from_pretrained(a.model_dir,device_map='auto',torch_dtype='auto').eval()
 emb=model.get_input_embeddings(); dev=next(model.parameters()).device
 texts=SHELL+CHESS+['the tracked ball','the white knight initially on g1']; template='Represent the candidate state or tracked entity exactly as text: {text}'
 arr=[]; records=[]
 with torch.inference_mode():
  for text in texts:
   prompt=template.format(text=text); ids=tok(prompt,return_tensors='pt',add_special_tokens=True).input_ids.to(dev)
   x=emb(ids)[0]
   # Fixed pooling: mean over non-special tokens, excluding BOS/EOS when present.
   if x.shape[0]>2: x=x[1:-1]
   v=x.float().mean(0).cpu().numpy(); arr.append(v/np.linalg.norm(v))
   records.append({'text':text,'prompt':prompt,'text_sha256':sha(text),'prompt_sha256':sha(prompt),'token_ids_sha256':sha(','.join(map(str,ids[0].cpu().tolist())))})
 out=a.out;out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,**{r['text_sha256']:v for r,v in zip(records,arr)})
 meta={'model':a.model,'model_dir':str(a.model_dir),'model_config_sha256':sha(json.dumps(model.config.to_dict(),sort_keys=True,default=str)),'embedding_shape':[len(arr),int(arr[0].shape[0])],'dtype':'float32','template':template,'pooling':'mean input embeddings excluding first/last token','states':records,'npz_sha256':sha(out.read_bytes())}
 out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps({'TEXT_CACHE_PASS':True,'path':str(out),'shape':[len(arr),int(arr[0].shape[0])],'sha256':meta['npz_sha256']},indent=2))
if __name__=='__main__':main()
