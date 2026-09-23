#!/usr/bin/env python3
"""Build separate semantic state cache using frozen text-only LM hidden states."""
from pathlib import Path
import argparse,hashlib,json,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
SHELL=['the ball is at the left position','the ball is at the middle position','the ball is at the right position']
CHESS=[f'the tracked knight is on {f}{r}' for r in range(1,9) for f in 'abcdefgh']+['the tracked knight is captured']
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','llava'],required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(a.model_dir)
 if a.model=='qwen': from transformers import Qwen3VLForConditionalGeneration as C
 else: from transformers import LlavaNextVideoForConditionalGeneration as C
 model=C.from_pretrained(a.model_dir,device_map='auto',torch_dtype='auto').eval(); lm=getattr(model,'language_model',None) or getattr(getattr(model,'model',None),'language_model',None)
 if lm is None: raise RuntimeError('model has no text-only language_model')
 texts=SHELL+CHESS+['the tracked ball','the white knight initially on g1']; template='Represent this candidate state or entity: {text}'; arr=[];meta=[]
 with torch.inference_mode():
  for text in texts:
   q=template.format(text=text); ids=tok(q,return_tensors='pt').input_ids.to(next(lm.parameters()).device); o=lm(input_ids=ids,output_hidden_states=True,return_dict=True); v=o.hidden_states[-1][0,-1].float().cpu().numpy(); v=v/np.linalg.norm(v); arr.append(v);meta.append({'text':text,'prompt':q,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'prompt_sha256':hashlib.sha256(q.encode()).hexdigest()})
 a.out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out,**{hashlib.sha256(x.encode()).hexdigest():v for x,v in zip(texts,arr)}); info={'model':a.model,'template':template,'pooling':'final semantic hidden state','shape':[len(arr),len(arr[0])],'states':meta};a.out.with_suffix('.json').write_text(json.dumps(info,indent=2));print(json.dumps({'SEMANTIC_CACHE_PASS':True,'shape':info['shape'],'path':str(a.out)}))
if __name__=='__main__':main()
