#!/usr/bin/env python3
"""Small, cache-only PSF failure diagnosis; never invokes vision forward."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd,torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
import sys;sys.path.insert(0,str(ROOT))
from staterev.psf import ShellAdapter,ChessAdapter,TextFeatureStore,PSF,split_grouped

def text_audit(cache,out):
 z=np.load(cache); X=np.stack(list(z.values())).astype('float64'); X/=np.linalg.norm(X,axis=1,keepdims=True)
 S=X@X.T; eig=np.linalg.svd(X-X.mean(0),compute_uv=False); rank=int((eig>eig[0]*1e-3).sum())
 out.write_text(json.dumps({'cache':str(cache),'n':len(X),'dim':X.shape[1],'cosine_min':float(S[np.triu_indices(len(X),1)].min()),'cosine_max':float(S[np.triu_indices(len(X),1)].max()),'cosine_mean':float(S[np.triu_indices(len(X),1)].mean()),'effective_rank':rank,'singular_values_top10':eig[:10].tolist()},indent=2))
 return json.loads(out.read_text())

class Local(nn.Module):
 def __init__(self,h,k): super().__init__(); self.p=nn.Sequential(nn.Linear(h,8,bias=False),nn.Linear(8,64)); self.head=nn.Linear(64,k)
 def forward(self,x): return self.head(F.normalize(self.p(x),dim=-1))
class RawLocal(nn.Module):
 def __init__(self,h,k): super().__init__(); self.head=nn.Linear(h,k)
 def forward(self,x): return self.head(x)
def local_fit(data,k,epochs=100,device='cpu',raw=False):
 m=(RawLocal(4096,k) if raw else Local(4096,k)).to(device); opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
 rows=[]
 for tr in data:
  for x,y in zip(tr.hidden, tr.state_ids): rows.append((x[-1],int(y)))
 X=torch.stack([x for x,y in rows]).to(device); y=torch.tensor([y for x,y in rows],device=device)
 for _ in range(epochs):
  opt.zero_grad(); loss=F.cross_entropy(m(X),y); loss.backward();opt.step()
 with torch.no_grad(): acc=float((m(X).argmax(1)==y).float().mean()); loss=float(F.cross_entropy(m(X),y))
 return {'train_accuracy':acc,'train_loss':loss,'rows':len(rows)}
def obs_fit(data,k,device='cpu'):
 return local_fit(data,k,100,device)
def changed(data):
 rows=[]
 for tr in data:
  prev=tr.initial_state_id
  for y in tr.state_ids.tolist(): rows.append(int(y!=prev));prev=y
 return {'changed':sum(rows),'unchanged':len(rows)-sum(rows),'changed_rate':float(np.mean(rows))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'outputs/psf_v1/diagnostic_v1');p.add_argument('--device',default='cuda:0');p.add_argument('--chess-root',type=Path,default=ROOT/'outputs/psf_v1/formal_precheck/chess_qwen_multitap');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 shell=ShellAdapter('qwen').load(10); chess=ChessAdapter('qwen',root=a.chess_root).load(10); shtr,shdev=split_grouped(shell,17); chtr,chdev=split_grouped(chess,17)
 res={'tiny_shell_local':local_fit(shtr,3,300,a.device),'tiny_chess_local':local_fit(chtr,65,300,a.device),'tiny_shell_raw':local_fit(shtr,3,300,a.device,True),'tiny_chess_raw':local_fit(chtr,65,300,a.device,True),'shell_change':changed(shell),'chess_change':changed(chess)}
 res['text_qwen']=text_audit(ROOT/'outputs/psf_v1/text_cache/qwen_text_embeddings.npz',a.out/'qwen_text_geometry.json');res['text_llava']=text_audit(ROOT/'outputs/psf_v1/text_cache/llava_text_embeddings.npz',a.out/'llava_text_geometry.json')
 # direct observation sanity uses a local linear head on the actual hidden taps.
 res['observation_chess_local']=obs_fit(chtr,65,a.device); res['observation_chess_raw']=local_fit(chtr,65,300,a.device,True)
 (a.out/'diagnostic_summary.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
