#!/usr/bin/env python3
"""Bounded projector gate and Full PSF v2 runs; cache-only."""
from pathlib import Path
import os,sys,json
import numpy as np,pandas as pd,torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from staterev.psf import ShellAdapter,ChessAdapter,PSF,Projector,TextFeatureStore,split_grouped,train,evaluate

def data(task,model,maxn=0):
 if task=='shell': return ShellAdapter(model).load(maxn)
 root=os.environ.get('PSF_CHESS_QWEN_ROOT') if model=='qwen' else None
 return ChessAdapter(model,root=root).load(maxn,'discovery')
def rows(ds):
 return [(x,int(y)) for tr in ds for x,y in zip(tr.hidden,tr.state_ids)]
def raw_fit(tr,dev,k,device):
 m=nn.Linear(4096,k).to(device);o=torch.optim.AdamW(m.parameters(),lr=3e-3)
 X=torch.stack([x[-1] for x,y in rows(tr)]).to(device); y=torch.tensor([y for x,y in rows(tr)],device=device)
 for _ in range(150): o.zero_grad();l=F.cross_entropy(m(X),y);l.backward();o.step()
 with torch.no_grad(): a=float((m(torch.stack([x[-1] for x,y in rows(dev)]).to(device)).argmax(1)==torch.tensor([y for x,y in rows(dev)],device=device)).float().mean())
 return a
class Obs(nn.Module):
 def __init__(self,kind,k): super().__init__(); self.p=Projector(4096,8,64,kind);self.w=nn.Parameter(torch.zeros(4));self.obs=nn.Sequential(nn.Linear(128,64),nn.Tanh());self.h=nn.Linear(64,k)
 def forward(self,x):
  x=self.p(x); z=(x*torch.softmax(self.w,0).view(1,4,1)).sum(1); return self.h(self.obs(torch.cat([z,z*0],-1)))
def obs_fit(tr,dev,k,kind,device):
 m=Obs(kind,k).to(device);o=torch.optim.AdamW(m.parameters(),lr=3e-3);X=torch.stack([x for x,y in rows(tr)]).to(device);y=torch.tensor([y for x,y in rows(tr)],device=device)
 for _ in range(200): o.zero_grad();l=F.cross_entropy(m(X),y);l.backward();o.step()
 with torch.no_grad():
  dX=torch.stack([x for x,y in rows(dev)]).to(device);dy=torch.tensor([y for x,y in rows(dev)],device=device)
  return float((m(dX).argmax(1)==dy).float().mean())
def main():
 out=ROOT/'outputs/psf_v1/stabilization_v2';out.mkdir(parents=True,exist_ok=True);device=os.environ.get('DEVICE','cuda:0'); configs=[('shell','qwen'),('shell','llava'),('chess','qwen'),('chess','llava')]; rec=[]
 for task,model in configs:
  ds=data(task,model);tr,dv=split_grouped(ds,17);k=3 if task=='shell' else 65;raw=raw_fit(tr,dv,k,device)
  for kind in ['full_linear','rank32','rank8_no_norm','rank8_old']:
   rec.append({'task':task,'model':model,'projector':kind,'raw_dev_accuracy':raw,'observation_dev_accuracy':obs_fit(tr,dv,k,kind,device),'passes_gate':False})
 df=pd.DataFrame(rec);df['passes_gate']=df.observation_dev_accuracy>=df.raw_dev_accuracy-.05;df.to_csv(out/'projector_comparison.csv',index=False);best=df[df.passes_gate].sort_values(['observation_dev_accuracy']).iloc[-1] if df.passes_gate.any() else None
 (out/'observation_gate_report.md').write_text('# Observation gate\n\n'+df.to_markdown(index=False)+'\n\nGate pass: '+str(bool(best is not None))+'\n')
 if best is None: print('OBSERVATION_GATE_FAIL');return
 kind=str(best.projector); cacheq=ROOT/'outputs/psf_v1/diagnostic_v1/text_cache/qwen_semantic.npz';cachel=cacheq
 commands=[]
 for task,model in configs+[('joint','qwen')]:
  od=out/f'full_{task}_{model}';cmd=f'python scripts/train_psf.py --train-tasks {"shell,chess" if task=="joint" else task} --models {model} --epochs 30 --changed-lambda 3 --projector-kind {kind} --strict-text-cache --text-cache-{model} {cacheq} --device {device} --out {od}'
  if task=='chess' or task=='joint': cmd='PSF_CHESS_QWEN_ROOT=outputs/psf_v1/formal_precheck/chess_qwen_multitap '+cmd
  os.system(cmd);commands.append(cmd)
 (out/'run_commands.json').write_text(json.dumps(commands,indent=2));print(json.dumps({'OBSERVATION_GATE_PASS':True,'selected_projector':kind,'runs':commands}))
if __name__=='__main__':main()
