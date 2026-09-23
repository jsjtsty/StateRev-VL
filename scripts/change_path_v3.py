#!/usr/bin/env python3
from pathlib import Path
import os,sys
import pandas as pd,torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from staterev.psf import ShellAdapter,ChessAdapter,split_grouped
OUT=ROOT/'outputs/psf_v1/change_path_v3';OUT.mkdir(parents=True,exist_ok=True)
def load(task,model):
    if task=='shell': return ShellAdapter(model).load()
    return ChessAdapter(model,root=os.environ.get('PSF_CHESS_QWEN_ROOT') if model=='qwen' else None).load(split='discovery')
class MLP(nn.Module):
    def __init__(self,inp,k): super().__init__(); self.net=nn.Sequential(nn.Linear(inp,128),nn.ReLU(),nn.Linear(128,k))
    def forward(self,x): return self.net(x)
def fit(ds,kind,k,device):
    tr,dv=split_grouped(ds,17); rows=[]
    for q in tr:
        prev=torch.tensor([q.initial_state_id]+q.state_ids[:-1].tolist())
        for i in range(len(q.state_ids)):
            x=q.hidden[i,-1]; xp=q.hidden[i-1,-1] if i else torch.zeros_like(x); one=F.one_hot(prev[i],k).float()
            parts={'current_only':[one,x],'delta_only':[one,x-xp],'current_previous':[one,x,xp],'full_concat':[one,x,xp,x-xp]}; rows.append((torch.cat(parts[kind]),int(q.state_ids[i])))
    m=MLP(rows[0][0].numel(),k).to(device); opt=torch.optim.AdamW(m.parameters(),lr=2e-3); X=torch.stack([x for x,y in rows]).to(device); y=torch.tensor([y for x,y in rows],device=device)
    for _ in range(120): opt.zero_grad(); loss=F.cross_entropy(m(X),y); loss.backward(); opt.step()
    out=[]
    with torch.no_grad():
        for q in dv:
            prev=[q.initial_state_id]+q.state_ids[:-1].tolist(); yy=q.state_ids.tolist(); xs=[]
            for i in range(len(yy)):
                x=q.hidden[i,-1];xp=q.hidden[i-1,-1] if i else torch.zeros_like(x);one=F.one_hot(torch.tensor(prev[i]),k).float();parts={'current_only':[one,x],'delta_only':[one,x-xp],'current_previous':[one,x,xp],'full_concat':[one,x,xp,x-xp]};xs.append(torch.cat(parts[kind]))
            pp=m(torch.stack(xs).to(device)).argmax(1).cpu()
            for i,(p,g) in enumerate(zip(pp,yy)): out.append({'correct':int(p==g),'changed':int(g!=prev[i]),'t':i+1})
    d=pd.DataFrame(out); return {'overall':d.correct.mean(),'changed':d[d.changed==1].correct.mean(),'unchanged':d[d.changed==0].correct.mean(),'n':len(d),'changed_n':int(d.changed.sum())}
def main():
    rec=[]
    for task,model in [('shell','qwen'),('shell','llava'),('chess','qwen')]:
        ds=load(task,model); k=3 if task=='shell' else 65
        for kind in ['current_only','delta_only','current_previous','full_concat']:
            rec.append({'task':task,'model':model,'representation':kind,**fit(ds,kind,k,os.environ.get('DEVICE','cuda:0'))})
    pd.DataFrame(rec).to_csv(OUT/'change_path_audit.csv',index=False);print(pd.DataFrame(rec).to_string(index=False))
if __name__=='__main__': main()
