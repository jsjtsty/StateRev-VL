#!/usr/bin/env python3
"""Discovery-only frozen hidden-event probe for the LLaVA replication cache."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/vetbench/llava_next_video_7b_replication_v1"
EVENTS = ("Left and Middle", "Middle and Right", "Left and Right")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--split", type=Path, default=ROOT/"outputs/vetbench/content_disjoint_split_v1/discovery_validation_split.json")
    ap.add_argument("--c-grid", default="0.1,1.0,10.0")
    a = ap.parse_args(); out = a.out
    behavior = pd.read_csv(out/"behavior.csv")
    split = json.loads(a.split.read_text()); disc=set(split["discovery_trajectories"]); val=set(split["validation_trajectories"])
    assert not disc & val
    with np.load(out/"hidden_states.npz") as z: h={k:z[k] for k in z.files}
    train=behavior[behavior.trajectory_id.isin(disc)].sort_values("target_prefix"); test=behavior[behavior.trajectory_id.isin(val)].sort_values("target_prefix")
    y=train.gt_event.to_numpy(); groups=train.trajectory_id.to_numpy(); cs=[float(x) for x in a.c_grid.split(",")]
    candidates=[]; best=None
    for layer in range(next(iter(h.values())).shape[0]):
        x=np.stack([h[k][layer] for k in train.target_prefix])
        for c in cs:
            fold=[]
            for tr,te in GroupKFold(3).split(x,y,groups):
                s=StandardScaler().fit(x[tr]); m=LogisticRegression(C=c,max_iter=2000).fit(s.transform(x[tr]),y[tr]); fold.append(float((m.predict(s.transform(x[te]))==y[te]).mean()))
            score=float(np.mean(fold)); r={"layer":layer,"C":c,"cv_accuracy":score,"fold_accuracy":fold}; candidates.append(r)
            rank=(-score,layer,c)
            if best is None or rank<best[0]: best=(rank,r)
    layer=int(best[1]["layer"]); c=float(best[1]["C"]); scaler=StandardScaler().fit(np.stack([h[k][layer] for k in train.target_prefix])); clf=LogisticRegression(C=c,max_iter=2000).fit(scaler.transform(np.stack([h[k][layer] for k in train.target_prefix])),y)
    artifact={"layer":layer,"C":c,"scaler":scaler,"clf":clf,"fit_trajectories":sorted(disc),"event_schema":list(EVENTS),"hidden_shape":list(next(iter(h.values())).shape)}; joblib.dump(artifact,out/"frozen_hidden_event_decoder.joblib")
    rows=[]
    for l in range(next(iter(h.values())).shape[0]):
        s=StandardScaler().fit(np.stack([h[k][l] for k in train.target_prefix])); m=LogisticRegression(C=c,max_iter=2000).fit(s.transform(np.stack([h[k][l] for k in train.target_prefix])),y); pred=m.predict(s.transform(np.stack([h[k][l] for k in test.target_prefix])))
        rows.append({"layer":l,"C":c,"validation_event_accuracy":float((pred==test.gt_event.to_numpy()).mean()),"selected_layer":l==layer})
    pd.DataFrame(rows).to_csv(out/"hidden_event_probe_validation.csv",index=False)
    (out/"frozen_hidden_event_decoder.json").write_text(json.dumps({"layer":layer,"C":c,"fit_trajectories":sorted(disc),"validation_trajectories":sorted(val),"selection":"3-fold GroupKFold on discovery only","candidate_scores":candidates},indent=2)+"\n")
    print(f"PROBE_PASS selected_layer={layer} C={c} validation_acc={rows[layer]['validation_event_accuracy']:.4f}")
if __name__=="__main__": main()
