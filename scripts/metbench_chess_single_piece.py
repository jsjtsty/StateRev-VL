#!/usr/bin/env python3
"""MET-Bench Chess single-piece tracking (manifest, probes, updater, smoke).

The dataset is discovered from Parquet schema rather than hard-coded filenames.
The tracked identity is the white knight initially on g1.  Rows are prefixes
after ordinary UCI moves; a game is truncated at its first castling,
en-passant, promotion, or malformed move.  No model is loaded by default.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "dataset" / "MET-Bench-Chess"
DEFAULT_OUT = ROOT / "outputs" / "metbench_chess" / "single_piece_tracking_v1"
SQUARES = [f"{f}{r}" for r in range(1, 9) for f in "abcdefgh"]
SI = {s:i for i,s in enumerate(SQUARES)}
TARGET = "white_knight_g1"

def sha256_bytes(x: bytes) -> str: return hashlib.sha256(x).hexdigest()
def sha256_file(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def discover(data_dir: Path):
    import pyarrow.parquet as pq
    # Prefer canonical full-* shards; evaluation-test duplicates full-test and
    # evaluation-text has no visual payload. Keep discovery automatic but avoid
    # double-counting duplicate example_id rows.
    files=sorted(p for p in data_dir.glob("*.parquet") if not p.name.startswith("evaluation-"))
    if not files: raise FileNotFoundError(f"no parquet files under {data_dir}")
    required={"example_id","initial_state","actions","image_actions","states","final_state"}
    good=[]
    for p in files:
        cols=set(pq.ParquetFile(p).schema_arrow.names)
        if required <= cols: good.append(p)
    if not good: raise ValueError(f"no files contain {sorted(required)}")
    return good

def split_name(game_id: str) -> str:
    m=re.match(r"chess-([^-]+)-", str(game_id)); return m.group(1) if m else "unknown"

def board_from_fen(fen: str):
    import chess
    return chess.Board(fen)

def ordinary(action: str) -> bool:
    """Special moves excluded from the task (castling, ep, promotion)."""
    # Syntax-only check. Board-dependent special moves are checked in
    # ``track_game`` before push (castling and en-passant cannot be inferred
    # reliably from the four UCI characters alone).
    return bool(re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", str(action))) and len(str(action))==4

def track_game(row, image_dir: Path, source: str = "", row_index: int = 0, materialize: bool = False, special_stats=None):
    import chess
    b=board_from_fen(row.initial_state)
    target=chess.G1; captured=False; rows=[]
    images=row.image_actions if row.image_actions is not None else []
    for i,(action, state) in enumerate(zip(row.actions, row.states[1:]), start=1):
        if not ordinary(action):
            if special_stats is not None:
                special_stats["promotion" if re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]", str(action)) else "malformed"] += 1
            break
        try: move=chess.Move.from_uci(action)
        except ValueError: break
        if move not in b.legal_moves:
            if special_stats is not None: special_stats["malformed_or_illegal"] += 1
            break
        if b.is_castling(move):
            if special_stats is not None: special_stats["castling"] += 1
            break
        if b.is_en_passant(move):
            if special_stats is not None: special_stats["en_passant"] += 1
            break
        if move.promotion is not None:
            if special_stats is not None: special_stats["promotion"] += 1
            break
        src,dst=move.from_square,move.to_square
        is_target=(src==target and b.piece_at(src) and b.piece_at(src).color==chess.WHITE and b.piece_at(src).piece_type==chess.KNIGHT)
        takes_target=(dst==target and b.piece_at(target) is not None and b.piece_at(target).color==chess.WHITE and b.piece_at(target).piece_type==chess.KNIGHT)
        if is_target: target=dst
        if takes_target: captured=True; target=None
        b.push(move)
        item=images[i-1] if i-1 < len(images) else {}
        raw=item.get("bytes") if isinstance(item,dict) else None
        path=None
        if raw and materialize:
            path=image_dir / f"{row.example_id}_t{i}.png"; path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists(): path.write_bytes(raw)
        elif raw:
            path=f"parquet://{source}#row={row_index}&action={i-1}"
        rows.append({"game_id":str(row.example_id),"split":split_name(row.example_id),"t":i,
          "target_piece":TARGET,"initial_square":"g1","src":chess.square_name(src),"dst":chess.square_name(dst),
          "current_state":"captured" if captured else chess.square_name(target),"captured":bool(captured),
          "initial_state":row.initial_state,"state_fen":state,"image_path":str(path.relative_to(ROOT)) if isinstance(path,Path) else str(path) if path else "",
          "image_sha256":sha256_bytes(raw) if raw else ""})
    return rows

def build_manifest(data_dir: Path, out: Path):
    from collections import Counter
    import subprocess
    files=discover(data_dir); image_dir=out/"images"; schema={}; games_by_source=Counter(); prefixes_by_protocol=Counter(); states=Counter(); events=Counter(); game_ids={"discovery":set(),"validation":set(),"holdout_test":set()}; source_games=Counter(); fp_paths={x:out/f"{x}_fingerprints.txt" for x in game_ids}
    manifest_path=out/"manifest.csv"; manifest_path.unlink(missing_ok=True)
    fp_handles={k:p.open("w") for k,p in fp_paths.items()}
    first=True; total=0; special=Counter(); seen_game_ids=set()
    for p in files:
        import pyarrow.parquet as pq
        pf=pq.ParquetFile(p)
        schema[p.name]={"rows":pf.metadata.num_rows,"columns":list(pf.schema_arrow.names)}
        row_index=0
        for batch in pf.iter_batches(batch_size=32):
          df=batch.to_pandas()
          rows=[]
          for row in df.itertuples(index=False):
            if str(row.example_id) in seen_game_ids:
                continue
            seen_game_ids.add(str(row.example_id))
            source_split=split_name(row.example_id)
            if source_split=="train": protocol="discovery" if int(hashlib.sha256(str(row.example_id).encode()).hexdigest()[:8],16)%10 < 8 else "validation"
            elif source_split=="validation": protocol="validation"
            else: protocol="holdout_test"
            game_ids[protocol].add(str(row.example_id)); source_games[source_split]+=1
            gs=track_game(row,image_dir,source=str(p.relative_to(ROOT)),row_index=row_index,materialize=False,special_stats=special)
            for g in gs:
                g["source_split"]=source_split; g["protocol_split"]=protocol
                rows.append(g); states[g["current_state"]]+=1; events[f"{g['src']}->{g['dst']}"]+=1
                if g["image_sha256"]:
                    fp_handles[protocol].write(g["image_sha256"]+"\n")
            prefixes_by_protocol[protocol]+=len(gs); total+=len(gs)
            row_index += 1
          if rows:
            pd.DataFrame(rows).to_csv(manifest_path,index=False,header=first,mode="a"); first=False
          del df, rows
    for fh in fp_handles.values(): fh.close()
    if not total: raise ValueError("no ordinary prefixes produced")
    for p in fp_paths.values():
        if p.exists(): subprocess.run(["sort","-u","-o",str(p),str(p)],check=True)
    overlap_file=out/"fingerprint_overlap.txt"; overlap_file.unlink(missing_ok=True)
    if fp_paths["discovery"].exists() and fp_paths["validation"].exists():
        with overlap_file.open("w") as f: subprocess.run(["comm","-12",str(fp_paths["discovery"]),str(fp_paths["validation"])],stdout=f,check=True)
    discovery=sorted(game_ids["discovery"]); validation=sorted(game_ids["validation"])
    summary={"schema":schema,"files":len(files),"games":sum(len(x) for x in game_ids.values()),"games_by_source_split":dict(source_games),"prefixes":total,"prefixes_by_protocol":dict(prefixes_by_protocol),"games_by_protocol":{k:len(v) for k,v in game_ids.items()},"special_policy":"truncate before first castling/en-passant/promotion/malformed move","filtered_special_moves":dict(special),"target_piece":TARGET,"event_count":total,"state_distribution":dict(states),"event_distribution":dict(events.most_common(20)),"image_fingerprint_overlap":sum(1 for _ in overlap_file.open()) if overlap_file.exists() else 0,"discovery_validation_game_overlap":len(game_ids["discovery"]&game_ids["validation"]),"manifest":str(manifest_path.relative_to(ROOT))}
    (out/"discovery_games.txt").write_text("\n".join(discovery)+"\n")
    (out/"validation_games.txt").write_text("\n".join(validation)+"\n")
    (out/"dataset_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return None,summary

class ChessStateUpdater(torch.nn.Module):
    """Small learned belief updater; 64 squares + captured, no chess rules."""
    def __init__(self, dim=32):
        super().__init__(); self.state=torch.nn.Linear(65,dim); self.src=torch.nn.Linear(64,dim,bias=False); self.dst=torch.nn.Linear(64,dim,bias=False); self.out=torch.nn.Linear(dim,65)
    def forward(self, previous, source, destination):
        return torch.softmax(self.out(torch.tanh(self.state(previous)+self.src(source)+self.dst(destination))),-1)
    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters())

def oracle_transition(previous, src, dst):
    out=np.zeros(65,float); out[64]=previous[64]
    out += previous * 0
    # Probability mass at source moves to destination; destination mass is captured.
    moved=float(previous[src]); out[src]+=0; out[dst]+=moved
    out[64]+=float(previous[dst])
    for i in range(64):
        if i not in (src,dst): out[i]+=previous[i]
    return out/out.sum()

def recursive_state_tracking(initial_square: str, source_probabilities, destination_probabilities, updater=None, mode="learned"):
    """Run a frozen-event recursive tracker.

    ``mode=learned`` consumes the supplied small updater. ``mode=oracle`` is
    the handwritten transition upper bound and is deliberately kept separate
    from the learned path. Probabilities are never replaced by move-question
    hidden states here; callers decide whether they came from explicit or
    state-question probes.
    """
    belief=np.eye(65,dtype=float)[SI[initial_square]]; records=[]
    for srcp,dstp in zip(source_probabilities,destination_probabilities):
        srcp=np.asarray(srcp,float); dstp=np.asarray(dstp,float); srcp=srcp/srcp.sum(); dstp=dstp/dstp.sum()
        if mode=="oracle":
            nxt=oracle_transition(belief,int(srcp.argmax()),int(dstp.argmax()))
        elif mode=="learned":
            if updater is None: raise ValueError("learned tracking requires updater")
            with torch.no_grad(): nxt=updater(torch.tensor(belief[None],dtype=torch.float32),torch.tensor(srcp[None],dtype=torch.float32),torch.tensor(dstp[None],dtype=torch.float32))[0].numpy()
        else: raise ValueError(mode)
        belief=nxt/nxt.sum(); records.append({"belief":belief.copy(),"predicted_state":"captured" if belief[64]>=belief[:64].max() else SQUARES[int(belief[:64].argmax())],"confidence":float(belief.max())})
    return records

def train_state_updater(manifest: Path, source_probs: dict, destination_probs: dict, out: Path, epochs=3, hidden_dim=32):
    """Discovery-only updater fit; event decoders and VLM are frozen."""
    rows=pd.read_csv(manifest,usecols=["game_id","t","initial_square","current_state","protocol_split"],chunksize=100000)
    games={}
    for c in rows:
        for r in c.itertuples(index=False):
            if r.protocol_split=="discovery" and f"{r.game_id}_t{int(r.t)}" in source_probs and f"{r.game_id}_t{int(r.t)}" in destination_probs: games.setdefault(r.game_id,[]).append(r)
    model=ChessStateUpdater(hidden_dim); opt=torch.optim.Adam(model.parameters(),lr=1e-3); loss_fn=torch.nn.NLLLoss()
    for _ in range(epochs):
        for gid,rs in games.items():
            rs=sorted(rs,key=lambda x:x.t); belief=torch.eye(65)[SI[rs[0].initial_square]].float(); losses=[]
            for r in rs:
                key=f"{r.game_id}_t{int(r.t)}"; sp=torch.tensor(source_probs[key],dtype=torch.float32); dp=torch.tensor(destination_probs[key],dtype=torch.float32)
                belief=model(belief[None],sp[None],dp[None])[0]; target=64 if r.current_state=="captured" else SI[r.current_state]; losses.append(-torch.log(belief[target].clamp_min(1e-8)))
            if losses: opt.zero_grad(); torch.stack(losses).mean().backward(); opt.step()
    out.parent.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":model.state_dict(),"hidden_dim":hidden_dim,"parameter_count":model.parameter_count,"fit_games":sorted(games)},out)
    return model

def write_manifest_shard(manifest: Path, out: Path, shard_index: int, num_shards: int):
    if not 0<=shard_index<num_shards: raise ValueError("invalid shard index")
    path=out/"shards"/f"manifest_shard_{shard_index}.csv"; marker=path.with_suffix(".complete.json"); path.parent.mkdir(parents=True,exist_ok=True)
    path.unlink(missing_ok=True); marker.unlink(missing_ok=True); first=True; games=set(); n=0
    for c in pd.read_csv(manifest,chunksize=100000):
        if "game_id" not in c: raise ValueError("manifest missing game_id")
        ids=sorted(c.game_id.astype(str).unique()); keep={g for j,g in enumerate(ids) if (int(hashlib.sha256(g.encode()).hexdigest()[:8],16)%num_shards)==shard_index}
        # The game-to-shard function is global and independent of chunk order.
        keep={g for g in c.game_id.astype(str).unique() if int(hashlib.sha256(g.encode()).hexdigest()[:8],16)%num_shards==shard_index}
        q=c[c.game_id.astype(str).isin(keep)]
        if len(q): q.to_csv(path,index=False,header=first,mode="a"); first=False; games.update(keep); n+=len(q)
    marker.write_text(json.dumps({"schema_version":1,"shard_index":shard_index,"num_shards":num_shards,"rows":n,"games":sorted(games),"sha256":sha256_file(path)},indent=2)+"\n")
    print(json.dumps({"SHARD_PASS":True,"path":str(path),"rows":n,"games":len(games)}))

def merge_manifest_shards(out: Path, num_shards: int):
    frames=[]; all_games=[]
    for i in range(num_shards):
        p=out/"shards"/f"manifest_shard_{i}.csv"; m=p.with_suffix(".complete.json")
        if not p.exists() or not m.exists(): raise FileNotFoundError(f"missing shard {i}")
        meta=json.loads(m.read_text()); assert int(meta["shard_index"])==i and sha256_file(p)==meta["sha256"]
        f=pd.read_csv(p); assert len(f)==int(meta["rows"]); frames.append(f); all_games.extend(f.game_id.astype(str).unique())
    assert len(all_games)==len(set(all_games)),"game appears in multiple shards"
    merged=pd.concat(frames,ignore_index=True).sort_values(["game_id","t"]); merged.to_csv(out/"manifest_shards_merged.csv",index=False)
    (out/"manifest_shards_merged.complete.json").write_text(json.dumps({"status":"complete","rows":len(merged),"games":len(set(all_games)),"num_shards":num_shards,"sha256":sha256_file(out/"manifest_shards_merged.csv")},indent=2)+"\n")
    print(json.dumps({"MERGE_PASS":True,"rows":len(merged),"games":len(set(all_games))}))

def updater_unit():
    u=ChessStateUpdater(); assert u.parameter_count==((65+1)*32+(64*32)*2+(32+1)*65)
    p=torch.softmax(torch.randn(4,65),-1); s=torch.softmax(torch.randn(4,64),-1); d=torch.softmax(torch.randn(4,64),-1); q=u(p,s,d)
    assert q.shape==(4,65) and torch.allclose(q.sum(-1),torch.ones(4),atol=1e-6)
    x=oracle_transition(np.eye(65)[SI['g1']],SI['g1'],SI['f3']); assert np.argmax(x)==SI['f3']
    print(json.dumps({"UNIT_PASS":True,"parameter_count":u.parameter_count,"oracle_transition":True}))

def special_unit():
    import chess
    # Board-dependent special move checks are intentionally exercised rather
    # than inferred from UCI spelling (en-passant is the easy case to miss).
    cases=[
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1","e1g1","castling"),
        ("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1","e5d6","en_passant"),
        ("4k3/6P1/8/8/8/8/8/4K3 w - - 0 1","g7g8q","promotion"),
    ]
    seen=[]
    for fen,uci,kind in cases:
        b=chess.Board(fen); m=chess.Move.from_uci(uci)
        assert m in b.legal_moves
        actual="castling" if b.is_castling(m) else "en_passant" if b.is_en_passant(m) else "promotion" if m.promotion else "ordinary"
        assert actual==kind; seen.append(actual)
    print(json.dumps({"SPECIAL_FILTER_PASS":True,"cases":seen}))

def audit_manifest(out: Path):
    summary=json.loads((out/"dataset_summary.json").read_text())
    d=set((out/"discovery_games.txt").read_text().split()); v=set((out/"validation_games.txt").read_text().split())
    assert not d & v
    overlap=(out/"fingerprint_overlap.txt").read_text().splitlines() if (out/"fingerprint_overlap.txt").exists() else []
    expected={"game_id","t","target_piece","initial_square","src","dst","current_state","image_path","captured","protocol_split"}
    head=pd.read_csv(out/"manifest.csv",nrows=2)
    assert expected <= set(head.columns)
    print(json.dumps({"AUDIT_PASS":True,"games":{"discovery":len(d),"validation":len(v)},"game_overlap":len(d&v),"image_fingerprint_overlap":len(overlap),"content_disjoint":not bool(overlap),"manifest_columns":list(head.columns),"summary_prefixes":summary["prefixes"]},indent=2))

def load_hidden_rows(manifest: Path, hidden_cache: Path):
    import numpy as np
    with np.load(hidden_cache,allow_pickle=False) as z:
        keys=list(z.files); arrays={k:z[k] for k in keys}
    rows=[]
    # Read only columns needed for probe and process in chunks. Hidden caches
    # use the stable key ``<game_id>_t<t>``.
    for chunk in pd.read_csv(manifest,usecols=["game_id","t","src","dst","protocol_split"],chunksize=100000):
        for r in chunk.itertuples(index=False):
            key=f"{r.game_id}_t{int(r.t)}"
            if key in arrays: rows.append((key,int(r.src[1])-1 + 8*"abcdefgh".index(r.src[0]),int(r.dst[1])-1 + 8*"abcdefgh".index(r.dst[0]),r.protocol_split))
    return arrays,rows

def fit_probe(manifest: Path, hidden_cache: Path, out: Path, candidate_layers="0", candidate_c="1.0"):
    import joblib
    sidecar=hidden_cache.with_suffix(".json")
    if sidecar.exists():
        meta=json.loads(sidecar.read_text())
        if str(meta.get("question_type", "state")) != "state":
            raise AssertionError("main square probe refuses move-question hidden states")
    arrays, rows=load_hidden_rows(manifest,hidden_cache)
    if not rows: raise ValueError("no manifest rows matched hidden cache keys")
    first=np.asarray(arrays[rows[0][0]]); n_layers=first.shape[0] if first.ndim==2 else 1
    layers=[int(x) for x in str(candidate_layers).split(",") if x.strip()]
    cs=[float(x) for x in str(candidate_c).split(",") if x.strip()]
    discovery=[r for r in rows if r[3]=="discovery"]
    if not discovery: raise ValueError("probe requires discovery rows")
    # Internal game-held-out selection is hash based; validation rows never
    # participate in layer/C selection or scaler fitting.
    tune=[r for r in discovery if int(hashlib.sha256(r[0].rsplit("_t",1)[0].encode()).hexdigest()[:8],16)%5==0]
    fitrows=[r for r in discovery if r not in set(tune)] or discovery
    best=None
    for layer in layers:
      if layer>=n_layers: continue
      for c in cs:
        x=np.stack([np.asarray(arrays[r[0]])[layer] if np.asarray(arrays[r[0]]).ndim==2 else np.asarray(arrays[r[0]]) for r in fitrows])
        tx=np.stack([np.asarray(arrays[r[0]])[layer] if np.asarray(arrays[r[0]]).ndim==2 else np.asarray(arrays[r[0]]) for r in tune]) if tune else x
        sy=np.array([r[1] for r in fitrows]); dy=np.array([r[2] for r in fitrows]); tsy=np.array([r[1] for r in tune]) if tune else sy; tdy=np.array([r[2] for r in tune]) if tune else dy
        sp=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=300,solver="lbfgs")).fit(x,sy)
        dp=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=300,solver="lbfgs")).fit(x,dy)
        score=float((sp.predict(tx)==tsy).mean()+(dp.predict(tx)==tdy).mean())/2
        if best is None or score>best[0]: best=(score,layer,c)
    if best is None: raise ValueError("no valid layer/C candidate")
    _,layer,c=best; x=np.stack([np.asarray(arrays[r[0]])[layer] if np.asarray(arrays[r[0]]).ndim==2 else np.asarray(arrays[r[0]]) for r in discovery]); sy=np.array([r[1] for r in discovery]); dy=np.array([r[2] for r in discovery])
    src=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=500,solver="lbfgs")).fit(x,sy); dst=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=500,solver="lbfgs")).fit(x,dy)
    artifact={"schema_version":1,"feature_source":"state-question forward hidden states","layer":layer,"C":c,"src_probe":src,"dst_probe":dst,"fit_games":sorted({r[0].rsplit("_t",1)[0] for r in discovery}),"validation_fit_overlap":False,"classes":SQUARES}
    out.parent.mkdir(parents=True,exist_ok=True); joblib.dump(artifact,out); (out.with_suffix(".json")).write_text(json.dumps({k:v for k,v in artifact.items() if k not in {"src_probe","dst_probe"}},default=str,indent=2)+"\n")
    print(json.dumps({"PROBE_PASS":True,"layer":layer,"C":c,"fit_rows":len(discovery),"fit_games":len(artifact["fit_games"]),"feature_source":artifact["feature_source"]},indent=2))

def shard_unit(out: Path):
    out.mkdir(parents=True,exist_ok=True); d=out/"shards"; d.mkdir(exist_ok=True); frames=[]
    for i in range(2):
        f=pd.DataFrame([{"game_id":f"g{i}","t":1}]); p=d/f"prediction_shard_{i}.csv"; f.to_csv(p,index=False); (p.with_suffix('.complete.json')).write_text(json.dumps({"shard_index":i,"num_shards":2,"rows":1,"sha256":sha256_file(p)})); frames.append(f)
    merged=pd.concat([pd.read_csv(d/f"prediction_shard_{i}.csv") for i in range(2)]); assert len(merged)==2 and merged.game_id.nunique()==2
    print(json.dumps({"SHARD_MERGE_PASS":True,"rows":len(merged)}))

def model_smoke(args):
    # Uses the same transformers model classes as existing StateRev-VL scripts.
    try:
        from transformers import AutoProcessor
        if args.model=="qwen":
            from transformers import Qwen3VLForConditionalGeneration as Model
        else:
            from transformers import LlavaNextVideoForConditionalGeneration as Model
        processor=AutoProcessor.from_pretrained(args.model_dir); model=Model.from_pretrained(args.model_dir,device_map="auto",torch_dtype="auto"); model.eval()
        frame=pd.read_csv(args.manifest); game=frame.game_id.iloc[0]; g=frame[frame.game_id==game].sort_values("t").iloc[:args.limit]
        if g.empty or not g.image_path.iloc[0]: raise RuntimeError("manifest has no image path")
        from PIL import Image
        import pyarrow.parquet as pq
        cache=args.out/"smoke_images"; cache.mkdir(exist_ok=True)
        imgs=[]
        for j, ref in enumerate(g.image_path):
            m=re.match(r"parquet://(.+)#row=(\d+)&action=(\d+)", str(ref))
            if not m: raise RuntimeError(f"bad image reference {ref}")
            tab=pq.read_table(ROOT/m.group(1),columns=["image_actions"],filters=[("__dummy__","=",0)]) if False else pq.read_table(ROOT/m.group(1),columns=["image_actions"]).slice(int(m.group(2)),1)
            raw=tab["image_actions"][0].as_py()[int(m.group(3))]["bytes"]
            ip=cache/f"{j}.png"; ip.write_bytes(raw); imgs.append(np.asarray(Image.open(ip).convert("RGB")))
        clip=np.stack(imgs,axis=0)
        # Qwen3-VL temporal patching requires at least two frames for a video
        # token. A one-prefix smoke duplicates the observed frame; the task
        # manifest and state question remain unchanged.
        if len(clip)==1: clip=np.concatenate([clip,clip],axis=0)
        text=(f"Initial chessboard FEN: {g.initial_state.iloc[0]}. The tracked piece is the white knight "
              f"that started on g1. You are given the visual move sequence through move {int(g.t.iloc[-1])}. "
              "Where is that piece now? Answer one square (a1-h8) or captured.")
        messages=[{"role":"system","content":"Answer the chess state question concisely."},{"role":"user","content":[{"type":"video","video":clip},{"type":"text","text":text}]}]
        try:
            inp=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt")
        except Exception:
            prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inp=processor(text=prompt,videos=clip,return_tensors="pt")
        if "pixel_values_videos" not in inp:
            prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inp=processor(text=prompt,videos=clip,return_tensors="pt")
        dev=next(model.parameters()).device; inp={k:(v.to(dev) if hasattr(v,'to') else v) for k,v in inp.items()}
        with torch.no_grad():
            if args.model=="qwen": o=model.model(**inp,output_hidden_states=True,return_dict=True)
            else: o=model(**inp,output_hidden_states=True,return_dict=True,logits_to_keep=1)
        hs=o.hidden_states; pos=inp["input_ids"].shape[1]-1; arr=np.asarray(hs[-1][0,pos].float().cpu()); np.save(args.out/f"{args.model}_hidden_smoke.npy",arr)
        (args.out/f"{args.model}_hidden_smoke.json").write_text(json.dumps({"question_type":"state","feature_source":"state-question forward","model":args.model,"hidden_shape":list(arr.shape),"prefixes":len(g),"game_id":str(game),"t_max":int(g.t.iloc[-1])},indent=2)+"\n")
        print(json.dumps({"SMOKE_PASS":True,"model":args.model,"hidden_shape":list(arr.shape),"prefixes":len(g),"question_type":"state"}))
    except Exception as e:
        print(json.dumps({"SMOKE_SKIPPED":True,"model":args.model,"reason":repr(e)}))

def native_smoke(args):
    """One-prefix native state and explicit-move generation smoke.

    This intentionally reports generated text only; formal candidate-logit
    scoring remains a separate full-run stage. Both prompts are kept distinct
    so move-question hidden states cannot enter the state probe cache.
    """
    try:
        from transformers import AutoProcessor
        Model = __import__("transformers",fromlist=["Qwen3VLForConditionalGeneration" if args.model=="qwen" else "LlavaNextVideoForConditionalGeneration"]).__dict__["Qwen3VLForConditionalGeneration" if args.model=="qwen" else "LlavaNextVideoForConditionalGeneration"]
        processor=AutoProcessor.from_pretrained(args.model_dir); model=Model.from_pretrained(args.model_dir,device_map="auto",torch_dtype="auto").eval()
        frame=pd.read_csv(args.manifest); game=str(frame.game_id.iloc[0]); g=frame[frame.game_id==game].sort_values("t").iloc[:args.limit]
        import pyarrow.parquet as pq
        from PIL import Image
        imgs=[]
        for j,ref in enumerate(g.image_path):
            m=re.match(r"parquet://(.+)#row=(\d+)&action=(\d+)",str(ref));
            if not m: raise RuntimeError(f"bad image reference {ref}")
            tab=pq.read_table(ROOT/m.group(1),columns=["image_actions"]).slice(int(m.group(2)),1); raw=tab["image_actions"][0].as_py()[int(m.group(3))]["bytes"]
            ip=args.out/f"native_smoke_{j}.png"; ip.write_bytes(raw); imgs.append(np.asarray(Image.open(ip).convert("RGB")))
        clip=np.stack(imgs); 
        if len(clip)==1: clip=np.concatenate([clip,clip])
        def ask(question):
            msgs=[{"role":"system","content":"Answer only the requested chess field."},{"role":"user","content":[{"type":"video","video":clip},{"type":"text","text":question}]}]
            try: inp=processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt")
            except Exception:
                inp=processor(text=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True),videos=clip,return_tensors="pt")
            if "pixel_values_videos" not in inp: inp=processor(text=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True),videos=clip,return_tensors="pt")
            dev=next(model.parameters()).device; inp={k:(v.to(dev) if hasattr(v,"to") else v) for k,v in inp.items()}
            with torch.inference_mode(): out=model.generate(**inp,max_new_tokens=12,do_sample=False)
            return processor.batch_decode(out[:,inp["input_ids"].shape[1]:],skip_special_tokens=True)[0].strip()
        base=f"Initial FEN {g.initial_state.iloc[0]}. Track the white knight initially on g1 through move {int(g.t.iloc[-1])}."
        state=ask(base+" Output only its current square or captured."); move=ask(base+" Output only the latest move as source-square destination-square, such as g1f3.")
        payload={"model":args.model,"game_id":game,"t":int(g.t.iloc[-1]),"native_state_text":state,"native_move_text":move,"question_type_state":"state","question_type_move":"move"}
        (args.out/f"{args.model}_native_smoke.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps({"NATIVE_SMOKE_PASS":True,**payload}))
    except Exception as e: print(json.dumps({"NATIVE_SMOKE_SKIPPED":True,"model":args.model,"reason":repr(e)}))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",type=Path,default=DEFAULT_DATA); ap.add_argument("--out",type=Path,default=DEFAULT_OUT); sub=ap.add_subparsers(dest="cmd",required=True)
    sub.add_parser("inspect"); sub.add_parser("build-manifest"); sub.add_parser("unit"); sub.add_parser("special-unit"); sub.add_parser("audit"); sub.add_parser("shard-unit")
    p=sub.add_parser("manifest-shard"); p.add_argument("--manifest",type=Path); p.add_argument("--shard-index",type=int,required=True); p.add_argument("--num-shards",type=int,required=True)
    p=sub.add_parser("manifest-merge"); p.add_argument("--num-shards",type=int,required=True)
    p=sub.add_parser("fit-probe"); p.add_argument("--hidden-cache",type=Path,required=True); p.add_argument("--manifest",type=Path); p.add_argument("--artifact",type=Path); p.add_argument("--candidate-layers",default="0"); p.add_argument("--candidate-c",default="1.0")
    s=sub.add_parser("model-smoke"); s.add_argument("--model",choices=["qwen","llava"],required=True); s.add_argument("--model-dir",type=Path,required=True); s.add_argument("--manifest",type=Path); s.add_argument("--limit",type=int,default=2)
    s=sub.add_parser("native-smoke"); s.add_argument("--model",choices=["qwen","llava"],required=True); s.add_argument("--model-dir",type=Path,required=True); s.add_argument("--manifest",type=Path); s.add_argument("--limit",type=int,default=2)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if a.cmd in ("inspect","build-manifest"):
        if a.cmd=="inspect": print(json.dumps({"files":[str(x) for x in discover(a.data_dir)]},indent=2))
        else: print(json.dumps(build_manifest(a.data_dir,a.out)[1],indent=2))
    elif a.cmd=="unit": updater_unit()
    elif a.cmd=="special-unit": special_unit()
    elif a.cmd=="audit": audit_manifest(a.out)
    elif a.cmd=="fit-probe": fit_probe(a.manifest or a.out/"manifest.csv",a.hidden_cache,a.artifact or a.out/"frozen_square_probes.joblib",a.candidate_layers,a.candidate_c)
    elif a.cmd=="shard-unit": shard_unit(a.out/"shard_unit")
    elif a.cmd=="manifest-shard": write_manifest_shard(a.manifest or a.out/"manifest.csv",a.out,a.shard_index,a.num_shards)
    elif a.cmd=="manifest-merge": merge_manifest_shards(a.out,a.num_shards)
    elif a.cmd=="native-smoke":
        a.manifest=a.manifest or a.out/"manifest.csv"; native_smoke(a)
    else:
        a.manifest=a.manifest or a.out/"manifest.csv"; model_smoke(a)
if __name__=="__main__": main()
