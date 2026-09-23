#!/usr/bin/env python3
"""Discovery-only optimization/interface diagnostic for PSF v5.

This is deliberately not a new model or a validation sweep.  It uses only
the existing discovery split and cached hidden/text features.  The diagnostic
asks two bounded questions:

1. Does the audit-style *full-data update* need more than 50 optimizer cycles?
2. Do the already-probed Qwen/LLaVA hidden layers work better than the
   quantile taps currently used by ``tap_layers``?

No held-out validation rows are read or written by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staterev.psf import (  # noqa: E402
    ChessAdapter,
    KeepSetPSF,
    ShellAdapter,
    TextFeatureStore,
    Trajectory,
    keep_set_loss,
)

OUT = ROOT / "outputs/psf_v1/keep_set_v5_optimization_diagnostic"
CACHE = {
    "qwen": ROOT / "outputs/psf_v1/diagnostic_v1/text_cache/qwen_semantic.npz",
    "llava": ROOT / "outputs/psf_v1/diagnostic_v1/text_cache/llava_semantic.npz",
}
SHELL_SPLIT = ROOT / "outputs/vetbench/content_disjoint_split_v1/discovery_validation_split.json"


class DeviceTextStore:
    """Read-only raw cache prefetched to the selected device.

    The learned text projector still runs on every optimizer cycle; only the
    frozen 4096-d input vectors are prefetched.  This changes no model
    computation and avoids repeatedly copying tiny arrays from CPU.
    """

    def __init__(self, path: Path, device: torch.device):
        z = np.load(path)
        self.path = path
        self.strict = True
        self._values = {k: torch.from_numpy(np.asarray(z[k], dtype="float32")).to(device)
                        for k in z.files}

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> torch.Tensor:
        k = self.key(text)
        if k not in self._values:
            raise KeyError(f"Missing frozen text feature for {text!r}")
        return self._values[k]


def load_data(task: str, model: str):
    if task == "chess":
        return ChessAdapter(model).load(split="discovery")
    ds = ShellAdapter(model).load()
    plan = json.loads(SHELL_SPLIT.read_text())
    ids = set(plan["discovery_trajectories"])
    return [x for x in ds if x.trajectory_id in ids]


def split_train_dev(data, seed=17, frac=0.2):
    rng = np.random.default_rng(seed)
    q = list(data)
    rng.shuffle(q)
    n = max(1, round(len(q) * frac))
    return q[n:], q[:n]


def preload(data, device):
    return [replace(x, hidden=x.hidden.to(device), state_ids=x.state_ids.to(device)) for x in data]


def raw_layer_data(task: str, model: str, layer: int):
    """Load a selected existing hidden layer and repeat it over four tap slots."""
    if task == "shell":
        adapter = ShellAdapter(model)
        behavior = pd.read_csv(adapter.behavior_path)
        plan = json.loads(SHELL_SPLIT.read_text())
        behavior = behavior[behavior.trajectory_id.isin(set(plan["discovery_trajectories"]))]
        hidden = np.load(adapter.hidden_path)
        states = adapter.states
        text = adapter.texts
        out = []
        for tid, g in behavior.sort_values(["trajectory_id", "t"]).groupby("trajectory_id"):
            rows = [r for r in g.itertuples() if f"{tid}_t{int(r.t)}" in hidden.files]
            if not rows:
                continue
            hs = []
            for r in rows:
                x = np.asarray(hidden[f"{tid}_t{int(r.t)}"])
                if x.ndim == 1:
                    one = x
                else:
                    one = x[min(max(int(layer), 0), x.shape[0] - 1)]
                hs.append(np.repeat(one[None], 4, axis=0))
            ids = torch.tensor([states.index(str(r.gt_state).title()) for r in rows])
            native = torch.tensor([
                states.index(str(r.state_pred).title())
                if str(r.state_pred).title() in states else -1 for r in rows
            ])
            out.append(Trajectory(
                model, "shell", str(tid), torch.tensor(np.stack(hs)),
                "the tracked ball", text,
                states.index(str(rows[0].initial_state).title()), ids, native, False,
            ))
        return out

    adapter = ChessAdapter(model)
    manifest = pd.read_csv(ROOT / "outputs/metbench_chess/single_piece_tracking_pilot_qwen_v1/pilot_manifest.csv")
    manifest = manifest[manifest.protocol_split.eq("discovery")]
    hidden = np.load(adapter.root / "hidden_discovery.npz")
    out = []
    for tid, g in manifest.sort_values(["game_id", "t"]).groupby("game_id"):
        rows = [r for r in g.itertuples() if f"{tid}_t{int(r.t)}" in hidden.files]
        if not rows:
            continue
        hs = []
        for r in rows:
            x = np.asarray(hidden[f"{tid}_t{int(r.t)}"])
            one = x if x.ndim == 1 else x[min(max(int(layer), 0), x.shape[0] - 1)]
            hs.append(np.repeat(one[None], 4, axis=0))
        ids = torch.tensor([adapter.squares.index(str(r.current_state)) for r in rows])
        out.append(Trajectory(
            model, "chess", str(tid), torch.tensor(np.stack(hs)),
            "the white knight initially on g1", adapter.texts,
            adapter.squares.index(str(rows[0].initial_square)), ids, None, False,
        ))
    return out


def task_pos_weight(data):
    unchanged = changed = 0
    for tr in data:
        y = tr.state_ids
        prev = torch.cat([torch.tensor([tr.initial_state_id], device=y.device), y[:-1]])
        c = y.ne(prev)
        unchanged += int((~c).sum())
        changed += int(c.sum())
    return unchanged / max(changed, 1)


def rows(model, data):
    out = []
    model.eval()
    with torch.no_grad():
        for tr in data:
            o = model.forward_trajectory(tr)
            y = tr.state_ids.to(o["final_probs"].device)
            prev = torch.cat([torch.tensor([tr.initial_state_id], device=y.device), y[:-1]])
            changed = y.ne(prev)
            for i in range(len(y)):
                out.append({
                    "task": tr.task_name, "trajectory_id": tr.trajectory_id,
                    "t": i + 1, "gt": int(y[i]), "changed": int(changed[i]),
                    "direct_pred": int(o["direct_logits"][i].argmax()),
                    "tracker_pred": int(o["track_probs"][i].argmax()),
                    "fusion_pred": int(o["final_probs"][i].argmax()),
                    "update_pred": int(torch.sigmoid(o["update_logits"][i]) >= 0.5),
                    "set_pred": int(o["set_logits"][i].argmax()),
                    "set_correct": int(o["set_logits"][i].argmax() == y[i]),
                })
    return pd.DataFrame(out)


def metrics(frame, tag, cycle):
    result = []
    for task, g in frame.groupby("task"):
        y = g.changed.to_numpy(dtype=int)
        p = g.update_pred.to_numpy(dtype=int)
        tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
        rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        changed = g[g.changed.eq(1)]
        changed_acc = float((changed["fusion_pred"] == changed["gt"]).mean()) if len(changed) else np.nan
        unchanged = g[g.changed.eq(0)]
        unchanged_acc = float((unchanged["fusion_pred"] == unchanged["gt"]).mean()) if len(unchanged) else np.nan
        det_ba = 0.5 * (rec + spec)
        set_acc = float(changed.set_correct.mean()) if len(changed) else np.nan
        balanced = float(np.nanmean([changed_acc, unchanged_acc]))
        score = float(np.nanmean([balanced, det_ba, set_acc]))
        result.append({
            "tag": tag, "cycle": cycle, "task": task, "n": len(g),
            "overall": float((g["fusion_pred"] == g["gt"]).mean()), "changed": changed_acc, "unchanged": unchanged_acc,
            "detector_balanced_accuracy": det_ba, "detector_precision": tp / max(tp + fp, 1),
            "detector_recall": rec, "detector_f1": 2 * (tp / max(tp + fp, 1)) * rec / max(tp / max(tp + fp, 1) + rec, 1e-8),
            "set_accuracy": set_acc, "predicted_set_rate": float(p.mean()),
            "true_set_rate": float(y.mean()), "tracker": float((g["tracker_pred"] == g["gt"]).mean()),
            "direct": float((g["direct_pred"] == g["gt"]).mean()), "balanced_score": score,
        })
    return pd.DataFrame(result)


def make_model(model_name, device, seed=17):
    torch.manual_seed(seed)
    return KeepSetPSF(
        (model_name,), fusion="fixed",
        text_stores={model_name: DeviceTextStore(CACHE[model_name], device)},
    ).to(device)


def train_condition(tag, task, model_name, train, dev, device, cycles, eval_every,
                    teacher_anneal, layer=None, seed=17):
    if layer is not None:
        # The layer data are constructed outside for reproducibility; this
        # branch is retained only to label the condition in the output.
        pass
    model = make_model(model_name, device, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    pos = task_pos_weight(train)
    best_score = -float("inf"); best_state = None; history = []
    by = {task: list(train)}
    for cycle in range(1, cycles + 1):
        model.train(); order = list(by[task]); rng.shuffle(order)
        progress = min(1.0, (cycle - 1) / max(teacher_anneal - 1, 1))
        teacher_prob = 1.0 - progress
        opt.zero_grad(set_to_none=True)
        losses = []
        for tr in order:
            loss, _ = keep_set_loss(model, tr, teacher_prob=teacher_prob, pos_weight=pos)
            (loss / len(order)).backward(); losses.append(float(loss.detach()))
        grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0)); opt.step()
        if cycle == 1 or cycle % eval_every == 0 or cycle == cycles:
            trm = metrics(rows(model, train), tag, cycle)
            dvm = metrics(rows(model, dev), tag, cycle)
            for f, split in [(trm, "train"), (dvm, "dev")]:
                f["split"] = split; f["teacher_prob"] = teacher_prob
                f["loss"] = float(np.mean(losses)); f["grad_norm_before_clip"] = grad_norm
            history += [trm, dvm]
            # Explicitly balanced checkpoint score; overall alone is not used.
            score = float(dvm.balanced_score.mean())
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    final_train = metrics(rows(model, train), tag, cycles).assign(split="best_train")
    final_dev = metrics(rows(model, dev), tag, cycles).assign(split="best_dev")
    return pd.concat(history + [final_train, final_dev], ignore_index=True), model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--teacher-anneal", type=int, default=40)
    ap.add_argument("--only", default="qwen_shell,qwen_chess",
                    help="comma-separated conditions: qwen_shell,qwen_chess,llava_shell,llava_chess,llava_shell_l1,qwen_shell_l24")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(a.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda": torch.set_float32_matmul_precision("high")
    print(f"Using device: {device}", flush=True)
    specs = {
        "qwen_shell": ("shell", "qwen", None),
        "qwen_chess": ("chess", "qwen", None),
        "llava_shell": ("shell", "llava", None),
        "llava_chess": ("chess", "llava", None),
        "llava_shell_l1": ("shell", "llava", 1),
        "qwen_shell_l24": ("shell", "qwen", 24),
    }
    selected = [x for x in a.only.split(",") if x]
    all_rows = []
    for name in selected:
        if name not in specs: raise ValueError(f"unknown condition {name}")
        task, model_name, layer = specs[name]
        if layer is None:
            data = load_data(task, model_name)
        else:
            data = raw_layer_data(task, model_name, layer)
        train, dev = split_train_dev(data, seed=17 + len(task))
        train = preload(train, device); dev = preload(dev, device)
        rows_df, _ = train_condition(name, task, model_name, train, dev, device,
                                     a.cycles, a.eval_every, a.teacher_anneal, layer)
        all_rows.append(rows_df)
        print(name, rows_df[rows_df.split.eq("best_dev")].to_dict("records"), flush=True)
    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(a.out / "optimization_curves.csv", index=False)
    best = result[result.split.eq("best_dev")].copy()
    best.to_csv(a.out / "best_discovery_metrics.csv", index=False)
    lines = [
        "# PSF v5 discovery-only optimization/interface diagnostic", "",
        "No held-out validation was used. Existing discovery hidden/text caches were reused; no VLM forward, architecture search, or third task was run.", "",
        "Training uses AdamW (2e-3, weight decay 1e-4), one full-data gradient-accumulated update per cycle, teacher probability annealed to zero, and balanced dev checkpoint selection.", "",
        "| condition | split | task | overall | changed | unchanged | detector BA | SET | balanced score |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in best.sort_values("tag").iterrows():
        lines.append(f"| {r.tag} | {r.split} | {r.task} | {r.overall:.3f} | {r.changed:.3f} | {r.unchanged:.3f} | {r.detector_balanced_accuracy:.3f} | {r.set_accuracy:.3f} | {r.balanced_score:.3f} |")
    lines += ["", "Interpretation rule: train and dev both low indicates optimization/interface failure; train high with dev low indicates generalization; detector high but SET low indicates selector failure."]
    (a.out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
