"""Phase-1 hidden-state probing pilot for StateRev-VL (representation localization ONLY).

For every prefix step in the mechanism manifest (t=1..5, all 50 trajectories by
default), re-runs the exact state-question forward pass used by the behavioral
audit (same prompt, same video window, same 8 fps processor sampling), stores the
hidden state at one fixed probe position - the LAST INPUT TOKEN before the
assistant turn begins (no generated answer tokens, hence no label leakage) - for
all transformer layers of Qwen3-VL-8B-Instruct (embedding + 36 layers).

Probe protocol (per seed, >=10 group splits by default):
  - group split by trajectory_id (no prefix of the same trajectory in train AND test);
  - ONE L2 logistic-regression probe per (target, layer) trained on TRAIN
    trajectories only; the regularization C is chosen by 3-fold CV grouped by
    trajectory INSIDE the train set (test is never seen);
  - the same probe is evaluated on the held-out test trajectories and reported
    per subset: all / event_correct+state_correct / event_correct+state_wrong /
    canonical revision success / canonical revision failure;
  - per test split: majority baseline per subset (largest test class frequency);
  - label-permutation null: TRAIN labels are permuted (n perms), the probe is
    refit with the same C and evaluated on the same held-out test rows - the
    decodability baseline under shuffled labels;
  - every held-out sample stores its probe prediction, GT-class probability and
    margin (probe_heldout_samples.csv) for later per-sample analysis.

Targets: current state S_t (Left / Middle / Right) and current event E_t
(Left-Middle / Middle-Right / Left-Right).

Already-extracted hidden states are REUSED (existing out-dir npz automatically,
plus any --reuse-npz files); the model is only run for missing keys.

--include-t0 additionally extracts the t=0 prefix (initial segment, no swap
shown) for every trajectory, as h_pre of the t=1 pairs of the state-retention
analysis (see scripts/build_state_retention_pairs.py). t=0 rows are excluded
from the probe evaluation itself.

No activation patching and no causal claims: this pilot only localizes where in
the stack the target information is linearly decodable.

Run from the project root, physical GPU 3 only:
  CUDA_VISIBLE_DEVICES=3 python scripts/run_hidden_state_probe.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _theory_of_space_utils import DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR
from run_state_rev_audit import (
    load_transformers_vl_model,
    setup_seeds,
    state_messages,
)
from run_vetbench_screening import (
    CUP_DIR,
    CUP_META,
    POSITION_NAMES,
    SWAP_OPTION_TEXT,
    _frame_cache,
    decode_frames,
    derive_ground_truth,
    load_metadata,
    prefix_frame_range,
    sampled_frame_count,
    video_processor_kwargs,
)

BEHAVIOR_AUDIT_DIR = DEFAULT_OUTPUT_DIR / "vetbench" / "behavior_audit_v2"
MANIFEST = BEHAVIOR_AUDIT_DIR / "mechanism_candidates.csv"
OUT_DIR = DEFAULT_OUTPUT_DIR / "vetbench" / "hidden_state_probe"
ALIGNED_T0_DIR = DEFAULT_OUTPUT_DIR / "vetbench" / "transformers_behavior_v1"

STATE_CLASSES = ("Left", "Middle", "Right")
EVENT_CLASSES = ("Left and Middle", "Middle and Right", "Left and Right")
STATE_INDEX = {c: i for i, c in enumerate(STATE_CLASSES)}
EVENT_INDEX = {c: i for i, c in enumerate(EVENT_CLASSES)}

# Reported evaluation subsets. "all" + the two joint classes + the two canonical
# revision subgroups (clean transition rows where the event was seen correctly
# and the previous state was known, split by whether the state was updated).
SUBSET_DEFS = {
    "all": lambda r: True,
    "event_correct_state_correct": lambda r: r["joint_class"] == "event_correct_state_correct",
    "event_correct_state_wrong": lambda r: r["joint_class"] == "event_correct_state_wrong",
    "revision_success": lambda r: bool(r["clean_revision"]) and bool(r["state_correct"]),
    "revision_failure": lambda r: bool(r["clean_revision"]) and not r["state_correct"],
}
SUBSETS = tuple(SUBSET_DEFS)
C_GRID_DEFAULT = (0.1, 1.0, 10.0)


def read_manifest(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("t", "frame_start", "frame_end", "clip_frames", "sampled_frames", "n_swaps_shown"):
            r[k] = int(r[k])
        r["sample_fps"] = float(r["sample_fps"])
        for k in ("event_correct", "state_correct", "is_transition", "prev_state_correct",
                  "clean_revision", "clean_maintenance"):
            r[k] = r[k] == "true"
    return rows


def collect_hidden_states(rows: list[dict], video_dir: Path, model, processor,
                          device: str, sample_fps: float,
                          states: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Run the state-question forward pass for every manifest row; return probe-position
    hidden states for all layers (L+1, hidden_size) per row plus metadata rows.
    Rows whose key already exists in `states` (previously extracted) are REUSED
    without a model forward."""
    n_layers = model.config.text_config.num_hidden_layers
    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_video[r["video"]].append(r)

    meta: list[dict] = []
    total = len(rows)
    done = 0
    for video, video_rows in by_video.items():
        prefix = Path(video_rows[0]["prefix_path"])
        video_path = prefix if prefix.is_absolute() else Path.cwd() / prefix
        if not video_path.exists():
            video_path = video_dir / video
        # skip the whole video if every one of its rows is already extracted
        if all(f"{r['trajectory_id']}_t{r['t']}" in states for r in video_rows):
            for r in sorted(video_rows, key=lambda x: x["t"]):
                key = f"{r['trajectory_id']}_t{r['t']}"
                hs_shape = states[key].shape
                meta.append(_probe_meta_row(r, key, sample_fps, hs_shape[0], hs_shape[1], True))
                done += 1
                print(f"  forward [{done}/{total}] {key} REUSED (no model forward)")
            continue
        container_frames = decode_frames(video_path)
        for r in sorted(video_rows, key=lambda x: x["t"]):
            key = f"{r['trajectory_id']}_t{r['t']}"
            if key in states:
                hs_shape = states[key].shape
                meta.append(_probe_meta_row(r, key, sample_fps, hs_shape[0], hs_shape[1], True))
                done += 1
                print(f"  forward [{done}/{total}] {key} REUSED (no model forward)")
                continue
            clip = container_frames[r["frame_start"]: r["frame_end"]]
            messages = state_messages(clip, r["initial_state"], r["n_swaps_shown"])
            vk = video_processor_kwargs(clip, sample_fps)
            # NOTE: do NOT pass enable_thinking here - it is not a parameter of this
            # transformers version's apply_chat_template, and any unknown kwarg makes
            # the base class REPLACE processor_kwargs with it (silently dropping the
            # fps / video_metadata needed for the 8 fps sampling). Qwen3-VL-8B's chat
            # template is identical with thinking on/off, so the prompt is unchanged.
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs=vk,
            )
            inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            probe_pos = inputs["input_ids"].shape[1] - 1
            # Use the base model (no lm_head): the full LM head would allocate a
            # (seq_len x vocab) logits tensor (~3.3 GiB for the 14.6k-token t=5 prefix)
            # and OOM the 40 GiB GPU. The base model still returns all layer
            # hidden states, which is all the probe needs.
            with torch.inference_mode():
                outputs = model.model(**inputs, output_hidden_states=True)
            hs = torch.stack(
                [h[0, probe_pos, :] for h in outputs.hidden_states], dim=0
            ).float().cpu().numpy()
            if hs.shape[0] != n_layers + 1:
                raise RuntimeError(f"Expected {n_layers + 1} hidden-state tensors, got {hs.shape[0]}")
            states[key] = hs
            meta.append(_probe_meta_row(r, key, sample_fps, hs.shape[0], hs.shape[1], False))
            done += 1
            print(f"  forward [{done}/{total}] {key} seq_len={inputs['input_ids'].shape[1]}")
        _frame_cache.pop(str(video_path), None)
        torch.cuda.empty_cache()
    return states, meta


def _probe_meta_row(r: dict, key: str, sample_fps: float, n_layers_stored: int,
                    hidden_size: int, reused: bool) -> dict:
    return {
        "key": key,
        "trajectory_id": r["trajectory_id"],
        "video": r["video"],
        "t": r["t"],
        "n_swaps_shown": r["n_swaps_shown"],
        "frame_range": [r["frame_start"], r["frame_end"]],
        "sample_fps": sample_fps,
        "probe_position": "last input token before assistant turn",
        "n_layers_stored": n_layers_stored,
        "hidden_size": hidden_size,
        "reused": reused,
        "gt_state": r["gt_state"],
        "gt_event": r["gt_event"],
        "joint_class": r["joint_class"],
        "clean_revision": r["clean_revision"],
        "clean_maintenance": r["clean_maintenance"],
        "state_correct_behavior": r["state_correct"],
        "event_correct_behavior": r["event_correct"],
    }


def add_t0_rows(rows: list[dict], gt_by_video: dict, sample_fps: float) -> list[dict]:
    """Synthesize the t=0 prefix row (initial segment, NO swap shown) for every
    trajectory that lacks one. These rows exist so the t=1 state-retention pairs
    have an h_pre (the hidden state BEFORE the first swap). The prompt is the same
    state question with n_swaps_shown=0 over frames [0, shuffle_start) - exactly
    the job run by the behavioral audit's prefix-state experiment (t=0).
    Behavior meta (state_correct) is read from the aligned prefix-state run when
    available; the row is otherwise identical in schema to the manifest rows."""
    t0_state_correct: dict[str, bool] = {}
    t0_file = ALIGNED_T0_DIR / "prefix_state.jsonl"
    if t0_file.exists():
        for line in t0_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if int(d["t"]) == 0:
                t0_state_correct[d["video"]] = bool(d["state_correct"])
    have = {(r["trajectory_id"], r["t"]) for r in rows}
    added: list[dict] = []
    for r0 in rows:
        tr, video = r0["trajectory_id"], r0["video"]
        if (tr, 0) in have:
            continue
        gt = gt_by_video[video]
        start, end = prefix_frame_range(0)
        n_frames = end - start
        dummy = np.zeros((n_frames, 1, 1), dtype=np.uint8)
        added.append({
            "trajectory_id": tr,
            "t": 0,
            "video": video,
            "prefix_path": r0["prefix_path"],
            "frame_start": start,
            "frame_end": end,
            "clip_frames": n_frames,
            "sampled_frames": sampled_frame_count(dummy, sample_fps),
            "sample_fps": sample_fps,
            "initial_state": POSITION_NAMES[gt["initial_pos"]],
            "n_swaps_shown": 0,
            "gt_state": POSITION_NAMES[gt["states"][0]],
            "gt_event": "",
            "joint_class": "",
            "is_transition": False,
            "event_correct": False,
            "prev_state_correct": False,
            "clean_revision": False,
            "clean_maintenance": False,
            "state_correct": t0_state_correct.get(video, False),
            "_t0_synthetic": True,
        })
    return rows + added


def group_split(trajectory_ids: list[str], seed: int, train_frac: float) -> tuple[set, set]:
    order = list(trajectory_ids)
    rng = random.Random(seed)
    rng.shuffle(order)
    n_train = max(1, int(round(train_frac * len(order))))
    return set(order[:n_train]), set(order[n_train:])


# --------------------------------------------------------------------------
# Probe evaluation: ONE probe per (target, layer, seed) trained on the TRAIN
# trajectories only (C chosen by inner group CV inside train), then evaluated
# on the held-out test trajectories for all reported subsets; label-permutation
# null and per held-out sample scores. Workers run in a single forked process
# pool; the (n, L+1, D) hidden-state tensor is shared with workers through a
# zero-copy mmap file so job arguments stay small.
# --------------------------------------------------------------------------


def _fit_lr(X, y, C, max_iter=2000):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=max_iter).fit(X, y)


def _select_c(X_tr, y_tr, trajs_tr, c_grid, rng_key: str):
    """Choose C by 3-fold CV grouped by trajectory, using TRAIN trajectories only
    (the test group is never seen). Ties keep the first (smallest) C."""
    from sklearn.preprocessing import StandardScaler
    rng = random.Random(rng_key)
    order = list(sorted(set(trajs_tr)))
    rng.shuffle(order)
    n_folds = min(3, len(order))
    folds = [set(order[i * len(order) // n_folds:(i + 1) * len(order) // n_folds])
             for i in range(n_folds)]
    best_c, best_score = 1.0, -1.0
    for C in c_grid:
        scores = []
        for hold in folds:
            te = [j for j, tr in enumerate(trajs_tr) if tr in hold]
            tr = [j for j, tr in enumerate(trajs_tr) if tr not in hold]
            if not te:
                continue
            yh, yt = y_tr[te], y_tr[tr]
            if len(set(yh.tolist())) < 2 or len(set(yt.tolist())) < 2:
                continue
            sc = StandardScaler().fit(X_tr[tr])
            acc = float((_fit_lr(sc.transform(X_tr[tr]), yt, C)
                         .predict(sc.transform(X_tr[te])) == yh).mean())
            scores.append(acc)
        if scores and float(np.mean(scores)) > best_score + 1e-12:
            best_score, best_c = float(np.mean(scores)), C
    return best_c


def _pad_proba(proba: np.ndarray, classes, n_classes: int) -> np.ndarray:
    """Map a (n, k) predict_proba (k <= n_classes, e.g. a train set missing one
    class) onto the full n-class layout."""
    P = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    for ci, c in enumerate(classes):
        P[:, int(c)] = proba[:, ci]
    return P


def _probe_worker(job: dict) -> dict:
    """One (seed, target, layer) probe job: C selection on train only, final
    probe, per-subset held-out accuracy + majority baseline, label-permutation
    null, and per held-out sample prediction / GT probability / margin.
    The hidden-state tensor arrives as a zero-copy mmap file path."""
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    seed, target, layer = job["seed"], job["target"], job["layer"]
    rows, index = job["rows"], job["index"]
    y_all, train_g, test_g = job["y_all"], job["train_g"], job["test_g"]
    c_grid, n_perms = job["c_grid"], job["n_perms"]
    n_classes, cls_names = job["n_classes"], job["cls_names"]
    gt_key = "gt_state" if target == "state" else "gt_event"
    X_all = np.load(job["x_path"], mmap_mode="r")

    train_idx = [i for i in range(len(rows)) if rows[i]["trajectory_id"] in train_g]
    test_idx = [i for i in range(len(rows)) if rows[i]["trajectory_id"] in test_g]
    y_tr, y_te = y_all[train_idx], y_all[test_idx]
    X_tr_raw, X_te_raw = X_all[train_idx, layer, :], X_all[test_idx, layer, :]

    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(X_tr_raw)
        X_tr, X_te = scaler.transform(X_tr_raw), scaler.transform(X_te_raw)
        c_best = _select_c(X_tr, y_tr, [rows[i]["trajectory_id"] for i in train_idx],
                           c_grid, f"{seed}-{target}-{layer}")
        clf = _fit_lr(X_tr, y_tr, c_best)
        P = _pad_proba(clf.predict_proba(X_te), clf.classes_, n_classes)
        pred_te = P.argmax(axis=1)

        subsets = {name: np.array([SUBSET_DEFS[name](rows[i]) for i in test_idx])
                   for name in SUBSETS}
        subsets = {name: mask for name, mask in subsets.items() if mask.any()}

        # label-permutation null: shuffle the TRAIN labels, refit with the SAME
        # selected C, evaluate on the same held-out test rows.
        rng = random.Random(f"perm-{seed}-{target}-{layer}")
        n_tr = len(y_tr)
        null_acc: dict[str, list[float]] = {name: [] for name in subsets}
        for _ in range(n_perms):
            y_perm = y_tr[rng.sample(range(n_tr), n_tr)]
            if len(set(y_perm.tolist())) < 2:
                continue
            c = _fit_lr(X_tr, y_perm, c_best)
            pp = _pad_proba(c.predict_proba(X_te), c.classes_, n_classes).argmax(axis=1)
            for name, mask in subsets.items():
                null_acc[name].append(float((pp[mask] == y_te[mask]).mean()))

    samples = []
    for j, i in enumerate(test_idx):
        srt = np.sort(P[j])[::-1]
        samples.append({
            "key": index[i],
            "trajectory_id": rows[i]["trajectory_id"],
            "t": rows[i]["t"],
            "gt_class": rows[i][gt_key],
            "pred_class": cls_names[pred_te[j]],
            "correct": bool(pred_te[j] == y_te[j]),
            "prob_gt": float(P[j, y_te[j]]),
            "prob_top1": float(srt[0]),
            "prob_top2": float(srt[1]),
            "margin": float(srt[0] - srt[1]),
            "in_event_correct_state_correct": bool(SUBSET_DEFS["event_correct_state_correct"](rows[i])),
            "in_event_correct_state_wrong": bool(SUBSET_DEFS["event_correct_state_wrong"](rows[i])),
            "in_revision_success": bool(SUBSET_DEFS["revision_success"](rows[i])),
            "in_revision_failure": bool(SUBSET_DEFS["revision_failure"](rows[i])),
        })

    sub_out = {}
    for name, mask in subsets.items():
        idx = np.flatnonzero(mask)
        class_counts = Counter(y_te[idx].tolist())
        sub_out[name] = {
            "test_n": int(len(idx)),
            "accuracy": float((pred_te[idx] == y_te[idx]).mean()),
            # largest class frequency (count / n), not the class label
            "majority_baseline": float(class_counts.most_common(1)[0][1] / len(idx)),
            "null": null_acc[name],
        }
    return {
        "seed": seed, "target": target, "layer": layer, "c_best": float(c_best),
        "train_n": len(train_idx), "test_n": len(test_idx),
        "subsets": sub_out, "samples": samples,
    }


def evaluate_probes(rows: list[dict], states: dict[str, np.ndarray],
                    split_seeds: list[int], train_frac: float,
                    c_grid: list[float], n_perms: int,
                    n_jobs: int) -> tuple[list[dict], list[dict], dict, dict[str, Any]]:
    """Run all probe jobs. Returns (result_rows, sample_rows, nulls, probe_meta)
    where nulls[(target, subset, layer_name)] is the list of per-seed null
    accuracy distributions (for saving the pooled permutation-null distribution).

    A single process pool is created (forked once); the hidden-state tensor is
    written to a temp .npy that workers access with mmap_mode='r' (zero-copy)."""
    import shutil
    import tempfile
    from concurrent.futures import ProcessPoolExecutor

    keys = [f"{r['trajectory_id']}_t{r['t']}" for r in rows]
    X_all = np.stack([states[k] for k in keys], axis=0).astype(np.float32)  # (n, L+1, D)
    n, L1, D = X_all.shape
    tmpdir = Path(tempfile.mkdtemp(prefix="probe_X_"))
    x_path = tmpdir / "X_all.npy"
    np.save(x_path, X_all)
    del X_all
    traj_ids = sorted({r["trajectory_id"] for r in rows})
    y_by_target = {
        "state": np.array([STATE_INDEX[r["gt_state"]] for r in rows]),
        "event": np.array([EVENT_INDEX[r["gt_event"]] for r in rows]),
    }

    result_rows: list[dict] = []
    sample_rows: list[dict] = []
    nulls: dict[tuple, list] = {}
    try:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for seed in split_seeds:
                train_g, test_g = group_split(traj_ids, seed, train_frac)
                print(f"[seed {seed}] train trajectories: {len(train_g)}, "
                      f"test trajectories: {len(test_g)}")
                for target in ("state", "event"):
                    jobs = [{
                        "seed": seed, "target": target, "layer": layer,
                        "x_path": str(x_path),
                        "rows": rows, "index": keys,
                        "y_all": y_by_target[target],
                        "train_g": train_g, "test_g": test_g,
                        "c_grid": list(c_grid), "n_perms": n_perms, "n_classes": 3,
                        "cls_names": STATE_CLASSES if target == "state" else EVENT_CLASSES,
                    } for layer in range(L1)]
                    for res in ex.map(_probe_worker, jobs, chunksize=2):
                        layer_name = "embedding" if res["layer"] == 0 else f"layer_{res['layer']:02d}"
                        for name, s in res["subsets"].items():
                            null = np.asarray(s["null"], dtype=np.float64) if s["null"] else np.array([])
                            result_rows.append({
                                "seed": seed, "target": target, "subset": name,
                                "layer": res["layer"], "layer_name": layer_name,
                                "c": res["c_best"],
                                "train_n": res["train_n"], "test_n": s["test_n"],
                                "accuracy": s["accuracy"],
                                "majority_baseline": s["majority_baseline"],
                                "null_mean": float(null.mean()) if len(null) else "",
                                "null_std": float(null.std()) if len(null) else "",
                                "null_p95": float(np.percentile(null, 95)) if len(null) else "",
                                "p_value": float((null >= s["accuracy"]).mean()) if len(null) else "",
                            })
                            nulls.setdefault((target, name, layer_name), []).append(list(null))
                        for s in res["samples"]:
                            sample_rows.append({
                                "seed": seed, "target": target,
                                "layer": res["layer"], "layer_name": layer_name,
                                **s,
                            })
                        all_s = res["subsets"].get("all")
                        print(f"  probe seed={seed} {target}/{layer_name} c={res['c_best']:.3f} "
                              f"acc(all)={'nan' if not all_s else round(all_s['accuracy'], 3)} "
                              f"(train={res['train_n']}, test={all_s['test_n'] if all_s else 0})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    probe_meta = {"n_rows": n, "n_layers": L1, "hidden_size": D,
                  "n_trajectories": len(traj_ids), "splits": list(split_seeds),
                  "train_frac": train_frac}
    return result_rows, sample_rows, nulls, probe_meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-1 hidden-state probing pilot: per-layer linear probes for the "
                    "current state S_t and event E_t at the last prompt token.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST,
                        help="mechanism_candidates.csv (drives which prefixes to forward)")
    parser.add_argument("--meta", type=Path, default=CUP_META)
    parser.add_argument("--video-dir", type=Path, default=CUP_DIR)
    parser.add_argument("--sample-fps", type=float, default=8.0)
    parser.add_argument("--max-trajectories", type=int, default=0,
                        help="Limit to the first N trajectories (0 = all). For smoke tests.")
    parser.add_argument("--split-seeds", type=str, default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated seeds for the trajectory group splits (>=10 default).")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--c-grid", type=str, default="0.1,1.0,10.0",
                        help="Comma-separated L2 regularization candidates; chosen per "
                             "(target, layer, seed) by 3-fold group CV inside the TRAIN set only.")
    parser.add_argument("--n-perms", type=int, default=100,
                        help="Label-permutation null repetitions per (target, layer, seed).")
    parser.add_argument("--n-jobs", type=int, default=min(16, os.cpu_count() or 1),
                        help="Parallel worker processes for the probe phase (CPU).")
    parser.add_argument("--reuse-npz", type=Path, action="append", default=[],
                        help="Previously saved hidden_states.npz to reuse (repeatable). "
                             "The existing out-dir npz is also reused automatically.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate manifest + GT and print the plan without loading the model.")
    parser.add_argument("--include-t0", action="store_true",
                        help="Also extract the t=0 prefix (initial segment, no swap shown) "
                             "for every trajectory. Needed as h_pre of the t=1 pairs of the "
                             "state-retention analysis. Only the missing cup_XXX_t0 keys are "
                             "run through the model; existing keys are reused. t=0 rows are "
                             "NOT part of the probe evaluation (probe targets are t=1..5).")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    gt_by_video = {e["video"]: derive_ground_truth(e) for e in load_metadata()}
    # sanity: manifest labels must match the derived GT
    for r in rows:
        gt = gt_by_video[r["video"]]
        assert 1 <= r["t"] <= 5 and r["n_swaps_shown"] == r["t"], f"bad prefix index: {r}"
        assert r["gt_state"] == POSITION_NAMES[gt["states"][r["t"]]], f"state GT mismatch: {r}"
        assert r["gt_event"] == SWAP_OPTION_TEXT[gt["swaps"][r["t"] - 1]], f"event GT mismatch: {r}"
        assert r["initial_state"] == POSITION_NAMES[gt["initial_pos"]], f"initial mismatch: {r}"
    if args.include_t0:
        rows = add_t0_rows(rows, gt_by_video, args.sample_fps)
        for r in rows:
            if r["t"] == 0:
                gt = gt_by_video[r["video"]]
                assert r["gt_state"] == POSITION_NAMES[gt["initial_pos"]], f"t0 GT mismatch: {r}"
                assert r["frame_start"] == 0 and r["n_swaps_shown"] == 0

    trajectories = sorted({r["trajectory_id"] for r in rows})
    if args.max_trajectories:
        keep = set(trajectories[: args.max_trajectories])
        rows = [r for r in rows if r["trajectory_id"] in keep]
        trajectories = sorted(keep)
    split_seeds = [int(s) for s in args.split_seeds.split(",") if s.strip()]
    c_grid = [float(c) for c in args.c_grid.split(",") if c.strip()]
    print(f"Manifest rows: {len(rows)} across {len(trajectories)} trajectories; "
          f"probe splits: {len(split_seeds)} (train_frac={args.train_frac}); "
          f"C grid: {c_grid}; perms/job: {args.n_perms}")

    # t=0 rows exist only as h_pre for the state-retention pairs; they are
    # prompt-trivial (initial position is named in the text) and are excluded
    # from the probe evaluation so historical probe results stay comparable.
    probe_rows = [r for r in rows if r["t"] >= 1]

    if args.dry_run:
        by_video = defaultdict(int)
        for r in rows:
            by_video[r["video"]] += 1
        print("Per-trajectory forward passes:", dict(list(by_video.items())[:3]),
              "... total", len(rows))
        print(f"Probe rows (t=1..5 only): {len(probe_rows)}")
        print(f"Subset sizes (probe rows): "
              f"{ {name: sum(1 for r in probe_rows if SUBSET_DEFS[name](r)) for name in SUBSETS} }")
        print("Dry run OK: manifest valid, GT cross-checked.")
        return

    setup_seeds(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse previously extracted hidden states: the existing out-dir npz
    # automatically, plus any explicit --reuse-npz files.
    states: dict[str, np.ndarray] = {}
    existing = args.out_dir / "hidden_states.npz"
    if existing.exists():
        with np.load(existing) as z:
            states.update({k: z[k] for k in z.files})
        print(f"Reusing {len(states)} saved hidden states from {existing}")
    for p in args.reuse_npz:
        with np.load(p) as z:
            added = {k: z[k] for k in z.files if k not in states}
            states.update(added)
        print(f"Reusing {len(added)} hidden states from {p}")

    n_missing = sum(1 for r in rows if f"{r['trajectory_id']}_t{r['t']}" not in states)
    if n_missing == 0:
        # probe-only rerun: every hidden state is already saved, no model load needed
        print(f"All {len(rows)} hidden states already saved - skipping model load "
              f"and all forward passes.")
        meta = [_probe_meta_row(r, f"{r['trajectory_id']}_t{r['t']}", args.sample_fps,
                                states[f"{r['trajectory_id']}_t{r['t']}"].shape[0],
                                states[f"{r['trajectory_id']}_t{r['t']}"].shape[1], True)
                for r in rows]
    else:
        print(f"Running state-question forward passes: {n_missing} new, "
              f"{len(rows) - n_missing} reused (probe position = last input token)...")
        model, processor = load_transformers_vl_model(args.model_dir)
        states, meta = collect_hidden_states(rows, args.video_dir, model, processor,
                                             args.device, args.sample_fps, states)
        del model
        torch.cuda.empty_cache()
    n_layers = next(iter(states.values())).shape[0] - 1
    np.savez(args.out_dir / "hidden_states.npz", **states)
    with open(args.out_dir / "probe_meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Saved {len(states)} hidden-state tensors ({n_layers + 1} x {states[next(iter(states))].shape[1]} each) to "
          f"{args.out_dir / 'hidden_states.npz'}")

    print("Training per-layer logistic probes (group split by trajectory, "
          "C selected on train only, label-permutation null)...")
    results, sample_rows, nulls, probe_meta = evaluate_probes(
        probe_rows, states, split_seeds, args.train_frac, c_grid, args.n_perms, args.n_jobs)

    with open(args.out_dir / "probe_results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "target", "subset", "layer", "layer_name", "c",
            "train_n", "test_n", "accuracy", "majority_baseline",
            "null_mean", "null_std", "null_p95", "p_value"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in r.items()})

    sample_fields = ["seed", "target", "layer", "layer_name", "key", "trajectory_id", "t",
                     "gt_class", "pred_class", "correct", "prob_gt", "prob_top1", "prob_top2",
                     "margin", "in_event_correct_state_correct", "in_event_correct_state_wrong",
                     "in_revision_success", "in_revision_failure"]
    with open(args.out_dir / "probe_heldout_samples.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sample_fields)
        writer.writeheader()
        for s in sample_rows:
            writer.writerow({k: (f"{s[k]:.6f}" if k in ("prob_gt", "prob_top1",
                                                         "prob_top2", "margin") else s[k])
                             for k in sample_fields})

    # per (target, subset, layer): mean/std over seeds + pooled permutation null
    layer_order = ["embedding"] + [f"layer_{i:02d}" for i in range(1, n_layers + 1)]
    probe_tables: dict[str, dict] = {}
    majority_tables: dict[str, dict] = {}
    for target in ("state", "event"):
        for subset in SUBSETS:
            table: dict[str, Any] = {}
            for layer_name in layer_order:
                sel = [r for r in results
                       if r["target"] == target and r["subset"] == subset
                       and r["layer_name"] == layer_name]
                if not sel:
                    table[layer_name] = {"mean": None, "std": None, "n_splits": 0}
                    continue
                accs = [r["accuracy"] for r in sel]
                pvals = [r["p_value"] for r in sel if r["p_value"] != ""]
                null_lists = nulls.get((target, subset, layer_name), [])
                pooled = (np.concatenate([np.asarray(v, dtype=np.float64)
                                           for v in null_lists])
                          if null_lists else np.array([]))
                table[layer_name] = {
                    "mean": float(np.mean(accs)),
                    "std": float(np.std(accs)),
                    "n_splits": len(accs),
                    "null_mean": float(pooled.mean()) if len(pooled) else None,
                    "null_std": float(pooled.std()) if len(pooled) else None,
                    "null_p05": float(np.percentile(pooled, 5)) if len(pooled) else None,
                    "null_p50": float(np.percentile(pooled, 50)) if len(pooled) else None,
                    "null_p90": float(np.percentile(pooled, 90)) if len(pooled) else None,
                    "null_p95": float(np.percentile(pooled, 95)) if len(pooled) else None,
                    "p_value": float(np.mean(pvals)) if pvals else None,
                    "null_distribution": [float(x) for x in pooled],
                }
            probe_tables[f"{target}/{subset}"] = table
            # majority baseline is layer-independent: one value per seed
            majs = {}
            for r in results:
                if r["target"] == target and r["subset"] == subset:
                    majs.setdefault(r["seed"], r["majority_baseline"])
            if majs:
                vals = list(majs.values())
                majority_tables[f"{target}/{subset}"] = {
                    "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    "n_splits": len(vals),
                }

    chance = 1.0 / 3
    summary = {
        "config": {
            "model_dir": str(args.model_dir),
            "manifest": str(args.manifest),
            "n_layers": n_layers,
            "probe_position": "last input token of the rendered state-question prompt "
                              "(before the assistant turn; no generated tokens -> no label leakage)",
            "forward_pass": "identical to the behavioral audit prefix-state input "
                            "(same prompt, window, 8 fps processor sampling; the video "
                            "spatial grid is set by Qwen3VLVideoProcessor.smart_resize, a "
                            "shared transformers function that vLLM also imports, so the "
                            "frame-count-dependent resize budget applies identically on "
                            "both sides)",
            "targets": {"state": list(STATE_CLASSES), "event": list(EVENT_CLASSES)},
            "probe_model": "StandardScaler + L2 LogisticRegression (lbfgs, max_iter=2000) "
                           "per layer; ONE probe per (target, layer, seed) trained on the "
                           "TRAIN trajectories only, then evaluated on held-out test "
                           "trajectories for all reported subsets",
            "c_selection": f"C in {c_grid} chosen by 3-fold CV grouped by trajectory "
                           "INSIDE the train trajectories only (test never seen); "
                           "ties keep the smallest C",
            "group_split": f"by trajectory_id, {len(split_seeds)} seeds {split_seeds}, "
                           f"train_frac={args.train_frac}; no trajectory crosses train/test",
            "subsets": {
                "all": "all held-out test rows",
                "event_correct_state_correct": "behavioral joint class EC+SC",
                "event_correct_state_wrong": "behavioral joint class EC+SW",
                "revision_success": "canonical revision (true transition + event correct "
                                    "+ previous state correct) with pred_t == GT_t",
                "revision_failure": "canonical revision with pred_t != GT_t",
            },
            "label_permutation_null": f"{args.n_perms} permutations of the TRAIN labels per "
                                      f"(target, layer, seed); the selected C is reused for "
                                      "the permuted refits; null value = held-out test "
                                      "accuracy of a probe trained on permuted labels; "
                                      "p_value = fraction of null values >= observed",
            "majority_baseline": "per test split (seed) and subset: largest class frequency "
                                 "among that split's test rows (layer-independent)",
            "n_forward_passes": len(meta),
            "reused_hidden_states": sum(1 for m in meta if m["reused"]),
            "n_rows": probe_meta["n_rows"],
            "n_trajectories": probe_meta["n_trajectories"],
            "hidden_size": probe_meta["hidden_size"],
        },
        "chance_baseline": chance,
        "majority_baseline": majority_tables,
        "probe": probe_tables,
        "subset_rows_in_manifest": {name: sum(1 for r in rows if SUBSET_DEFS[name](r))
                                    for name in SUBSETS},
        "interpretation_note": "representation localization only (linear decodability on "
                               "held-out trajectories); no causal claims, no activation "
                               "patching in this phase",
    }
    (args.out_dir / "probe_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nprobe_results.csv rows: {len(results)}; "
          f"probe_heldout_samples.csv rows: {len(sample_rows)}")
    print(f"Outputs in {args.out_dir}: hidden_states.npz, probe_meta.jsonl, "
          f"probe_results.csv, probe_heldout_samples.csv, probe_summary.json")


if __name__ == "__main__":
    main()
