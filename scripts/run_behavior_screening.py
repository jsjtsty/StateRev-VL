from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from _theory_of_space_utils import DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR


REVISION_PATH = Path("data/processed/theory_of_space_relative_revision.jsonl")
MAINTAIN_PATH = Path("data/processed/theory_of_space_relative_maintain.jsonl")
OUT_DIR = DEFAULT_OUTPUT_DIR / "behavior_screening"

LABELS = ("LEFT", "RIGHT", "ABOVE", "BELOW")
LABEL_RE = re.compile(r"\b(LEFT|RIGHT|ABOVE|BELOW)\b", re.IGNORECASE)
STRICT_FLIPS = {("LEFT", "RIGHT"), ("RIGHT", "LEFT"), ("ABOVE", "BELOW"), ("BELOW", "ABOVE")}
DISTANCE_BUCKETS = [(2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, float("inf"))]

SYSTEM_PROMPT = (
    "You are a spatial reasoning assistant. You always answer with exactly one word, "
    "either LEFT, RIGHT, ABOVE, or BELOW. No explanations."
)

SINGLE_IMAGE_PROMPT = (
    "The image below is a top-down view of a room layout (north is up). "
    "Where is the {target} relative to the {reference}? "
    "Answer with exactly one word: LEFT, RIGHT, ABOVE, or BELOW."
)

BOTH_DEFAULT_PROMPT = (
    "The first image shows an earlier state, and the second image shows the current state. "
    "Where is the {target} relative to the {reference} in the CURRENT state? "
    "Answer with exactly one word: LEFT, RIGHT, ABOVE, or BELOW."
)

BOTH_REMIND_PROMPT = (
    "The first image shows an outdated earlier state. IMPORTANT: only the second image "
    "reflects the current state and must be treated as the sole current evidence. "
    "Where is the {target} relative to the {reference} in the current state? "
    "Answer with exactly one word: LEFT, RIGHT, ABOVE, or BELOW."
)

CONDITIONS = ("old_only", "new_only", "both_default", "both_remind")


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def axis_separation(pos: dict, ref_pos: dict) -> tuple[float, float]:
    dx = float(pos["x"]) - float(ref_pos["x"])
    dz = float(pos["z"]) - float(ref_pos["z"])
    return max(abs(dx), abs(dz)), min(abs(dx), abs(dz))


def primary_separation(pos: dict, ref_pos: dict) -> float:
    return axis_separation(pos, ref_pos)[0]


def revision_clarity(record: dict) -> float:
    """Clearness of the relative position in both old and new states.

    Higher = larger minimum primary-axis separation and smaller maximum
    secondary-axis separation across the two states.
    """
    ref = record["reference_pos"]
    primary_old, secondary_old = axis_separation(record["old_pos"], ref)
    primary_new, secondary_new = axis_separation(record["new_pos"], ref)
    return min(primary_old, primary_new) - max(secondary_old, secondary_new)


def distance_bucket(primary: float) -> tuple[float, float]:
    for lo, hi in DISTANCE_BUCKETS:
        if lo <= primary < hi:
            return (lo, hi)
    return DISTANCE_BUCKETS[-1]


def select_revision_samples(records: list[dict], n: int, seed: int, strict_flip: bool) -> list[dict]:
    pool = records
    if strict_flip:
        pool = [r for r in records if (r["old_label"], r["new_label"]) in STRICT_FLIPS]

    by_target: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in pool:
        by_target[(rec["scene_id"], rec["target_object_id"])].append(rec)

    chosen: list[dict] = []
    for target_records in by_target.values():
        best = max(target_records, key=lambda r: (revision_clarity(r), -len(r["sample_id"])))
        chosen.append(best)

    rng = random.Random(seed)
    rng.shuffle(chosen)
    if n and len(chosen) > n:
        chosen = chosen[:n]
    return chosen


def select_maintain_samples(records: list[dict], revision_samples: list[dict], n: int, seed: int) -> list[dict]:
    """Match maintain controls to the revision set by label and primary-distance buckets."""
    by_target: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in records:
        by_target[(rec["scene_id"], rec["target_object_id"])].append(rec)
    one_per_target = [min(v, key=lambda r: r["sample_id"]) for v in by_target.values()]

    target_dist: dict[tuple[str, tuple[float, float]], int] = Counter()
    for rec in revision_samples:
        bucket = distance_bucket(primary_separation(rec["new_pos"], rec["reference_pos"]))
        target_dist[(rec["new_label"], bucket)] += 1

    if n and n != len(revision_samples):
        total = sum(target_dist.values())
        if total:
            exact = {key: value * n / total for key, value in target_dist.items()}
            scaled = {key: int(value) for key, value in exact.items()}
            remaining = n - sum(scaled.values())
            for key in sorted(exact, key=lambda k: (exact[k] - scaled[k], target_dist[k]), reverse=True)[:remaining]:
                scaled[key] += 1
            target_dist = Counter(scaled)

    by_label_bucket: dict[tuple[str, tuple[float, float]], list[dict]] = defaultdict(list)
    for rec in one_per_target:
        key = (rec["label"], distance_bucket(primary_separation(rec["target_pos"], rec["reference_pos"])))
        by_label_bucket[key].append(rec)

    rng = random.Random(seed)
    selected: list[dict] = []
    for key, count in sorted(target_dist.items()):
        pool = by_label_bucket.get(key, [])
        rng.shuffle(pool)
        selected.extend(pool[:count])
    rng.shuffle(selected)
    return selected


def build_messages(record: dict, condition: str) -> list[dict]:
    target = record["target_name"]
    reference = record["reference_name"]
    if condition == "old_only":
        text = SINGLE_IMAGE_PROMPT.format(target=target, reference=reference)
        images = [record["old_image"]]
    elif condition == "new_only":
        text = SINGLE_IMAGE_PROMPT.format(target=target, reference=reference)
        images = [record["new_image"]]
    elif condition == "both_default":
        text = BOTH_DEFAULT_PROMPT.format(target=target, reference=reference)
        images = [record["old_image"], record["new_image"]]
    elif condition == "both_remind":
        text = BOTH_REMIND_PROMPT.format(target=target, reference=reference)
        images = [record["old_image"], record["new_image"]]
    else:
        raise ValueError(f"Unknown condition: {condition}")

    content: list[dict] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def expected_label(record: dict, condition: str) -> str:
    return record["old_label"] if condition == "old_only" else record["new_label"]


def parse_label(text: str, thinking: bool = False) -> str | None:
    """Extract the answer label from a raw generation.

    With thinking enabled the raw output contains a <think>...</think> block;
    only the text after the closing tag is considered, since the reasoning may
    itself mention the label words.
    """
    if thinking and "</think>" in (text or ""):
        text = text.split("</think>", 1)[1]
    match = LABEL_RE.search(text or "")
    if not match:
        return None
    return match.group(1).upper()


def candidate_pair(expected_label: str) -> tuple[str, str]:
    """Legal answers for a two-alternative forced-choice item.

    Horizontal relations are compared only against LEFT/RIGHT, vertical
    relations only against ABOVE/BELOW, using the expected label's axis.
    """
    if expected_label in ("LEFT", "RIGHT"):
        return ("LEFT", "RIGHT")
    return ("ABOVE", "BELOW")


def load_model_and_processor(model_dir: Path):
    """Load the local Qwen3.5 VL model and processor (shared by screening scripts)."""
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(model_dir)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")
    return model, processor


def score_candidates(
    model,
    processor,
    messages: list[dict],
    candidates: tuple[str, ...],
    device: str,
    processor_kwargs: dict | None = None,
) -> dict[str, float]:
    """Score each candidate by its continuation log-probability given the identical prompt + images.

    The prompt is rendered once with ``add_generation_prompt=True``; each candidate
    token is appended to that exact prefix, so only the candidate word differs.
    Forced-choice scoring always renders the prompt with an empty think block
    (``enable_thinking=False``): the candidate is the direct answer token, so its
    probability is well defined regardless of the ``--thinking`` setting used for
    free generation.

    Fairness notes:
    - Every candidate is scored with the same leading space (``" " + candidate``),
      matching the tokenizer's space-prefixed generation tokens, and only the
      candidate's own tokens contribute to its score (no prompt tokens are scored).
    - The raw score is the sum of log-probs over the candidate's tokens, which would
      penalise longer candidates purely by length. We therefore divide by the
      candidate token count (per-token mean log-prob). For equal-length candidates
      this is equivalent to the sum (same argmax); for unequal lengths it removes the
      systematic length bias.
    """
    base = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs=processor_kwargs,
    )
    prompt_len = base["input_ids"].shape[1]
    prompt_ids = base["input_ids"][0]
    mm_type_ids = base["mm_token_type_ids"][0]
    vision_inputs = {
        k: v
        for k, v in base.items()
        if k not in ("input_ids", "attention_mask", "mm_token_type_ids")
    }

    scores: dict[str, float] = {}
    with torch.inference_mode():
        for candidate in candidates:
            cand_ids = torch.tensor(
                processor.tokenizer.encode(" " + candidate, add_special_tokens=False),
                dtype=torch.long,
            )
            input_ids = torch.cat([prompt_ids, cand_ids]).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            mm_extended = torch.cat([mm_type_ids, torch.zeros(len(cand_ids), dtype=mm_type_ids.dtype)]).unsqueeze(0)
            full = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "mm_token_type_ids": mm_extended,
                **vision_inputs,
            }
            full = {k: v.to(device) if hasattr(v, "to") else v for k, v in full.items()}
            # Only compute logits for the positions needed to score the candidate
            # tokens. logits_to_keep=N returns the last N logits; the candidate
            # token at sequence position prompt_len+j needs logits[prompt_len+j-1],
            # i.e. rows [prompt_len-1, prompt_len+len(cand_ids)-1], so we request
            # len(cand_ids)+1 rows and use the first len(cand_ids) of them. This
            # avoids materialising the full (T, vocab) logits tensor, which is
            # prohibitive for long video inputs.
            outputs = model(**full, return_dict=True, logits_to_keep=len(cand_ids) + 1)
            log_probs = F.log_softmax(outputs.logits[0].float(), dim=-1)
            seq_log_prob = 0.0
            for j, token_id in enumerate(cand_ids.tolist()):
                seq_log_prob += log_probs[j, token_id].item()
            scores[candidate] = seq_log_prob / len(cand_ids)
    return scores


def run_inference(
    model,
    processor,
    messages: list[dict],
    device: str,
    max_new_tokens: int,
    thinking: bool,
    processor_kwargs: dict | None = None,
) -> str:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=thinking,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs=processor_kwargs,
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    generated = outputs[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(generated, skip_special_tokens=not thinking)[0].strip()


def summarize(results: list[dict]) -> dict:
    revision = [r for r in results if r["kind"] == "revision"]
    maintain = [r for r in results if r["kind"] == "maintain"]
    summary: dict = {
        "n_revision_samples": len({r["sample_id"] for r in revision}),
        "n_maintain_samples": len({r["sample_id"] for r in maintain}),
    }

    def condition_stats(records: list[dict]) -> dict:
        stats: dict = {}
        for condition in CONDITIONS:
            rows = [r for r in records if r["condition"] == condition]
            if not rows:
                continue
            correct = sum(r["correct"] for r in rows)
            unparsed = sum(r["parsed_label"] is None for r in rows)
            stats[condition] = {
                "n": len(rows),
                "accuracy": correct / len(rows),
                "unparsed": unparsed / len(rows),
            }
        return stats

    summary["revision_conditions"] = condition_stats(revision)
    summary["maintain_conditions"] = condition_stats(maintain)

    total_rows = len(results)
    unparsed_rows = sum(1 for r in results if r["parsed_label"] is None)
    summary["parsing_failure_rate"] = (unparsed_rows / total_rows) if total_rows else None

    maintain_correct = sum(r["correct"] for r in maintain)
    summary["maintain_overall_accuracy"] = (maintain_correct / len(maintain)) if maintain else None

    by_sample: dict[str, dict] = defaultdict(dict)
    for row in revision:
        by_sample[row["sample_id"]][row["condition"]] = row

    def answered(rows: dict, condition: str, label_key: str) -> bool:
        row = rows.get(condition)
        return bool(row and row["parsed_label"] == row[label_key])

    def sample_fields(rows: dict) -> dict:
        return next(iter(rows.values()))

    perception_clean = [
        s
        for s, rows in by_sample.items()
        if answered(rows, "old_only", "old_label") and answered(rows, "new_only", "new_label")
    ]
    both_default_clean = [
        s
        for s in perception_clean
        if answered(by_sample[s], "both_default", "new_label")
    ]
    strong_inertia = [
        s
        for s in perception_clean
        if sample_fields(by_sample[s])["old_label"] != sample_fields(by_sample[s])["new_label"]
        and answered(by_sample[s], "both_default", "old_label")
    ]
    strong_inertia_recovered = [
        s
        for s in strong_inertia
        if answered(by_sample[s], "both_remind", "new_label")
    ]

    summary["perception_clean"] = {
        "n": len(perception_clean),
        "both_default_accuracy": (len(both_default_clean) / len(perception_clean)) if perception_clean else None,
    }
    summary["strong_inertia"] = {
        "n": len(strong_inertia),
        "rate_of_perception_clean": (len(strong_inertia) / len(perception_clean)) if perception_clean else None,
        "both_remind_recovery_n": len(strong_inertia_recovered),
        "both_remind_recovery_rate": (len(strong_inertia_recovered) / len(strong_inertia)) if strong_inertia else None,
        "sample_ids": sorted(strong_inertia),
    }
    return summary


def build_mode_comparison(comp_rows: list[dict], summaries: dict[str, dict]) -> dict:
    overall: Counter[str] = Counter()
    per_key: dict[tuple[str, str], dict] = {}
    for row in comp_rows:
        key = (row["kind"], row["condition"])
        entry = per_key.setdefault(key, {"n": 0, "free_correct": 0, "two_correct": 0, "agree": 0})
        entry["n"] += 1
        entry["free_correct"] += int(row["free_correct"])
        entry["two_correct"] += int(row["two_choice_correct"])
        entry["agree"] += int(row["free_label"] == row["two_choice_label"])
        if row["free_correct"] and row["two_choice_correct"]:
            overall["both_correct"] += 1
        elif not row["free_correct"] and not row["two_choice_correct"]:
            overall["both_wrong"] += 1
        elif row["free_correct"]:
            overall["free_only_correct"] += 1
        else:
            overall["two_choice_only_correct"] += 1
        if not row["free_correct"] and row["two_choice_correct"]:
            overall["two_choice_fixes_free"] += 1
        if row["free_correct"] and not row["two_choice_correct"]:
            overall["two_choice_breaks_free"] += 1

    per = {
        f"{kind}/{condition}": {
            **entry,
            "free_accuracy": entry["free_correct"] / entry["n"],
            "two_choice_accuracy": entry["two_correct"] / entry["n"],
            "agreement_rate": entry["agree"] / entry["n"],
        }
        for (kind, condition), entry in sorted(per_key.items())
    }
    return {
        "n_rows": len(comp_rows),
        "overall": dict(overall),
        "per_kind_condition": per,
        "predicted_marginal_free": dict(Counter(r["free_label"] for r in comp_rows)),
        "predicted_marginal_two_choice": dict(Counter(r["two_choice_label"] for r in comp_rows)),
        "revision_metrics": {
            "perception_clean": {
                "free": summaries["free"]["perception_clean"],
                "two_choice": summaries["two_choice"]["perception_clean"],
            },
            "strong_inertia": {
                "free": summaries["free"]["strong_inertia"],
                "two_choice": summaries["two_choice"]["strong_inertia"],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavior screening of state revision/inertia on Qwen3.5-9B "
                    "(free generation and/or two-alternative forced-choice probability scoring)."
    )
    parser.add_argument("--revision-jsonl", type=Path, default=REVISION_PATH)
    parser.add_argument("--maintain-jsonl", type=Path, default=MAINTAIN_PATH)
    parser.add_argument("--revision-mode", choices=["strict_flip", "all"], default="strict_flip",
                        help="strict_flip keeps only LEFT<->RIGHT / ABOVE<->BELOW; all keeps every label transition.")
    parser.add_argument("--n-revision", type=int, default=0,
                        help="Max revision samples per changed target (0 = keep all eligible).")
    parser.add_argument("--n-maintain", type=int, default=0,
                        help="Target maintain count (0 = match the selected revision count).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-think-tokens", type=int, default=16384,
                        help="Token budget for free generation when --thinking on (reasoning can be long).")
    parser.add_argument("--thinking", choices=["on", "off"], default="off",
                        help="Whether Qwen3.5 may emit a <think>...</think> reasoning block "
                             "during free generation. Two-choice scoring always uses an empty "
                             "think block so the answer probability is well defined.")
    parser.add_argument("--scoring-mode", choices=["two_choice", "free", "both"], default="two_choice",
                        help="two_choice: compare log-probabilities of the two legal answers "
                             "(default diagnostic mode); free: original free generation; "
                             "both: run both and write a comparison.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Build and print the prompt plan without loading the model.")
    args = parser.parse_args()

    revision = select_revision_samples(
        load_jsonl(args.revision_jsonl),
        args.n_revision,
        args.seed,
        strict_flip=(args.revision_mode == "strict_flip"),
    )
    maintain = select_maintain_samples(
        load_jsonl(args.maintain_jsonl),
        revision,
        args.n_maintain,
        args.seed + 1,
    )
    print(f"Selected {len(revision)} revision and {len(maintain)} maintain samples.")
    if args.n_maintain and len(maintain) < args.n_maintain:
        print(f"Note: only {len(maintain)} maintain samples available for the requested {args.n_maintain} "
              "(label/distance buckets underfilled).")
    elif not args.n_maintain and len(maintain) < len(revision):
        print(f"Note: maintain sample pool could not fully mirror the revision distribution "
              f"({len(maintain)} < {len(revision)}).")
    print("Revision transitions:", dict(sorted(Counter(r["transition"] for r in revision).items())))
    print("Maintain labels:", dict(sorted(Counter(r["label"] for r in maintain).items())))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.out_dir / "selected_samples.json"
    selected_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "revision": [r["sample_id"] for r in revision],
                "maintain": [r["sample_id"] for r in maintain],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote selected sample ids: {selected_path}")

    plan: list[dict] = []
    for record in revision:
        for condition in args.conditions:
            plan.append({
                "kind": "revision",
                "sample_id": record["sample_id"],
                "condition": condition,
                "record": record,
                "expected": expected_label(record, condition),
            })
    for record in maintain:
        for condition in args.conditions:
            plan.append({
                "kind": "maintain",
                "sample_id": record["sample_id"],
                "condition": condition,
                "record": record,
                "expected": expected_label(record, condition),
            })

    if args.dry_run:
        for item in plan[:3] + plan[-3:]:
            messages = build_messages(item["record"], item["condition"])
            print("\n---", item["kind"], item["sample_id"], item["condition"], "expected:", item["expected"])
            if args.scoring_mode in ("two_choice", "both"):
                print("    candidates:", candidate_pair(item["expected"]))
            for message in messages:
                role = message["role"]
                content = message["content"]
                if isinstance(content, str):
                    print(f"[{role}] {content}")
                else:
                    kinds = [c.get("type") for c in content]
                    text = next((c.get("text") for c in content if c.get("type") == "text"), "")
                    print(f"[{role}] content_types={kinds}")
                    print(f"[{role}] text={text}")
        print(f"\nDry run OK: {len(plan)} prompt rows would be generated.")
        return

    model, processor = load_model_and_processor(args.model_dir)

    modes = []
    if args.scoring_mode in ("two_choice", "both"):
        modes.append("two_choice")
    if args.scoring_mode in ("free", "both"):
        modes.append("free")

    results_paths: dict[str, Path] = {}
    streams: dict[str, object] = {}
    for mode in modes:
        path = args.out_dir / f"results_{mode}.jsonl"
        results_paths[mode] = path
        streams[mode] = open(path, "w", encoding="utf-8")

    try:
        for idx, item in enumerate(plan, start=1):
            record = item["record"]
            messages = build_messages(record, item["condition"])
            for mode in modes:
                if mode == "two_choice":
                    candidates = candidate_pair(item["expected"])
                    scores = score_candidates(model, processor, messages, candidates, args.device)
                    choice = max(candidates, key=lambda c: scores[c])
                    row = {
                        "scoring": "two_choice",
                        "kind": item["kind"],
                        "sample_id": item["sample_id"],
                        "scene_id": record["scene_id"],
                        "condition": item["condition"],
                        "target_name": record["target_name"],
                        "reference_name": record["reference_name"],
                        "old_image": record["old_image"],
                        "new_image": record["new_image"],
                        "old_label": record.get("old_label"),
                        "new_label": record.get("new_label"),
                        "expected_label": item["expected"],
                        "candidate_labels": list(candidates),
                        "candidate_scores": scores,
                        "score_gap": abs(scores[candidates[0]] - scores[candidates[1]]),
                        "parsed_label": choice,
                        "raw_output": None,
                        "correct": choice == item["expected"],
                    }
                else:
                    thinking = args.thinking == "on"
                    max_tokens = args.max_think_tokens if thinking else args.max_new_tokens
                    raw = run_inference(model, processor, messages, args.device, max_tokens, thinking)
                    parsed = parse_label(raw, thinking)
                    row = {
                        "scoring": "free",
                        "kind": item["kind"],
                        "sample_id": item["sample_id"],
                        "scene_id": record["scene_id"],
                        "condition": item["condition"],
                        "target_name": record["target_name"],
                        "reference_name": record["reference_name"],
                        "old_image": record["old_image"],
                        "new_image": record["new_image"],
                        "old_label": record.get("old_label"),
                        "new_label": record.get("new_label"),
                        "expected_label": item["expected"],
                        "raw_output": raw,
                        "parsed_label": parsed,
                        "correct": parsed == item["expected"],
                    }
                streams[mode].write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  [{idx}/{len(plan)}] {mode} {row['kind']} {row['sample_id']} {row['condition']}: {row['parsed_label']}")
    finally:
        for stream in streams.values():
            stream.close()

    summaries: dict[str, dict] = {}
    for mode in modes:
        rows = load_jsonl(results_paths[mode])
        summary = summarize(rows)
        summaries[mode] = summary
        (args.out_dir / f"summary_{mode}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if len(modes) == 1:
            (args.out_dir / "results.jsonl").write_text(
                results_paths[mode].read_text(encoding="utf-8"), encoding="utf-8"
            )
            (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for mode, summary in summaries.items():
        print(f"\n== {mode} summary ==")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if "two_choice" in summaries and "free" in summaries:
        free_by_key = {
            (r["sample_id"], r["condition"]): r for r in load_jsonl(results_paths["free"])
        }
        comp_rows = []
        for row in load_jsonl(results_paths["two_choice"]):
            free_row = free_by_key[(row["sample_id"], row["condition"])]
            comp_rows.append({
                "kind": row["kind"],
                "sample_id": row["sample_id"],
                "condition": row["condition"],
                "expected_label": row["expected_label"],
                "free_label": free_row["parsed_label"],
                "two_choice_label": row["parsed_label"],
                "free_correct": free_row["correct"],
                "two_choice_correct": row["correct"],
                "candidate_scores": row.get("candidate_scores"),
                "score_gap": row.get("score_gap"),
            })
        (args.out_dir / "mode_comparison_rows.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in comp_rows) + "\n",
            encoding="utf-8",
        )
        comparison = build_mode_comparison(comp_rows, summaries)
        (args.out_dir / "mode_comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\n== free vs two_choice comparison ==")
        print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
