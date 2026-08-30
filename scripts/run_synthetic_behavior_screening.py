from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from _theory_of_space_utils import DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR
from run_behavior_screening import (
    CONDITIONS,
    candidate_pair,
    load_model_and_processor,
    parse_label,
    run_inference,
    score_candidates,
    summarize,
)


REVISION_PATH = Path("data/synthetic_state/synthetic_revision.jsonl")
MAINTAIN_PATH = Path("data/synthetic_state/synthetic_maintain.jsonl")
OUT_DIR = DEFAULT_OUTPUT_DIR / "synthetic_behavior"

SYSTEM_PROMPT = (
    "You are a spatial reasoning assistant. You always answer with exactly one word, "
    "either LEFT or RIGHT. No explanations."
)

SINGLE_IMAGE_PROMPT = (
    "The image below contains a red circle and a blue square. "
    "Where is the red circle relative to the blue square? "
    "Answer with exactly one word: LEFT or RIGHT."
)

BOTH_DEFAULT_PROMPT = (
    "The first image shows an earlier state, and the second image shows the current state. "
    "Where is the red circle relative to the blue square in the CURRENT state? "
    "Answer with exactly one word: LEFT or RIGHT."
)

BOTH_REMIND_PROMPT = (
    "The first image shows an earlier state, and the second image shows the current state. "
    "IMPORTANT: the second image is the authoritative visual evidence of the current state; "
    "the first image only represents the past state and should be ignored. "
    "Where is the red circle relative to the blue square in the current state? "
    "Answer with exactly one word: LEFT or RIGHT."
)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_messages(record: dict, condition: str) -> list[dict]:
    if condition == "old_only":
        text = SINGLE_IMAGE_PROMPT
        images = [record["old_image"]]
    elif condition == "new_only":
        text = SINGLE_IMAGE_PROMPT
        images = [record["new_image"]]
    elif condition == "both_default":
        text = BOTH_DEFAULT_PROMPT
        images = [record["old_image"], record["new_image"]]
    elif condition == "both_remind":
        text = BOTH_REMIND_PROMPT
        images = [record["old_image"], record["new_image"]]
    else:
        raise ValueError(f"Unknown condition: {condition}")
    content: list[dict] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def expected_relation(record: dict, condition: str) -> str:
    return record["old_relation"] if condition == "old_only" else record["new_relation"]


def correct_answer_margin(record: dict, condition: str, scores: dict[str, float]) -> float:
    expected = expected_relation(record, condition)
    wrong = "RIGHT" if expected == "LEFT" else "LEFT"
    return scores[expected] - scores[wrong]


def make_two_choice_row(record: dict, condition: str, scores: dict[str, float]) -> dict:
    expected = expected_relation(record, condition)
    choice = max(scores, key=scores.get)
    margin = correct_answer_margin(record, condition, scores)
    return {
        "scoring": "two_choice",
        "sample_id": record["sample_id"],
        "pair_id": record["pair_id"],
        "kind": record["change_type"],
        "change_type": record["change_type"],
        "condition": condition,
        "target": record["target"],
        "anchor": record["anchor"],
        "old_image": record["old_image"],
        "new_image": record["new_image"],
        "old_relation": record["old_relation"],
        "new_relation": record["new_relation"],
        "old_label": record["old_relation"],
        "new_label": record["new_relation"],
        "expected_relation": expected,
        "expected_label": expected,
        "LEFT_score": scores["LEFT"],
        "RIGHT_score": scores["RIGHT"],
        "candidate_scores": scores,
        "score_gap": abs(scores["LEFT"] - scores["RIGHT"]),
        "correct_answer_margin": margin,
        "predicted_relation": choice,
        "parsed_label": choice,
        "raw_output": None,
        "correct": choice == expected,
    }


def make_free_row(record: dict, condition: str, raw: str, thinking: bool) -> dict:
    expected = expected_relation(record, condition)
    parsed = parse_label(raw, thinking)
    return {
        "scoring": "free",
        "sample_id": record["sample_id"],
        "pair_id": record["pair_id"],
        "kind": record["change_type"],
        "change_type": record["change_type"],
        "condition": condition,
        "target": record["target"],
        "anchor": record["anchor"],
        "old_image": record["old_image"],
        "new_image": record["new_image"],
        "old_relation": record["old_relation"],
        "new_relation": record["new_relation"],
        "old_label": record["old_relation"],
        "new_label": record["new_relation"],
        "expected_relation": expected,
        "expected_label": expected,
        "predicted_relation": parsed,
        "parsed_label": parsed,
        "raw_output": raw,
        "correct": parsed == expected,
    }


def paired_analysis(rows: list[dict]) -> dict:
    """Pair revision/maintain rows that share the same pair_id and new_image."""
    rev_by_pair = {
        row["pair_id"]: row
        for row in rows
        if row["kind"] == "revision" and row["condition"] == "both_default" and row.get("correct_answer_margin") is not None
    }
    mnt_by_pair = {
        row["pair_id"]: row
        for row in rows
        if row["kind"] == "maintain" and row["condition"] == "both_default" and row.get("correct_answer_margin") is not None
    }
    pair_ids = sorted(set(rev_by_pair) & set(mnt_by_pair))

    per_pair = []
    for pair_id in pair_ids:
        rev_row = rev_by_pair[pair_id]
        mnt_row = mnt_by_pair[pair_id]
        revision_margin = rev_row["correct_answer_margin"]
        maintain_margin = mnt_row["correct_answer_margin"]
        per_pair.append({
            "pair_id": pair_id,
            "new_relation": rev_row["new_relation"],
            "maintain_margin": maintain_margin,
            "revision_margin": revision_margin,
            "margin_delta": revision_margin - maintain_margin,
            "maintain_predicted": mnt_row["predicted_relation"],
            "revision_predicted": rev_row["predicted_relation"],
            "maintain_correct": mnt_row["correct"],
            "revision_correct": rev_row["correct"],
            "prediction_differs": mnt_row["predicted_relation"] != rev_row["predicted_relation"],
        })

    def mean(values: list[float]) -> float | None:
        return statistics.mean(values) if values else None

    maintain_margins = [p["maintain_margin"] for p in per_pair]
    revision_margins = [p["revision_margin"] for p in per_pair]
    deltas = [p["margin_delta"] for p in per_pair]
    negative_deltas = [d for d in deltas if d < 0]
    prediction_differs = [p for p in per_pair if p["prediction_differs"]]

    return {
        "n_pairs": len(per_pair),
        "mean_maintain_margin": mean(maintain_margins),
        "mean_revision_margin": mean(revision_margins),
        "mean_margin_delta": mean(deltas),
        "margin_delta_negative_rate": (len(negative_deltas) / len(deltas)) if deltas else None,
        "n_margin_delta_negative": len(negative_deltas),
        "n_prediction_differs": len(prediction_differs),
        "prediction_differs_rate": (len(prediction_differs) / len(per_pair)) if per_pair else None,
        "maintain_both_default_accuracy": (
            sum(1 for p in per_pair if p["maintain_correct"]) / len(per_pair)
        ) if per_pair else None,
        "revision_both_default_accuracy": (
            sum(1 for p in per_pair if p["revision_correct"]) / len(per_pair)
        ) if per_pair else None,
        "per_pair": per_pair,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic state-revision behavior screening (two-alternative forced choice)."
    )
    parser.add_argument("--revision-jsonl", type=Path, default=REVISION_PATH)
    parser.add_argument("--maintain-jsonl", type=Path, default=MAINTAIN_PATH)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--scoring-mode", choices=["two_choice", "free", "both"], default="two_choice")
    parser.add_argument("--thinking", choices=["on", "off"], default="off")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-think-tokens", type=int, default=1024)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    revision = load_jsonl(args.revision_jsonl)
    maintain = load_jsonl(args.maintain_jsonl)
    all_records = revision + maintain
    print(f"Loaded {len(revision)} revision + {len(maintain)} maintain samples.")

    plan = [
        {"record": record, "condition": condition}
        for record in all_records
        for condition in args.conditions
    ]

    if args.dry_run:
        for item in plan[:2] + plan[-2:]:
            messages = build_messages(item["record"], item["condition"])
            print("\n---", item["record"]["sample_id"], item["condition"],
                  "expected:", expected_relation(item["record"], item["condition"]))
            if args.scoring_mode in ("two_choice", "both"):
                print("    candidates:", candidate_pair(expected_relation(item["record"], item["condition"])))
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
    thinking = args.thinking == "on"

    modes = []
    if args.scoring_mode in ("two_choice", "both"):
        modes.append("two_choice")
    if args.scoring_mode in ("free", "both"):
        modes.append("free")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_paths: dict[str, Path] = {}
    streams: dict[str, object] = {}
    for mode in modes:
        path = args.out_dir / f"results_{mode}.jsonl"
        results_paths[mode] = path
        streams[mode] = open(path, "w", encoding="utf-8")

    try:
        for idx, item in enumerate(plan, start=1):
            record = item["record"]
            condition = item["condition"]
            messages = build_messages(record, condition)
            for mode in modes:
                if mode == "two_choice":
                    scores = score_candidates(
                        model, processor, messages, candidate_pair(expected_relation(record, condition)), args.device
                    )
                    row = make_two_choice_row(record, condition, scores)
                else:
                    max_tokens = args.max_think_tokens if thinking else args.max_new_tokens
                    raw = run_inference(model, processor, messages, args.device, max_tokens, thinking)
                    row = make_free_row(record, condition, raw, thinking)
                streams[mode].write(json.dumps(row, ensure_ascii=False) + "\n")
                if idx % 50 == 0 or idx == len(plan):
                    print(f"  [{idx}/{len(plan)}] {mode} {row['sample_id']} {row['condition']}: {row['predicted_relation']}")
    finally:
        for stream in streams.values():
            stream.close()

    summaries: dict[str, dict] = {}
    for mode in modes:
        rows = load_jsonl(results_paths[mode])
        summary = summarize(rows)
        summaries[mode] = summary
        (args.out_dir / f"summary_{mode}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if len(modes) == 1:
            (args.out_dir / "results.jsonl").write_text(
                results_paths[mode].read_text(encoding="utf-8"), encoding="utf-8"
            )
            (args.out_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    two_choice_rows = load_jsonl(results_paths["two_choice"]) if "two_choice" in results_paths else []
    if two_choice_rows:
        analysis = paired_analysis(two_choice_rows)
        (args.out_dir / "paired_analysis.json").write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\n== paired analysis ==")
        compact = {k: v for k, v in analysis.items() if k != "per_pair"}
        print(json.dumps(compact, ensure_ascii=False, indent=2))

    for mode, summary in summaries.items():
        print(f"\n== {mode} summary ==")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
