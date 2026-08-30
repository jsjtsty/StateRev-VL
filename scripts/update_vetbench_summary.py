from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_vetbench_screening import OUT_DIR, build_summary


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild VET-Bench summary.json from result JSONL files.")
    parser.add_argument("--results-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--single-swap", type=Path)
    parser.add_argument("--tracking", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    single_swap_path = args.single_swap or args.results_dir / "results_single_swap_tracking.jsonl"
    tracking_path = args.tracking or args.results_dir / "results_tracking.jsonl"
    output_path = args.output or args.results_dir / "summary.json"

    summary = build_summary(load_jsonl(single_swap_path), load_jsonl(tracking_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {output_path}")


if __name__ == "__main__":
    main()
