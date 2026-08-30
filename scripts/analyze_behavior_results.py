from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from _theory_of_space_utils import DEFAULT_OUTPUT_DIR, locate_theory_of_space_source, load_run_json


RESULTS_PATH = DEFAULT_OUTPUT_DIR / "behavior_screening" / "results.jsonl"
DIAG_DIR = DEFAULT_OUTPUT_DIR / "behavior_screening" / "diagnostics"
REVISION_INDEX = Path("data/processed/theory_of_space_relative_revision.jsonl")
MAINTAIN_INDEX = Path("data/processed/theory_of_space_relative_maintain.jsonl")
LABELS = ("LEFT", "RIGHT", "ABOVE", "BELOW")
SINGLE_IMAGE_CONDITIONS = ("old_only", "new_only")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def error_type(expected: str, predicted: str) -> str:
    if predicted == expected:
        return "correct"
    pair = {expected, predicted}
    if pair in ({"LEFT", "RIGHT"}, {"ABOVE", "BELOW"}):
        return "same_axis_flip"
    if expected in ("LEFT", "RIGHT") and predicted in ("ABOVE", "BELOW"):
        return "horizontal_to_vertical"
    if expected in ("ABOVE", "BELOW") and predicted in ("LEFT", "RIGHT"):
        return "vertical_to_horizontal"
    return "other"


def confusion_matrix(rows: list[dict]) -> dict:
    matrix: dict[str, dict[str, int]] = {}
    for label in LABELS:
        row = Counter(r["parsed_label"] or "UNPARSED" for r in rows if r["expected_label"] == label)
        matrix[label] = {k: row.get(k, 0) for k in (*LABELS, "UNPARSED")}
    return matrix


def print_confusion(title: str, rows: list[dict]) -> None:
    matrix = confusion_matrix(rows)
    correct = sum(r["correct"] for r in rows)
    print(f"\n== {title}: n={len(rows)} acc={correct / len(rows):.3f}")
    header = "expected \\ predicted | " + " | ".join(f"{k:>7s}" for k in (*LABELS, "UNPARSED"))
    print(header)
    print("-" * len(header))
    for expected in LABELS:
        values = [matrix[expected][k] for k in (*LABELS, "UNPARSED")]
        print(f"{expected:>18s} | " + " | ".join(f"{v:>7d}" for v in values))
    return matrix


def per_relation_stats(rows: list[dict]) -> dict:
    stats: dict[str, dict] = {}
    for expected in LABELS:
        subset = [r for r in rows if r["expected_label"] == expected]
        if not subset:
            continue
        correct = sum(r["correct"] for r in subset)
        types = Counter(error_type(r["expected_label"], r["parsed_label"]) for r in subset)
        stats[expected] = {"n": len(subset), "accuracy": correct / len(subset), "error_types": dict(types)}
    return stats


def per_sample_stats(rows: list[dict]) -> dict:
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sample[row["sample_id"]].append(row)

    single_error_counts: Counter[int] = Counter()
    both_single_wrong: list[str] = []
    any_condition_wrong: list[str] = []
    for sample_id, sample_rows in by_sample.items():
        single_rows = [r for r in sample_rows if r["condition"] in SINGLE_IMAGE_CONDITIONS]
        single_wrong = sum(not r["correct"] for r in single_rows)
        single_error_counts[single_wrong] += 1
        if single_wrong == len(single_rows):
            both_single_wrong.append(sample_id)
        if any(not r["correct"] for r in sample_rows):
            any_condition_wrong.append(sample_id)

    return {
        "n_samples": len(by_sample),
        "single_image_wrong_count_distribution": dict(sorted(single_error_counts.items())),
        "n_samples_wrong_in_both_single_image_conditions": len(both_single_wrong),
        "samples_wrong_in_both_single_image_conditions": sorted(both_single_wrong),
        "n_samples_with_any_error": len(any_condition_wrong),
    }


def per_scene_stats(rows: list[dict]) -> dict:
    single = [r for r in rows if r["condition"] in SINGLE_IMAGE_CONDITIONS]
    by_scene: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in single:
        grouped[row["scene_id"]].append(row)
    for scene_id, scene_rows in sorted(grouped.items()):
        correct = sum(r["correct"] for r in scene_rows)
        by_scene[scene_id] = {
            "n_single_image_rows": len(scene_rows),
            "correct": correct,
            "errors": len(scene_rows) - correct,
        }
    return by_scene


def world_to_pixel_map(run_name: str) -> dict[tuple[int, int], tuple[float, float]]:
    layout = locate_theory_of_space_source()
    meta = load_run_json(layout, run_name, "meta_data.json")
    mapping: dict[tuple[int, int], tuple[float, float]] = {}
    for item in meta.get("topdown_map", {}).get("mapping", []):
        world = item.get("world", {})
        pixel = item.get("pixel", {})
        if "x" in world and "z" in world and "x" in pixel and "y" in pixel:
            mapping[(int(world["x"]), int(world["z"]))] = (float(pixel["x"]), float(pixel["y"]))
    return mapping


def load_font(size: int) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_marker(draw: ImageDraw.ImageDraw, center: tuple[float, float], color: str, label: str, radius: int, font: ImageFont.ImageFont) -> None:
    cx, cy = int(center[0]), int(center[1])
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=5)
    draw.line((cx - radius - 8, cy, cx + radius + 8, cy), fill=color, width=4)
    draw.line((cx, cy - radius - 8, cx, cy + radius + 8), fill=color, width=4)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = cx - radius - text_w - 10
    text_y = cy - text_h // 2
    draw.rectangle((text_x - 4, text_y - 4, text_x + text_w + 4, text_y + text_h + 4), fill=(255, 255, 255, 0))
    draw.text((text_x, text_y), label, fill=color, font=font)


def annotate_panel(image: Image.Image, pixel_map: dict, target_pos: dict, ref_pos: dict, target_name: str,
                   ref_name: str, gt_label: str, seen: bool, predicted: str | None, panel_tag: str) -> Image.Image:
    scale = 1.5
    img = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(img)
    font_label = load_font(20)
    font_gt = load_font(26)
    font_badge = load_font(22)
    s = scale

    target_px = pixel_map.get((int(target_pos["x"]), int(target_pos["z"])))
    ref_px = pixel_map.get((int(ref_pos["x"]), int(ref_pos["z"])))

    if ref_px:
        draw_marker(draw, (ref_px[0] * s, ref_px[1] * s), "#1565C0", f"anchor {ref_name}", int(14 * s), font_label)
    if target_px:
        draw_marker(draw, (target_px[0] * s, target_px[1] * s), "#C62828", f"target {target_name}", int(14 * s), font_label)

    gt_text = f"GT {panel_tag}: {gt_label}  ({target_name} {gt_label.lower()} {ref_name})"
    draw.text((12, 10), gt_text, fill=(0, 0, 0), font=font_gt)
    draw.text((14, 12), gt_text, fill=(255, 255, 255), font=font_gt)

    if seen:
        draw.rectangle((0, 0, img.width - 1, img.height - 1), outline="#FB8C00", width=12)
        badge = f"SEEN | model: {predicted}"
        draw.rectangle((img.width - 420, 10, img.width - 10, 56), fill=(0, 0, 0))
        draw.text((img.width - 406, 16), badge, fill=(255, 255, 255), font=font_badge)
    return img


def make_diagnostic_figure(row: dict, index_record: dict, pixel_map: dict, out_path: Path) -> None:
    old_img = Image.open(row["old_image"])
    new_img = Image.open(row["new_image"])
    seen_condition = row["condition"]
    target_name = row["target_name"]
    ref_name = row["reference_name"]
    predicted = row["parsed_label"]
    ref_pos = index_record["reference_pos"]

    if index_record.get("change_type") == "no_change" or "target_pos" in index_record:
        gt = index_record["label"]
        pos = index_record["target_pos"]
        panels = [(old_img, "old", gt, pos), (new_img, "new", gt, pos)]
    else:
        panels = [
            (old_img, "old", index_record["old_label"], index_record["old_pos"]),
            (new_img, "new", index_record["new_label"], index_record["new_pos"]),
        ]

    rendered = []
    for image, panel_tag, gt_label, target_pos in panels:
        rendered.append(
            annotate_panel(
                image,
                pixel_map,
                target_pos,
                ref_pos,
                target_name,
                ref_name,
                gt_label,
                seen=(seen_condition == "old_only" and panel_tag == "old") or (seen_condition == "new_only" and panel_tag == "new"),
                predicted=predicted if ((seen_condition == "old_only") == (panel_tag == "old")) else None,
                panel_tag=panel_tag,
            )
        )
    left, right = rendered
    gap = 16
    header_h = 70
    canvas = Image.new("RGB", (left.width + right.width + gap * 3, header_h + max(left.height, right.height) + gap * 2), "white")
    draw = ImageDraw.Draw(canvas)
    font_header = load_font(24)
    header = (
        f"{row['sample_id']} | {row['kind']} | {seen_condition} | "
        f"expected={row['expected_label']} predicted={predicted} correct={row['correct']}"
    )
    draw.text((gap, 16), header, fill=(0, 0, 0), font=font_header)
    legend = "markers: blue=anchor (reference), red=target; orange border = image the model saw; GT text on each panel"
    draw.text((gap, 48), legend, fill=(90, 90, 90), font=load_font(18))
    canvas.paste(left, (gap, header_h))
    canvas.paste(right, (gap * 2 + left.width, header_h))
    canvas.save(out_path)


def load_index_map() -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in (REVISION_INDEX, MAINTAIN_INDEX):
        for record in load_jsonl(path):
            records[record["sample_id"]] = record
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze behavior screening results and export diagnostic figures.")
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--n-diag", type=int, default=20, help="Number of random single-image error figures to export.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", type=Path, default=DIAG_DIR)
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    index_map = load_index_map()
    report: dict = {}

    for kind in ("revision", "maintain"):
        for condition in SINGLE_IMAGE_CONDITIONS:
            subset = [r for r in rows if r["kind"] == kind and r["condition"] == condition]
            matrix = print_confusion(f"{kind} {condition}", subset)
            report[f"{kind}_{condition}_confusion"] = matrix
            report[f"{kind}_{condition}_per_relation"] = per_relation_stats(subset)
        kind_rows = [r for r in rows if r["kind"] == kind and r["condition"] in SINGLE_IMAGE_CONDITIONS]
        report[f"{kind}_per_sample"] = per_sample_stats([r for r in rows if r["kind"] == kind])
        report[f"{kind}_per_scene_single_image"] = per_scene_stats(kind_rows)

        # flip-oriented error summary for single-image conditions
        errors = [r for r in kind_rows if not r["correct"]]
        type_counts = Counter(error_type(r["expected_label"], r["parsed_label"]) for r in errors)
        predicted_marginal = Counter(r["parsed_label"] for r in kind_rows)
        report[f"{kind}_single_image_error_types"] = dict(type_counts)
        report[f"{kind}_single_image_predicted_marginal"] = dict(predicted_marginal)
        print(f"\n== {kind} single-image (old_only+new_only): n={len(kind_rows)} errors={len(errors)}")
        print("   error types:", dict(type_counts))
        print("   predicted marginal:", dict(predicted_marginal))

    report["parsing_failure_count"] = sum(1 for r in rows if r["parsed_label"] is None)

    # export diagnostic figures for 20 random single-image errors
    error_rows = [r for r in rows if r["condition"] in SINGLE_IMAGE_CONDITIONS and not r["correct"]]
    rng = random.Random(args.seed)
    rng.shuffle(error_rows)
    picked = error_rows[: args.n_diag]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    pixel_cache: dict[str, dict] = {}
    for row in picked:
        scene = row["scene_id"]
        if scene not in pixel_cache:
            pixel_cache[scene] = world_to_pixel_map(scene)
        out_name = f"diag_{row['sample_id']}_{row['condition']}.png"
        out_path = args.out_dir / out_name
        make_diagnostic_figure(row, index_map[row["sample_id"]], pixel_cache[scene], out_path)
        exported.append(out_name)

    report["exported_diagnostics"] = exported
    report_path = args.out_dir / "analysis_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nExported {len(exported)} diagnostic figures to {args.out_dir}")
    print(f"Analysis summary: {report_path}")


if __name__ == "__main__":
    main()
