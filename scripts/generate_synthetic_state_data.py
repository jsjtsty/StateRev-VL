#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
StateRev-VL 极简合成视觉状态数据生成器

核心设计：
对于同一个当前场景 new_image，同时构造两种历史：

1. maintain:
   old_relation == new_relation
   例如 RIGHT -> RIGHT

2. revision:
   old_relation != new_relation
   例如 LEFT -> RIGHT

因此 maintain 和 revision 可以共享完全相同的当前图片，
唯一差别是此前的视觉历史。

目标物体：
    红色圆形

参照物体：
    蓝色方块

第一版仅研究：
    LEFT / RIGHT
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image, ImageDraw


REL_LEFT = "LEFT"
REL_RIGHT = "RIGHT"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic_state"),
        help="输出目录",
    )

    parser.add_argument(
        "--n-scenes",
        type=int,
        default=100,
        help=(
            "基础当前场景数量。"
            "每个当前场景生成 1 条 revision + 1 条 maintain，"
            "所以默认 100 会得到总共 200 条样本。"
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--min-separation",
        type=int,
        default=140,
        help="目标和参照物在水平方向上的最小中心距离",
    )

    parser.add_argument(
        "--max-y-offset",
        type=int,
        default=40,
        help="两个物体纵向中心允许的最大差距",
    )

    parser.add_argument(
        "--object-size",
        type=int,
        default=56,
        help="目标圆和参照方块的基础尺寸",
    )

    parser.add_argument(
        "--jitter",
        type=int,
        default=18,
        help="旧图中坐标随机扰动范围",
    )

    parser.add_argument(
        "--n-distractors",
        type=int,
        default=0,
        help="干扰物数量。第一轮建议保持 0。",
    )

    return parser.parse_args()


def opposite_relation(relation: str) -> str:
    if relation == REL_LEFT:
        return REL_RIGHT
    if relation == REL_RIGHT:
        return REL_LEFT
    raise ValueError(relation)


def relation_from_positions(
    target_xy: Tuple[int, int],
    anchor_xy: Tuple[int, int],
) -> str:
    return REL_LEFT if target_xy[0] < anchor_xy[0] else REL_RIGHT


def sample_current_positions(
    rng: random.Random,
    image_size: int,
    min_separation: int,
    max_y_offset: int,
    object_size: int,
    relation: str,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    返回：
        target_xy, anchor_xy

    坐标表示物体中心。

    关键点：
    根据 LEFT / RIGHT 关系，先限制 anchor 的可采样范围，
    保证 target 一定有足够空间满足 min_separation。
    """

    pad = object_size + 30

    min_coord = pad
    max_coord = image_size - pad

    # 先检查当前参数组合是否存在合法解
    if max_coord - min_coord < min_separation:
        raise ValueError(
            f"当前参数无法满足最小间距："
            f"image_size={image_size}, "
            f"object_size={object_size}, "
            f"pad={pad}, "
            f"min_separation={min_separation}"
        )

    # anchor 的 y 坐标
    anchor_y = rng.randint(
        pad + 60,
        image_size - pad - 60,
    )

    if relation == REL_LEFT:
        # target 必须位于 anchor 左边至少 min_separation
        #
        # target_x >= min_coord
        # 所以 anchor_x 至少应该：
        # anchor_x >= min_coord + min_separation

        anchor_low = max(
            image_size // 2 - 50,
            min_coord + min_separation,
        )

        anchor_high = min(
            image_size // 2 + 50,
            max_coord,
        )

        if anchor_low > anchor_high:
            raise ValueError(
                f"LEFT 无合法 anchor 范围: "
                f"[{anchor_low}, {anchor_high}]"
            )

        anchor_x = rng.randint(
            anchor_low,
            anchor_high,
        )

        target_low = min_coord
        target_high = anchor_x - min_separation

        target_x = rng.randint(
            target_low,
            target_high,
        )

    elif relation == REL_RIGHT:
        # target 必须位于 anchor 右边至少 min_separation
        #
        # target_x <= max_coord
        # 所以 anchor_x 至多应该：
        # anchor_x <= max_coord - min_separation

        anchor_low = max(
            image_size // 2 - 50,
            min_coord,
        )

        anchor_high = min(
            image_size // 2 + 50,
            max_coord - min_separation,
        )

        if anchor_low > anchor_high:
            raise ValueError(
                f"RIGHT 无合法 anchor 范围: "
                f"[{anchor_low}, {anchor_high}]"
            )

        anchor_x = rng.randint(
            anchor_low,
            anchor_high,
        )

        target_low = anchor_x + min_separation
        target_high = max_coord

        target_x = rng.randint(
            target_low,
            target_high,
        )

    else:
        raise ValueError(
            f"Unsupported relation: {relation}"
        )

    target_y = anchor_y + rng.randint(
        -max_y_offset,
        max_y_offset,
    )

    target_y = max(
        pad,
        min(image_size - pad, target_y),
    )

    return (
        (target_x, target_y),
        (anchor_x, anchor_y),
    )

def sample_old_positions(
    rng: random.Random,
    current_target: Tuple[int, int],
    current_anchor: Tuple[int, int],
    desired_relation: str,
    image_size: int,
    min_separation: int,
    max_y_offset: int,
    object_size: int,
    jitter: int,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    为旧图生成位置。

    anchor 会发生轻微移动，
    目标物体也重新采样，
    但保证 old_relation 等于 desired_relation。

    这样 maintain 图不会与 new 图像素完全相同。
    """

    pad = object_size + 30

    anchor_x = current_anchor[0] + rng.randint(-jitter, jitter)
    anchor_y = current_anchor[1] + rng.randint(-jitter, jitter)

    anchor_x = max(
        image_size // 2 - 80,
        min(image_size // 2 + 80, anchor_x),
    )

    anchor_y = max(
        pad + 30,
        min(image_size - pad - 30, anchor_y),
    )

    if desired_relation == REL_LEFT:

        high = anchor_x - min_separation

        if high <= pad:
            anchor_x = image_size // 2

            high = anchor_x - min_separation

        target_x = rng.randint(
            pad,
            high,
        )

    elif desired_relation == REL_RIGHT:

        low = anchor_x + min_separation

        if low >= image_size - pad:
            anchor_x = image_size // 2

            low = anchor_x + min_separation

        target_x = rng.randint(
            low,
            image_size - pad,
        )

    else:
        raise ValueError(desired_relation)

    target_y = anchor_y + rng.randint(
        -max_y_offset,
        max_y_offset,
    )

    target_y = max(
        pad,
        min(image_size - pad, target_y),
    )

    return (target_x, target_y), (anchor_x, anchor_y)


def draw_scene(
    image_size: int,
    target_xy: Tuple[int, int],
    anchor_xy: Tuple[int, int],
    object_size: int,
    rng: random.Random,
    n_distractors: int = 0,
) -> Image.Image:

    # 使用略微偏灰的背景，避免纯白过于人工
    image = Image.new(
        "RGB",
        (image_size, image_size),
        (242, 242, 242),
    )

    draw = ImageDraw.Draw(image)

    half = object_size // 2

    # 蓝色方块：参照物
    ax, ay = anchor_xy

    draw.rectangle(
        [
            ax - half,
            ay - half,
            ax + half,
            ay + half,
        ],
        fill=(40, 90, 210),
        outline=(20, 50, 140),
        width=3,
    )

    # 红色圆形：目标物体
    tx, ty = target_xy

    draw.ellipse(
        [
            tx - half,
            ty - half,
            tx + half,
            ty + half,
        ],
        fill=(220, 55, 55),
        outline=(150, 25, 25),
        width=3,
    )

    # 可选干扰物。
    # 第一轮请设置为 0。
    distractor_colors = [
        (60, 170, 80),
        (230, 180, 40),
        (150, 80, 190),
        (60, 180, 180),
    ]

    for _ in range(n_distractors):

        size = rng.randint(
            int(object_size * 0.6),
            int(object_size * 1.1),
        )

        x = rng.randint(
            size + 10,
            image_size - size - 10,
        )

        y = rng.randint(
            size + 10,
            image_size - size - 10,
        )

        color = rng.choice(distractor_colors)

        shape = rng.choice(
            ["circle", "rectangle"]
        )

        h = size // 2

        if shape == "circle":
            draw.ellipse(
                [
                    x - h,
                    y - h,
                    x + h,
                    y + h,
                ],
                fill=color,
            )

        else:
            draw.rectangle(
                [
                    x - h,
                    y - h,
                    x + h,
                    y + h,
                ],
                fill=color,
            )

    return image


def write_jsonl(path: Path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():

    args = parse_args()

    rng = random.Random(args.seed)

    output_dir: Path = args.output_dir

    current_dir = output_dir / "images" / "current"
    revision_old_dir = output_dir / "images" / "revision_old"
    maintain_old_dir = output_dir / "images" / "maintain_old"

    current_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    revision_old_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    maintain_old_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    revision_rows = []
    maintain_rows = []

    current_left = 0
    current_right = 0

    for i in range(args.n_scenes):

        pair_id = f"scene_{i:05d}"

        # 强制整体 LEFT / RIGHT 基本均衡
        if i % 2 == 0:
            new_relation = REL_LEFT
            current_left += 1
        else:
            new_relation = REL_RIGHT
            current_right += 1

        # ------------------------------
        # 生成当前状态
        # ------------------------------

        new_target_xy, new_anchor_xy = sample_current_positions(
            rng=rng,
            image_size=args.image_size,
            min_separation=args.min_separation,
            max_y_offset=args.max_y_offset,
            object_size=args.object_size,
            relation=new_relation,
        )

        new_image = draw_scene(
            image_size=args.image_size,
            target_xy=new_target_xy,
            anchor_xy=new_anchor_xy,
            object_size=args.object_size,
            rng=rng,
            n_distractors=args.n_distractors,
        )

        new_image_path = (
            current_dir / f"{pair_id}.png"
        )

        new_image.save(new_image_path)

        # -----------------------------------
        # Maintain:
        # old relation == new relation
        # -----------------------------------

        maintain_old_relation = new_relation

        maintain_target_xy, maintain_anchor_xy = sample_old_positions(
            rng=rng,
            current_target=new_target_xy,
            current_anchor=new_anchor_xy,
            desired_relation=maintain_old_relation,
            image_size=args.image_size,
            min_separation=args.min_separation,
            max_y_offset=args.max_y_offset,
            object_size=args.object_size,
            jitter=args.jitter,
        )

        maintain_image = draw_scene(
            image_size=args.image_size,
            target_xy=maintain_target_xy,
            anchor_xy=maintain_anchor_xy,
            object_size=args.object_size,
            rng=rng,
            n_distractors=args.n_distractors,
        )

        maintain_old_path = (
            maintain_old_dir / f"{pair_id}.png"
        )

        maintain_image.save(
            maintain_old_path
        )

        # -----------------------------------
        # Revision:
        # old relation != new relation
        # -----------------------------------

        revision_old_relation = opposite_relation(
            new_relation
        )

        revision_target_xy, revision_anchor_xy = sample_old_positions(
            rng=rng,
            current_target=new_target_xy,
            current_anchor=new_anchor_xy,
            desired_relation=revision_old_relation,
            image_size=args.image_size,
            min_separation=args.min_separation,
            max_y_offset=args.max_y_offset,
            object_size=args.object_size,
            jitter=args.jitter,
        )

        revision_image = draw_scene(
            image_size=args.image_size,
            target_xy=revision_target_xy,
            anchor_xy=revision_anchor_xy,
            object_size=args.object_size,
            rng=rng,
            n_distractors=args.n_distractors,
        )

        revision_old_path = (
            revision_old_dir / f"{pair_id}.png"
        )

        revision_image.save(
            revision_old_path
        )

        # -----------------------------
        # 基础一致性校验
        # -----------------------------

        assert (
            relation_from_positions(
                new_target_xy,
                new_anchor_xy,
            )
            == new_relation
        )

        assert (
            relation_from_positions(
                maintain_target_xy,
                maintain_anchor_xy,
            )
            == maintain_old_relation
        )

        assert (
            relation_from_positions(
                revision_target_xy,
                revision_anchor_xy,
            )
            == revision_old_relation
        )

        # -----------------------------
        # Maintain record
        # -----------------------------

        maintain_rows.append(
            {
                "sample_id": f"{pair_id}_maintain",
                "pair_id": pair_id,

                "change_type": "maintain",

                "target": "red circle",
                "anchor": "blue square",

                "old_relation": maintain_old_relation,
                "new_relation": new_relation,

                "old_image": str(maintain_old_path),
                "new_image": str(new_image_path),

                "old_target_xy": list(maintain_target_xy),
                "old_anchor_xy": list(maintain_anchor_xy),

                "new_target_xy": list(new_target_xy),
                "new_anchor_xy": list(new_anchor_xy),
            }
        )

        # -----------------------------
        # Revision record
        # -----------------------------

        revision_rows.append(
            {
                "sample_id": f"{pair_id}_revision",
                "pair_id": pair_id,

                "change_type": "revision",

                "target": "red circle",
                "anchor": "blue square",

                "old_relation": revision_old_relation,
                "new_relation": new_relation,

                "old_image": str(revision_old_path),
                "new_image": str(new_image_path),

                "old_target_xy": list(revision_target_xy),
                "old_anchor_xy": list(revision_anchor_xy),

                "new_target_xy": list(new_target_xy),
                "new_anchor_xy": list(new_anchor_xy),
            }
        )

    # -----------------------------
    # 保存 JSONL
    # -----------------------------

    revision_jsonl = (
        output_dir / "synthetic_revision.jsonl"
    )

    maintain_jsonl = (
        output_dir / "synthetic_maintain.jsonl"
    )

    write_jsonl(
        revision_jsonl,
        revision_rows,
    )

    write_jsonl(
        maintain_jsonl,
        maintain_rows,
    )

    # -----------------------------
    # 保存整体信息
    # -----------------------------

    metadata = {
        "seed": args.seed,
        "image_size": args.image_size,
        "n_base_scenes": args.n_scenes,

        "n_revision": len(revision_rows),
        "n_maintain": len(maintain_rows),

        "new_relation_left": current_left,
        "new_relation_right": current_right,

        "target": "red circle",
        "anchor": "blue square",

        "min_separation": args.min_separation,
        "max_y_offset": args.max_y_offset,
        "object_size": args.object_size,
        "n_distractors": args.n_distractors,

        "important_design": (
            "Revision and maintain records with the same pair_id "
            "share exactly the same current image."
        ),
    }

    with (
        output_dir / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 70)
    print("StateRev-VL synthetic data generated")
    print("=" * 70)

    print(
        f"Revision : {len(revision_rows)}"
    )

    print(
        f"Maintain : {len(maintain_rows)}"
    )

    print(
        f"Current LEFT : {current_left}"
    )

    print(
        f"Current RIGHT: {current_right}"
    )

    print()

    print(
        f"Revision index: {revision_jsonl}"
    )

    print(
        f"Maintain index: {maintain_jsonl}"
    )

    print(
        f"Images: {output_dir / 'images'}"
    )

    print()

    print(
        "关键性质：相同 pair_id 的 revision / maintain "
        "共享完全相同的 new_image。"
    )


if __name__ == "__main__":
    main()