"""Minimal controllable shell-game renderer (side view, numpy/OpenCV).

Used for the tracking-algorithm study: swap sequences are chosen by the
caller (so "shortcut" and ground-truth answers can be decorrelated), and the
scene can be rendered with opaque or transparent cups and any number of cups.

Timeline (seconds): lift 0.5 -> hold 1.0 -> lower 0.5 -> k x (swap 1.0 + pause
0.25) -> end hold 0.5. During a swap the two cups exchange x positions; the
first cup travels "in front" (lower on screen, slightly larger), the second
"behind", so their paths never visually coincide. The ball always moves with
the cup covering it and is drawn underneath the cup body, so it is fully
occluded when cups are opaque.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import cv2
import numpy as np

LIFT, HOLD, LOWER, SWAP, PAUSE, END = 0.5, 1.0, 0.5, 1.0, 0.25, 0.5


@dataclass
class Scene:
    n_cups: int
    init: int                      # ball's starting cup position (0 = leftmost)
    swaps: list[tuple[int, int]]   # position pairs swapped, in order
    transparent: bool = False
    width: int = 448
    height: int = 336
    fps: int = 8
    cup_color: tuple[int, int, int] = (40, 90, 200)     # BGR
    bg_color: tuple[int, int, int] = (200, 215, 225)
    table_color: tuple[int, int, int] = (60, 100, 140)
    ball_color: tuple[int, int, int] = (40, 40, 220)
    cup_colors: list | None = None   # optional per-cup colors (distinguishable cups)
    extra: dict = field(default_factory=dict)

    def positions(self) -> np.ndarray:
        margin = self.width * 0.14
        return np.linspace(margin, self.width - margin, self.n_cups)

    def duration(self) -> float:
        return LIFT + HOLD + LOWER + len(self.swaps) * (SWAP + PAUSE) + END

    def ball_trace(self) -> list[int]:
        """Ball position after each swap (ground truth states S_1..S_k)."""
        s, out = self.init, []
        for a, b in self.swaps:
            s = b if s == a else a if s == b else s
            out.append(s)
        return out


def _cup_poly(cx: float, base_y: float, scale: float, w: float) -> np.ndarray:
    bw, tw, h = 0.19 * w * scale, 0.13 * w * scale, 0.27 * w * scale
    return np.array([[cx - bw / 2, base_y], [cx + bw / 2, base_y],
                     [cx + tw / 2, base_y - h], [cx - tw / 2, base_y - h]], np.int32)


def _ease(u: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(max(u, 0.0), 1.0))


def render(scene: Scene) -> np.ndarray:
    """Return uint8 RGB frames, shape [T, H, W, 3]."""
    W, H = scene.width, scene.height
    pos = scene.positions()
    ground = H * 0.72
    n = scene.n_cups
    frames = []
    total = scene.duration()
    for fi in range(int(math.ceil(total * scene.fps))):
        t = fi / scene.fps
        img = np.empty((H, W, 3), np.uint8)
        img[:] = scene.bg_color
        cv2.rectangle(img, (0, int(ground)), (W, H), scene.table_color, -1)
        # slot -> cup identity; cups keep identity, positions permute
        xs = pos.copy()
        ys = np.full(n, ground)
        scales = np.ones(n)
        lift = 0.0
        if t < LIFT:
            lift = _ease(t / LIFT)
        elif t < LIFT + HOLD:
            lift = 1.0
        elif t < LIFT + HOLD + LOWER:
            lift = 1.0 - _ease((t - LIFT - HOLD) / LOWER)
        # which cup (identity) sits at which slot after completed swaps
        slot_of = list(range(n))        # slot_of[cup_id] = current slot
        cup_at = list(range(n))         # cup_at[slot] = cup_id
        ball_cup = scene.init           # ball follows cup identity
        t_sh = t - (LIFT + HOLD + LOWER)
        moving = None
        if t_sh >= 0:
            cyc = SWAP + PAUSE
            cur = int(t_sh // cyc)                       # swap index whose cycle we are in
            in_motion = cur < len(scene.swaps) and (t_sh - cur * cyc) < SWAP
            done = min(len(scene.swaps), cur if in_motion else cur + 1)
            for a, b in scene.swaps[:done]:
                ca, cb = cup_at[a], cup_at[b]
                cup_at[a], cup_at[b] = cb, ca
                slot_of[ca], slot_of[cb] = b, a
            if in_motion:
                moving = (scene.swaps[cur], _ease((t_sh - cur * cyc) / SWAP))
        for cid in range(n):
            xs[cid] = pos[slot_of[cid]]
        if moving is not None:
            (a, b), u = moving
            ca, cb = cup_at[a], cup_at[b]
            xs[ca] = pos[a] + (pos[b] - pos[a]) * u
            xs[cb] = pos[b] + (pos[a] - pos[b]) * u
            d = math.sin(math.pi * u)
            ys[ca] = ground + 0.10 * H * d; scales[ca] = 1 + 0.12 * d     # front
            ys[cb] = ground - 0.10 * H * d; scales[cb] = 1 - 0.12 * d     # behind
        lift_px = lift * 0.20 * W
        # draw order: behind cups first, then others by y
        order = np.argsort(ys)
        ball_drawn = False

        def draw_ball():
            bx, by = xs[ball_cup], ys[ball_cup]
            r = int(0.03 * W * scales[ball_cup])
            cv2.circle(img, (int(bx), int(by - r - 1)), r, scene.ball_color, -1, cv2.LINE_AA)

        for cid in order:
            if cid == ball_cup:
                draw_ball(); ball_drawn = True
            poly = _cup_poly(xs[cid], ys[cid] - lift_px, scales[cid], W)
            col = scene.cup_colors[cid] if scene.cup_colors else scene.cup_color
            if scene.transparent:
                over = img.copy()
                cv2.fillPoly(over, [poly], col, cv2.LINE_AA)
                img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
                cv2.polylines(img, [poly], True, tuple(int(c * 0.6) for c in col), 2, cv2.LINE_AA)
            else:
                cv2.fillPoly(img, [poly], col, cv2.LINE_AA)
                cv2.polylines(img, [poly], True, tuple(int(c * 0.6) for c in col), 2, cv2.LINE_AA)
        if not ball_drawn:
            draw_ball()
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def sample_frames(frames: np.ndarray, src_fps: int, dst_fps: float) -> np.ndarray:
    idx = np.round(np.arange(0, len(frames), src_fps / dst_fps)).astype(int)
    idx = idx[idx < len(frames)]
    return frames[idx]


def write_mp4(frames: np.ndarray, path, fps: int) -> None:
    h, w = frames.shape[1:3]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()


def timeline(events):
    """events: list of ('reveal',) | ('swap', a, b) | ('wait', seconds).
    Returns segments [(t0, t1, event)] ; a reveal lasts LIFT+HOLD+LOWER, a swap SWAP+PAUSE."""
    segs, t = [], 0.0
    for ev in events:
        dur = {'reveal': LIFT + HOLD + LOWER, 'swap': SWAP + PAUSE, 'wait': None}[ev[0]]
        dur = ev[1] if ev[0] == 'wait' else dur
        segs.append((t, t + dur, ev))
        t += dur
    return segs, t + END


def render_events(scene: Scene, events) -> np.ndarray:
    """Like render(), but driven by an explicit event list (reveals and waits
    can be placed anywhere). scene.swaps / scene.init are ignored except init."""
    W, H = scene.width, scene.height
    pos = scene.positions()
    ground = H * 0.72
    n = scene.n_cups
    segs, total = timeline(events)
    frames = []
    for fi in range(int(math.ceil(total * scene.fps))):
        t = fi / scene.fps
        img = np.empty((H, W, 3), np.uint8)
        img[:] = scene.bg_color
        cv2.rectangle(img, (0, int(ground)), (W, H), scene.table_color, -1)
        xs, ys, scales = pos.copy(), np.full(n, ground), np.ones(n)
        slot_of, cup_at = list(range(n)), list(range(n))
        ball_cup = scene.init
        lift, moving = 0.0, None
        for t0, t1, ev in segs:
            if ev[0] == 'swap':
                if t >= t0 + SWAP:
                    a, b = ev[1], ev[2]
                    ca, cb = cup_at[a], cup_at[b]
                    cup_at[a], cup_at[b] = cb, ca
                    slot_of[ca], slot_of[cb] = b, a
                elif t >= t0:
                    moving = ((ev[1], ev[2]), _ease((t - t0) / SWAP))
            elif ev[0] == 'reveal' and t0 <= t < t1:
                u = t - t0
                lift = _ease(u / LIFT) if u < LIFT else 1.0 if u < LIFT + HOLD else 1.0 - _ease((u - LIFT - HOLD) / LOWER)
        for cid in range(n):
            xs[cid] = pos[slot_of[cid]]
        if moving is not None:
            (a, b), u = moving
            ca, cb = cup_at[a], cup_at[b]
            xs[ca] = pos[a] + (pos[b] - pos[a]) * u
            xs[cb] = pos[b] + (pos[a] - pos[b]) * u
            d = math.sin(math.pi * u)
            ys[ca] = ground + 0.10 * H * d; scales[ca] = 1 + 0.12 * d
            ys[cb] = ground - 0.10 * H * d; scales[cb] = 1 - 0.12 * d
        lift_px = lift * 0.20 * W
        order = np.argsort(ys)
        ball_drawn = False
        for cid in order:
            if cid == ball_cup:
                bx, by = xs[ball_cup], ys[ball_cup]
                r = int(0.03 * W * scales[ball_cup])
                cv2.circle(img, (int(bx), int(by - r - 1)), r, scene.ball_color, -1, cv2.LINE_AA)
                ball_drawn = True
            poly = _cup_poly(xs[cid], ys[cid] - lift_px, scales[cid], W)
            col = scene.cup_colors[cid] if scene.cup_colors else scene.cup_color
            cv2.fillPoly(img, [poly], col, cv2.LINE_AA)
            cv2.polylines(img, [poly], True, tuple(int(c * 0.6) for c in col), 2, cv2.LINE_AA)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames)
