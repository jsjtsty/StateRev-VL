#!/usr/bin/env python3
"""Figures for the stale-present preprint. Numbers are copied from outputs/tracking_algo_v1/stats_ci.md,
synthword2_summary.md, layermask_gemma3_12b.jsonl, recency_heads_test_qwen3vl8b_*.jsonl and real7_*.jsonl
(the last three are recomputed here when present)."""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / 'outputs/tracking_algo_v1'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})


def fig_models():
    # (model, last-frame, [lo, hi], video, [lo, hi]); M m=4 D_fin=0.5 s and chess captures m=1 D_fin=0.5 s
    M = [('Gemma-3-12B', 1.00, None, 0.30, (0.18, 0.42)), ('Idefics3-8B', 1.00, None, 0.37, (0.25, 0.50)),
         ('Qwen3-VL-8B', 1.00, None, 0.40, (0.28, 0.53)), ('InternVL3.5-8B', 1.00, None, 0.57, (0.43, 0.70)),
         ('Qwen3.5-9B', 1.00, None, 0.58, (0.45, 0.70)), ('LLaVA-OV-7B', 1.00, None, 0.68, (0.57, 0.80)),
         ('Qwen3-VL-32B', 1.00, None, 0.80, (0.70, 0.90)), ('InternVL3.5-GPT-OSS', 1.00, None, 0.97, (0.92, 1.00)),
         ('Qwen3.6-27B', None, None, 0.99, None)]
    C = [('Qwen3-VL-8B', 0.86, (0.79, 0.92), 0.20, (0.12, 0.28)), ('InternVL3.5-GPT-OSS', 0.70, (0.61, 0.79), 0.39, (0.29, 0.49)),
         ('InternVL3.5-8B', 0.83, (0.75, 0.90), 0.55, (0.45, 0.65)), ('Gemma-3-12B', 0.95, (0.90, 0.99), 0.61, (0.51, 0.70)),
         ('LLaVA-OV-7B', 0.65, (0.55, 0.74), 0.62, (0.52, 0.71)), ('Qwen3-VL-32B', 0.96, (0.92, 0.99), 0.63, (0.54, 0.72)),
         ('Qwen3.5-9B', 0.90, (0.84, 0.95), 0.90, (0.84, 0.95)), ('Qwen3.6-27B', None, None, 0.995, None)]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for ax, rows, title in ((axes[0], M, '(a) Synthetic layouts, 4 prior states'), (axes[1], C, '(b) Real chess captures')):
        for i, r in enumerate(rows):
            name, img, _, vid = r[0], r[1], r[2], r[3]
            if img is not None:
                ax.plot([vid, img], [i, i], color=GRID, lw=2, zorder=1)
                ax.scatter(img, i, s=26, color=ORANGE, zorder=3, edgecolor='white', linewidth=0.8)
            ci = r[4]
            if ci:
                ax.plot(ci, [i, i], color=BLUE, lw=1, alpha=0.6, zorder=2)
            ax.scatter(vid, i, s=26, color=BLUE, zorder=3, edgecolor='white', linewidth=0.8)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
        ax.set_xlim(0, 1.04); ax.set_xlabel('accuracy on "what is at the end?"')
        ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(title, fontsize=8, color=INK, loc='left')
    axes[0].scatter([], [], color=ORANGE, s=26, label='last frame only'); axes[0].scatter([], [], color=BLUE, s=26, label='full video (95% CI)')
    fig.legend(loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(HERE / 'fig_models.pdf', bbox_inches='tight')


def fig_timeline():
    # pooled image-only − video gap (5 models, n=67 each), averaged over fps (no fps effect)
    pos = {(1, 1): 0.245, (1, 4): 0.285, (4, 1): 0.455, (4, 4): 0.550}
    order = {(1, 1): 0.035, (1, 4): 0.010, (4, 1): 0.185, (4, 4): 0.335}
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.0), sharey=True)
    for ax, d, t in ((axes[0], pos, '(a) "which letter is at slot p?"'), (axes[1], order, '(b) "what is the final order?"')):
        for dur, col, mk in ((1, BLUE, 'o'), (4, ORANGE, 's')):
            y = [d[(1, dur)], d[(4, dur)]]
            ax.plot([1, 4], y, color=col, lw=2, marker=mk, ms=5, mec='white', mew=0.8)
            ax.text(4.15, y[1], f'{dur} s / state', color=INK2, va='center', fontsize=7)
        ax.set_xticks([1, 4]); ax.set_xlim(0.6, 5.6); ax.set_xlabel('number of prior states m')
        ax.set_title(t, fontsize=8, color=INK, loc='left'); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel('last-frame − video accuracy'); axes[0].set_ylim(0, 0.62)
    fig.tight_layout(); fig.savefig(HERE / 'fig_timeline.pdf', bbox_inches='tight')


def pstale(rows, key, stale='pen', dom=None):
    get = (lambda r: r[dom][key]) if dom else (lambda r: r[key])
    return float(np.mean([get(r) == r[stale] for r in rows]))


def fig_layers():
    series = {}
    # synthetic: Qwen3-VL-8B (Addendum L data, D_fin <= 1 s), Gemma-3-12B (Addendum Y)
    L = [json.loads(l) for l in open(OUT / 'recency_heads_test_qwen3vl8b_layers.jsonl')]
    L = [r for r in L if r['dfin'] <= 1.0]
    b0 = pstale(L, 'none')
    series[('syn', 'Qwen3-VL-8B')] = (36, [(0, 8), (9, 17), (18, 26), (27, 35)],
                                      [b0 - pstale(L, f'text_mask_pen_L{a}-{b}') for a, b in [(0, 8), (9, 17), (18, 26), (27, 35)]])
    G = [json.loads(l) for l in open(OUT / 'layermask_gemma3_12b.jsonl')]
    gb = [(6 * k, 6 * k + 5) for k in range(8)]
    series[('syn', 'Gemma-3-12B')] = (48, gb, [pstale(G, 'none') - pstale(G, f'pen_L{a}-{b}') for a, b in gb])
    for m, name, nl in (('qwen3vl8b', 'Qwen3-VL-8B', 36), ('llava_ov7b', 'LLaVA-OV-7B', 28), ('gemma3_12b', 'Gemma-3-12B', 48)):
        R = [json.loads(l) for l in open(OUT / f'real7_{m}.jsonl')]
        bands = [tuple(map(int, k.split('_L')[1].split('-'))) for k in R[0]['real'] if k.startswith('prior_L')]
        series[('real', name)] = (nl, bands, [pstale(R, 'none', 'stale', 'real') - pstale(R, f'prior_L{a}-{b}', 'stale', 'real') for a, b in bands])
    col = {'Qwen3-VL-8B': BLUE, 'Gemma-3-12B': ORANGE, 'LLaVA-OV-7B': AQUA}
    mk = {'Qwen3-VL-8B': 'o', 'Gemma-3-12B': 's', 'LLaVA-OV-7B': '^'}
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=True)
    for ax, dom, t in ((axes[0], 'syn', '(a) Synthetic videos'), (axes[1], 'real', '(b) Real frames, clean prior state (n=52)')):
        for (d, name), (nl, bands, ys) in series.items():
            if d != dom:
                continue
            xs = [((a + b + 1) / 2) / nl for a, b in bands]
            ax.plot(xs, ys, color=col[name], lw=2, marker=mk[name], ms=5, mec='white', mew=0.8, label=name)
        ax.axhline(0, color=INK2, lw=0.6); ax.set_xlim(0, 1); ax.set_xlabel('relative depth of the masked band')
        ax.set_title(t, fontsize=8, color=INK, loc='left'); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7, loc='upper left')
    axes[0].set_ylabel('drop in P(stale answer)')
    fig.tight_layout(); fig.savefig(HERE / 'fig_layers.pdf', bbox_inches='tight')
    for k, (nl, bands, ys) in series.items():
        print(k, [f'L{a}-{b}:{y:+.3f}' for (a, b), y in zip(bands, ys)])


if __name__ == '__main__':
    fig_models(); fig_timeline(); fig_layers()
