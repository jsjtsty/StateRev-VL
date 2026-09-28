# The Stale Present

**Video-language models often report the previous state instead of the current one.**

📄 **Paper (PDF):** [paper/stale_present_arxiv/main.pdf](paper/stale_present_arxiv/main.pdf)
&nbsp;·&nbsp; LaTeX source: [paper/stale_present_arxiv/](paper/stale_present_arxiv/)

Tianyang Sun, Harbin Institute of Technology

---

## What this is about

Ask a video-language model a simple question: *what is shown at the end of the video?* The answer is plainly visible in the final frames. Every model we tested answers it almost perfectly when it sees **only the last frame**. With the **whole video**, many of them answer with the state that came **just before**. We call this failure the *stale present*.

- On synthetic scenes with four earlier layouts and a 0.5 s final layout, the video accuracy of eight open models ranges from **0.30 to 0.80**. From the last frame alone it is **1.00**.
- On real chess games, Qwen3-VL-8B names the **captured** piece instead of the capturing one in 59–78% of trials.

Using controlled stimuli and preregistered tests, we find:

1. **What drives it.** The effect needs at least one earlier state of the same kind as the final one. It grows with the number of such states and shrinks as the final state is shown longer. Frame rate does not matter.
2. **Where it happens.** It is a mid-layer retrieval by the question tokens. Blocking the question tokens' attention to the previous state's visual tokens, only within a narrow band of middle layers, removes most stale answers in Qwen3-VL-8B and Gemma-3-12B.
3. **Real footage.** On Perception Test videos, history costs about 10 points of accuracy over a same-pipeline "frozen video" control. When a clean previous state is present, real frames and rendered tiles give effects of the same size, and the same mid-layer masking removes them.
4. **A fix and its limits.** Present-contrastive decoding (PCD) subtracts the score of the video truncated before the last change, and it removes the stale answers. For pure present-state questions, though, it never beats simply reading the last frame. It helps only when the question needs the history as well as the present.

**Limits.** The newest model we tested, Qwen3.6-27B, is immune. In natural shuffling videos, only some models concentrate their errors on the previous state. We present the stale present as a documented, mechanistically localized failure of current mid-sized open video VLMs, not as a universal property.

## Repository layout

| Path | Contents |
|---|---|
| `paper/stale_present_arxiv/` | The preprint: `main.tex`, `appendix.tex`, `refs.bib`, figures (`figs/make_figs.py` regenerates them), and the compiled `main.pdf`. |
| `docs/tracking_algo_v1/PREREG.md` | The preregistration file. Every addendum (hypotheses and thresholds) was written before the matching data were generated or the models were run. Addenda L–AH belong to this paper; A–K cover earlier shell-game tracking experiments. |
| `docs/tracking_algo_v1/NIGHT_LOG.md` | Chronological lab log with all results, including failed predictions. |
| `docs/tracking_algo_v1/RESEARCH_REPORT_zh.md` | Full research report (in Chinese). |
| `scripts/tracking_algo_*.py` | Stimulus generators, model evaluation, interventions and statistics (see below). |
| `staterev/shellsim.py` | The 2D renderer used for the synthetic cup scenes. |

Other files in the repository come from earlier exploratory work on state tracking in video VLMs and are not needed to reproduce the paper.

## Scripts by paper section

| Paper section | Experiment | Script(s) |
|---|---|---|
| Shared | model loading, input building, letter scoring | `tracking_algo_eval.py` |
| §4 Phenomenon | layouts with *m* prior states (M) | `tracking_algo_now2.py` |
| §4 Phenomenon | real chess games (R) | `tracking_algo_chess_now.py` |
| §5 Drivers | final dwell (L), minimal displays (N), disks (O) | `tracking_algo_now.py`, `tracking_algo_now3.py`, `tracking_algo_now4.py` |
| §5 Drivers | last-frame anchoring (P), chain of thought | `tracking_algo_anchor.py`, `tracking_algo_now_cot.py` |
| §5 Drivers | timeline factorial on word tiles (AD) | `tracking_algo_synthword2.py`, `tracking_algo_synthword2_stats.py` |
| §6 Mechanism | attention shares, recency heads (Q), Qwen band masking | `tracking_algo_now_attn.py`, `tracking_algo_recency_heads.py` |
| §6 Mechanism | band masking for image-list models (Y) | `tracking_algo_layermask.py` |
| §7 Real footage | pilot (Z), dwell (AA), timelines (AB), tiles (AC) | `tracking_algo_real.py`, `tracking_algo_real2.py`, `tracking_algo_real3.py`, `tracking_algo_synthword.py` |
| §7 Real footage | previous state (AE), frozen control (AF), real vs. tiles (AG), masking (AH) | `tracking_algo_real4.py` – `tracking_algo_real7.py` and the matching `*_stats.py` |
| §8 Fix | PCD, change-point masking, history questions, noise (S–W), gate (X) | `tracking_algo_fix.py`, `tracking_algo_gate.py` |
| All | bootstrap CIs, McNemar tests, Holm correction | `tracking_algo_stats.py` |

The scripts write per-item results to `outputs/tracking_algo_v1/`. Generated data, model outputs and model weights are **not** in the repository.

## Environment

We ran all models locally in bf16 on 4× A100-40GB with Python 3.12, PyTorch 2.14 and Transformers 5.15. Weights come from the Hugging Face Hub (Qwen3-VL, Qwen3.5, Qwen3.6, Qwen2.5-VL, LLaVA-OneVision, InternVL3.5, Gemma-3, Idefics3). External data:

- chess game records: MET-Bench-Chess;
- real footage: the Perception Test videos distributed with MVBench, plus the official Perception Test annotations (`all_valid.json`).

## Citation

```bibtex
@misc{sun2026stalepresent,
  title  = {The Stale Present: Video-Language Models Often Report the Previous State Instead of the Current One},
  author = {Sun, Tianyang},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/jsjtsty/StateRev-VL}
}
```
