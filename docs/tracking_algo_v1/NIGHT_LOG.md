# Tracking-algorithm study: night log (2026-09-23)

Everything here is new data from the controlled renderer (`staterev/shellsim.py`). Hypotheses were pre-registered in `PREREG.md` (main set plus Addenda A–D) before each set was run. Findings are marked **[confirmatory]** or **[exploratory]**.

## 0. One-paragraph summary

Across 5 open VLMs (Qwen3-VL-8B, Qwen3.5-9B, Qwen3.6-27B, Qwen2.5-VL-7B, LLaVA-NeXT-Video-7B), native shell-game answers are at chance whenever the ball is occluded, including with a single swap, and 0.87–1.00 when the cups are transparent. Answers do not depend on the swaps at all.

The pre-registered position-anchored shortcut (from G0) does **not** replicate; it is dropped.

Mechanistically, the models perceive every swap perfectly: the swapped pair is decodable at 1.00 from the video tokens during that swap, and Qwen3.5 answers per-swap perception questions 100% correctly. They also register the ball whenever it is visible. But updating the occluded ball is shallow:
- the state after the first swap is decodable (0.97, and it survives 4 s of delay);
- in random sequences, later updates collapse to chance (S2 0.4–0.6, S3+ chance). Structured sequences (a guaranteed first ball-move) allow a second update (0.86), so there is no hard one-step limit; the exact rule is open.

The same composition, given as text, also fails when the model must answer immediately (Qwen3.5 k=1: 0.28), and succeeds with written step-by-step reasoning (~1.0 up to k=4). The bottleneck is **latent (no-CoT) state update**, not perception. Even where the internal state exists (k=1: 0.97 decodable), the native answer ignores it (0.24–0.33).

## 1. Native behaviour (G1, 870 videos + 300 colored; `init` prompt)

| model | opaque3 | transp3 | colored3 | k1 | opaque4 / 5 |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 0.39 | 0.87 | 0.43 | 0.23 | 0.30 / 0.12 |
| Qwen3.5-9B | 0.31 | 0.89 | 0.31 | 0.53 | 0.25 / 0.23 |
| Qwen3.6-27B | 0.31 | **1.00** | **0.65** | 0.50 | 0.26 / 0.26 |
| Qwen2.5-VL-7B | 0.38 | 0.55 | 0.41 | 0.27 | 0.13 / 0.17 |
| LLaVA-NeXT-Video-7B | 0.37 | 0.37 | 0.38 | 0.27 | 0.27 / 0.22 |

- **[confirmatory]** H1 (shortcut) fails in all models. H2 (transparency helps) holds for the three Qwen3.x models and Qwen2.5.
- **[confirmatory]** H4 (colored cups) holds only for 27B (0.65).
- On opaque videos, answers collapse to one position (e.g. Qwen3-VL answers Middle 82%).
- **[confirmatory] Minimal pairs (Addendum A).** Inserting a GT-changing swap flips the answer no more than an irrelevant swap does, in every model (e.g. 27B 0.26 vs 0.28). Native answers ignore the swaps.
- **[exploratory] Thinking.** Qwen3-VL-8B-Thinking with a 1024-token budget, answer forced at the end: 0.37 (n=57). It never finished reasoning within budget.

## 2. Perception vs composition

- **[exploratory] Per-swap perception** (each swap clip asked separately):
  - Qwen3.5-9B: 1.00. Composing its own perceptions symbolically gives 1.00 final accuracy.
  - Qwen3-VL-8B: 0.69, confusing Middle–Right with Left–Middle.
- **[exploratory] Text only, true swaps given** (60 problems per k):

| k | Qwen3.5 direct | Qwen3.5 CoT | Qwen3-VL direct | Qwen3-VL CoT |
|---|---|---|---|---|
| 1 | 0.28 | 1.00 | 0.67 | 1.00 |
| 2 | 0.38 | 1.00 | 0.45 | 1.00 |
| 3 | 0.37 | 1.00 | 0.40 | 0.90 |
| 4 | 0.32 | 0.93 | 0.38 | 0.92 |
| 6 | 0.27 | 0.62 | 0.30 | 0.63 |

Answer-only responses collapse to "A".

## 3. Where is the state? Hidden-state probes (logistic, 5-fold CV)

### 3a. Last token, opaque3 (n=300)

| label | Qwen3-VL-8B | Qwen3.5-9B |
|---|---|---|
| initial position | 1.00 | 1.00 |
| swapped pairs | 0.9–1.0 | 0.9–1.0 |
| final position | 0.33–0.38 | 0.31–0.37 |
| state after swap j (Qwen3-VL L24) | S1 0.86, S2 0.46, S3–S6 chance | S1 0.60, later chance |

### 3b. Video tokens, group right after swap j [exploratory]

| | S1 | S2 | S3–S6 | pair j during swap j |
|---|---|---|---|---|
| Qwen3-VL-8B, opaque, L20–24 | **0.97** | 0.36–0.39 | 0.28–0.47 | 1.00 |
| Qwen3.5-9B, opaque, L18 | **0.92** | 0.37 | 0.28–0.43 | 1.00 |
| Qwen3.5-9B, transparent | 1.00 | 1.00 | 1.00 | |

The additive symbolic baseline for S1 (init + pair) is 0.48, so 0.97 means a real, non-linear one-step update.

**Replicated across architectures and scale** [exploratory]. On opaque video tokens, S1 / S2 / S3–S6:

| model | S1 | S2 | S3–S6 | transparent |
|---|---|---|---|---|
| Qwen3.6-27B (L34) | 0.96 | 0.37 | 0.33–0.42 | all ≈1.00 |
| Qwen2.5-VL-7B (L16) | 0.97 | 0.37 | 0.27–0.43 | all ≈1.00 |

This holds even though Qwen2.5-VL's native answers are completely collapsed.

### 3c. Horizon set (Addendum C), last token L24, n=90 per condition

| condition | Qwen3-VL-8B | native accuracy |
|---|---|---|
| K1: one swap right after the start, 0 / 2 / 4 s delay | **0.98 / 0.97 / 0.97** | 0.33 / 0.24 / 0.26 |
| K3_m1 / K4_m1: one swap after a mid-video reveal | 0.49 / 0.41 | 0.44 / 0.30 |
| K3_m2 / K3_m3 / K4_m4 | 0.36 / 0.28 / 0.34 | |

- **[confirmatory] H5 ("one update since the last sighting") is falsified**, and specifically in the pre-registered second failure mode: the tracked update is special to the first swap of the video.
- The re-sighted position itself is decodable during the reveal and after the cups are lowered (1.00). The model registers the new sighting but does not apply the next swap to it.
- Qwen3.6-27B replicates this. Best layer per condition, post hoc: K1 0.96–0.98; one swap after a reveal 0.53–0.57; m≥2 0.37–0.46.
- Qwen3.5 shows the same pattern with weaker last-token signal: K1 0.54–0.63 vs m=1-after-reveal 0.34–0.40. At n=90 this is underpowered; a power check shows S1 at n=90 ≈ 0.62, versus 0.92 at n=300.
- **The representation–readout dissociation is striking.** At k=1, the state is present at 0.97 in the hidden, yet the native answer is at chance.

### 3d. Binding set (Addendum D): invalid by design

- B_dist_first and B_dist2_first can be solved linearly from per-position swap-involvement counts (symbolic baseline 1.00), so their 1.00 hidden decodability is uninformative.
- Only B_ball_first is valid (symbolic 0.57, hidden 1.00): the first-swap update survives a later distractor swap.
- Native accuracy (Qwen3-VL): B_ball_first 0.53 vs 0.29 for the other two conditions.
- H6 needs a redesigned distractor that is not linearly decodable.


### 3e. Interference (Addendum E) and second update (Addendum F), last token

All sets are 3 opaque cups, n=150 per condition. Symbolic linear baselines are 0.33–0.55, so linear shortcuts are excluded.

| condition | Qwen3-VL-8B L24 | Qwen3.5-9B L18 | Qwen3.6-27B best L |
|---|---|---|---|
| E_R: reveal, random | 0.99 | 0.75 | |
| E_WR: reveal, wait, random | 0.99 | 0.78 | |
| E_DR: reveal, distractor, random | **0.98** | 0.63 | |
| F_DVR: reveal, distractor, reveal, random | 1.00 | 0.65 | 1.00 |
| F_MR: reveal, ball-move, random | **0.86** | 0.55 | 0.89 |
| F_MVR: reveal, ball-move, reveal, random | **0.86** | 0.44 | 0.87 |

- **[confirmatory] H7 ("first event only") is falsified.** A preceding distractor does not block the update.
- **[confirmatory] H8 ("one ball-move per video") is falsified for Qwen3-VL and 27B.** A second update IS decodable (0.86–0.89) when the first swap is a guaranteed ball-move. It is only weakly present in Qwen3.5 (0.44–0.55).
- So the "first-swap-only" account from 3c is **too strong**. In random sequences (opaque3, n=300), decodability still collapses with depth: S1 0.97 → S2 0.4–0.6 → S3+ chance. Horizon K3_m1 was 0.49.
- But in structured sequences a second update exists. The collapse is structure-dependent, not a hard one-step limit.
- A post-hoc split of opaque3 by whether swap 1 moved the ball is inconclusive (n = 30–200).

### 3f. Depth set (exploratory; n=900, reveal + 4 uniformly random swaps)

Decodability of S_j at the video tokens right after swap j (chance ≈ 0.36):

| model | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| Qwen3-VL-8B L24 | 1.00 | 0.56 | 0.47 | 0.37 |
| Qwen3.5-9B L18 | 1.00 | 0.43 | 0.44 | 0.38 |
| Qwen3.6-27B L34 | 1.00 | 0.49 | 0.47 | 0.36 |

- Last-token S2 for Qwen3-VL is 0.65.
- The collapse does **not** depend on how many times the ball has moved so far (0 / 1 / 2+ moves give the same accuracy).
- With adequate power: one update is perfect, the second is weak, the third and fourth are at chance, in all three models.
- The F_MR / F_MVR result (0.86) is therefore a structured-sequence exception still to be explained. One candidate is fewer (init, pair) combinations when the first swap always moves the ball.

## 4. Interpretation (current best account)

1. Perception is not the bottleneck. Each swap, the initial position and every re-sighting are encoded near-perfectly.
2. Latent state update is shallow and fragile.
   - The first occluded swap is applied reliably, around L8–24 on the video tokens, in all 4 probed Qwen-family models. It survives delay and distractors.
   - In random swap sequences, decodability collapses by the second or third update.
   - Addenda E/F show this is **not** a strict one-update or first-event limit: with a guaranteed first ball-move, a second update reaches 0.86–0.89 in Qwen3-VL and 27B.
   - The exact rule governing when updates fail is **still open**.
3. Readout does not use even the one computed state: native accuracy stays at chance where the hidden state gives 0.97.
4. Explicit step-by-step reasoning in text solves the composition. With video input, the 8B Thinking model does not get there within 1k tokens.

## 5. Caveats

- One synthetic renderer, a simple 2D side view.
- Mechanism results cover four Qwen-family models (Qwen2.5-VL, Qwen3-VL, Qwen3.5, Qwen3.6); LLaVA is not yet probed on the new data. No non-Qwen model has token-level results.
- Last-token vs video-token probes differ in strength.
- The Thinking evaluation is budget-limited.
- The binding test is invalid (see 3d).

## Engineering

- Installed `flash-linear-attention` 0.5.2 (via proxy); fla vs torch-fallback predictions agree on 30/30 items.
- Qwen3.6-35B-A3B is skipped: it offloads to CPU on 2×40 GB.

## 6. Day 2 (2026-09-23, continued)

### 6a. Causal activation patching (Addendum G; Qwen3-VL-8B, 240 target/source pairs, one-swap videos)

- Source and target share the initial position and timing but end in different positions.
- Frozen last-token probes were fit on 240 separate videos.

| patched tokens | L8 | L16 | L24 | L32 |
|---|---|---|---|---|
| during-swap video tokens → probe24 reports the source state | 0.86 | **0.99** | 0.00 | 0.00 |
| post-swap video tokens → probe24 reports the source state | 0.08 | 0.00 | 0.00 | 0.00 |
| all video tokens | 1.00 | 0.98 | 0.00 | 0.00 |
| same-state control (probe accuracy) | 0.87 | 1.00 | 1.00 | 1.00 |

Native answer under the during-swap patch:
- "source" answers stay at 0.33–0.35, the unpatched rate;
- the logit margin moves only +0.42 [0.26, 0.58].

Verdicts:
- **[confirmatory] H9 as worded is falsified.** Post-swap tokens do not causally carry the state read at the last token. They do represent the one-step state (decodable 1.00), but it is not what gets read.
- The causally used information sits on the **during-swap** tokens and is transferred to the query position between L16 and L24. After L24 the video tokens no longer matter.
- **[confirmatory] H10 (readout gap) holds.** Even a fully transplanted internal state does not change the model's answer.

Mechanistic picture: no recurrent object-state memory is maintained along the video. The query token reads the event tokens at L16–24 and composes them there. That supports one composition step, not a chain. This matches the Head 0 localization at L24 in the earlier VET-Bench study.

### 6b. Non-Qwen replication (LLaVA-NeXT-Video-7B, Llama-based; depth set, n=900, video tokens at 2 fps)

| S1 | S2 | S3 | S4 |
|---|---|---|---|
| 1.00 (L11+) | 0.44–0.51 | 0.40 | 0.35–0.37 |

This is the same depth collapse as the 3 Qwen models.

### 6c. Training-free method: perceive-then-reason

Ask about each swap window separately, then let the model compose its own perceived swaps in written step-by-step text.

- Synthetic opaque3 (partial run, n=137): Qwen3.5-9B D3 **0.97** vs native 0.31.
- Qwen3-VL-8B is limited by its per-swap perception (0.66).
- Caveat: swap windows come from metadata (oracle segmentation).

### 6d. REAL VET-Bench cup videos (50 videos × 5 swaps)

| method | Qwen3.5-9B | Qwen3-VL-8B | Qwen3.6-27B |
|---|---|---|---|
| native answer, start given / perceived | 0.30 / 0.34 | 0.38 / 0.34 | 0.30 / 0.28 |
| native with reasoning (thinking, 2048-token budget, answer forced) | 0.32 | 0.36 (Thinking model) | – |
| per-swap perception (250 swaps) | **1.00** | 0.86 | **1.00** |
| start position perception | 1.00 | 1.00 | 1.00 |
| perceive + exact rule (D1, upper bound) | 1.00 | 0.38 | 1.00 |
| **perceive-then-reason (D3, model composes in writing)** | **0.86** | 0.40 | **1.00** (start given and start perceived) |
| stepwise: one clip at a time, current state given in text, answer only | 0.32 (per step 0.34) | 0.28 (per step 0.33) | – |

Reference: VET-Bench's Molmo2-SGCoT (fine-tuned on 300 samples) reports 0.92.

Findings:
- The failure transfers from our renderer to the real benchmark.
- The training-free decomposition recovers most of the gap for a model with good per-swap perception.
- **Stepwise prompting fails even for a single update**: state in text plus one swap clip, answer-only, gives chance. Answer-only composition fails even at depth one, matching text-only k=1 = 0.28. **Verbalization is necessary**: the model must write both the perceived swap and the reasoning steps.
- Remaining D3 errors come from written composition over 5 steps (text-only CoT: k=5 0.84, k=6 0.76).
- An alternative "state lines + stated rule" prompt was worse (0.19–0.38; possibly a parsing issue, not investigated).

