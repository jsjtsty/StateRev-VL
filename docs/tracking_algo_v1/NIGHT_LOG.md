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


### 6e. Label-free perceive-then-reason on real VET-Bench (no swap windows from metadata)

**Segmentation.** Segments are cut model-free at local minima of pixel-motion energy (min separation 1 s for cup, 1.5 s for card) and at motion onsets/offsets. Segments shorter than 0.5 s are dropped.

**Per-segment questions.** Each remaining segment gets two questions:
1. "do two cups swap? yes/no";
2. if yes, "which two?" (3-way).

A first version used a single 4-way question that included "no swap". It failed: Qwen3.5 collapsed to answer A, and 27B called the 0.4 s cup-lowering segment a swap. That version was replaced; its results are in `*_autoseg.jsonl`.

| model / task | swap sequence exactly right | D1 (exact rule) | D3 (written reasoning) | native |
|---|---|---|---|---|
| Qwen3.5-9B, cup | **1.00** | 1.00 | **0.92** | 0.30 |
| Qwen3.5-9B, card | 0.92 | 0.92 | 0.74 | 0.38 |
| Qwen3.6-27B, cup | 0.78 | 0.80 | 0.80 | 0.30 |
| Qwen3.6-27B, card | 0.76 | 0.80 | 0.80 | 0.32–0.38 |
| Qwen3-VL-8B, cup | 0.00 | 0.28 | 0.20 | 0.38 |
| Qwen3-VL-8B, card | 0.18 | 0.56 | 0.52 | 0.30–0.38 |

- The start position is perceived from the video in all cases (100%). `vis` equals `init` everywhere.
- 27B's errors come from the yes/no detector answering "no" on real swaps. With oracle windows, 27B is 1.00.
- Qwen3.5 card: D3 < D1. The written composition for cards loses 0.18 against the exact rule.

### 6f. Two-swap causal patching (Addendum H; Qwen3-VL-8B, 480 targets)

**Unpatched baseline (targets, last token L24).** Probe accuracy: S2 (final) 0.985, S1 0.99, pair1 1.00, pair2 1.00. Native answer 0.377. Probe CV on the 480 training videos: S2 0.94, S1 0.998.

Table entries are the fraction of pairs whose last-token L24 probe reports the SOURCE value. The unpatched baseline for this is ≤ 0.01 everywhere.

| source / patched tokens | L8: S2 | L8: S1 or pair | L16: S2 | L16: S1 or pair |
|---|---|---|---|---|
| dp1 (pair1 differs), swap1 tokens | **0.62** | S1 0.72, pair1 0.81 | 0.20 | S1 **0.84**, pair1 0.90 |
| dp1, swap2 tokens | 0.22 | S1 0.14 | **0.40** | S1 0.06 |
| dp1, post-swap2 tokens | 0.15 | 0.02 | 0.05 | 0.00 |
| dp2 (pair2 differs), swap1 tokens | 0.18 | pair2 0.01 | 0.04 | pair2 0.00 |
| dp2, swap2 tokens | 0.57 | pair2 0.70 | **0.69** | pair2 **0.99** |
| dp2, post-swap2 tokens | 0.17 | pair2 0.11 | 0.05 | pair2 0.00 |

Native answers follow the source's final state in 0.29–0.36 of pairs (chance level) under every condition.

**Hypothesis verdicts**
- **H11a is supported.** Each swap's own tokens carry that event to the query at L16: pair2 0.99, S1 0.84.
- **H11b is falsified. Composition happens.** When dp1 differs only in the first swap, patching swap2's tokens at L16 moves the final state to the source in 0.40 of pairs, while the S1 and pair1 probes stay unchanged (0.06 and 0.00). Swap2's own content is identical between source and target. So by L16 the swap2 video tokens carry the composed state S2 = f(S1, pair2), not just their own event. At L8 the same patch gives only 0.22: the composition forms at the swap2 video tokens between L8 and L16.
- Consistent with this, patching swap1 tokens at L8 changes S2 in 0.62 of pairs (the change is downstream-composed). The same patch at L16 changes S2 in only 0.20: by then S1 has already been read into the swap2 tokens.

**Revised mechanism.**
- The state is updated sequentially **in the video-token stream**: the tokens of swap j read the state from the swap j−1 tokens in the early–middle layers (L8–L16).
- The query reads the most recent event tokens.
- The native answer ignores all of it. The readout gap holds under causal intervention for k=2.
- The earlier "one latent update only" account is withdrawn for k=2. Where the chain breaks (k=3 or 4; final S4 at the last token is 0.39) is measured in Addendum I.

### 6g. Final state vs number of swaps (Addendum I; n=600 per k)

5-fold CV decodability of the final state. Majority baseline 0.35–0.37.

**Symbolic linear baselines** (logistic regression):
- one-hot of init and each pair: k=1 0.38, k=2 0.45, k=3 0.34, k=4 0.36;
- one-hot init × last pair: 1.00 at k=1, ≤0.38 for k ≥ 2.

Decoding above these baselines at k ≥ 2 therefore requires composition.

**Qwen3.5-9B**

| k | last token (best layer / L32) | video group right after the last swap (best / L4) |
|---|---|---|
| 1 | 0.995 / 0.995 | 0.995 / 0.72 |
| 2 | **0.96** / 0.94 | 0.94 / 0.42 |
| 3 | **0.76** / 0.71 | 0.67 / 0.36 |
| 4 | **0.54** / 0.47 | 0.46 / 0.34 |

Reading:
- The internal state degrades gradually, about −0.2 per additional swap after k=2. There is no hard one-step limit.
- At the post-swap video tokens, the state forms in L8–L14, consistent with the Addendum H composition locus.
- The native answer is at chance for all k. The readout gap is largest at k = 1–2, where the state is near-perfect internally.


### 6h. CORRECTION: the decodable "final state" is a history lookup, not a state variable

**The test.**
- Final-state probes are re-evaluated with GroupKFold, where a group is the full history tuple (init, pair_1..pair_k).
- Test histories are therefore never seen in training.
- A history-invariant "ball location" code would transfer; a code over history conjunctions would not.

**Results** (last token; the "random split" column uses the same probe with StratifiedKFold):

| data | random split | held-out history |
|---|---|---|
| finalk k=1, Qwen3-VL L24 / Qwen3.5 L18 | 1.00 / 0.995 | **0.00 / 0.00** |
| finalk k=2, Qwen3-VL L24 / Qwen3.5 L18 | 0.96 / 0.95 | **0.04 / 0.11** |
| positive control: init, k=1–2 (both models) | — | 1.00 |
| positive control: pair1, k=2 | — | 0.88 / 0.95 |
| **transparent cups** (transp3, k=2–6, 285 unique histories), Qwen3-VL all layers | — | **1.00** |
| opaque cups (opaque3, same design), Qwen3-VL | — | 0.27–0.34 (chance) |

Held-out accuracy far below chance at k=1–2 is the signature of a probe that combines additive or conjunctive history features. Held-out combinations have systematically different labels than the additive prediction.

**Consequences (supersede 6f/6g and parts of 3/6a).**
1. With opaque cups, neither model has a history-invariant representation of where the ball is, at any layer, even after one swap.
   - The random-split decodability (0.99 / 0.96 / 0.75 / 0.55 for k=1–4) reflects how often each history tuple is seen in training (9 / 27 / 81 / 243 tuples for 600 videos). It does not reflect state tracking.
   - With visible balls, a location code exists (1.00 held-out).
2. The "readout gap" ("the model knows the state but does not say it") is **withdrawn**. The answer is at chance because there is no state variable to read.
3. Addenda G and H patching used random-split probes.
   - They show event information flow: each swap's tokens carry that swap's content to the query at L8–L16, and swap2 tokens integrate swap1 information between L8 and L16.
   - The integrated content is a history conjunction, not a location state.
   - "Composition" in 6f should read "integration of event history".
4. Methodological point: random-split linear probes of "world state" in sequence tasks with a finite history space are confounded by history lookup. Held-out-history evaluation is required.
   - This likely affects prior probing claims in the literature.

**Revised account.**
- Video VLMs perceive each swap and the initial position, and carry both to the query.
- They do not form a latent location state for occluded objects: no algorithmic composition happens.
- Perceive-then-reason works because it externalizes the state as text.

**6h addendum: all models, held-out-history final-state probe at the last token** (transp3 / opaque3; k=2–6; 285 unique histories per set)

| model | transparent | opaque |
|---|---|---|
| Qwen3-VL-8B | 1.00 | 0.27–0.34 |
| Qwen3.5-9B | 1.00 | 0.29–0.32 |
| Qwen3.6-27B | 0.99–1.00 | 0.26–0.40 |
| Qwen2.5-VL-7B | 0.55–0.80 | 0.33–0.36 |

finalk held-out-history results, k=3/4 (both models): 0.04–0.31, all ≤ chance. The random-split k=4 symbolic tuple baseline is 0.88 (not 1.00) because many histories are singletons.

**Colored cups (colored3, each cup a distinct color), held-out-history final-state probe, last token**

| model | early layer | middle layer | last layer | native answer |
|---|---|---|---|---|
| Qwen3-VL-8B | 0.32 (L16) | **0.64** (L28) | 0.59 (L36) | 0.43 |
| Qwen3.5-9B | 0.28 (L14) | 0.38 (L25) | 0.48 (L32) | 0.31–0.37 |

With colored cups, a partially history-invariant location code appears in late layers. It exceeds the native answer. The 27B model (native 0.65) is pending.

With distinct colors, the target can be re-identified by appearance at the end: the color seen at the reveal, then the final layout. No motion tracking is required.

**Working account.**
- Models localize an occluded object by **appearance re-identification**: a partial, late-layer code appears when the cups are distinguishable.
- They do not localize it by **tracking through motion**: no code appears for identical cups, at any k.

**Qwen3.5-9B two-swap patching (Addendum H replication; probes at L18, random-split, so they read history conjunctions)**
- The only substantial effects come from patching at L8:
  - swap1 tokens → pair1 probe 0.64;
  - swap2 tokens → pair2 probe 0.46.
- Patching at L16 has no effect on any probe (≤ 0.02 above baseline). In this hybrid DeltaNet/attention model, event information has already left the video tokens by L16.
- Native answers never follow the source.

**27B card with det2 (two-phrasing detector):** no change (sequence exactly right 0.76; D1 = D3 = 0.80). Missed swaps are not fixed by rephrasing.

**Colored cups, Qwen3.6-27B:** held-out-history probe 0.31 (L26), **0.81** (L49), 0.81 (L64); native answer 0.65.

Across the 3 models, the strength of the late-layer appearance-based location code tracks native accuracy on colored cups:

| model | code | native |
|---|---|---|
| Qwen3.5 | 0.48 | 0.34 |
| Qwen3-VL | 0.64 | 0.43 |
| 27B | 0.81 | 0.65 |

The code is always higher than the answer (a modest readout gap exists here, where a real state code exists).

### 6i. Addendum J: appearance re-id vs motion (Qwen3-VL-8B, Qwen3.5-9B; n=300 paired sequences)

The last-token location code uses held-out-history GroupKFold, at the fixed late layer (Qwen3-VL L28, Qwen3.5 L32).

| model | condition | native | code (fixed layer) | code (best layer) |
|---|---|---|---|---|
| Qwen3-VL | C-full | 0.41 | 0.57 | 0.60 |
| Qwen3-VL | C-teleport | **0.55** | **0.98** | 0.98 |
| Qwen3-VL | I-full | 0.35 | 0.09 | 0.22 |
| Qwen3.5 | C-full | 0.40 | 0.36 | 0.36 |
| Qwen3.5 | C-teleport | 0.39 | **0.91** | 0.91 |
| Qwen3.5 | I-full | 0.32 | 0.15 | 0.25 |

Paired differences, C-full − C-teleport (bootstrap 95% CI):
- native: Qwen3-VL −0.14 [−0.19, −0.09]; Qwen3.5 +0.01 [−0.04, +0.06];
- code: Qwen3-VL −0.41 [−0.47, −0.35]; Qwen3.5 −0.55 [−0.60, −0.49].

**Verdicts**
- J1 and J2 are supported, and more strongly than predicted: removing the motion does not hurt; it **helps**.
- The intact swap video degrades the appearance-based location code by 0.4–0.55.
- J3 holds: identical cups are at or below chance.

**Unresolved.** C-teleport differs from C-full in two ways: (a) no intermediate layouts or continuous motion; (b) a prolonged initial layout. Addendum K separates them.

### 6j. Addendum K results (n=300 paired sequences; colored cups)

Held-out-history location code at the fixed late layer, and native accuracy:

| swap period shows | Qwen3-VL code | Qwen3.5 code | Qwen3-VL native | Qwen3.5 native |
|---|---|---|---|---|
| full motion | 0.63 | 0.45 | 0.45 | 0.38 |
| steps (static intermediate layouts) | 0.54 | 0.49 | 0.49 | 0.35 |
| tele (initial layout frozen) | 0.99 | 0.92 | 0.56 | 0.39 |
| telefinal (final layout from the start) | 0.98 | 0.96 | 0.60 | 0.39 |
| blank (no cups) | 0.98 | 0.94 | **0.64** | 0.33 |

**Verdicts**
- K1 is supported: seeing intermediate layouts, with or without continuous motion, destroys the code.
- K1' is rejected.
- K2 is supported: the prolonged initial layout is not the cause.
- Paired CIs: code full − blank is −0.35 [−0.40, −0.30] for Qwen3-VL and −0.49 [−0.55, −0.44] for Qwen3.5.

**Ingredients (held-out-history probes, L28 / L32)**

| probe | full | steps | blank |
|---|---|---|---|
| ball-cup color (bound at the reveal) | 0.97–1.00 | 0.97–1.00 | 0.97–1.00 |
| final position of each color | 0.80 / 0.87 | 0.68 / 0.71 | 0.99 / 1.00 |

- The binding is intact.
- The representation of the fully visible final layout is corrupted by earlier layouts.
- The lookup (ball location) degrades further.

**Behavior: "At the very end, what color is the cup in the <pos> position?"** The answer is fully visible in the final 0.75 s.

| model | blank | tele | full | steps |
|---|---|---|---|---|
| Qwen3-VL | 1.00 | 0.87 | 0.91 | 0.70 |
| Qwen3.5 | 1.00 | 0.98 | 0.85 | 0.67 |

- The errors are almost always the color at that position in the **penultimate layout**:
  - Qwen3-VL: 1.00 (full) and 0.88 (steps) of errors;
  - Qwen3.5: 0.96 and 0.91.
- The models report a one-step-stale "present".

**Open question**
- In steps, the penultimate layout dwells 1.0 s and the final layout 0.75 s. Is the answer the most-viewed layout (dwell weighting) or a failure of recency? Addendum L tests this.
- Blank, where the final layout is also only 0.75 s, gives 1.00. So the effect requires competition from a previous layout.

**27B, Addendum J native:** C-full 0.65, C-teleport **0.79**, I-full 0.35. Paired difference −0.15 [−0.22, −0.08]. Removing motion helps 27B's answers too. The code analysis is pending.

**27B cup, det2:** D3 0.82 (was 0.80); swap sequence exactly right 0.80.

## 7. Night 3 (2026-09-24): the "stale present"

### 7a. Addendum L (dwell): Qwen3-VL / Qwen3.5, "color at position p at the very end", n=60 per cell

| D_pen → D_fin | Qwen3-VL acc / P(pen) | Qwen3.5 acc / P(pen) |
|---|---|---|
| 0.5 → 0.5 | 0.55 / 0.35 | 0.58 / 0.38 |
| 0.5 → 4.0 | 0.92 / 0.07 | 0.87 / 0.08 |
| 2.0 → 0.5 | 0.45 / 0.47 | 0.47 / 0.50 |
| 2.0 → 4.0 | 0.93 / 0.07 | 0.83 / 0.17 |

- Accuracy is driven by the final dwell, D_fin; D_pen has little effect.
- L1 is supported. L1' is not (at D_fin = 4 s, P(pen) is 0.07–0.08).

### 7b. Addendum M (video mode): m layouts after the reveal

| m | D_fin | Qwen3-VL acc / P(pen) | Qwen3.5 acc / P(pen) |
|---|---|---|---|
| 1 | 0.5 | 0.92 / 0.02 | 1.00 / 0.00 |
| 1 | 2.0 | 1.00 / 0.00 | 1.00 / 0.00 |
| 2 | 0.5 | 0.57 / 0.43 | 0.48 / 0.47 |
| 2 | 1.0 | 0.70 / 0.28 | 0.68 / 0.28 |
| 2 | 2.0 | 0.82 / 0.18 | 0.77 / 0.23 |
| 4 | 0.5 | 0.40 / 0.43 | 0.58 / 0.35 |
| 4 | 2.0 | 0.73 / 0.18 | 0.75 / 0.17 |

- When the only earlier layout is the reveal layout (m=1), the present is reported perfectly even at 0.5 s.
- **One extra static layout (m=2) halves accuracy.** Errors are almost always the immediately preceding layout: 52–59 of 55–70 errors are one step back.
- M1 is supported as "m ≥ 2 vs m = 1", not as a graded effect: m=4 ≈ m=2.
- Candidate reason: the reveal layout is visually distinct (cups lifted, ball visible). Competition arises between layouts of the SAME kind (cups down), and the model does not resolve which of them is latest.

### 7c. Addendum M: image-list mode (same frames, each a separate image with "Frame i (t s):")

| m | D_fin | Qwen3-VL video → images | Qwen3.5 video → images |
|---|---|---|---|
| 2 | 0.5 | 0.57 → **0.85** | 0.48 → **0.75** |
| 2 | 2.0 | 0.82 → 0.92 | 0.77 → 0.82 |
| 4 | 0.5 | 0.40 → 0.70 | 0.58 → 0.82 |
| 4 | 2.0 | 0.73 → 0.93 | 0.75 → 0.83 |

- Image mode reduces the stale present substantially but does not remove it: P(pen) at D_fin = 0.5 s is 0.12–0.23. M2 is partly supported.
- Group-alignment check: every final-layout onset falls on a temporal-group boundary (even frame index), so within-group blending is NOT the cause.
- Token check: the per-frame spatial grid is identical (20×28) in both modes. Video mode merges 2 frames per temporal group, so a state dwelling D s gets half as many tokens as in image mode.

**Working hypothesis (token-mass voting).**
- The reported "present" is a vote over similar-looking states weighted by the number of tokens (dwell) each received, with only a weak recency prior.
- This is consistent with L (D_fin dominates), M (image mode ≈ doubling the final state's tokens) and m=1 (the reveal layout differs in kind, since the cups are lifted, so it does not compete).

**Correction to the working hypothesis (same night).** Token-mass voting is contradicted by two results:
- (i) Addendum K "tele": the initial cups-down layout dwells 2.5–5 s against a 0.75 s final layout, yet the final layout is reported at 0.87–0.98.
- (ii) Addendum L: quadrupling D_pen changes accuracy only a little (0.55 → 0.45).

Better account, **change-order confusion**:
- With a single change (tele; m=1), the present is reported correctly.
- With ≥ 2 changes, the model often reports the state after the PENULTIMATE change (P ≈ 0.4–0.5 at short D_fin). It cannot tell which change was the latest.
- A longer final dwell (more final tokens, or image mode with explicit per-frame timestamps) tips the choice toward the true last state.

Addendum N (minimal digit/ball displays, m=1 vs m=3 changes) tests whether this generalizes.

### 7d. Addendum N: minimal displays (single digit or single ball; n=50 per cell)

- Both models are at 0.92–1.00 in every cell, including m=3 changes with D_fin = 0.25 s.
- **N1 is falsified.** The stale present is NOT a general temporal failure. A single changing item is always reported correctly.
- The effect requires multi-object layouts, where the answer depends on binding a feature (color) to a location at a specific time.

**Hypothesis: temporal illusory conjunctions.**
- Color–position bindings from different moments mix.
- Errors are colors that occupied the queried position at an earlier moment: 100% in Addendum K, against 72–79% expected by chance.

Addendum O tests object count × query type.

**27B replications**
- Addendum J code (L49, held-out history): C-full 0.85, C-teleport **1.00**, I-full ≤ 0.25. Paired difference −0.15 [−0.19, −0.11]. Native: 0.65 / 0.79 / 0.35.
- Addendum N (minimal digit/ball displays): 0.98–1.00 in all cells.

### 7e. Addendum O: minimal colored disks (n=50 per cell; m=3 changes)

Accuracy / P(pen), Qwen3-VL | Qwen3.5:

| query | n | D_fin 0.5 | D_fin 2.0 |
|---|---|---|---|
| obj ("where is the <color> disk") | 1 | 0.84/0.16 \| 1.00/0.00 | 1.00 \| 1.00 |
| obj | 2 | 0.52/0.32 \| 0.60/0.32 | 0.64 \| 0.78 |
| obj | 3 | 0.50/0.48 \| 0.48/0.46 | 0.74 \| 1.00 |
| pos ("what is in the <p> position") | 1 | 0.54/0.46 \| 0.72/0.28 | 0.74 \| 0.84 |
| pos | 2 | 0.36/0.46 \| 0.42/0.42 | 0.58 \| 0.70 |
| pos | 3 | 0.60/0.36 \| 0.76/0.24 | 0.66 \| 0.86 |

- **O1 is supported for obj queries**: n ≥ 2 drops accuracy by 0.24–0.52 relative to n=1, and errors are the penultimate state.
- **New finding: position-keyed queries fail even with one object** (0.54 / 0.72). An object-keyed query about the same information succeeds.

**Revised account.**
- The present is reported correctly only when the answer is the current value of a single, uniquely identified object's feature.
- Any lookup keyed by position, or requiring disambiguation among several objects, retrieves a temporal mixture. The penultimate state wins about 25–50% of the time at short final dwell.
- This works like **temporal illusory conjunctions**.

### 7f. Scale: 27B shows NO stale present

- Addendum L: 0.98–1.00 in every cell.
- Addendum M (video mode): 0.98–1.00, including m=4 at D_fin = 0.5 s.

The stale present is an 8–9B phenomenon (Qwen3-VL-8B, Qwen3.5-9B). It is not a universal property of video VLMs.

27B still shows motion interference in the colored shell game (Addendum J code: 0.85 full vs 1.00 teleport), so the two effects are dissociable.

A scale sweep (Qwen3-VL 2B / 4B / 8B / 32B) has been started. The models are downloading.

### 7g. Addendum P: last-frame anchoring and an image-only control

Accuracy by input: video only / video + last frame as an image / last frame image only.

| set | Qwen3-VL | Qwen3.5 |
|---|---|---|
| M m=4, D_fin 0.5 | 0.40 / 0.65 / **1.00** | 0.58 / 0.87 / **1.00** |
| M m=4, D_fin 2.0 | 0.73 / 0.82 / 1.00 | 0.75 / 0.83 / 1.00 |
| O obj n=3, D_fin 0.5 | 0.50 / 0.70 / 1.00 | 0.48 / 0.90 / 1.00 |
| O pos n=1, D_fin 0.5 | 0.54 / 0.56 / 1.00 | 0.72 / 0.96 / 1.00 |
| K full (ball question) | 0.45 / 0.45 / — | 0.38 / 0.41 / — |
| J I-full (identical) | — / 0.34 / — | — / 0.33 / — |

Note: n=2 disk cells are hard even from the single image (0.56–0.90), because of the empty slot. They are excluded from the interference claim.

**Verdicts**
- P1 fails: anchoring helps but does not restore ≥ 0.9, especially for Qwen3-VL.
- P2 fails: there is no gain on the colored shell-game ball question.
- P3 holds.

**Finding: proactive interference.**
- For the 8–9B models, the final state is perceived perfectly in isolation (1.00).
- The preceding video overrides it. Even when the clean final frame is appended as a separate image, the penultimate state from the video is still chosen in 10–44% of cases (Qwen3-VL).
- 27B shows no such interference (7f).

### 7h. Other models and the 27B shell game

**Addendum M (video), stale present**

| model | m=1, D_fin 0.5 | m=2, D_fin 0.5 | m=4, D_fin 0.5 | m=4, D_fin 2.0 |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 0.80 | 0.47 (P(pen) 0.48) | 0.47 (0.43) | 0.65 |
| LLaVA-NeXT-Video-7B | 0.20 (P(pen) 0.53) | 0.33 | 0.37 | 0.33 |

- Qwen2.5-VL replicates the Qwen3-VL / Qwen3.5 pattern.
- LLaVA is at or below chance even at m=1, and prefers the EARLIER layout (P(pen) 0.53–0.67 at m=1). An image-only control is queued to tell perception failure from interference.
- LLaVA on O is near chance in all cells.

**27B, Addendum K native (shell-game ball question)**

| full | steps | tele | telefinal | blank |
|---|---|---|---|---|
| 0.66 | 0.57 | 0.80 | 0.69 | **0.86** |

- Paired difference, full − blank: −0.20 [−0.26, −0.13].
- Intermediate cups-down layouts hurt 27B too, even though 27B has no stale present in the simple "what color is at p" task. The shell-game interference is therefore not only a stale-present effect.

### 7i. Stale present: last-token attention (Qwen3-VL-8B, Addendum M m ≥ 2, eager attention; exploratory)

The measure is the head-averaged, per-temporal-group attention from the last token, as final / (final + penultimate):
- ≈ 0.46–0.53 at most layers, for both correct and stale trials. **There is no recency preference in attention**: the final and penultimate layouts receive equal per-token attention.
- The only separation is at layer 25: correct 0.64 vs stale 0.52. The video attention mass also peaks around L22–25 (0.15–0.17).
- A per-head analysis at L20–30 is running.

### 7j. Recency heads and the scale sweep (in progress)

**Per-head last-token attention** (Qwen3-VL-8B, Addendum M, m ≥ 2; 235 correct, 105 stale trials)
- A small set of heads at layers 24–27 separates correct from stale trials:

  | head | final share, correct | final share, stale |
  |---|---|---|
  | L25H29 | 0.78 | 0.48 |
  | L24H30 | 0.79 | 0.51 |
  | L25H13 | 0.76 | 0.52 |
  | L27H26 | 0.76 | 0.57 |

- The top 8 are saved as `recency_heads_qwen3vl8b.json` (0-based layer index).
- On correct trials these heads select the latest layout; on stale trials they split evenly.
- A causal test (Addendum Q, on the independent Addendum L data) is running.

**Scale sweep, Qwen3-VL (Addendum M video, D_fin 0.5)**

| size | m=1 | m=2 | m=4 |
|---|---|---|---|
| 2B | 0.42 | 0.38 | 0.42 |
| 4B | 0.83 | 0.55 | 0.45 |
| 8B | 0.92 | 0.57 | 0.40 |
| 32B | pending | pending | pending |

Qwen3.6-27B (another family): 1.00 / 1.00 / 0.98.

**Addendum O, Qwen3-VL-2B / 4B:** stale everywhere (obj n=3 at D_fin 0.5: 0.34 / 0.50).

### 7k. Addendum Q result: the recency heads are not causal

Intervention at the last query position only, on the Addendum L data (n=480). Accuracy / P(pen) at D_fin ≤ 1 s:

| condition | acc | P(pen) |
|---|---|---|
| none | 0.600 | 0.333 |
| suppress-pen (selected heads) | 0.592 | 0.342 |
| random heads | 0.596 | 0.333 |
| suppress-final (selected heads) | 0.583 | 0.346 |

**Q1 and Q3 are falsified** (Q2 holds trivially). The L24–27 heads correlate with the outcome, but blocking their access to either layout at the answer position changes nothing. The decision is made earlier or elsewhere.

A broader exploratory intervention is running: all layers and heads, at all text positions after the video, mask the penultimate, first or final layout.

**Image-only controls (last frame alone)**
- Qwen2.5-VL-7B: 1.00 on M and on O (n = 1, 3). Its stale present is interference, as with Qwen3-VL and Qwen3.5.
- LLaVA-NeXT-Video-7B: 0.22–0.74 even from the single image. It cannot perceive this display reliably and is excluded from the stale-present analysis.

**27B with last-frame anchoring**
- O and M: ≈ 1.00.
- K full / steps: 0.71 / 0.72 (vs 0.66 / 0.57 native).
- J I-full: 0.33.

### 7l. Broad text-side masking (exploratory; Qwen3-VL-8B, Addendum L data)

Keys of one layout are masked for every head and layer, at all TEXT query positions after the video. The video tokens themselves are untouched.

Accuracy / P(pen):

| condition | D_fin ≤ 1 s | D_fin 4 s |
|---|---|---|
| none | 0.60 / 0.33 | 0.93 / 0.07 |
| mask penultimate | **0.80 / 0.075** | 0.97 / 0.01 |
| mask the layout before it (L1) | 0.55 / 0.43 | 0.89 / 0.11 |
| mask final | 0.17 / 0.64 | 0.28 / 0.58 |

- **Most stale errors arise at text-side retrieval.** When the question tokens cannot read the penultimate layout, stale answers fall from 0.33 to 0.075.
- The remaining gap to the image-only ceiling (0.80 vs 1.00) is attributable to the video encoding: final-layout tokens contaminated by earlier frames.
- Masking the final layout makes the model report the penultimate one (P(pen) 0.64). The two layouts compete for the same answer slot.
- A layer-range localization is running.

**Layer-range localization** (the penultimate layout is masked for all text positions, only within the given layers). Accuracy / P(pen) at D_fin ≤ 1 s:

| layers masked | acc | P(pen) |
|---|---|---|
| none | 0.60 | 0.33 |
| L0–8 | 0.60 | 0.33 |
| L9–17 | 0.66 | 0.23 |
| **L18–26** | **0.81** | **0.08** |
| L27–35 | 0.60 | 0.34 |

**The stale retrieval happens at the text positions in layers 18–26.** Masking these layers alone reproduces the full effect of masking every layer.

The answer-position-only intervention (7k) had no effect. So the retrieval is carried by earlier text tokens, e.g. the question's position words, and not by the final token.

### 7m. Qwen3-VL-32B: the stale present persists at scale

**Addendum M (video), m=1 / m=2 / m=4:**
- D_fin 0.5: 1.00 / 0.90 / 0.80;
- D_fin 2.0: 1.00 / 0.97 / 0.97.

**Addendum O, D_fin 0.5 (D_fin 2.0):**

| query | n=1 | n=2 | n=3 |
|---|---|---|---|
| pos | **0.46 / P(pen) 0.54** (0.68) | 0.46 (0.74) | 0.84 (0.94) |
| obj | 0.72 (0.98) | 0.56 (0.70) | 0.78 (0.96) |

- The last frame alone gives 1.00 on M and O.
- In the Qwen3-VL family, scale (2B → 32B) reduces but does NOT remove the stale present.
  - On cups, 32B is much better than 8B.
  - On minimal disks with position-keyed queries, 32B is as stale as 8B (0.46).
- Only Qwen3.6-27B (a newer generation) is free of it.
- The phenomenon spans three model generations (Qwen2.5-VL, Qwen3-VL, Qwen3.5) and sizes from 2B to 32B.

### 7n. Where along the question the stale retrieval happens, and whether CoT fixes it (Qwen3-VL-8B)

**Text-span masking of the penultimate layout, layers 18–26 only** (Addendum L data, D_fin ≤ 1 s). Accuracy / P(pen):

| span masked | acc | P(pen) |
|---|---|---|
| none | 0.60 | 0.33 |
| all text | 0.81 | 0.08 |
| last token only | 0.63 | 0.30 |
| video end up to the position word | 0.75 | 0.16 |
| after the position word | 0.73 | 0.18 |

The retrieval is distributed along the question. Each half carries about half of the effect, and the answer token alone carries little.

**CoT prompt** (describe each moment, then answer; greedy, 400 tokens). Accuracy (P(pen)), video-only value in brackets:

| cell | CoT | video only |
|---|---|---|
| O pos n=1 | **0.94 (0.06)** | [0.54] |
| O obj n=3 | 0.68 (0.16) | [0.50] |
| O pos n=3 | 0.68 (0.22) | [0.60] |
| M m=4 | 0.45 (0.23; 18/60 unparsed) | [0.40] |

The written descriptions hallucinate extra rearrangements and wrong timestamps, often running out of tokens. CoT helps for simple displays but is unreliable for multi-layout sequences.

**Qwen3.5 CoT** (video only in brackets):
- O pos n=1: 0.50 (P(pen) 0.50) [0.72];
- O obj n=3: 0.84 [0.48];
- O pos n=3: 1.00 [0.76];
- M m=4: 0.57 (0.35) [0.58].

CoT effects are inconsistent across cells and models. Some cells help, one gets worse. It is not a reliable fix.

### 7o. Qwen2.5-VL layer-range masking, and chess (Addendum R, Qwen3-VL-8B)

**Qwen2.5-VL-7B, text-side masking of the penultimate layout** (Addendum L data, D_fin ≤ 1 s). Accuracy / P(pen):

| layers masked | acc | P(pen) |
|---|---|---|
| none | 0.39 | 0.37 |
| L14–20 | 0.44 | 0.28 |
| all | 0.45 | 0.25 |
| other ranges | ≈ none | ≈ none |

This is only a partial replication. The best band is middle-late (L14–20 of 28, cf. L18–26 of 36 in Qwen3-VL), but most stale answers survive masking. For Qwen2.5 more of the effect sits in the video encoding.

**Chess (real MillionBase move windows).**
- **src queries are invalid.** The correct answer is "empty", and Qwen3-VL almost never answers "empty", even from the final image alone (0.04–0.10). This is an option bias.
- **dst queries.** m=1 and m=3 are similar, so R2 is not supported. R1 is borderline:

  | input | acc | P(pen) |
  |---|---|---|
  | video | 0.66–0.74 | 0.20–0.26 |
  | image only | 0.82–0.90 | 0.00–0.12 |

- **Amendment (exploratory, R').** Regenerate dst queries restricted to CAPTURES, so that both the penultimate and final contents are pieces, avoiding the "empty" bias.

### 7p. Chess, captures only (R'; Qwen3-VL-8B; n=100 per cell)

Question: "what is on square <capture square> at the end?" Options: the capturing piece (correct), the captured piece (penultimate), and another piece.

| cell | image only | video | P(pen) with video |
|---|---|---|---|
| m=1, D_fin 0.5 | 0.86 | **0.20** | **0.78** |
| m=1, D_fin 2.0 | 0.88 | 0.48 | 0.50 |
| m=3, D_fin 0.5 | 0.81 | 0.36 | 0.59 |
| m=3, D_fin 2.0 | 0.90 | 0.44 | 0.54 |

**On real game sequences, the model mostly reports the captured piece as still standing on the square.** From the final image alone it identifies the capturing piece.

- This is the strongest stale-present effect so far: a gap of 0.4–0.66.
- The effect appears already at m=1. There the only previous state is a board of the same kind, which fits the "same-kind competition" account; in Addendum M, the m=1 predecessor was the visually distinct reveal.

**Chess captures, other models.** Video accuracy vs image-only, with P(pen) for video:

| model | m=1, D_fin 0.5 | m=3, D_fin 0.5 | m=3, D_fin 2.0 |
|---|---|---|---|
| Qwen3.5-9B | 0.90 vs 0.90 | **0.75** vs 0.94 (P(pen) 0.24) | 0.80 vs 0.97 |
| Qwen3-VL-32B | **0.63** (P(pen) 0.35) | **0.72** (0.27) | 0.94 |

- The 32B image-only run and the 27B run are pending.
- Non-capture dst, 32B / 27B: video 0.84–0.88 / 0.92–1.00; 32B image-only 0.96–1.00.
- The effect in real chess is strongest for Qwen3-VL-8B, clear for Qwen3-VL-32B at short final dwell, and milder for Qwen3.5.

**Chess, final results**
- Qwen3-VL-32B, captures, image-only: 0.96–0.99. The video gap at D_fin 0.5 is therefore 0.27–0.33 (video 0.63 / 0.72).
- Qwen3.6-27B, captures, video: 0.99–1.00. No effect.
- Qwen3.5-9B, non-capture dst, video: 0.94–0.98.

Summary of the chess result: a clear stale present for Qwen3-VL 8B and 32B, a mild one for Qwen3.5-9B (only m=3), and none for Qwen3.6-27B.

## 8. Training-free fixes (Addendum S)

The change detector was amended before any outcome-driven tuning. The original mean-difference threshold missed single-piece chess moves, and PCD there equaled native. It now marks a new state when > 0.1% of pixels change by > 40 levels. All results below use the amended detector.

**Qwen3-VL-8B.** Accuracy (P(pen)); the image-only column is the ceiling from 7g/7p.

| cell | native | CPM (L18–26) | PCD α=1 | PCD α=0.5 | image only |
|---|---|---|---|---|---|
| chess m=1, D_fin 0.5 | 0.20 (0.78) | 0.72 (0.25) | **0.84 (0.04)** | 0.53 | 0.86 |
| chess m=1, D_fin 2.0 | 0.48–0.52 | 0.83 | 0.90 | 0.76 | 0.88 |
| chess m=3, D_fin 0.5 | 0.36 (0.59) | 0.66 (0.28) | 0.74 (0.12) | 0.44 | 0.81 |
| chess m=3, D_fin 2.0 | 0.45 | 0.88 | 0.89 | 0.68 | 0.90 |
| O pos n=1, D_fin 0.5 | 0.54 | 0.80 | 0.78 | 0.62 | 1.00 |
| O pos n=3, D_fin 0.5 | 0.60 | 0.88 | 0.92 | 0.80 | 1.00 |
| O obj n=3, D_fin 0.5 | 0.50 | **0.94** | **0.94** | 0.84 | 1.00 |
| M m=4, D_fin 0.5 | 0.38–0.40 | **0.90** | 0.87 | 0.60 | 1.00 |
| M m=4, D_fin 1.0 | 0.70 | 0.97 | 0.98 | 0.90 | 1.00 |
| N digit / ball (no-harm control) | 0.96–1.00 | 1.00 | 1.00 | 0.96–1.00 | — |

**Verdicts**
- S1 is partly supported: chess improves by +0.3 to +0.5, but P(pen) stays at 0.25–0.28, above the 0.15 threshold.
- S2 is supported for Qwen3-VL-8B. Other models are running.
- S3 is supported: no harm on the single-item controls.

**Both fixes bring "end-state" questions close to the image-only ceiling.** Their value beyond "just look at the last frame" must be shown on tasks that need both history and present, such as the colored shell game (next).

**PCD, other models.** Native → PCD α=1; P(pen) in brackets.

| cell | Qwen3.5-9B | Qwen3-VL-32B |
|---|---|---|
| chess m=1, D_fin 0.5 | 0.90 → 0.98 | 0.63 (0.35) → **0.98 (0.00)** |
| chess m=3, D_fin 0.5 | 0.75 (0.24) → 0.94 (0.00) | 0.72 (0.27) → **0.98** |
| O pos n=1, D_fin 0.5 | 0.72 → **1.00** | 0.46 (0.54) → **0.98** |
| O obj n=3, D_fin 0.5 | 0.48 (0.46) → **0.96** | 0.78 → 0.96 |
| M m=4, D_fin 0.5 | 0.58 → 0.88 | 0.80 → 0.98 |
| N digit / ball | 1.00 → 1.00 | 0.92–1.00 → 1.00 |

S2 is supported for all three Qwen models. PCD brings end-state questions to the image-only ceiling (≈ 0.94–1.00) and never hurts the controls.

**History + present (colored shell game, Addendum K data; identical-cup control).** Neither fix helps:

| model / set | native | PCD 1.0 | CPM | CPM-ends |
|---|---|---|---|---|
| Qwen3-VL-8B, K full | 0.45 | 0.43 | 0.39 | 0.43 |
| Qwen3-VL-8B, K steps | 0.49 | 0.52 | 0.46 | 0.51 |
| Qwen3-VL-8B, I-full (control) | 0.35 | 0.26 | 0.35 | 0.34 |

- CPM-ends keeps the first static (reveal) run and the final state, masking only the middle, at L18–26 for text queries.
- The shell-game deficit is therefore NOT the stale-present retrieval: masking intermediate layouts at the text side does not recover the "blank" advantage (0.64).
- Consistent with 6j, the damage from intermediate layouts is already in the video-token representation, where the final-layout code is at 0.80 vs 0.99.
- **Scope of the fixes:** they correct present-state readout. They do not repair the appearance-based location lookup that is corrupted during encoding.

### 8b. Addendum T (history + present chess questions): both predictions fail, and PCD is harmful

The question is "At the end, what is on the square where the <captured piece> stood at the beginning?" Accuracy (P(stale = the named captured piece)):

| model / cell | native video | PCD α=0.5 | PCD α=1 | image only |
|---|---|---|---|---|
| Qwen3-VL-8B, m=1, D_fin 0.5 | 0.67 (0.00) | 0.68 | **0.30 (0.38)** | 0.27 (0.62) |
| Qwen3-VL-8B, m=1, D_fin 2.0 | 0.78 (0.00) | 0.78 | 0.28 (0.33) | 0.33 (0.52) |
| Qwen3-VL-8B, m=3, D_fin 0.5 | 0.80 (0.00) | 0.82 | 0.42 (0.42) | 0.65 (0.17) |
| Qwen3-VL-8B, m=3, D_fin 2.0 | 0.82 (0.00) | 0.92 | **0.05 (0.92)** | 0.63 (0.10) |
| Qwen3-VL-32B | pending | pending | pending | 0.00 (0.57–0.78) |
| Qwen3.5-9B | pending | pending | pending | 0.08–0.17 (0.30–0.48) |

**T1 fails.** When the question names the captured piece, the video answer is NEVER the stale piece: P = 0.00 in all cells. Naming the piece seems to cue that it has left the square.

**T2 fails, and PCD α=1 is harmful.** Here the truncated "past" video is not a proxy for the stale answer: the question itself refers to the past. Subtracting it pushes probability onto the captured piece.

**Scope of PCD.** It applies to pure present-state questions only ("what is at X at the end"). It must not be used when the question references history. α=0.5 is neutral to slightly positive here.

The image-only control shows the opposite bias: without history, models pick the named piece (32B: 0.57–0.78).

### 8c. Non-Qwen-vision models (InternVL3.5-8B: InternViT + Qwen3 LLM; LLaVA-OneVision-7B: SigLIP + Qwen2 LLM)

**Image-only controls (last frame)**
- Both models: 1.00 on M and on O (n = 1, 3).
- Chess captures: InternVL 0.78–0.83; LLaVA-OV 0.60–0.73.

Native video → PCD α=1:

| cell | InternVL3.5-8B | LLaVA-OV-7B |
|---|---|---|
| M m=4, D_fin 0.5 | **0.57 (P(pen) 0.30) → 0.87** | **0.68 (0.27) → 0.95** |
| M m=4, D_fin 1.0 | 0.83 → 0.92 | 0.68 → 0.93 |
| O obj n=3, D_fin 0.5 | 0.96 → 0.96 | 0.70 (0.20) → 0.86 |
| O pos n=1, D_fin 0.5 | 0.74 (0.26) → 1.00 | 0.64 (0.36) → 0.72 |
| chess m=1 / m=3, D_fin 0.5 | 0.55 / 0.48 → 0.76 / 0.73 | 0.62 / 0.68 → 0.64 / 0.54 |
| N digit / ball (control) | 1.00 → 1.00 | 0.98 / 0.58 → 1.00 / 0.88 |

- **The stale present replicates with non-Qwen vision encoders**: M and O show image-only 1.00 vs video 0.57–0.74.
- **PCD fixes it** on M and O for both models.
- On chess, LLaVA-OV shows no video-vs-image gap (0.62–0.68 vs 0.60–0.73), so there is nothing to fix. InternVL shows a gap of about 0.3, closed by PCD to 0.73–0.76.

**Addendum T, Qwen3.5-9B**
- Native P(stale) is 0.18–0.28 (vs 0 for Qwen3-VL-8B).
- PCD α=1 helps at m=1, D_fin 0.5 (0.50 → 0.82).
- m=3 cells are low for all conditions (0.2–0.32).
- 32B is pending.

**Addendum T, Qwen3-VL-32B.** Native accuracy (P(named captured piece)) → PCD α=0.5 / α=1:
- m=1, D_fin 0.5: 0.03 (0.82) → 0.43 / 0.62;
- m=1, D_fin 2.0: 0.05 (0.80) → 0.32 / 0.50;
- m=3, D_fin 0.5 and 2.0: 0.00 (0.95) → ≤ 0.02 / 0.13–0.18.

32B almost always answers the named captured piece. It does the same from the final image alone (0.57–0.78), so this is at least partly name priming rather than a stale present. PCD α=1 partially counteracts it at m=1.

Addendum T is model-dependent and confounded by naming the piece in the question. **It is not usable as evidence for or against PCD's value beyond the last frame.** A cleaner history + present design is needed, one that does not name the stale answer.

### 8d. Addendum U: history-referenced present questions (the stale answer is never named)

**U-chess: "what is on the square where the first move of the video ended?"** Exchange windows, 100 per D_fin.

| model | image only | native (P(stale)) | PCD α=0.5 | PCD α=1 |
|---|---|---|---|---|
| Qwen3-VL-8B | 0.36 / 0.38 | 0.46 (0.40) / 0.51 (0.37) | 0.59 / 0.71 | **0.65 / 0.82** |
| Qwen3-VL-32B | 0.44 / 0.46 | 0.52 (0.47) / 0.50 (0.44) | 0.72 / 0.73 | **0.89 / 0.88** |
| Qwen3.5-9B | 0.50 / 0.41 | 0.67 (0.32) / 0.70 (0.27) | 0.84 / 0.91 | 0.80 / 0.88 |
| InternVL3.5-8B | 0.34 / 0.39 | 0.59 (0.31) / 0.63 (0.31) | 0.81 / 0.77 | 0.75 / 0.78 |
| LLaVA-OV-7B | 0.40 / 0.41 | 0.56 (0.28) / 0.38 (0.41) | 0.53 / 0.43 | 0.51 / 0.45 |

(Pairs are D_fin 0.5 / 2.0.)

- **U1 holds**: P(stale) is 0.27–0.47 in all five models.
- **U2 holds for 4 of 5 models** on chess: PCD α=1 is +0.13 to +0.38 over native and +0.30 to +0.45 over image-only. LLaVA-OV gains nothing.
- **This is the missing evidence.** The question requires the history (image-only is near chance). Native answers are contaminated by the previous state, and PCD corrects them. PCD therefore adds value beyond "look at the last frame".

**U-disks: "what color is at the position where the <c> disk was at the beginning?"** This fails for every model:
- native 0.23–0.54;
- image-only guessing 0.11–0.59;
- PCD gives no gain.

The models do not retrieve the reference position from the first layout (position-keyed retrieval was already the hardest case in 7e). The failure is in resolving the history reference, before any present-state readout.

### 8e. Addendum V: detector-free PCD-Δ (the past is the video minus its last Δ s)

Accuracy (Qwen3-VL-8B | InternVL3.5-8B):

| cell | native | PCD (detector) | PCD-Δ0.5 | PCD-Δ1.0 |
|---|---|---|---|---|
| chess m=1, D_fin 0.5 | 0.20 \| 0.55 | 0.84 \| 0.76 | 0.84 \| 0.76 | 0.72 \| 0.74 |
| chess m=3, D_fin 0.5 | 0.36 \| 0.48 | 0.74 \| 0.73 | 0.74 \| 0.73 | 0.65 \| 0.72 |
| O obj n=3, D_fin 0.5 | 0.50 \| 0.96 | 0.94 \| 0.96 | 0.94 \| 0.96 | 0.78 \| 0.96 |
| M m=4, D_fin 0.5 | 0.40 \| 0.57 | 0.87 \| 0.87 | 0.87 \| 0.87 | 0.75 \| 0.85 |
| M m=4, D_fin 1.0 | 0.70 \| 0.83 | 0.98 \| 0.92 | 0.93 \| 0.92 | 0.98 \| 0.92 |
| U-chess, D_fin 0.5 | 0.46 \| 0.59 | 0.65 \| 0.75 | 0.65 \| 0.75 | 0.62 \| 0.83 |
| **U-chess, D_fin 2.0** | 0.51 \| 0.63 | 0.82 \| 0.78 | **0.44 \| 0.35** | **0.53 \| 0.47** |

- With Δ = D_fin, PCD-Δ equals detector PCD.
- With Δ > D_fin (Δ1.0 at D_fin 0.5), it keeps most of the gain: about 75% for Qwen3-VL and about 95% for InternVL.
- **With Δ < D_fin, it hurts, and falls below native.** The "past" clip then still contains the present state, and the contrast subtracts the correct answer.
- V1 is partly supported. PCD needs a reasonable estimate of when the current state began. For real footage this needs a robust (e.g. feature-based) state-change detector, not a fixed window.

### 8f. Addendum W: PCD under simulated camera noise (σ=6 noise, ±2 px jitter, ±3% flicker)

Accuracy (P(stale)), Qwen3-VL-8B | InternVL3.5-8B:

| cell | native (noisy) | PCD, pixel detector | PCD, motion-compensated detector |
|---|---|---|---|
| chess m=1, D_fin 0.5 | 0.39 (0.57) \| 0.53 | 0.39 \| 0.63 | **0.76 (0.09) \| 0.71** |
| chess m=3, D_fin 0.5 | 0.42 \| 0.57 | 0.23 \| 0.64 | **0.67 \| 0.71** |
| M m=4, D_fin 0.5 | 0.38 (0.50) \| 0.53 | 0.28 \| 0.92 | **0.92 (0.02) \| 0.93** |
| M m=4, D_fin 1.0 | 0.75 \| 0.82 | 0.20 \| 0.95 | **0.97 \| 0.93** |
| U-chess, D_fin 0.5 | 0.46 \| 0.50 | 0.28 \| 0.67 | **0.61 \| 0.72** |
| U-chess, D_fin 2.0 | 0.46 \| 0.62 | 0.35 \| **0.27** | **0.61 \| 0.72** |

- The pixel detector agrees with the clean t_c on only 19 of 520 noisy videos.
- **W1 holds**: PCD with the pixel detector collapses under noise. For Qwen3-VL it is at or below native everywhere, and it drops to 0.27 for InternVL on U-chess at D_fin 2.0.
- **W2 holds**: PCD with the motion-compensated detector recovers the clean-video gains on every cell and both models, including the history-referenced U-chess questions.
- The PCD pipeline (a motion-compensated change detector plus a present-contrastive decode) is thus robust to realistic sensor and camera noise at this level. Real footage remains untested.

### 8g. Non-Qwen LLM backbones (exploratory, not preregistered; 2026-09-27)

Models: Gemma-3-12B-it, Idefics3-8B-Llama3, InternVL3.5-GPT-OSS-20B-A4B-Preview. Frames are given as a numbered image list (≤32 frames, last frame kept). Queue: `scripts/run_tracking_algo_queue_nonqwen.sh`. Logs: `logs/{anchor_imgonly,chesscap_*_img,fix_pcd,fixU,fixU_img}_<model>.log`.

Each cell is image-only → native (P(stale)) → PCD α=1.

**Gemma-3-12B**
- chess m=1, D 0.5: 0.95 → 0.61 (0.37) → 0.93
- M m=4, D 0.5: 1.00 → 0.30 → 0.73
- O obj n=3, D 0.5: 1.00 → 0.58 → 0.82
- U-chess: 0.39/0.45 → 0.45/0.45 → 0.74/0.64

**Idefics3-8B**
- The chess board is not perceived (image-only 0.43–0.52), so the chess cells are uninformative.
- M: 1.00 → 0.37 → 0.75
- O obj n=3: 0.86 → 0.36 → 0.92
- **PCD hurts the N_ball control: 0.44 → 0.04.**

**InternVL-GPT-OSS**
- Near-immune on the synthetic cells: M 0.97, O obj 0.90–1.00.
- One exception: O pos n=1, D 0.5 drops to 0.48 (image-only 1.00, PCD 1.00).
- Strong stale present on chess: 0.70–0.85 → 0.39–0.56 (0.43–0.57) → 0.68–0.79.
- U-chess: 0.31/0.40 → 0.49/0.48 → 0.67/0.58.

**U-disks** fails for all three models (as with Qwen).

**Takeaways**
- The stale present is not specific to Qwen LLMs, nor to Qwen's 2-frame temporal token merging.
- PCD transfers to the non-Qwen models, but with a control failure on Idefics3.

### 8h. Stats, Addendum Y (layer masking) and Addendum Z (real footage) (2026-09-28)

**Stats.** Bootstrap CIs and McNemar/Holm for every §5–§6 cell are in `stats_ci.md` (`scripts/tracking_algo_stats.py`).

**Idefics3 N_ball.** The PCD failure is a degenerate control: the model answers C on 100% of items under every input. PCD amplifies ~0.1-nat noise.

**Y (`tracking_algo_layermask.py`; n=320: M m4 f0.5/f1.0 + chess m1/m3 f0.5).** Masking text→image attention to the penultimate state.

Gemma-3-12B:
- no mask 0.547 → all layers 0.709 (McNemar 54:2): **Y1 holds**.
- L18–23 alone 0.678 (81% of the gain): **Y2 holds**. Relative depth 0.38–0.48, vs Qwen 0.50–0.72.
- 8 global layers alone 0.700; 40 sliding-window layers alone 0.603.
- Masking the final state: 0.138. Masking an earlier state: −0.03. **Y3 holds.**

Idefics3-8B:
- all layers 0.45 → 0.50 (29:13, p=0.02): Y1 fails.
- Hypothesis: the stale state enters at the encoding stage.

**Z (`tracking_algo_real.py`; Perception Test videos in MVBench perception.zip; 79 letter-order + 23 bag items; 2 fps, 448 px).**

Letters, image-only − native, with 95% CI:

| Model | Difference |
|---|---|
| Qwen3-VL-8B | +0.03 |
| Qwen3-VL-32B | −0.05 |
| Qwen3.5-9B | 0.00 |
| InternVL3.5-8B | +0.06 |
| LLaVA-OV-7B | **+0.14 [0.03, 0.25]** |
| Gemma-3-12B | +0.06 |
| Pooled | **+0.04 [−0.00, 0.08]** |

- Z0 holds (image-only 0.63–0.73). **Z1 fails.**
- Z2: the stale option takes 0.69 of native errors (p=6e-4), but also 0.61 of image-only errors, so it is a confusable distractor. Not diagnostic.
- Bag: native is far better than image-only (not single-frame decidable).
- PCD fails on real footage: the robust detector puts t_c at the median 98% of the video, because hands and camera move.
- Per the prereg, the remaining 142 videos are not fetched.

**Scheduling.** GPUs 0/1 idled after the Y runs. The queue tails were replaced by `run_tracking_algo_queue_rest2.sh`; the running dumps were untouched. The X dumps (gemma3_12b, internvl35_8b, llava_ov7b, internvl_gptoss) are still running. Addendum X is written up after they finish.

### 8i. Real-footage follow-ups AA–AF (2026-09-28)

- **AA** (last change → 0.5 s / 2 s / full, crop view, 6 models): pooled image-only − native +0.011 [−0.030, +0.051]; AA2 +0.006. Fails. Dwell length was not why Z failed. Summary: `real2_summary.md`.
- **AB/AC** (real vs synthetic tiles, natural/splice/static timelines): on synthetic tiles the position question shows gaps of +0.13 to +0.49 on the 7–12B models; 32B is at ceiling. Real-frame gaps are small and mixed. Real shuffles are multi-step: of 74 reconstructable items, 8 have 1 change, 24 have 2 and 42 have 3–13. The Z "stale option" (the beginning order) is the true penultimate state in only 10 of 73 items.
- **AD** (synthetic tiles, m×fps×dur, 5 models, n=67; 32B not run): main effects on the gap:
  - m 1→4: +0.24 (pos and order);
  - dur 1→4 s: +0.07;
  - fps: ≈0.
  - P(stale) follows the same pattern. `synthword2_summary.md`.
- **AE** (question targets the reconstructed penultimate state, 6 models):
  - pos D=0.5: +0.123 [+0.076, +0.169];
  - penultimate share of errors: −0.024 (ns).
  - Exploratory: ≤2 changes +0.22; held ≥2 s +0.17.
  - `real4_summary.md`.
- **AF** (frozen = last frame repeated through the video path; preregistered before running):
  - history effect (frozen − native), pos D=0.5: +0.097 [+0.051, +0.144]. Just under the 0.10 threshold, so AF1 formally fails.
  - pipeline effect: +0.025 (ns). AF2: −0.066 (ns).
  - P(pen) rises frozen → native only for InternVL3.5 (0.18 → 0.31) and LLaVA-OV (0.21 → 0.31), and weakly for 32B (0.21 → 0.26).
  - `real5_summary.md`.
- **Ops:** the first 32B AB/AC launches OOMed next to AF on GPU0 (their DONE lines in ab_done/ac_done are false; a note was appended). The abc and ae queues were killed, and all 32B runs were redone with `run_tracking_algo_queue_32b.sh` on GPUs 1,2 (q32b_done.txt).
- **AG** (rev. 1, recorded before running). Checking the stimuli showed that most AE intermediate "states" are transit orders (hands carrying letters). With resting states (letter centroids still across adjacent 1-fps samples), 45/73 items have exactly one resting prior state, 7 have two and 22 have none. AG therefore uses m=1: the prior resting state held 4 s, then the final state 0.5 s, at 2 fps; real frames vs rendered tiles; held vs frozen vs image-only; 52 items.
  - Position questions:
    - real: frozen − held +0.096 [+0.048, +0.144] (AG1 formally just misses 0.10); P(stale) held − frozen +0.103 [+0.054, +0.151] (AG2 holds).
    - syn: +0.131 and +0.106.
    - domain difference: +0.035 (ns).
  - Order questions: ≈0 in both domains.
  - `real6_summary.md`.

### 8k. Addendum AH: layer masking on the AG real clips (2026-09-28)
- Prereg in PREREG.md (AH) before running. `scripts/tracking_algo_real7.py`, `run_tracking_algo_queue_ah.sh`, stats `tracking_algo_real7_stats.py` → `real7_summary.md`.
- Sanity: `none` answers agree 100% with AG held answers (all 3 models, both domains).
- Real, position, P(stale) none → prior_all: Qwen3-VL-8B 0.404 → 0.250 (9:1, p=0.02; AH1 holds), LLaVA-OV 0.423 → 0.212 (12:1, p=0.003), Gemma 0.346 → 0.250 (6:1, p=0.13).
- Qwen bands: L9–17 −0.115, L18–26 −0.096 (62% of prior_all; AH2 ≥50% met but not the best band → AH2 fails), L0–8/L27–35 0.
- LLaVA-OV: L14–20 alone −0.19 (11:1); other bands small. Syn domain same band (0.48 → 0.21).
- Gemma: prior_global 0.346 → 0.231 (6:0, p=0.03); L12–17 best band.
- fin_all collapses toward stale in all models (AH3 holds).
