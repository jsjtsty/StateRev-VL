# Universal Semantic Transition Filter (USTF) v1 Handoff

Date: 2026-09-21

## Objective

Implement a single transition/observation core shared across Shell, Chess,
and (by interface only) any future task C, and across Qwen/LLaVA backbones,
trained without event labels, task ids, or handwritten transition rules.
This change is implementation + unit tests + a real tiny-scale sanity gate
only. No formal/large training suite was run, and per the tiny-gate result
below, none should be started yet (see "Tiny Gate Result" and "Recommended
Next Step").

## Implementation

```text
staterev/ustf.py      core model, filter recursion, loss, training loop,
                       task-C adapt interface, tiny-gate runner
scripts/ustf.py        CLI: tiny-gate / param-report / joint-smoke / task-c-demo
tests/test_ustf.py      12 unit tests (synthetic data only, no cache dependency)
```

`staterev/psf.py` was **not modified**. USTF reuses `Trajectory`,
`TextFeatureStore`, `ShellAdapter`, `ChessAdapter` from it verbatim -- those
already match the required interface (frozen multi-layer hidden, target
text, candidate texts, initial state id, per-step GT state id,
trajectory/task/model id) and already read from real caches keyed by
`sha256(text)`, so no new dataset adapter or new VLM forward pass was
introduced. `tests/test_psf.py` (8 tests) still passes unchanged.

## 1. Actual Model Structure

```text
BackboneProjector (per backbone): nn.Linear(4096, 128), full linear, no
  forced unit-norm. Reused for BOTH visual hidden taps and candidate/target
  text features -- this is the one deliberate deviation from staterev.psf's
  PSF family, which keeps visual and state-text projectors separate. Both
  live in the same backbone's native 4096-dim space, and every downstream
  comparison only ever happens inside one backbone's projected space, so
  merging them roughly halves the parameter cost per backbone and keeps the
  total well inside the requested 1-2M budget.

USTFCore (shared, backbone- and task-agnostic):
  layer_mix_change, layer_mix_obs   -- two 4-way softmax weights over the
                                        fractional-depth taps from
                                        staterev.psf.tap_layers
  change_encoder   Linear(4*128,128) -> GELU -> Linear(128,128)
                     v_t = f(H_t, H_{t-1}, H_t-H_{t-1}, target)
  transition_scorer Linear(5*128,128) -> GELU -> Linear(128,1)
                     score(i,j) = F(v_t, c_i, c_j, c_j-c_i, target), vectorized
                     over the full K x K grid, softmax over j -> T_t(i,j)
  observation_scorer Linear(3*128,128) -> GELU -> Linear(128,1)
                     O_t(j) = G(H_t, c_j, target), softmax -> p_obs

Recursive filter (staterev/ustf.py:USTF.forward_trajectory):
  p_prior = belief @ T_t
  p_obs   = softmax(O_t)
  p_t     = softmax(log(p_prior) + log(p_obs))     # normalized product,
                                                     # not a fixed 0.5 blend
  belief_{t+1} = p_t                                 # fully free recursion
```

No task id, event label, or fixed class index ever reaches `USTFCore`. `K`
(candidate count) is read off the input each call: 3 for Shell, 65 for
Chess, arbitrary for anything else.

Loss (`ustf_loss`): `L = L_transition + 0.5*L_observation + L_rollout`,
exactly as specified --

- `L_transition = -log T_t[S_{t-1}, S_t]` (teacher-forced indices; `T_t`
  itself never depends on belief, so this needs no separate forward pass)
- `L_observation = CE(p_obs, S_t)`
- `L_rollout = CE(p_t, S_t)`, read off the fully free recursive `posterior`

## 2. Parameter Count

Single backbone (Qwen only), `outputs/ustf_v1/parameter_counts_qwen.json`:

| component | params |
|---|---:|
| per-backbone projector (visual + state-text, shared module) | 524,416 |
| visual change encoder | 82,176 |
| transition scorer | 82,177 |
| observation scorer | 49,409 |
| layer-mixture weights | 8 |
| **shared core total** | **213,770** |
| **total trainable** | **738,186** |
| fraction of 7B | 0.0105% |
| fraction of 8B | 0.0092% |

Two backbones (Qwen + LLaVA), `outputs/ustf_v1/parameter_counts_qwen_llava.json`:

| component | params |
|---|---:|
| per-backbone projector x2 | 1,048,832 |
| shared core (unchanged) | 213,770 |
| **total trainable** | **1,262,602** |
| fraction of 7B | 0.0180% |

Both are inside the requested 1-2M budget and far under 0.1% of a 7B/8B VLM.
The "state-text projector" line item the spec asks for is numerically
identical to the visual projector because it is the same module (see
Structure section) -- reported separately in the JSON only to match the
requested report shape.

## 3. Tiny Gate Result -- **FAIL (Chess); PASS (Shell)**

Run: `python scripts/ustf.py tiny-gate --out outputs/ustf_v1` (Shell/Qwen 5
trajectories, Chess/Qwen discovery 10 games, 400 epochs, lr=5e-4,
weight_decay=0, grad-clip=1.0, full report at
`outputs/ustf_v1/tiny_gate/tiny_gate_report.json`).

| task | transition acc | rollout acc | observation acc | pre-train obs acc | gate needs |
|---|---:|---:|---:|---:|---|
| Shell (K=3) | 0.96 | **1.00** | 0.76 | 0.40 | >=0.95 trans/rollout, obs>=0.5 & +0.10 |
| Chess (K=65) | 0.49 | 0.49 | 0.49 | 0.00 | same |

Candidate-order permutation invariance: **1.0** (all trajectories, both
tasks) -- see item 5.

```json
"gate": {
  "transition_pass": false, "rollout_pass": false,
  "observation_pass": false, "permutation_pass": true,
  "overall_pass": false
}
```

Shell alone passes every criterion cleanly (100% free-rollout accuracy on
its own 5 overfit trajectories). Chess plateaus around 49% on all three
metrics after 400 epochs and stops improving (dev_rollout_accuracy flat at
0.60 combined for epochs 140-400; loss still inching down but accuracy no
longer moving). 49% is far above chance for K=65 (1.5%), so the shared core
is learning real signal on Chess, not memorizing nothing -- but it is far
below the 95% overfit bar the spec sets for a tiny gate.

**Per the spec's own stop condition ("若 tiny gate 失败，停止，不进入更大训练"),
formal/larger USTF training should not be started yet.** See Recommended
Next Step.

### Diagnosis

Three optimizer configurations were tried on the real Chess/Qwen 10-game
slice before settling on the reported one (see `git log` -- none of the
intermediate runs are committed, only the final config):

1. lr=2e-3, weight_decay=1e-4, clip_norm=1.0, 60 epochs: chess stuck ~0.47.
2. lr=2e-3, weight_decay=0, clip_norm=5.0, 60 epochs: unstable, both tasks
   regressed (shell dropped to 0.40).
3. lr=5e-4, weight_decay=0, clip_norm=1.0, 400 epochs (**reported above**):
   smooth monotonic loss decrease, shell converges to 100%, chess plateaus
   at ~49-58% around epoch 140 and does not move further through epoch 400.

This looks like an optimization/capacity plateau specific to Chess's K=65
pairwise scoring, not a training-loop bug (permutation invariance holds
throughout; Shell on the identical code path reaches 100%). It is also
consistent with pre-existing evidence in this repo that the Chess/Qwen
hidden cache carries a comparatively weak raw signal for exact-square
identity: `outputs/metbench_chess/single_piece_tracking_pilot_qwen_v1/PILOT_REPORT.md`
reports a dedicated destination-square logistic-regression probe (fit on
the full 200-game discovery split, not just 10 games) reaching only 28.9%
row-level validation accuracy, with joint (src AND dst) move decoding at
0.3%. USTF's observation scorer has to recover state identity from the same
underlying hidden representation through a compatibility function (H_t vs.
c_j) rather than a dedicated per-square classification head, which trades
per-task capacity for cross-task sharing -- that tradeoff is doing real work
here: on only 10 games it has not yet resolved the K=65 space to 95%.

## 4. Is the Transition Scorer Shareable?

Yes, architecturally verified, not just asserted:

- `test_cross_backbone_shared_core_separate_projectors` and
  `test_variable_k_joint_task_forward` (`tests/test_ustf.py`) confirm the
  identical `USTFCore` instance (same `transition_scorer`/
  `observation_scorer`/`change_encoder` object, same parameter count) is
  called for Shell (K=3) and Chess (K=65) and for Qwen and LLaVA, with no
  branching on task or backbone inside `USTFCore`.
- `scripts/ustf.py joint-smoke` (`outputs/ustf_v1/joint_forward_smoke.json`)
  ran real forward passes -- untrained model, so accuracy numbers there are
  not meaningful, only shape/wiring -- across all four (task, backbone)
  combinations against the real Shell/Chess/Qwen/LLaVA caches and confirmed
  `single_shared_core_module: true` and
  `per_backbone_projectors_are_distinct_modules: true`.
- `adapt_to_new_task` asserts `sum(p.numel() for p in model.core.parameters())`
  is byte-identical before and after every mode (`zero-shot`, `few-shot`,
  `from-scratch`) -- Task C never grows the core (see item 6 of the spec /
  section 11 below).

Whether it transfers *well* (i.e. generalizes, not just shares weights) is
untested -- the tiny gate above only established learnability on Shell, not
Chess, so no transfer claim should be made yet.

## 5. Candidate-Order Permutation

**Robust.** `permutation_invariance_check` reindexes a random permutation of
`candidate_texts` (and remaps `state_ids`/`initial_state_id` to match),
reruns the forward pass, and maps the permuted `prior`/`posterior` back to
the original ordering with the inverse permutation. This is an exact
architectural property (broadcasting over `states[i]`/`states[j]` has no
positional dependence once candidate identity is tracked), verified both by
a dedicated unit test on an untrained model (`test_candidate_permutation_invariance`)
and by the tiny-gate run on the **trained** model across all 15 real
trajectories: `permutation_invariant_fraction: 1.0`, zero failures.

## 6. Do Qwen and LLaVA Share One Core?

Yes, both by construction and by a real forward-pass smoke test. `USTFCore`
is instantiated once per `USTF` model regardless of how many backbones are
registered (`self.core = USTFCore(...)`, no per-backbone branch). Only
`BackboneProjector` is per-backbone. `joint-smoke` ran real forward passes
against the actual cached hidden states for both Qwen and LLaVA on both
Shell and Chess (`outputs/ustf_v1/joint_forward_smoke.json`) and confirmed
identical core-module identity and correctly-shaped per-backbone outputs
(K=3 for Shell, K=65 for Chess, independent of backbone). No joint Qwen+LLaVA
*training* was run -- only forward-pass wiring was checked, consistent with
the "smoke, not formal suite" scope for this change.

## 7. Worth Entering Bounded Validation?

**Not yet, not as-is.** The tiny gate is a hard go/no-go per the spec, and
it fails on Chess. Proceeding to bounded validation (let alone formal
training) now would risk reporting numbers built on an architecture/
optimization configuration that has not demonstrated it can even memorize
its own 10-game overfit set for the harder task. Shell alone already passes
cleanly, so the Shell-only path is technically gate-clear, but the spec's
gate is explicitly joint ("Shell/Qwen 5条 + Chess/Qwen 10局"), and a
Shell-only pass does not validate the shared core against the K=65 regime
that Chess (and any future large-K task C) will need.

Suggested before re-attempting the tiny gate (none of this has been run):

1. Increase `hidden` width in `USTFCore`'s three MLPs (currently 128) --
   cheap in the parameter budget (roughly doubling stays well under 2M) and
   was not tried in this pass.
2. Try a longer schedule with a decayed learning rate (e.g. 800-1200 epochs,
   lr warm-restart or cosine decay) rather than a single fixed lr=5e-4 --
   the loss was still (slowly) decreasing at epoch 400.
3. Consider whether Chess needs its own diagnostic: fit a plain per-step
   65-way linear probe directly on the same 10-game `H_t` slice (bypassing
   the transition/observation-scorer's compatibility-function design) to
   establish an upper bound on what 10 games of this hidden cache can
   support at all, before concluding the shared-core design is the
   bottleneck rather than the data slice.
4. Only after Chess clears >=95% on its own tiny slice should the joint
   Shell+Chess tiny gate be re-run end to end.

## 8. Next Formal Command (not executed)

Once the tiny gate passes for both tasks, the next step this interface
already supports without new code is a Shell+Chess joint fit followed by
the Task-C zero/few-shot protocol demo turned into a real held-out split
evaluation. Do not run this until item 7 is resolved:

```bash
python scripts/ustf.py tiny-gate \
  --shell-trajectories 5 --chess-games 10 \
  --epochs 800 --lr 5e-4 --patience 800 \
  --hidden 256 \
  --out outputs/ustf_v1_retry
```

## Section 13 Checklist (what was and wasn't done)

Done: code implementation, 12 unit tests (synthetic, no cache dependency),
Shell/Chess dataset adapter reuse (no new adapter code), candidate
permutation test (unit + real trained-model check), tiny overfit run on
real Shell/Qwen(5) + Chess/Qwen(10) data, joint forward-pass compatibility
smoke across Qwen/LLaVA x Shell/Chess, Qwen/LLaVA cache-schema test
(`test_real_cache_schema_qwen_llava_if_present`, plus the real
`joint-smoke` run against both caches).

Not done (per instructions, and additionally blocked by the tiny-gate
result): formal/large training suite, a real third task, any new VLM
forward pass, any new data split, circuit/head experiments.

## Files

```text
staterev/ustf.py
scripts/ustf.py
tests/test_ustf.py
outputs/ustf_v1/tiny_gate/tiny_gate_report.json
outputs/ustf_v1/tiny_gate/tiny_gate_predictions.csv
outputs/ustf_v1/tiny_gate/tiny_gate_history.csv
outputs/ustf_v1/tiny_gate/tiny_gate_model.pt
outputs/ustf_v1/parameter_counts_qwen.json
outputs/ustf_v1/parameter_counts_qwen_llava.json
outputs/ustf_v1/joint_forward_smoke.json
outputs/ustf_v1/task_c_interface_demo.json    (synthetic task, interface-only)
```
