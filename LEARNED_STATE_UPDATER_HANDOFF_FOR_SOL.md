# Learned State Updater Experiment Handoff

Date: 2026-09-18

## Objective

Implemented a lightweight repair method for the shell-game task:

```text
frozen VLM + frozen hidden-event decoder + learned state updater
```

The existing strict content-disjoint split and existing Qwen3-VL/LLaVA hidden-event caches were reused. No new data split or VLM forward pass was introduced.

## Implementation

Main implementation:

```text
scripts/learned_state_updater.py
```

Runner:

```text
scripts/run_learned_state_updater.sh
```

The updater supports the current 3-state / 3-event shell-game:

```text
P(S_t) = sum_e q_t(e) * P(S_{t-1}) * T_e
```

Each event has a learned `3 x 3` row-stochastic transition matrix. The updater therefore has:

```text
3 events x 3 previous states x 3 current states = 27 parameters
```

The base VLM and hidden-event decoder remain frozen. Updater fitting uses discovery trajectories only.

## Existing Data and Split

Split reused without modification:

```text
outputs/vetbench/content_disjoint_split_v1/discovery_validation_split.json
```

The split contains:

```text
discovery: 30 trajectories
validation: 20 trajectories / 100 prefixes
```

At cache load time the script rechecks:

- discovery/validation trajectory overlap;
- exact pixel fingerprint overlap;
- exact hidden tensor overlap.

Each fitted artifact also records SHA256 hashes for the split, behavior cache, hidden cache, and decoder.

## Cache Adapters

### Qwen

```text
behavior:
  outputs/vetbench/mechanism_gate_final/behavior.csv

hidden:
  outputs/vetbench/hidden_state_probe/hidden_states.npz

decoder:
  outputs/vetbench/hidden_event_recursive_content_disjoint_v1/frozen_event_decoder.joblib

native event probabilities:
  outputs/vetbench/recursive_state_recovery_v1/shards/
```

### LLaVA

```text
behavior:
  outputs/vetbench/llava_next_video_7b_replication_v1/behavior.csv

hidden:
  outputs/vetbench/llava_next_video_7b_replication_v1/hidden_states.npz

decoder:
  outputs/vetbench/llava_next_video_7b_replication_v1/frozen_hidden_event_decoder.joblib
```

LLaVA event probabilities are read from its existing `behavior.csv` event score fields.

## Implemented Commands

Unit test:

```bash
python scripts/learned_state_updater.py unit
```

Fit both models:

```bash
bash scripts/run_learned_state_updater.sh fit
```

Fit in parallel:

```bash
bash scripts/run_learned_state_updater.sh fit-parallel
```

Run four validation combinations:

```bash
bash scripts/run_learned_state_updater.sh evaluate
```

The four combinations are:

```text
Qwen updater   -> Qwen validation
LLaVA updater  -> LLaVA validation
Qwen updater   -> LLaVA validation
LLaVA updater  -> Qwen validation
```

Run few-shot curves:

```bash
bash scripts/run_learned_state_updater.sh curves
```

Full workflow:

```bash
bash scripts/run_learned_state_updater.sh all
```

## Smoke Tests

Completed successfully:

- updater unit test;
- Qwen 2-trajectory smoke test;
- LLaVA 2-trajectory smoke test;
- Qwen/LLaVA temperature calibration smoke test;
- Qwen-to-Qwen artifact loading;
- Qwen-to-LLaVA artifact loading;
- LLaVA-to-Qwen artifact loading;
- LLaVA-to-LLaVA artifact loading.

Smoke output:

```text
outputs/vetbench/learned_state_updater_v1/smoke/
```

All four cross-model smoke combinations achieved `1.0` on the two discovery trajectories used for the engineering smoke test. This is not a held-out scientific result.

## Completed Formal Results

Results are stored under:

```text
outputs/vetbench/learned_state_updater_v1/
```

Each validation CSV contains 100 prefixes for each method, with 20 validation trajectories.

### Validation Summary

The learned updater and handwritten-rule recursion produced identical results in all four combinations.

| Source updater | Target model | Overall | t>=2 | t>=3 | t=1 | t=2 | t=3 | t=4 | t=5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen | Qwen | 0.890 | 0.8625 | 0.8167 | 1.00 | 1.00 | 0.85 | 0.80 | 0.80 |
| LLaVA | LLaVA | 1.000 | 1.0000 | 1.0000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Qwen | LLaVA | 1.000 | 1.0000 | 1.0000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| LLaVA | Qwen | 0.890 | 0.8625 | 0.8167 | 1.00 | 1.00 | 0.85 | 0.80 | 0.80 |

Trajectory-cluster bootstrap 95% CIs are included in the JSON summaries. For example, Qwen-to-Qwen overall CI is `[0.78, 0.98]`; LLaVA-to-LLaVA overall CI is `[1.00, 1.00]`.

### Native State Baseline

The canonical strict content-disjoint recursive baseline uses the native
predictions stored in `behavior_audit_v2/mechanism_candidates.csv`, which gives
Qwen `37%` overall. The learned-updater cache adapter separately reports `31%`
from `mechanism_gate_final/behavior.csv` baseline rows. These are different
saved native behavior caches and must not be silently combined.

Native VLM state-answer accuracy:

| Model | Overall | t>=2 | t>=3 |
|---|---:|---:|---:|
| Qwen, canonical content-disjoint recursive cache | 0.37 | 0.3125 | 0.3333 |
| Qwen, alternate mechanism-gate behavior cache | 0.31 | 0.325 | 0.350 |
| LLaVA | 0.28 | 0.300 | 0.3167 |

Thus hidden-event recursion is much stronger than native state generation on this task.

### Few-Shot Curves

Five fixed-seed repeats were run for each trajectory count.

#### Qwen

| Discovery trajectories | Overall | t>=2 | t>=3 | t=5 |
|---:|---:|---:|---:|---:|
| 1 | 0.330 | 0.318 | 0.293 | 0.300 |
| 2 | 0.508 | 0.458 | 0.403 | 0.360 |
| 5 | 0.762 | 0.725 | 0.667 | 0.630 |
| 10 | 0.892 | 0.865 | 0.820 | 0.810 |
| 20 | 0.890 | 0.863 | 0.817 | 0.800 |
| 30 | 0.890 | 0.863 | 0.817 | 0.800 |

#### LLaVA

| Discovery trajectories | Overall | t>=2 | t>=3 | t=5 |
|---:|---:|---:|---:|---:|
| 1 | 0.318 | 0.307 | 0.270 | 0.260 |
| 2 | 0.544 | 0.503 | 0.463 | 0.420 |
| 5 | 0.856 | 0.843 | 0.820 | 0.780 |
| 10 | 1.000 | 1.000 | 1.000 | 1.000 |
| 20 | 1.000 | 1.000 | 1.000 | 1.000 |
| 30 | 1.000 | 1.000 | 1.000 | 1.000 |

## Learned Transition Structure

The learned matrices show the expected event-specific shell-game permutation structure. On the small smoke fit, rows corresponding to observed state/event combinations approach one-hot transitions. The learned updater has 27 parameters, but the task structure makes the effective solution close to the handwritten transition rule.

The formal learned updater and handwritten recursion are numerically equivalent at the reported validation metrics. Therefore the current experiment demonstrates that the transition rule can be learned from discovery data, but does not demonstrate an accuracy improvement over the hand-coded rule.

## Temperature Calibration

Discovery-only temperature fitting selected:

```text
Qwen:  T = 0.25
LLaVA: T = 0.25
```

The calibrated discovery NLL was lower than raw NLL for both models:

```text
Qwen:
  raw NLL:        0.0031859965
  calibrated NLL: 3.0089866e-09

LLaVA:
  raw NLL:        0.0028565326
  calibrated NLL: 1.5072823e-09
```

Important limitation: the completed validation summaries used the default `temperature=1.0`. The calibrated validation recursion was not included in the formal summary. The temperature files are available, but a separate validation pass is required before making claims about calibrated recursive accuracy or persistent-error reduction.

## Output Files

Updater artifacts:

```text
outputs/vetbench/learned_state_updater_v1/qwen_updater.joblib
outputs/vetbench/learned_state_updater_v1/llava_updater.joblib
```

Artifact metadata and SHA256 manifests:

```text
outputs/vetbench/learned_state_updater_v1/qwen_updater.json
outputs/vetbench/learned_state_updater_v1/llava_updater.json
outputs/vetbench/learned_state_updater_v1/qwen_manifest.json
outputs/vetbench/learned_state_updater_v1/llava_manifest.json
```

Validation summaries:

```text
outputs/vetbench/learned_state_updater_v1/qwen_to_qwen_summary.json
outputs/vetbench/learned_state_updater_v1/llava_to_llava_summary.json
outputs/vetbench/learned_state_updater_v1/qwen_to_llava_summary.json
outputs/vetbench/learned_state_updater_v1/llava_to_qwen_summary.json
```

Validation predictions:

```text
outputs/vetbench/learned_state_updater_v1/qwen_to_qwen_validation.csv
outputs/vetbench/learned_state_updater_v1/llava_to_llava_validation.csv
outputs/vetbench/learned_state_updater_v1/qwen_to_llava_validation.csv
outputs/vetbench/learned_state_updater_v1/llava_to_qwen_validation.csv
```

Few-shot curves:

```text
outputs/vetbench/learned_state_updater_v1/qwen_fewshot_curve.csv
outputs/vetbench/learned_state_updater_v1/llava_fewshot_curve.csv
```

Logs:

```text
outputs/vetbench/learned_state_updater_v1/logs/
```

## Interpretation for Sol

The safest current interpretation is:

> A 27-parameter transition updater trained only on discovery trajectories can recover the shell-game state transition structure from frozen hidden-event probabilities. The learned updater transfers across Qwen3-VL and LLaVA-NeXT-Video caches without measurable degradation relative to the handwritten rule. However, because the learned and handwritten recursions are identical on the current validation results, this is evidence of learnability and cross-model compatibility, not evidence of an accuracy gain.

The strongest held-out result is the model-specific difference in hidden-event quality:

- LLaVA hidden-event recursion: 100% validation accuracy;
- Qwen hidden-event recursion: 89% validation accuracy.

The updater itself does not explain this difference; the event decoder quality does.

## Recommended Next Step

Before writing a final paper claim about calibration, run only the missing calibrated validation evaluations:

```bash
python scripts/learned_state_updater.py evaluate \
  --source-model qwen \
  --target-model qwen \
  --artifact outputs/vetbench/learned_state_updater_v1/qwen_updater.joblib \
  --temperature 0.25 \
  --out outputs/vetbench/learned_state_updater_v1

python scripts/learned_state_updater.py evaluate \
  --source-model llava \
  --target-model llava \
  --artifact outputs/vetbench/learned_state_updater_v1/llava_updater.joblib \
  --temperature 0.25 \
  --out outputs/vetbench/learned_state_updater_v1
```

Do not overwrite the existing summaries if preserving the raw-temperature results is important; use a separate output directory or extend the evaluator with an explicit calibrated filename suffix.
