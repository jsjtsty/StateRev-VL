# Native Baseline Alignment and Temperature Validation Summary

Date: 2026-09-18

This report summarizes two small-scope checks performed offline from existing
StateRev-VL artifacts. No new large-scale VLM forward was launched.

## 1. Qwen Native State: 31% vs 37%

### Shared protocol

Both results use the same strict content-disjoint split:

```text
outputs/vetbench/content_disjoint_split_v1/discovery_validation_split.json
```

Coverage is identical:

```text
20 validation trajectories / 100 validation prefixes
```

The validation prefix IDs, GT states, state-question prompt family, and row
aggregation are aligned.

### Source comparison

| Result | Native prediction source | Overall accuracy | Role |
|---|---|---:|---|
| 37% | `behavior_audit_v2/mechanism_candidates.csv` | 0.37 | Canonical strict content-disjoint recursive baseline |
| 31% | `mechanism_gate_final/behavior.csv`, `condition=baseline` | 0.31 | Alternate saved behavior-cache result |

The two prediction vectors match on `87/100` prefixes and differ on `13/100`.
The discrepancy is therefore caused by different saved native behavior caches,
not by split, GT labels, prefix coverage, or aggregation.

The decode paths also differ:

- 37% source: generated answer text parsed by `run_state_rev_audit.parse_tracking_option`.
- 31% source: first-token/logit greedy answer decode from `state_rev_input_pipeline.first_token_logits`.

Both code paths use the same `run_state_rev_audit.state_messages` and
`state_question_text` prompt family.

### Per-step comparison

| Source | Overall | t>=2 | t>=3 | t=1 | t=2 | t=3 | t=4 | t=5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Canonical 37% cache | 0.370 | 0.3125 | 0.3333 | 0.60 | 0.25 | 0.15 | 0.35 | 0.50 |
| Alternate 31% cache | 0.310 | 0.3250 | 0.3500 | 0.25 | 0.25 | 0.15 | 0.35 | 0.55 |

### Decision

Use **37%** as the single authoritative Qwen native-state baseline for the
current strict content-disjoint hidden-event recursive report. Retain 31% only
as a separately labeled alternate behavior-cache result.

Do not silently rewrite the original JSON/CSV artifacts.

Alignment outputs:

```text
outputs/vetbench/native_state_alignment_audit_v1/qwen_native_state_comparison.csv
outputs/vetbench/native_state_alignment_audit_v1/alignment_report.md
outputs/vetbench/native_state_alignment_audit_v1/alignment_report.json
```

The existing handoff was updated accordingly:

```text
LEARNED_STATE_UPDATER_HANDOFF_FOR_SOL.md
```

## 2. Temperature Validation

Temperature was not re-fitted. The discovery-only values were reused:

```text
Qwen:  T = 0.25
LLaVA: T = 0.25
```

The calibrated evaluations were written to a new directory:

```text
outputs/vetbench/learned_state_updater_v1_calibrated/
```

No `T=1.0` outputs were overwritten.

### Qwen: T=1.0 vs T=0.25

| Method | Temperature | Overall | t>=2 | t>=3 | t=5 | Persistent error | Wrong trajectories |
|---|---:|---:|---:|---:|---:|---:|---:|
| Learned updater | 1.00 | 0.89 | 0.8625 | 0.8167 | 0.80 | 4/20 = 0.20 | 4/20 |
| Learned updater | 0.25 | 0.89 | 0.8625 | 0.8167 | 0.80 | 4/20 = 0.20 | 4/20 |

The handwritten-rule recursion has the same values.

### LLaVA: T=1.0 vs T=0.25

| Method | Temperature | Overall | t>=2 | t>=3 | t=5 | Persistent error | Wrong trajectories |
|---|---:|---:|---:|---:|---:|---:|---:|
| Learned updater | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0/20 = 0.00 | 0/20 |
| Learned updater | 0.25 | 1.00 | 1.00 | 1.00 | 1.00 | 0/20 = 0.00 | 0/20 |

The handwritten-rule recursion has the same values.

### Interpretation

Temperature calibration changes event confidence but does not change the
event argmax on this cache. Consequently, the hard recursive state path is
unchanged, including:

- overall accuracy;
- t>=2 and t>=3 accuracy;
- t=5 accuracy;
- persistent error rate;
- wrong trajectory count.

For the current data, calibration does not reduce error propagation. The
remaining Qwen errors are caused by event-path mistakes that remain unchanged
under `T=0.25`.

### Calibrated output files

```text
outputs/vetbench/learned_state_updater_v1_calibrated/qwen_to_qwen_validation.csv
outputs/vetbench/learned_state_updater_v1_calibrated/qwen_to_qwen_summary.json
outputs/vetbench/learned_state_updater_v1_calibrated/llava_to_llava_validation.csv
outputs/vetbench/learned_state_updater_v1_calibrated/llava_to_llava_summary.json
```

The evaluator now reports:

- `persistent_error_rate`;
- `persistent_error_trajectory_count`;
- `wrong_trajectory_count_any_step`;
- `wrong_trajectory_count_final_t5`.

## Final Conclusions

1. The authoritative Qwen native state baseline is **37%**, not 31%, for the current strict content-disjoint recursive experiment.
2. The 31% result is valid only as an alternate result from a different saved native behavior cache and decode path.
3. Both Qwen and LLaVA discovery-only temperature calibration selected `T=0.25`.
4. Applying `T=0.25` does not change recursive state accuracy or persistent errors for either model.
5. The learned updater remains identical to the handwritten transition rule on the reported validation metrics.
6. No evidence currently supports claiming calibration as an accuracy or error-propagation improvement.

