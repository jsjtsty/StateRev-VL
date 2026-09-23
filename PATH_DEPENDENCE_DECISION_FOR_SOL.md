# StateRev-VL Path-Dependence Decision Report

## Executive conclusion

The held-out experiment provides strong evidence that earlier visual history
changes the model's internal hidden representation even when the target and
donor are matched on:

`t, initial state, S_{t-1}, E_t, S_t`.

The result does **not** show an equally strong change in the model's native
state commitment. Across all four conditions, the native categorical
prediction was identical for every one of the 36 pairs. The current result is
therefore best described as:

> history-sensitive internal representation with weak or invariant native
> state readout.

This supports a path-dependent / non-Markov representation hypothesis, but
not a claim that historical context directly changes the final native answer.

## Protocol and audit

- 36 fixed held-out pairs, `t>=3` only.
- 17 validation target trajectories had an eligible discovery donor.
- Pair counts: `t=3: 11`, `t=4: 12`, `t=5: 13`.
- Discovery and validation trajectories were disjoint.
- Four conditions were evaluated:
  - `target_real`
  - `donor_history_target_current`
  - `target_history_donor_current`
  - `donor_real`
- Primary inference: target-prefix aggregation, then target-trajectory cluster bootstrap and sign permutation.
- Pair rows are descriptive; `n=36` is not an independent sample size.
- All 8 shards completed and the merged artifact contains 144 rows.

The run used Transformers `5.16.1` and PyTorch `2.14.0+cu126`; all four
conditions within the run used the same versions. This differs from the
original project environment specification and should be recorded for exact
reproduction.

## Main history contrasts

The primary contrast holds the target current-event window fixed:

`donor_history_target_current - target_real`

The mirrored contrast holds the donor current-event window fixed:

`donor_real - target_history_donor_current`

### State hidden decoder

For `t>=3`:

| endpoint | target-history contrast | donor-history contrast |
|---|---:|---:|
| L24 state probability | `+0.329 [0.121, 0.522]` | `+0.535 [0.374, 0.687]` |
| L32 state probability | `+0.331 [0.177, 0.486]` | `+0.504 [0.382, 0.628]` |
| L36 state probability | `+0.371 [0.219, 0.509]` | `+0.514 [0.416, 0.616]` |

All intervals above are target-trajectory-cluster bootstrap 95% CIs. The
corresponding sign-permutation p-values are approximately `0.0075/0.0001`,
`0.0009/<0.001`, and `0.0005/<0.001` respectively.

The current-window-only control was close to zero in `t>=3`:

- L24: approximately `+0.010`, CI `[-0.042, +0.067]`;
- L32: approximately `-0.025`, CI `[-0.096, +0.041]`;
- L36: approximately `+0.018`, CI `[-0.070, +0.104]`.

Thus the hidden-state effect is primarily associated with the substituted
earlier history, not simply with replacing the current event window.

### Event decoder

The event endpoint also changed under history substitution despite the
current event window being held fixed. For `t>=3`, the target-history
contrast was:

- L24: `+0.133 [0.025, 0.274]`;
- L32: `+0.162 [0.065, 0.293]`;
- L36: `+0.174 [0.072, 0.306]`.

This is an important caution. The event decoder is not a pure causal event
readout under this construction: its output is history-sensitive too. The
finding supports contextual entanglement of the final-token representation,
but should not be phrased as “history changes the perceived current event.”

## Native state readout

The native answer did not change categorically:

- Every condition had the same predictions: 29 `Middle`, 7 `Left`.
- Native accuracy was 12/36 (`33.3%`) in every condition.
- There were no prediction changes, no wrong-to-correct flips, and no
  correct-to-wrong flips in any history contrast.

For `t>=3`, the target-history contrast changed the native logits on average
by approximately:

- Left: `-0.180`;
- Middle: `+0.124`;
- Right: `-0.050`.

However, the GT margin effect was not reliable:

- target-history contrast: `+0.201 [-0.174, +0.602]`;
- donor-history contrast: `+0.233 [-0.172, +0.654]`.

Therefore the history-dependent hidden representation is not being converted
into a categorical native state update in this experiment.

## Step dependence

The hidden history effect is strongest and most stable at `t=4` and `t=5`.
At `t=3`, the target-history contrast is weaker and often includes zero, while
the mirrored donor-history contrast remains positive for several late-layer
state endpoints. This is consistent with a progressively accumulating history
effect, but the step-specific sample sizes are only 11, 12, and 13 prefixes.

The native categorical answer remains unchanged at every step.

## Interpretation

The result rules against the narrow claim that the final hidden state is a
pure function only of the matched tuple `(S_{t-1}, E_t, S_t)`, at least for
the tested decoder endpoint. Earlier history remains recoverable or changes
the geometry of the final representation.

It does **not** by itself establish that the model maintains a separate
abstract world-state variable. Three explanations remain compatible:

1. a genuinely path-dependent internal state;
2. history-dependent nuisance/context features entangled with state and event;
3. a decoder-sensitive representation whose native readout projects away the
   history difference.

The strongest supported claim is therefore:

> Earlier visual history causally influences late hidden representations under
> matched current transition conditions, while the native state readout is
> comparatively invariant and remains categorically weak.

## Decision

### GO: representation-level follow-up

The result is strong enough to continue a representation-level mechanism
study, especially testing whether the history effect is localized to a
specific layer/module and whether it predicts later commitment failures.

### NO-GO: native path-dependent state claim

Do not claim that history substitution changes the model's native current-state
belief. Native logits and categorical predictions do not support that claim in
the present held-out sample.

### Recommended framing

The project should move from “does the model update state?” to:

> visual events are incorporated into a history-sensitive latent state, but
> the model's downstream native commitment/readout discards or compresses much
> of that information.

Relevant artifacts:

- `outputs/vetbench/path_dependence_v1/path_dependence_results.csv`
- `outputs/vetbench/path_dependence_v1/path_dependence_contrasts.csv`
- `outputs/vetbench/path_dependence_v1/path_dependence_summary.json`
- `outputs/vetbench/path_dependence_v1/path_dependence_report.md`
