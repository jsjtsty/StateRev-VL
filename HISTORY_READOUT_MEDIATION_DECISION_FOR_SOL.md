# History-to-Readout Mediation Decision

## Conclusion

The corrected experiment supports a representation-to-readout bottleneck.
Injecting the history-sensitive activation into the target input reliably
changes the held-out state decoder at L24, L32, and L36, but does not produce a
reliable native state-logit change. The current-window control is close to
zero.

The appropriate claim is:

> Earlier visual history is causally present in the late hidden representation,
> but its effect is largely suppressed or discarded by the native state
> readout.

This does not support the stronger claim that history substitution changes the
model's native categorical state commitment.

## Audit

- Fixed held-out path manifest: 36 pairs, 17 target trajectories, `t>=3`.
- Conditions: history, current-window control, donor-real.
- Layers: L24, L32, L36.
- Both sufficiency and necessity were evaluated.
- Inference: target-prefix aggregation followed by target-trajectory cluster
  bootstrap and sign permutation.
- Pair rows are descriptive; pairs are not independent observations.
- All eight shards completed; merged output has 648 rows.
- L36 was corrected to use the post-final-norm endpoint matching the existing
  hidden cache.
- Self-patch native logit error was exactly zero in the saved output.

## Primary `t>=3` effects

### History condition

| layer | state-probe sufficiency | state-probe necessity | event-probe sufficiency | native GT-logit sufficiency |
|---:|---:|---:|---:|---:|
| L24 | `+0.329 [0.116,+0.520]` | `+0.329 [0.116,+0.520]` | `+0.133 [0.024,+0.277]` | `+0.133 [-0.041,+0.322]` |
| L32 | `+0.331 [0.177,+0.485]` | `+0.331 [0.177,+0.485]` | `+0.162 [0.064,+0.294]` | `+0.094 [-0.179,+0.388]` |
| L36 | `+0.371 [0.220,+0.508]` | `+0.371 [0.220,+0.508]` | `+0.174 [0.072,+0.309]` | `+0.091 [-0.211,+0.416]` |

The native GT-logit necessity effects were also not reliable:

- L24: `+0.025 [-0.143,+0.201]`;
- L32: `+0.059 [-0.211,+0.349]`;
- L36: `+0.091 [-0.211,+0.416]`.

The state and event effects are significant or directionally stable, while
native effects remain statistically compatible with zero.

### Current-window control

The current-window condition produced small effects:

- state probe: L24 `+0.010 [-0.042,+0.066]`, L32 `-0.025 [-0.096,+0.042]`,
  L36 `+0.018 [-0.071,+0.104]`;
- event probe: L24 `-0.004 [-0.015,+0.007]`, L32 `+0.002 [-0.033,+0.029]`,
  L36 `+0.012 [-0.023,+0.045]`;
- native GT logit: L24 `+0.017 [-0.030,+0.063]`, L32 `+0.034 [-0.011,+0.078]`,
  L36 `+0.029 [-0.020,+0.079]`.

This control does not show a comparable hidden-state shift.

## Sufficiency and necessity interpretation

For the state and event decoders, sufficiency and necessity recover the same
history-versus-target representation difference, as expected from decoding the
patched endpoint. This verifies that the activation patch actually transfers
the representation-level history signal in both directions.

For native logits, the bidirectional effects are small and their confidence
intervals include zero. Thus the history signal reaches the decoder-visible
representation but is not reliably transmitted into the native state output.

## Step profile

The representation effect is weakest at t=3 and larger at t=4/t=5:

- state-probe history sufficiency at L24: t3 `+0.025`, t4 `+0.466`, t5
  `+0.555`;
- at L32: t3 `+0.093`, t4 `+0.466`, t5 `+0.391`;
- at L36: t3 `+0.157`, t4 `+0.484`, t5 `+0.465`.

Native GT-logit effects do not show a comparably stable stepwise increase.
The main result therefore concerns representation sensitivity, not increasing
native state commitment with depth.

## Decision

### Restricted GO: representation mechanism

Continue only with analyses that explain why the native readout suppresses the
history-sensitive signal, for example a fixed readout geometry analysis or a
pre-registered comparison of native logits against hidden-state directions.

### NO-GO: native path-dependent commitment claim

Do not claim that the model's native answer is path-dependent based on this
experiment. Native categorical commitment was not reliably changed.

### Stop condition

Do not continue the Head0/L32 rescue line or increase intervention amplitude
solely to force categorical flips. The combined evidence already indicates a
distributed commitment bottleneck rather than a missing single activation.

## Artifacts

- `outputs/vetbench/history_readout_mediation_v1/history_readout_mediation_results.csv`
- `outputs/vetbench/history_readout_mediation_v1/history_readout_mediation_summary.json`
- `outputs/vetbench/history_readout_mediation_v1/history_readout_mediation_report.md`
