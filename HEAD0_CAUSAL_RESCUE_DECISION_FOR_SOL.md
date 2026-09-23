# StateRev-VL Head 0 Causal Rescue：供 GPT-5.6 Sol 决策

## Executive decision

**Result: Restricted GO for causal state-margin rescue; NO-GO for strong categorical/behavioral rescue.**

Patching a strong, state-matched donor's L24 Head 0 final-token activation into a weak target significantly increases the target model's native GT-state margin. The effect remains strong at `t>=2` and `t>=3`, exceeds ordinary and fixed-random matched donors, and reverses direction when the donor implies a different state. However, the intervention rarely changes the categorical answer, and the strict GT-logit-specificity endpoint is not positive on its own. Head 0 therefore carries causally usable, state-directed information, but a single unscaled Head 0 transplant is generally insufficient to overcome the model's downstream commitment threshold.

The strongest defensible interpretation is:

> L24 Head 0 is a state-directed causal readout channel whose activation can improve native state margins, especially for weak targets, but the model's final categorical commitment is constrained by additional distributed/downstream components.

Do not describe this experiment as successful behavioral correction or as a complete state circuit.

## 1. Protocol and audit

The intervention remained frozen from prior localization:

- Qwen3-VL decoder L24, one-based;
- Head 0 only;
- final prompt token only;
- pre-`o_proj` head-concat slice `[0:128]`;
- target video, prompt, input IDs, pixels, grid, and all other activations unchanged.

VET-Bench contains only 50 trajectories, all already used in the earlier 30/20 discovery-validation split. No untouched third trajectory split exists. This rescue experiment therefore uses:

- targets: the frozen 20 validation trajectories;
- donor pool: the disjoint 30 discovery trajectories;
- 100 possible target prefixes;
- 97 eligible prefixes with an exact same-state donor, a 97% matching rate;
- 3 preregistered exclusions at `t=2` because no discovery-pool donor matched the required `t/S_prev/event/S_t` tuple.

The rescue intervention is new and Head 0 was frozen before it, but this is not an independent new-trajectory replication. Primary inference first aggregates at target prefix and then treats target trajectory as the cluster. Pair counts are not used as independent `n`.

Sanity checks passed:

- all 8 shards are complete: 485 rows = 97 targets x 5 conditions;
- all 20 target trajectories are represented;
- target input IDs, pixels, and video grid are identical across conditions;
- self-patch effects are exactly zero;
- donor and target trajectories are disjoint except by definition for self patch;
- donor matching fields satisfy the frozen manifest.

One offline reporting bug was corrected before this report: `baseline-incorrect` now strictly means `baseline GT margin < 0`, yielding 63 rather than 64 samples. The excluded case was a zero-margin tie. No GPU result changed.

## 2. Primary strong-donor result

The primary intervention uses the strongest available donor satisfying:

```text
same t, same S_prev, same current event, same GT S_t, different trajectory
```

Donor strength was fixed by donor baseline GT margin before rescue execution.

| subset | Delta GT margin | 95% cluster CI | permutation p | GT-state specificity |
|---|---:|---:|---:|---:|
| overall | `+0.404` | `[+0.302,+0.504]` | `<0.0001` | `+0.011 [-0.060,+0.095]` |
| `t>=2` | `+0.504` | `[+0.374,+0.632]` | `<0.0001` | `+0.023 [-0.067,+0.128]` |
| `t>=3` | `+0.465` | `[+0.333,+0.604]` | `<0.0001` | `-0.019 [-0.137,+0.123]` |
| weak | `+0.745` | `[+0.523,+0.991]` | `<0.0001` | `+0.169 [-0.006,+0.389]` |
| strong | `+0.171` | `[+0.045,+0.318]` | `0.0207` | `-0.101 [-0.269,+0.013]` |
| baseline margin `<0` | `+0.605` | `[+0.415,+0.807]` | `<0.0001` | `+0.095 [-0.036,+0.253]` |

The native GT margin result is robust and persists in the genuinely multi-step subsets. It is also substantially larger for the median-defined weak group than for the strong group.

The stricter specificity endpoint is not independently positive: its CI crosses zero overall, at `t>=2`, and at `t>=3`. The positive margin effect therefore should not be paraphrased as "Head 0 selectively increases only the GT logit." At prefix level, strong-donor patching changes multiple native logits; it improves the GT state relative to the currently strongest competitor, but the GT logit is not consistently the largest-increasing logit among all three states.

## 3. Comparison with matched donors

Strong-donor selection matters. The trajectory-cluster effects on GT margin were:

| subset | strong donor | ordinary matched donor | fixed-random matched donor |
|---|---:|---:|---:|
| overall | `+0.404` | `+0.114` | `+0.184` |
| `t>=2` | `+0.504` | `+0.126` | `+0.233` |
| `t>=3` | `+0.465` | `+0.210` | `+0.240` |
| weak | `+0.745` | `+0.331` | `+0.407` |

Paired main-minus-control effects were:

| subset | strong minus ordinary | strong minus random |
|---|---:|---:|
| overall | `+0.290 [+0.202,+0.377]` | `+0.220 [+0.132,+0.310]` |
| `t>=2` | `+0.378 [+0.265,+0.489]` | `+0.271 [+0.159,+0.387]` |
| `t>=3` | `+0.254 [+0.135,+0.375]` | `+0.225 [+0.108,+0.350]` |
| weak | `+0.414 [+0.241,+0.597]` | `+0.338 [+0.213,+0.475]` |

All listed CIs exclude zero. This is evidence that rescue depends on donor activation quality, rather than any arbitrary Head 0 replacement.

Ordinary and random same-state donors nevertheless also produce positive average GT-margin movement. This is compatible with graded state information in Head 0. Their negative average strict-specificity scores show that arbitrary matched donors are not clean state-selective rescuers.

## 4. Different-event directional control

The different-event control matches `t` and `S_prev` but uses a donor whose event implies a different resulting state. It should not rescue the target GT. That prediction is supported:

| subset | Delta target GT margin | Counterfactual donor-state margin shift |
|---|---:|---:|
| overall | `-0.233 [-0.364,-0.099]` | `+0.444 [+0.286,+0.616]` |
| `t>=2` | `-0.203 [-0.335,-0.076]` | `+0.447 [+0.272,+0.649]` |
| `t>=3` | `-0.240 [-0.375,-0.112]` | `+0.456 [+0.267,+0.671]` |

The counterfactual movement is significant under trajectory-cluster inference. This is the strongest specificity evidence in the rescue study: Head 0 does not merely add generic confidence. Its causal direction changes with the state semantics of the donor activation.

This result complements, but does not erase, the null same-state `GT-state specificity` endpoint. Together they indicate a state-directed but distributed/non-isolated readout contribution.

## 5. Weak-target hypothesis

The preregistered median split supports larger margin gain for weak targets:

```text
weak minus strong rescue gain = +0.626
95% CI = [+0.347,+0.929]
trajectory-paired permutation p = 0.0006
```

The continuous within-trajectory relationship is weaker:

```text
slope(delta GT margin ~ baseline GT margin) = -0.046
95% CI = [-0.104,+0.021]
p = 0.172
```

Therefore the data support a coarse weak-versus-strong interaction, but not a reliable monotonic law that every decrement in baseline margin produces a larger gain. This distinction should be retained in any paper claim.

## 6. Categorical rescue and harm

Continuous native-margin movement rarely crosses the decision boundary:

- baseline prediction incorrect: 64/97 targets, including one zero-margin tie;
- observed wrong-to-correct flips under strong donor: 3/64;
- strict baseline margin `<0`: 63 targets;
- strict incorrect targets becoming positive-margin: 1/63;
- strict incorrect targets becoming nonnegative-margin: 2/63;
- weak group: zero categorical flips despite a large average margin gain;
- observed correct-to-wrong harmful flips: 0/33.

The trajectory-cluster wrong-to-correct estimate is `0.050` overall and `0.075` at `t>=2`, but its bootstrap CI includes zero. The sign-permutation test is also uninformative for this sparse endpoint.

The observed zero harmful flips is encouraging but should not be interpreted as a population CI of exactly zero. A nonparametric bootstrap of an all-zero observed endpoint is degenerate and cannot rule out a nonzero harm rate in new data.

Thus the experiment demonstrates subthreshold belief/readout movement, not reliable behavioral rescue.

## 7. Mechanistic interpretation

The rescue result sharpens the existing circuit picture:

1. Raw visual transplantation establishes causal event input.
2. Late-layer block and L24 attention patching establish native-state mediation.
3. Held-out Head 0 localization isolates most of the L24 attention native-state effect, but only a minority of its event effect.
4. Strong same-state Head 0 donors raise target GT margins.
5. Different-event Head 0 donors lower target GT margins and raise donor-implied counterfactual margins.
6. Most targets still do not change categorical answers.

The combined result favors:

```text
distributed event encoding/routing
             -> Head 0 state-directed readout contribution
             -> downstream/distributed commitment bottleneck
```

It disfavors both extremes:

- **not no-effect:** Head 0 has robust, directionally semantic causal influence;
- **not a complete state circuit:** Head 0 alone rarely rescues the final decision and does not uniquely increase the GT logit.

## 8. Recommended next decision

### Recommended scientific status

Use the following labels:

- `GO` for a validated Head 0 state-directed causal contribution;
- `Restricted GO` for continuous native-margin rescue;
- `NO-GO` for claiming robust categorical behavioral rescue;
- `NO-GO` for claiming Head 0 is the complete visual-event-to-state circuit.

### Recommended experimental action

Do not resume broad layer/head search. The localization question has reached diminishing returns.

If one final targeted experiment is scientifically necessary, the most informative next test is a preregistered Head 0 dose-response/rescue test on the already frozen target-donor pairs, using interpolation/extrapolation along the donor-minus-target Head 0 direction. It should ask whether increasing intervention strength produces monotonic GT-margin movement and decision-boundary crossings while different-event donors produce the opposite state-specific direction. This would directly test whether the weak categorical rescue is merely subthreshold or requires additional heads/downstream components.

If time or compute is limited, stop here and frame the contribution as **causal state-belief modulation with a downstream commitment bottleneck**, not model correction.

## 9. Files

- `outputs/vetbench/head0_causal_rescue_v1/rescue_manifest.json`
- `outputs/vetbench/head0_causal_rescue_v1/rescue_pair_results.csv`
- `outputs/vetbench/head0_causal_rescue_v1/rescue_target_aggregated.csv`
- `outputs/vetbench/head0_causal_rescue_v1/rescue_summary.json`
- `outputs/vetbench/head0_causal_rescue_v1/head0_causal_rescue_report.md`

