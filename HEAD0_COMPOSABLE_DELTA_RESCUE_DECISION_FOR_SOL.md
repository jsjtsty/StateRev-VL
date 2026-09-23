# StateRev-VL Composable Head 0 + L32 Delta Rescue：供 GPT-5.6 Sol 决策

## Executive decision

**结论：没有证据支持正协同；有证据支持 same-state 下的可叠加但次加性 margin movement；不支持继续单通道 rescue 线。**

本实验使用真正可组合的 downstream residual-delta intervention：

```text
delta32 = h32_donor_baseline - h32_target_baseline
h32' = h32_current + beta * delta32
```

joint 条件中，`h32_current` 来自已经完成 L24 Head 0 donor patch 的 target forward，因此没有被 L32 overwrite。与此前完整 block-output replacement 不同，这次 interaction 在定义上可识别。

在 strong same-state donor 下，joint 的 GT margin 高于 Head0-only 和 L32-delta-only，但：

```text
interaction = Delta_joint - Delta_Head0 - Delta_L32delta
```

在 overall、`t>=2`、`t>=3` 以及 weak/baseline-incorrect 子集都为负，且主要 beta 下 CI 明显低于 0。说明 Head 0 与 L32 delta 的贡献存在饱和、非线性或部分重叠，而不是正向协同。Joint 可以比任一单独干预更强，但没有超过两者效应之和。

Different-event donor 下，joint 会继续推动 donor-implied alternative state，并在 `t>=2/t>=3` 造成很高的 harmful flip 率。因此这不是一个安全的 categorical rescue mechanism。

## 1. Validity and artifact audit

结果目录：`outputs/vetbench/head0_downstream_delta_rescue_v1/`。

- 97 个 target prefixes。
- 20 条 target trajectories。
- 2 类 donor：`strong_same_state`、`different_event`。
- beta：`0.25`、`0.5`、`1.0`。
- 4 个条件：baseline、Head0-only、L32-delta-only、Head0+L32-delta。
- 预期并完成 2328 行 pair-level 结果。
- pair-level 仅作 descriptive；主要统计先按 target prefix 聚合，再按 target trajectory 做 cluster bootstrap 和 sign permutation。
- target/donor pairs 继承 frozen joint-rescue manifest，没有重新挑选。
- `delta32` 明确来自未干预 baseline target/donor captures。
- `beta=0` 的 joint=Head0-only 代数和 unit sanity 通过。
- beta>0 的 joint 与 L32-delta-only 不恒等；记录的最大 native-logit 差异为 `2.74679`。

因此，这次结果没有被此前的 full-block occlusion 问题解释掉。

## 2. Strong same-state donor

数值为 target-prefix 聚合后、trajectory-cluster bootstrap 的 mean，括号内为 95% CI。

### Native GT-margin

| beta | Head0-only | L32-delta-only | Head0+L32-delta |
|---:|---:|---:|---:|
| 0.25 | +0.404 [+0.302,+0.505] | +0.770 [+0.602,+0.935] | +1.102 [+0.842,+1.357] |
| 0.50 | +0.404 [+0.302,+0.505] | +1.356 [+1.053,+1.659] | +1.464 [+1.098,+1.836] |
| 1.00 | +0.404 [+0.302,+0.505] | +1.616 [+1.196,+2.078] | +1.640 [+1.185,+2.138] |

主多步子集：

| subset | beta | Head0-only | L32-delta-only | joint |
|---|---:|---:|---:|---:|
| t>=2 | 0.25 | +0.504 [+0.374,+0.631] | +0.939 [+0.726,+1.152] | +1.347 [+1.021,+1.675] |
| t>=2 | 0.50 | +0.504 [+0.374,+0.631] | +1.646 [+1.264,+2.030] | +1.779 [+1.316,+2.257] |
| t>=2 | 1.00 | +0.504 [+0.374,+0.631] | +1.940 [+1.385,+2.545] | +1.977 [+1.378,+2.630] |
| t>=3 | 0.25 | +0.465 [+0.333,+0.606] | +0.917 [+0.692,+1.154] | +1.263 [+0.925,+1.638] |
| t>=3 | 0.50 | +0.465 [+0.333,+0.606] | +1.567 [+1.167,+2.012] | +1.633 [+1.158,+2.185] |
| t>=3 | 1.00 | +0.465 [+0.333,+0.606] | +1.790 [+1.152,+2.567] | +1.790 [+1.100,+2.631] |

Joint 在所有这些条件下都比 Head0-only 强。相对于 L32-delta-only，beta=0.25 和 0.5 的 joint margin 也更高；beta=1 的增益很小且 CI 包含 0：

- overall：joint - L32-only `+0.024 [-0.029,+0.079]`。
- `t>=2`：`+0.037 [-0.029,+0.106]`。
- `t>=3`：约 `0.000 [-0.077,+0.085]`。

所以“joint 比任一单独干预更强”在连续 margin 上部分成立，但在主要多步 beta=1 条件下，并没有稳定超过 L32-delta-only。

## 3. Interaction：是否存在正协同？

### Strong same-state

| beta | overall | t>=2 | t>=3 | weak | baseline margin<0 |
|---:|---:|---:|---:|---:|---:|
| 0.25 | -0.072 [-0.110,-0.034] | -0.095 [-0.143,-0.045] | -0.119 [-0.179,-0.058] | -0.168 [-0.296,-0.064] | -0.151 [-0.267,-0.059] |
| 0.50 | -0.296 [-0.357,-0.236] | -0.370 [-0.450,-0.293] | -0.398 [-0.481,-0.317] | -0.594 [-0.803,-0.411] | -0.480 [-0.675,-0.320] |
| 1.00 | -0.380 [-0.462,-0.297] | -0.467 [-0.565,-0.367] | -0.465 [-0.567,-0.362] | -0.702 [-0.897,-0.526] | -0.579 [-0.765,-0.419] |

所有 CI 都位于 0 下方，sign permutation 也支持负向 interaction。负 interaction 随 beta 增大，且 weak 与 baseline-incorrect 组更强，表明 Head 0 contribution 并没有被 downstream delta 充分转化为额外 independent gain；两者更可能作用于重叠或受限的 commitment channel。

这不是此前 full-block intervention 的代数伪 interaction。这里确实执行了 additive downstream delta，但结果仍然不支持 positive synergy。

## 4. Categorical commitment

### Strong same-state donor

Joint wrong→correct rate：

| subset | beta=0.25 | beta=0.50 | beta=1.00 |
|---|---:|---:|---:|
| overall | +0.083 [+0.017,+0.167] | +0.150 [+0.067,+0.242] | +0.289 [+0.168,+0.422] |
| t>=2 | +0.117 [+0.017,+0.242] | +0.192 [+0.083,+0.317] | +0.325 [+0.175,+0.483] |
| t>=3 | +0.075 [0,+0.200] | +0.117 [+0.025,+0.242] | +0.275 [+0.125,+0.442] |
| weak | +0.037 [0,+0.093] | +0.120 [+0.042,+0.208] | +0.296 [+0.153,+0.458] |
| baseline margin<0 | +0.075 [+0.017,+0.150] | +0.142 [+0.067,+0.225] | +0.281 [+0.164,+0.410] |

Joint categorical correction increases with beta and is clearly detectable at beta=1. However, this should not be mistaken for positive interaction: L32-delta-only already produces substantial categorical movement, and joint beta=1 is nearly the same as L32-only.

The joint correct→wrong rate under strong same-state is 0 in the reported subsets where a valid baseline-correct denominator exists; the corresponding sparse endpoint should not be interpreted as proof of zero future harm.

### Different-event donor

At beta=1:

- target GT margin: `-2.798 [-3.792,-1.901]` overall;
- `t>=2`: `-2.802 [-3.976,-1.746]`;
- `t>=3`: `-2.896 [-4.054,-1.856]`;
- correct→wrong: `0.824 [0.676,0.941]` overall;
- `t>=2`: `0.875 [0.688,1.000]`;
- `t>=3`: `0.857 [0.643,1.000]`.

The joint donor-implied movement is strongly semantic but highly destructive when donor event conflicts with target GT. At beta=0.5 and 1.0, joint is more harmful than L32-delta-only in target GT margin:

- beta=1, overall joint - L32-only: `-0.418 [-0.544,-0.297]`;
- beta=1, `t>=2`: `-0.452 [-0.592,-0.321]`;
- beta=1, `t>=3`: `-0.473 [-0.637,-0.319]`.

Head 0 does not protect against the downstream delta's conflicting semantic direction.

## 5. Weak and baseline-incorrect targets

For strong same-state donors at beta=1:

- weak margin gain: `+3.361 [2.171,4.729]`;
- baseline margin `<0` gain: `+2.672 [1.745,3.694]`;
- weak wrong→correct: `+0.296 [0.153,0.458]`;
- baseline-incorrect wrong→correct: `+0.281 [0.164,0.410]`.

These gains are larger than Head0-only, but the interaction is more negative in these groups, respectively `-0.702 [-0.897,-0.526]` and `-0.579 [-0.765,-0.419]`. Thus weak cases benefit most in absolute terms, while also showing the clearest saturation/overlap rather than synergy.

## 6. Final answers

### A. Do Head 0 and downstream delta show identifiable positive synergy?

**No.** The intervention makes synergy identifiable, but same-state interaction is consistently negative across beta and the main `t>=2/t>=3` subsets. Joint can exceed either single intervention because both contribute to the same direction, but it does not exceed their sum.

### B. Does joint intervention convert Head 0's continuous movement into more categorical correction?

**Partly, but the improvement is attributable mainly to the strong L32-delta component, not a demonstrated Head0-to-downstream synergy.** At beta=1, joint wrong→correct is `+0.289` overall and `+0.325` at `t>=2`, but L32-delta-only is already `+0.289` overall and `+0.225` at `t>=2`; joint is only marginally above L32-only and not consistently so at `t>=3`. Different-event harmful flips also show that stronger commitment movement is not intrinsically safe.

### C. If there is still no synergy, should the rescue line stop?

**Yes, for the current single-channel rescue line.** The experiment has answered the composability question sufficiently: Head 0 provides a causal state-directed margin contribution, L32 provides a stronger downstream commitment/readout shift, and their additive combination is subadditive with donor-dependent harm. Continuing to tune Head 0/L32 amplitudes is unlikely to establish a clean, safe categorical rescue mechanism without introducing a new hypothesis and intervention.

## 7. Recommended scientific framing

Use:

> L24 Head 0 and L32 residual-delta interventions produce semantically directed native-state movement, but their combination is subadditive rather than synergistic. Downstream state commitment behaves as a distributed, capacity-limited channel: stronger interventions can cross some categorical boundaries, but conflicting event controls reveal substantial harmful transfer.

Avoid:

- “Head 0 and L32 cooperate synergistically.”
- “Joint intervention solves the categorical commitment problem.”
- “L32 is a clean causal state variable.”
- “No harm” based only on sparse/zero same-state harmful flips.

## 8. Artifact references

- `outputs/vetbench/head0_downstream_delta_rescue_v1/delta_rescue_manifest.json`
- `outputs/vetbench/head0_downstream_delta_rescue_v1/delta_rescue_pair_results.csv`
- `outputs/vetbench/head0_downstream_delta_rescue_v1/delta_rescue_target_results.csv`
- `outputs/vetbench/head0_downstream_delta_rescue_v1/delta_rescue_summary.json`
- `outputs/vetbench/head0_downstream_delta_rescue_v1/delta_rescue_report.md`

