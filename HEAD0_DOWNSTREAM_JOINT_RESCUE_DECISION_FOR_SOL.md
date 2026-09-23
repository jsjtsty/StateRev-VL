# StateRev-VL Frozen Head 0 + Downstream Joint Rescue：供 GPT-5.6 Sol 决策

## Executive decision

**A：没有证明 Head0 + downstream 的正协同，也没有证明 joint 突破了 categorical commitment。**

L32/L36 block-only 在 strong same-state donor 下比 Head0-only 产生大得多的 native GT-margin 增益，并且 categorical wrong→correct crossing 明显增加；但是按当前 frozen intervention 定义，`Head0 + L32/L36` 的结果逐行精确等于对应 block-only。原因是完整 final-token downstream block-output replacement 会覆盖该 token 的全部上游 residual，包括 L24 Head 0 的贡献。因此观察到的 interaction 不是正 interaction，而是 downstream replacement 的结构性 occlusion。

最稳妥的结论是：**L32/L36 是更强的 downstream state-readout/commitment intervention，但当前 joint 设计不能检验 Head 0 与 downstream 的加性或协同关系。** Different-event donor 下它也会产生强烈 donor-implied counterfactual movement，同时造成大量 harmful flips，说明强度很高但不是安全的 rescue。

## 1. Artifact and validity audit

使用产物：`outputs/vetbench/head0_downstream_joint_rescue_v1/`。

- 97 个 target prefixes；20 条 target trajectories；2 类 donor；6 个条件；共 1164 行。
- shard、target coverage、condition/donor coverage 和 duplicate key 检查通过。
- baseline 在两种 donor 记录中的重复 logits 一致。
- `Head0 + L32` 与 `L32 only` 的逐 pair logits 最大差异为 0（浮点容差内）。
- `Head0 + L36` 与 `L36 only` 的逐 pair logits 最大差异为 0（浮点容差内）。
- pair-level 只作 descriptive；推断先按 target prefix 聚合，再按 target trajectory 做 bootstrap/sign permutation。
- 这 50 条轨迹此前已经用于 discovery/validation，因此不是 untouched new-trajectory replication。

## 2. Strong same-state donor

数值为 trajectory-cluster mean，括号内为 95% cluster bootstrap CI。

### Native GT-margin movement

| subset | Head0 only | L32 only | L36 only | Head0+L32 | Head0+L36 |
|---|---:|---:|---:|---:|---:|
| overall | +0.404 [+0.302,+0.505] | +1.624 [+1.185,+2.093] | +1.659 [+1.221,+2.134] | +1.624 [+1.185,+2.093] | +1.659 [+1.221,+2.134] |
| t>=2 | +0.504 [+0.374,+0.632] | +1.952 [+1.380,+2.570] | +1.996 [+1.423,+2.624] | +1.952 [+1.380,+2.570] | +1.996 [+1.423,+2.624] |
| t>=3 | +0.465 [+0.333,+0.606] | +1.798 [+1.131,+2.590] | +1.844 [+1.175,+2.650] | +1.798 [+1.131,+2.590] | +1.844 [+1.175,+2.650] |

因此 downstream block-only 的连续 margin effect 在 `t>=2`、`t>=3` 仍然很强。L36 的 margin effect 略大于 L32：

- overall：L36 比 L32 高约 `+0.035`；
- `t>=2`：高约 `+0.045`；
- `t>=3`：高约 `+0.046`。

这些差异很小，不能据此宣称 L36 明显优于 L32；从当前结果看，两者都是强 downstream intervention。

### Categorical crossing

| subset | Head0 only wrong→correct | L32 only | L36 only | Head0+L32 | Head0+L36 |
|---|---:|---:|---:|---:|---:|
| overall | +0.050 [0,+0.133] | +0.339 [+0.202,+0.483] | +0.289 [+0.164,+0.421] | +0.339 [+0.202,+0.483] | +0.289 [+0.164,+0.421] |
| t>=2 | +0.075 [0,+0.200] | +0.375 [+0.217,+0.542] | +0.325 [+0.175,+0.483] | +0.375 [+0.217,+0.542] | +0.325 [+0.175,+0.483] |
| t>=3 | +0.050 [0,+0.150] | +0.325 [+0.158,+0.508] | +0.275 [+0.117,+0.442] | +0.325 [+0.158,+0.508] | +0.275 [+0.117,+0.442] |

L32 的 categorical rescue rate 高于 L36，尽管 L36 的 continuous margin 略大。这说明 margin gain 和 categorical commitment 并非同一个 endpoint，L36 的最大 margin 不能直接转译为最佳行为 rescue。

Strong same-state 下 correct→wrong 的 cluster estimate：

- L32：overall `+0.059 [0,+0.176]`；`t>=2` `+0.062 [0,+0.188]`；`t>=3` `+0.071 [0,+0.214]`。
- L36：数值相同。

这些 harm CI 包含 0，但不能把观察到的低/零显著性解释为绝对安全。

### Weak and baseline-incorrect targets

Strong same-state 下：

- weak：L32 `+3.346 [2.214,4.641]`，L36 `+3.374 [2.234,4.699]`；
- baseline margin `<0`：L32 `+2.681 [1.773,3.669]`，L36 `+2.683 [1.779,3.670]`。

这表明 downstream patch 对弱/错误 baseline 的 margin 推动尤其大。但仍不能只凭连续 margin 断言可靠 correction；categorical crossing 受 decision boundary 和竞争 logits 共同决定。

## 3. Different-event donor：方向性与风险 control

Different-event donor 不应提高 target GT，而应推动 donor-implied state。

### Target GT margin

| subset | Head0 only | L32 only | L36 only |
|---|---:|---:|---:|
| overall | -0.232 [-0.365,-0.103] | -2.374 [-3.247,-1.582] | -2.419 [-3.277,-1.632] |
| t>=2 | -0.203 [-0.332,-0.077] | -2.348 [-3.387,-1.404] | -2.383 [-3.403,-1.450] |
| t>=3 | -0.240 [-0.371,-0.117] | -2.425 [-3.437,-1.523] | -2.421 [-3.412,-1.529] |

### Donor-implied counterfactual margin

| subset | Head0 only | L32 only | L36 only |
|---|---:|---:|---:|
| overall | +0.444 [+0.287,+0.616] | +3.218 [+2.196,+4.332] | +3.270 [+2.239,+4.395] |
| t>=2 | +0.447 [+0.274,+0.649] | +3.388 [+2.136,+4.755] | +3.432 [+2.162,+4.822] |
| t>=3 | +0.456 [+0.271,+0.671] | +3.481 [+2.231,+4.848] | +3.481 [+2.219,+4.865] |

这确认 L32/L36 不是 generic confidence boost：different-event donor 下它们强烈推动 donor-implied alternative state，并显著压低 target GT margin。与此同时，overall correct→wrong 为约 `0.824 [0.676,0.941]`，`t>=2` 约 `0.875 [0.688,1.000]`，`t>=3` 约 `0.857 [0.643,1.000]`。因此 downstream full-block transplant 是强烈的 semantic intervention，但对原本正确的 target 极具破坏性。

## 4. Interaction 是否为正？

定义：

```text
interaction = Delta_joint - Delta_Head0 - Delta_Block
```

对于 strong same-state donor，trajectory-cluster interaction 为：

- L32：overall `-0.404 [-0.505,-0.302]`；`t>=2` `-0.504 [-0.632,-0.374]`；`t>=3` `-0.465 [-0.606,-0.333]`。
- L36：数值相同。

对于 different-event donor，interaction 为正：

- L32/L36：overall `+0.232 [+0.103,+0.365]`；
- `t>=2`：`+0.203 [+0.077,+0.332]`；
- `t>=3`：`+0.240 [+0.117,+0.371]`。

这些 interaction 数值不能作为 synergy 证据。由于 `Delta_joint = Delta_Block` 恒成立，代数上：

```text
interaction = -Delta_Head0
```

所以 strong same-state 中 Head0 本身为正，interaction 必然为负；different-event 中 Head0 对 target GT 为负，interaction 必然为正。它只是上游 Head0 effect 被 full downstream replacement 覆盖的结果，不是模块间协同或拮抗的可识别估计。

## 5. 对三个最终问题的回答

### A. Head0 + downstream 是否突破 categorical commitment？

**本实验不能支持这个说法。** L32/L36-only 确实比 Head0-only 有更多 wrong→correct crossing，且在 `t>=2/t>=3` 仍存在；但 joint 与 block-only 完全相同，所以不能归因于 Head0 与 downstream 的联合，也不能证明一个可泛化、安全的 commitment rescue。different-event donor 的高 harmful-flip 率进一步反对把它称作稳健 rescue。

### B. 哪个 downstream block 更有增益？

按连续 GT margin，L36 略高；按 wrong→correct crossing，L32 略高。差异没有形成清晰、可靠的单一赢家。更重要的是，L32/L36 都强于 Head0-only，但两者都是 full-block overwrite intervention，不能解释其内部是“转换”还是“直接读出/覆盖”。

### C. 若 joint 仍主要只推 margin，是否应停止 rescue 线？

**应停止当前这种单通道/完整 block replacement rescue 线。** 结果已经足够支持 distributed commitment bottleneck 的解释：Head0 能推动 native margin，L32/L36 能更强地改变 readout/commitment，但 semantic movement 与 categorical correction 并不等价；强 downstream intervention 还会在 different-event control 下大规模破坏答案。

不建议把当前 interaction 写成正协同，也不建议继续用 full final-token block replacement 做 joint rescue。若科学问题必须继续，唯一合理的后续应是另行预注册一种真正可组合的 intervention（例如 downstream residual delta 的受控注入或小剂量 interpolation），并重新定义 interaction；这已经不是本 frozen 实验的结论。

## 6. 最终标签

- `GO`：L32/L36 存在强 native margin / semantic readout influence；
- `Restricted GO`：downstream block 可以提高部分 categorical crossing，但存在明显 donor-dependent harm；
- `NO-GO`：声称 Head0 + downstream 已证明正协同；
- `NO-GO`：声称已突破一个稳定、可泛化的 categorical commitment barrier；
- `GO`：将结果纳入“distributed commitment bottleneck / full-block downstream overwrite”解释。

## 7. Artifact references

- `outputs/vetbench/head0_downstream_joint_rescue_v1/joint_rescue_manifest.json`
- `outputs/vetbench/head0_downstream_joint_rescue_v1/joint_rescue_pair_results.csv`
- `outputs/vetbench/head0_downstream_joint_rescue_v1/joint_rescue_target_results.csv`
- `outputs/vetbench/head0_downstream_joint_rescue_v1/joint_rescue_summary.json`
- `outputs/vetbench/head0_downstream_joint_rescue_v1/joint_rescue_report.md`

