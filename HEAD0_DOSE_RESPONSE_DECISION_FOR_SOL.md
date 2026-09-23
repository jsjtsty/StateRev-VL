# StateRev-VL Head 0 Dose-Response：供 GPT-5.6 Sol 决策

## Executive decision

**连续 causal state-margin：GO。categorical rescue：Restricted GO。Head 0 单独完成 state commitment：NO-GO。**

在冻结的 Qwen3-VL L24 Head 0、final prompt token、pre-`o_proj` slice 干预下，沿

```text
h_alpha = h_target + alpha * (h_donor - h_target)
```

增加 strong same-state donor 的剂量，会稳定、近似单调地提高 native GT-state margin。这个效应在真正的多步条件 `t>=2` 和 `t>=3` 仍然存在。different-event donor 则使 donor 所隐含的 counterfactual state margin 随剂量增加，说明响应具有事件/状态语义方向，而不是单纯增加置信度。

但是，连续 margin 移动很少跨越 categorical decision boundary；在 `alpha=2` 仍只有少数严格错误样本被纠正。Head 0 更像一个可因果控制的 state-directed readout channel，而不是足以完成最终 state commitment 的完整电路。`alpha=2` 还带来更大的整体 logit 位移和 entropy 改变，因此不能把最大剂量直接称为干净的 rescue。

## 1. Protocol and validity

- 固定 layer：Qwen3-VL decoder L24。
- 固定 head：Head 0。
- 固定位置：final prompt token，attention `o_proj` 之前的 head-concat slice。
- alpha：`[0, 0.25, 0.5, 1, 1.5, 2]`。
- donor：已有 frozen strong same-state donor，以及已有 different-event donor；没有重新挑 pair。
- 主统计：先按 target prefix 聚合，再以 target trajectory 为 cluster 做 bootstrap CI。
- pair 不是独立样本；50 条轨迹此前已经用于 discovery/validation，因此本实验不是 untouched new-trajectory replication。

产物完整：97 个 target prefixes、20 条 target trajectories、2 类 donor、6 个 alpha，共 1164 行 pair-level 结果。`alpha=0` 的最大误差为 0；`alpha=1` 与 Head 0 rescue v1 的 logits 最大绝对误差为 `4.44e-16`。这些检查支持剂量插值实现和旧实验的兼容性。

## 2. A：是否存在清晰 dose-response？

### Strong same-state donor：GT margin

数值为 trajectory-cluster mean，括号内为 95% bootstrap CI；endpoint 是相对 alpha=0 的 `Delta GT margin`。

| alpha | overall | t>=2 | t>=3 |
|---:|---:|---:|---:|
| 0.25 | +0.097 [+0.055,+0.137] | +0.126 [+0.073,+0.176] | +0.117 [+0.056,+0.177] |
| 0.5 | +0.212 [+0.154,+0.268] | +0.257 [+0.187,+0.325] | +0.238 [+0.165,+0.308] |
| 1 | +0.404 [+0.303,+0.501] | +0.504 [+0.377,+0.627] | +0.465 [+0.333,+0.602] |
| 1.5 | +0.612 [+0.449,+0.771] | +0.762 [+0.562,+0.956] | +0.700 [+0.500,+0.915] |
| 2 | +0.831 [+0.628,+1.031] | +1.030 [+0.778,+1.279] | +0.954 [+0.696,+1.244] |

固定的 dose-response 判据要求相邻 alpha 的 effect 方向一致、相邻 CI 下界不低于 0，且 dose slope 的 CI 为正。该判据在 overall、`t>=2`、`t>=3` 均通过。对应的 cluster dose slope 为：

- overall：`+0.414 [0.310, 0.515]`
- `t>=2`：`+0.513 [0.385, 0.639]`
- `t>=3`：`+0.474 [0.345, 0.617]`

分组上，alpha=2 的 margin gain 为：

- weak：`+1.641 [1.157, 2.166]`
- strong：`+0.291 [0.101, 0.520]`
- baseline margin `<0`：`+1.276 [0.866, 1.716]`

weak 组增益明显更大，支持“弱 commitment 更可被 Head 0 推动”的粗粒度交互。但逐 prefix 的连续回归斜率为 `-0.046 [-0.104, 0.021]`，`p=0.172`，所以不能声称 baseline margin 每下降一点，rescue gain 就严格线性增加。

### Different-event donor：donor-implied counterfactual margin

对于 different-event donor，使用 donor event 所隐含 state 的 margin endpoint：

| alpha | overall | t>=2 | t>=3 |
|---:|---:|---:|---:|
| 1 | +0.444 [+0.289,+0.614] | +0.447 [+0.275,+0.647] | +0.456 [+0.271,+0.669] |
| 2 | +0.901 [+0.557,+1.278] | +0.959 [+0.564,+1.405] | +0.946 [+0.515,+1.433] |

该方向性在 `t>=2` 和 `t>=3` 没有消失。它是 Head 0 含有语义方向信息的关键证据：换成隐含另一状态的 donor，不是产生同向的 generic confidence，而是推动另一状态的 native margin。

## 3. B：categorical rescue 是否显著且实用？

结论必须与连续 margin 分开。

Strong same-state donor 的 trajectory-cluster wrong-to-correct 估计为：

- alpha=1：`+0.050 [0, 0.133]`
- alpha=2：`+0.079 [0.013, 0.163]`

alpha=2 下，严格 baseline margin `<0` 的 63 个目标中只有 `3/63` 跨过 decision boundary；首次 crossing 分布为 alpha=1、1.5、2 各 1 个。也就是说，alpha=2 的 cluster-level categorical effect 可以统计检测到，但行为修正规模仍然很小，不能称为可靠的 model correction。

观察到的 strong same-state correct-to-wrong flip 为 0；这在本数据上令人放心，但 all-zero endpoint 的非参数 bootstrap 是退化的，不能据此证明未来样本 harm rate 为 0。

Different-event donor 在 alpha=2 有：

- donor-implied state flips：`+0.184 [0.026, 0.368]`
- correct-to-wrong：`+0.221 [0.059, 0.412]`
- donor-state boundary crossing：`3/36`

这进一步说明大剂量能推动语义方向，但也说明当 donor event 与 target GT 不一致时，过强干预会造成实质性错误。它不应被当成 rescue 成功，而应作为方向性和剂量安全性的 control。

## 4. C：是否支持 downstream/distributed commitment bottleneck？

**支持，但不是唯一证明。**

证据链是：

1. alpha 从 0 到 2 时，same-state donor 的 native GT margin 在 overall、`t>=2`、`t>=3` 都稳定增加。
2. weak targets 的连续增益远大于 strong targets。
3. 但严格错误样本中绝大多数仍没有跨越 boundary；弱组即使平均 margin gain 很大，也没有形成可靠的 categorical flip。
4. different-event donor 能把 margin 推向 donor-implied state，说明 Head 0 不是只改变无方向的 confidence。

最符合的工作模型是：

```text
distributed event encoding/routing
        -> L24 Head 0 state-directed readout contribution
        -> additional downstream/distributed commitment bottleneck
```

这不能区分具体 bottleneck 是后续层、其他 heads、logit readout 竞争，还是多个因素共同造成。当前实验支持“Head 0 contribution + commitment bottleneck”，不支持“Head 0 是完整 state circuit”。

## 5. alpha=2 的非特异性和安全性

alpha=2 的 strong same-state 结果：

- desired-state specificity：`+0.036 [-0.127, 0.232]`，CI 跨 0；
- centered three-state logit L2：`+0.950 [0.777, 1.141]`；
- entropy change：`+0.096 [0.048, 0.145]`；
- new unintended prediction：`+0.030 [0, 0.060]`。

因此不能声称 alpha=2 只提升 GT logit。它造成了可观的整体三状态 logit 位移和分布变化。对于 different-event donor，alpha=2 的 harmful flip 已为 `+0.221 [0.059, 0.412]`，显示过强的 semantic intervention 可能明显破坏原本正确的答案。

`alpha=1` 或 `1.5` 是更保守的操作点候选；具体选择取决于优先级是减少非特异性扰动还是最大化 margin movement。alpha=2 不应作为默认安全 rescue 强度。

## 6. 最终回答

### A. 是否存在清晰 dose-response？

**是。** Strong same-state donor 的 GT margin 在 alpha 网格上稳定增加，且 overall、`t>=2`、`t>=3` 都通过预注册 dose-response 判据。Different-event donor 的 donor-implied counterfactual margin 也呈相同的剂量方向。

### B. 增强 Head 0 是否显著提高 categorical rescue？

**alpha=2 下统计上可检测，但实践上很弱，因此 Restricted GO。** 只有 3/63 严格 baseline-incorrect targets crossing；连续 margin movement 远强于分类翻转。

### C. 是否支持 downstream/distributed commitment bottleneck？

**一致支持，但不是唯一证明。** Head 0 能够因果地把 native margin 向语义指定方向推动，却通常不足以完成 categorical commitment；这与 downstream/distributed bottleneck 相符。

## 7. 给 GPT-5.6 Sol 的建议

建议采用以下最终标签：

- `GO`：Head 0 存在可复现、双向、语义定向的 causal state-margin contribution；
- `Restricted GO`：连续 native-margin rescue / dose-response；
- `NO-GO`：Head 0 单独实现可靠 categorical behavioral rescue；
- `NO-GO`：将 Head 0 描述为完整或唯一的 visual-event-to-state circuit。

不建议再进行宽泛的 layer/head 搜索。若必须做最后一个实验，应只做预注册的窄范围 causal rescue，测试多个 downstream/head 的联合干预能否把连续 margin movement 转化为 categorical commitment，并保留 different-event harm control。否则可以停止实验，把论文主张限定为：

> L24 Head 0 is a causally controllable, semantically state-directed readout channel, while final categorical commitment remains distributed and subthreshold for most targets.

## 8. Artifact references

- `outputs/vetbench/head0_dose_response_v1/dose_response_manifest.json`
- `outputs/vetbench/head0_dose_response_v1/dose_response_pair_results.csv`
- `outputs/vetbench/head0_dose_response_v1/dose_response_target_results.csv`
- `outputs/vetbench/head0_dose_response_v1/dose_response_summary.json`
- `outputs/vetbench/head0_dose_response_v1/dose_response_report.md`
