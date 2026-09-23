# StateRev-VL：当前机制实验初步结论

## 给 GPT-5.6 Sol 的提交说明

本报告总结 `StateRev-VL` 当前已经完成的 validity、counterfactual mechanism、layer/module localization 和部分 L24 head discovery 结果。由于时间限制，未完成 validation head intervention 和 joint candidate-head intervention。因此，本文明确区分 held-out causal evidence 与 discovery-only hypothesis，不将未验证的 head 候选写成最终机制结论。

## 1. 研究问题

核心问题是视觉交换事件如何参与动态状态更新：

\[
S_{t-1} + E_t \rightarrow S_t
\]

其中 `E_t` 是当前 swap event，`S_t` 是模型需要维护的当前杯子状态。

主要反事实构造为：

\[
S_{cf}=apply\_swap(S_{prev,target}, E_{source})
\]

实验将 source trajectory 的真实 current-swap visual window 移植到 target trajectory，target 的其他历史保持不变。

## 2. 最重要的已验证结果

### 2.1 视觉 event intervention 成功

Raw visual current-window transplantation 能可靠地将模型内部 event representation 从 target event 推向 source event：

```text
E_target -> E_source
```

已有 event intervention efficacy 约为 `0.95`。这说明后续 state movement 不是仅由 pair label 或分析脚本中的 counterfactual 标签造成，而是由真实视觉窗口替换触发。

### 2.2 Native state logits 向 counterfactual state 移动

修正 pair-key join 后，使用模型自身 native Left/Middle/Right logits 计算的 counterfactual movement 约为：

| subset | native cf-vs-target movement |
|---|---:|
| overall | `+1.11` |
| `t>=2` | `+0.60` |
| `t>=3` | `+0.62` |

这不是错误 join 得到的 native accuracy。统计遵循 source pairs 先在 target prefix 内聚合，再以 target trajectory 为 cluster 进行 bootstrap/permutation；pair 数不能当作独立样本量。

`t>=2` 和 `t>=3` 仍有稳定 movement，因此结果不依赖 `t=1`。这一点很重要，因为初始状态 `S0` 在 prompt 中直接提供，`t=1` 不能承担主要机制结论。

### 2.3 Held-out late-layer block mediation

在 discovery/validation trajectory split 下，冻结层选择后，held-out validation 复现了从约 L24 到 L36 的 late-layer native mediation。以 `t>=2` 为例：

| layer | native sufficiency | native necessity |
|---:|---:|---:|
| L24 | `+0.184`，CI `[+0.027,+0.325]` | 正确负向 |
| L28 | `+0.424` | 正确负向 |
| L32 | `+0.511` | 正确负向 |
| L36 | `+0.592`，CI `[+0.322,+0.851]` | `-0.592`，CI `[-0.851,-0.322]` |

在 `t>=3`，L36 native sufficiency 约为 `+0.576`，95% CI `[+0.281,+0.876]`。

因此，目前最稳固的因果结论是：

> Qwen3-VL 存在一个可在 held-out trajectories 上复现的 late-layer activation pathway，该 pathway 会影响模型自身的 current-state readout，并使其朝视觉反事实所蕴含的 state 移动。

## 3. Event 到 state 的瓶颈

严格 event decoder 已修复为 discovery-only fit：decoder 的 scaler、PCA、regularization 和 C selection 均只使用 discovery trajectories，validation 不参与训练或交叉验证。

event effect 明显强于 native-state effect。例如 validation、attention-output L24、`t>=2`：

| endpoint | sufficiency | necessity |
|---|---:|---:|
| event | `+0.153`，CI `[+0.092,+0.222]` | `-0.106`，CI `[-0.160,-0.060]` |
| native state | `+0.080`，CI `[+0.005,+0.148]` | `-0.122`，CI `[-0.177,-0.068]` |

block-output mediation 也显示类似模式：event representation 的 causal effect 较强，而最终 native state commitment 明显较弱。

这支持：

```text
visual event encoding  ->  late-layer state readout
             strong              attenuated
```

当前可以称为 `event-to-state attenuation` 或 `event-to-state bottleneck`。但现有层分辨率下，event 与 state 的 onset 都约从 L24 开始，不能声称已经证明 event causal onset 严格早于 state causal onset。

## 4. Attention 与 MLP module decomposition

### 4.1 Attention

在 validation、`t>=2`：

| layer | event sufficiency | native-state sufficiency | event necessity | native-state necessity |
|---:|---:|---:|---:|---:|
| L24 | `+0.153` | `+0.080` | `-0.106` | `-0.122` |
| L28 | `+0.027` | `+0.041` | `-0.057` | `-0.025` |
| L32 | `+0.018` | `-0.030` | `-0.021` | `+0.009` |

L24 attention output 有小幅双向一致的 event/state effect，但 L28/L32 没有形成稳定、连续、双向的 standalone native-state module region。

### 4.2 MLP

在 validation、`t>=2`：

| layer | event sufficiency | native-state sufficiency | event necessity | native-state necessity |
|---:|---:|---:|---:|---:|
| L24 | `+0.068` | `-0.047` | `-0.065` | `+0.001` |
| L28 | `+0.058` | `+0.041` | `-0.044` | `-0.064` |
| L32 | `+0.013` | `-0.013` | `-0.007` | `+0.002` |

MLP 有弱 event mediation，但没有稳定的 native-state 双向 effect。因此不能据此声称 MLP 是明确的 state transformation module。

### 4.3 Module-level interpretation

目前最谨慎的解释是：

> late-layer block-level pathway 已被验证；attention 和 MLP 单独 patch 尚未稳定解释完整 native state effect。该 pathway 可能是 distributed 的，也可能依赖 attention/MLP 与后续 residual/readout 的联合 interaction。

不能把 L24 attention 的小幅 effect升级为完整 event-to-state circuit。

## 5. L24 held-out head localization

### 5.1 Attention 结构审计

L24 有 32 个 query heads、8 个 KV heads、128-dimensional head 和 GQA group size 4。干预点位于 head concat 后、`o_proj` 前，不是错误的 post-`o_proj` slicing。all-head patch 与完整 concat patch 一致，self patch 为数值零。

### 5.2 Frozen discovery candidates

只使用 discovery trajectories 冻结的候选为：

- event-routing group：Heads `[2,29]`；
- event-to-state candidate：Head `[0]`。

validation 未重新选择 head。完整 validation 包含 352 pairs、66 target prefixes 和 20 target trajectories。

### 5.3 Head 0 held-out validation

| subset | event suff. | event nec. | native-state suff. | native-state nec. | specificity suff. | specificity nec. |
|---|---:|---:|---:|---:|---:|---:|
| overall | `+0.019` | `-0.022` | `+0.181` | `-0.229` | `+0.193` | `-0.207` |
| `t>=2` | `+0.025` | `-0.030` | `+0.086` | `-0.125` | `+0.087` | `-0.102` |
| `t>=3` | `+0.035` | `-0.026` | `+0.103` | `-0.106` | `+0.104` | `-0.088` |

关键 trajectory-cluster CIs：

- `t>=2` native sufficiency：`+0.086`，CI `[+0.023,+0.142]`；
- `t>=2` native necessity：`-0.125`，CI `[-0.189,-0.070]`；
- `t>=3` native sufficiency：`+0.103`，CI `[+0.021,+0.184]`；
- `t>=3` native necessity：`-0.106`，CI `[-0.145,-0.065]`。

Head 0 也显著强于 matched controls：

| paired native-state contrast | sufficiency | necessity |
|---|---:|---:|
| `t>=2`, main minus same-event | `+0.102` `[+0.047,+0.155]` | `-0.119` `[-0.197,-0.049]` |
| `t>=2`, main minus history | `+0.094` `[+0.045,+0.143]` | `-0.126` `[-0.209,-0.059]` |
| `t>=3`, main minus same-event | `+0.099` `[+0.035,+0.161]` | `-0.089` `[-0.150,-0.030]` |
| `t>=3`, main minus history | `+0.110` `[+0.051,+0.168]` | `-0.121` `[-0.173,-0.064]` |

native specificity 的对应 paired CIs 也均不跨零。self patch 最大绝对误差约为 `8.6e-17`。

因此 Head 0 从 discovery candidate 升级为 held-out validated L24 final-prompt-token event-to-native-state causal head。

### 5.4 Heads 2/29 未通过 validation

冻结的 Heads 2/29 joint group 在 `t>=2` 的 event sufficiency 为 `+0.023`，但 necessity 为 `-0.013` 且 CI 跨零；native-state effects 方向不一致。因此不能将 Heads 2/29 报告为 validated event-routing heads。

### 5.5 Mediation accounting

相对于完整 L24 attention-output intervention，Head 0 恢复：

| subset | native suff./nec. | event suff./nec. |
|---|---:|---:|
| `t>=2` | `1.07x / 1.02x` | `0.16x / 0.28x` |
| `t>=3` | `1.30x / 1.03x` | `0.17x / 0.29x` |

ratio 大于 1 可能来自非线性或 suppressive interactions，不能解释为可加的 variance explained。Head 0 可以解释 L24 attention 的大部分 native-state effect，但只能解释少数 event effect；大部分 event routing 仍然是 distributed 的，或依赖其他 heads/head interactions。

### 5.6 解释边界

Head 0 intervention 位于 L24 final prompt token。它定位的是 late-layer event-to-state readout，而不是完整的 visual-token routing circuit。因此不能声称 Head 0 单独完成视觉 event extraction，也不能声称整个 L24 attention effect 可以线性分解为单 heads。

## 6. 当前机制分类

现有证据最支持：

1. `validated late-layer block-level state-readout mediation`；
2. `strong event encoding with weaker downstream state commitment`；
3. `validated L24 Head 0 final-token event-to-native-state pathway`；
4. `distributed event routing outside Head 0`。

现有证据不支持：

1. Head 0 是完整 visual-event extraction circuit；
2. 已定位完整的 distributed-head event-routing circuit；
3. event representation 等价于稳定、抽象的 current-state representation；
4. 强 categorical state commitment 已被证明。

## 7. 最终判定

### 对 layer/module 机制

`Restricted GO`

反事实视觉 event 通过 held-out reproducible late-layer pathway 影响 native state readout，且在 `t>=2/t>=3` 仍成立。

### 对 individual-head localization

`GO: validated Head 0 final-token event-to-state readout head`

Head 0 的 sufficiency、necessity、native specificity 和 matched-control contrasts 均通过 held-out validation。Heads 2/29 未通过 validation。

### 对 causal rescue

`Restricted GO for a narrowly scoped Head 0 causal-rescue experiment`

若继续，应只围绕冻结的 Head 0 做预注册 rescue，不再进行跨层或全 head sweep。当前结果本身已经足以报告 head-level readout localization。

## 8. 可直接提交的摘要

> Raw visual transplantation reliably changes the model's internal event representation from `E_target` to `E_source`, with intervention efficacy of approximately 0.95. Correctly joined native logits move toward the counterfactual state `S_cf`, including at `t>=2` and `t>=3`. Bidirectional block-output mediation identifies a held-out late-layer causal region beginning around L24. A strict discovery-only event decoder shows stronger event mediation than downstream native-state commitment, supporting an event-to-state bottleneck. Within L24 attention, the discovery-frozen Head 0 candidate validates on 20 held-out trajectories: at `t>=2`, native-state sufficiency is +0.086 (95% CI [+0.023,+0.142]) and necessity is -0.125 (95% CI [-0.189,-0.070]); at `t>=3`, the corresponding effects are +0.103 and -0.106. Native specificity and paired contrasts against same-event and matched-history controls also remain significant. Head 0 recovers approximately all of the L24 attention native-state effect but only 16-29% of its event effect. The frozen Heads 2/29 event-routing group does not validate bidirectionally. The evidence therefore supports a validated L24 Head 0 final-token event-to-native-state readout pathway embedded within a more distributed event-routing circuit. It does not support Head 0 as a complete visual-event extraction circuit or imply strong categorical state commitment.

## 9. 产物索引

- `outputs/vetbench/circuit_localization_v1/`
- `outputs/vetbench/circuit_localization_v2/`
- `outputs/vetbench/head_localization_l24_v1/head_localization_final_report.md`
- `outputs/vetbench/head_localization_l24_v1/head_final_effects.csv`
- `outputs/vetbench/head_localization_l24_v1/head_control_contrasts.csv`
- `outputs/vetbench/head_localization_l24_v1/head_localization_final_summary.json`
