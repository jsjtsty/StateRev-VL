# StateRev-VL：视觉交换事件到动态状态更新的机制审计

日期：2026-09-12  
项目：StateRev-VL  
任务：VET-Bench cup/shell game  
模型：Qwen3-VL-8B-Instruct

## 摘要

本项目研究多模态 Transformer 是否把视觉交换事件 E_t 与历史状态
S_{t-1} 组合成当前状态 S_t：

    S_{t-1} + E_t -> S_t

在统一 Transformers 输入和 forward pipeline 下，事件信息非常强地存在于
隐藏状态中；前一状态只有中等强度，当前状态表示明显更弱。原始视觉
current-window transplantation 能稳定地把内部 event representation 从
E_target 推向 E_source，并使 native state logits 朝 counterfactual state
移动，且在 t>=2、t>=3 仍存在。但 categorical argmax 很少翻转，且三状态
特异性不够稳定。

后续因果定位显示，主要 native mediation 位于 late layers，约从 L24 开始；
L24 attention 具有双向 event/native-state effect。L24 的 Head 0 是一个经过
held-out 验证的 event-to-native-state causal head。Head 0 的干预产生清晰的
连续 margin movement 和 dose-response，也能被错误 event donor 反向驱动，
但单独或与 L32 downstream intervention 组合时，categorical commitment 仍然
有限。历史替换可以显著改变 late hidden representation，却没有可靠改变
native prediction 或 native GT margin。

当前最稳妥的结论是：

> 模型能够读取视觉 event，并且该 event causal influence 可以到达 native
> state readout；然而多步状态维护中的 categorical commitment 很弱。结果
> 更符合 late distributed commitment bottleneck / representation-readout
> gap，而不是已经证明存在一个干净、独立、可泛化的 abstract current-state
> variable。

这是一项机制审计结果，不应把“线性可解码”自动写成“模型功能性使用了该
representation”。

## 1. 项目问题与实验对象

### 1.1 数据与输入

- VET-Bench cup/shell game。
- 50 条 trajectory，每条 5 次 swap，共 250 个 prefix samples。
- 状态空间：Left、Middle、Right。
- 每次事件是两个位置之间的交换；共有六种 directed transition：
  L->M、M->R、R->L、M->L、R->M、L->R。
- 主 regime：controlled_8fps。
- 采样帧数约为 t=1..5 的 31/47/63/79/95。
- 所有主要行为、hidden-state、native-logit 和 intervention 输入都复用
  scripts/state_rev_input_pipeline.py。

统一 pipeline 保存并检查 sampled frame indices、frame fingerprints、
input_ids SHA256、pixel_values shape/hash、video_grid_thw、prompt hash、
model/backend/version。机制实验采用与 hidden extraction 一致的
Transformers forward 路径。

### 1.2 模型与实现环境

项目早期记录的环境为 Transformers 5.15.1、PyTorch 2.13；目标服务器最近
实际运行记录为 Transformers 5.16.1、PyTorch 2.14.0+cu126。vLLM 与
Transformers 的行为存在实质差异，因此最终机制分析以 Transformers-aligned
artifacts 为准；版本差异应在复现实验时明确记录。

推理使用 bf16、greedy decoding、do_sample=False、max_new_tokens=32。
隐藏状态 probe 的主位置是最后一个 input token。

### 1.3 统计原则

pair rows 仅作 descriptive。多个 source pair 属于同一个 target prefix 时，
先在 target prefix 内聚合，再以 target trajectory 为独立 cluster。
置信区间使用 trajectory-cluster bootstrap，方向检验使用 trajectory-level
paired sign/permutation。870 pairs 绝不能当作 n=870 个独立样本。

主机制结论优先看 t>=2 和 t>=3；t=1 中 S_0 直接写在 prompt 中，不能承担
主要的多步状态维护结论。

## 2. 初始假设与审计策略

最初的工作假设是：模型可能在 hidden state 中保留 S_{t-1} 和 E_t，但
没有完成 state composition，表现为 stale / belief inertia。

审计过程中发现，stale 不是可靠的主线：模型的 Middle response bias、S0
persistence 和其他答案偏置可以产生类似表面行为。因此实验路线被调整为：

1. 统一输入、backend 和统计口径；
2. 区分 event/state 的 representation 与 native behavior；
3. 直接删除或移植当前视觉 event；
4. 验证 intervention efficacy；
5. 用 native logits、OOD decoder 和 causal activation intervention 分开
   检验 event-to-state pathway；
6. 最后定位 layer、module 和 head，并测试 rescue 是否能突破 categorical
   commitment。

## 3. Validity 与 reproducibility audit

### 3.1 Backend mismatch

早期行为结果来自 vLLM，而 hidden states/native logits 来自 Transformers。
同一视频、问题和输入下，两者并不等价：

| 指标 | 旧 vLLM | Transformers-aligned |
|---|---:|---:|
| state correct | 91/250 = 36.4% | 81/250 = 32.4% |
| event correct | 122/250 = 48.8% | 173/250 = 69.2% |
| clean revision | 51 | 60 |
| revision success | 15 | 17 |
| canonical stale failure | 31 | 26 |
| clean maintenance | 7 | 23 |

state answer agreement 仅 73.6%，event answer agreement 仅 56.4%。因此旧
vLLM behavior 不能直接与 Transformers hidden-state mechanism 结果拼接。
最终主分析使用 Transformers-aligned behavior。

### 3.2 0.549 与 0.658 的 probe discrepancy

这是统计量误读，不是同一个 estimator 得出两个结果。

- 0.549 是 10 个 trajectory-group split 的 mean point estimate 中，跨层
  取最大值的结果，位于 L33 附近。
- 0.658 是某个 split 的 maxT observed value，即先在该 split 的 37 层中
  取最大值，再作为 observed statistic。

后续 audit 脚本在相同输入上并列输出 split、layer、subset、PCA/scaler、C、
metric、maxT/null、pooling/token position，确认两者不是 hidden states 不同，
而是 aggregation 顺序不同。不能把这两个数字当成同一 probe 的重复实验
结果。

### 3.3 输入充分性与 frame coverage

2 fps event prompt 的行为准确率约 0.688；但每个 t 的视频在 swap window
结束时停止，没有 settled post-swap frame。模型看到的是 swap 过程中的 5 个
sampled frames 和更早历史帧。event accuracy 随 t 约为 0.92/0.70/0.78/0.58/0.46，
Left-Right 是最难的 swap，约 0.494。

从 2 fps 增到 8 fps 没有显著改善：state 约从 0.324 到 0.340，event 约从
0.688 到 0.644，paired trajectory CI 均跨 0。raw frame count 不是主要解释。

### 3.4 旧文档与代码问题

审计修正了两个文档问题：

- “state-correct all 195” 实际相关 candidate subset 是 91；代码 mask 正确。
- “prev wrong” 标签与代码含义相反；revision success 实际要求
  prev_state_correct=True。

另有若干实现问题在后续阶段修复：错误 pair-key join、event decoder
validation leakage、L36 pre/post-final-norm endpoint 不一致、tuple return
处理、self/patched column 缺失、以及完整 block replacement 覆盖上游 Head 0
贡献的问题。

## 4. 行为基线：event-state gap 与 stale 重新解释

在 Transformers-aligned 250 rows 上：

- state accuracy：81/250 = 32.4%；
- event accuracy：173/250 = 69.2%；
- clean revision：60；其中 success 17，failure 43；canonical stale 26；
- t=0 初始状态准确率 50/50，因为状态名称直接在 prompt 中。

响应分布明显偏向 Middle：L=0.208、M=0.740、R=0.052。
response-bias null 的 stale rate 为 0.439--0.716，而真实模型 stale rate 约
0.363，并未高于 null。因而 stale 不是独立的机制 signature；很多错误答案
是 Middle 或其他位置，而不是简单重复 swap 前状态。

在同一 regime 内，event-correct 但 state-wrong 的比例约为 event-correct
rows 的三分之二。2 fps 为 121/172，8 fps 为 106/161；其中只有少部分是
stale，其余是其他偏置答案。这说明 event-to-state gap 存在，但不能用
belief inertia 单独描述。

文本 cue rescue 进一步显示，给出正确的 previous state + event 文本可以
提高答案；只给 previous state 的收益很小。text-only correct cue 的表现
不低于 video+cue。该结果证明模型能按显式文字完成 swap computation，但不
证明它能从视觉 event 自主形成当前 state。

## 5. Ordinary probing 与 symbolic composition

### 5.1 Hidden-state linear decodability

在 state-question forward 的 hidden states 上，使用 train-only scaler、PCA、
trajectory-group split、L2 logistic probe 和 maxT/permutation：

| regime | event balanced accuracy | current state | previous state |
|---|---:|---:|---:|
| 2 fps | 1.000 | 0.564 | 0.608 |
| controlled 8 fps | 1.000 | 0.549 | 0.598 |

event 是强线性 code；previous state 是中等强度 code；current state 明显弱。
这支持 event encoded, state representation weaker 的表示层描述，但不
等于证明模型在生成答案时功能性读取了该 code。

### 5.2 Symbolic composition

把 out-of-fold 解码出的 previous state 与 event 重新应用交换规则，得到
symbolic state：

- overall 约 0.576；
- true transitions 约 0.558；
- canonical stale subset 约 0.538；
- canonical stale 上 native state answer 为 0%。

这说明一个外部显式组合器可以利用被解码的 operands；它同时提示 native
Transformer readout 没有可靠完成同样的 composition。但因为 symbolic
composition 使用的是外部 probe 的预测，不能当作模型内部存在 abstract
state variable 的证明。

## 6. Mechanism Gate：从控制到真实 event transplant

### 6.1 旧 freeze control 不充分

旧方法把当前 swap window 的帧替换成该 window 第一帧。这样仍可能保留
event-specific 的静态视觉线索，因此不能证明 current event evidence 被删除。

no_current_event 条件后来定义为：当前 t window 的所有 sampled frames 都
替换为 swap window 开始前最后一张 sampled frame 的重复；frame count、slot、
timestamps、grid 和 token budget 保持不变。该条件真正移除了 current-window
视觉内容，同时保持输入长度和位置结构尽量一致。

### 6.2 Raw visual transplantation

source-target pair 固定满足：same t、same S_{t-1}、same fps/frame count/grid、
不同 event，且 S_cf = apply_swap(S_prev_target,E_source) 与 S_target 不同。
hybrid video 保留 target history，只替换 target current-swap window 为 source
真实 sampled frames。controls 包括 target self、same-event source、matched
history 和 temporal shuffle；shuffle 不被当作主要 negative control，因为
event identity 可能保留。

event intervention efficacy 约 0.95，说明 raw visual transplant 确实能把
event representation 推向 source event。后续所有 state interpretation 都
把该 efficacy 作为前提。

## 7. Native state logits 与 OOD state decoder

### 7.1 修正 pair-key 后的 native movement

原始 pair key 是 target_t_from_source，baseline/GT key 是 target_t。
修正映射后，对每个 pair 使用：

    native_cf_shift =
    [(L_cf - L_target)_hybrid - (L_cf - L_target)_baseline]

结果：

| subset | native cf shift |
|---|---:|
| overall | +1.11，95% CI 约 [+0.90,+1.33] |
| t>=2 | +0.596，95% CI 约 [+0.444,+0.743] |
| t>=3 | +0.624，95% CI 约 [+0.435,+0.819] |

这是真实 native logits 的连续移动，不是错误 join 得到的 accuracy。

### 7.2 三状态特异性

进一步分别计算 delta_logit_cf、delta_logit_target、delta_logit_third，
以及 native_specificity = delta_cf - max(delta_target,delta_third)。

主结果：

- t>=2 specificity 约 +0.131，CI [-0.024,+0.280]；
- t>=3 specificity 约 +0.164，CI [+0.003,+0.328]，但稳定性有限；
- flip-to-cf probability change 很小，t>=2 约 +0.009；
- main specificity 相对 same-event/history controls 的差异更支持主方向，
  但 t>=2 的主 specificity CI 仍跨 0。

因此该阶段是 Restricted GO，不是 Strong GO。可说 native logits 有方向性
counterfactual influence，不能说已证明 categorical native state 被特异地
更新。

### 7.3 OOD current-state decoder

为排除普通 decoder 通过 (S_prev,E) 纠正推算 S_t，建立 directed-route
OOD decoder：

- A：L->M、M->R、R->L；
- B：M->L、R->M、L->R；
- Train A -> Test B，Train B -> Test A；
- leave-one-route-out 完全隐藏一种 directed transition；
- scaler、PCA、C selection 只在 train trajectories/routes 中拟合。

OOD state shift：

| protocol | t>=2 | t>=3 |
|---|---:|---:|
| A->B | +0.590 | +0.514 |
| B->A | +0.592 | +0.536 |
| leave-one-route-out | +0.291 | +0.215 |

这些 movement 支持某种跨 route 的 state-related representation，但 leave-one-
route-out 的效应明显弱于 A/B split；不能据此宣称完全 abstract、route-free
current-state code。综合结论是 IID_and_OOD_state_shift，但限定为
GO-restricted-representation。

## 8. Circuit localization：layer 与 module

### 8.1 Split 与层级扫描

固定 discovery/validation trajectory split，discovery 只用于选择 causal
region，validation 只用于冻结后验证。粗扫层为
[4,8,12,16,20,24,28,32,36]，再在 discovery 中对连续 region 加密。

最终验证显示 late-layer residual mediation 从约 L24 开始：

| validation subset | L24 sufficiency | L28 | L32 | L36 |
|---|---:|---:|---:|---:|
| overall | +0.382 | +0.921 | +1.094 | +1.201 |
| t>=2 | +0.184 | +0.424 | +0.511 | +0.592 |
| t>=3 | +0.240 | +0.449 | +0.511 | +0.576 |

necessity 为对应负方向。该 region 在 held-out validation 复现，且
matched-history 在晚层接近 0，支持 late native mediation，但不证明 effect
只来自 current event。

### 8.2 Strict event decoder leakage 修复

此前 event endpoint decoder 使用了全部 baseline trajectories，导致 validation
leakage。修复后 decoder 只在 discovery trajectory 的 baseline hidden states
上拟合 scaler/PCA/C，并冻结到 validation。event 与 native state 的 layer
profiles 才可进行严格 ordering 比较。

### 8.3 Attention 与 MLP

validation t>=2 的 L24 attention 同时具备：

- event sufficiency 正向；
- native-state sufficiency 正向；
- event necessity 负向；
- native-state necessity 负向；
- CI 均支持预期方向。

L24 MLP 可以稳定介导 event，但没有稳定的 native-state 双向 effect。因此
当前最合理的 module-level 描述是：L24 attention 参与 event-to-native-state
causal pathway；MLP 更像 event representation 的稳定介导点，但不足以单独
解释 native state readout。

layer ordering 的严格 held-out event endpoint 受限于 endpoint 和 layer
profile 的统计稳定性，不能把“先看到 event”直接写成已经证明的生物学式
顺序。可靠结论是 event mediation 与 native mediation 在 late layer 共存，
而非已精确证明一个唯一 onset layer。

## 9. L24 Head 0 localization

Qwen3-VL L24 attention 审计结果：

- 32 attention/query heads；
- 8 key/value heads；
- GQA group size = 4；
- head dimension = 128；
- intervention 在 head concat、o_proj 之前的单 head slice；
- 没有用 post-o_proj 切片假装单 head。

unit tests 验证 all-head patch 与完整 attention-output patch 一致，self patch
为数值零。

head discovery 只用 discovery trajectories，validation 不重新选择 head：

- Head 0：event-to-native-state candidate，并在 validation 上通过双向方向性
  检验；
- Heads 2/29：event-routing candidate，但 joint event necessity CI 跨 0，
  native-state effect 不稳定，不能称为 validated routing circuit。

Head 0 validation effect：

| subset | event suff. | event nec. | native-state suff. | native-state nec. |
|---|---:|---:|---:|---:|
| overall | +0.019 | -0.022 | +0.181 | -0.229 |
| t>=2 | +0.025 | -0.030 | +0.086 | -0.125 |
| t>=3 | +0.035 | -0.026 | +0.103 | -0.106 |

Head 0 可以解释 L24 attention 的大部分 native-state effect，但只能解释一小
部分 event effect。比率超过 1 时可能来自非线性或 suppressive interaction，
不能解释为可加的 variance explained。结论是：Head 0 是 validated final-
token event-to-native-state readout head，不是完整视觉 event extraction circuit。

## 10. Head 0 rescue、dose-response 与 downstream rescue

### 10.1 Head 0 causal rescue

由于 50 条 trajectory 已被 discovery/validation 使用，rescue 没有真正的全新
trajectory replication；使用 validation target 与 disjoint discovery donor，
必须标为条件性复现。

strong same-state donor 的 Head 0 patch：

| subset | delta GT margin | wrong->correct |
|---|---:|---:|
| overall | +0.404 | +0.050 |
| t>=2 | +0.504 | +0.075 |
| t>=3 | +0.465 | +0.050 |

weak target 的 margin gain 约 +0.745；weak-minus-strong gain 约 +0.626，支持
baseline 较弱时更容易获得连续 rescue，但 slope CI 仍不足以建立普遍规律。
GT-state specificity 本身不够稳定，不能称为强 categorical rescue。

different-event donor 会产生相反方向的 donor-implied movement，说明 Head 0
不是无语义的随机 logit 扰动；它能传递 event semantics，但不保证最终 state
commitment。

### 10.2 Dose-response

alpha 为 0/0.25/0.5/1/1.5/2 时，same-state donor 的 overall GT margin gain
约为 0、+0.097、+0.212、+0.404、+0.612、+0.831；t>=2 为
0、+0.126、+0.257、+0.504、+0.762、+1.030。说明连续 dose-response 清晰。

alpha=2 时 wrong-to-correct 约 +0.079，但严格 negative-margin targets 仅
3/63 crossing。大 alpha 增加 logit displacement 和少量 unintended prediction；
different-event donor 在 alpha>=1.5 时 harmful flip 明显增加。因此连续
margin 可被放大，但 categorical commitment 仍是瓶颈。

### 10.3 Full downstream block joint rescue 的结构性失败

旧 joint experiment 将 L32/L36 完整 final-token block output overwrite 到
target。该操作覆盖了之前的 Head 0 contribution，因此 Head0+block 必然等于
block-only；不能用它判定 positive synergy。这个结果是 intervention definition
的 occlusion，不是 Head 0 无效。

### 10.4 Composable residual-delta rescue

后续改用：

    delta32 = h32_donor - h32_target
    h32' = h32_current + beta * delta32

其中 joint 条件的 h32_current 来自已经完成 Head 0 intervention 的 forward，
且 delta 来自 baseline cache。beta=0 与 Head0-only 相等，joint 与 L32-delta-
only 不再恒等。

strong same-state donor 结果：

| subset, beta=1 | Head 0 | L32 delta | joint |
|---|---:|---:|---:|
| overall | +0.404 | +1.616 | +1.640 |
| t>=2 | +0.504 | +1.940 | +1.977 |
| t>=3 | +0.465 | +1.790 | +1.790 |

joint categorical wrong-to-correct 在 t>=2 约 +0.325，t>=3 约 +0.275，但
correct-to-wrong 也上升；different-event donor 下 joint 产生强烈反向 movement
和大量 harmful flips。整体没有稳定、清晰的正 interaction，不能宣称 Head 0
与 L32 存在可组合 rescue synergy。最合理解释仍是 distributed downstream
commitment bottleneck。

## 11. Offline heterogeneity 与 path dependence

### 11.1 Heterogeneity

现有 native transplant 显示 soft 与 hard endpoint 分离：

- t>=2 native cf-vs-target shift 约 +0.596；
- t>=3 约 +0.623；
- 但 GT margin gain 可以为负；
- categorical crossing 近似为零。

因此不能用 argmax 不翻转否定所有 state-related effect，也不能用 logit
movement 宣称 categorical state update。

### 11.2 Held-out path-dependence design

固定 discovery=30 trajectories、validation=20 trajectories。t>=3 产生 36
eligible validation-target/discovery-donor pairs，覆盖 17 target trajectories；
t=3/4/5 分别 11/12/13 pairs。四条件为：

1. target history + target current window；
2. donor history + target current window；
3. target history + donor current window；
4. donor history + donor current window。

### 11.3 Path-dependence result

history replacement 强烈改变 hidden state decoder：t>=3 L24/L32/L36 约
+0.329/+0.331/+0.371。current-window-only control 接近 0。

但 native predictions 在所有条件完全不变：29 Middle、7 Left，accuracy
12/36=33.3%，没有 wrong-to-correct 或 correct-to-wrong flip。native GT-margin
history contrast：

- donor-history/target-current minus target-real：+0.201，CI
  [-0.174,+0.602]；
- donor-real minus target-history/donor-current：+0.233，CI
  [-0.172,+0.654]。

因此 history 会改变 final representation，但 native readout 相对不变。
event decoder 也随 history substitution 改变，所以不能说该实验证明 history
特异改变了 event perception；更准确的是 final representation 对 history/context
敏感。

### 11.4 History-to-readout mediation

修正 L36 endpoint 后，t>=3 history sufficiency 的结果为：

| layer | state probe | event probe | native GT logit |
|---:|---:|---:|---:|
| L24 | +0.329 [+0.116,+0.520] | +0.133 [+0.024,+0.277] | +0.133 [-0.041,+0.322] |
| L32 | +0.331 [+0.177,+0.485] | +0.162 [+0.064,+0.294] | +0.094 [-0.179,+0.388] |
| L36 | +0.371 [+0.220,+0.508] | +0.174 [+0.072,+0.309] | +0.091 [-0.211,+0.416] |

necessity native effects CI 也跨 0；current-window control 的 state/native
effects接近 0。这个结果支持 representation-to-readout bottleneck：历史
敏感信息可以进入 late hidden representation，却没有稳定地改变 native readout。

## 12. 失败实验与修复记录

### 12.1 Stale / inertia 主线

response-bias null 显示 stale 没有超过偏置 null，因此降级为背景现象，不再
作为主要机制结论。

### 12.2 旧 freeze

重复 current-window 第一帧仍可能留下静态 event cue。改为 no_current_event：
用 window 开始前最后一帧重复填满整个当前 window。

### 12.3 Pair-key join

target_t_from_source 与 target_t 未正确映射会产生错误 native accuracy。
已改为同一 target trajectory/prefix 的显式 join，并重新计算 native cf shift。

### 12.4 Event decoder leakage

早期 event endpoint decoder 使用全部 baseline trajectories。已改为 discovery-only
fit，validation 完全冻结，避免 layer/event ordering leakage。

### 12.5 L36 endpoint

L36 block output 是 pre-final-norm，而已有 hidden cache decoder 对应 post-final-
norm hidden_states[36]。直接混用会产生退化的 0/1 probe。修复为在正确的
post-final-norm endpoint 上 decode，并完成 self-patch sanity。

### 12.6 Full block overwrite 与 composable delta

完整 L32/L36 overwrite 会抹掉 Head 0 effect，故不能研究 synergy。后续改为
baseline-derived residual delta 加到 Head0-intervened current activation，解决
了恒等问题，但仍未获得稳定正协同。

### 12.7 运行工程问题

多卡 shard 运行中曾出现：缺少 validation CSV、环境 import path 缺失、输出
列缺失、tuple 当 dict 解包、以及 shell 脚本在 shard 未完成时提前进入 offline
analysis。后续脚本增加了 completion verification、tqdm、GPU shard、CPU
analysis、manifest audit 和 unit/smoke test。正式实验应先检查所有 shard 的
completion marker 与 expected columns，再运行离线分析。

## 13. 综合机制模型

    视觉 current window
             |
             v
    event representation E_t ----------- strong, causally transplantable
             |
             v
    late L24 attention / Head 0 ---------- native logit influence
             |
             v
    distributed downstream readout ------- weak categorical commitment
             |
             v
    argmax current-state behavior -------- sparse / biased / often unchanged

历史路径也能改变 late representation，但通常不能改变 native categorical
readout：

    history substitution -> hidden representation change -> native readout mostly invariant

因此最符合当前全部结果的解释是：

- event code 强；
- previous-state code 中等；
- current-state code 普通 probe 下较弱；
- event transplant 能使 native logits 连续朝 counterfactual 移动；
- Head 0 是 late event-to-state readout pathway 的一个 validated component；
- categorical commitment 需要更分布式或更强的 downstream state commitment；
- hidden representation 与 native readout 之间存在明显 gap；
- multi-step history 可能加剧该 gap。

## 14. 已建立的结论、限制性结论与未建立结论

### 已建立

1. 在统一 Transformers pipeline 下，视觉 event 比 current state 更容易被
   hidden-state linear decoder 读取。
2. response bias 足以混淆 stale，因此 stale 不是主要机制证据。
3. raw visual current-window transplant 能稳定改变 event representation。
4. event transplant 会使 native state logits 朝 S_cf 移动，且 t>=2/t>=3
   仍存在。
5. late-layer residual mediation 从约 L24 开始，L24 attention 有 held-out
   双向 event/native-state effect。
6. L24 Head 0 是 validated final-token event-to-native-state causal component。
7. Head 0 存在连续 dose-response，different-event donor 可造成语义相反的
   movement。
8. 历史改变 late hidden representation，但不可靠改变 native prediction。

### 限制性结论

1. native counterfactual movement 是 continuous soft influence，不等于稳定
   categorical state update。
2. OOD decoder movement 支持跨 route 的 state-related information，但
   leave-one-route-out 效应较弱，不能宣称完全抽象的 current-state code。
3. Head 0 更像 event-to-state readout component，不是完整 event-routing circuit。
4. Head 0 + downstream residual delta 没有显示稳定正协同；结果支持或提示
   distributed commitment bottleneck，但不是唯一可能解释。
5. rescue 使用已有 50 trajectories，未提供完全独立的新 trajectory replication。

### 尚未建立

- 没有证明模型内部存在干净、独立、可泛化的 S_t latent variable。
- 没有证明 linear decodability 等于 generation-time functional use。
- 没有证明 history 改变 native categorical state。
- 没有证明单个 Head 0 是唯一或充分的视觉 event extraction head。
- 没有证明 attention 与 MLP effect 可以线性相加。
- 没有证明 t=1 的强 effect 代表真正的多步 state maintenance。

## 15. 最终判断与后续建议

### 当前主结论

最稳妥的论文级表述是：

> Qwen3-VL 在 cup/shell task 中保留并可因果传递视觉交换事件信息；该
> information 在 late L24 attention、尤其 Head 0 的 final-token readout
> pathway 上能够连续地影响 native state logits。然而这种影响在多步条件
> 下很少转化为特异的 categorical state commitment。历史可改变内部表示，
> 但 native readout 对其相对不敏感。整体结果支持一个 distributed
> commitment bottleneck / representation-readout gap，而不是一个已经确认
> 的 clean abstract-state computation。

### GO / NO-GO

- 对“视觉 event 能否影响 native state belief”：GO，限制为 continuous native
  logit influence。
- 对“是否已证明特异 categorical counterfactual state update”：Restricted GO，
  t>=2/t>=3 specificity 和 flip evidence 不够强。
- 对“是否存在 Head 0 单头即可 rescue commitment”：NO-GO。
- 对“是否进入 exhaustive head sweep”：当前不建议。已有 Head 0 localization
  已足以支持 narrowly scoped component claim；继续扫描应只有在预先定义的
  新问题能区分 distributed commitment、downstream readout 或 causal rescue
  时才值得。
- 对“是否停止 single-channel rescue line”：建议停止把更大 alpha 或更多
  downstream overwrite 作为主要路线；它主要增加 soft margin，不稳定解决
  categorical commitment。

### 更有价值的后续方向

若继续项目，优先做能区分机制模型的实验，而不是重复普通 probing：

1. 针对 t>=2/t>=3 的 success/weak-update heterogeneity，预注册 native
   margin、event efficacy、history sensitivity 的联合模型。
2. 做 late-layer native readout 的 distributed intervention，例如冻结多个
   validated components 后测试是否能恢复 categorical boundary，但必须先
   预注册组合规则，避免 post hoc rescue search。
3. 使用独立任务或第二个模型检验 event representation 强、state commitment
   弱是否可迁移。
4. 如没有新的可区分预测，应停止 rescue 线，把工作转向完整报告与可复现性
   发布。

## 16. Artifact 索引

### 输入、行为与 probing

- scripts/state_rev_input_pipeline.py
- outputs/vetbench/validity_gate_v1/validity_gate_report.md
- outputs/vetbench/composition_analysis_v1/composition_report.md
- outputs/vetbench/probe_analysis_v2/final_probe_report.md
- outputs/vetbench/state_retention_analysis_v1/retention_report.md

### Mechanism gate 与 native counterfactual

- outputs/vetbench/mechanism_gate_v1/mechanism_gate_report.md
- outputs/vetbench/mechanism_gate_final/REPORT_FOR_GPT56_SOL.md
- outputs/vetbench/mechanism_gate_final_v2/mechanism_final_verdict.md
- outputs/vetbench/mechanism_native_specificity_v1/native_specificity_report.md

### Circuit 与 head localization

- outputs/vetbench/circuit_localization_v1/circuit_localization_report.md
- outputs/vetbench/circuit_localization_v2/circuit_localization_report.md
- outputs/vetbench/head_localization_l24_v1/head_localization_final_report.md
- outputs/vetbench/head_localization_l24_v1/joint_head_report.md

### Rescue 系列

- outputs/vetbench/head0_causal_rescue_v1/head0_causal_rescue_report.md
- outputs/vetbench/head0_dose_response_v1/dose_response_report.md
- outputs/vetbench/head0_downstream_joint_rescue_v1/joint_rescue_report.md
- outputs/vetbench/head0_downstream_delta_rescue_v1/delta_rescue_report.md

### Heterogeneity、path 与 history readout

- outputs/vetbench/offline_heterogeneity_v1/offline_heterogeneity_report.md
- outputs/vetbench/path_dependence_v1/path_dependence_report.md
- outputs/vetbench/history_readout_mediation_v1/history_readout_mediation_report.md
- scripts/history_readout_mediation.py
- scripts/analyze_history_readout_mediation.py
- scripts/run_history_readout_mediation_8gpu.sh

## 一句话结论

StateRev-VL 没有发现一个简单的“模型看到了 event 却完全没有任何 state
影响”的 null；它发现的是更具体也更有价值的现象：视觉 event 能进入并
因果影响 late native state readout，但影响主要停留在连续 logits 层面，
多步 categorical commitment 被一个分布式 downstream representation/readout
瓶颈显著削弱。
