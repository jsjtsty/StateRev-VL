# StateRev-VL 研究现状总览

更新：2026-09-23。本文汇总截至目前的全部研究结论，取代散落在各 `*_FOR_SOL.md` 中的阶段性判断；与旧文档冲突时以本文为准（见 §6 "已推翻的结论"）。

## 0. 一句话结论

视频 VLM 能在 hidden state 中近乎完美地编码视觉事件，却不能把事件与历史状态组合成当前状态；原生回答接近随机，而在同一批 hidden 上做外部组合即可达到 89–100%。因果上，事件影响经 L24 attention（尤其 Head 0）到达状态读出，但只产生连续 logit 移动，很少形成类别翻转。通用的端到端学习型滤波器（PSF、USTF）至今没有超过简单的"事件解码 + 转移"方案。

## 1. 数据、模型与协议

| 任务 | 数据 | 状态空间 | 划分 |
|---|---|---|---|
| Shell（VET-Bench cup） | 50 条轨迹 × 5 次交换 = 250 个 prefix | Left / Middle / Right | content-disjoint：discovery 30 / validation 20 |
| Chess（MET-Bench，单棋子跟踪） | 300 局，2,852 个 prefix，t≤10 | 64 格 + captured（实际只出现 g1/f3/d4 等 5–6 个） | discovery 200 局 / validation 100 局 |

- 模型：Qwen3-VL-8B-Instruct（主）、LLaVA-NeXT-Video-7B（复现）。
- hidden：state-question forward 的最后一个输入 token。Qwen Shell 有 37 层，LLaVA Shell 有 33 层，LLaVA Chess 有多层；**Qwen Chess 只存了最后一层**。
- 统计：以轨迹为 cluster 做 bootstrap；主结论看 t≥2、t≥3，因为 t=1 时初始状态直接写在 prompt 里。
- 行为以 Transformers 前向为准。vLLM 与之不等价：state 答案一致率只有 73.6%，event 答案一致率 56.4%。

## 2. 机制审计（Shell / Qwen）

### 2.1 行为与有效性
- 原生准确率：state 32.4%，event 69.2%，严重偏向 Middle（L/M/R = 0.21/0.74/0.05）。
- "stale / 信念惯性"不是独立机制：真实 stale 率约 0.36，没有超过 response-bias null（0.44–0.72）。这条已降级为背景现象。
- 在 event 答对的样本中，约 2/3 的 state 是错的，说明事件→状态的缺口真实存在。
- 给文字提示（上一状态 + 事件）可以救回；只给上一状态几乎无效。说明模型能按文字规则计算交换，但不能从视觉自主组合。
- 帧率从 2 fps 提到 8 fps 无显著改善。每个 t 的视频在交换窗口结束处截断，没有交换完成后的静止帧。

### 2.2 表示
| 线性探针 | event | 上一状态 | 当前状态 |
|---|---|---|---|
| 平衡准确率 | 1.00 | 约 0.60 | 约 0.55 |

用外部符号组合"解码出的上一状态 + 事件"约得 0.58；而在 canonical stale 子集上原生回答是 0%。

### 2.3 因果证据
- **原始视觉移植**（只替换当前交换窗口）：事件干预有效率约 0.95。原生 counterfactual logit 在 t≥2 移动 +0.60 [0.44, 0.74]，t≥3 移动 +0.62。
- **但类别层面几乎不翻转**：三状态特异性在 t≥2 为 +0.13，CI 跨 0；在 t≥3 为 +0.16，CI 勉强为正。结论是 Restricted GO。
- **OOD 状态解码**（跨有方向 route 训练/测试）：A→B 与 B→A 约 +0.59；留一 route 只有约 +0.25–0.29。存在跨 route 的状态相关信息，但不能说是完全抽象的状态变量。
- **层定位**：从约 L24 开始的后层残差介导（validation t≥2：L24 +0.18，L36 +0.59）。L24 attention 有双向 event / 原生状态效应；MLP 只稳定介导 event。
- **L24 Head 0**：在 held-out 上验证为 event→原生状态的读出头（t≥2 充分性 +0.086，必要性 −0.125），不是完整的事件提取电路。Head 2/29 未通过验证。
- **救援实验**：
  - Head 0 注入：GT margin +0.40–0.50，但 wrong→correct 只有 5–7.5%；dose-response 清晰，而 alpha=2 时 3/63 越过分类边界。
  - 与 L32 残差增量组合：没有稳定正协同，还伴随有害翻转。
  - 结论：单头救援 NO-GO；更大 alpha 这条路线不再继续。
- **历史路径**：替换历史会显著改变后层 hidden（t≥3 状态探针约 +0.33–0.37），但原生预测完全不变（36 个 pair 零翻转）。存在表示→读出的缺口。

### 2.4 综合模型
视觉事件 →（强、可移植）事件表示 → L24 attention / Head 0 →（连续的 logit 影响）→ 分布式下游读出 →（很弱的类别提交）→ 原生答案。更符合 **distributed commitment bottleneck / representation–readout gap**，而不是存在一个干净的 S_t 变量。

## 3. 外部状态追踪（冻结 VLM）

### 3.1 Shell（content-disjoint validation，100 个 prefix）
| 方法 | Qwen | LLaVA |
|---|---|---|
| 原生 state | **0.37**（权威口径；另一份缓存为 0.31） | 0.28 |
| hidden 事件探针 → 规则递归 | **0.89**（事件准确率 0.95，用 L24） | **1.00**（用 L1） |
| oracle 事件 → 规则 | 1.00 | 1.00 |

- 27 参数的可学习 updater 与手写规则结果完全相同，并能跨 Qwen/LLaVA 迁移；few-shot 约 10 条轨迹即饱和。
- 温度校准（T=0.25）不改变 argmax，对错误传播没有影响。
- **LLaVA 的 L1 层 100% 未经审计**：可能利用了帧数或时间伪特征。任何依赖 LLaVA Shell 的结论都要先完成审计。

### 3.2 Chess（validation 100 局，948 个 prefix）
| 方法 | Qwen | LLaVA |
|---|---|---|
| 原生 state | 0.59 | 0.43（输出被截断，解析器敏感，不是干净基线） |
| 64 类 src/dst 走子探针 | dst 0.29，联合走子 0.003 | — |
| 直接 65 类状态探针 | 0.90 | 0.98 |
| compact 事件（unaffected / moved / captured）递归 | **0.957** | 0.977 |
| 加权融合（tracker + direct） | 0.962 | 0.98 |

compact 事件的做法有效，完整的走子解码做不到。

## 4. 通用学习型滤波器

### 4.1 PSF v1–v5（GRU / KEEP-SET / state-delta，共享 core，无事件标签）
- 形式预检：Shell 0.36，Chess 0.48。
- stabilization v2：观测门控只在 Shell 上通过。
- KEEP/SET v5 bounded：Shell 0.41–0.42，Chess/Qwen 0.79，joint Chess 0.86。
- state-delta v4：tiny Shell 只有 0.48。
- change-path 审计（teacher forcing）：Shell 在 delta-only 下最好也只有 0.48。
- teacher-forcing 审计里 detector 和 SET 都是 100%，说明失败出在 free rollout 与优化上。
- **所有版本在 Shell 上都没有过门槛。**

### 4.2 USTF（本阶段）
**结构**：
- 每个 backbone 一个 4096→128 的 projector，加一个共享 core。
- core 包括：change encoder、pairwise 转移打分器、观测打分器。
- 融合方式为 Bayes 乘积，全程自由递归。

**v1 的失败原因**：候选文本向量两两余弦约 0.996，Shell 与 Chess 共用 projector 后，Chess 塌缩为永远预测 f3。之前"Chess 信号弱"的诊断是错的。

**修复阶梯**：
- P0a：候选集合标准化。
- P1：转移拆成 stay / move，并加 ⟨A·v, c_j−c_i⟩ 匹配。
- P1b/P1c：观测加双线性匹配，融合加可学习可信度 β_t。
- 训练日程：5 轮 warmup + 前 40 轮不启用早停。
- 默认结构 P1c：Qwen 单 backbone 820K 参数，Qwen+LLaVA 1.34M，占 7B 的 0.019%。

**tiny gate**：
- v1 通过 1/3 个种子；P0a 及之后的变体都是 3/3。
- 正式 gate（P1c，默认种子）通过：Chess 与 Shell 的 transition / rollout 均为 1.00；候选顺序打乱测试 100% 通过。

**held-out**（discovery 内按加载顺序切分：Shell 20/10/20，Chess 100/50/50；5 个种子，新训练日程）：

| 测试集 | USTF P1c | 线性状态探针 | v1 |
|---|---|---|---|
| Qwen Chess | 0.963 ± 0.012 | 0.86 | 0.58 |
| LLaVA Chess | 0.974 ± 0.008 | 0.95 | — |
| Qwen Shell | 0.70 ± 0.09 | 0.51 | 0.36 |
| LLaVA Shell | 0.42 | 0.46 | — |

- 跨 backbone：Qwen+LLaVA 联合训练对各自结果影响在 0–3 个点内；Qwen 训练的 core 冻结后只训 LLaVA projector，LLaVA Chess 约 0.90。core 在形式上与 backbone 无关。
- 起作用的是转移先验：只用先验 ≈ 完整滤波。只用观测时 Chess 约 0.8，Shell 接近随机。
- 无效的改动：特征 dropout、随机截取子轨迹、hidden LayerNorm（P1d：Shell 0.64 / 0.43）。

**关键对照（审视阶段补做，USTF 使用的 hidden 层和数据划分）**：

| | Qwen Shell | LLaVA Shell |
|---|---|---|
| 单层线性事件探针 + 交换规则 | **0.94**（L9） | **0.94**（L8） |
| USTF P1c | 0.64–0.70 | 0.42 |

- 三杯设定下，事件可由（S_{t−1}, S_t）唯一推出：球移动了，交换的就是这两个位置；球没动，交换的就是另外两个位置。所以探针并没有使用额外监督。
- 结论：USTF 在同等信息下落后 25–50 个点；Shell **不受输入特征限制**。
- 94% 里的层是看过测试结果挑的，数字偏乐观；但严格口径下的 0.89 足以支撑同一结论。

**USTF 落后的原因**：
1. 没有共享的事件瓶颈。球没动的步本身能说明是哪两个杯子交换，但在 USTF 里只监督了"停留"，没有把这个信息传给事件；探针方案则让每一步都监督同一个事件变量。
2. "视觉变化对齐文本差"不适用于对称交换：L↔M 同时需要 c_M−c_L 和 c_L−c_M 两个方向，双线性匹配在结构上做不到。
3. 候选文本（left / middle / right）本身不含转移信息，Shell 上的"语义"部分帮不上忙。

**评估局限**：
- held-out 测试集被用于选择 P1c 和训练日程，数字偏乐观。
- 划分与原有的 canonical split 不同，不能直接与 §3 的数字比较。
- Chess 实际上是 g1→f3→d4 的近 3 状态链；训练中没见过的格子（e5）和 captured 上，USTF 与探针都是 0%。"语义迁移"没有证据。
- Qwen Chess 只有一层 hidden，层混合在这里不起作用。

## 5. 结论分级

**已建立**
1. 事件可以被强线性解码，当前状态弱，原生回答接近随机；外部组合能恢复到 0.89–1.00。这一现象在 Qwen 与 LLaVA、Shell 与 Chess 上都成立。
2. 视觉事件移植会使原生 logit 朝 counterfactual 方向移动（t≥2、t≥3 仍成立）。
3. 后层介导从约 L24 开始；L24 Head 0 是经 held-out 验证的 event→状态读出组件，并有连续的 dose-response。
4. 替换历史会改变后层表示，但不改变原生预测。
5. stale 不是独立机制；vLLM 与 Transformers 的行为不可混用。

**限制性结论**
- 原生状态的更新是连续 logit 移动，不是类别更新；特异性证据较弱。
- 跨 route 的状态信息存在，但留一 route 的效应弱。
- Head 0 与下游组合没有稳定的救援协同，"分布式提交瓶颈"是最合理的解释，但不是唯一可能。
- 学习型 updater 与手写规则等价，没有精度增益。
- USTF 可以跨任务、跨 backbone 共享 core，在 Chess 上追平任务专用方法，在 Shell 上明显落后。

**尚未建立**
- 模型内部存在可泛化的抽象 S_t 变量。
- 线性可解码等于生成时被功能性使用。
- 任何学习型通用滤波器优于"事件解码 + 转移"。
- 对没见过的状态或转移的语义泛化。
- LLaVA Shell 的 L1 100% 是否为伪特征。
- 在 50 条轨迹之外的独立复现（目前所有结论都在同一批 Shell 数据上）。

## 6. 已推翻或修正的结论

| 旧结论 | 修正 |
|---|---|
| stale / 信念惯性是主机制 | 没有超过 response-bias null，已降级 |
| 探针 0.549 与 0.658 相互矛盾 | 聚合顺序不同（跨层 max 的均值 vs 单个 split 的 maxT），不是 hidden 不同 |
| Head 0 + 整块覆盖没有协同 | 整块覆盖会抹掉 Head 0 的贡献，属于实验定义问题；改用可组合增量后，仍然没有稳定协同 |
| USTF v1 在 Chess 上失败是因为 Chess 信号弱 | 实为候选文本塌缩导致；修复后约 0.96 |
| "Chess 49% 远高于随机，说明学到了东西" | 实为全部预测同一格 |
| Shell 受限于 prefix hidden 特征，需要重跑 VLM | 错误。同一批 hidden 用事件探针 + 规则可达 0.94；问题在 USTF 的结构 |
| Qwen 原生 Shell 准确率 0.31 | 权威口径为 0.37（两份缓存的解码路径不同） |

## 7. 评估与计划

- **外部定位**：VET-Bench 原论文（arXiv 2603.08436）已经指出 SOTA VLM 在该任务上接近随机，并用 Molmo2-SGCoT 提升到 90% 以上。"hidden 知道、输出用不上"这一类发现本身不新。我们的差异化在于：机制定位（事件被感知但没有被组合，瓶颈在后层提交），以及区分感知与组合的诊断框架。
- **判断**：以现有内容投 CCF-A 主会不乐观（主观估计中稿概率 <15%）。主要短板：规模小（50 条轨迹）、机制只在一个模型上完成、方法没有超过简单基线、存在直接竞争工作。
- **建议主线**：做成一篇分析论文，核心论点是"视频 VLM 感知事件却不组合状态"。按新主线估计，当前进度约 30%。
  1. 用 shellgame 生成器扩数据到几百到几千条；同时使用 card 游戏、MET-Bench Chess，并评估 Theory-of-Space 能否作为第三个任务。
  2. 在 4–6 个模型和不同规模上画"事件可解码度 / 原生状态 / 外部组合"随 t 变化的曲线，并用 Molmo2-SGCoT 做机制对照：它修好的是感知还是组合？
  3. 在扩展数据上、至少 2 个模型中复现后层定位。
  4. 提出一个能改善原生回答的修复（后层状态注入或小 LoRA），并与 SGCoT 对比。
  5. USTF 降为组件或基线。如果要保留方法线，改为"潜在操作瓶颈"（USTF-LO：T_t = Σ_m q_t(m)·T_m），并必须通过没见过状态的测试。
- **第一道门槛（2–3 周）**：扩展数据后，缺口是否在 ≥4 个模型上稳定复现。是：按 CCF-A 推进，目标约为 ICML / ACL / ICCV 2027 周期（截稿时间以官网为准）。否：转投 CCF-B 或 workshop。
- **评估纪律**：此后一律使用 canonical split（Shell content-disjoint、Chess validation）；测试集只在最终确认时使用一次。

## 8. 产物索引

- 机制审计：`STATE_REV_VL_COMPLETE_RESEARCH_REPORT.md`（完整数字与路径），`outputs/vetbench/{validity_gate_v1,mechanism_gate_final*,circuit_localization_v2,head_localization_l24_v1,head0_*,path_dependence_v1,history_readout_mediation_v1}/`
- 外部追踪：`outputs/vetbench/{hidden_event_recursive_content_disjoint_v1,learned_state_updater_v1*,llava_next_video_7b_replication_v1,native_state_alignment_audit_v1}/`，`outputs/metbench_chess/{single_piece_tracking_pilot_qwen_v1,qwen_compact_event_validation_v1,qwen_hybrid_state_tracker_v1,llava_compact_event_replication_v1,llava_native_state_audit_v1}/`
- PSF：`staterev/psf.py`，`scripts/train_psf.py`，`outputs/psf_v1/*/REPORT.md`
- USTF：`staterev/ustf.py`，`scripts/ustf.py`（`tiny-gate` / `heldout` / `param-report` / `joint-smoke` / `task-c-demo`；变体 v1、p0a–p1d），`tests/test_ustf.py`（18 个测试），`outputs/ustf_v1/{tiny_gate,heldout,logs}/`
