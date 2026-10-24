# OVMOT 方向 A/B/C 精确新颖性审计

日期：2026-10-24。状态：`HOLD_AB_COUPLED_STATE_MACHINE; STOP_C_STANDALONE_ATTENTION`。
范围：只回答 association ambiguity/deferred state、detector uncertainty/belief update、track-set identity state 三条路线中最值得走什么。先复用本地 `set_association_literature_audit_20261024.md`、`novelty_protocol_association_state_handoff_20261024.md`、`trajectory_set_gate_full_protocol_20261024.md` 和 `covtrackpp_code_boundary_20261023.md`，再补查直接近邻。本审计不把摘要当全文；正文已下载/抽取的 11 篇列在文末。检索结论可随时被更强一手工作推翻，不能写“首次”。

## 1. 判定

- **单独 B：STOP。** box/detector uncertainty 进 Kalman/association 已被直接占满，不能再把“完整 belief”或“lost/recovery”作为独立新意。
- **单独 C：STOP。** 历史观测集合/attention、跨帧 track-query、轨迹集合图连接均有直接先例；本项目的全量 FIT 信息门也未支持普通集合编码（见 §4）。
- **单独 A：HOLD，不能按原版 PendingState 写法。** **A×B 耦合的因果状态机是唯一仍值得先验证的窄候选**：歧义产生的是“占用多少证据量、每支状态如何演化、何时不可逆提交”的**联合 belief**，而不是换一个 pair score 或换一个 swap 规则。必须先通过自然 event replay 信息门，才允许训练网络。

## 2. 精确先例与可写差异

表中 `机制` 指已核验的实际方法，不是标题推测。`建议` 是相对本项目的判定。

| 机制 | 代表论文与可核验位置 | 公式/实验 | 与本项目可写的差异 | 建议 |
|---|---|---|---|---|
| Detector 预测分布进入 IoU 歧义与关联 | **UTrack**, ECCV 2024 UnCV Workshop，[arXiv:2408.17098](https://arxiv.org/abs/2408.17098)，Sec.3.1–3.2；[官方代码](https://github.com/DLR-MI/UTrack) | NMS 采样得到每个框的 `δa`；Sec.3.2 将区间重叠的 IoU 视为不可判，并用 phase/box-size disambiguator 排序。完整系统还比较 phase 与 size 两个控件。 | 我们不能再把“用预测分布判断 IoU 是否歧义”作为核心新点。**A×B 的差异必须是歧义随后改变分支的证据量和状态寿命，而不是只选一个 tie-breaker。** | STOP 宽泛的 uncertainty-aware association |
| box covariance 进入 Kalman 更新 | **UncertaintyTrack**, ICRA 2024，[arXiv:2402.12303](https://arxiv.org/abs/2402.12303)，Sec.III-A/B、Fig.2；[代码](https://github.com/TRAILab/UncertaintyTrack) | Prob-YOLOX 给 `Σ_box`；转成 `[cx,cy,a,h]` 后替换 KF 的 measurement covariance `R`。另测试 Confidence Ellipse Filtering、Box Relaxation、Entropy Greedy Matching；BDD100K 和 MOT17 与 ByteTrack/OC-SORT 比较，ID switch 约降 19%。 | “detector box covariance 做完整 belief update”是已有实现。我们只能写：**歧义分支占用独立 belief，并让观测不确定性控制分支证据强度**；仍须超过 constant-R/score-only。 | STOP 独立 novelty；保留为强对照 |
| 置信度耦合的 Kalman 在 OV-MOT | **OVSORTv2**, IEEE SPL 33:2355/2655–2659（页码源有冲突，以 DOI 为准），[DOI:10.1109/LSP.2026.3705385](https://doi.org/10.1109/LSP.2026.3705385)；官方只取得摘要，未读正文 | 摘要明确为 CWK（confidence-weighted Kalman update）与 CNAK（confidence-noise adaptive Kalman）；并加入 wavelet conv。 | 即使换成 OV-MOT，**低置信检测放大 `R`、降低更新增益**已有直接先例。不能写“首次把 OV 检测不确定性送入滤波器”。 | STOP；若候选退化为 `R=σ(score)` 立即停 |
| 预测协方差决定匹配搜索宽度；deferred identity commitment | **ARC-SORT**, arXiv 2026，[2610.22306](https://arxiv.org/abs/2610.22306)，Sec.3.4–3.6、4.6；**仅预印本** | Eq.2 `e=clip(α·max(σx/w,σy/h),e_min,e_max)` 扩张 IoU；Eq.4 计算 assignment swap margin；低 margin 时快照外观，Eq.5 累加 swap log-likelihood，窗口 `W=30`，最多一次 swap。正文明确 **“labels only, filter states untouched”**。 | 最危险的 A 近邻。差异只能落在：**分支不是单一标签假设，而是各自维护 appearance/box/birth/lifetime 的因果状态**；提交必须一次性确定 canonical memory，且必须证明比 no-write、single pending、两分支 swap 更强。 | **HOLD A×B；A 原版 STOP_AS_ARC_SORT_OVERLAP** |
| 对重叠配对在线纠错并阻止错误外观写入 | **FC-Track**, arXiv 2026，[2603.12758](https://arxiv.org/abs/2603.12758)，Sec.III-A/B，Algorithm 1；仅预印本 | 用 IoA 找出重叠 track 对；高重叠时冻结 appearance 更新；用 prime/aux track 外观比较，满足阈值就重新指派。MOT17/20 test 主表另报 81.73/82.81/66.95（MOTA/IDF1/HOTA）。 | “防错误外观传播/在线局部纠正”已被覆盖。我们若保留 quarantine，必须主动比较 FC-Track 式“冻结+局部分配”、同预算 medoid/last-view 和 no-write。 | **A 的 ordinary quarantine 降为控制；不能当独立贡献** |
| 生命周期记忆污染、拒绝不可靠更新、集合记忆 | **LiAM-SAM**, arXiv 2026，[2609.28078](https://arxiv.org/abs/2609.28078)，Sec.3.3–3.4、Fig.1 | 自适应 context memory 只接收“reliable, diverse”参考；相邻参考有 IoU 门；强交互时用运动/几何分辨重复与串识，并把不可靠轨迹从短/长记忆同时排除。DanceTrack 上以 SAM2 系统做大幅整体消融。 | 不能把“memory pollution + gated write + attention memory”当新问题。若 OVMOT 候选仍用普通 reliability gate，`STOP_AS_MEMORY_GATE`。 | STOP 普通 memory-integrity 叙事；保留作概念对照 |
| 多观测历史集合与集合关联 | **MeMOT** [2203.16761](https://arxiv.org/abs/2203.16761)；**GTR** [2203.13250](https://arxiv.org/abs/2203.13250)；**MOTIP** [2403.16848](https://arxiv.org/abs/2403.16848)；**NOOUGAT** [2509.02111](https://arxiv.org/abs/2509.02111) | MeMOT 对短/长期历史做 attention 聚合；GTR 对整片段检测集合联合分配；MOTIP 以历史 track token 对当前检测预测 ID；NOOUGAT 在线上/离线统一历史轨迹集合到新片段集合的连接。 | “track set/attention identity state”已有问题定义与强实现。可写的窄差异只能是**观测可靠性/竞争身份的显式概率状态**，且必须超过这些集合读出与等预算均值/EMA。 | STOP 泛化 C；只可作为分支状态里的 evidence token |
| 统一 learned multi-cue association | **CAMELTrack**, arXiv 2025，[2505.01257](https://arxiv.org/abs/2505.01257)，Sec.3.2、Table 1；[代码](https://github.com/TrackingLaboratory/CAMELTrack) | Temporal Encoder 聚合各 tracklet 的 cue token，GAFFE 在 tracklet+detection 全集上做 context-aware self-attention，再输出可匹配 embedding；显式宣称学习替代 tracklet representation/fusion/multi-stage matching。 | 再加一个 pair attention、cross-attention 或“时空融合头”不足以形成差异。论文中的差异应落在**状态机与不确定证据的分配**，不是 backbone/attention 容量。 | STOP 仅升级 attention 网络 |
| Bounded multi-trajectory selection under delayed disambiguation | **EviTrack**, arXiv 2026，[2605.19283](https://arxiv.org/abs/2605.19283)，Sec.1、3–4；**作者署名为 independent researcher，仅合成实验** | 维护有界轨迹前缀，以累计预测证据和 likelihood ratio 选择；合成 delayed-disambiguation 基准上与 SIS/BPF 比较。 | 它不是视觉 MOT、没有 memory/assignment 后果；但足以压掉“有界候选轨迹 + 延迟承诺”的宽泛新颖性。我们的 MOT 特异性必须体现在 canonical state 与共享检测/身份分配。 | 可借鉴，不能写成首创；不把合成证据外推到 MOT |
| 完整 MHT/JPDA 式延迟决策 | 经典 MHT，**Reid 1979** [DOI:10.1109/TAC.1979.1102124](https://doi.org/10.1109/TAC.1979.1102124)；JPDA 原文见 **Fortmann et al. 1983** [DOI:10.1109/JOE.1983.1145560](https://doi.org/10.1109/JOE.1983.1145560) | MHT 保留跨帧 association hypotheses，按后验/m-best 剪枝；JPDA 对可行联合关联做边缘化。 | 我们不主张“保留多个关联假设”本身新。**成本/范围限定**（只对少数高歧义事件开分支、分支携带 OV 身份状态）是工程约束，不是理论首创；要写成可测的 bounded approximation。 | 必引；不得用 MHT 名字包装普通 pending ID |
| 在线视觉 MOT 的延迟/近在线修订 | **MHT revisited**，ICCV 2015，[CVF 全文](https://openaccess.thecvf.com/content_iccv_2015/papers/Kim_Multiple_Hypothesis_Tracking_ICCV_2015_paper.pdf)；**Near-Online Multi-Target Tracking With Aggregated Local Flow Descriptor**, ICCV 2015，[DOI:10.1109/ICCV.2015.347](https://doi.org/10.1109/ICCV.2015.347) | 前者显式扩展 MHT 的 appearance 与假设管理；后者使用短时间窗/未来信息做近在线轨迹关联。 | “future causal commitment”只能在**当前时刻之前可用、固定小延迟**下实现；不能使用完整未来，也不应宣称在线严格无延迟。 | 必须区分 online-causal 与 near-online；禁写 future-aware online |

## 3. 唯一建议探索的 A×B 耦合机制

### 3.1 可写对象

对每个高歧义 detection，保留 **至多 K 个 assignment hypotheses**；每条 hypothesis 不只是标签，而是拥有自己的：

`m_k = (appearance evidence e_k, geometric belief μ_k/P_k, evidence mass w_k, birth/lifetime state)`。

观测框协方差 `Σ_obs` 决定该观测能贡献多少有效证据，**不是给 pair score 乘一个 confidence**。若观测是低质量框，它既不能强力更新 `P_k`，也不能以独立帧的身份全额累加 `w_k`。当分支后验或证据质量不能满足预声明条件时，分支保持 provisional；满足时才向 canonical state 提交一次，并清除其余分支。

建议形式（用于设计，不声称通用推导）：

```text
for each branch k:
  μ_k^- = F μ_k
  P_k^- = F P_k F^T + Q
  S_k = H P_k^- H^T + Σ_obs(z_t)
  r_k = z_t - H μ_k^-
  loglik_k = -1/2 [r_k^T S_k^-1 r_k + log det S_k]

  # 关键：有效证据质量由新旧不确定性共同决定
  q_k = information_gain(P_k^-, S_k) · agreement_k(e_k, A_obs)
  w_k <- normalize(w_k exp(loglik_k));  delta_k <- clip(q_k, 0, q_max)
```

这里的 `q_k` 必须和**同预算控制**比较，例如 constant mass、score mass、仅 covariance、仅 appearance agreement。不能把 `q_k` 解释成校准概率；需要 NLL/coverage/ECE 或至少 rank calibration 证据。

### 3.2 可发表的精确差异（待实证）

相对 ARC-SORT：ARC 快照外观并以 swap likelihood 事后改标签，且**明确不改 filter state**。本候选让每条矛盾分支沿时间独立推进 μ/P/记忆与生命周期，在提交前 canonical memory 不吸收争议观测。相对 FC-Track：本候选不仅是重叠时冻结更新与局部重指派，还保留**多帧、分支级状态和固定预算**。相对 LiAM-SAM：对象不是 SAM2 mask 记忆完整度，而是 detector-level OVMOT 的 assignment belief 和 canonical evidence write。相对 MHT：只对少数事件做固定 `K`、固定窗口的稀疏信念分支，并明确报告预测成本与遗漏。

这仍只是**机制可分辨性**，不是新颖性保证。可写成论文的前提是：在自然事件上存在足够污染后效，并且本机制比全部普通控制稳定 repair>break、改善后续 continuity；若这些没有出现，放弃“新状态机”的说法。

### 3.3 最小信息门（先 CPU replay，不训练）

使用现成 FIT cache 与冻结 COVTrack 输出，固定 source folds；GT 只在 replay 后评分：

1. **Parity**：无歧义事件时，原 tracker 的 ID、box、memory、lifecycle 逐字段一致；任何差异先当实现错误。
2. **自然事件**：统计 top1/top2 margin 冲突、appearance/geometry 分歧、之后 1/4/8 步的 GT identity repair/break、canonical memory 污染持续长度。只看 GT-free 触发，不能看 GT 选择事件。
3. **四臂**：`released`；`no-write on conflict`；`single pending`；`K-branch A×B`。每个分支与普通控制使用相同检测、候选、寿命和 compute budget。
4. **注入检查**：固定位置错链、交换、重复、空帧、缺失 geometry；确认分支确实改变下一帧 assignment/memory，而不是只改最终 ID。只改输出 → `STOP_AS_RERANKER`。
5. **统计单位**：source→video→GT trajectory；不得把数百万 pair 当独立样本。报告 source macro、lag1/lag4、leave-one-source、repair/break 和 FP/FN/延迟。

停止条件：事件后效小到无法支撑；单纯 `no-write` 或 `single pending` 已取得全部收益；仅一个 source/lag 有效；A×B 的收益完全由 threshold、低分框过滤、普通 appearance bank 或额外 compute 解释；分支不改变后续状态。

### 3.4 若信息门通过，网络应怎样做

只训练一个**轻量 evidence scorer**，不重训 backbone：输入相对 appearance、box innovation、`Σ_obs`、原帧 gap、cycle/assignment ambiguity 及 history validity；输出每个观测对各分支的 log-likelihood 与有效证据质量。训练监督来自 fit 中可确认的 same/different identity anchors 和分支后效，未知不作负例；按 source 留出。网络容量、训练预算和输入特征必须与 MLP/linear/logistic controls 完全相同。它与 OV-MOT 分类 posterior 分开：类别一致性不能决定身份 membership。

## 4. 为什么 C 不再进入训练

本项目全量 FIT 集合门已完成：403 视频/341 source，四臂评价中 `fixed_attention` 相对 `raw_ema` 在两个 lag 均为负（宏平均约 −3.56/−4.65 pp）；`raw_mean`、`history_max` 也未稳定超过 EMA；`last` 只在 lag1 为 +0.89 pp，lag4 为 −0.18 pp。该结果属于 FIT 探索性信息门，不是独立泛化或 TETA，但足以拒绝**当前这个 ordinary set-attention 设计**进入训练。未来若有新机制，应该是分支状态/证据质量，不是把 `attention` 再加一层。证据入口：`trajectory_set_gate_full_protocol_20261024.md`、`results/trajectory_set_gate_fullfit_eval_20261024_v1/`。

## 5. 推荐组合叙事（可多贡献，但不是拼盘）

题目式表述可以是：

> **Ambiguity-Aware Associative State for Open-Vocabulary Multi-Object Tracking: Evidence Allocation, Bounded Branching, and Causal Commitment**

主贡献建议控制在两条：

1. **A×B 联合关联状态**：assignment ambiguity 不只是 label uncertainty，还决定观测能在 branch 中贡献多少有效证据；观测框不确定性进入唯一分配与 belief 演化，而不是仅做 `R` 缩放或 score gate。可验证对象：低质量观测应在保持 ID 连续性的同时减少错误 memory 写入与虚假恢复。
2. **有界分支提交与污染后效诊断**：固定小 `K`、固定延迟窗口、每支独立状态，在证据条件满足时一次提交；以后续 1/4/8 步 repair/break、AssocA/TETA 和 memory contamination 证明其价值。对照 ARC-SORT 的 label-only swap、FC-Track 的写入冻结、MHT/JPDA 和 no-write。

可选的第三条只作**应用型贡献**：把同一 belief 状态用于 OV-MOT 的 class posterior 分离。必须严格证明身份 membership 与 class uncertainty 是不同变量，分别报告 ClsA/AssocA；不能把分类改善算成关联收益，也不能让 class agreement 决定 identity。若做不出独立收益，删掉这条，不要凑。

这条故事的边界是清楚的：不是“我们又做了一个 attention 网络”，不是“首次不确定性 MOT”，而是一个**围绕错误关联后果的因果状态模型**。论文能把问题、机制、失败后效和计算预算讲成一条线。

## 6. 必须避免的 claim

- 不写 **first / novel** 于：MHT、JPDA、deferred assignment、provisional track、uncertainty-aware association、detector covariance Kalman、historical attention、track-set association、memory contamination 这些方向本身。
- 不写泛化的 “full belief” 而实际只改 `R`、IoU 或 pair score；不把 `Σ_obs` 的乘权叫成 belief update。
- 不把 ARC-SORT 的 margin + swap 复述为 A；不把 FC-Track 的 overlap write-freeze 或 LiAM-SAM 的 reliable memory 当自创；它们是必须跑的强基线。
- “future causal commitment”只能表示**当前帧之前获取的证据在后续帧提交**，不能表示在线算法使用未来帧。固定延迟须报告；没有未来帧的实验不能称 near-online。
- 不把 GT identity/purity 或未来 frame 用于分支生成/提交；GT 只用于事后评分与预先限定的 oracle 上限。
- 不只报 ID switch 或 MOTA：至少给 LocA/AssocA/ClsA/TETA 的完整 base/novel 结果，且不混 TAO 稀疏 val、dense C-TAO-val、MOT17 的 HOTA/AssA。
- 不从 val 挑阈值再称独立泛化；novel 类支持视频数很少时只报支持与类别级结果，不写“新类普适提升”。
- 不把 55M candidates 当统计样本量，不把单一视频的修复当机制成立；报告 source macro、leave-one-source、两个 lag。
- 不把离线 tracklet merge 的增益算作在线 commit 的增益；ARC-SORT 的 merge 就是反例，必须分开。
- 不把 CAMEL/CIP/TFA/MGA 式普通可学习 context fusion 的收益据为己有；新方法应固定 backbone 并单独报告信息/预算控制。
- 不声称本项目集合门或任何缓存实验已证明新方法；当前只有文献判定与实验设计。

## 7. 证据等级与缺口

已读全文（PDF 抽取，方法/实验位置在上表）：UTrack、UncertaintyTrack、ARC-SORT、FC-Track、LiAM-SAM、EviTrack、DMTrack、DeconfuseTrack、CAMELTrack、COVTrack++；此外复用本地 MeMOT/GTR/MOTIP/NOOUGAT 的正文审计。**OVSORTv2 只读到官方摘要与元数据**，正文未获得；因此只用于说明“confidence-aware Kalman in OVMOT 已存在”，不引用其未核实数值。SOVTrack 正文未获得，不据此作机制或数字判断。ARC-SORT、FC-Track、LiAM-SAM、EviTrack 均为 2026 预印本，日期与同行评审状态不应当作已接收事实，必须持续复查。

本地来源证据目录：`OVMOT_RESEARCH/tmp_analysis/novelty_abc_sources_20261024/pdfs/`（11 个 PDF 与 `pdftotext -layout` 文本）。本报告未改 baseline、checkpoint、sealed cache 或实验协议，未使用 GPU。
