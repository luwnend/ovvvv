# 候选：历史条件下的受约束外观预测

2026-10-24。状态：**HOLD，仅数学原型与可证伪设计；神经网络协议尚未冻结或执行，新颖性未建立。** 本文件是独立新设计，不替换任何已封印 protocol、factorial design、baseline、缓存或结果。这里的 causal 仅指推理输入严格来自当前时刻以前，不代表已经识别了物理因果关系。

## 现有机制与实际证据

COVTrack 的 `FeatureFusionModule` 把绝对 box4 和 semantic512 投影到 appearance 维度，再用联合 gate、cycle confidence 和 residual 改写当前 appearance；历史 memory 用 EMA。位置因此是 node cue，时间没有作为显式的外观状态演化变量。相关实码为 `repos/COVTrack/ovtrack/models/roi_heads/ovtrack_roi_head.py:1393` 与 `repos/COVTrack/ovtrack/models/trackers/ovtracker.py:405`、`:529`、`:637`。

现有 COVTrack++ 论文审计已覆盖 MGA 的空间包含/child cross-attention、TAQ cycle gate，以及 TCP 的跨帧检测 confidence 传播。当前候选不能将使用空间上下文、时间记忆、可靠性、残差、attention 或预测 appearance 本身当作空白。2026-10-23 的代码边界审计说明当时公开树仍对应 COVTrack；本次没有重新核验远端仓库，不能把该日期的公开代码状态当作永久事实。

本次只读 `results/fit_lag_identity_support_20261024_v1/summary.json`，SHA256 `9764e2a6f35e4e0a5ce3ab319d15ca4642afda5383289710a5e9dd66e48974c1`。读取时独立 output audit 已为 `PASS_INDEPENDENT_FIT_LAG_IDENTITY_SUPPORT_OUTPUT_AUDIT`；summary 内部状态仍保留 producer 原始的 pending-independent 标签。人口是 341 sealed FIT source、405 allocation/403 cached video、13791 frame、613536 observation；合法 manual anchor 21661。两个无缓存 allocation 1720/1974 不扩充 sealed source。

同预测类别、common-CV、已知正负都可用的 query：

| sampled lag | hard query | rawcos 错误 | 单独 CV 分数修复 raw 错误 | 单独 CV 分数破坏 raw 正确 | raw 错误 source |
|---|---:|---:|---:|---:|---:|
| 1 | 2949 | 46 | 26 | 289 | 29 |
| 4 | 2142 | 149 | 60 | 533 | 50 |

全局必要支持条件通过，有 156 个 source-held-out fold 同时具备两个固定 lag 的训练侧与评价侧正负 hard-query 支持。最终门仍是 `HOLD_PENDING_SEPARATE_TRAINING_PROTOCOL`。上表是稀疏 known subset 的条件排序，未知从未作负例；既不是完整候选 assignment，也不是官方 tracking 改善，更不证明外观变化可预测。直接以几何替代 appearance 的破坏远多于修复，是必须保留强 appearance 和动态 appearance 控制的原因。

## 唯一候选假说

在一部分真实同身份观测中，短时外观变化的方向可由**过去的 raw appearance 轨迹与 box 尺度/运动轨迹**预测，且把该演化限制在一个共享的小子空间，可以减少普通自由残差预测造成的身份漂移。可检验对象是：在相同已知 query/候选、相同训练源和监督预算下，受约束输运能否比普通角速度外推与无约束预测更好地保留同类实例区分，并在两个固定 lag 和未见 source 上呈现 repair > break。

这不是“几何越可信就提高其权重”。它先预测 memory 到当前时刻的 appearance 状态，再比较实际 query 与该预测的创新。训练与评分必须区分三个问题：过去能否预测 feature 演化；该预测是否提高身份区分；完整在线 assignment 是否改善。第一个成立不推出后两个。

## 数学对象与约束

单位 raw appearance 为 `x_p,x_m ∈ R^256`，时间满足 `t_p<t_m<=t_q`，单位始终为原视频 frame-index；无 FPS 不称秒。历史仅以 released ID 定位，duplicate/no-ID 与跨歧义历史按现有规则无效。`q` 的 appearance 不进入状态预测器。

共享正交基 `U∈R^(256×r), U^T U=I`。有限历史 encoder 以 `x_p,x_m` 的过去差异、过去 box 的归一化位移/log尺度变化、真实 history gap 和有效性 mask 产生 `K_m=-K_m^T`，再定义

```text
Δ = t_q - t_m
T_m(Δ) = I + U [ exp(Δ K_m) - I ] U^T
xhat_m(q) = T_m(Δ) x_m
appearance innovation score(q,m) = x_q^T xhat_m(q)
```

预测器在读当前 query appearance 或 query position **之前**产生 `K_m`；本候选不允许以 query appearance 自适应旋转 memory 来获得平凡高匹配。当前 query box 仅进入单独的 motion innovation 对照/评分槽；过去 box 运动可条件化 appearance 的演化，但不能据此声称 box motion 识别了 viewpoint 或 ego motion。无相机观测时不声称相机补偿。semantic512 和分类 posterior/readout 冻结，不进入输运 generator，也不接 identity loss 梯度。

`r` 与 encoder 规格不在这里搜索；只有通过支持与碰撞审查后才在独立训练协议一次性声明。最小数学原型接受调用方提供的 `U,K`，不拟合它们；无合法历史严格回退 last raw appearance，保留原因 mask。

正交性只保证**同一算子下**的范数/内积及 `U` 正交补不变。不同 memory 接收不同 `T_m`，跨身份距离仍可压缩甚至完全坍缩；身份信息也可能恰好位于 `span(U)`。所以“低秩正交 = 身份不变”是错误归因。后续训练至少要保留同类 known-different identity margin/ranking 项；即使训练负例不坍缩，也不构成未见身份的保证。无约束残差、纯正交但无 discriminative loss 都是必要控制。

同一固定 `K` 的 `T(0)=I`、`T(a+b)=T(b)T(a)` 与 inverse reciprocity 是结构恒等式；通过它们只验证实现。真实路径一致性只有在独立 history 更新得到的 `K_i,K_j` 上比较 `T_j(Δ2)T_i(Δ1)x_i` 与从 `i` 的历史直接预测长 gap 的结果才有数据内容；动态、遮挡或视角变化会使此假设失效，不能把固定 generator 半群恒等作学习证据。协议不以 query truth 选择哪条路径更有利。

## 必须先做的无训练强控制

**constant angular-velocity appearance extrapolation** 是普通动态 appearance 控制，绝非创新：由 `x_p→x_m` 最短大圆弧的 past-only 方向，以 `Δ/(t_m-t_p)` 外推。令 `θ=acos(x_p^T x_m)`、`v=(cosθ x_m-x_p)/sinθ`，则

```text
xhat = cos(ρ θ) x_m + sin(ρ θ) v,  ρ=(t_q-t_m)/(t_m-t_p).
```

同点取 last；无历史、zero previous、对跖方向未定义或 caller 标记歧义时取 last。非法 last/nonfinite、future-before-last、有效 history 时间倒置须拒绝。超长 gap 可能转圈，不暗中截断或由 GT 决定退回；先显式报告 ratio/angle 并用反例展示失效，若要限制有效范围必须另行预声明。错误但唯一的 released ID 不能被该数学函数识别，必须做污染历史挑战，不能因函数返回 valid 就称物理身份可靠。

先完成固定数学核及三轮合成，再另行设计真实 FIT 控制输出；不能立即用其结果选 lag/source/阈值。root 已实施的 `tmp_analysis/causal_appearance_transport.py` 是这一层，不代表已跑真实数据。

## 最小原型接口与可观测性

```text
previous_raw, last_raw: [M,256], original frozen rows
previous_time, last_time, query_time: [M], original frame indices
history_valid: bool[M], explicit missing/duplicate/no-ID reason
past_box_change: [M,4], normalized displacement and log-size ratio
past_box_valid: bool[M,4]
U: [256,r], orthonormal; K: [M,r,r], skew-symmetric
prediction: [M,256]; transformed_valid/reason: [M]
pair score: [Q,M], query appearance only used after prediction is sealed
```

上表是未来完整系统接口，**当前原型只实现下面两个数学核**，没有 history encoder、past-box encoder、pair score 或全量真实数据 runner：

```python
angular_velocity_prediction(previous, last, previous_time, last_time,
                            query_time, history_valid)
# numpy [M,D] 输入（D不限256），返回 dict：prediction、directed_arc、
# coincident_static、antipodal_fallback、missing_or_zero_history、
# history_angle_radians、extrapolation_ratio。输出单位化；不改输入。

low_rank_orthogonal_transport(features, basis, generator, time_gap, valid)
# torch features[M,D] / basis[D,R] / generator[M,R,R] / gap[M] / bool[M]
# 返回 torch[M,D]；不额外单位化，保留输入范数；invalid行逐值返回输入。
# generator由调用方提供，原型不声称验证其外部生成路径或学会了预测。
```

当前 `time_gap` 必须与 generator 的反时间单位一致；训练协议必须固定为原帧单位。两个数学核本身没有 proposal、ID、身份标签、分类、box 或 tracker 访问。history_valid 必须由合法 past-only locator 在外层生成，不能拿当前query匹配正确性填mask。

训练期当前可用的是缓存 raw/semantic/box/time 和 eligible FIT manual identities。未知不是 negative；no-positive known subset 不能生成 no-match/dustbin 标签。输运预测损失仅用合法 same-identity endpoint；跨时间独立 generator 路径损失需要至少三个合法 same-identity anchors，**目前 fullFIT pair 支持没有证明这种 triplet 支持**，须先另做 source/trajectory/gap 覆盖审计。缓存无法反传 backbone/ROI/head；若以后训练，仅声明新增 history encoder/U/评分头，semantic 分支与 detector 显式冻结并核梯度。

推理期有过去观测、原帧号、预测 boxes、released history locator 和当前 appearance；没有 GT 身份、GT类别、未来历史或对当前预测是否正确的标签。unknown/新轨迹照常进入完整候选，不作监督筛除。候选的内在不可辨识仍在：相似物体在相同位置交叉、同一物体突变与错误历史可产生相同可见输入；本模型不能给出点可识别保证。

## 最少必要对照与判据

同一 sealed FIT 人口与 source fold、固定 lag `{1,4}`、相同 candidate/mask/tie、同等参数/训练步与 seed 预算：

1. last raw cosine；独立 identity adapter cosine（沿用冻结 2×2 的 appearance 问题）。
2. 普通 constant angular-velocity raw extrapolation，以及普通位置 CV/relative-geometry pair head。第一项不能被 static raw baseline 代替。
3. 等预算 unconstrained appearance predictor，输入完全相同；等预算 appearance+geometry pair MLP 不做 transport，避免只归因网络容量/一般融合。
4. 受约束 transport，另有相同 `U,K` 架构但去掉跨更新路径损失、去掉 known-different ranking 项的控制。固定半群单元性质不算该消融的结果。
5. 保持端点与监督不变、只破坏过去顺序/gap/geometry 的预声明 falsification。若打乱后相同，不能声称学习了时间/空间演化。history 污染挑战和真实合法困难同身份新视角要同时报告，不能拒绝一切新视角来获得“稳定”。

上列是后续训练设计的必要比较，当前尚未执行网络 sweep。先用无训练数学核淘汰不成立实现，再以独立支持审计决定是否值得冻结一个有限训练协议；不根据这些结果来重新挑最优人口。source/video/GT trajectory 是相关单位，5500万 pair 不当作5500万独立重复；报告所有预声明 source fold、source宏平均与dataset block、成对repair/break，不能只报micro或成功fold。候选与query级记录只作条件诊断，官方assignment效果需以后另行冻结完整的在线重放协议；既有用户研究授权继续适用，不增加人为重复审批。

停止门：

- 任一未来/query appearance 进入 generator、无效历史被当有效、时间单位偷换、正交/方向/mask数学失败：停止实现路径。
- 合法三anchor轨迹与hard same-category支持不能跨至少两个原source分布，或路径目标只能靠未知身份推定：停止路径训练，不填伪标签。
- 已见source上预测误差下降但未见source两个lag的身份repair不超过break，或不超过无训练角速度/等预算pairMLP：停止该机制，不用更换lag/source补救。
- 去掉路径约束无损，或打乱时间/gap/past geometry无损：停止相关机制解释；保留普通预测baseline定位。
- 增益仅由单source/GT轨迹或大间隔训练重复主导，或known conditional gain在完整候选和在线因果回放消失：停止泛化/部署主张。
- 近邻已具同样history-conditioned constrained transport与对应训练约束，且我们没有额外可验证对象：停止新颖性主张。可保留其作为baseline工程研究。

## 已知碰撞与未完成检索

本次实时读取的原始摘要：

- [Temporal Dynamic Appearance Modeling for Online Multi-Person Tracking, arXiv:1510.02906](https://arxiv.org/abs/1510.02906)：已利用 temporal appearance dynamics、mid-level semantic feature variation、在线增量更新来改善association。HF Markdown/API均404，已回退arXiv摘要成功；本次没有完整方法逐式阅读。因此历史动态外观不是新意。
- [Discriminative Appearance Modeling With Multi-Track Pooling, CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Kim_Discriminative_Appearance_Modeling_With_Multi-Track_Pooling_for_Real-Time_Multi-Object_Tracking_CVPR_2021_paper.html)：官方摘要明确 multi-track memory update、在线hard tracking episode与bilinear LSTM。动态记忆及同类干扰判别训练不是新意；本次只读取官方摘要，未排除其全部公式。

高优先级碰撞线索：2026 Neurocomputing [Focusing on temporal feature variations: Improving adaptability of Re-identification in MOT](https://www.sciencedirect.com/science/article/pii/S0925231226010349)（搜索摘要SFP feature prediction/alignment，全文403），以及2023 [Multiple Object Tracking With Appearance Feature Prediction and Similarity Fusion](https://doi.org/10.1109/ACCESS.2023.3279868)。后者root已成功读取[DOAJ元数据摘要](https://doaj.org/article/f06ab0cd43ea4798ba6eb64fbb947f3c)：摘要明确由past frames以transformer预测当前feature，并使用global/local appearance与similarity fusion；期刊全文仍受访问限制，不能称逐式查重完成。还需针对 orthogonal/Lie-group dynamic feature transport、Koopman/latent dynamics、view-equivariant ReID、continuous-time neural state-space 做有范围的原文碰撞审查；本文件的低秩指数算子来自标准数学，不以名称新造当创新。

本候选的最窄可研究对象只是：**在冻结OVMOT表征和稀疏合法身份监督下，跨历史更新的可验证输运一致性，是否在同类干扰中比普通动态appearance预测更好地保留身份区分。** 当前答案未知；数学正确、FIT pair支持与候选合理性不能代替这个结果。

## 独立数学核检查

独立审计 `audit_causal_appearance_transport_independent_20261024.py` 只导入原语，不导入 root checker、不打开真实特征或监督。核验的原语 SHA256 为 `8b534c5f133880165e3e46abbd93e49584ebf8e20bec9c58b46ab425fa8f5924`。pinned Python3.8 CPU 实际工具 session `59647`、exit `0`，26项通过；结果为 `causal_appearance_transport_independent_synthetic_20261024.json`，真实执行回执为 `causal_appearance_transport_independent_execution_20261024.json`。

独立oracle包括手算正/反方向与非等间隔、4π超长gap绕回、空输入、无效/对跖/同点/zero-history mask、输入未变、二维封闭旋转与正交补、float32/float64容差、对generator角速度及输入feature的解析梯度。反例明确展示不同正交算子可将两条不同身份向量变成同一点，错误但唯一history仍会被当作有效输入。固定半群检查只证数学结构；以上均是合成实现证据，不是已学输运、真实预测收益、训练就绪或新颖性证据。
