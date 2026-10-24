# 身份外观适配与时空几何的 source OOF 对照

当前阶段：**15/15 模型、统一评价、fresh 全输出审计及 finalizer 已完成并实际 exit0**；2026-10-24 06:30:27 Asia/Shanghai 完成。现场独立终验通过；当前固定 identity adapter 未改善原始外观特征。

这是普通 adapter 与关联 head 的机制对照，用于决定后续研究方向。已经看过
FIT 的支持和基线，因此结果只属于探索性 source OOF；不能称为新独立泛化集，
也不能以普通 adapter、attention 或融合的组合本身声称论文创新。

## 要回答的问题

1. 在固定合法 identity 监督下，256 维特征经小 adapter 后，其**纯 cosine**
   排序是否改善？只在可学习 head 后改善，不足以证明特征本体改善。
2. 同一个冻结外观下，正确相对几何是否超过外观 head、只保留时间/缺失提示、
   以及参考位置错位的对照？只有正确几何单独提供增量，才能支持时空利用解释。

全部候选、标签预算、epoch、初始化种子和训练 query 人口固定，不选最好 seed，
不只训练曾被几何修复的 query，不把未知身份作负例。

## 数据与隔离

原 341 sealed FIT source、403 个缓存视频、13,791 帧及 613,536 条 retained
观测全部保留。原 allocation 的两个未采视频继续单列，不添加 source role。
sampled lag 为 1、4；真实原帧号用于时间和几何，空帧不跨越。

新 source 分折 seed 为 `identity_spacetime_source_oof_v1`。在
`(dataset block, original role)` 内按 `SHA256(seed+NUL+source)` 排序后取 index
modulo 5；旧 mean/pair 仅为 metadata 分层，不复用旧 nuisance/calibrator。
Root 已只用 metadata 独立核算 source 分配与候选数量：

| 留出折 | source | 视频 | 帧 | lag1 候选 | lag4 候选 |
|---|---:|---:|---:|---:|---:|
| 0 | 74 | 90 | 3,050 | 7,000,967 | 6,303,699 |
| 1 | 72 | 94 | 3,117 | 6,308,527 | 5,661,591 |
| 2 | 69 | 84 | 2,865 | 5,781,528 | 5,215,908 |
| 3 | 64 | 66 | 2,323 | 4,873,993 | 4,421,776 |
| 4 | 62 | 69 | 2,436 | 5,131,222 | 4,652,953 |

数据层独立检查已通过 split→source→eligibility→合法身份字段的解码顺序；
留出、tune、ineligible 及类别/错误字段的 poison 值不进入 JSON value decoder。
这只是数据层预审，不代表整个 runner、训练或收益已通过。

## 固定网络与强对照

CPU float64 单线程，三个 seed 为 20261024、20261025、20261026。每个 fold/seed
训练一个 256→64→256 恒等初始化 residual identity adapter，50 epoch 后冻结；
精确零输入 mask 防止 bias 将零向量变成有效外观。

每种原始/适配外观分别配四个 19→32→1 residual head，20 epoch：

| 条件 | 提供给 head 的信息 |
|---|---|
| G0 | 仅外观，全部 geometry/time/mask 槽为零 |
| G1 | 正确相对位置、宽高比例、原帧差、CV innovation/IoU 与有效性 mask |
| Gmask | 只保留原帧差、history gap 和 masks |
| Gperm | 完整 GT-free reference 池内固定错位七个几何数值，保留自身时间和 masks |

另存 raw 和 adapter 的纯 cosine，共十列。geometry head 不向 adapter 反传。
归一化仅使用该折 train 的合法 loss 人口，所有 arm 共用原始 geometry normalizer；
先归一再干预。query loss 为完整 P×N，权重为 source→官方 GT trajectory→唯一
query observation→可用 lag 等权，batch 使用 `N/B × sum(global_weight×loss)`。

## 执行、封存与读出

每个 fold/seed 为独立训练子进程，仅解码其 train-source 的合法身份。其留出
candidate 始终是完整 retained Q×M，包含 unknown；所有十五个模型的 checkpoint、
normalizer、十列 score、rowmap 和 receipt 全局 SHA 封存后，才统一连接评价身份。
后台 supervisor 必须记录真实 child wait/exit，最后由新进程独立重算输出。

评价覆盖两种 scope、两种 population、all-nonempty 与 P+N 子集，分别报告每个
seed、source、旧 role 和 dataset block；给出 raw paired repair/break、层级宏、
固定共同 query 跨 lag、几何强对照、2×2 interaction 与 source 删除敏感性。
共同 query 的 reference/history 仍随 lag 变化，不能将其差异解释为时间因果作用。

all-nonempty 读出沿用 parent 的已知候选子集定义：没有已知 positive 的 query
无法在该子集中命中 positive，因此其 correctness 为零。这是稀疏标签覆盖限制，
不能据此认定部署关联错误或确定没有匹配；机制主读出只用 P+N 俱全人口。

全量评分三 seed 合计 166,056,492 条候选记录、十列 float64 原始存储约 12.37 GiB。
磁盘启动门为至少 18 GiB；启动前实际空闲48,115,036,160 bytes。最终同版toy计时按已封存hard人口外推训练约61.4分钟、全评分约19.2分钟；含真实IO/几何/评价/独立复核规划100–240分钟。这是启动前估计；现完整实测为约 228 分钟（含约 60 分钟 fresh 全输出独审），CPU，无 GPU。
冻结 baseline、上游 COVTrack、原数据、旧 sealed 结果及另一会话训练均受保护。

## 当前证据位置

- 固定建议：`tmp_analysis/identity_spacetime_followup_recommendation_20261024.md`
- metadata folds：`tmp_analysis/identity_spacetime_source_folds_20261024.json`
- metadata-only 核算：`tmp_analysis/identity_spacetime_oof_metadata_preparation_20261024.json`
- 数据层独立预审：`tmp_analysis/identity_spacetime_oof_data_independent_preflight_20261024.json`
- 正式协议：`tmp_analysis/identity_spacetime_oof_protocol_20261024.json`，SHA `a7169f8674d8e019782820bc7b4a99ff4b2082bdeaa0a731e60352a488ed9be1`。
- 完整独立预审：`tmp_analysis/identity_spacetime_oof_independent_preflight_20261024.json`，actual exit0、20.267秒，SHA `9fb4633e125a6f646a9de55642ad376da4642df220ff3a87673fe0c9b2dc100c`。
- 启动与进程凭据：`results/identity_spacetime_oof_execution_20261024_v1/launch_receipt.json`、`supervisor_identity.json`、`coordinator.started.json`。最终凭据为 `completion_receipt.json`、`control_brief.json`、`independent_output_audit.json` 及 execution 目录的 `execution_receipt.json`、`finalization.exit.json`；均已现场核验。

## 实现审查与实际运行

数据层、trainer、evaluator、coordinator和finalizer各完成三轮合成，独立预审另实际运行固定50/20 epoch的身份突变挑战、29项全15模型封印异常、8项真实detached进程成功/失败检查和18评分块/5折normalizer/956条SQLite的独立QA。合成/预审数值复核的最大误差为2.66e-15；真实完整实验的15个worker与统一evaluator均exit0，fresh完整output audit最大绝对误差为5.684e-14，raw逐候选bitwise精确。临时共同query数量的显式override不适用于真实实验，真实固定1331门不变。

实际入口审查修复了日志NumPy整数JSON序列化错误；独立auditor还统一了缺历史框的两种等价null表示。它们属于实现/格式问题，不是方法性能失败。真实任务全程 CPU。十五个模型预测全局封存先于统一 identity 评价；完成后才读取最终指标，没有提前选择 seed/fold。

## 最终结果与判定

协议为已看 FIT 上的探索性 source OOF；主分母是 same-predicted-category / common-CV-covered / 已知 P+N 的 lag1 **2,949 query / 172 source**、lag4 **2,142 / 160**。两 lag 共同子集另为 **1,331 query / 107 source**，不能混用分母。所有宏差均以 source→GT trajectory→query 层级等权计算，单位 pp。

| seed | lag | raw 错误数 | adapter 错误数 | A1G1 错误数 | adapter−raw 宏差 | A1G1−raw 宏差 |
|---|---:|---:|---:|---:|---:|---:|
| 20261024 | 1 | 46 | 87 | 77 | -2.4913 | -1.8540 |
| 20261024 | 4 | 149 | 221 | 212 | -4.3822 | -3.4361 |
| 20261025 | 1 | 46 | 85 | 74 | -1.8682 | -1.4527 |
| 20261025 | 4 | 149 | 210 | 205 | -3.2013 | -2.8949 |
| 20261026 | 1 | 46 | 93 | 79 | -2.0518 | -1.6633 |
| 20261026 | 4 | 149 | 201 | 196 | -3.5139 | -2.5827 |

纯 adapter 的三 seed 平均宏差为 lag1 **−2.1371 pp**、lag4 **−3.6991 pp**；只是固定种子的描述均值，不将 seed 当独立 source。A1 的 G1 相对 G0/Gmask/Gperm 在完整各 lag 人口中均正，且逐 source 删除后保留方向，但 A1G1 仍落后 raw。A0 的 G1−G0 在 seed20261026/lag4 为 −0.0325 pp，不能主张几何普遍稳健。

固定共同 1,331 query 上，adapter 的纯外观宏差仍全部负；geometry 的补救和 2×2 interaction 则随 seed/lag 变化。seed20261025/lag4 的 A1 G1−G0 为 **−0.1241 pp**、G1−Gmask 为 **−0.3466 pp**。A1G1 的 micro 在六个共同子集 cell 均落后 raw；宏平均有一个 cell（seed20261024/lag4）高于 raw **0.0286 pp**，所以结论是未一致恢复 raw，不能写成所有宏 cell 均负。完整共同集表和删除敏感性见[独立判读](../tmp_analysis/identity_spacetime_oof_independent_completion_audit_20261024_简报.md)。

### 三轮各两项的负结果检查

1. 核全部 15 个真实训练 trace 的 50/20 epoch、batch/step/finite 与固定层级权重；核 sealed checkpoint 有限且原零初始化 residual 权重已更新，实际 label-mutation QA 仍有 SHA 绑定。训练 surrogate 首/末范围约 0.0692–0.0816 / 0.000660–0.000871，只是不同优化步骤上的描述，不能当完整固定状态 loss 或收敛证明。
2. 绑定 fresh auditor 对每候选全部十列重算（最大绝对差 5.684e−14，raw bitwise 精确及 8,208 cell 父基线精确）；逐模型核 train/heldout 隔离与合法 identity projection，unknown 不作负例。独审没有重放 optimizer。
3. 核全部 seed/lag 的固定 hard 人口负方向；再分开核共同 1,331 query、几何对照和 interaction，避免以某一正向 cell 或较弱 adapter 的补救代替 raw 改善。

检查脚本 `tmp_analysis/diagnose_identity_adapter_negative_20261024.py` 实际 exit0；收据 `identity_adapter_negative_diagnosis_20261024.json`。未发现可解释该负结果的执行、标签投影或打分错误；不能认定唯一失败原因。**停止将这一固定 adapter 配置作为改进，不扩大同配方训练；不外推为所有身份学习或时序网络失败。**

### 完成链

现场终验重新 hash 3,717 sealed members；70 launch pins、68 global pins 与 7 个保护状态一致。15 worker、evaluator、coordinator、fresh auditor、finalizer 全部有真实 parent-wait exit0。global seal SHA `42e47475…`、summary SHA `b630f89c…`、fresh audit SHA `af4afb5c…`，completion 中绑定全部路径/SHA。独立终验 `tmp_analysis/identity_spacetime_oof_independent_completion_audit_20261024.json` SHA `d4f4f7e6…`。全量 `evaluation/summary.json` 约 563 MB，常规续接先读 99 KB 的 `control_brief.json` 和终验简报，不重复整份打印。

下一步先验证严格 scene residual 是否给完整候选的合法 P/N 提供额外身份信息；既有 donor 的 fullpool-MNN h 预测不能代替该门。新颖性、backbone 改善、官方 tracking 收益和独立泛化仍未建立。
