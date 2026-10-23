# 身份外观适配与时空几何的 source OOF 对照

当前阶段：完整独立预审已通过，2026-10-24 02:42:27 Asia/Shanghai 真实 CPU 后台任务已启动；尚无评测结果。

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
磁盘启动门为至少 18 GiB；启动前实际空闲48,115,036,160 bytes。最终同版toy计时按已封存hard人口外推训练约61.4分钟、全评分约19.2分钟；含真实IO/几何/评价/独立复核规划100–240分钟。这是估计，完整实测尚待退出。
冻结 baseline、上游 COVTrack、原数据、旧 sealed 结果及另一会话训练均受保护。

## 当前证据位置

- 固定建议：`tmp_analysis/identity_spacetime_followup_recommendation_20261024.md`
- metadata folds：`tmp_analysis/identity_spacetime_source_folds_20261024.json`
- metadata-only 核算：`tmp_analysis/identity_spacetime_oof_metadata_preparation_20261024.json`
- 数据层独立预审：`tmp_analysis/identity_spacetime_oof_data_independent_preflight_20261024.json`
- 正式协议：`tmp_analysis/identity_spacetime_oof_protocol_20261024.json`，SHA `a7169f8674d8e019782820bc7b4a99ff4b2082bdeaa0a731e60352a488ed9be1`。
- 完整独立预审：`tmp_analysis/identity_spacetime_oof_independent_preflight_20261024.json`，actual exit0、20.267秒，SHA `9fb4633e125a6f646a9de55642ad376da4642df220ff3a87673fe0c9b2dc100c`。
- 启动与进程凭据：`results/identity_spacetime_oof_execution_20261024_v1/launch_receipt.json`、`supervisor_identity.json`、`coordinator.started.json`。结束凭据尚未产生，不能以启动当完成。

## 实现审查与实际运行

数据层、trainer、evaluator、coordinator和finalizer各完成三轮合成，独立预审另实际运行固定50/20 epoch的身份突变挑战、29项全15模型封印异常、8项真实detached进程成功/失败检查和18评分块/5折normalizer/956条SQLite的独立QA。完整临时实验的15个真实worker与统一evaluator均exit0，fresh完整output audit通过，全部十列最大误差2.66e-15。临时共同query数量的显式override不适用于真实实验，真实固定1331门不变。

实际入口审查修复了日志NumPy整数JSON序列化错误；独立auditor还统一了缺历史框的两种等价null表示。它们属于实现/格式问题，不是方法性能失败。真实任务的supervisor、coordinator及首个worker已核验PID/boot/start-ticks/cmdline绑定，CUDA关闭。全部模型预测全局封存前不读出真实OOF指标。
