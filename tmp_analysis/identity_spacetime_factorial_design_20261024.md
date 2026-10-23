# Identity appearance 与时空利用的固定机制对照

本文件仅固定后续实验问题与支持门；当前实现只做 FIT-only lag 支持审计，
不训练、不搜索参数、不重放新 tracker、不改五模式结果。12-source 局部诊断
只说明需要检查跨 source 支持；它不选择本次 source、query、lag 或参数。

## 要区分的两种改变

| 条件 | appearance | geometry |
|---|---|---|
| A0G0 | frozen raw256 | geometry 与 mask 槽置零 |
| A1G0 | identity-only residual adapter | 同上 |
| A0G1 | frozen raw256 | 相对 geometry8 与有效性 mask |
| A1G1 | 同一个训练后冻结的 adapter | 同上 |

这是支持通过后可实施的 2×2 对照，当前证据尚不足以训练。Adapter 的有限接口为
`normalize(x + W2 GELU(W1 x))`，256→64→256，恒等初始化；仅接合法身份监督，
不输入 geometry、semantic 或类别正确性。每个 source fold 内只训练一次 adapter，
再冻结并复用于 G0/G1。四臂评分头保持相同输入槽和结构；G0 将 geometry/mask
槽置零，G1 提供真实输入。semantic512 和原分类 readout 始终冻结，semantic
cosine 仅是本支持审计的固定 control。训练步数、loss、seed、结构和标准化须在
后续独立训练协议中另行冻结；当前不根据结果选择它们。

Appearance 改善必须在 geometry 关闭、**纯 adapter cosine** 读出时仍出现；
同一 frozen appearance 下 G1 相对 G0 的变化才归于几何利用。联合 arm 的改善
不能代替这两个检查。普通 adapter、对比学习与相对几何本身不是新颖性证据。

## 当前支持审计的人口与封印

人口是原 full collection 中全部 sealed FIT source groups，保留没有合法身份
或 hard query 的 source。按冻结 source_roles 与 full_collection_plan 决定人口，
不按局部 12-source 指标或候选分数选择。现有 metadata 为 341 source、403 有帧
video、13791 frame、613536 retained observations；另两个无 anchor allocation
video 没有缓存帧，在人口记录中明确保留为未采集。

lag 固定为 sampled-step `{1,4}`，时间始终是原视频 frame_index 差；lag4 不等于
4 个原视频帧。每个 video 的每个 query frame 与精确向前 lag 的 sampled frame
生成 **全部 retained Q×M**。包括未知、released id=-1、空帧；不跳过空 reference。
候选不按 GT 类别、身份、原预测 ID 或可修复性筛选。FIT source 中 heldout 类别
对象仍作为未知候选存在；监督仅允许既有 audited eligible FIT manual identities，
不打开 class-heldout targets、tune/test 身份或任何类别正确性。

按 video×lag 分 shard，用只含六个 float64 固定分数的 `.npy` memmap 落盘：
raw256 cosine、absolute xyxy cosine、普通 IoU、负归一化位移 norm、负 CV innovation
norm、semantic512 cosine。每 video 独立行元数据保留 box、image/dump row、预测类、
released ID、原帧号、完整 past-history 时间/box/缺失原因。帧对 offset 和 query-major
Q×M 规则唯一确定每个候选的端点，无需重复存储 row/image/video 字段。

CV history 只用 released ID 定位 reference 之前最近的独立原帧；非负重复 ID 当前
行及跨该歧义的下一个 unique 行都没有 CV history。没有 ego compensation。
所有 GT-free shard、行元数据、逐帧对 receipt 及 SHA 完成全局封印之后，才导入
监督投影 helper 和读取 eligible FIT identity payload。未知 identity=-1，不作负例。
manual join 使用 immutable GT-free index 的 `(image_id,dump_row)`；既有 old-ID
edge/pair 支持不用于过滤候选或定义新 truth。

## 报告与支持门

每个 source（连同原 mean/pair role、dataset block）、lag、all-candidate/same-predicted-
category、all-known/common-CV scope 都报告完整候选、已知正负、未知、source/video/GT
trajectory 支持和 CV 缺失。六分数的 CV ranking 必须共用同一 known mask；非 CV
scope 不报告 CV score。Top1 以最高分、最低 immutable reference dump_row 破 tie。
同时报告全部 nonempty-known query 与至少一个 known positive+negative 的主子集。
query 级 rawcos↔各 cue correct/wrong 转移、repair/break、GT/source 错误集中度与
lag1/lag4 同 query 的交集都固定报告；不按 GT 给 tie 选有利候选。

known subset 中没有 positive 只代表当前合法身份覆盖不足，不等于实际 unmatched；
unknown reference 可能就是同一身份。本审计的 no-positive 只报 coverage，不能转成
negative 或 null/no-match 训练标签。未来若 pair head/attention 引入 dustbin，须先有
完整帧标注与可见性，或另行冻结合法 known-negative 无匹配协议。Geometry 的不确定性
与缺失 mask、候选 cycle ambiguity 是独立输入/支持维度；不能用缺失历史或无标签
代替几何可靠性，也不能把 cycle ambiguity 混称为 appearance 本体改善。

支持门只检查必要可辨识条件，不把通过称作训练就绪：至少两个 source 有同类
hard query、至少两个 source 有 rawcos 错误可观察、存在跨 lag 的同 query hard
支持、并存在 source 留一后训练侧与评价侧都有正负 query 的 fold。每个原 source
都列出 fold 的支持情况；不能删除失败 fold 后选择最好来源。错误/repair 若仍集中
单一 source，标记该阻塞，保留完整数据并停止在支持审计；不重新挑 lag 或门槛。
结果最终仍为 HOLD，支持通过后另行冻结并审查具体训练协议，不会由脚本自动启动。

分析单位是 source/video/GT trajectory；候选和 query 可以共享身份与历史，不能
将 5500 万 pair 或一条轨迹的多个观测当作独立重复。当前审计不输出置信区间、
显著性、power、官方 TETA 或因果 assignment 效果，也不开训练。

## 资源上界与预计成本

冻结 metadata 给出 lag1 29096237、lag4 26255927，共 55352164 候选；六个 float64
score 原始占 2656903872 bytes（约 2.474 GiB）。最大单帧 77 row，单帧对至多
5929 候选；单个 score block 约 278 KiB。生成只保留最近五个 frame 的 raw/semantic
与 past-ID history；每 video 元数据最多 41 frame。`.npy` memmap 只按帧对写/读，
不 concatenate 全数据、不一次加载 2.5 GiB。汇总状态按 341 source×2 lag×4 scope
有界；详细 query 与 identity 支持用 SQLite 落盘，逐块交易，不把所有 pair 放 RAM。

CPU 单线程、禁用 CUDA。由先前 20.48 s / 1210939 候选粗略线性外推约 15.6 min；
本次 streaming、hash、SQLite、磁盘不同，预留 15–45 min，仅为估计，未为校时实跑。
程序记录每阶段时间、块数、最大块和 RSS；不根据运行中指标改变人口或配置。
