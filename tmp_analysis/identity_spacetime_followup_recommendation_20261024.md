# Identity adapter × geometry：下一阶段固定协议建议

状态：`PROPOSED_FIXED_EXPLORATORY_IDENTITY_SPACETIME_OOF_PROTOCOL_NO_TRAINING_RUN`。
本阶段只写建议，不训练，不生成新预测，不修改已封印 producer/protocol/design
或结果。支持审计已通过独立全量输出核验；这只排除了当前实现/支持边界阻塞，
不是网络有效、训练充分或新颖性证据。下一实现须按本文件产生可审实现、三轮
synthetic 与独立预审，再执行固定实验；不是增加用户重复批准要求。

## 原角色不能直接改称新 train/test

原 `full_mean_pair_gate_protocol_20261023.json:pair_design.fit_roles` 明确：
`mean_role` 拟合 quality、wrong-evidence mean 和 shared calibrator；`pair_role`
内部做 5×3 source CV，pair-role labels 不得进入 nuisance 拟合/选择/校准。原角色
用于隔离 nuisance 与 pair 机制，并非身份网络的天然 train/test 划分。

因此撤回“直接 pair_role 训练、mean_role 评估即为干净 holdout”的建议：这种
物理 source 分离本身可以做一个新实验，但不能沿用原语义；若继续复用 mean-role
训练过的 nuisance/calibrator，更可能把评估标签带回预测。两侧 source 的支持/
错误分布也已经被检查，不能称为未触碰的确认性 holdout。原角色文件保持不变。

新身份分支另建 source OOF 分配：旧角色只作 metadata 阻断与分层报告；不复用
旧 mean-quality、posterior residual、质量校准器、分类错误标签或其模型选择。
原 detector/checkpoint 的训练边界继承现有 cache，不借此宣称 detector-unseen。

## 科学问题与可证伪预测

1. 同一合法 identity 监督预算能否改善 raw256 的实例区分？A1 的**纯 cosine**
   必须相对 A0 纯 cosine 改善；只在可学习 pair head 后改善不足以支持 appearance
   本体改善。即使成立，也只是缓存后的 adapter 改善，不是 backbone/ROI-head 改善。
2. 在同一个 frozen appearance 下，真实相对 geometry 能否带来超过 appearance
   head、time/missing-mask 和失配 geometry 对照的增量？若不能，拒绝“学会更好利用
   时空”的解释。没有 ego compensation，CV 模型仍有长 gap 与旧 ID 历史错误风险。

支持审计的预定 same-predicted-category/common-CV、p+n query 集合：lag1 为
2949 query/172 source，raw 错 46（29 source）；lag4 为 2142/160，错 149（50 source）。
两 lag 共同 1331 query/107 source，raw 错为 16 与 99，两边都错 8。它说明长 lag
困难在共同 query 中仍存在，不能把两个全量分母直接当纵向比较。

单 cue 替换仍损失更大：CV repair/break 为 26/289、60/533；semantic 为 19/250、
46/292。修复分布跨 source，原局部 +1 没有扩成全量优势。不能因 CV 修复 26/60
就只训练这些 query，也不能据此挑 raw-error source 或重选 lag。

## 人口、source 隔离与固定 folds

仅使用现有 341 sealed FIT source、403 cached video、13791 frame；不读取 tune、
test、class-heldout targets 或 GT 类别正确性。两个未采 allocation video 1720/1974
继续单独报告，不添加 source role。allretained 的未知/heldout 对象仍保留在候选中，
监督只来自现 audited eligible FIT manual identities。

新 fold seed 字符串固定为 `identity_spacetime_source_oof_v1`。按
`(source_group prefix/dataset_block, original mean_role/pair_role)` 分 metadata strata；
各 stratum 内按 `SHA256(seed + NUL + source_group)` 升序，第 i 个 source 分到
`i % 5`。source 的全部 video、GT 身份、观测、lag1/4、reference/history 都同 fold。
原角色仅用于阻断，不能作为数值预测特征。所有 341 source 记录在 fold manifest，
无监督/无 hard query source 保留，禁止按支持或结果重分配、删除 fold。

这条固定规则的 metadata/既有支持核算如下；核算只报告，不调整规则：

| held-out fold | source 数 | 原 mean/pair 数 | lag1 hard/raw错 | lag4 hard/raw错 |
|---|---:|---:|---:|---:|
| 0 | 74 | 37/37 | 798/9 | 570/29 |
| 1 | 72 | 37/35 | 531/7 | 379/25 |
| 2 | 69 | 35/34 | 443/9 | 313/22 |
| 3 | 64 | 32/32 | 683/10 | 535/40 |
| 4 | 62 | 31/31 | 494/11 | 345/33 |

每 fold 在其余四 fold 训练，held-out source 只用于封印后的读出；不设结果驱动的
inner search、早停或 checkpoint 选择，固定最终 epoch。标准化、adapter、pair head
及所有统计仅来自该 fold train sources。五 fold 的 OOF 预测共同组成结果，每个
query 只接受未训练过该 source 的模型；不把某 fold 结果用于下一 fold 调参。
独立测试应突变 model f 的 held-out labels，验证 f 的训练、checkpoint、标准化、
预测 SHA 与 metadata folds 不变；同一 source 在其它 model 的 train 中合法出现，
因此不能要求全部五个模型的训练/预测 SHA 都不变。全部 fold/seed 的 OOF 预测
先全局封印，再统一计算 held-out 指标，避免 fold 间人工读出反馈。

这是已看过支持/基线后的探索性 source OOF；不能宣称新的独立泛化集或确认性检验。
未来泛化证据仍需另行冻结未用于当前设计的合法数据，不能把现 tune 擅自打开。

## 稀疏合法标签与固定训练目标

candidate 永远是同 video 精确 sampled lag{1,4} 的完整 retained Q×M，原帧号决定
几何时间，空帧不跳过。未知 identity=-1，不是负例。监督 join 只用 immutable
`(image_id,dump_row)` 对齐，known positive/negative 由 `(video_id,GT track_id)` 决定。
原 released ID 仅定位过去 history；不作为 truth、positive 或候选过滤器。

训练 query 固定为 train sources 中 **same cached predicted-category、all-known**
且同时有至少一个 positive 和 negative 的 query-lag，两个 lag 都用，不按 CV、分数、
raw 错误或几何修复性筛 query。每个 query 内保留全部合法 P×N 对，不 hard-mine 或
抽取最有利负例。只在训练 loss 投影中使用已知标签；完整候选仍全部生成/评分。

Identity adapter 使用共享
`z = normalize(normalize(raw) + W2 GELU(W1 normalize(raw)))`，256→64→256，
W2/b2=0 恒等初始化，W1 Xavier、b1=0。对原始 raw norm=0 的行，在 adapter 输出
末端显式乘 input-norm>0 mask，保证精确归零；不能只依赖 normalize(0)，因为
learnable biases 会使零输入变非零。输入无任何
geometry、semantic、GT 类别、绝对身份 ID 或 ambiguity。每个 fold/seed 只训练一次
adapter，之后冻结，复用于所有 G 条件；geometry loss 不能反传到它。

Adapter 的 query-lag loss 固定为
`mean_{p in P,n in N} softplus((cos(zq,zn)-cos(zq,zp))/0.1)`。
不是以 class posterior 为身份标签，不使用 unmatched/null、伪 positive 或旧轨迹纯度。
层级权重固定：equal source → equal official query GT trajectory → equal unique
query observation → equal available lag → equal P×N comparison。每个 fold train
重新计算权重；missing-support source 不参与 loss 数值，但保留覆盖与零支持报告。
反向 query 不自动复制为独立样本，重复端点/跨 lag 的相关性明确保留。

known subset 没有 positive 只代表合法标签覆盖不足；unknown reference 可能是真同
身份。此类 query 只报 coverage，不能造 null/no-match/negative 训练标签。未来若
引入 dustbin，先需要完整帧标注/可见性或另行冻结合法 known-negative 无匹配协议。

## 2×2、固定 head 与强对照

| arm | appearance | geometry head slots |
|---|---|---|
| A0G0 | frozen 原 raw cosine | geometry/time/mask/history 全部零 |
| A1G0 | frozen identity adapter cosine | 同上 |
| A0G1 | frozen 原 raw cosine | 真实相对 geometry、time/history、mask |
| A1G1 | 同一个 frozen adapter cosine | 同上 |

另输出 A0 与 A1 的纯 cosine，不能以 A1G0 的 head 收益代替纯表征检验。
head 固定是 `score=s/0.1 + MLP(input)`，input 为 19 维：appearance cosine s，
8 个 geometry 值 `[dx,dy,log_w_ratio,log_h_ratio,log1p(original_gap),ex,ey,CV_IoU]`，
`log1p(history_original_gap)`，对应 9 个 valid masks。MLP 为 19→32→1、GELU、
output layer=0 初始化。所有 arm 同结构/槽/epoch/优化/seed，G0 把全部非 s 槽置零。
这是固定小型 residual association head；当前不扩为 attention/候选图网络。

geometry8 使用既有已审核模块，history 与 NaN/mask 规则完全沿用已封印候选；
CV_IoU 只在 history valid 时有效，static overlap 不能冒充 CV_IoU。标准化人口与 loss
一致：train 的 same-predcat、all-known、p+n query-lag。设 query-lag 全局层级权重
为 w_i，每个 positive edge 的边际权重为 `w_i/(2|P_i|)`，negative edge 为
`w_i/(2|N_i|)`；不能把全部 known refs 等权而让 P/N 数改变 source 权重。每个数值
维度仅取 valid 条目、再全局归一对应边际权重，求 population mean/std；zero-valid
维度固定 mean=0/std=1，其余 std floor=`1e-6`，无 clip/search。所有 A/G arm 共用
同一套 train-fold **原始未干预 geometry** normalizer：先归一、缺失填零保留 mask，
再执行 G0/Gmask 置零或 Gperm 置换；不为 arm 另拟合。s 不标准化。没有 camera H，
没有未来帧，也没有可解释为动力学真值的 uncertainty 标准差。log history gap 是
单独的可靠性 covariate，不能称作校准好的 uncertainty。

每个 appearance 必须再跑两个同结构/预算控制：

- **Gmask**：只保留 original gap、history gap 和全部 masks；七个位置/大小/
  innovation/CV-IoU 数值槽置零。G1 若不超过此控制，只能归于时间/缺失覆盖提示，
  不能说利用了相对几何。G0 的 zero mask 本身亦是条件定义，不声称参数容量完全相等。
- **Gperm**：在每个 GT-free query/reference frame pair 内，对 reference 按
  `(cached predicted category,history-valid)` 分组，组内按 immutable dump_row 升序
  循环移一列，置换七个几何数值，保持自身 gap、history gap、masks。必须先在完整
  retained reference 集合中构造 donor/mapping，再投影 loss-known P/N；不能先按
  GT/known 状态删 reference。group<2 的列保持原样。它只是 GT-free reference
  错位控制，**不保证异 GT、无信息或破坏全部几何**；同身份重复 reference/重复 box
  可让数值仍相同。覆盖分别报告 mapping 非恒等与几何数值实际改变（都只看 GT-free
  mapping/value）；训练与 held-out 同样固定处理，不跨 source/video/time、不改
  candidate/label。完整 Gperm 与 G1 都报告；实际改变部分另作 paired 读出，
  不能按 GT 或收益定义覆盖子集。

G1 相对 G0 的收益若在 Gmask/Gperm 下也出现，不能归于正确时空对齐。semantic512
cosine 保持原固定 control（不送入新 head）；cycle ambiguity 保持独立概念，本轮
不加入，避免同时改变 appearance、geometry 与候选集合上下文。

优化配置建议固定为：CPU 单线程，seeds `{20261024,20261025,20261026}`；每个
fold/seed 的 adapter AdamW lr=1e-3、weight_decay=1e-4、50 epoch；每个 frozen
appearance/geometry arm 的 head 同优化器、20 epoch；batch 为 64 query-lag，
seeded permutation 每 epoch 覆盖每个训练 query-lag 恰一次，grad norm clip=5。
不使用 validation loss 早停，不衰减或扫 lr，不选 seed。全局层级权重和为 1，batch
梯度用 `N/B * sum(weight_i * loss_i)`，N 是该 fold 总训练 query-lag 数，B 是本 batch
实际 query-lag 数（含最后不足 batch）。loss_i 已为该 query-lag 全部 P×N 的均值，
无二次 query 归一；不得逐 batch 重归一权重而改变 source 权重。head loss 是相同
P×N 的 `softplus(score_n-score_p)`。
此配置是一个可证伪的小容量实验，不是性能最优承诺；资源上界在实现预审时估算，
不能因结果不理想加步数。所有 seed 单独报告，seed 平均只是初始化鲁棒性描述。

## 同 query 跨 lag 读出与失败判据

先为每 fold 所有 held-out retained candidates 写出每 arm/seed 的完整 GT-free
score shard、row mapping、训练来源/标准化/checkpoint SHA 并封印，再 join held-out
eligible manual labels。训练时使用 train labels：在共享 JSONL 中先词法投影 split，
再 source_group，跳过 held-out source 后才能解码身份。不能直接调用会加载全部
FIT 身份的旧 helper 来构建训练表。监督投影模块可为 train 导入；在 model f 的隔离
训练进程中，其 held-out 身份值在全部 OOF 全局封印前不得解码或进入 f 的训练/预测
路径；同一 source 在其它 model 的训练标签可合法解码。全部 fold/seed 预测封印后
再统一评价，不能先看某 fold 指标指导其余 fold。拒绝只保存已知候选或预测正确行。

主要读出是预定 same-predcat/common-CV、p+n 的 conditional ranking：按各 lag 全量
分母和两个 lag 同 query 交集（当前 1331）分别报告，固定最高 score、最低 immutable
reference dump_row 破 tie。训练/evaluation 的 unknown 不负例、no-positive coverage
规则一致。另列 all-known population（不要求 CV）、all-candidate scope 与所有
nonempty-known query；不因 geometry 缺失给不同 arm 换分母。

每个 arm/seed 都报告 rawcos paired repair/break、两 lag correct/wrong 转移、
source→GT trajectory→query 等权 correctness、micro count、每 source/原角色/
dataset block 与 CV/missing coverage。零支持 source/fold 仍显示。主要比较是：
纯 A1−A0；A0G1−A0G0 与 A1G1−A1G0；对应 G1−Gmask/Gperm；以及 2×2 interaction。
跨 lag 比较只在同 query 的固定交集上计算，并保留不同参考集合的说明。

若纯 A1 在同 query OOF、两个 lag 中不能同向改善，不支持 appearance 本体改善；
若真实 G1 不能在两个 lag 都超过同 appearance 的 G0/Gmask/Gperm，不支持正确
几何利用；若收益在任一固定 seed 或 source 删除后反号，明确报告鲁棒性失败，
不选择 seed/source/lag 宣称成功。按完整 source 删除一次的敏感性可派生，不重新
训练或改模型。没有预声明效应大小/power，方向支持也不能称充分效果证据；不把
pair/query 作为独立样本做显著性或把 55m 候选当训练样本量。

## 不能据此宣称

本实验没有更新 backbone/ROI/CLIP readout，没有真实逐帧 tracker/memory/assignment
闭环、未匹配目标、ego compensation、TETA/AssocA 或 detector-unseen 类别/视频
评估；稀疏 known subset 的 top1 不是部署全候选 top1。source OOF 仍处在已检查
FIT 人口，不能宣称 test/generalization。普通 adapter/MLP/geometry 控制本身不构成
新颖性，也不修复五模式主结果。若假设失败，保留完整 arm 与分母，不改名、重挑源
或转为任意加权的结果。

## 依据与固定 artifact

- 支持输出 `results/fit_lag_identity_support_20261024_v1/summary.json` SHA
  `9764e2a6f35e4e0a5ce3ab319d15ca4642afda5383289710a5e9dd66e48974c1`。
- 独立全量审计 `independent_output_audit.json`：
  `PASS_INDEPENDENT_FIT_LAG_IDENTITY_SUPPORT_OUTPUT_AUDIT`，SHA
  `99e0d3467529f23c85926591799ae624dfa0802c8d0a597c675b8c89968c6606`。
- 原 `source_roles.json` SHA
  `afa61df067f392b84ba917e1c536a9b64b869dbda19a2236b0a05b6f5c51cb8b`；
  原 mean/pair 协议 SHA
  `fc10e338c0160e389008aa34de13dd384cc0466bdd9df08ba7f937104de00718`。
- 本次只新增此建议文件；上述封印 artifacts、design 与训练状态保持原状。
