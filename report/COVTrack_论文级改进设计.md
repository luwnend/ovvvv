# COVTrack 论文级改进设计

> 日期：2026-10-23；本版替换已撤回的旧分析与推断。
> 目标：从 COVTrack 出发，找到有明确机制、可消融、可证伪的方法改进。
> 当前状态：完整val读出与自然误判诊断已推进到20项控制；固定inverse-density读出未超过普通quality控制，新机制方法尚未成立。
> 完整结果：`report/COVTrack_完整val轨迹读出_20261023.md`。普通聚合增益不等于 TCES 方法已成立。
> 后续机制阶段：`report/COVTrack_自然依赖核验_20261023.md`；主官方复核以独立completion receipt为准。
> 另一个会话的训练目录 `results/covtrack_ctao_train20_bs24_20261022` 受保护，不停止、不覆盖、不修改。

## 1. 研究目标与固定对照

最终目标仍是有效的方法改进；诊断和负结果用于选择方法，不能替代最终方法目标。
不把新增阈值、mask 修正或重命名聚合公式当作论文级创新。
不为了凑贡献数，把尚无证据的分类、关联、生命周期读端拼成三个模块。

固定基线为 `work/covtrack_base`、官方 `ctao_public.pth`、完整 `validation_ours_v1.json`。
发布 A 组参数：`match_score_thr=0.37 / max_per_img=80 / memo_frames=50 / momentum_embed=0.4`。
完整 val 为 988 视频、36,375 帧、112,798 标注；有 GT 的验证类为 296 类：261 base、35 novel。
官方复现实测 Base TETA 39.554、Novel TETA 34.205，采用发布实现原样作为冻结对照。
历史 OVTrack、val60、val200 的结果均不能替代此基线或完整 val 结论。
所有结果绑定 split、checkpoint、检测及跟踪参数、类别列映射、评测器与输入输出哈希。

## 2. 已核实的实现链路

```text
冻结检测器 / ROI 双路分类
  → 按发布类别列 mask 做 text/image 指数融合
  → NMS / 检测截断 / tracker distractor 过滤
  → appearance、location、semantic 的置信度融合与关联
  → 轨迹 memo
  → 全轨迹逐观测标签的多数投票 TCC
```

COVTrack 已有可学习的多线索融合；“让语义或位置参与关联”不是新的空缺。
融合头存在实际训练调用，不能把早期延迟监督描述为完全没有监督。
原分类读出保留逐帧 argmax 后投票，未实现本文候选的显式依赖模型。
当前已有代码主要是双路后验缓存、固定轨迹聚合、官方 TETA 读出与独立诊断工具。
TCES 的可学习可靠性状态、身份混合推断、联合关联及生命周期读端仍是实验候选，没有完整实现。

### 2.1 发布融合 mask 与正确评测分组必须分开

令 `m(c)` 为发布代码的列 mask，`a=1/3`，实际融合为：

```text
m(c)=true : p(c) ∝ p_image(c)^(2/3) × p_text(c)^(1/3)
m(c)=false: p(c) ∝ p_text(c)^(2/3) × p_image(c)^(1/3)
```

不能把这两个分支的权重解释反，也不能用 GT 类别为整行选择指数。
`class_name.py` 的有效验证 mask 有 297 项、86 个 true；它不等于 35 类 rare 分组。
已定位：该 mask 对应 `cid in lvis_novel_label_ids`；正确索引为 `cid-1`。
后者与 annotation `frequency=='r'` 在 296 列上完全一致，前者有 93 列不一致。
其中 72 个 base 类被标 true，21 个 rare 类被标 false，仅 14 个 rare 类标 true。
此行为来自上游发布代码；上游、baseline、缓存副本的 `class_name.py` 哈希相同。

保留发布 A 组不变；单列 corrected-mask 对照，归类为实现校准，不计为方法贡献。
当前缓存上的 mask 修正只改变既有观测的类别读出，保留原框、ID、分数与筛选结果。
它不等于修改 ROI 分类后重跑 NMS、检测阈值和关联的完整系统反事实。
评测 Base/Novel 始终用真实 frequency 分组，不用发布 mask 分组。

### 2.2 稀疏 val 的生命周期边界

官方 val 的 `frame_id` 是采样步号，视频通常约 40 个采样步；A 组 memo=50 很少到期。
不能把这些步号差解释为物理遮挡时长，也不能把大多数断轨归因于 memo 过期。
连续 C-TAO 训练标注不能替代独立连续验证集；dense val 缺失时不推进 dormant 主贡献。
此约束不否认关联失败和漏检存在，只限制可支持的生命周期解释。

## 3. 已纠正的证据问题

旧报告已归档于 `results/tces_audit_20261023/prior_reports`；审计见 `tmp_analysis/novelty_audit_20261023.md`。

| 问题 | 影响与处理 |
|---|---|
| `cov_tces_label.py` 仅按 local track ID 分组 | 不同视频会重置 ID；统一使用 video + track key |
| `macro_novel.py` 用 `p_text` 计算所谓 TCC | 不能作为官方 fused TCC 的失败证据 |
| 聚合前丢弃无 GT 匹配观测 | GT 改变了方法输入；必须先用全部观测聚合，再独立评分 |
| `fusion_adaptive.py` 按 GT rare 状态选整行指数 | 既不等于官方逐列融合，也不属于可部署推理 |
| 独立 nearest-GT 的多对一统计 | 是诊断，不等于官方 TETA 的匹配与聚类协议 |
| 按观测、按轨迹、按类别宏平均混用 | 三者是不同估计量，不能直接相减或替代 ClsA |
| 轨迹温度被当作排序改进 | 正标量 `softmax(E/T)` 不改变 top-1/top-k 排序 |

旧 val60 上“sumlog 有 novel 增益”及“TCES 已无信号”的说法均不能直接采用。
完整观测、原 TCC 对齐与官方指标均已核验。发布 sumlog 的 Base/Novel TETA 为 39.774/35.771（冻结对照 39.554/34.205）；据此继续机制研究，不用旧表决定停止，也不直接训练。

## 4. 与已有方法的真实边界

| 方法 | 已占据的机制 | 本研究不能仅据此宣称新颖 |
|---|---|---|
| COVTrack | appearance/location/semantic 的置信度融合 | 新增语义 cue、置信度门控或残差强度 |
| COVTrack++ MCF | 联合多线索估计可靠性并控制融合 | 将 bbox/cls confidence 接到另一处权重 |
| COVTrack++ MGA | 空间包含关系、可靠子区域、cross-attention | 嵌套框或部件上下文聚合 |
| COVTrack++ TCP | 同预测类别约束的跨帧标量置信传播 | 普通历史置信度传播或检测恢复 |
| TRACT TCR | appearance bank、历史 category bank、置信分段投票 | 多帧记忆、在线可靠标签更新 |
| TRACT TFA/TSE | 分数采样、SA/MLP 特征聚合、属性文本及读出选择 | 泛称轨迹池化、注意力聚合、多源标签选择 |
| OVTR CIP | 类别相关图像 embedding 经 attention/MLP 更新 track query | 泛称在线类别状态或时序类别信息传播 |

TCP 的恢复检测可继续作为后续帧 source；不能把它说成完全不能跨多帧传播。
OVTR 的本地 CIP 传递的是 learned feature/query，不是直接上一帧后验 EMA。
TRACT 本地缺少 TraCLIP 模型主体与实际 tracker 文件；边界来自公开原文及现有入口，不能声称完成其全部代码排除。
COVTrack++ 已实时核验论文官方链接、公开分支与代码树：目前仅main，HEAD `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`，248个blob均与冻结COVTrack一致，尚未识别可核验的MGA/TCP新增实现。不另外克隆同一旧树并冒充++复现；MCF代码与++论文公式也不能直接声称逐式一致。
完整源码和原始论文边界见`tmp_analysis/covtrackpp_code_boundary_20261023.md`及其live-sources证据。

一般领域的边界也已补查：NAN/QAN（2017）已有attention/quality pooling，PFE（ICCV2019）明确讨论视频冗余破坏条件独立，Trick & Rothkopf（AISTATS2022）已有显式相关Dirichlet分类器融合；redundancy attention、temporal covariance pooling、UGG也分别占据互补聚合、二阶统计和不确定身份传播。相关性、GLS、attention或mixture本身不能作为新贡献。

## 5. 主研究候选：依赖感知的双路类别证据

**实验候选 A，尚未成立。** 科学问题是：重复视角是否被误当作多份独立证据，从而稳定放大错误类别？
这不同于“单帧分数低就降权”；它考察一个观测在已有历史之外增加多少信息。
先固定检测、ID、框、完整观测和一种明确命名的融合 mask，构造：

```text
e_t(c) = (1-a(c)) log p_text,t(c) + a(c) log p_image,t(c)
e_t(c) = μ_k(c) + ε_t(c)
Cov(ε_t(c), ε_u(c)) = σ_c² K_k[t,u]
```

K 只能由推理可见特征估计；appearance 相似度不自动等于类别错误的协方差。
μ 的最大类别也不必是真类；系统性分支偏差不能靠减少方差自动修复。
标准 GLS 可作诊断控制：`w=K⁺1/(1ᵀK⁺1)`，`E(c)=Σ_t w_t e_t(c)`。
非负最小方差约束、简单 view-component pooling 也只能作为已有统计工具与对照。
套用 GLS、换一个注意力权重或聚类阈值本身不足以构成创新。
必须证明自然轨迹存在该依赖失效，且解释力与收益超过分数采样、随机等预算采样及普通池化。
精确重复观测不应增加有效信息；对每份复制行添加独立噪声会破坏这一性质。
一般的 sumlog 还可能重复累计类别先验，但普通全局 prior correction 已做过，不能重新包装。
log clipping、缓存精度和核尺度需要固定并记录敏感性，不用 novel GT 挑最优后声称独立泛化。

### 5.1 已执行的固定核控制：停止此具体读出

完整val以`K=((1+cos(raw track-head))/2)^4`、exact-evidence去重、inverse row-density权重读出；不使用GT选择观测。双mask各10项控制已做精确固定几何TETA分类回放，六个已有官方控制逐class/video/alpha计数完全一致，两主density另做完整官方复核。

| 相对各自full sumlog | ΔBase TETA | ΔNovel TETA |
|---|---:|---:|
| 发布mask inverse density | -0.392320 | +0.166917 |
| corrected mask inverse density | -0.264938 | +0.037240 |

普通score-weighted log在两个mask下都同时超过density的Base与Novel，因此当前固定核读出没有增量价值；不继续用val搜索核尺度包装贡献。
同身份的已匹配观测内，控制精确采样lag和score rank，外观相近组同错类别率+5.4955 pp、端点错误率反而-1.7486 pp，支持“错误方向更集中”的描述，但不能直接视为条件残差协方差或可用的风险权重。主分析仅795条受支持轨迹，novel只有13/35类有对比支持。

下一步候选必须把**单帧平均偏差/质量**与**自然共同误判过程**分开。要在base训练的类/视频留出上证明对共同误差的预测增量，并胜过同容量quality/attention与简单相关融合；不能从本次pair关联直接启动完整训练或宣称技术成立。

### 5.2 学习目标的退化检查与可执行边界

同GT身份条件下，hard同错并不是独立的新过程标签：两帧argmax相同则错误指示相同，不同则至少一帧错，`samewrong=agreement*e_i*e_j`。只学这个标签的pair网络可能仅学会质量与类别一致性。另一个退化是使用`mu=p*(1-pi)`中心化wrong-evidence：同GT下每个坐标残差始终同号，正的对角乘积本身不能证明自然依赖。

新目标规范先以平均错误、单帧质量及shared calibrator为控制；主假说为在不含双方当前候选概率的共同参考条件下，直接估计wrong-evidence条件均值，再对out-of-source-group残差的对角乘积/trace做预测。参考只由预测轨迹中双方之外的观测构建，不用GT挑参考。它仍可能包含共享真实标签不确定性和语义难度，称为**中心化wrong-evidence surrogate**；既不是完整协方差，也不预先许诺PSD、GLS或attention方法创新。

具体输入、交叉项、loss、交叉拟合、同容量quality+agreement/attention/相关融合控制及支持门槛见`tmp_analysis/base_error_learning_objective_20261023.md`。精确合成检查只证明代数与退化案例。1,299帧真实pilot及fit/tune支持已独立通过；固定单帧quality控制的层级等权tune Brier由0.2337降至0.1916，但仍低估错误率，实际监督类只重叠3类。由此先扩采全部fit/tune，不能直接进入pair方法训练。扩采元数据独立PASS，用户暂停时尚未启动；完整数值及限定见`report/COVTrack_base观测采集_20261023.md`。未匹配观测保持unknown；test和class holdout不用于选择目标或超参。

## 6. 必须同时区分的竞争解释：身份混合

**实验候选 B，尚未成立。** 同一轨迹内的语义冲突可能来自类别噪声，也可能来自多个身份被关联混合。
不能用类别一致性定义轨迹纯度，再以同一个一致性证明该轨迹可靠，这会循环强化错误主类。
先用只读 GT 诊断划分高纯轨迹与混合轨迹，量化 GT 净化的上界，但不让 GT 净化进入方法输入。
再比较无 GT 的 appearance component/medoid 选择是否保留上界的一部分。
若候选 A 只改善混合轨迹，类别证据相关性的解释不充分，应转向身份污染模型。
若净化上界本来很小，停止候选 B；不直接训练复杂混合状态。
候选 B 后续可能维护竞争身份成分及各自类别后验，但需区别于已有 mixture tracking 与 tracklet splitting。
先测固定 ID 的分类读出，再单独讨论改 ID；不得把两类收益混在一个消融里。

## 7. 三轮证伪，每轮至少两项检查

| 轮次 | 检查一 | 检查二 | 当前证据状态 |
|---|---|---|---|
| 1：实现与评测 | video/track key、全部观测、类别列、TCC tie-break、框/ID/score 哈希一致；原 TCC 必须对齐 | GT 只进入独立评分/oracle；无 GT 帧筛选、GT 分支选择、二次 TCC | 已通过全量缓存、TCC、官方 raw/count 与保护字段核验；见阶段报告 |
| 2：机制与控制 | 精确复制/单帧极限、自然同错率与错误基率；主exact-lag/score控制、另列置信敏感性 | 三seed等预算random、score/FPS、full sumlog、score/entropy weighting | 当前固定核已执行，density未胜过普通score weighting；近似复制干预和可学习error-kernel仍未执行 |
| 3：理论与范围 | single-matched-ID与三类mixed scope分开，逐类支持与mask内差值；不把unmatched身份判为已知 | 六项官方控制回放，主输出完整官方count/raw parity及保护核验 | 具体density的停止结论由本阶段独立completion receipt封存；可学习A与身份B均仍未成立 |

每轮失败先核验原因；版本号 `v3` 或 `dbg11` 不等于完成三轮科学审查。
实现错误会使依赖它的结论失效；不能把修复过程算作理论被证伪。
OVTrack 历史负结果只在原协议成立，不把一个 pair 探针或一个小子集外推成 COVTrack 理论失败。

## 8. 指标、支持数与停止逻辑

主读出用官方 TETA，报告 Base/Novel 的 TETA、LocA、AssocA、ClsA；固定框/ID时核验 Loc/Assoc 数组一致。
诊断另报逐类/轨迹/观测 top-k、支持数和纯度分层，明确它们不是官方 ClsA。
35 个 novel 类中 26 类仅出现在一个 GT 视频；完整 val 并不等于每类都有独立多视频证据。
固定 296 类的 paired video bootstrap 可作会话内经验视频构成扰动诊断，不作独立泛化或显著性主证。
视频重采样后无支持类仍记 0，使其 Δ 为 0，可能将宏平均效应向 0 拉；它不同于重选有 GT 类后重跑 TAO。
报告逐类 GT 视频数、有效分类匹配支持和确定性 Δ；可补 leave-one-video-out influence，不称为置信区间。
不得将区间排 0 当成训练许可；单视频类无法靠更多 bootstrap 重复获得跨视频可识别性。

top-1 改善而 top-10/50 不变可以是有效排序修复，不能据此自动停止。
纯排序方法的 top-k 变化应解释其收益与代价；新增证据的主张需要额外证据，不能由排名变化直接推出。
Base/Novel 权衡与最小有意义改进应在看新候选结果前明确；不靠事后挑类或全局 bias 掩盖退化。
若原 TCC 不对齐、GT 泄漏或保护字段变化，停止解释结果并先修实现。
若一个具体核/结构不能预测目标误差关系、或没有超过质量/预算控制，停止该具体实现；当前固定inverse-density已满足停止条件。不能据一个相似度核失败，将整个可学习共同误判过程候选判为理论不可能。
若身份净化上界小或无 GT 成分选择无法保留收益，停止候选 B。
若只有子集收益而完整 val 不成立，记录范围限制，不能据此启动新训练。

## 9. 执行顺序与训练门槛

1. corrected-mask sumlog官方控制已补齐：Base/Novel TETA=39.837629/36.311125。20项分类回放、自然pair诊断及主官方独立复核见本阶段报告，不把mask修正归入新方法。
2. 终止当前固定appearance-density读出；保留普通sumlog、score/entropy weighting和各预算控制，后续方法必须直接比较。
3. 先完成base-training可观测性与类别/视频留出方案：核验真实原始GT与C-TAO补帧标签、训练/val元数据去重、完整词表映射和已有缓存覆盖。不能直接把val的296维词表移作全部base训练词表。
4. 对可学习共同误判过程，先以独立base训练缓存检验质量条件化的增量预测能力；如果可用监督、类别迁移或oracle headroom不成立，不投入完整readout训练。身份B则先测净化上界与class-blind污染可辨识性。
5. 有超过简单质量/依赖控制的可信信号之后，才进入独立训练目录；新训练只用允许base监督。GPU遵守用户最新授权：允许使用非4GB且启动前没有chenhc进程的设备，已有其他用户进程不构成禁止；保护本账号现有任务。base类留出是readout的pseudo-novel检验，不能称为detector未见类别；当前val选择始终标为探索性，独立泛化另验。

冻结 baseline、上游仓库、既有训练与归档报告；新读出和新候选产物使用独立路径及 receipt。
完成状态要求退出标记、完整指标、输入输出哈希与受保护文件检查通过；启动任务不等于完成实验。
