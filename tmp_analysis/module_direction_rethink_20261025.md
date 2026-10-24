# OVMOT 模块级方向重审：从更多证据转向证据所有权

日期：2026-10-25
状态：`DESIGN_AND_FALSIFICATION_ONLY`。本文只读现有代码、结果和文献审计，没有启动 GPU、训练、采集或修改冻结产物。文中“建议实现”不等于已经实现。

## 判断

**主方向选一个：类别条件的因果证据所有权账本（Causal Evidence Ownership, CEO）。** 它不再问“怎样让同一轨迹的多帧证据权重更聪明”，而问“父框里的这块视觉证据到底属于父对象、子检测，还是背景；同一块证据能否被多个对象重复计入”。

这个方向有三个与既有失败直接对应的支点：

1. 当前轨迹多帧错误高度相关，继续增加同源帧或学习一个普通的帧权函数，没有提供新的独立性。像素区域的所有权是一个比 posterior 相关性更具体、可干预的变量。
2. 现有 ROI 信息门否定了“历史局部 token 提供普通身份证据”；它没有否定“局部证据属于哪个对象”这一分类问题。CEO 只在检测到的包含关系内做有符号、类别条件的归因，不把 token 当成新的全局身份描述子。
3. `person→shirt/jacket`、`car→taxi/pickup` 等错误表明，部分区域同时在支持父框和子框。普通 part attention 会把子证据加给父对象；CEO 要减去子框能够解释的那部分父类别正证据。

第二方向“证据创新滤波”作为潜在的第二篇/第二贡献，先用 CPU 判定是否值得。第三方向“所有权约束的集合关联”只在 CEO 已经证明存在真实所有权信号后考虑；当前不应先改匹配。

## 已确认的约束

### 数值与人口

- 冻结 COVTrack 完整 val：Base/Novel TETA `39.554/34.205`。
- corrected-mask sumlog：Base/Novel TETA `39.838/36.311`，增益全在 ClsA；LocA、AssocA 数组逐项不变。它是任何分类模块必须超过的强控制。
- 已完成的普通历史集合信息门中，`fixed_attention` 相对 `raw_ema` 在两个 lag 的 hierarchical macro 都下降（lag1 `0.9219` vs `0.9575`，lag4 `0.8904` vs `0.9369`）；而 last 在 lag1 高、lag4 又低。因此后续任何 CEO 对照都必须保留 raw256/EMA/last/mean，不能只拿 sumlog 当强控制。
- 同一轨迹两帧同时出错的概率为 `0.653`，独立基线为 `0.113`，约 `5.8x`。`log p_text` 由语义 embedding 线性重建的 `R²=0.967`。这两项共同解释了为什么普通 sumlog、帧选择和线性语义重加权很快饱和。
- 可判定的 val 分类人口是 **9,734 条预测轨迹**，其中 9,466 条只匹配一个 GT 类，268 条匹配多个 GT 类并被排除；9,466 不是 GT 轨迹数。val annotation 有 5,473 条 GT 轨迹，完整预测有 324,639 条预测轨迹。
- 全局 part→whole 名称映射在 9,466 条单一 GT 类预测轨迹上的微平均准确率是 `0.39246→0.56677`，但 291 个有 GT 类的宏平均是 `0.24086→0.23843`。**微平均不能替 TETA/ClsA。** 该映射还把小支持子类完全压没：`cab 13→0`、`jacket 3→0`、`pickup 2→0`、`shirt 2→0`。
- 后续 source 留出复核把整体—部件关系固定并重放：development 仅 fold 0，heldout 为 fold 1–4；heldout 的 7,233 条单一 GT 类事件预测轨迹覆盖 1,739 条可见 GT 轨迹、677 个 source。全局映射在 heldout 为 `1,363 changed / 1,266 repaired / 20 broken`，97 个有支持类的类宏从 0.249965 到 0.24714（**−0.283 pp**，这是事件级诊断值，不是官方 ClsA）；同一父边重复语义门为 `155/147/0`、类宏 `+0.018 pp`；父轨迹实际发射语义证据为 `291/278/1`、类宏 `+0.023 pp`。后两项增益极低且仍主要由 `person` 驱动。**结论已降级为零参数强基线/诊断，不再作为主网络创新。** 原先全 val 的 `383/352/2` 仅保留作实现修正历史，不应替代 source-heldout 结果。
- 2,574 个有包含 child 的父轨迹中，1,519 个父轨迹终类错误；其中 280 个父轨迹（18.43%）的某一 child **逐帧 detector label 众数**等于父框误判类。这是筛选上界，不是 child 遮挡造成的因果修复率。`child_mode` 与 child 轨迹终类是两个字段；5,109 条边中 2,083 条两者不同。
- 关联冲突策略中 appearance/geometry 选择不一致时，修复/破坏为 `15/175`，12 个 source 无一为正。纯几何片段合并的官方结果也下降。不能因为 GT oracle 显示局部关联头寸，就直接修改匹配。

### 代码入口

真实工作副本为 `work/covtrack_base`。

- `ovtrack/models/roi_heads/ovtrack_roi_head.py:1584` `OVTrackRoIHeadUncertainty`；`:2230` `get_classification_head_features`；`:3122` `simple_test_bboxes`；`:2931` `uncertainty_simple_test`。
- `ovtrack/models/roi_heads/ovtrack_roi_head.py:1393` 是 `FeatureFusionModule`。它已有 appearance/bbox/semantic 的 gate 和 cycle confidence；不要重新命名 gate。
- `ovtrack/models/roi_heads/track_heads/quasi_dense_embed_head.py:101` 把 `[N,256,7,7]` 经四层 Conv、flatten `256×7×7=12,544`、FC1024、`fc_embed` 压成 256 维。身份 token 的真正插口在 flatten 之前。
- 分类侧 `_bbox_forward` 能拿到 `bbox_feats [N,256,7,7]`；当前公开返回只保留 text/image 的 512 维 embedding 和 posterior，没有把二维网格交给 tracker。
- `ovtrack/models/trackers/ovtracker.py:481` 是 `OVTrackerUncertainty.match`；`:674-684` 逐 detection 取最大 score 并清空已占用列，是顺序依赖的贪心关联。
- `ovtrack/models/trackers/ovtracker.py:405` 将 fused appearance 写回单条 EMA，`:443` 每 track 只暴露最近一个 memo。

本项目的 geometry-ROI 信息门在 122 个合法 anchor、129 个正对、241 个 hard negative 上显示：box-aligned 历史 token 的表面增益主要来自同一 local track 的历史连续；严格过去且跨 local track 时，相对 raw 的 source-macro AUROC 为 `−0.2613`。因此本文不设计“用更多历史 token 做 identity attention”。CEO 的新变量是**当前帧内区域的类别证据所有权**，验证时必须把历史 token、box prior 和同 track 平滑作为强控制。

## 主候选 A：类别条件因果证据所有权账本

### A1. 科学问题与可写 claim

研究问题限定为：

> 在开放词表检测把完整对象拆成嵌套的父/子框时，父轨迹的类别后验有多少正证据实际由子框区域产生；这部分证据能否由类别条件、过去可见的归属关系预测，并在不改变框、检测分数和轨迹 ID 的情况下从父轨迹的类别账本中扣除？

可写的贡献边界是：

1. **诊断**：开放词表 MOT 中父子检测造成父轨迹类别证据重复计账，且主要集中在少数整体—部件混淆族；报告跨 source 的支持与离散度。
2. **机制**：类别条件的正证据 explain-away，而不是父子 cross-attention 的加法聚合。同一个子框对父类别 `shirt` 与 `person` 的影响必须不同。
3. **因果监督**：用真实图像输入处的子区域遮挡产生 teacher，学习可部署的归属账本；不是用类别共现或手工关系表当监督。
4. **时序归属**：子框对父框的占用必须在过去帧稳定，单帧突发框不能重写整个轨迹标签。

不能写的 claim：首次使用包含关系、首次做 part attention、首次做 relation reasoning、首次把子特征给父对象。COVTrack++ 的 MGA 已经用 IoC、TAQ 和 parent-query/child-key-value 做加法聚合；JointDet、DSA、PartCATSeg、VLPart、FreeCP、ProCal、contextual relabelling 分别覆盖了关系验证、检测删除、部件建模、类别去冗余和上下文重打分。

### A2. 与 MGA、普通 attention 的明确差别

把父检测 `p` 的子检测集合记为 `C_p`。对父类别 `c`：

```text
L_pc = log p(c | parent ROI)                         # 基线类别账本
w_jr ∈ [0,1], sum_{j∈C_p} w_jr <= 1                 # 父 ROI 第 r 格被 child j 占用
D_pjc = sum_r w_jr * max(E_pc(r), 0)                # child 区域内父 c 的正证据
a_pjc = h(q_j, g_jp, phi(t_j, t_c), s_{j,t})        # 类别条件归属强度
a_pjc = a_pjc * (1 - q_j(c))                        # child 自己支持 c 时少扣
L_tilde_pc = L_pc - λ * sum_{j∈C_p} a_pjc D_pjc
```

其中 `q_j` 是部署时 child 的 1203 类 posterior，`g_jp` 是 IoC、面积比和相对位置，`phi(t_j,t_c)` 是 child 与父类别的文本关系表示，`s_{j,t}` 只由过去帧的包含稳定性、child 轨迹持续时间和 posterior 稳定性构成。

机制约束：

- **只扣减，不聚合。** `D` 只取父类别的正证据；child evidence 不经过 attention 加回父特征。若实现退化成 additive child pooling，直接判 collision。
- **类别条件。** 同一 child 对 `shirt` 和 `person` 的扣除量不同。去掉文本关系、child posterior 或类条件后若结果不变，贡献只剩 mask/scalar gate，停止。
- **不重复占用。** 同一父 ROI 网格格点被多个 child 覆盖时，`w_jr` 归一化到总占用 `≤1`，避免多个 child 对同一块证据重复扣减。
- **无 child 时严格恒等。** 无可部署 child、关系无效、比分不足、missing history 时输出精确 fallback 到基线 posterior；不允许 bias 把空集合变成有效修正。
- **类别账本在轨迹级聚合。** 基线可以是 TCC 或 sumlog，但每个父观测只贡献一次 ownership correction；不能先扣帧、再做一次 TCC 把修正重复累计。

### A3. 因果 teacher 与可训练头

只在训练 split 上生成 teacher。对固定父框 `p`，保持父框坐标不变，分别计算：

```text
full:         log p(c | image, parent box p)
child-mask:   log p(c | image with the detected child box masked, parent box p)
area-mask:    log p(c | image with same-area, position-matched control mask)
T_pjc = max(0, full_c - child-mask_c)
```

父框在 masked 图上也使用同一坐标，不重新跑检测/NMS，避免用框变化冒充证据变化。真实 child mask、面积匹配 mask、父框内非 child 等面积 mask 和错误父框 child mask 必须是四次独立前向。训练时 GT 只用于确认父/子轨迹的合法类别和采样；部署输入只使用 detector 输出和过去可见状态。

小头 `OwnerLedger` 的输入是：

- `[N_p,256,7,7]` 父 ROI 空间特征，来自 `_bbox_forward` 的 `bbox_feats`；
- 每个 child 的 `[q_j, g_jp, t_j, s_{j,t}]`；
- 父文本 embedding `t_c`（或从冻结词表取到的类别 embedding）。

输出 `[N_p,N_child,K]` 的类别条件归属，`K` 只保留基线和 child 的候选竞争类（例如 top-16 + true/sampled class），不直接学习 1203×child 的大表。`K` 候选必须由部署可见的 posterior/文本邻近确定，不能按 GT 选。参数预算先控制在 `1M` 以内，backbone、CLIP projection、检测头和 tracker 冻结。

损失：

```text
L_teacher = Huber(hat_T_pjc, T_pjc) + λ_ce CE(corrected_parent, train_label)
            + λ_sparse * smooth_l1(hat_T outside candidate set, 0)
```

`T` 只作为训练教师；线上不能重跑遮挡。线上若 owner 不自信，`a=0` 并走基线。

### A4. 实现落点

最小侵入的三步：

1. 新文件 `ovtrack/models/roi_heads/evidence_ledger_head.py`：`OwnerLedger`、ROI/child mask 投影、类别条件输出和严格 fallback。
2. 修改 `OVTrackRoIHeadUncertainty._bbox_forward` 的可选返回值，暴露 `bbox_feats [N,256,7,7]`；默认 `return_spatial=False`，不改变原 inference。在 `simple_test_bboxes` 只对 NMS 后保留的 detection 行取得 token，返回 `evidence_tokens`、`det_inds` 和原始 posterior。
3. 新脚本 `tools/evaluate_evidence_ledger.py` 在**固定框/固定 ID**上只重写 `category_id`。检测框、score、track ID、输出顺序、Loc/Assoc 数组必须逐项相同。第一版不把 ledger 分数接回 `OVTrackerUncertainty.match`，避免同时改变 ClsA 和 AssocA。

第二版若要把 token 接入训练，新增 `configs/uncertainty-ovtrack-teta/ovtrack_r50_evidence_ledger.py`，用新 work dir；不得覆盖 `work/covtrack_base` 的冻结配置和 `ctao_public.pth`。track-level 推理可以用独立脚本，不要求重跑 detector。

### A5. 预注册实验阶梯

**G0：零参数关系与统计重放（CPU，几分钟）**

- 修正 `semantic_containment_repeat` 的同一父边实体条件，固定阈值后再跑；
- 只报告逐类宏平均、官方固定人口 ClsA 的重放、source 支持、child recall；
- 全类别名称映射、反向映射、随机同基数映射、类别频率先验、纯 IoC/面积 gate 都作为强基线；
- 任何“微平均提升但类宏平均下降”的规则不能进入下一步。
- 先用已冻结且不读结果的 video/source hash split（development fold 0；heldout fold 1–4）做 G0，禁止按 provider 名称或类别结果重新切分；后续正式的 provider 迁移验证另算外部检验。只报告每个 source 的覆盖率与增益，不能把一个 source 或 `person` 当独立样本。现有 11 对映射是看全 val 混淆后的探索规则，只能作强控制，不能作为新方法的独立关系；
- 当前 `source_split.json` 把 AVA 的若干 video 归为同 source，但另一个冻结事件表没有 AVA source 标识，现有 diagnostic 的 limitation 明说它把 video 当 group；因此该 heldout 只能作探索性 VAL support，不能当独立泛化。要生成可审计的 provider-level 迁移结果，必须先把所有事件/预测轨迹可逆连接到 `video_source.json`，否则 G0 只能报告“source 缺失/未覆盖”，不能用默认值补全。
- 在现有留出 fold 上，语义 child GT 的支持很稀疏（全局语义子类合计约 146 个 GT 轨迹；分到 provider/source 后有的类只有 0–3 条）。不能把单个 source 的单类修复写成通用 part—whole 机制；至少要求修复跨多个 source，且 child-class recall 的下降被明确计入。



**G1：真实遮挡因果门（GPU pilot，目标 20 分钟内）**

从 FIT source 的设计折中固定选 120 个 parent-child 事件，覆盖至少 6 个 source，每个事件跑 full/child/area/non-child 四个前向；同一 parent 的多 child 作为同一事件聚类，不把 edge 数当独立样本。先只做前向和封存，再接身份/类别正确性。

预注册通过条件：

- 真实 child mask 的 parent top-1 repair/break `≥2:1`；
- source-cluster bootstrap 的单侧 95% 下界大于 0；
- 至少 3 个 source 有修复，删除最大修复 source 后方向不翻；
- 与 area/position-matched mask 相比，增益至少保留一半；
- 若只有 `person` 一个类对贡献，主模块只能作为窄范围应用，不进入通用训练。

若 pilot 只跑 120 个事件，通过只代表“有因果信号”，不宣称泛化。扩展前必须使用新输出目录并重新冻结采样和 random mask 种子。

**G2：source-OOF 学习与归因门**

用 `source→video→GT trajectory` 分组，5 个外层 source 折；每折训练时绝不解码留出 source 的类别正确性。整个 config、词表、阈值、类别候选和随机种子在外层折启动前冻结。五折全部模型和分数先封存，再统一接入身份/类别。报告：

- 纯 `raw256` / 无 ROI token；
- 全局 sumlog；
- 零参数 containment；
- `OwnerLedger`；
- class-agnostic ownership；
- scalar confidence gate；
- additive child attention（MGA-like）；
- learned class table（不用文本关系）；
- random relation text；
- 正确 token 与随机/错位 token；
- 错误父框 child mask 或错误时间 child。

停止条件：学习头没有在五折 source-macro 上同时超过“零参数 containment”和“class-agnostic gate”，或者 `Huber(teacher)` 在留出 source 上显著高，直接 STOP；不能以训练损失下降代替泛化。

**G3：只改类别的官方闭环**

生成新的 prediction JSON，保持 bbox、score、track ID 和行顺序逐项不变，只改类别；先做固定人口重建审计，再跑官方完整 TETA。必须分别报 Base/Novel 的 TETA、LocA、AssocA、ClsA，以及宏/微轨迹 top-1、逐类支持、child recall。

相对 **corrected-mask sumlog `39.838/36.311`**，最小可写要求：

- 不出现 LocA/AssocA 数组变化；
- 不能使用“9,466 微平均”替代官方 ClsA；现有正确的 corrected-mask sumlog 是 `Base/Novel ClsA 20.397666/9.824471`，任何映射的类宏下降都要单列；

- Base 与 Novel TETA 不反向；
- 至少一个 split 的 ClsA 提升有 source-cluster 下界大于 0；
- 类宏平均不下降；小支持 child 类不得因路由被整体清零；
- 收益不能只来自 `person` 或单一 video。

若只有 val 微平均改善而 class macro/TETA 不改善，判 STOP。若只有 Base 改善而 Novel 下降，不能写“开放词表普适提升”，只能写“Base 类整体—部件污染诊断”。

### A6. 当前证据不能证明什么

当前 parent/child 边来自固定 val 的 9,466 条可判定预测轨迹子图，268 条歧义轨迹及其余未进入分类事件的预测没有入图。现有 280/1,519 是预测轨迹上的筛选上界，既不是按 GT 轨迹去重的总体，也不是因果修复率。G0 必须先把 `global_tid` 可逆映射回 `(provider, video_id, local_track_id)` 并封存；若不能用 metadata-only 重建，就不能把该比例外推到全部 val。

当前也没有官方完整 TETA 的新输出。所有 `0.5668`、`352/2` 之类数值都是固定轨迹类别事件上的描述性读出，不是 Base/Novel ClsA。它们的唯一作用是定义基线和停止条件。

### A7. 合成原型的实现状态

`tmp_analysis/evidence_ownership_head.py` 仍只是独立合成契约原型，尚未接入 COVTrack。独立复核版本历史：

- `eab492b8`：FAIL。owner 选择、per-parent 上限、1203 类宽度、NaN fail-closed、posterior detach 均有问题；
- `c4684e66`：FAIL。这些多数修复，但空 geometry shape 未验证、接口不含时间、relation 仍退化为共享空间标量；
- `68fa8839`：又发现 empty shape 只靠 assert、`class_width` 不要求精确匹配、time dtype 未限制、`max_children` 口径混淆；partial-overlap 探测本身也需重做。
- 当前修订在 `68fa8839` 基础上改为正式异常校验、精确 1203 类宽、`max_children_per_parent` 与全局 `max_edges` 分离、严格整数 frame time；作者契约通过。
- `833a4bf6` 的最终独立 CPU 复核 15/15 PASS：上述 shape、宽度、time、上限、partial overlap、per-class relation、梯度均通过。**只放行到真实遮挡 pilot**；仍无机制收益、source-OOD 或 TETA。

作者测试覆盖：无 child 精确恒等、只减不增、重叠不双扣、严格更大 child 换主、tie 给低 edge、per-parent 上限、1203 类、NaN 拒绝、未来时间拒绝、posterior detach、父类 embedding 单独扰动只影响该类、有限梯度。它们只是放行到下一次独立复核的证据，不是收益或因果结论。

此外，当前 relation 是“父类 embedding 与 child query 的 learned per-class scalar”加上 child posterior 的类别乘子，足以排除纯共享 gate，但还不能证明学习到了 `part-of` 文本关系。正式实验需要随机类关系、learned class table、class-agnostic 与只换 child posterior 的强控制。

### A8. 成本

- G0：现有 NPZ/JSON，CPU `5–10 min`。
- G1：120 个事件 × 4 个前向；按既有 full collector 的 `0.68 s/帧` 量级，纯前向约 `5.4 min`，加模型加载、I/O、四臂特征和审计预计 `10–18 min`。这是当前允许的短 pilot；若现场预计超过 20 分钟，先暂停会话。
- G1 扩展到 400 个事件：约 1,600 个前向，纯前向已约 18 分钟，加 I/O 预计 `25–40 min`，应作为需交接的后台任务，不直接由当前对话阻塞等待。
- G2 teacher 缓存若覆盖 3,000 帧 × 2 个 mask，按 0.68 s/帧约 68 分钟，属于长任务；先冻结 launch manifest、输出目录和 provenance，再启动。
- `OwnerLedger` 头本身若在已缓存 `[N,256,7,7]` 上训练，参数量小于 1M，训练时间预计短于一次全 FIT detector 采集；完整 COVTrack 20 epoch fine-tune 不应在 G1/G2 通过前启动。

## 第二候选 B：证据创新滤波，而不是更多帧求和

### B1. 问题

同一轨迹的类别错误高度持续，`sumlog` 把重复证据当独立证据；但此前 inverse-density、帧选择、linear semantic aggregation 都已经失败。新候选不应再学“哪帧更好”，而应学“这一帧相对已见轨迹，带来了多少不可由历史解释的类别证据”。

定义每个观测的中心化类别对数证据 `e_t(c)`。用一个类共享的、过去可见的递推状态预测下一帧：

```text
h_t = GRU([e_{t-1}, r_t, delta_t, valid_t], h_{t-1})
mu_t(c) = g(h_t, c)                       # 只由 t 前信息预测
innov_t(c) = e_t(c) - mu_t(c)
L_pc = e_0(c) + sum_{t>=1} w_t innov_t(c)
```

`w_t` 不是普通 confidence weight；它由 predicted innovation precision 和时间/外观变化决定，且对精确重复观测应趋于 0。重点是“只累计不能由历史解释的差异”，不是再给 sumlog 乘一个分数。

### B2. 为什么它与既有失败不同

它不试图预测某帧的 correct/wrong 标签，也不估计一个完整协方差；它优化的是下一帧类别证据的因果创新。因为 `e_t` 是类共享的，模型可以应用到 novel 类；不要求每类都有训练轨迹。必须用 C-TAO 的连续轨迹构造伪轨迹，并在 fit source 上做 source-OOF。

### B2.1 当前实现准备状态

`tmp_analysis/diagnose_conservative_memory_update_20261025.py` 已经写出但还没有完整运行；短探测发现按 pair 展开历史 `[P,8,256]` 的内存/广播开销过大，且输出目录已经预创建，当前不能把它写成已通过的信息门。必须先改成按 video/pair 分块的实现，并通过三项合成检查：
（1）无历史的严格 fallback；
（2）相同历史重复观测的 innovation 为 0；
（3）未来帧改变不会影响当前状态。

之后才可在已封存的 403 视频、341 source 的 GT-free features 上跑 CPU 信息门。现有普通 set-gate 也提示“last 与 EMA 各有胜负”，因此 B 的通过线不能只看平均 micro，必须是两个 lag、多个 source 都超过各自的普通控制。当前没有 B 的收益结论。

### B3. 最小实现与停止线

实现文件先放在独立离线目录，例如 `tmp_analysis/evidence_innovation_filter_20261025.py`，输入固定 full-FIT 的 `p_fused`/raw256/box/original frame index，不接 tracker。

- 单位：`source → GT trajectory → 唯一 query observation`；
- 对照：TCC、full sumlog、sumprob、EMA、last-frame、score/entropy weighting、普通 scalar logistic gate、同参数量 MLP、打乱时间顺序、反向时间；
- 必须验证：相同 observation 重复输入时 innovation 为 0；未来帧不得进入 `h_t`；
- 先用 `5 min` CPU pilot，只有跨 source、跨 lag 均超 sumlog 才扩全量；
- 若提升来自单一 source、只在 base 类、或不能通过 pseudo-novel holdout，STOP。

这个方向的论文故事是“相关证据的有效样本量”，但它与 temporal attention、state space/RTS、TRACT 类别 bank 和普通 EMA 的碰撞较强。因此只作为第二贡献，不能替代 CEO 的因果遮挡门。

## 第三候选 C：所有权约束的集合关联（延后）

现有贪心匹配的后果是顺序依赖和局部列竞争；GT oracle 的潜在头寸也主要来自碎片合并。但不能直接做 Hungarian 或“多帧 attention”。只有当 CEO 已证明局部 token 的所有权能预测类别证据时，才允许测试一个小版本：

```text
owned_embedding(o) = pool(w_self,r * Z_o,r)
s(i,k) = baseline_appearance_geometry(i,k) + delta(owned_embedding(i), owned_embedding(k))
```

在每帧用有 dustbin 的 one-to-one assignment 取代逐 detection 贪心列清零。约束：同一观测不能被两个 track 使用；child 与 parent 可同时存在，但同一区域不能同时作为两个独立身份的完整证据；所有分数保持原候选池和原阈值，不按类别映射做硬合并。

最小判定是同一 12-source smoke 的 release-state replay：必须改变后续 memo/assignment，不能在最终 JSON 上事后改 ID；以未来 1/4/8 个采样观测的 contamination 与 official AssocA 为主。与 pure Hungarian、Sinkhorn、TrackTrack-like joint assignment、no-write、random mask、area-matched ownership 全套比较。任一项只改最终标签、不能超过普通全局匹配、或只在单一 source 正，则 STOP。该候选优先级最低，不应先做。

## 方法排序与论文口径

| 优先级 | 候选 | 先验判断 | 为什么值得/不值得 | 一旦失败的止损 |
|---|---|---|---|---|
| 1 | A：因果证据所有权账本 | **GO to G0/G1 only; relation routing STOP** | 直接解释父—子共同占据同一视觉区域；有可利用的 ROI 空间入口；可用真实遮挡产生因果监督 | 真实遮挡不超面积控制、只剩 person、或类宏下降：STOP |
| 2 | B：证据创新滤波 | HOLD | 直接针对 5.8x 错误相关；但 temporal filter/EMA 先例很多，收益可能小 | 不能五折超 sumlog/pseudo-novel：STOP |
| 3 | C：所有权约束集合关联 | HOLD/LOW | 关联确有局部 oracle 头寸；但纯几何、冲突切换和全局匹配均为负/近邻密集 | 只在输出后改 ID 或单一 source：STOP |

论文最能站住的组合是：**A 的因果诊断 + A 的 explain-away 机制 + 可能的 C 作为同一所有权原则在关联侧的延伸**。B 只有在独立 source-OOF 上明确超过 sumlog 时才作为第二贡献；不建议为了“多模块”把 A/B/C 硬拼成三头怪兽。

## 不采用的思路

- 不再训练普通 geometry-ROI 历史 attention；已有 gate 证明增益主要是同 track 连续。
- 不再用 scene residual、角速度外推或静态 box 拼接作为主创新；这些是负结果。
- 不再用全类别名称映射来追求微平均；类别宏平均方向相反，而且会破坏真实子类。
- 不把父子包含关系当成类别表重打分；必须有像素/遮挡证据和类别条件。
- 不把 `child_mode`、GT 选择 memory、未来帧或 val 类别混淆作为部署输入。
- 不先改 tracker 阈值/贪心顺序来挽救分类模块。
- 不把“轻量 attention/gate”本身写成技术深度；技术论证必须落在证据所有权、可辨识性和反事实监督上。

## 现在的最小动作

1. 先完成 G0 的口径修正：单一 GT 类预测轨迹、逐类宏平均、同一父边 repeat、平票拒绝、source 聚类；保留现有结果为独立归档，不覆盖。
2. 冻结 CEO 的候选类别集合、child 采样、四个 mask 臂、source 折和失败阈值。
3. 实现 `OwnerLedger` 的合成契约测试，至少三轮、每轮两项：无 child 精确 fallback、重复 child 不重复扣减、空白/无效 token；单 child mask 与 ROI 网格映射、错误 child 不越过类别门；真实遮挡后重前向、parent box 固定、token 行 locator 与 NMS 行一致。
4. 只有合成与独立预审 exit 0，才启动 G1 的短 GPU pilot；任何预计超过 20 分钟的采集/训练先暂停当前对话并保留 manifest、进程身份、GPU UUID 和完成回执。

当前没有新方法 TETA，也没有理由把 CEO 写成已验证的创新。关系路由的 source-heldout 结果已经说明：**零参数规则本身不是可训练新方法**；它只能清理 CEO 的基线与反事实门槛。
