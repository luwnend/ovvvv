# Part–whole / containment 结果独立复核

日期：2026-10-25  
范围：只读审查下列脚本、缓存、边表与结果；不训练、不使用 GPU、不修改已有实验产物。

**结论：当前结果不能直接作为“global semantic routing”候选进入 source-heldout，更不能据此开训。**
类别 ID 映射、同视频包含边和原始逐轨迹分数均可复现，没有发现跨视频混边、GT 选边或 TAO category-ID 与 296 列索引混淆。阻断问题在指标口径和条件门实现：

1. 全局名称映射把**微平均轨迹准确率**从 0.39246 提到 0.56677，但同一分母上的**逐类宏平均准确率反而从 0.24086 降到 0.23843**。TETA 的 ClsA 是逐类、逐 video/alpha 的宏平均，不能用 +17.43 个百分点的微平均差代表 TETA 或 ClsA 提升。
2. 9,466 是“在 alpha=0 的官方分类事件中只匹配到单一 GT 类的**预测轨迹**数”，不是 val 的 GT 轨迹数（annotation 中为 5,473 条），也不是全部 324,639 条预测轨迹。还有 268 条预测轨迹同时匹配多个 GT 类，被静默排除。
3. `semantic_containment_repeat` 的 212/201/1 不能作为实现结果：判断“重复包含”时，任意一条 child 边满足 `common_frames>=2` 即可，而语义匹配可以来自另一条只共同出现 1 帧的父边。按连接实体修正后为 383 个候选、352 修复、2 破坏；原报告的 201 修复是错误口径。
4. 364 个 semantic-containment 改动中，347 个修复来自 `person` GT，来自 `person→part` 的选择效应极强；当前不是已证明的多类别机制。

可以将当前关系规则保留为 source-heldout 中的**零参数强基线**，前提是先冻结映射、改用类宏平均/官方 TETA，并修正下面列出的统计与因果边界。不能继续把它描述为已经验证的网络创新。

## 1. 输入、代码与证据绑定

主要输入：

- `results/dependence_audit_20261023/teta_events/classification_events.npz`
  - SHA-256：`fc11440c3ce750ee4918e1290e32cc5b6a20c4fde20aca1ae70f19fee6e0e996`
  - 383,115 个事件，9,734 条预测轨迹，296 个评估类，988 个视频。
- `results/dependence_audit_20261023/readout_cache.npz`
  - 324,639 条预测轨迹，1,529,908 个输出观测。
- `results/containment_parent_child_tracks_20261024_v1/edges.jsonl`
  - SHA-256：`1f280d0c5c6b4fa389057a8a1910f8579174a409dbc612867ba3a8a11f52820d`
  - 5,109 条唯一有序 parent–child 边。
- `/raid/datasets/ring_data/data/tao/annotations/validation_ours_v1.json`
  - 36,375 张图、988 个视频、112,798 个 annotation、5,473 条 `(video_id, track_id)` GT 轨迹。

被审脚本：

- `tmp_analysis/part_whole_class_prior_audit_20261024.py`
  - SHA-256：`ade368dbc5d765574203f34da173301232e53f17df083f70ddbf5f6cfeb94eb1`
- `tmp_analysis/containment_conditioned_class_gate_20261024.py`
  - SHA-256：`788df488d016a13a6001dd4baca4acd596acad6fd6b214c6adab49e197979ee5`
- `tmp_analysis/containment_parent_child_tracks_20261024.py`
  - SHA-256：`a3cd94c57fe7646f1bacb952b17257021e8923aa8fae7fa2bf3b42b2331bee54`

结果：

- `results/part_whole_class_prior_audit_20261024_v1/result.json`
  - SHA-256：`96ce665814bdb7b3e434f049ff90b320443128ea91fc7f4250801de50684adcc`
- `results/containment_conditioned_class_gate_20261024_v1/result.json`
  - SHA-256：`4c318f5120b5b1f05fddb7da608da7512c56c870920159d4146d32f2748a2d7f`

## 2. 检查方法和轮次

**第一轮：代码静态核验。** 追溯 `build_teta_classification_events.py`，确认 `gt_class_index` 是评估类列号，`category_ids[gt_class_index]` 才得到 TAO category ID；检查边生成器只用 `det_bboxes`、`det_labels`、`track_ids` 建立包含关系，GT 在边建立后才用于打分；检查两个门控脚本的分母、映射方向、计数和重复门布尔表达式。

**第二轮：现有缓存独立重算。** 不导入被审脚本，直接读取 NPZ/JSON，并用独立字典和集合逻辑重算：

- GT 类歧义；
- base/global/semantic-containment/repeat 的 changed/repaired/broken/correct；
- 微平均、类宏平均、逐类支持与 source 分布；
- parent/child 是否同视频、自环、重复边和边的轨迹时间关系。

对 20 个边数最多的视频（621 条边）重新扫描 `results/cov_tces_v3_fullval/*.npz`，用原始检测框重算 IoC>0.8 与共同帧。有效包含帧计数在 589/621 条边上与记录完全一致，其余 32 条也未见超报；未发现低报。所有 5,109 条边同视频，无自环。

## 3. 确认正确、无需改动

### 3.1 TAO category ID 与 296 列索引

`classification_events.npz` 字段：

```text
gt_class_index: int32, [0, 295]
category_ids:   int64, 296 个 TAO category ID, [1, 1202]
```

296 个 `(category_ids, class_names)` 对全部与 annotation 的 `(id, name)` 一致。`pred_category_id` 已在 1–1202 的 TAO ID 空间；它没有再被当作 296 列索引。两个门控脚本的

```python
category_ids[gt_class_index]
```

使用正确。名称映射使用 annotation 的 `name→id`，方向是 child label → whole label，也正确。1203 个类别名唯一，没有同名冲突。

### 3.2 alpha=0 与 GT 类歧义

alpha=0 下有 64,627 个事件、9,734 条预测轨迹。使用全部 10 个 alpha 仍为相同 9,734 条轨迹，单一 GT 类轨迹仍是 9,466 条；因此 alpha=0 没有丢失轨迹或改变可判定样本。

9,734 条事件轨迹中：

- 9,466 条只匹配一个 GT 类；
- 260 条匹配两个 GT 类；
- 8 条匹配三个 GT 类。

268 条歧义轨迹全部只属于一个视频，没有跨视频混类。原脚本排除歧义轨迹的双集构建逻辑在实现上一致。它没有用“第一条 GT 类”强行填值，这一点正确。

### 3.3 `pred_category_id` 和门控方向

`pred_category_id[0]` 是发布 TCC 读取，所有值都在 296 个 TAO category ID 内。全局映射方向与报告一致，原始 `changed=1789/repaired=1671/broken=21/correct=5365` 已逐条独立复现。

semantic-containment 原值 `364/348/1/4062` 也已逐条复现。它不是由于错误类别索引造成的。

### 3.4 包含图的视频边界

- 5,109 条边全部满足 parent 与 child 属于同一视频。
- 无 `parent_gid == child_gid`。
- 无重复 `(parent_gid, child_gid)`。
- 每条边 `common_frames>=1`，最小 1、最大 33。
- 5,109 条边涉及 2,574 个 parent、3,504 个 child。

边生成器在同一个 NPZ 帧内枚举 parent/child，没有读取其他帧来断言当帧包含，因此不存在时间断裂式“未来相干边”。

### 3.5 几何坐标与退化框

`det_bboxes[:, :4]` 已是 xyxy，直接计算交集、面积和 child 中心，坐标尺度一致。child/parent 面积比限定在 `[0.01, 1]`；IoC 是 child 区域被 parent 覆盖的比例，并要求 child 中心落在 parent 内。

独立抽样的 3,311 个同帧候选没有零面积或负面积框。当前 `max(...,1e-12)` 的写法在一般退化框下会返回 0 或很大值，不会像纯 `0/0` 那样得到 NaN；但今后重建边时仍应在进入 IoC 前显式拒绝非正面积框。

## 4. 确认错误与必须修正的口径

### E1. 全局映射的主要数字是微平均；类宏平均反向

固定 9,466 条单一 GT 类预测轨迹重算：

| readout | 正确数 | 微平均准确率 | 291 个有 GT 类的宏平均 | 296 个评估类含缺失零项的宏平均 |
|---|---:|---:|---:|---:|
| Base TCC | 3,715 | 0.39246 | 0.24086 | 0.23679 |
| Global name map | 5,365 | 0.56677 | **0.23843** | **0.23440** |

因此 `0.3925→0.5668` 只能称为“这些已匹配预测轨迹上的微平均 top-1 准确率”。它不能作为 TETA 或 ClsA 提升证据。TETA 的 ClsA 在同一类别内对 video/alpha 聚合，再对类别做等权平均；大量 `person` 修复不一定产生等幅的类宏收益。

映射还完全摧毁了几个小支持子类：

| GT 类 | 支持轨迹 | Base 正确 | 映射后正确 |
|---|---:|---:|---:|
| `cab_(taxi)` | 38 | 13 | 0 |
| `jacket` | 14 | 3 | 0 |
| `pickup_truck` | 42 | 2 | 0 |
| `shirt` | 15 | 2 | 0 |
| `skirt` | 2 | 1 | 0 |

所以“global map 明显更好”并不成立。必须同时报告类宏 ClsA、child recall、逐类支持，以及缺失 GT 类的处理规则。

### E2. 9,466 不是 GT 轨迹分母

该集合的构造是：

```text
官方 TETA classification events
→ alpha=0 下出现过的预测 global_track_id
→ 只保留恰好匹配一个 GT 类的预测轨迹
```

因此：

- val annotation 有 5,473 条 GT 轨迹；
- tracker 发布 324,639 条预测轨迹；
- 官方分类事件覆盖 9,734 条预测轨迹；
- 9,466 条是可判定单一 GT 类的预测轨迹。

这三者不能互换。事件缓存不保留 `gt_track_id`，所以仅凭该缓存无法给出“覆盖了多少条 GT 轨迹”、未匹配 GT 的分布，或按 GT 轨迹去重的分母。报告中的 “9,466 条 track” 应改为“9,466 条单一 GT 类分类事件预测轨迹（268 条歧义匹配排除）”。

如果论文实验单位是 source → video → GT trajectory，必须另建包含 `gt_track_id` 的评测表；不能把预测轨迹当作 GT 轨迹。候选/边数也不是独立样本数。

### E3. `semantic_containment_repeat` 布尔条件连接了错误实体

原逻辑：

```python
longcont[g] = target
if
    any(edge.child_gid == g and edge.common_frames >= 2)
    and any(parents[p] == target for p in parents_of_g)
```

两个 `any` 可以落在不同父轨迹：一条与 `g` 的任意父子边共同出现两帧，而类别匹配来自另一条只共同出现一帧的父边。独立重算：

```text
原报告：212 changed, 201 repaired, 1 broken
连接实体修正后：383 changed, 352 repaired, 2 broken
```

因此原 212 行不仅偏低，也把“重复包含语义连接”定义错了。修正版应使用同一 `(parent_gid, child_gid)` 边同时满足语义目标与 `common_frames>=2`。

### E4. 多数父类路由存在平票依赖插入顺序

`containment` 臂使用：

```python
Counter(parent_preds).most_common(1)[0][0]
```

在语义 child 集合内，有 93 个 child 的父类支持出现平票；`Counter.most_common` 在这种情况下依赖首次插入顺序，而不是确定性冻结规则。当前 `semantic_containment` 不需要多数路由，所以它的 364 不受这 93 个平票影响；但如果之后实现 learned/majority parent routing，必须显式定义平票策略（拒绝路由、按固定 category ID，或按置信度），不能依赖字典遍历。

### E5. `child_mode` 与 `child_pred` 不是同一量

边表字段：

- `child_pred`：child 轨迹的发布终类；
- `child_mode`：该 child 在包含帧上的逐帧 detector label 众数；
- `explains_parent_wrong`：使用 `child_mode == parent_pred`，不是 child 轨迹终类。

5,109 条边里有 2,083 条两者不同，涉及 1,435 个 parent。由于 9,466 条轨迹的发布读出基本是单类标签，这个字段名很容易被误读成“child 的最终类别”。

按三种定义独立统计：

| 解释定义 | 唯一 parent 数 | 占 1,519 个错误 parent |
|---|---:|---:|
| 原 `child_mode == parent_pred` | 280 | 18.43% |
| `child_pred == parent_pred` | 316 | 20.80% |
| 原 edge 级 `explains_parent_wrong` 去重 | 280 | 18.43% |

原报告采用 280/18.43%，实现与字段定义一致；但必须在论文和后续脚本中明确它是“逐帧 detector 标签众数”，不能写成“子轨迹终类等于父误判类”。

### E6. “重复包含”没有证明因果稳定性

`common_frames` 是整段轨迹的历史总共同帧数，不是当前时刻之前的前缀计数。用它把一个关系称为 “temporally stable”，在回顾性诊断里可以，但部署时若直接使用整段轨迹边，会用未来帧决定当前分类。

此外，当前边只要求某一帧包含，未要求同一 child 在轨迹生命周期内持续属于 parent，也未要求 parent/child 轨迹身份正确。它不能作为“child 遮挡导致 parent 误分类”的因果修复率。即使 common_frames>=2，也可能只是一个错误 child 在两次观测中因框重叠而成立。

### E7. semantic-containment 的类别与 source 集中度

修正前的 364 个 semantic-containment 候选中：

- 348 个修复；
- 1 个破坏；
- 其中 347 个修复来自 `person` GT；
- 1 个修复来自 `car_(automobile)` GT；
- 被破坏的是 1 个 `cab_(taxi)` GT。

它覆盖 7 个 source（HACS 166、YFCC100M 73、LaSOT 56、AVA 50、Charades 12、BDD 4、ArgoVerse 3），但修复数主要来自 3 个 source，且 `person` 占 99.7% 的修复。单一 GT 类主导意味着目前不能写成“普适的 part–whole 关系机制”。

同时，这个 364 只是 9,466 条可判定预测轨迹的 3.85%、错误轨迹的一小部分；比全局映射 1,789 次改动小得多。它是高精度、低覆盖的规则，不是全局映射的等价替代。

### E8. `global_name_prior` 的结果字段不可直接作为紧凑映射读取

`containment_conditioned_class_gate` 把

```json
"global_name_prior": {"<global_track_id>": "<category_id>", ...}
```

写入 9,466 个 key 的字典，导致结果 JSON 达 196 KB，且看起来像类别 ID 表。这不是数值错误；`score()` 确实用同一 index 表计算了 1,789/1,671/21。但来源可读性差，后续应只存 `mapping`、聚合计数和逐类/逐 source 表，并给原始 per-track 预测另存一个明确命名的表。

## 5. 未验证风险与不能下结论的部分

1. **映射是否预先冻结无法从结果自证。** 映射由类别名词手工定义，但 `part_whole_class_prior_audit` 同时声称“seen val confusion”。如果在看到 val 混淆后选择 11 个映射，则它是后验选择，不是泛化方法。仓库时间戳和脚本不能证明选择顺序；应保留在 val 之前的设计记录或在 heldout 上重新冻结。
2. **`semantic` 表中没有 `hat`、`helmet`、`backpack` 等部件。** 当前规则漏掉大量可能关系；不能把 364 当作包含信息的上界，也不能因漏掉关系而低估网络潜力。但同样不能用扩大词表后再看 val 的方式调参。
3. **没有官方完整 TETA。** 本轮只验证了分类事件重放和轨迹微/宏统计；没有生成只改 `category_id` 的新 prediction JSON，也没有跑完整官方评测。分类事件重放的 Loc/Assoc 不变只适用于固定框/ID 的条件；任何训练或新几何都不能复用旧 Loc/Assoc。
4. **没有 source-heldout。** 当前七个 source 都参与了同一 val 汇总；同一 source 的 video 相关性也没有按 source→video 做 cluster bootstrap。source 级数字只是描述性切片。
5. **没有真实图像遮挡。** 包含关系来自检测框重叠，不证明 child 像素遮挡 parent 或 parent 像素证据被 child 占用。真正的因果门仍需像素级 child mask、面积/位置匹配控制遮罩和错误父框控制遮罩。
6. **边只对可判定的 9,466 条预测轨迹中的 parent/child 建图。** 268 条歧义轨迹、未匹配轨迹和全部未出现在 classification events 中的预测轨迹没有进入边表。部署时不能假设这些样本都没有包含关系。
7. **尚未检查 `child_mode` 众数平票。** 边生成器使用 `Counter(...).most_common(1)` 取逐帧 label 众数；平票策略同样依赖插入顺序。由于这不是 semantic 门的唯一依据，未单独重算其影响，但应在重建边时为平票记录状态或拒绝。

## 6. 是否可进入 source-heldout

**不能以当前 global mapping 的 `0.5668` 或 conditional 的 `0.4291` 作为已验证候选进入 heldout。** 当前最可靠的可用结论是：

- TAO 类别 ID 与 296 列评估索引没有混淆；
- 原始边表没有跨视频混边或未来帧造边；
- 9,466 是预测轨迹子集，不是 GT 轨迹总数；
- 全局映射的微平均提升伴随类宏平均下降，不能直接支撑 TETA/ClsA 结论；
- repeat 门有已确认的实体连接 bug；
- 语义包含门是高精度、低覆盖且高度集中于 `person` 的探索规则。

可以进入 heldout 的形态是：把**零参数名称映射**和**修正后的 semantic containment**作为强基线，同时在 heldout 视频/source 上预先冻结以下内容：

- `person`/`car`/`truck` 的整体—部件词表；
- child→whole 映射；
- `IoC>0.8`、面积比 `[0.01,1]`、`common_frames>=2` 的具体定义；
- 平票与无 GT 类别的处理；
- 是否只用当前与过去帧构造边。

heldout 需同时报告：官方完整 TETA/ClsA（Base 与 Novel 分开）、逐类 precision/recall、source/video 级聚类区间、child recall、随机同基数映射、反向映射、尺寸/位置门限和纯类别频率先验。至少要证明 semantic containment 在 source-heldout 上 repair 明显多于 break，且不能只由 `person` 一个类驱动。

只有这一步通过，才值得进入“ROI 局部证据 + 类别条件 evidence ownership”的网络部分。当前 ROI 信息门已经表明普通局部注意力不足以支撑这条路线；因此网络设计必须超过上述零参数关系基线，而不是只超过一个弱阈值。

## 7. 建议的下一步（按依赖顺序）

1. 先写一个独立的 `track_score_audit`，输出“预测轨迹 / 单一 GT 类预测轨迹 / 歧义预测轨迹 / annotation GT 轨迹”四个明确口径，禁止再使用含糊的 `tracks` 字段。
2. 在冻结映射前按 source/video hash 划分 development 与 heldout；development 只用于设计，不把 val 全体的混淆结果反馈到词表。
3. 修正 `semantic_containment_repeat` 的同一父边条件，并重跑为归档新目录；本次审计不得覆盖旧 JSON。
4. 对语义门输出逐类宏平均、逐 source 和逐 child recall；对 93 个多数平票明确拒绝或冻结规则。
5. 生成只改 `category_id` 的新 prediction JSON，保持 bbox、score、track ID、顺序和匹配不变；跑官方完整 TETA，分开 Base/Novel、ClsA/LocA/AssocA。
6. 只有 heldout 关系规则先通过，才做真实图像 child mask 与面积/位置匹配遮挡；否则把 part–whole 网络方向记为 STOP，回到与类别关系无关的分类/时序模块。

