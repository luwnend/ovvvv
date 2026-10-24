# EvidenceOwnershipLedger v2 独立复审

日期：2026-10-25
对象：`/home/chenhc/ov/OVMOT_RESEARCH/tmp_analysis/evidence_ownership_head.py`
冻结快照 SHA-256：`c4684e660842e00299ac6943e8af1419575a27753b12c1f96b8e4b2ad7c69c69`（14,981 bytes）
审计模式：只读主文件；独立 CPU 探针；未启动 GPU、未训练、未修改冻结 baseline。

## 判定

**FAIL。不得进入独立遮挡特征采集，亦不得训练或接回 COVTrack。**

上一轮 eab492b8 的多数实现错误已经修复：owner 选择、per-parent 上限、1203 类、异常数值、explain-away 反向和 posterior detach 均有实质改善。但本版本仍有三项与是否可定义独立遮挡采集门槛直接相关的失败，其中“类别条件退化为共享空间标量”使方法创新主张尚未成立。

该判定属于**实现与接口未完成**，尚不能把类别方向整体定义为理论失败。

## 审计快照与版本漂移

审计期间主文件曾出现 `c4f5…`、`4cc5…`、`0aab…` 和 `c4684e…` 多个哈希。最后一次复核时，主文件与冻结快照逐字节一致（diff 为空）。本报告只对这 **一个** 哈希有效；若文件再次变化，必须重新审计，不能沿用本结论。

独立探针及原始结果：

- 探针：`tmp_analysis/owner_ledger_v2_audit_20261025.py`（SHA-256 `5bcac6a6b3676cd116344b7a7109c862dc75b851e93d36417571fa5f81a5a716`）
- 证据：`tmp_analysis/owner_ledger_v2_audit_evidence_20261025.json`
- JSON SHA-256：`f39dca5bd82fbabbff539999567079ddacc461ff5738cfa6c5e36ed22c88db52`
- 环境：PyTorch `1.10.0+cu113`，CPU float32。探针在运行前校验被测源码哈希，避免在本文件继续被编辑时把结果转给新版本。

每轮 7 项，共三轮；每轮均至少包含两项主动反例，而不是仅依赖自带 `synthetic_contracts()`。自带测试本身在该哈希上通过，但未覆盖下列三个失败项。

## 已经成立的契约

| 上一轮 FAIL / 本轮要求 | 本轮结果 | 独立证据 |
|---|---|---|
| per-parent 而非全局 `max_children` | **PASS** | 2 个父各 1 个 child 被接受；同一个父 2 个 child 在 `max_children=1` 时被拒绝 |
| `[0,1,0]` 顺序不得让后 edge 覆盖最高分 | **PASS** | parent 序列 `[0,1,0]`、分数 `[1,9,5]` 时，parent 0 由 edge 2 拥有，parent 1 由 edge 1 拥有；没有 edge 0 串入 |
| 选最高合法分数 | **PASS** | 完全重叠的两个 child 分别用 `1 vs 5` 和 `5 vs 1` 测，均只扣较高者对应的类 |
| 严格 tie 给 lower edge | **PASS** | 分数相同但 child 类别不同；正序由 edge 0 扣除，反转 child payload 后仍由 edge 0 扣除 |
| 完全/部分重叠只扣一次 | **PASS** | 部分重叠的共享格只有一个 owner；把候选分数反转后 owner 随之反转 |
| 默认 1203 类与完整词表 | **PASS** | 实际调用默认构造的 `parent_tokens[2,256,2,2]`、`class_embedding[1203,256]`，输出为 `[2,1203]`，有限且非正 |
| NaN / Inf 六个张量输入 fail-closed | **PASS** | parent、child、overlap、posterior、embedding、geometry 各自 NaN 与 Inf 共 12 个输入探针全部抛出异常 |
| 越界、负索引和浮点 edge | **PASS** | parent / child 的 `-1,1,99` 均拒绝，浮点 edge 也拒绝；非空 edge 的空 child token 亦拒绝 |
| explain-away 方向 | **PASS** | 同一父类 child support `q=0,.25,.5,1` 时，父类扣除量为 `0` 和递增的负值；没有使用 `1-q` |
| posterior detach | **PASS** | 对输入 posterior 设 `requires_grad` 并反传，`grad is None`；但类别轴仍会随 posterior 值变化 |
| 父正证据空间限制 | **PASS** | 三个格中只有中间格有正父证据，仅该格被扣；其余格精确为零。全零父 token 精确恒等 |
| 梯度有限且 child_query 不是死路 | **PASS** | 关系末端权重非零初始化下，所有参数梯度有限，`child_query` 首层与末层最大绝对梯度均非零；parent/child/geometry 输入梯度也有限 |
| 空 edge 精确恒等 | **PASS** | 输入四维合法空 edge 时逐元素全零 |
| 非 tie edge 的输入排序不变 | **PASS** | 三 edge 连同 child payload 一起重排后，输出在 float32 容差内一致 |

这里的“父正证据”只是归一化 parent token 对类别文本 embedding 的逐格正内积；它通过与空间限制契约，但**不是**经分类器校准的 evidence mass。

## 三轮中仍失败的项目

| 失败项 | 结果 | 反例与根因 |
|---|---|---|
| 空输入/形状 fail-closed | **FAIL（实现）** | 空 edge 时 geometry 用 `[0,4]` 而非约定的 `[0,5]`，模块仍返回 `[1,3]` 全零。空 edge 在 shape 检查之前提前返回，`child_geometry` 只检查有限性，没有检查尾维和 child token 行数 |
| 未来信息 | **FAIL（接口）** | signature 中只有 parent/child tokens、edge、overlap、posterior、embedding、geometry；没有 time/frame、track 身份或观测状态字段。额外传 `future_tokens` 会 TypeError，但真实调用方仍可把未来 child 放进合法 `child_tokens`。模块无法自行证明严格过去 |
| 类条件不能退化为 class-agnostic spatial gate | **FAIL（核心机制）** | `child_query` 不接收 posterior，也没有 per-class/cross-class relation。`relation` 的输入是“每格的父类投影 + 一个共享 child relation”，输出恰好是 `[H,W]` 的**每格一个标量**。主动路径为 `q_child[c] × sigmoid(relation[h,w]) × ownership[h,w] × parent_positive[c,h,w]`。均匀 support 加同 embedding 时三个类输出完全相同。可学习部分是共享空间 relation、child 几何/occupancy，以及 posterior 给每类乘的系数；不是真正的类别条件关系 |

此外，overlap 对合法区间外数值采用 `clamp(0,1)`：NaN 会拒绝，但 `-1` 与 `2` 被静默夹取。该行为本身可以接受为设计选择，但与“所有非法输入 fail-closed”的严格版本不同，必须明确成契约。

## 与上轮相比的实际修复

这些不是“测试写法变化”：

1. 先解析 `n_parent`，再使用 per-parent `bincount`；不再全局计 child 数。
2. owner 比较当前格已选最高原始 score，严格 `>` 使同分保留更低 edge；不再按位置错位覆盖。
3. 六个浮点输入做有限性检查，edge 检查 int64 与上下界。
4. 默认 1203 类可以在实际 `[1203,256]` 词表上运行。
5. `child_posterior.detach()` 放在计算前；类别权重为 `q_child[c]`，方向与 child 支持父类时扣更多一致。
6. 自带上限测试改为构造 `max_children=1`，不再用默认 `max_children=4` 假测。

仍属于实现缺口的三项，修复方向分别很明确：空 edge 前的完整 shape 校验；显式 frame/track observation 与过去时间约束；带类对参数的 relation 或每类 child evidence 投影。三者都应在重新独立审计前完成。

## 进入“独立遮挡特征采集”的前置条件

**当前不允许。** 不是训练门槛，而是采集契约尚不能证明“目标 child 与正确时间关系 + 明确可解释的类条件”三件事。

重新审计通过的最低要求：

1. 空 edge 的输入 shape 和几何值全部严格检查；任何非法形状、非有限值、越界或不一致都拒绝。
2. 调用接口显式接收 frame/time 和 child 的观测状态，拒绝 `child_time > parent_time` 及未来帧；测试必须用真实未来 child 做反例。
3. `relation` 至少能输出 `[H,W,C]`（父类条件）或定义 `[C_parent,C_child]` 的类对关系。必须用 posterior 与父类 evidence 形成主动路径，并通过零初始化、常数条件、打乱类关系测试证明不是全类共享标量。
4. 父正证据与零父证据保持精确恒等；对均匀 posterior、恒等 embedding 和随机 child relation 做控制，确认没有隐藏的共享 spatial gate 获得全部扣分。
5. 把错误 child、同一 child 在不同类别支持、完全/部分/零重叠、`[0,1,0]` 分数、严格 tie、1203 类实际 shape 全部重跑，并保持真实遮挡前的独立 source/video 划分。

通过上述实现审计**只代表可以开始独立遮挡特征采集**。采集仍需新目录、固定哈希、完整 receipt，冻结父框、检测器、NMS、轨迹 ID 与类别表；不得用 GT 选择 owner、阈值或关系。采集不证明机制增益，之后必须先通过真实 child/错误 child/面积匹配 mask/非 child mask 的因果门槛，才可训练或进行官方 TETA 评测。

## 最终结论

- **owner ledger 的实现大方向已修复，但版本仍为 FAIL。**
- 最有价值的修复证据是 `[0,1,0]` 与 tie 反例、per-parent 上限、1203 类真实 shape、反向 explain-away、posterior detach 和有限梯度。
- 阻断采集的根本原因不是普通测试覆盖不足，而是**类别条件核心仍退化成共享空间标量**；即使采集了 child ROI，也无法据此验证独立创新。
- 当前只允许继续修代码与进行 CPU 合成反例；不允许启动真实 GPU 遮挡采集、训练或修改类别输出。
