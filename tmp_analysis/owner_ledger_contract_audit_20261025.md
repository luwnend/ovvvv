# EvidenceOwnershipLedger 合成契约独立审计

日期：2026-10-25  
审计对象：`tmp_analysis/evidence_ownership_head.py`  
源码 SHA-256：`eab492b8800ea9c9c78d392812d0bde2e9809621742e3173f332e9bae12c2c38`（11,725 bytes）

## 判定

**FAIL。当前版本不得进入真实 GPU 遮罩 pilot，也不得接入训练或作为方法实验的强基线。**

审计期间同一路径被另一协作者连续修改。本报告冻结上述哈希；后续文件若改变，必须重新审计，不能把本报告的结果转给新版本。现场最终哈希仍与该快照一致。

本报告只使用 CPU 合成输入和独立探针，没有修改主脚本，没有启动 GPU 或训练。

## 已验证成立的部分

| 契约 | 结果 | 证据 |
|---|---|---|
| 无 edge 时精确恒等 | PASS | 形状 `[3,2]`，逐元素全零 |
| 父 token 全零或全部与词表正交 | PASS | correction 逐元素精确为零 |
| 输出只扣不减 | PASS | 100 次随机有限输入均为非正且有限 |
| 单个完全重叠 child 与两个完全相同的 child | PASS | 两者输出逐元素相同，没有双重扣除 |
| 两个部分重叠 child 的共享格 | PASS | 直接 assignment 探针只产生 owner 0 或 1，未同时扣两次 |
| 父类别正证据的空间位置 | PASS | 只在一个格点有父正证据时，只扣除该格；`0.5` 与解析值一致 |
| 类别 posterior 无跨类串扰 | PASS | 改变类别 1 的 child support 不改变类别 0 的 correction |
| 初始化的梯度路径 | PASS，但有条件 | `child_mask`、`child_query`、child token 都有有限非零梯度 |
| 自带 `synthetic_contracts()` | PASS | 当前快照执行成功并返回七项 true |

自带测试通过并不证明完整契约。它没有覆盖以下失败项。

## 三轮独立检查与反例

测试脚本还从主脚本独立复制了 assignment 循环，避免用端到端 sigmoid 输出代替 owner 选择规则。所有数值来自 CPU float32。

### 第一轮：身份、维度、边界、极限

| 检查 | 结果 |
|---|---|
| 空 edge 精确零 | PASS |
| 零/负父证据精确零 | PASS |
| 100 次随机输入有限且非正 | PASS，0 个失败 seed |
| `max_children=1` 时两条 edge | **FAIL：被接受并返回结果** |
| `max_children=0` 时一条 edge | **FAIL：被接受并返回结果** |
| NaN overlap | **FAIL：不拒绝，输出含 NaN** |
| NaN geometry | **FAIL：不拒绝，输出含 NaN** |
| Inf posterior | **FAIL：不拒绝**；当前输出碰巧有限，不能作为契约保证 |
| 默认 `class_width=256`、1203 类 | **FAIL：`ValueError: class width exceeds fixed relation width`** |
| 空 `child_tokens` 但非空 edge | **FAIL：卷积前的形状/空输入契约不足**；同长度的正常空 edge 路径才恒等 |
| parent / child 越界索引 | 部分失败：越界会泄漏为底层 `IndexError`，负 child 索引会被 Python 负索引接受 |
| 错误 geometry 维度 | PASS：shape 错误会抛异常 |
| geometry 与 overlap 的几何一致性 | **FAIL：完全冲突的 geometry 也会被接受** |

`max_children` 只在构造函数中保存，forward 从未读取。它不是硬上限，也不定义“每父”还是“全局”上限。

### 第二轮：assignment 与重叠

直接 assignment 反例使用固定 owner 分数，已排除 sigmoid 变换带来的混淆。

| 检查 | 结果 |
|---|---|
| 相同分数 tie 给 lower edge | **FAIL：得到后一个 edge** |
| 前一个 child 分数更高 | **FAIL：后一个仍可成为 owner** |
| 后一个 child 分数更高 | PASS：后一个成为 owner |
| 两个部分重叠、共享中心格 | PASS：共享格只有一个 owner |
| 父 0 的 edge 序列 `[0,1,0]`，分数 `[1,9,5]` | **FAIL：最终 owner 为 2，最高分 edge 1 被错误覆盖** |

根因在 `forward` 的重选逻辑：比较 `score[...] > current_score` 的当前格，而不是比较该格已有 owner 的原始分数；后续 edge 会用自己的位置索引去覆盖既有 owner。即使单父通常把 edge 连续排列，这个循环也没有契约保证这种输入顺序。低层 API 的注释称“larger predicted occupancy”，实际比较的是 `predicted_occupancy * overlap`。注释、变量和代码语义应统一为选择分数。

完全重叠 child 的“只扣一次”成立，部分重叠的“每格最多一个 owner”也成立；但“每个格由最高合法分数 owner 所得”以及“严格 tie 稳定给 lower edge”不成立。

### 第三轮：类别条件、父证据、梯度

| 检查 | 结果 |
|---|---|
| 无 child posterior 时父类扣除 | `2.0` |
| child 对同一父类 posterior 为 `1.0` 时扣除 | `0.0` |
| 期望语义：child 支持父类时 explain-away 增加 | **FAIL，方向相反** |
| posterior 标签跨类串扰 | PASS，类别轴是对角的 |
| posterior 是否进入 `child_query` | 否 |
| posterior 是否仍会接收 gradient | **YES** |
| 单一正父证据格 | PASS，空间作用范围正确 |
| 初始化时 child mask/query/token 梯度 | PASS，均有限非零 |
| 将 relation 末端零初始化 | **FAIL：child query 梯度完全消失** |
| 全类别均匀 posterior 时的类别门 | **FAIL：只剩一个共享标量形状，不能区分真正类条件关系** |

类别公式需要先统一。设计文档的公式写 `a = h(q_j, ...) * (1 - q_j(c))`；代码实现也是 `1 - q_j(c)`。但“explain-away”的遮挡 teacher 定义是：child 支持类别 `c` 时，父类别 `c` 的正证据更多归 child 解释，因此应扣更多。这两个目标不能同时用同一个 `1 - q_j(c)` 表达，除非训练 teacher 和方向另行定义。

当前 `child_query` 只接收 child token/geometry，没有接收 posterior；类别信息只经过 `available[c]` 与全类共享的空间 relation 分数。它仍有类别轴，但并不是“每个父类别—child 类别关系都由一个网络表示”。在声称超过 scalar gate 前，必须把类别关系单独参数化，并做 class-agnostic、learned-class-table 和随机 relation 的消融。

`child_posterior` 未 detach 时能回写梯度。即使上游不优化该输入，也会扩大计算图和训练接口的假设；必须明确它是观测常量，或显式训练，不能兼有两种身份。

父证据使用的是归一化 parent token 与各父类文本 embedding 的内积，再逐格 `clamp_min(0)`。它是一份“父类别投影正分”，不是经过完整 ROI 分类器的校准 evidence mass。方向与空间局限通过；语义名称和与教师的一致性仍需校准。

## 与 class-agnostic 基线的机制边界

不能凭模块名称或“类别条件有改变”就进入 pilot。现有实现最容易学到的路径是：

```text
共享空间 child occupancy × 全类共享 spatial relation × 每类 (1-q_child[c]) × 父类正投影
```

当父 evidence 大致均匀，或 `child_query` 学不到有效类关系时，这条路径可以退化为一个乘在父类分数图上的共享几何/空间 gate。因此在同一冻结输入上必须至少比较：

| arm | 作用 |
|---|---|
| `parent_positive` 无 child | 恒等基线 |
| scalar confidence/occupancy gate | 类无关门 |
| class-agnostic spatial ownership | 无类条件的空间归因 |
| learned class table，不用文本关系 | 学习类对的强控制 |
| random / permuted class relation | 检查文本关系是否真的提供信息 |
| additive child pooling / MHA | 区分 explain-away 与普通父子注意力 |
| 正确 child 与错 child、真实 mask 与面积匹配 mask、过去与未来 | 因果与泄漏控制 |

只有独立 source 的 teacher top-1 与 heldout `ClsA` 同时超过上述控制和零参数 containment，才能把模块写成有效创新。低层测试通过最多说明可以开始 pilot，不能证明机制有增益。

## 实现失误还是理论失败

下面这些问题属于明确的实现错误，修复后可以重新审计：

1. assignment 比较用了位置错位的当前分数，且 tie 未固定 lower edge；
2. `max_children` 未执行；
3. 1203 类与默认 `class_width` 不匹配；
4. NaN/Inf、越界索引、空 tensor 和几何一致性没有 fail-closed；
5. relation 末端初始化依赖“非零小权重”来保留 child query 梯度；
6. posterior 没有边界 detach，导致意外的梯度回写。

下面问题属于理论/接口口径，必须先决定再改代码：

1. `1 - q_child[c]` 与遮挡 explain-away teacher 的方向是否一致；不一致会让训练目标互相抵消；
2. `parent_positive` 是校准 evidence mass、父类别分数图，还是仅作 spatial support；三者的损失和归一化不同；
3. 类别关系是学习网络、固定文本关系的函数，还是只通过 posterior 与父投影形成；决定后者的“类条件”容易被 class-agnostic 控制击败；
4. `max_children` 是每个父轨迹、每个帧观测还是每个 batch 的上限；以及超过上限时是拒绝还是确定性保留 top-K；
5. 真实 GPU pilot 的前置条件。当前只能运行遮挡采集和特征缓存审计，不能训练该头，也不能用其输出改类别。

## 是否允许进入真实 GPU 遮罩 pilot

**不允许基于当前头启动“训练/评价 evidence ownership”的 GPU pilot。**

可以对独立的全/child/area/non-child ROI 前向做只读采集，前提是：

- 父框、检测器、NMS、轨迹 ID 和类别表保持冻结；
- 使用新的输出目录和固定哈希；
- 采集 receipt 完整写出；
- 不为了让该头通过而调整数据集或结果；
- 不在采集阶段用 GT 选 owner、阈值或 relation。

采集完成只说明 teacher 数据可用，不能替代本报告的 FAIL。修复头以后必须重新执行三轮、每轮至少两项反例，并重新核查 1203 类、真实 source-heldout 和官方 TETA `ClsA`，再决定是否接回 COVTrack。

本审计没有新增训练产物，也没有修改任何冻结 baseline 或结果。
