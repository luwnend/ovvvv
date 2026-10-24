# EvidenceOwnershipLedger v5 独立审计（417f734e）

日期：2026-10-25  
主文件：`tmp_analysis/evidence_ownership_head.py`  
冻结副本：`/raid/datasets/chc_data/partwhole_review_20261025_v1/evidence_ownership_head_417f734e.py`  
源码 SHA-256：`417f734e63228637a153657c4106801e2f16a641209154b21ec7ea7ef4ce6ae3`  
大小：21,054 bytes。审计期间主文件先与冻结副本 SHA 一致；结束后被另一协作者继续编辑，当前新 SHA 为 `f5228ec95dfbb7351c7420c2ea1e5149c737b55bb7c56184f82289de4fda40c4`。本报告只对冻结副本 `417f734e` 负责，主文件在审计期间未被本审计修改。新 SHA 尚未完成独立重审。  
环境：Python 3.8.20、PyTorch 1.10.0+cu113，CPU float32；全程 `CUDA_VISIBLE_DEVICES=""`，未启动 GPU、训练、数据采集或遮挡前向。

## 判定

**FAIL。唯一变更 `GroupNorm(8, hidden) → InstanceNorm2d(hidden, affine=True)` 没有修复目标失败，并引入了新的合法输入回归。**

这次修复针对的是单行 `train()`，但 `InstanceNorm2d` 归一化的是“每个样本的每个通道在空间轴上”的统计量。当合法 ROI 是 `[E, C, 1, 1]` 时，每个样本的每个通道只有一个空间元素，训练模式仍抛：

```text
ValueError: Expected more than 1 spatial element when training,
got input size torch.Size([1, 8, 1, 1])
```

实际模块默认 `hidden=128`，1203 类默认模型在 `1×1` 上同样抛出 `torch.Size([1, 128, 1, 1])`。因此这不是只在非默认小配置出现的边角问题。

更重要的是，v4 的 `GroupNorm` 在“合法 batch 2、`1×1`”上本来可以通过，因为 `GroupNorm` 把 `N×H×W` 合在一起算统计；v5 改成每个样本独立的空间归一化后，该输入现在失败。审计的真实对照为：

| 输入（均 `train()`） | v4 `GroupNorm` | v5 `InstanceNorm2d` |
|---|---|---|
| `N=1, 1×1` | FAIL（`1 value per channel`） | FAIL（`1 spatial element`） |
| `N=2, 1×1` | **PASS** | **FAIL（回归）** |
| `N=1, 2×2` | PASS | PASS |
| `N=2, 2×2` | PASS | PASS |
| `N=1, 1×2` | PASS | PASS |

因此：**不允许基于该快照训练、接回 COVTrack、启动依赖该头训练模式的 GPU 遮挡 pilot，或将其输出用于类别改写。** 只读的 `eval()` 遮挡特征采集也不应使用这个版本：默认 `1×1` 在 `eval()` 也失败；在明确 ROI 空间形状之前，不能把采集结果当作可复现的模块入口。

若要修复，建议改为语义明确的 `LayerNorm` 类算子，或在 `child_mask` 前加入显式、可测试的 `1×1` 分支；同时保留 batch 2、`1×1` 的回归测试。任何改动都需要新 SHA 和完整三轮重审，不能把本结论或 v4 的通过项迁移给新版本。

## 证据和审计文件

- 独立 v5 探针：`tmp_analysis/owner_ledger_v5_audit_20261025.py`  
  SHA-256 `6b65182e3c746bf9c2a232cf1b5cd46d183dc9d133f2e1348407373d142fd6c4`
- v5 证据 JSON：`tmp_analysis/owner_ledger_v5_audit_evidence_20261025.json`  
  SHA-256 `d100935d134fa8e1c5c35899f9c02a03a54812f23c366de8425a31c0595fbc35`
- v4 对照快照：`/raid/datasets/chc_data/partwhole_review_20261025_v1/evidence_ownership_head_833a4bf.py`  
  SHA-256 `833a4bf6dcd7e6af2763b602f975ee9a0e7fea09e68dde9a5b7904f84305af6b`
- 将 v4 的独立探针绑定到 v5 副本的迁移复核：`/tmp/owner_ledger_v5_v4probe_audit.py`  
  SHA-256 `53f159f70204c3ba567bd807b0d1b0749f854ff4939ba513e8979227147a1966`  
  证据：`/tmp/owner_ledger_v5_v4probe_evidence_20261025.json`，SHA-256 `fc1bdf6e0bd8db6af663d50bc4f5de8932868e68a7ae8a439ae18449252e8382`

两套探针均在执行前校验冻结副本 SHA，执行后复核主文件和副本未漂移。本文不把 `/tmp` 文件当作长期结果；正文保留了关键的通过项与失败项。

本审计结束时，主文件被另一协作者加入一个 `StableSpatialNorm`（内部为 `LayerNorm`）类，但 `child_mask` 的当前分支仍保留 `InstanceNorm2d`。这个新编辑不是 `417f734e` 的一部分，也不能用本报告的失败结论替代它的审计；下一次审计必须使用新的稳定 SHA 副本。

## v5 独立探针：三轮、每轮至少两项

探针共 20 项主动反例，三个轮次都有边界或对照，不只调用作者自带的 `synthetic_contracts()`。第一轮包含输入批量和空间形状矩阵，第二轮检查输入/索引契约，第三轮检查 owner、类别条件、梯度及 batch 独立性。

| 轮次 | 检查 | 结果 |
|---|---|---|
| 1 | `train()` 单个 parent，合法 `1×1` | **FAIL**：`Expected more than 1 spatial element` |
| 1 | `train()` 两个 parent，`1×1` | **FAIL**：同样的空间统计错误 |
| 1 | `train()` 单个 parent，`2×2` | PASS：shape `[1,3]`，有限、非正 |
| 1 | `train()` batch 2，`2×2` 全 1 | PASS：shape `[2,3]`，有限、非正 |
| 1 | `eval()` 单个 parent，`1×1` | **FAIL**：当前版本在 eval 也失败 |
| 1 | v4/v5 归一化输入矩阵 | **FAIL（预期对照未满足）**：同时暴露单行 `1×1` 未修复和 batch 2 回归 |
| 2 | `python -O` 下非法空 overlap 形状 | PASS：`[0,2,3]`、`[0,3,2]`、`[1,2,2]` 均拒绝 |
| 2 | 空 edge，`1×1`，`train()` | PASS：返回精确零，不进入卷积 |
| 2 | 空/非空 `class_width` 不等于 posterior 列数 | PASS：均拒绝 |
| 2 | 时间形状、整数 dtype、非有限、未来信息 | PASS：拒绝混合 dtype、float/bool、NaN/Inf 和未来 child |
| 2 | 六类输入张量的 NaN/Inf | PASS：parent/child token、overlap、posterior、embedding、geometry 均 fail-closed |
| 2 | per-parent 与全局 `max_edges` 上限 | PASS：上限分离且生效 |
| 3 | `[0,1,0]` 局部最大 owner、严格 tie 给低 edge | PASS |
| 3 | 完全重叠不双扣、部分重叠分区分账 | PASS：合并与单区结果逐元素差 `0.0` |
| 3 | 类条件 posterior、单类文本扰动、posterior detach | PASS |
| 3 | 零父证据精确零 | PASS |
| 3 | `train()` `2×2` 梯度 | PASS：末层 child mask、child query、relation 梯度有限且非零；输入梯度有限 |
| 3 | `train()` `1×1` 梯度 | **FAIL**：未到达反传，前向即抛空间统计异常 |
| 3 | `train()` batch 内单个 parent 输出独立 | PASS：batch 输出与逐 parent 运行一致 |
| 3 | 1203 类、`train()` `2×2` | PASS：shape `[1,1203]`，有限、非正 |
| 3 | 1203 类、`train()` `1×1`（迁移探针） | **FAIL**：默认 `hidden=128`，`Expected more than 1 spatial element` |

### 探针自身的修正记录

第一轮曾因探针构造错误产生额外失败：随机张量被重复采样、tie 用例把类嵌入误当作 posterior、batch 独立性误切了共享类嵌入和 overlap；这些错误已修正后重跑。表中的失败不是由这些测试写法造成的，尤其 `1×1` train 失败有源码行栈，batch-2 `1×1` 的 v4/v5 对照也由同一矩阵直接证明。保留这条记录是为了区分“探针失误”和“模块失败”，避免把测试脚本问题误报为理论结论。

## 与 v4 契约的迁移

v4 报告记录三轮 16 项、15 PASS、1 FAIL：`833a4bf6` 的 `train()` 单行 `GroupNorm` 失败，其余 shape、时间、上限、owner/tie、重叠、类条件、1203 类、梯度等通过。本审计对 v5 做了两件事：

1. 用新 v5 探针重跑 20 项主动反例，确认除归一化路径外未看到输入/owner/类别条件退化；第二轮 6 项全部 PASS，第三轮除 `1×1` 外全部 PASS。
2. 把 v4 的独立探针只改绑定 SHA 后在 v5 副本上重跑。结果为 **14 PASS、2 FAIL**：失败的是“实际 1203 类非空前向”和“单行 train 不复现归一化崩溃”。这两项都源于同一个 `InstanceNorm2d` 空间统计问题；v4 的 15 项契约没有被证明整体保留为 15/15，因为默认入口路径已变。

这不是把 v4 的 PASS 继承下来，而是新 SHA 的独立迁移证据。旧报告中的 `833a4bf6` 结论、作者自带测试和本轮 v5 结论必须分别绑定各自 SHA。

## 实现失误还是理论失败

当前失败首先应归为**实现失误**，不是 CEO/证据所有权的理论否定：

- 目标异常栈明确来自 `child_mask` 的归一化算子对 `1×1` 空间输入拒绝；
- 同一 v4 契约在 `2×2` 上有有限、非正输出和有限梯度；
- v4 的 `GroupNorm` 在 batch 2、`1×1` 可通过，v5 的空间独立归一化反而使合法输入失败；
- 因此应修实现并重新审计，不能据此断言类别条件 explain-away 无效。

这也暴露了一个接口缺口：模块文档把 `1×1` 视为合法 `[E,D,H,W]`，但没有在构造函数中固定空间尺寸，也没有说明训练时禁止 `1×1`。即使采用新算子，也要明确“ROI 最小空间尺寸/是否允许退化为全局 pooling”的约定，避免在真实接入时把形状错误误判为机制失败。

## 是否允许短 GPU 只读遮挡 pilot

**本次审计不允许。** 用户要求“如果全部 PASS 明确允许短 GPU 只读遮挡 pilot；若否明确 FAIL 点”。本次不是全部 PASS：

- `train()` `1×1` 未修复；
- `train()` batch 2 `1×1` 出现新回归；
- 默认 1203 类 `train()` `1×1` 入口失败；
- `eval()` `1×1` 同版本也失败。

因此不能启动基于 `417f734e` 的 GPU 采集或训练。修复归一化并明确最小空间尺寸后，用**新 SHA、固定 ROI 形状、新输出目录和完整 receipt**重审；通过后再安排短 pilot。若只是为了验证 teacher/因果信号而使用 v4 `833a4bf6`，它的 eval 模式 `1×1` 与 `2×2` 都可运行，但仍需满足冻结父框、检测器、NMS、轨迹 ID、类别表和只读采集等条件；这不能替代本 v5 的 FAIL，也不能证明模块增益、source-heldout 泛化或官方 TETA/ClsA。

## 结论摘要

- **快照 `417f734e…`：FAIL。** 唯一变更没有修复目标失败，并让 v4 原本可运行的 batch 2、`1×1` 训练输入失败。
- v5 第二轮输入契约、owner/tie、重叠分账、类别条件和 `2×2` 梯度仍通过；不能把这些通过写成训练可用或方法收益。
- 当前问题是实现层面的归一化/空间尺寸契约问题，优先修实现并重审；不把本次失败外推为“证据所有权方向理论无效”。
- 未训练、未采集、未改主文件、未停止或修改任何其他会话进程。BDD 不属于本会话，本审计未读取或启动它。
- 审计后主文件的 `f5228ec9…` 编辑未被本报告执行或认证；其新类尚未接入 `child_mask`，不能据此宣称 `1×1` 问题已经修复。
