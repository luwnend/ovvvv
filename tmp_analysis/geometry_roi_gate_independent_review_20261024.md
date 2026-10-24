# Geometry ROI information gate: independent review

结论：**STOP（as an identity module）**。原 gate 的 FGFA 增量
主要来自“当前观测 + 局部 track 历史”；跨 track 与严格过去信号
均没有稳定增量。局部空间 token 也没有超过 raw256。

## 复算与协议

- 独立读取 72 个评测 NPZ，重建 122 个 anchor、6 个 source、3 个 fold。
- 重建 552 个非 unknown pair，其中 370 个是 primary pair
  （129 同 GT、241 同预测类别 hard negative）。标签只用于评分；没有用 GT
  过滤候选池或选择 memory。
- unknown 为 0；仍保留 unknown 入候选池并计 unresolved 的规则。
- 作者的 552 pair、7 个 arm 的 source-macro AUROC 与 top-1 计数
  已逐项对齐，见 JSON 的 `author_parity`。

## 关键混淆

- 129 个同 GT pair 中，98 个来自同一个 local track；
  241 个 hard-negative pair 中，17 个也来自同一个 local track。
- 历史累计使用“当前 + 过去”，不使用未来；不存在直接未来泄漏。
- 57 个 top-1 query 中，50 个可在候选池看到未来锚点；
  47/57 的 FGFA top-1 选择指向同 track，
  其中 36/57 指向未来同 track 行。后一行的历史包含 query，
  所以这只能证明局部 track 连续性，不能证明新的跨轨迹关联。
- 57 个 query 只对应 20 条 GT 轨迹；按轨迹汇总会暴露这一点。
- 所有 FGFA 消融只是重新评分固定候选池；不对 memory 做 GT 选择，
  也不改变部署时的候选构造。

## 指标

| arm | source-macro AUROC | pooled AUROC | top-1 acc | top-1 repair/break vs raw |
|---|---:|---:|---:|---:|
| raw_current | 0.9164 | 0.8521 | 0.7895 | 0/0 |
| fgfa_inclusive_current | 0.9503 | 0.9155 | 0.8421 | 6/3 |
| fgfa_exclusive_strict_past | 0.6551 | 0.6728 | 0.5789 | 2/14 |
| token_global_current | 0.8887 | 0.8128 | 0.6491 | 0/8 |
| spatial_topk_current | 0.8773 | 0.8195 | 0.6842 | 2/8 |
| geometry_only | 0.7708 | 0.7134 | 0.5439 | 1/15 |

## 判定

- FGFA inclusive pair 相对 raw 的增量：+0.0339。
- 严格过去（不含当前）的增量：-0.2613；跨 local track 的增量亦不足以证明独立身份信息。
- 跨 local track 子集的 FGFA 相对 raw 增量：-0.0290；（token_global / centroid / top-k 均不优于 raw 的 source 级指标）。
- 原始 FGFA top-1：正确 48/57；相对 raw 修正 6 次、破坏 3 次。
- 将同 track 未来选择中的 current-match 权重减半后：正确 50/57；修正 6 次、破坏 1 次。
  这是固定 50/50 权重的消融，不是按 GT 调参；仍需在因果候选池
  下复验，不能直接作为部署收益。
- top-1 的修复/破坏与轨迹级 macro 指标表明：pair AUROC 高不等于
  部署时能减少关联错误；可用信号主要是局部轨迹历史连续性。
- 因此：**不训练**当前类别条件局部 token attention；
  **不把** geometry ROI gate 当作主要创新点或论文指标。

数据缓存：`/raid/datasets/chc_data/geometry_roi_full_20261024_v1`
保留 514 MiB，用于复核；不是训练缓存。

CPU only；未加载模型、未训练、未使用 GPU。

## 严格因果候选池（禁止同 track 未来行）

| arm | eligible queries | top-1 correct | repair/break vs raw |
|---|---:|---:|---:|
| raw_current | 48 | 38 | 0/0 |
| fgfa_inclusive_current | 48 | 40 | 3/1 |
| fgfa_half_current_split | 48 | 40 | 3/1 |
| fgfa_exclusive_strict_past | 48 | 25 | 1/14 |

该池只有 48 个 query（来自 20 条 GT 轨迹），不能单凭它声称稳定增益。
