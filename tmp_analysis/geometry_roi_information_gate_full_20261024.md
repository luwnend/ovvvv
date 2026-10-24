# Geometry-ROI 局部 token 全量信息门（2026-10-24）

**结论：`STOP` 训练“类别/局部 token attention”这一路线。** 6 source、3
source-pair fold、122 个 FIT 人工 identity anchor 上的固定特征评价实际完成；
局部空间 token 相对**同样使用历史窗口的 raw256/全局均值**没有产生稳定的身份
增量。它可以保留为描述性诊断，不能作为网络模块收益或论文创新证据。

## 证据边界

- Token 目录：`/raid/datasets/chc_data/geometry_roi_full_20261024_v1`，
  276 帧、15,153 retained rows、12 视频、6 source。18 帧 pilot 重叠页逐成员
  字节一致。
- 官方 collector 在写完全部页后，因 manifest 路径被替换，postflight 报
  `Manifest changed`，因此**没有** terminal `collection_receipt.json`。
  这不是形式上的 collector `PASS`。逐页独立审计 276/276 通过：NPZ 成员、
  float16 空间张量、有限值、row locator、bbox、raw256 与父缓存完全一致。
- 新 receipt
  `results/geometry_roi_full_audit_receipt_20261024_v2/audit_receipt.json`
  只证明 payload 被完整独立核验，并显式记录缺失 terminal receipt 与
  `_v1 → _v2` 路径事后改写。它不是补造的 collection receipt。
- 冻结 gate manifest：
  `results/geometry_roi_gate_manifest_20261024_v1/manifest.json`，
  SHA256 `4d2b2b775b4a4d207829cfcb359b3ec8ab34e28291847570ef5d4c7ba7ec109a`。
  评价集为 129 个同 GT 跨步对、241 个同预测类别异 identity hard-negative
  对。每个 fold 分别有 30/39/60 个正对与 37/132/72 个 hard-negative 对。
  这些 pair 共享端点、视频与轨迹，不是独立样本。

## 全量固定特征结果

主指标为 source 宏 AUROC；top1 固定 57 个同时含合法正例与 hard-negative 的
query，GT 只用于评分，未知 identity 候选保留但选中后记为 unresolved。

| arm | 同身份 vs hard-neg AUROC | top1 正确 | 相对 raw repair/break |
|---|---:|---:|---:|
| 当前 raw256 | 0.9164 | 45/57 | — |
| 历史 raw256 因果均值 | 0.9480 | 48/57 | 5/2 |
| 当前 token global mean | 0.8887 | 37/57 | 0/8 |
| 历史 token global mean | 0.9373 | 48/57 | 5/2 |
| 当前 token center | 0.9025 | 40/57 | 1/6 |
| 空间 top-k + 固定几何先验 | 0.8773 | 39/57 | 2/8 |
| box-aligned 历史局部 token 均值 | 0.9503 | 48/57 | 6/3 |
| 空间 token 随机置换控制 | 0.8888 | 41/57 | 2/6 |
| token global mean + 几何先验 | 0.8785 | 35/57 | 0/10 |

看似最好的 box-aligned 历史局部 token 对**历史 raw 均值**只多修 1 次、也破坏
1 次，source 宏 top1 差 `−0.0794 pp`；对**历史 token global mean** 同样
`−0.0794 pp`。历史均值带来的表面增益可由 raw 与 global-token 共有的时间
平滑解释，不是局部空间证据的独立贡献。空间 top-k 则比历史 global mean 的
source 宏低 `18.49 pp`，比随机空间置换低 `3.89 pp`。

因此本阶段不训练普通局部 attention、类别条件 token attention 或历史 token
memory，也不把 alignment 的微小 AUROC 变化包装成新模块。若后续重启，
必须先出现新的可证伪机制，而不是换池化、增加 attention head 或只用单帧 raw
作对照。

## 复现与核验

```bash
/home/chenhc/miniconda3/envs/covtrack/bin/python \
  /home/chenhc/ov/OVMOT_RESEARCH/tmp_analysis/geometry_roi_information_gate_20261024.py \
  --cache-dir /raid/datasets/chc_data/geometry_roi_full_20261024_v1 \
  --receipt /home/chenhc/ov/OVMOT_RESEARCH/results/geometry_roi_full_audit_receipt_20261024_v2/audit_receipt.json \
  --out /home/chenhc/ov/OVMOT_RESEARCH/results/geometry_roi_information_gate_full_20261024_v3

/home/chenhc/miniconda3/envs/covtrack/bin/python \
  /home/chenhc/ov/OVMOT_RESEARCH/tmp_analysis/independent_audit_geometry_roi_information_gate_20261024.py
```

- 作者单测：14 项，3 轮均 exit `0`。独立审计：重读标签、token 与
  receipt，独立重实现全部 arm 和 top1，3 轮均 exit `0`。
- 全量 evaluator 主输出：
  `results/geometry_roi_information_gate_full_20261024_v3/evaluation_result.json`。
- 独立复算：
  `results/geometry_roi_information_gate_full_20261024_v3/independent_audit_full.json`。
- 三轮单测与独立复算的实际 stdout/stderr/exit code 都在同目录 `tests/` 和
  `independent_audit_rounds/`。

**实现限制：**`raw256`、`fgfa`、`topk` 等是冻结缓存上的固定统计，没有训练
参数，也不是部署 tracker。`fgfa` 只做 box-normalized 同网格平均，没有
flow warp。当前 top1 只评价已知合法 identity 的固定候选集，不外推 unknown
候选、no-match 或在线闭环收益。论文如提及此阶段，只能写成负对照/可证伪
信息门，不能写成 SOTA 或局部 attention 已有效。

## 当前判定

| 项目 | 判定 |
|---|---|
| ROI token payload 完整性 | PASS（带 provenance hold） |
| 三折固定特征评价器 | PASS |
| 局部 token 超过同预算历史均值 | **未通过** |
| 空间匹配超过随机 token 置换 | **未通过** |
| 训练局部 attention/类别条件 token 模块 | **STOP** |
