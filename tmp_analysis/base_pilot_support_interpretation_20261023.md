# Pilot natural-error support interpretation — 2026-10-23

**结论：pilot 足以建立低容量、类别共享的质量基线；不足以将配对残差解释为新的错误依赖机制。** 扩采选择为 sealed plan 的全部 fit/tune 人工 anchor 帧，不按 tune 错误、特定类别表现或 test outcomes 筛选。当前 test 与 class-holdout 自然错误结果保持关闭。

## 已核验的监督支持

来源：`results/base_error_collection_20261023_v3/support_v1/support_receipt.json`（`PASS_FIT_TUNE_SIGNAL_SUPPORT_AUDIT`）、`support_summary.json` 和独立封存的 GT-free 索引。全部1,382条监督观测仅来自 fit/tune 视频×146个 readout-fit 类；36个 class-holdout 类的 correctness 行为0。原始 source JSON 包含 test partition 标签，但没有 test prediction/GT join、GT frame map 或错误结果。

| 当前 full1203 稀疏人工-anchor 协议 | Fit | Tune |
|---|---:|---:|
| 可监督观测 / hard errors | 1,067 / 615 | 315 / 228 |
| 未加权 hard error rate | 57.64% | 72.38% |
| Source groups / videos / GT trajectories | 14 / 24 / 113 | 7 / 9 / 41 |
| 实际监督类别 | 15 | 12 |
| eligible manual anchors 得到监督比例 | 1,067/2,138 = 49.91% | 315/647 = 48.69% |
| primary same-predtrack AND same-GT pairs | 6,876 | 1,581 |
| primary pair GT trajectories / predicted tracks | 93 / 128 | 25 / 35 |
| both-wrong / same-wrong-class pairs | 4,078 / 1,953 | 1,152 / 334 |

Fit/tune 实际监督类别仅重叠 **3类：bus(173)、car(207)、person(793)**。这是 readout-fit 集合内部的支持与分布差异，不能称为预留类别 holdout 的结果。Fit 的 person/car 占854/1,067=80.04%，tune 占159/315=50.48%；不能据此将平均误差差距归因为某个单一机制。

## 配对数量、集中度与 reference 可用性

Exhaustive pairs 重复使用相同端点和轨迹，8,457对不是8,457个独立样本。Fit/tune 最大单 source 的 pair 份额分别为22.06%/30.80%；最大单 GT trajectory 份额为10.78%/20.56%。在 both-wrong 事件中，这两种最大份额进一步达到28.37%/36.63%和18.17%/28.21%；same-wrong-class 的最大单 source 份额为26.78%/32.04%。因此模型损失、crossfit 和不确定性必须尊重 source group/轨迹重复观测，不能按 pair 总量扩大证据强度。

按**所有 emitted 观测**构建的 GT-free 预测轨迹长度≥4，因而去掉两端后仍有≥2个 reference 的 primary pairs：**fit 6,859/6,876对，覆盖14源组、113条预测轨迹；tune 1,575/1,581对，覆盖7源组、31条预测轨迹**。这是输入可用性统计；reference 不按 GT identity、correctness、class holdout 或 oracle purity 选择。reference 仍可能含错误身份，不能据此声称轨迹干净或支持密集邻帧推断。

## 最小可辩护目标与已运行基线

先学习单观测 hard-error probability 的类别共享低容量质量函数，并保留 soft-risk 作为诊断；不使用 GT class ID、source ID 或 GT-derived identity purity 作为输入。B0 为 fit 平均错误，B1 为固定容量的共享质量基线，fit 按 source group crossfit，tune 只评价。`mean_quality_v1/diagnostic_summary.json` 的独立审计结果：source→GT trajectory→observation 加权 Brier，fit 为 **0.25888→0.20814**，tune 为 **0.23373→0.19156**。Tune B1 平均预测错误0.59835，实际0.68796，尚未校准；这些结果只支持质量有描述性可预测信号。Bootstrap 区间条件于已拟合预测且不重新拟合，不能代替训练不确定性；更强的 inner source-group CV 质量对照仍缺失。

后续若扩采支持足够，最小机制候选仍是独立规格中类别共享、直接回归 wrong-evidence mean 后的 centered wrong-evidence trace surrogate。必须先建立 mean 校准，并与 quality、quality+agreement 的同容量 pair 控制比较；不得将 `b_i*b_j` 称为 covariance，不得从 hard same-wrong-class 频次推断新的依赖。相同预测标签时两个 hard errors 相同，预测标签不同时 joint-wrong 由端点错误决定；plug-in `p*(1-pi)` mean 还存在同身份坐标乘积非负的代数退化。详见 `base_error_learning_objective_20261023.md` 与其 CPU 合成检查。本次不启动该 pair learner。

## 不依赖 outcomes 的完整 fit/tune 扩采入口

新准备器：`tmp_analysis/prepare_base_error_full_collection.py`。新 stage：`results/base_error_full_collection_20261023/`。复用原 sealed plan 的 source-group split、146/36类别划分，并重新按唯一 `(video_id, original frame_index, track_id)`、C-TAO `is_ori=True`、唯一 synset bridge 和 bbox误差≤1e-6重建 exact original-manual 交集。只选择 fit/tune。

| 完整选中元数据 | Fit | Tune | Total |
|---|---:|---:|---:|
| Allocated videos | 405 | 45 | 450 |
| 非空采集视频 / source groups | 403 / 341 | 45 / 42 | 448 / 383 |
| Manual anchor frames | 13,791 | 1,492 | 15,283 |
| Exact manual targets | 42,020 | 4,888 | 46,908 |
| Fit-category manual targets | 39,903 | 4,557 | 44,460 |
| 有fit-category标签的类别 / 其中≥2源组 | 134 / 65 | 38 / 9 | union 140 |

完整 fit/tune 的 fit-category 支持交集为32类。分配视频1720、1974没有 exact anchors，明确列为未采集；不能作为两个 tracker-reset 视频计入。相对 pilot fit/tune，增加 **14,258帧、415个非空视频、362个源组、43,939个manual targets、41,675个fit-category targets**。这些只是标签与帧的上限，不保证能匹配到相应自然错误；完整池仍受 person/car 和长尾单源类别影响。

按已完成 pilot 的394.595秒/1,299帧线性外推，完整15,283帧约 **77.4分钟**，新增部分约72.2分钟。估计排除了审计、I/O和不同视频内容的变化；完整采集应独立重跑，因为旧 NPZ 绑定旧 annotation SHA，不能复制充当 full run。

`full_inference.json` 只含白名单字段、完整1203类别、空 annotations/tracks。`manual_anchor_targets.json` 独立保存监督、source annotation IDs及旧官方类别桥；`frame_id=sample_step`逐视频从0计数，`frame_index/source_frame_id`保留，memo50仍是稀疏采样步。Producer 自检已通过；独立 `independent_metadata_audit.json` 也已重建完整52,911个原始交集并通过 full fit/tune 15,283帧/46,908 anchors 核验。本准备与解读没有启动 GPU、采集或训练。旧 plan、pilot及其准备器、runner、verifier、support脚本前后hash相同。

## Bound evidence

- Parent plan SHA256: `f37f1ecc52734953ea5269407ab0aa8013e650e97bed4dde3cdd399a11e13ba5`.
- Pilot support receipt SHA256: `f0a0624f597086183a6638ddb1bd8a090fcd2c2acb1a25777114eb10a04df1f6`.
- Full prep script SHA256: `6effccf38b142428fca9dbfe826fe580af1f2136fe34ed3f5d5aa079fb034601`.
- Full collection plan SHA256: `cbb14c6fd41135d887b46ce1b5869f0068b76922bf5c40f87f78ec2c2a715f9b`.
- C-TAO source SHA256: `4269b1a5f026350cf45fbbf61689894f7d45d9bd2730331e73ab6f841d9b3adc`.
- Official source SHA256: `7eb551fdeeeebc76b876ae255f91dc5662c7270a125955c5f1be2d9bd30921d0`.
