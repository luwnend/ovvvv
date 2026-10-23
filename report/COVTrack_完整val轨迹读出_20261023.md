# COVTrack 完整 val：轨迹类别读出阶段报告

> 2026-10-23；状态：4 个候选读出与 1 个发布基线的官方 TETA 已完成，汇总核验通过。
> 这是方法开发的阶段结果。普通 sumlog 是控制实验；可学习 TCES、依赖建模与身份拆分尚未成立。

## 1. 已得到的结论

完整官方 val 上，发布后验 sumlog 比冻结 TCC 的 Base TETA **+0.220**、Novel TETA **+1.566**；Base/Novel ClsA 分别 **+0.661/+4.698**。
旧分析中的跨视频 ID 混合、把 text 当作 fused TCC、GT 先筛观测等问题已定位；其失败结论已从正式文档删除，不能继续作为停止方向的证据。
现在可以确认完整后验的轨迹读出有可用信号。该结果没有证明新的可靠性状态、依赖模型或联合关联方法有效，也未触发新训练。

## 2. 冻结协议与完整性

- Split：`data_tao/annotations/validation_ours_v1.json`，988 视频 / 36,375 图像 / 112,798 GT 标注；296 有 GT 类，按 `frequency` 分为 261 base / 35 novel。
- 模型：发布 COVTrack R50，`ctao_public.pth`，SHA256 `e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c`。本轮继续用已有发布权重缓存进行只读控制，没有新启动权重推理。
- A 参数：`thr=.37 / max_per_img=80 / memo=50 / momentum_embed=.4 / max_fusion_ratio=2 / confused_features=True`。
- 全部 1,529,908 输出观测 / 324,639 轨迹，键为 `(video_id, local_track_id)`；109,024 条单观测轨迹也保留。
- 聚合在 GT 评分之前完成；GT 不决定观测选择、整行融合分支或方法输出。corrected mask 只用公开类别词表的 frequency 元数据。
- 36,371 个 dump，4 个无 dump 图像均无原始输出；原 TCC 与发布 JSON 完全一致。5 个逐帧 argmax 差异只影响 1 条 TCC，真实局部 TETA 全 16 项原始数组严格相同。
- 候选只改 `category_id`；行顺序、框、score、ID 均保留，无二次 TCC。官方 TETA50 直接读取结果。
- 原始 Loc/Assoc 全数组逐类严格一致；官方逐视频 TP/FN/FP 重建 ClsA，误差门限 1e-12，核验通过。ΔTETA = ΔClsA / 3。

缓存、列映射、公式与 TCC 对齐：`results/tces_audit_20261023/manifest.json`、`cache_readback_verification.json`。
评测器源码哈希由各 `readouts/<rule>/receipt.json` 绑定；原始 summary pickle 只含显示舍入值，本轮额外保存未舍入数组与计数。

## 3. 官方完整 val 结果

| 固定轨迹读出 | Base TETA | Base ClsA | Novel TETA | Novel ClsA |
|---|---:|---:|---:|---:|
| 发布 TCC（冻结对照） | 39.554 | 19.547 | 34.205 | 3.507 |
| 发布后验 sumprob | 39.684 | 19.936 | 34.292 | 3.768 |
| 发布后验 sumlog | 39.774 | 20.208 | 35.771 | 8.205 |
| corrected mask + TCC | 39.433 | 19.185 | 34.389 | 4.057 |
| corrected mask + sumprob | 39.509 | 19.410 | 34.708 | 5.015 |

五行的 Base LocA/AssocA 均为 **57.155/41.960**；Novel 均为 **58.173/40.936**。
发布后验 sumprob 增益小于 sumlog。后者是普通 log pooling 控制，不计为论文创新；它应成为后续新读出的直接对照。

corrected mask 是实现诊断。发布 mask 对应 `cid in lvis_novel_label_ids`，与正确 `cid-1` 有 93 个前景列不一致（72 base、21 rare）；背景另列保留。
mask 改正后的固定读出有 Base 退化，不能自动替换冻结发布对照。
同一 sumprob 下，改 mask 的 Base/Novel TETA 变化为 **-0.175/+0.416**；同一 corrected mask 下，sumprob 相对 TCC 为 **+0.075/+0.319**。
这些对照只重读原检测框和轨迹，未重跑 NMS、检测筛选或关联。corrected-mask sumlog 尚未做官方评估，当前表不是完整的 mask×聚合器消融矩阵。

## 4. 收益集中程度与支持数

sumlog 的 Base 为 **88 类改善 / 52 类退化 / 121 类不变**；Novel 为 **8/2/25**。
Novel 逐类变化如下（百分点）：

| Novel 类 | ΔClsA | GT 视频数 |
|---|---:|---:|
| gorilla | +66.755 | 1 |
| drone | +45.834 | 4 |
| barbell | +38.483 | 1 |
| saxophone | +14.357 | 1 |
| ax | +7.414 | 3 |
| matchbox | +5.649 | 1 |
| shawl | +5.632 | 3 |
| lawn_mower | +0.036 | 6 |
| hockey_stick | -1.485 | 5 |
| army_tank | -18.245 | 8 |

前三个获益类占 Novel 净 ClsA 增益的 **91.9%**，其中 gorilla、barbell 都只有 1 个 GT 视频。
35 个 novel 类中 26 类只出现在 1 个 GT 视频，29 类不超过 2 个；本轮是同一 val 上的探索性结果，不能称为跨视频泛化已验证。

固定类集合、删去一个视频的 sumlog ΔClsA 范围：Base **[0.434, 0.917]**；Novel **[2.791, 5.181]**。这是影响分析，不是置信区间。
另做配对视频重采样 2,000 次（seed 20261023），从 TP/FN/FP 重新合并后计算宏平均；完整结果在 `paired_summary_1c8ffa976ed8.json`。
普通视频 bootstrap 平均每次会漏掉约 10 个 novel 类，无支持类 Δ 贡献 0，可能将宏平均效应向 0 拉；该区间仅作经验视频构成扰动诊断，不作显著性、训练许可或独立泛化证据。

## 5. 独立 GT 诊断（不等于官方 ClsA）

类别无关的逐图 Hungarian 最大总 IoU 匹配，再保留 IoU≥.5：65,711 个匹配观测 / 10,219 条匹配轨迹，35 个 rare 类均有支持。
GT 只评分已完成的全观测读出；该匹配不是 TETA 的轨迹对齐与本地聚类匹配。

| 全观测聚合后读出 | Rare 类宏 top-1 | Rare 类宏 top-10 |
|---|---:|---:|
| 原 TCC | 5.924 | 19.662 |
| fused sumprob | 6.280 | 50.948 |
| fused sumlog | 11.715 | 50.550 |
| text sumprob | 4.842 | 43.268 |
| image sumprob | 15.466 | 55.832 |
| corrected mask sumprob | 8.170 | 81.170 |

TCC 的 top-k 按票数排序，无票类并列受类别 ID tie-break 影响；不能据此把 top-k 差距全部解释为新语义证据。
8,573 条轨迹仅匹配到一个 GT 身份，1,646 条混合；匹配观测加权纯度 0.931，轨迹平均纯度 0.946。
Rare 的 1,720 个匹配观测中 1,461（84.94%）位于单身份轨迹，说明分类余量并不只来自身份混合；但未匹配观测无法核实身份，不能称为轨迹已完全净化。
纯身份分层仍有 sumlog/分支选择余量；跨分支 GT oracle 只是诊断上界，不能证明无 GT 路由有效。

## 6. 下一步：针对自然冗余的证伪

不急于训练。先在固定发布后验上检查“外观相近的观测是否反复产生同一种分类错误”，控制同轨时间间隔、置信度与身份混合；单纯复制帧实验不能证明自然失效。
已有 dump 含 `track_feats[N,256]`、`cls_feats[N,512]` 和三路完整后验，均可 CPU 回读，无须重新推理；appearance 在 cosine 前需 L2 normalize。
若自然错误相关性成立，再比较依赖平衡控制、普通 sumlog、等预算随机/分数采样及 confidence 权重；GLS、注意力或聚类权重本身均不是新颖性证明。
发布 mask 与 corrected mask 各用自身的普通池化作对照；下一阶段如采用 sumlog，需先补 corrected-mask sumlog 官方控制，不能把 mask 修复量并入方法收益。
每轮至少两项检查：实现/GT 隔离 → 机制干预/等预算控制 → 纯混身份分层/完整 val 与逐类支持。后两轮尚未完成，不把调试次数计成科学审查次数。
训练由另一个会话负责；不停止、不修改 `results/covtrack_ctao_train20_bs24_20261022`，本轮未启动任何新训练。

## 7. 可复核产物

- 缓存与审计：`results/tces_audit_20261023/manifest.json`、`track_cache.npz`、`cache_readback_verification.json`。
- 每组预测、未舍入指标、raw 与计数：`results/tces_audit_20261023/readouts/<rule>/`，五组 receipt 均 COMPLETE。
- 汇总：`results/tces_audit_20261023/paired_summary_1c8ffa976ed8.json`；旧 `paired_summary.json` 仅为基线自检。
- 独立诊断：`diagnostic.json`、`scoreinput_manifest.json`；局部 argmax 控制：`argmax_tcc_control_local_teta.json`。
- 源码：`tmp_analysis/build_track_posterior_cache.py`、`evaluate_track_readouts.py`、`summarize_track_readouts.py`、`score_track_cache_diagnostic.py`。
- 日志：`logs/tces_audit_20261023_*.log`；新颖性与评测审计：`tmp_analysis/novelty_audit_20261023.md`、`evaluator_audit_20261023.md`。
- 正式文档中的旧错误表格与结论已删除；旧快照只作审计溯源，见 `results/tces_audit_20261023/prior_reports/README.md`。
- 退出状态和保护字段的最终核验见审计目录 `completion_receipt.json`。
