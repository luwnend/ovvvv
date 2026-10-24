# Evidence ownership：FIT 事件预算与阶段停点

日期：2026-10-25  
状态：`GT_FREE_BUDGET_PASS; CONTRACT_REVISION_PENDING_REVIEW; NO_TRAINING`。本节记录已完成的可部署事件预算和独立实现复核，不作方法收益或因果结论。

## 事件预算

只用冻结 FIT 缓存的检测框、标签、released ID，按 `area_ratio∈[0.01,0.8]`、`IoC>0.8` 和 child 中心落在 parent 内建立父—子边；每个 `(video, parent track, child track)` 只保留最早一帧，避免把重复观测当独立事件。脚本为 `tmp_analysis/audit_fit_containment_events_20261025.py`，输出在 `/home/chenhc/ov/OVMOT_RESEARCH/results/fit_containment_event_budget_20261025_v2/`。

| 量 | 数值 |
|---|---:|
| 扫描帧 | 15,283 |
| 原始边 | 425,844 |
| 去重 parent-child 边 | **227,001** |
| unique parent frames | 165,504 |
| unique parent tracks | 46,704 |
| unique child tracks | 104,029 |
| 有事件的视频 | 448/448 |
| 有事件的 source | 383 |
| area ratio p10 / p50 / p90 | 0.015 / 0.061 / 0.241 |
| IoC p10 / p50 / p90 | 0.879 / 1.000 / 1.000 |

七个数据 block 均有覆盖：HACS 95、LaSOT 88、Charades 74、AVA 61、YFCC100M 48、BDD 48、ArgoVerse 34 个视频。最常见的预测类对包括 `car→wheel` 4,951、`person→shoe` 1,140、`car→taillight` 3,730、`person→short_pants` 813 条去重边。这个分布**偏向部件检测**，不能把事件数量等同于 independent error support；顶层类对高度集中，pilot 必须按 source→GT trajectory 抽样和汇总。

回执：`PASS_GT_FREE_BUDGET_AUDIT`，脚本 SHA `52a9a230…`，边文件 SHA `80f1aace…`，summary SHA `30ba7a06…`，未使用 GT、未训练、未用 GPU。输入 frame execution SHA 与冻结 full collector 一致。

## 当前方法状态

- `semantic containment` 和全名映射已经降到零参数强基线。source-heldout 上后者类宏下降；语义门增益极小、高精度低覆盖且主要由 `person` 驱动，不能当网络创新。
- CEO 仍是唯一值得做因果 pilot 的模块方向：父 ROI、child occupancy、类别条件 explain-away；直接使用真实图像遮罩 teacher。
- 现有合成 `evidence_ownership_head.py` 在有独立复核后已修订，但仍未通过独立重审。阻断项包括：关系头需要真正逐类条件而不是只给所有类乘同一个空间标量；owner 规则、`max_children`、NaN/索引 fail-closed、1203 类宽度、posterior detach 等必须在正式真数据前再核。作者测试通过不等于方法可用。

## 下一道门

不直接训练。先重新独立审当前头版本；然后把事件预算转成固定的、source 分层的小 pilot manifest：每个 (source, GT trajectory) 只贡献一个 parent/child 事件，覆盖多个 block；对同一 parent 使用 child mask、面积/位置匹配 mask、非 child 等面积 mask 和错误父框 child mask 做真实图像前向。只有 child mask 超过面积匹配控制且跨 source 修复多于破坏，才允许采集更大 teacher 缓存或训练。

若独立审仍失败，该阶段记为理论/实现失败；若因果 pilot 不通过，只停止 CEO 的普通 part—whole 形式，不把结果外推为所有局部证据或所有关联学习无效。
