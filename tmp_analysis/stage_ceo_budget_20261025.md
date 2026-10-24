# Evidence ownership：FIT 事件预算与阶段停点

日期：2026-10-25  
状态：`GT_FREE_BUDGET_PASS; CONTRACT_REVISION_PENDING_REVIEW; NO_TRAINING`。本节记录已完成的可部署事件预算和独立实现复核，不作方法收益或因果结论。

## 事件预算

只用冻结 FIT 缓存的检测框、标签、released ID，按 `area_ratio∈[0.01,0.8]`、`IoC>0.8` 和 child 中心落在 parent 内建立父—子边；每个 `(video, parent track, child track)` 只保留最早一帧，避免把重复观测当独立事件。脚本为 `tmp_analysis/audit_fit_containment_events_20261025.py`，输出在 `/home/chenhc/ov/OVMOT_RESEARCH/results/fit_containment_event_budget_20261025_v4/`。

| 量 | 数值 |
|---|---:|
| 扫描帧（FIT only） | 13,791 |
| 原始边 | 388,367 |
| 去重 parent-child 边 | **205,999** |
| unique parent frames | 150,465 |
| unique parent tracks | 42,108 |
| unique child tracks | 93,909 |
| 有事件的视频 | 405/405 FIT |
| 有事件的 source | 341/341 FIT |
| area ratio p10 / p50 / p90 | 0.015 / 0.061 / 0.241 |
| IoC p10 / p50 / p90 | 0.879 / 1.000 / 1.000 |

full FIT 扫描覆盖 7 个数据 block；最大 source 仅占 4.79% 的去重边。最常见的预测类对仍包括 `car→wheel`、`person→shoe`、`car→taillight`，明显偏向部件检测。这个分布**偏向部件检测**，不能把事件数量等同于 independent error support；顶层类对高度集中，pilot 必须按 source→GT trajectory 抽样和汇总。

父轨迹级预算另为 **42,108 个 parent track contexts**，最长 36 个观测帧、中位 1 帧；不是每条边都是独立事件。固定 341 个 FIT source 都有支持，每个 source 至少 8 个 parent、最多 1,927 个。跨 1,036 个 parent 类、35,246 个 parent-child 类对。后续 pilot 必须以 parent 为最小抽样簇，child 类对计数只用于描述覆盖率。

身份支持联合是独立只读描述：205,999 条关系中 `both_known` 仅 1,821（680 同 GT、1,141 不同 GT）、`parent_known` 42,937、`child_known` 3,151、`unknown` 158,090，覆盖 336 个 source。unknown 没有被当负例。这条支持表只说明可做后续诊断的合法端点覆盖，不能把 1,821 条当作训练集大小；真实 teacher/分类评测仍需按 source→GT trajectory 去重并确认每个 source 的正负支持。

回执：`PASS_GT_FREE_BUDGET_AUDIT`，脚本 SHA `13a1d393…`，边文件 SHA `b9d969a4…`，summary SHA `937d4444…`，未使用 GT、未训练、未用 GPU。输入 frame execution SHA `e0e327a8…` 与冻结 full collector 一致。

## 当前方法状态

- `semantic containment` 和全名映射已经降到零参数强基线。source-heldout 上后者类宏下降；语义门增益极小、高精度低覆盖且主要由 `person` 驱动，不能当网络创新。
- CEO 仍是唯一值得做因果 pilot 的模块方向：父 ROI、child occupancy、类别条件 explain-away；直接使用真实图像遮罩 teacher。
- 现有合成 `evidence_ownership_head.py` 在有独立复核后已修订，但仍未通过独立重审。阻断项包括：关系头需要真正逐类条件而不是只给所有类乘同一个空间标量；owner 规则、`max_children`、NaN/索引 fail-closed、1203 类宽度、posterior detach 等必须在正式真数据前再核。作者测试通过不等于方法可用。

## 已通过的实现契约

当前 `evidence_ownership_head.py` 快照 `c0ea9872`：`833a4bf6` 的单行 `train()` GroupNorm 错误已修；InstanceNorm 方案在 1×1 ROI 仍失败，已换成通道 LayerNorm（StableSpatialNorm）。作者测试和 (1,1,1)/(2,1,1)/(1/2,2,2) 等 train 前向+梯度均通过，等待最终独立复核。`833a4bf6` 的 15/15 PASS 不含 train-mode 单行或 1×1 测试：`python -O` 下 shape 仍拒绝、1203 类宽精确、int32/int64 时间且拒未来、per-parent/全局上限分离、partial overlap 各格一 owner、关系输出 per-class、无 child 精确恒等、随机输入 finite/nonpositive、梯度有限。此结论只对快照有效；后续编辑必须重新审计。它放行真实遮挡特征采集，不放行训练或方法收益 claim。

## 下一道门

不直接训练。先重新独立审当前头版本；然后把事件预算转成固定的、source 分层的小 pilot manifest：每个 (source, GT trajectory) 只贡献一个 parent/child 事件，覆盖多个 block；对同一 parent 使用 child mask、面积/位置匹配 mask、非 child 等面积 mask 和错误父框 child mask 做真实图像前向。只有 child mask 超过面积匹配控制且跨 source 修复多于破坏，才允许采集更大 teacher 缓存或训练。

若独立审仍失败，该阶段记为理论/实现失败；若因果 pilot 不通过，只停止 CEO 的普通 part—whole 形式，不把结果外推为所有局部证据或所有关联学习无效。
