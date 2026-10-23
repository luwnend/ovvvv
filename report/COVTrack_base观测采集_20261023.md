# COVTrack base观测采集与暂停续接 — 2026-10-23

当前已完成独立full-vocabulary导出接口、CPU合约和4帧真实smoke；1,299帧pilot正在采集。**尚未完成pilot封存、自然错误支持或新模型训练，研究目标继续。** 本报告记录接口阶段边界，后续结果须以实际回执更新。

## 已完成的可核验接口

独立源码副本为`work/covtrack_base_error_20261023`，原`work/covtrack_tces_v2`的101个源码文件逐项hash未变，新copy只修改ROI/MOT两文件并新增dump utility。冻结公开`ctao_public.pth` SHA为`e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c`；既有训练的config/launcher与正式baseline均受保护。

实际采集词表为LVIS-v1 1203前景+背景，保存`posterior_category_ids=[1..1203,-1]`、实际337列novel mask及text/image融合指数。完整词表改变softmax竞争、融合和检测/关联输出，不能等同历史val296后验或将裁剪结果声称为同缓存复现。`fix_bg=False`、ensemble开启、GT可视化关闭。阈值`.37`、max80、memo50、momentum`.4`、fusion2记录在每次run config；memo单位为选中帧的采样步，并非密集物理时间。

每帧保存原始track/cls特征、三路后验、proposal→NMS→tracker keep→emitted映射、原始/采样时间、video/image/local track ID。`id=-1`单列保留并用emitted flag区分。零观测帧也必须有NPZ和receipt；不能静默跳过异常。聚合键固定为`(video_id,local_track_id)`，GT只能在后续监督/评分层出现。

## 三轮接口核验与修复

| 轮次 | 核验一 | 核验二 |
|---|---|---|
| 1 元数据 | image/video负类别和非穷尽类别侧通道改为白名单 | categories的标注聚合计数删除；注入拒绝检查通过 |
| 2 CPU合约 | 真实4帧pipeline复现旧`KeyError: frame_id`，桥接后精确恢复`0,1,0,1` | 1204列/337mask/rowmap、空帧和负ID合成检查通过 |
| 3 真实小测 | 4帧150行与model实际emitted结果精确一致，两次tracker对象重置 | 独立NPZ/receipt/hash/融合审计实际exit0；71源码与7保护项通过 |

旧准备目录`results/base_error_collection_20261023`和`_v2`保留为实施偏离来源，不能作为合格缓存使用；`_v2/smoke_v1`实际exit1，未产生模型观测。合格准备目录是`results/base_error_collection_20261023_v3`，真实通过的run是其`smoke_v2`。

smoke融合重建最大绝对误差`1.8621592845491364e-7`。真实smoke没有空帧或负ID，这两项当前仅有CPU合成证据。完整谱系：`tmp_analysis/base_error_collection_deviations_20261023.json`、`base_collection_protocol_review_20261023.md`、`smoke_v2/independent_collection_audit.json`。

## pilot范围及暂停续接

pilot为41视频/28源组/1,299稀疏人工anchor帧，另存3,953精确人工anchor目标，fit可用标签上限2,138。选择这些帧依据标签可用性；每个选中帧保留全部GT-free模型输出，不能称密集自然错误总体。视频/源组切分不代表released detector从未见过这些数据；class holdout是readout的pseudo-novel。

启动的run：`results/base_error_collection_20261023_v3/pilot_collect_v1`。PID4166653，原工具session91862，日志`execution_logs/pilot_collect_v1.log`。耗时初估6–10分钟。设备固定UUID`GPU-2a55197e-2c19-45ff-cf33-13031951f9d0`，启动前无chenhc进程且约36GB空闲，原有另一用户进程按最新用户授权允许共卡。本账号既有训练PID3546537/3546538及4GB显示卡不使用。

已另启动独立后台CPU收尾程序`tmp_analysis/finish_base_error_collection.py`：等本run完成后运行两个已通过smoke的validator，将结果写入`pilot_collect_v1/background_completion_audit.json`。它不启动GPU工作、训练、GT匹配或重试。若原进程中断且没有完成回执，会明确记为不完整；不覆盖部分输出。原runner的真实exit须从session91862读取，后台程序无法凭收尾文件杜撰原进程exit。

稍后继续时先检查：

```bash
tail -n 5 results/base_error_collection_20261023_v3/execution_logs/pilot_collect_v1.log
cat results/base_error_collection_20261023_v3/pilot_collect_v1/collection_receipt.json
cat results/base_error_collection_20261023_v3/pilot_collect_v1/background_completion_audit.json
```

只有raw缓存封存及独立核验通过，才独立审查并运行`tmp_analysis/build_base_error_support.py`。该脚本尚未独立审查/运行，状态见`base_support_review_handoff_20261023.json`。匹配必须由所有原始TAO GT参与且不使用预测类别，再仅用精确manual-anchor交集生成fit/tune监督；未匹配为unknown，test自然错误不计算。先完成全emitted GT-free轨迹聚合，再打开监督，不能用GT选择读出观测。

## 学习目标边界

已写`tmp_analysis/base_error_learning_objective_20261023.md`和精确合成检查。hard同错受agreement平凡约束；`p*(1-pi)`中心化还会导致同身份每坐标乘积恒非负。两者不能直接声称学到共同错误协方差。候选是参考条件均值直接回归与out-of-source-group soft residual；仍需平均错误、quality+agreement、校准、同容量attention/相关融合控制。当前未训练，也未证明其新颖性或指标收益。

旧TCES错误失败结论已从正式文档移除，历史来源保持撤回标记。当前接口修复与合成检查不能替代真实机制证据。
