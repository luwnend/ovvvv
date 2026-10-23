# OVMOT research

COVTrack 上的开放词汇多目标跟踪研究：先核验机制和完整验证集结果，有把握后再训练。

- [项目进度](report/项目进度.md)：仅记录大阶段结论与下一步。
- [方法设计与新颖性边界](report/COVTrack_论文级改进设计.md)。
- [完整 val 轨迹读出报告](report/COVTrack_完整val轨迹读出_20261023.md)。
- [方法调研笔记](report/OVMOT_方法调研笔记.md)。

冻结对照：发布 COVTrack R50 / ctao_public.pth / 官方完整 TAO validation / A 参数 .37 / 80 / 50 / .4。
发布 TCC Base/Novel TETA=39.554/34.205；普通 sumlog=39.774/35.771。
普通池化是控制实验，尚未构成新的机制方法；26/35 novel 类只有一个 GT 视频，完整 val 结果仍需范围与泛化核验。

本仓库保存经过核验的研究脚本、设计文档及轻量指标/哈希回执。大型数据、权重、逐帧后验和预测保留在本地研究目录，不上传。
`evidence/` 是对应阶段 `results/` 中 JSON 回执的副本；其中绝对路径记录原始执行机器，不能据此假定这些文件已在仓库中。
当前本地工作目录为 `/home/chenhc/ov/OVMOT_RESEARCH`；仓库工作副本为 `/home/chenhc/ov/ovvvv`。阶段结束后同步已核验的脚本/文档并提交。

上游：[COVTrack](https://github.com/zekunqian/COVTrack)，冻结源码 commit `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`。
遵守数据集及上游代码许可；本仓库未重新发布上游仓库、数据或 checkpoint。
