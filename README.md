# OVMOT research

COVTrack 上的开放词汇多目标跟踪研究：先核验机制和完整验证集结果，有把握后再训练。

- [项目进度](report/项目进度.md)：仅记录大阶段结论与下一步。
- [方法设计与新颖性边界](report/COVTrack_论文级改进设计.md)。
- [完整 val 轨迹读出报告](report/COVTrack_完整val轨迹读出_20261023.md)。
- [自然误判与依赖读出阶段](report/COVTrack_自然依赖核验_20261023.md)。
- [base完整词表采集](report/COVTrack_base观测采集_20261023.md)：全量15,283帧/681,602输出已完成（实际2.91小时），full support及独立全输出审计通过。
- [强质量控制](report/COVTrack_full质量控制_20261023.md)：nested source-CV主模型fit/tune Brier=0.147018/0.136580，独立审计通过；普通控制，不是新方法TETA收益。
- [fit身份混入支持](report/COVTrack_fit身份混入支持_20261023.md)：560,371条固定fit边中819条已知不同身份，涉及135源组；大量unknown保持未知。真实执行与独立全查通过，尚不构成训练条件。
- [特征与时空改进](report/COVTrack_特征与时空改进_20261024.md)：五种外观记忆对照及相对时空模块已加入。TAO train固定fit部分12源组、697帧的回放与1,210,939完整候选诊断独立通过；memory首要F1与released均0.824266。391个条件query中CV替代外观修4/破63，不支持直接加运动权重；尚无方法收益。
- [当前阶段续接](tmp_analysis/fit_lag_identity_support_handoff_20261024.md)：完整FIT固定lag `{1,4}` 的55,352,164候选与独立全量复算已完成、双方实际exit0；341源组/403视频。同预测类别CV-covered的共同1,331query上raw外观错16→99，长间隔困难值得处理，但CV直接替代仍破坏更多。必要支持门通过，尚无训练或新方法收益。[轻量完整支持](evidence/fit_lag_identity_support_20261024_v1/support_brief.json)与[完成回执](evidence/fit_lag_identity_support_20261024_v1/completion_receipt.json)。
- [身份特征与几何分离对照](tmp_analysis/identity_spacetime_followup_recommendation_20261024.md)：独立source-OOF、identity adapter × geometry及强控制建议，未训练；[动态外观候选](tmp_analysis/causal_appearance_transport_candidate_20261024.md)：已加入无训练角速度外推与低秩正交算子，三轮合成通过，真实数据收益与新颖性未建立。前一阶段[特征续接](tmp_analysis/feature_spacetime_handoff_20261024.md)与[full溯源](tmp_analysis/full_stage_handoff_20261023.md)保留。
- [方法调研笔记](report/OVMOT_方法调研笔记.md)。

冻结对照：发布 COVTrack R50 / ctao_public.pth / 官方完整 TAO validation / A 参数 .37 / 80 / 50 / .4。
发布 TCC Base/Novel TETA=39.554/34.205；普通 sumlog=39.774/35.771。
普通池化是控制实验，尚未构成新的机制方法；26/35 novel 类只有一个 GT 视频，完整 val 结果仍需范围与泛化核验。

后续20项固定几何分类回放与自然误判诊断显示：同身份匹配观测内存在重复同错类别集中，但当前固定appearance inverse-density读出在两mask下均被普通分数加权同时超过Base与Novel。停止该具体读出，继续核验base训练监督、词表和类别/视频留出，验证可学习共同误判过程。两主候选另做完整官方评估，计数/原始LocA与AssocA/输入输出哈希以阶段完成回执为准。

当前G条件矩候选在真实pair拟合前被精确反例证伪其混合conditioning部署公式，停止该具体公式及“内在协方差”新颖性归因；不外推所有关系网络失败。冻结协议、三轮每轮两项检查与决定回执保留。GT-free索引623,878边及全部3,947,118候选已独立复算；索引与身份支持均不等于方法有效。整体研究继续，尚无新方法TETA或新训练结果。

本仓库保存经过核验的研究脚本、设计文档及轻量指标/哈希回执。大型数据、权重、逐帧后验和预测保留在本地研究目录，不上传。
`evidence/` 是对应阶段 `results/` 中 JSON 回执的副本；其中绝对路径记录原始执行机器，不能据此假定这些文件已在仓库中。
当前本地工作目录为 `/home/chenhc/ov/OVMOT_RESEARCH`；仓库工作副本为 `/home/chenhc/ov/ovvvv`。阶段结束后同步已核验的脚本/文档并提交。
同步助手：`tmp_analysis/sync_research_stage.py`，按显式allowlist复制轻量文件，生成每文件SHA清单；Git提交和推送在检查diff后执行。原始结果中的绝对路径不随Git快照迁移。

上游：[COVTrack](https://github.com/zekunqian/COVTrack)，冻结源码 commit `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`。
遵守数据集及上游代码许可；本仓库未重新发布上游仓库、数据或 checkpoint。
