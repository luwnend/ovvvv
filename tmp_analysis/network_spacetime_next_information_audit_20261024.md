# 时空信息改善网络特征：实际入口与缺失约束审计

状态：`READ_ONLY_CODE_AUDIT_NO_NEW_RESULTS_READ`。2026-10-24。只写本说明；未修改冻结脚本/缓存、未启动计算、未读取新主 OOF 或真实 donor-out 结果。本结论不依赖它们是否有效。

**当前没有足够证据提出一个可直接开跑的新时空特征机制。** 现有向量缓存支持普通 identity metric adaptation 和关联证据对照；若要解释为“用历史可见细节修复当前实例特征”，缺的是局部对应/可见性观测，以及区分实例细节与类别、坐标、背景的训练目标。给现有向量再加 attention 或残差门控不会自动补上这些信息。这里不声称 raw256 完全不含可学习的时变信息，也不声称所有表征改进都必须使用显式对应。

## 1. 实际执行路径

科学 work copy 根为 `work/covtrack_base_error_20261023/ovtrack/models/`。冻结配置 `results/base_error_full_collection_20261023/full_collect_v1/collection_config.py:40` 选择 `OVTrackRoIHeadUncertainty`，tracker 为 `OVTrackerUncertainty`，`confused_features=True`、`momentum_embed=0.4`。

| 入口 | 核实的数据流 | 对“时空特征改善”的含义 |
|---|---|---|
| `mot/ovtrack.py:180`；`roi_heads/ovtrack_roi_head.py:2872` | 当前图像 backbone/FPN → ROI `[N,256,7,7]` → track head。`track_heads/quasi_dense_embed_head.py:101` 为四层 conv 后 flatten、FC 和 256 维投影；不是显式时序提取器。 | ROI 空间细节在进入 tracker 前已被压缩；缓存 raw256 是此 head 输出。 |
| `roi_heads/ovtrack_roi_head.py:1393` | MCF 将绝对 xyxy、平均 semantic512 分别投影，和 raw track feature 经过 gate/residual；输出 `raw + residual * max_ratio*(1-assoc_conf)^2`。 | 已有可学习跨 cue 残差；输入没有 dt、局部空间 token、可见性或运动对齐对应。换门控名字不构成新信息。 |
| 同文件 `:2488`、`:2980`；`trackers/ovtracker.py:505` | 训练从 raw key/ref 算三种 cycle confidence，detach 后双侧融合；推理 ROI 返回 raw，tracker 用 current raw 对历史 fused EMA 算 confidence，再仅融合 current。 | 存在可核实的训练/推理数据流差异；仅凭源码不能认定它造成了性能问题。 |
| `trackers/ovtracker.py:405`、`:640`、`:690` | fused current 用于匹配，再以 `0.6*old+0.4*current` 写回 identity memory；semantic 保存最近一次，框也取最近一次。 | 历史不是一个未经融合的干净视觉目标。错误关联/修正可能反馈进入 memory，但真实污染程度尚未测量。离线 pair 缓存不提供改 assignment 后的完整反事实 memory。 |
| `losses/cyc_loss.py:102` | cue 归一化相似度经双向 row-softmax，取往返矩阵对角。 | 这是候选集合相关的返回概率；1×1 时无论绝对 cosine 都返回 1。它不提供独立的同身份、可见性或物理运动真值。绝对 xyxy cosine 也不具公共平移不变性。 |

同一 ROI 文件的 `OVTrackRoISLAck`（`:3533`）包含另一套 `FeatureInteractionModule`（`:5697`）；冻结配置未选择它。其训练作 key/ref cross-attention，所见推理入口 `:5000` 用 current/current；模块计算的 self-attention 输出未进入后续 concat。这不能作为当前采集使用历史空间 attention 的证据。

COVTrack++ 的已核对论文快照 `covtrackpp_live_sources_20261023/S4.SS2.txt`、`S4.SS3.txt` 分别描述 MCF 和 MGA：后者是同帧 IoC containment、TAQ 与 parent-query/child-key-value 聚合，不是跨时刻同物体局部对应。所审公开代码树未独立定位 ++ MGA/TCP 实现；旧 MCF 平方系数也不应与 ++ 论文线性式宣称完全一致。代码/论文边界沿用 `covtrackpp_code_boundary_20261023.md`。

## 2. 已停止控制能排除什么

沿用 `spacetime_next_mechanism_design_audit_20261024.md` 的历史读出：same-predicted-category/common-CV/合法 known P+N 中，lag1 CV repair/break 为 **26/289**，lag4 为 **60/533**。角度外推即使限于“所有已知 positive history 同 GT”层，仍分别为 **5/98**、**30/1138**。

因此不能再把“全局 embedding 沿历史方向确定性外推”当作已有支持的模型，也不能把角度失败完全归因于这些已知 positive history 的 ID 污染。这些结果不解释像素 CV 失败的唯一原因，不证明所有历史干净，更不能否定有监督的局部时序学习。

当前缓存有 raw256、平均 semantic512、框/时间/proposal 和 posterior；没有分开的 text/image512、投影前特征、ROI 空间张量、背景对应/flow、pose/depth 或 visibility。平均语义不能唯一还原两分支；压缩后的 raw256 没有唯一恢复原 ROI 细节的依据。离线向量训练也没有原 backbone/ROI/head 的计算图。现有 identity adapter × geometry 设计已经是普通对照，本审计不将其再次包装成新机制。

## 3. 一个条件候选及其可反驳条件

若后续明确要研究“特征修复”，可具体化为**跨时刻可见局部特征补全**。它属于已有视频特征聚合/补全范畴；本次没有建立新颖性或有效性。

| 要素 | 必须写清的契约 |
|---|---|
| 输入 | 当前被遮挡的图像前向所得 ROI spatial tokens；过去时刻的候选 ROI tokens；独立的合法对应/warp、有效可见区域与原帧 gap。过去支持的部署选择必须因果且 GT-free。GT 身份只能用于合法训练，不能纯化推理 memory。 |
| 目标 | 对训练中原本可见、被人工遮住的当前区域，预测 frozen teacher 在未遮挡当前图像上的局部特征；另有合法同身份监督与同类不同身份 hard negatives。真实已遮挡区域没有同帧可见 teacher 真值，不能借此声称重建真实隐藏外观。 |
| 防止直接复制 | student 在图像输入处遮挡并重新前向，阻断目标区域经 backbone 感受野、skip、semantic、对应估计器或 visibility 旁路进入。仅在卷积后将几个 token 置零、或只对 teacher stop-gradient，均不充分。teacher 目标不得作为 predictor 输入。 |
| 防止错误解释 | 用同类合法负例约束实例区分；坐标/背景干预检查捷径；固定错误身份/错误时间支持检查历史依赖。它们是可反驳检查，不是消除全部捷径的证明。局部重建降低而身份检索不变，只说明重建目标改善。 |
| 最强普通对照 | 同容量、同 teacher/身份损失/训练预算的当前帧 masked reconstruction；相同历史和对应的 **warp-and-average/EMA**；同预算未对齐 temporal pooling。各臂共享支持数和实际推理选择器。仅胜过 raw256 不足；只胜未对齐平均而未胜 warp-average，证据落在对齐而非学习修复。 |
| 与 COVTrack/++ 的可描述区别 | 操作发生在 flatten/投影前，以同物体跨时刻局部可见对应为传输单位；MCF 是全局向量跨 cue 残差，论文 MGA 是同帧 containment 聚合。这个差异只定义测试对象，不自动提供方法新颖性。 |

成功读出必须包含关闭几何/类别关联加分后的纯 appearance 身份检索，保持合法 P/N、source 隔离及相同读出人口；否则只支持关联偏置或 memory 效果。跨源 token 重建与 identity 读出都需报告。在线跟踪仍需独立闭环验证，不能由 pair 排名或重建 loss 替代。

**当前信息缺口：** 缓存不具备上述空间 token、独立对应/可见性与受控 masked-image 前向，故这个候选不能用现有向量实现。后续进入该候选时先补齐这些观测和强对照；当前两项冻结实验继续按原计划运行。本报告是代码判断与待验证设计，没有新采集、训练或机制收益。
