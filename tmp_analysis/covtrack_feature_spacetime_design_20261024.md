# 可测的特征与相对时空接口

当前五模式仅干预推理记忆的特征状态：首要 `raw_confidence` 保留原 fused
评分和 EMA，用独立 raw256 EMA 计算 appearance cycle confidence。其余四模式
是固定对照。它们没有改善 backbone、ROI/head 表征，也没有重新训练网络。
`relative_track_geometry.py` 目前仅实现并通过相对几何的 CPU 单元验证，尚未
估计真实相机运动、接入评分或进行网络学习。不能从这些实现声称有效或新颖。

## 已有缓存能做什么

冻结缓存保留 raw identity embedding `[N,256]`、平均 semantic embedding
`[N,512]`、box/time、完整词表三个 posterior 分支。identity/semantic embedding
为 float16；不能称 CPU 缓存重放与原 float32/GPU 精确等同。完整每帧 retained
行（含原 `id=-1`）和空帧必须保留，先报告 released 重放分区/ID 的首次分歧。

先用现有缓存检验五模式的 raw/fused 状态一致性，并独立检查相对几何张量的
数值、缺失历史覆盖率、时间间隔分布及与 raw appearance 的可区分性。按 source
隔离，仅用合法 FIT manual identities 解释新预测；不复用旧轨迹 pair 标签评价
新 ID。当前单元测试不需要新特征采集或训练。相对几何不应直接按任意固定权重
加到 appearance score 后宣称收益。

分开的 text/image region512、投影前 ROI/region1024、空间 ROI tensor 没有单独
缓存。平均 semantic512 不能唯一恢复两个分支，posterior 也不能恢复被投影/归一
化丢弃的分量。要研究分支互补、区域细节、遮挡/部位或重新训练 ROI/head，必须
新建带明确 row/proposal 对齐的独立缓存或重新前向；不得覆盖现冻结缓存。

## 下一个 pair 网络的明确输入

设当前 query 数为 `Q`、活动 memory 数为 `M`。输入必须是相对 edge 张量，
不把绝对位置硬拼到 identity node embedding。建议接口为：

```text
raw_query:        float32 [Q,256]
raw_memory:       float32 [M,256]     # 未经 context fusion 的历史 raw EMA
semantic_query:   float32 [Q,512]     # 单独冻结的语义 cue
semantic_memory:  float32 [M,512]     # 按已声明的 last-observation 规则
geometry:         float32 [Q,M,8]
geometry_valid:   bool    [Q,M,8]
history_valid:    bool    [M]
camera_valid:     bool    [M]         # 几何模块当前采用整次调用统一是否供 H
```

`geometry[...,0:8]` 固定含：归一化相对位移 `dx,dy`、`log(wq/wm),log(hq/hm)`、
`log1p(original_frame_gap)`、归一化 CV innovation `ex,ey`、CV-predicted-box IoU。
位移和 size 比例相对 camera-warped memory；归一化 scale 为 query/memory 宽高
逐轴几何平均。时间始终为原视频帧号差，不能用 sampled-step 差冒充秒数；没有
FPS 时不推断秒。CV 速度使用 previous→memory 的实际原帧间隔，每个 memory
单独计算。输入网络前对无效 innovation/CV-IoU 填零并提供逐维 mask；零是填充
值，不能表示观测到了零速度。`prediction_type='static_reference'` 的 IoU 可以
另名作为 static overlap cue，不能混称有效 CV-IoU。标准化统计仅由训练源计算，
不以某个任意截断替代缺失/异常检查。

appearance 另算原始 dot product、cosine 及两侧范数；dot 的尺度与 norm 有关，
不能把其变化全部解释成身份变化。semantic cosine/posterior disagreement 是另
一组语义 cue，不把 CLIP 类别投影直接改作 identity embedding。候选集合歧义
（top1-top2 margin、row/column softmax、cycle consistency）单独保留为 context
输入；它依赖 `Q,M` 和其他候选，区别于某条 edge 的绝对 dot/cosine，不用它
替代 pair appearance。未来 pair 网络可输出一个单独的 association logit，
并与仅 raw appearance、appearance+ambiguity、appearance+relative geometry
等相同监督预算的对照比较。当前没有实现或训练该网络。

相机 H 由外部 GT-free 图像估计器提供，方向必须是各 memory_frame→query_frame。
只有所有 memory 在同一原帧时才能共享一个 `[3,3]` H；不同帧用 `[M,3,3]`。
若速度称已补偿，还必须提供逐 memory 的 previous→memory H。没有该历史 H
时不能把原像素位移当作补偿速度。单应变换只建模平面/主导相机运动；其 bbox
角点包络和恒速外推是模型假设，不能解释为真实物体动力学。无 H 的结果明确为
未补偿模式。在线用途必须遵守当前 query 前的历史，不使用未来帧估计相机或速度。

## 真正改变 identity 表征的有限路径

最小可学习路径是在 raw256 后放共享的残差 identity adapter，例如
`z=normalize(x + W2 GELU(W1 x))`，256→64→256，初始化为恒等。缓存能训练的
只有 adapter、pair association head 和显式新 geometry/ambiguity encoder；
backbone、ROI extractor、track-head conv/fc、CLIP projection 没有可回传的缓存
计算图，不能通过离线向量训练更新它们。若需要更新 track-head 本身，必须另行
配置真实 ROI 输入和 checkpoint，从明确参数名单开始重训。

监督仅使用可靠 same-identity positives，包括预先固定的跨 lag/view 正例；
hard negatives 优先同类 different-identity，让模型学习实例区分而非类别捷径。
不同身份不能从 unmatched/unknown 或预测轨迹不纯推导。稀疏已知 pair 的源/轨迹
集中度和同类 hard-negative 支持必须先检查，不能因为已有 819 条 different-GT
边就启动训练。图像增强产生 view 变换正例需要新 ROI/图像前向；缓存只能利用
已经存在的不同观测，不能伪造未缓存的视觉变化。优化过程中冻结 semantic512
和类别 readout；语义是独立提示而不是 identity loss 的无约束梯度通路。

训练新的 edge encoder/pair logit 必须有对应身份监督、无匹配目标和按 source
隔离的验证；原 `FeatureFusionModule` 的 bbox_transform 输入 `[N,4]` 是 node
绝对框，不能不重训就替换成 `[Q,M,8]` edge。若重构或微调它的 gate/residual
层，相关新 projection、gate、residual 和 association loss 都要明确训练；保留
原完整 readout/语义分支作为冻结控制。既有 MCF 已有门控，普通 adapter、对比
学习和相对几何融合本身都不是新颖性依据，验证前不启动或称有效。

## 源码可核实的梯度边界

当前科学 work copy 的 `mot/ovtrack.py:_freeze_detector` 只枚举 backbone、
neck、RPN、roi_head.bbox_head。`roi_head.projection` 和
`projection_for_image` 不在这份显式列表中；
`get_classification_head_features` 在 projection 后生成 semantic cue，训练
fusion 路径传入该 cue，`FeatureFusionModule` 对 assoc feature 可选 detach，
对 cls cue 没有对应显式 detach。因此存在 association loss 到这些 projection
的静态代码路径。没有检查另一会话 live optimizer 的 parameter groups、实际
requires_grad、执行分支和梯度，不能据此断言现训练发生了语义漂移；本任务没有
触碰该训练。后续新实验若要求语义冻结，应明确冻结/断开这两层并实际验证参数
与梯度名单，而不能只依赖 `freeze_detector=True` 名称。

几何独立复核以最终 `relative_track_geometry.py` SHA
`c5a085f8c56da990bc7f1cb2bedc7116c7ba32223e279f4ef678ed7447c4064a`
为准，收据为 `relative_track_geometry_independent_review_20261024_v2.json`。
先前 singular-H 和 projective-scale 两个反例保留在旧 review 中，已在最终实现
修复并重新通过独立合成检查。相机估计器精度和真实 tracking/network 效果仍未验证。
