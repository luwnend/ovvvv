# 时序局部补全、masked teacher 与身份损失：近邻查重

2026-10-24。状态：`STOP_NOVELTY_GENERIC_TEMPORAL_COMPLETION_TEACHER_ID`。比较对象为 `network_spacetime_next_information_audit_20261024.md` 中的条件候选；读取 SHA256 为 `199b771851d8584623f49824316e4459afe45c8cd055250d6bfc7fa73310b123`。先检查本地已有调研，再只选下列 **4 篇**原始工作，未扩展候选网络。来源、版本与公开代码定位见 `temporal_roi_completion_sources_20261024/source_manifest.json`。

**当前描述属于已有路线的组合，不能据此主张新方法。** VRSTC 已直接包含跨帧局部补全、人工遮挡训练和身份损失；FGFA 已有显式运动对齐后的空间特征聚合；DropMAE 已指出并干预 masked reconstruction 依赖同帧捷径的问题；MVD 已用冻结 image/video teacher 的高层特征作为 masked prediction 目标。换成 ROI token、限制 past-only、将像素目标换成 teacher feature，尚未给出超出这些近邻的独立机制。

这是有范围的碰撞判断，不是声称四篇中某一篇逐式实现了当前全部契约，也不是穷尽文献后的首创性排除。

## 1. 原始方法与普通对照

| 原始工作与已读范围 | 输入 → 目标；局部对齐与时间方向 | 已有普通对照与对当前候选的判定 |
|---|---|---|
| **VRSTC / STCnet，CVPR 2019**，[原文](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_VRSTC_Occlusion-Free_Video_Person_Re-Identification_CVPR_2019_paper.pdf)，Secs.3.1–3.6、4.1、5.2–5.3.1/Table1 | 人工遮挡当前行人图像 → spatial generator 粗补全 → 利用 previous/next 局部特征细化。Sec.3.3 用 3×3 patch 的 cosine、softmax 和邻帧 patch 加权合成；原始未遮挡当前图像提供 L1 目标，local/global discriminator 约束生成，冻结预训练 ID guider 的身份 CE 保持身份。生成后再训练 ReID。部署遮挡检测还使用整段视频的 region mean，不是 past-only。 | Table1 已比较无补全 NL、Spa、参数匹配的当前帧 Spa+AE、邻帧但无 attention 的 Spa+TAE、Spa+Tem，以及逐步添加 discriminator/ID guider。**补全＋局部时序匹配＋身份约束，连同容量/无匹配控制均已有。** 当前候选的过去支持、teacher-token 目标不是逐式同一，但不自动建立新颖性；其局部匹配/启发式遮挡分数也不是独立可见性真值。 |
| **FGFA，ICCV 2017**，[原文](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Flow-Guided_Feature_Aggregation_ICCV_2017_paper.pdf)，Sec.3、Sec.4.2/Table1、Algorithm1 | 邻帧 dense feature maps 经 FlowNet flow、bilinear warp 到当前位置，再以 embedding cosine 的逐位置时间 softmax 加权；检测监督端到端训练。默认窗口 `i−K…i+K`，含未来帧。没有 masked teacher 或身份目标。 | Table1 分别检查单帧、无 warp 的均匀平均、无 warp 的 adaptive weighting、warp＋adaptive weighting、冻结部分网络。**对齐后补入时序可见特征是已有方法**；应作为相同支持/相同 warp 下的强普通控制，不能只比较 raw256。 |
| **DropMAE，CVPR 2023 会议版**，[原文](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_DropMAE_Masked_Autoencoders_With_Spatial-Attention_Dropout_for_Tracking_Tasks_CVPR_2023_paper.pdf)，Secs.3.1–3.4、4.2、6 | 同视频两帧 patch 拼接、随机 mask；encoder 只看 visible tokens，decoder 预测 masked pixels。ASAD 根据跨帧匹配强度采样删除同帧 attention，迫使模型利用另一帧线索；没有外部 warp、可见性标签或 ReID loss。预训练重建双帧，不是部署时 past-only predictor。 | 比较 ImageNet MAE、双帧 TwinMAE、DropMAE，并检查 dropout 比例及 encoder/decoder 施加位置。**“不能让当前空间捷径代替时序补全”已有明确问题和训练干预。** 当前候选需在相同 masking/支持预算下比较，不能将遮挡操作本身称为新约束。 |
| **MVD，CVPR 2023**，[原文](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Masked_Video_Distillation_Rethinking_Masked_Feature_Modeling_for_Self-Supervised_Video_CVPR_2023_paper.pdf)，Secs.3.1–3.3/Algorithm1、4.3/Tables7–9 | student 从 masked video 的 visible tokens 预测缺失位置的高层特征；固定 image teacher 对各未遮挡帧编码，固定 video teacher 对未遮挡 clip 编码，两 decoder 分别以 Smooth L1 匹配 target。tube masking 沿时间共享空间 mask；没有 identity loss、独立 ROI 对应或过去支持选择契约。 | 比较 image/video/co-teaching、固定/EMA teacher、附加 pixel loss，以及 full per-token distillation 与 masked reconstruction。**冻结未遮挡教师提供 masked feature target 已有直接先例。** 将其接到 ROI/身份监督上是任务组合，须另证独立技术贡献。 |

本地预检使用 `appearance_transport_literature_audit_20261024.md`、`novelty_audit_20261023.md` 及相关设计/代码边界说明。已有历史全局 embedding 预测、普通质量聚合等记录不能替代上述局部补全查重；本报告也未重新追逐其未取得的全文或另开宽泛检索。

## 2. 公开实现核对边界

本次只读实现，没有安装、执行或复现实验。

- **FGFA**：官方仓库 commit `f01243e23a2e2be0a91d0a5a927433686305a054`。[网络代码](https://github.com/msracver/Flow-Guided-Feature-Aggregation/blob/f01243e23a2e2be0a91d0a5a927433686305a054/fgfa_rfcn/symbols/resnet_v1_101_flownet_rfcn.py#L889) 实现 `GridGenerator/BilinearSampler`；`:1062–1081` 在缓存窗口上 warp、时间 softmax、加权求和。由原始作者仓库确认，不依赖第三方重写。
- **DropMAE**：官方仓库 commit `4de150de7ea56623b5d4fdc1cd23019238dc4c27`。[models_mae.py](https://github.com/jimmy-dq/DropMAE/blob/4de150de7ea56623b5d4fdc1cd23019238dc4c27/models_mae.py#L26) 的 `mask_drop_out_dec` 计算跨帧匹配强度并采样同帧 attention；`:166–170` 在 softmax 前置 `−inf`；`:412–438` 对两帧 masked pixels 算 reconstruction loss。使用 CVPR 2023 PDF，未混用 arXiv 后来的改题版本。
- **MVD**：官方仓库 commit `45b877a6d18202a81528eeb31ff2da2c0e9ef2ea`。[engine_for_pretraining.py](https://github.com/ruiwang2021/mvd/blob/45b877a6d18202a81528eeb31ff2da2c0e9ef2ea/engine_for_pretraining.py#L54) 的 teacher 分支在 `no_grad`、`eval` 下编码未遮挡输入；`:79–93` 仅取 masked positions 作两种 teacher target。默认 distillation loss 在 `run_mvd_pretraining.py:42` 为 SmoothL1。
- **VRSTC**：上述方法判断来自原始论文。有限的作者/标题/代码检索未独立确认对应公开实现；不能将其他任务的同名 `STCNet` 仓库配给它，也不据未检索到断言代码不存在。

## 3. 仅剩一个可证伪的任务缺口

本次四篇的所读部分没有核实到以下完整部署对象：**past-only、真实 GT-free 支持选择、可能错链的在线历史，在相同标注和支持预算下，修复后的纯 appearance 对同类不同身份仍有额外区分力。** 这是尚未验证的任务缺口，不能用“visibility-aware/identity-preserving”改名充当实现或创新。

最小问题保持同一 ROI/token、teacher、mask、合法身份标签、因果支持选择器与训练预算，比较：当前帧 masked reconstruction、相同历史/对应的 warp-and-average，以及学习补全。再对预先固定、具有合法标签的支持做错误身份/错误时间干预，保持数量、gap 和可见区域预算；unknown 不充当负类，GT 不进入部署选择器。

学习补全须同时表现为：真实对齐支持带来 held-source 同类身份检索增量；失配支持下没有仅因借入他人细节而改善重建、损坏身份区分。身份读出关闭几何和类别关联加分，人口及 hard negatives 固定。若只胜未对齐平均，证据属于对齐；若只降低 reconstruction loss，证据属于重建；若不胜相同信息的 warp-average，则尚无学习修复必要性的证据。单纯忽略全部历史虽可抗污染，也不能满足“使用历史带来增量”。这些都是普通、可反驳的控制要求，本次没有生成新的方法或收益结论。

当前缓存缺 ROI spatial tokens、独立对应/可见性和受控 masked-image 前向的事实不变。已有冻结实验按各自协议继续；本报告未读新 OOF/donor-out 评价、未改冻结代码/缓存、未启动新采集或训练。
