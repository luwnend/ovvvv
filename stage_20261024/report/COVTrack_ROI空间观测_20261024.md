# ROI空间观测：采集与映射能力验收

2026-10-24；**四帧实际采集及独立验收通过**。本项为普通观测工具，没有训练或新方法效果。Python为`/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python`，实际外层wait exit0、25.9168秒，timeout300秒未触发。GPU启动前owner/显存检查符合授权，未改动既有chenhc训练。

## 实际采到什么

`OVTrackRoIHeadUncertainty._track_forward`的ROI `[N_pre,256,7,7]`→4ConvModule→CHW flatten→FC ReLU→fc_embed256。hook保存ROI输入、最后完整ConvModule输出与raw；均立即detach.clone，原生全batch/device/dtype重放FC，torch.equal验证native raw。实际框按scale_factor进入模型坐标；pretracker_det_row/keep映射决定retained顺序。RPN posterior的proposal row仅是locator，不能作为track ROI行索引。

| source / video | prefix帧 / image | N_pre | retained | 原帧号 |
|---|---|---:|---:|---:|
| ArgoVerse / 1271 | 0 / 214313 | 69 | 50 | 150 |
| ArgoVerse / 1271 | 1 / 214314 | 80 | 54 | 240 |
| BDD / 801 | 0 / 148612 | 80 | 69 | 0 |
| BDD / 801 | 1 / 148642 | 80 | 68 | 30 |

选帧仅按sealed FIT source/video metadata的字典序首项，首两个dataset block各两帧；未按身份、类别、visibility或收益挑选。共241行，保存ROI/conv各float32 `[N,256,7,7]`、raw `[N,256]`以及两空间框与不可变row/proposal映射。

## 验收范围

独立离线审计实际exit0/4.71秒：24协议pins、71原模型源码hash、7保护状态及4NPZ逐文件SHA一致，保存pretracker row=parent keep非零位置，proposal/原xyxy精确，已存raw→float16 uint16位与parent cache全等。runtime原生全batch FC重放由冻结代码、运行receipt和actualexit链支持，独审没有重跑模型/FC。**保存retained subset网格的单独FC重放仍未验证**，不承诺改变GEMM批量后末位严格相等。

协议`tmp_analysis/roi_spatial_observation_protocol_20261024.json` SHA `ea49049e…`；collector SHA `ad7c3948…`，trace SHA `e9fea532…`，三轮各两项合成检查均exit0。输出`results/roi_spatial_observation_20261024_v1`，执行`results/roi_spatial_observation_execution_20261024_v1/actual_exit.json`；独立收据`tmp_analysis/roi_spatial_observation_independent_completion_audit_20261024.json` SHA `e3400041…`。

## 能支持的下一步

空间token现可从冻结网络真实采集；没有证明它们比raw256更可辨、等于独立可见局部patch、提供真实对应/相机/visibility，或带来身份/跟踪/泛化/创新收益。原历史完整运行的prompt/image/inputtensor及所有数值后端flags未封存，不能将重放不一致直接归为机制失败。

若重启局部时序学习，须另补合法past-only对应、受控图像遮挡前向与相同信息/预算warp-average等强对照。VRSTC/FGFA/DropMAE/MVD已覆盖泛化补全/teacher/ID组合，本次collector不解除该新颖性STOP。大批采集应先冻结存储和资源协议，当前不复制全量数据或训练网络。
