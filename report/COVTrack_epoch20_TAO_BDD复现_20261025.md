# COVTrack 20 epoch 复现：TAO val/test 与 BDD100K zero-shot 审计

日期：2026-10-25（2026-10-26 更新 BDD 正式协议）。本文只报告实际已完成的推理与评测，以及能够从冻结源码和原始论文核验的条件。BDD100K 未参与训练、微调或按 BDD 分数选参。

## 结论

TAO val 复现通过：本次训练的 `epoch_20.pth` 得到 Combined/Base/Novel TETA **38.828 / 39.443 / 34.239**；发布基线为 **39.554 / 34.205**，差值 **-0.111 / +0.034**。这是本次训练 checkpoint 的有效 TAO 复现结果。

BDD100K zero-shot 的 **COVTrack 正式结果**为 **42.745 TETA**（LocA 42.584、AssocA 37.353、ClsA 48.299），使用完整 200 视频、官方 BDD COCO 标注、DetPro BDD 8 类 prompt、`BDDVideoDataset`、COVTrack 自有 `OVTrackRoIHeadUncertainty + FeatureFusionModule + OVTrackerUncertainty`、`dynamic_rcnn_thre=True`、`.37/80/50/.4` 和 `exhaustive=False`。独立 OVTR BDD evaluator 对同一预测复核为 **42.703 / 42.304 / 37.510 / 48.295**，差异仅 0.042 TETA。论文 Table X 的 COVTrack 为 **45.8 / 44.1 / 39.4 / 53.9**，当前差 **3.055 TETA**，主要剩余差距为 LocA **1.516**、AssocA **2.047**、ClsA **5.601**。

另有一条容易混淆的 **43.675** 结果：它使用同一官方 GT/prompt、我们自己的 checkpoint 和 COVTrack 代码，但沿用 OVTrack/旧 TAO 的 `.0001/50/.5/10/.8` 推理参数；它是共享推理参数的对照，**不是 COVTrack 自有配置的正式复现结果**。师兄提到的 45.8 使用 COVTrack 方法的自有推理配置，不能用 OVTrack 的配置替代。

公开 COVTrack 仓库（`main`，HEAD `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`）没有 BDD 专用训练或评测脚本；其 BDD wrapper 明确写的是 `TETA(exhaustive=True)`。`exhaustive=False` 是原作者与师兄沟通后的评测口径，不是公开 README 的默认值；报告中必须保留这个 provenance，不能把 42.745 或对照 43.675 当作仓库默认 exhaustive BDD 数值。

## TAO val/test

checkpoint：

```text
/home/chenhc/ov/OVMOT_RESEARCH/results/covtrack_ctao_train20_bs24_20261022/epoch_20.pth
sHA256: 3205bfc9e6fd570532ff12ece58c2037a9f1c039daa09a5a37a3964dd37f3a44
```

训练于 2026-10-24 14:58:01 完成。TAO val 使用 `validation_ours_v1.json`；`results/covtrack_ctao_epoch20_valtest_20261024/val.completion.json` 记录实际 exit 0、PASS、8,158.2 秒（约 2.27 小时）。

| split | Combined TETA | Base TETA | Novel TETA | Combined LocA | Combined AssocA | Combined ClsA |
|---|---:|---:|---:|---:|---:|---:|
| val（本次） | 38.828 | 39.443 | 34.239 | 57.104 | 42.007 | 17.373 |
| Base（本次） | — | 39.443 | — | 57.009 | 42.033 | 19.287 |
| Novel（本次） | — | — | 34.239 | 57.808 | 41.811 | 3.097 |
| val（发布） | — | 39.554 | 34.205 | — | — | — |
| test（本次） | 不可计算 | — | — | — | — | — |

TAO test 的 `test_ours.json` 有 1,419 视频、52,155 帧，但 GT annotations 为 0，本地只能生成预测、不能算 TETA。test 预测在 `results/covtrack_ctao_epoch20_test_20261024_v2/`，不作为分数。首轮运行因源码硬编码 `filename.index('val')` 失败，修复在隔离源码副本中进行；修复后的预测产物与验证不改变 val 结果。

## BDD100K 数据与正式配置

官方 BDD COCO 标注（与师兄使用同一文件；200 视频、39,973 帧、465,512 标注）：

```text
/home/hub/ovtrack-official/data/bdd/annotations/box_track_20/bdd_box_track_val_cocofmt_siyuan.json
SHA256: 074ff79555483296cf7ccadddeeceeec7a83452c900506dc46588ed3a3e65d5d
```

DetPro BDD prompt（8 类，逐类使用训练学到的 context 生成，不是 LVIS bank 的前 8 行）：

```text
work/covtrack_epoch20_test_src_20261024/bdd_protocol_assets/detpro_prompt_bdd.pt
SHA256: 013a0ecbdc10cfb950a02eb42d89b14565de6c13ee7227a058c87c79899ab947
```

类别 ID 与名称：

| id | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| name | pedestrian | rider | car | truck | bus | train | motorcycle | bicycle |

隔离源码 `work/covtrack_epoch20_test_src_20261024` 使用同一 checkpoint。**COVTrack 正式配置**为 `tmp_configs/ovtrack_r50_bddmot_covtrack_bdd_20261026.py`：

```text
dataset: BDDVideoDataset
classes: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle
roi_head: OVTrackRoIHeadUncertainty
feature_fusion_head: FeatureFusionModule
tracker: OVTrackerUncertainty
dynamic_rcnn_thre=True
prompt_path=detpro_prompt_bdd.pt  # 8x512
test_cfg.rcnn: score_thr=0.0001, max_per_img=80
tracker: match_score_thr=0.37, memo_frames=50, momentum_embed=0.4
TETA: exhaustive=False
```

日志证据：`logs/covtrack_epoch20_bdd_covtrack_protocol_full_20261026.log` 记录 prompt 路径存在、`text features shape torch.Size([8, 512])` 及本地 `epoch_20.pth`；配置 SHA 与 checkpoint/prompt/GT 哈希见输出目录 `manifest.json`。

正式产物：

```text
results/covtrack_epoch20_bdd_covtrack_protocol_full_20261026/bdd_track_coco.json
results/covtrack_epoch20_bdd_covtrack_protocol_full_20261026/bdd_track_scalabel.json
results/covtrack_epoch20_bdd_covtrack_protocol_full_20261026/bdd_bbox.json
results/covtrack_epoch20_bdd_covtrack_protocol_full_20261026/manifest.json
results/covtrack_epoch20_bdd_covtrack_protocol_full_independent_20261026/result.json
```

官方 BDD wrapper 路径为 **42.745 / 42.584 / 37.353 / 48.299**；独立 OVTR BDD evaluator 为 **42.703 / 42.304 / 37.510 / 48.295**。差异来自两个 evaluator 对本地 cluster 的处理；正式比较保留各自 evaluator 标签，不混用。4 个没有任何轨迹的帧仍保留在 Scalabel 输出中，独立评测覆盖 39,973 帧。

## BDD 对照实验

所有对照均使用同一 checkpoint，10 视频切片用于快速排除配置方向；10 视频子集只作选择诊断，不能替代完整 val 分数。

| 实验 | 视频 | 主要配置 | TETA | LocA | AssocA | ClsA | 判断 |
|---|---:|---|---:|---:|---:|---:|---|
| 早期手工 GT/转换 | 200 | 自建 COCO GT、现场 CLIP 文本头、dynamic=True、默认 tracker | 37.621 | 39.421 | 37.547 | 35.896 | 非正式协议，已被正式结果取代 |
| 错误 custom 类名 | 200 | bdd 简写名、custom tracker | 27.414 | 9.633 | 29.093 | 43.517 | 无效，不能报告 |
| TAO 发布参数 | 10 | max80、`.37/50/.4` | 34.034 | 44.732 | 32.460 | 24.911 | 下降，排除 |
| LVIS DetPro 前 8 行 | 10 | `prompt_path` 前 8 行、max80 | 21.116 | 9.679 | 20.629 | 33.041 | 输入错配，排除 |
| 现场 CLIP 文本头 | 10 | 原生名、max80、dynamic=True | 21.117 | 9.679 | 20.629 | 33.041 | 文本头错配，排除 |
| 官方 BDD 协议 + OVTrack/旧 TAO 推理参数 | 10 | 官方 GT、DetPro BDD、`.0001/50/.5/10/.8`、False | 42.464 | 45.374 | 36.386 | 45.630 | 共享参数对照 |
| 官方 BDD 协议 + OVTrack/旧 TAO 推理参数 | 200 | 官方 GT、DetPro BDD、`.0001/50/.5/10/.8`、False | 43.675 | 41.019 | 39.191 | 50.814 | 已知对照，不能报作 COVTrack 自有配置 |
| COVTrack 自有配置 | 10 | 官方 GT、DetPro BDD、`.37/80/50/.4`、False | 42.130 | 47.502 | 33.909 | 44.980 | 快速验证通过 |
| **COVTrack 自有配置** | **200** | 官方 GT、DetPro BDD、`.37/80/50/.4`、False | **42.745** | **42.584** | **37.353** | **48.299** | **COVTrack 正式结果** |

前几轮的 37.621 不能解释为模型本身的 BDD 能力，因为同时叠加了自建 GT、错误文本头、动态阈值和不同包装器。改用官方 BDD 输入后，即使沿用 OVTrack/旧 TAO 推理参数也达到 43.675；把阈值恢复为 COVTrack 自有配置后，为 42.745。两个编号回答的是不同问题，不能将较高的 43.675 写成 COVTrack 自有配置的正式结果。

## 已排除的输入/评测错误

- 类别顺序：早期 `TaoDataset` 配置使用 LVIS 风格的简写类名，`cat_ids` 为 `[2,3,4,5,6,7,8]`；它的 `person` 在 LVIS 风格转换文件上能找到对应项，但 `pedestrian/rider` 的语义链与官方 `BDDVideoDataset` 的 `[1..8]` 并不等价。正式配置改用官方 `BDDVideoDataset` 和原生 BDD 类名，避免这条歧义路径。
- 评测上限：早期结果每帧最多 47 条预测，截断前后分数一致；COVTrack 正式配置将 `max_per_img` 设为 80，完整 200 视频保存了 1,208,126 条轨迹记录，最终 TETA 由官方 wrapper 与独立 evaluator 双重复算。
- 动态 R-CNN 分数阈值：COVTrack 自有配置保留 `dynamic_rcnn_thre=True`，8 类下自动阈值约 0.111；OVTrack 式对照关闭该动态阈值并使用 0.0001，两者分数不可当作同一推理配置。
- 推理实际生成：普通 GT 446,840，预测 1,019,664 条，轨迹约 200,213 条；类别无关 IoU@0.5 召回约 59.4%、精度约 26.3%，匹配框 IoU 中位 0.818。同帧 IoU≥0.5 重复框为 0 对。问题指向候选数量/关联碎片化，并非简单同帧 NMS。
- 轨道合并：相邻帧、同类、IoU≥0.5 仅 84 对，10 视频 TETA 36.404 → 36.367，下降，停止该方向。
- 官方权重 `ctao_public.pth` 在早期 10 视频配置下为 36.055，不代表它们在正式协议下的差异；该对比使用错误类名和 prompt，因此不能用于判断模型权重优劣。

## 与论文条件和公开代码的差异

COVTrack++ 原文 Table X 报告 COVTrack **45.8 / 44.1 / 39.4 / 53.9**，并说明 zero-shot 使用 TAO 训练模型直接跑 BDD100K val，不用 BDD100K subset，不做 fine-tune 或 domain adaptation；调参只用 TAO val，BDD 不参与选择。原文还说明 detector/NMS 设置沿用检测器/OVTrack 设定。

OVTrack 原文补充材料写的是：BDD 实验使用视频中出现的类别作为 text prompts；TETA 推理使用 `|P|=50` 且类别无关 NMS。公开 COVTrack 代码没有发布 BDD 专用配置，也没有公开“按视频类别构造 prompts”的转换脚本；本次正式配置使用官方 BDD 8 类 prompt 和同一官方 GT，并在原代码 BDD wrapper 中只改 `exhaustive=False`。这个 False 口径来自原作者沟通，公开仓库默认仍是 True。

公开 COVTrack tree 中没有 `scripts/evaluate_bdd100k.sh` 或 `evaluate_bdd1000.sh`；GitHub API 递归树也仅返回 BDD dataset wrapper、`tools/to_bdd100k.py` 和 `ovtrack/core/to_bdd100k`。因此不把外部二手脚本当成官方协议，也不把公开默认的 True 分数与沟通后的 False 分数混报。

## 当前判断与后续

已完成 TAO val 的有效复现，以及 BDD 完整 val 的可追溯 zero-shot 结果。COVTrack 自有配置的正式 BDD 距论文差 3.055 TETA；早期基于 37.621 的“训练不足/协议尚缺根因”判断由错误输入组合造成，应废弃。官方 BDD 协议下沿用 OVTrack/旧 TAO 参数的 43.675 是共享参数对照，但不是 COVTrack 自有配置的正式复现。

当前可复核的结论以 COVTrack 自有配置的正式协议为准。残差 3.055 不复用于 BDD 调参或模型选择；若继续缩小，应单独预注册“检测候选/分类头/关联器”的比较，并保持 TAO 选参和 BDD 不参与选择的边界。

## 产物与校验

| artifact | sha256 |
|---|---|
| `epoch_20.pth` | `3205bfc9e6fd570532ff12ece58c2037a9f1c039daa09a5a37a3964dd37f3a44` |
| COVTrack 正式 BDD COCO 预测 | `3002e5c19ff5063e3a7d327423937012cd333a2a5cc0485b80f7298132d39c25` |
| COVTrack 正式 BDD Scalabel 预测 | `4e0497dbe61e41a0ead5066e4f5efdd7fe66e98392038cb04251c9e265794892` |
| COVTrack 正式 BDD 检测输出 | `1fee4833794c6dcb5c521a9e7e58256b4a15e77d16642de14f06d4808cf37228` |
| COVTrack 正式配置 | `6dbd8b14ffb97e78cb50c918dfba3136ae4b7d19b19a08ae632d1c6b8e09c21e` |
| COVTrack 独立 BDD TETA 回执 | `04d469849447f1151dfdfd372ed7754439b0157bb34d29c927b7e870a119757d` |
| 完整 val `tao_track.json` | `3512db84bfdf129fc33d4be18f3982f3a34f2a8417f51169b675aa06e670a430` |
| 完整 val 独立 TETA 回执 | `d056693ae4557faa5f803935895eadda9e56136fc3393353e3d74f5c9f4fb4b4` |
| 10 视频 A 参数预测 | `9d9a2061e8afd88fb0d8218c3eb73aefdcfe84ba63fdeeb2fbf06c07abedeb05` |
| 10 视频自定义 8 类预测 | `bbe191c796cc1fc1af71f044311502da302f6baa786976b0f471e9d3618df01e` |
| 自定义 8 类独立 TETA 回执 | `17c9a90925c3873dccfaa314abbf1b429519fafefe24edfbae7809d3c3eca3fd` |

COVTrack 正式 BDD 独立评测回执：`results/covtrack_epoch20_bdd_covtrack_protocol_full_independent_20261026/result.json`。
