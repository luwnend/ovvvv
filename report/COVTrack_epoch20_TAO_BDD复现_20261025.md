# COVTrack 20 epoch 复现：TAO val/test 与 BDD100K zero-shot 审计

日期：2026-10-25。本文只报告实际已完成的推理与评测，以及能够从冻结源码和原始论文核验的条件。BDD100K 未参与训练、微调或按 BDD 分数选参。

## 结论

TAO val 复现通过：本次训练的 `epoch_20.pth` 得到 Combined/Base/Novel TETA **38.828 / 39.443 / 34.239**；发布基线为 **39.554 / 34.205**，差值 **-0.111 / +0.034**。这是本次训练 checkpoint 的有效 TAO 复现结果。

BDD100K zero-shot 的最佳已验证结果为 **37.621 TETA**（LocA 39.421、AssocA 37.547、ClsA 35.896），使用完整 200 视频、模型原生 8 类映射、`max_per_img=50` 和训练默认 tracker 参数。独立 OVTR BDD evaluator 对同一预测复核为 **37.582**。论文 Table X 的 COVTrack 为 **45.8 TETA**（LocA 44.1、AssocA 39.4、ClsA 53.9），当前仍有 **8.179** 的总差距，主要来自 LocA 与 ClsA。因此 BDD 行尚未完成同条件复现；不能把 37.621 写成论文 45.8 的复现值。

公开 COVTrack 仓库（`main`，HEAD `9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`）没有 BDD 专用训练或评测脚本；远端树只包含 TAO 配置、BDD dataset wrapper 和转换工具。README 明确指出 COCO 格式推理后还需要转成 BDD100K 格式评测，但没有发布这个实验的 class prompts、BDD 输入转换设置或精确推理命令。作者代码与原论文的完整 BDD 协议间仍有未说明的环节。

## TAO val/test

checkpoint：

```text
/home/chenhc/ov/OVMOT_RESEARCH/results/covtrack_ctao_train20_bs24_20261022/epoch_20.pth
sHA256: 3205bfc9e6fd570532ff12ece58c2037a9f1c039daa09a5a37a3964dd37f3a44
```

训练于 2026-10-24 14:58:01 完成。TAO val 使用 `validation_ours_v1.json`；`results/covtrack_ctao_epoch20_valtest_20261024/val.completion.json` 记录实际 exit 0、PASS、8,158.2 秒（约 2.27 小时）。

| split | Combined TETA | Base TETA | Novel TETA | LocA | AssocA | ClsA |
|---|---:|---:|---:|---:|---:|---:|
| val（本次） | 38.828 | 39.443 | 34.239 | 57.009 | 42.033 | 19.287 |
| val（发布） | — | 39.554 | 34.205 | — | — | — |
| test（本次） | 不可计算 | — | — | — | — | — |

TAO test 的 `test_ours.json` 有 1,419 视频、52,155 帧，但 GT annotations 为 0，本地只能生成预测、不能算 TETA。test 预测在 `results/covtrack_ctao_epoch20_test_20261024_v2/`，不作为分数。首轮运行因源码硬编码 `filename.index('val')` 失败，修复在隔离源码副本中进行；修复后的预测产物与验证不改变 val 结果。

## BDD100K 数据与正式配置

原始数据：

```text
/raid/datasets/ring_data/data/BDD MOT/bdd100k
```

转换后完整 val：

```text
tmp_analysis/bdd100k_box_track_val_20261024_cocovid.json
```

共 200 视频、39,973 帧、465,511 标注；普通 GT 446,840，crowd ignore 18,671。类别 ID 与名称：

| id | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| name | pedestrian | rider | car | truck | bus | train | motorcycle | bicycle |

转换脚本为 `tmp_analysis/make_bddmot_cocovid_20261024.py`；独立 OVTR BDD TETA 入口为 `tmp_analysis/score_bdd_teta_with_ovtr_20261024.py`，evaluator 位于 `tmp_analysis/ovtr_bdd_teta_20261024`，已补 crowd ignore 支持。

最佳有效配置（隔离源码 `work/covtrack_epoch20_test_src_20261024`）：

```text
custom_classes=False
num_classes=8
only_validation_categories=False
max_per_img=50
match_score_thr=0.5
memo_frames=10
momentum_embed=0.8
prompt_path=None  # 现场按 BDD_CLASSES 生成 8x512 CLIP 文本特征
```

日志证据：`logs/covtrack_epoch20_bddmot_val200_native8_20261024.log` 记录 `prompt path None`、`text features shape torch.Size([8, 512])`。

正式产物：

```text
results/covtrack_epoch20_bddmot_val200_native8_20261024/tao_track.json
results/covtrack_epoch20_bddmot_val200_native8_20261024/OVTrack/teta_summary_results.pth
results/covtrack_epoch20_bddmot_val200_native8_bddteta_20261025/result.json
```

内置 TAO/TETA 路径为 **37.621 / 39.421 / 37.547 / 35.896**；独立 OVTR BDD evaluator 为 **37.582 / 39.171 / 37.693 / 35.882**。两者只相差 0.039，支持产物完整性和评测实现基本一致；正式比较保留各自 evaluator 标签，不混用。

## BDD 对照实验

所有对照均使用同一 checkpoint，10 视频切片用于快速排除配置方向；10 视频子集只作选择诊断，不能替代完整 val 分数。

| 实验 | 视频 | 配置变化 | TETA | LocA | AssocA | ClsA | 判断 |
|---|---:|---|---:|---:|---:|---:|---|
| 错误 custom 类名 | 200 | `custom_classes=True`，bdd 简写名、默认 tracker/50 | 27.414 | 9.633 | 29.093 | 43.517 | 无效，不能报告 |
| 原生 8 类 | 10 | 默认 tracker，max50 | 36.404 | 43.376 | 34.553 | 31.282 | 有效对照 |
| 原生 8 类 | 200 | 默认 tracker，max50 | **37.621** | **39.421** | **37.547** | **35.896** | 当前最佳正式结果 |
| max80 单变量 | 10 | `max_per_img=80` | 36.682 | 43.971 | 34.853 | 31.222 | 小幅提升，未全量 |
| TAO 发布参数 | 10 | max80、`.37/50/.4` | 34.034 | 44.732 | 32.460 | 24.911 | 下降，排除直接迁移 |
| DetPro 文本头 | 10 | `prompt_path` 前 8 行、max80、默认 tracker | 21.116 | 9.679 | 20.629 | 33.041 | 大幅下降，排除 |
| 官方权重 | 10 | `ctao_public.pth`，原生 8 类 | 36.055 | 44.492 | 34.312 | 29.361 | 同类配置未达论文量级 |

DetPro 对照的 `prompt_path='saved_models/pretrained_models/detpro_prompt.pt'` 实际取前 8 行，但该文件是 1203 类 LVIS 词表，8 个 BDD 类在其中的位置并非前 8；日志和配置已记录，因此只能作为负对照，不能说明 DetPro 本身就是正确 BDD 条件。改用 8 个类名现场生成文本头的结果同样只有 21.116。两项都说明“仅换文本头”不足以解释论文分数。

## 已排除的输入/评测错误

- 类别顺序：`TaoDataset` 的 `cat_ids` 按 `bdd_classes_20261024.txt` 顺序为 `[1,2,3,5,4,8,7,6]`，而模型内部 `BDD_CLASSES` 顺序为 person/rider/car/truck/bus/train/motorcycle/bicycle；输出中的 ID 与对应类别一致，不需要重映射。对错误结果的类别混淆诊断曾提示错标，但经过源码核对，直接重映射不是正确修复，相关临时文件不进入正式结果。
- 评测上限：完整预测每帧最多 47 条，未超过 TETA 的 50 条限制；把完整预测按每帧前 50 条截断后重新评测仍为 37.582，四项指标完全相同，因此不是保存/读取预测时截断导致。
- 动态 R-CNN 分数阈值：`dynamic_rcnn_thre=True` 时，`ovtrack_roi_head.py` 将测试分数阈值设为 `1/(类别数+1)*1.001`；8 类下约为 0.111，而 `custom_classes=True` 还会把 tracker 初始/对象分数阈值改为约 0.211/0.161。两者都是源码为减少词表类别而设计的正常推理路径，不列为当前配置错误，也没有据此启动新的长跑。
- 推理实际生成：普通 GT 446,840，预测 1,019,664 条，轨迹约 200,213 条；类别无关 IoU@0.5 召回约 59.4%、精度约 26.3%，匹配框 IoU 中位 0.818。同帧 IoU≥0.5 重复框为 0 对。问题指向候选数量/关联碎片化，并非简单同帧 NMS。
- 轨道合并：相邻帧、同类、IoU≥0.5 仅 84 对，10 视频 TETA 36.404 → 36.367，下降，停止该方向。
- 官方权重 `ctao_public.pth` 在相同 10 视频配置下为 36.055，而本次训练 `epoch_20.pth` 为 36.682；这进一步说明差距不是单纯训练不足，BDD 检测/推理协议尚未完全复原。

## 与论文条件和公开代码的差异

COVTrack++ 原文 Table X 报告 COVTrack **45.8 / 44.1 / 39.4 / 53.9**，并说明 zero-shot 使用 TAO 训练模型直接跑 BDD100K val，不用 BDD100K subset，不做 fine-tune 或 domain adaptation；调参只用 TAO val，BDD 不参与选择。原文还说明 detector/NMS 设置沿用检测器/OVTrack 设定。

OVTrack 原文补充材料写的是：BDD 实验使用视频中出现的类别作为 text prompts；TETA 推理使用 `|P|=50` 且类别无关 NMS。公开 COVTrack 代码没有发布 BDD 专用配置，也没有公开“按视频类别构造 prompts”的转换脚本；当前完整 200 视频实验用模型原生 8 类词表，是最接近公开可检索条件的做法，但仍不能证明与原作者逐项一致。

公开 COVTrack tree 中没有 `scripts/evaluate_bdd100k.sh` 或 `evaluate_bdd1000.sh`；GitHub API 递归树也仅返回 BDD dataset wrapper、`tools/to_bdd100k.py` 和 `ovtrack/core/to_bdd100k`。因此不把外部二手脚本当成官方协议。

## 当前判断与后续

已完成 TAO val 的有效复现，以及 BDD 完整 val 的可追溯 zero-shot 结果。BDD 的剩余差距无法用 TAO 参数、DetPro 前 8 行、单类别重映射、评测上限或轨道合并解释；现有证据倾向于检测候选/关联以及原文未公开的 BDD prompt 与输入转换协议，但尚未用原作者产物隔离到单一原因。

后续若继续复现，应优先向作者获取 BDD 原始转换 JSON、每个视频的 prompt/词表、checkpoint 与完整推理命令，再做一次同数据同 evaluator 的对照。没有这些材料时，继续在单一参数或自造脚本上试调会把探索性选择混入论文比较，不宜启动新的长跑。

## 产物与校验

| artifact | sha256 |
|---|---|
| `epoch_20.pth` | `3205bfc9e6fd570532ff12ece58c2037a9f1c039daa09a5a37a3964dd37f3a44` |
| 完整 val `tao_track.json` | `3512db84bfdf129fc33d4be18f3982f3a34f2a8417f51169b675aa06e670a430` |
| 完整 val 独立 TETA 回执 | `d056693ae4557faa5f803935895eadda9e56136fc3393353e3d74f5c9f4fb4b4` |
| 10 视频 A 参数预测 | `9d9a2061e8afd88fb0d8218c3eb73aefdcfe84ba63fdeeb2fbf06c07abedeb05` |
| 10 视频自定义 8 类预测 | `bbe191c796cc1fc1af71f044311502da302f6baa786976b0f471e9d3618df01e` |
| 自定义 8 类独立 TETA 回执 | `17c9a90925c3873dccfaa314abbf1b429519fafefe24edfbae7809d3c3eca3fd` |

完整 val 独立评测回执：`results/covtrack_epoch20_bddmot_val200_native8_bddteta_20261025/result.json`。
