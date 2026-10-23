# Base-training natural-error learning readiness — 2026-10-23

**结论：训练数据入口与独立元数据规划已就绪；当前尚未达到新训练所需证据条件。** 本次没有采集推理观测或启动训练，不能把现有 sparse validation 诊断当成训练数据，也不能把下述框数当成已经得到的自然错误样本数。

本报告仅盘点已有注释、配置、源码和已知 cache manifest，并新增本报告、metadata-only 规划及其生成器。未使用 GPU，未修改 `repos/COVTrack`、work 副本、正式 `report/` 或另一会话的 `results/covtrack_ctao_train20_bs24_20261022`。

## 1. Live annotation inventory

`data_tao` 与 `work/covtrack_base/data/tao` 均指向 `/raid/datasets/ring_data/data/tao`，不能重复计算。训练配置 `work/covtrack_base/configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py:265` 使用 `saved_models/ctao_dataset/ctao_base.json`，不是 `train_ours.json`。

| 注释 | Videos | Images | Boxes | GT tracks | 类别元数据 / 实际有标签类别 |
|---|---:|---:|---:|---:|---:|
| `work/covtrack_base/saved_models/ctao_dataset/ctao_base.json` | 500 | 490,210 | 1,489,637 | 2,588 | 1,203 / 182 |
| `data_tao/annotations/train.json` | 500 | 18,274 | 54,639 | 2,647 | 1,230 / 216 |
| `data_tao/annotations/train_ours.json` | 500 | 18,274 | 54,639 | 2,647 | 1,230 / 216 |
| `data_tao/annotations/validation_ours_v1.json` | 988 | 36,375 | 112,798 | 5,473 | 1,203 / 296 |

`ctao_base.json` 的 annotation category frequency：1,378,391 `f` 框、111,246 `c` 框、**0 `r` 框**；实际有标签类别为121个 `f` +61个 `c`。完整 LVIS-v1 base 词表有866类，但当前 C-TAO 轨迹监督只覆盖182类。不能称为“866类均有自然错误训练样本”。README 的公开 C-TAO 总框数不能替代当前这个 base-only 文件的实测框数。

密集 C-TAO 的17,131帧带 `is_ori=True`，其余473,079帧为补帧；2,588条轨迹按全部密集标注计长度中位数536帧。**密集补全框和原始人工 GT 必须区分。** 即使 `is_ori=True`，也不能自动把该帧所有框视为人工 GT。

本次进一步与原始 TAO train 核对：以 `(video_id, original frame_index, track_id)` 对齐，只接受旧类别 synset 唯一映射到1203词表且 bbox 四坐标相等（容差 `1e-6`）的交集。结果：

- C-TAO 原始帧有53,263框；其中**52,911框**与原始 TAO GT 精确对应，bbox 最大绝对差为0。
- 另352框在原始 GT 没有对应 key，排除出首批监督目标。
- 原始旧1230词表216个有标签类别中208个有唯一新词表 synset 对应，8个无对应；不能按数值 ID 直接拼接两个版本。原始 TAO 可唯一映射的 base 框为53,264个，与上述精确交集并非同一集合。
- 本规划只使用52,911个精确交集人工 anchor 作为监督目标；其余 C-TAO 框保留独立来源标记，不作为本规划的真实分类错误标签。

## 2. Train/val independence and vocabulary boundary

Train与validation：video ID交集0；去掉 `train/` / `val/` 前缀后的 video name、image relative path 交集均为0。进一步按 AVA `movie`、HACS `youtube_id` 合并为 YouTube 源视频，ArgoVerse按 `sequence` 合并相机，其余按 dataset/clip 合并：train432组、validation849组，源组交集0。这是**元数据去重**，没有图像内容去重，也未做 Charades subject 或 YFCC uploader holdout。

`class_name.py` 的866个 base 和337个 novel ID（代码0-based label转1-based category）分别与 `frequency in {f,c}` / `frequency==r` 完全一致。182个当前有标签 base 类中136个在val296词表内，**46个在外**。

活动配置的 `only_validation_categories=True` 会在 `OVTrackRoIHeadUncertainty` 构造时裁剪 text features 到val296类（`ovtrack_roi_head.py:1818`），推理后再恢复为原始 category ID。因此只替换 annotation path 不能得到完整 train/base 词表后验。

独立采集的合法源码配置入口是保持 `num_classes=1203`、`custom_classes=False`，并将 `only_validation_categories=False`、`only_test_categories=False`；预期导出1203前景+背景，共1204列，明确保存 category IDs `[1..1203, -1]`。监督目标仍限于base人工anchor，保留novel文本列不构成novel GT监督。此入口由源码静态核对支持，**尚未运行模型/schema smoke**。

`work/covtrack_tces_v2` 的现有 dump hook 按后验实际列数选行（ROI head:3488，MOT:237–267），没有显式写死297；但逐帧npz未保存 `posterior_category_ids`。现有 `tmp_analysis/build_track_posterior_cache.py` 则固定val296顺序、背景和 `log(296)` 归一化，不能直接用于train1203。需要独立副本扩展schema并证明 `det_labels+1`、posterior列与fusion mask一致。类别655仅存在 `speaker_(stero_equipment)` / `speaker_(stereo_equipment)` 拼写差；应保留数值ID并记录别名，不能按显示名重新排序。

**Train full1203 与已有val296是不同 vocabulary regime，也改变类别竞争和检测筛选分布。** 不能仅靠列重排或裁剪就声称后验、几何输出或学习分布等价；后续manifest必须记录实际候选词表与mask，并在schema/迁移smoke中核验这种差异。

## 3. Existing cache availability

已有完整 `results/cov_tces_v3_fullval` 及 `results/tces_audit_20261023/track_cache.npz` 对应 `validation_ours_v1.json`：988 videos、36,371非空dump文件、1,529,908观测、324,639预测轨迹、297列后验。`results/dependence_audit_20261023` 的readout cache也来自同一fullval。它们可用于既有诊断和接口核对，**不得作为训练样本**。

在本次 `results` / `tmp_analysis` 文件名与manifest浅查范围，未找到train/C-TAO同schema的完整 posterior+appearance cache。两个 `covtrack_ctao_train20*` 目录的launch manifest指向C-TAO训练注释，未声明采集cache；没有扫描大dump内容或全盘数据。因此结论是“本次限定范围未找到”，并非证明全盘不存在。

## 4. Sealed metadata-only plan

规划：`tmp_analysis/base_error_learning_plan_20261023.json`。生成器：`tmp_analysis/build_base_error_learning_plan.py`。种子为 `20261023-base-natural-error-metadata-v1`；每个源数据集内按source group的SHA256排序切分约80/10/10，保留完整源组，帧和轨迹视为组内重复观测。

| Full metadata split | Videos | Source groups | Manual anchor images | Manual anchor boxes | GT tracks |
|---|---:|---:|---:|---:|---:|
| Fit | 405 | 343 | 13,791 | 42,020 | 2,067 |
| Tune | 45 | 42 | 1,492 | 4,888 | 250 |
| Test | 50 | 47 | 1,733 | 6,003 | 271 |

按frequency分层，将182个有标签base类分为146个拟合候选类和36个class holdout。**这个holdout仅对新的readout learner构成pseudo-novel，不是detector未见类**；released detector/CLIP可能已经见过这些base类与训练视频。full fit视频实际只覆盖其中134个拟合类别；full test有9个class-holdout类别、289个相应框，不代表36类都被充分检验。

最小pilot在7个源数据集的每个split中，按固定hash选择含至少一条≥6人工anchor轨迹的源组：每源数据集2 fit、1 tune、1 test源组。包含整个组，故不是简单28个视频。

| Pilot | Videos | Source groups | Anchor images | Manual anchor boxes | Boxes eligible for fitting/tuning under fit-category rule |
|---|---:|---:|---:|---:|---:|
| Fit | 24 | 14 | 767 | 2,297 | **2,138** |
| Tune | 9 | 7 | 258 | 672 | 647 |
| Test | 8 | 7 | 274 | 984 | 903 (evaluation only) |
| Total | **41** | **28** | **1,299** | **3,953** | — |

Pilot fit仅19个有标签类别，其中3个预留class holdout；test只有3个class-holdout类别、81个相应框。上述只是可用GT上限，真实检测匹配、错误和独立轨迹支持尚未知，**不能直接据此声称可训练、具有统计功效或验证了机制**。首批应定位为schema与学习信号检查。

JSON枚举了500个视频的split/source group、pilot的1,299个image ID/路径/时序、3,953个人工anchor annotation ID、完整词表与class holdout ID及输入SHA256。`metadata_validation=PASS_METADATA_ONLY`：源组split互斥、拟合/holdout类别互斥、52,911框总数相符、1,299个pilot文件路径全部存在；未做图像内容校验或模型smoke。

生成器只读输入并拒绝覆盖既有输出。已使用原JSON的 `created_utc`，在临时目录完整重建，得到 **`PASS_BYTE_IDENTICAL`**（同一JSON SHA256）；临时重建文件随临时目录清理。复现时使用 `--output` 指定新路径，`--created-utc` 指定原时间戳；该核验未运行模型。

首批时间轴仅包含至少有一个人工anchor的稀疏TAO帧，是按标签可用性选择的训练样本，不代表完整视频错误总体。它可以保留这些帧上的所有GT-free预测，再独立匹配人工anchor生成监督；不能将未匹配观测当负样本。若研究需要密集邻帧依赖，必须另封存连续上下文采集manifest，不能把本pilot称作dense-frame证据。

## 5. Minimal next executable collection and remaining items

最小独立集合已经由JSON明确到视频、帧、anchor与类别，但**完整采集命令尚不可执行**。下一阶段应在独立副本与新输出目录中依次完成：

1. 以已冻结released `ctao_public.pth` 固定模型，不读取另一会话的动态epoch checkpoint；按JSON生成保留原始ID、时序的pilot annotation与独立collection config。
2. 扩展dump与cache schema：显式记录vocab regime、category IDs、actual fusion mask、p_text/p_image/p_fused、原始未时间融合的appearance/cls features、box/score、proposal/local/global IDs、video/image/frame_index/时间戳与零观测帧receipt。源码、配置、模型、注释、类别mapping与逐帧清单都绑定hash。
3. 先通过schema和类别映射smoke；再采集GT-free预测，封存raw cache；之后在独立监督层做class-blind geometry matching，仅采用精确人工anchor目标并保留unmatched/contaminated/补全来源状态。
4. 在拟合前明确可学习机制、残差/共享错误目标、控制项和损失。只用fit视频×fit类别优化；tune仅用tune视频的fit类别；test与class holdout不参与选择。按source group或视频/轨迹聚类报告支持与不确定性。

当前缺少train原始cache、合法完整词表的schema证明、对应监督target和已明确的可学习机制。自然相似性与错误依赖诊断、启发式readout收益或相关矩阵本身均不能替代这些条件。本规划是下一阶段的metadata入口。

## 6. Provenance

| Input/artifact | SHA256 |
|---|---|
| C-TAO base annotation | `4269b1a5f026350cf45fbbf61689894f7d45d9bd2730331e73ab6f841d9b3adc` |
| Original TAO train annotation | `7eb551fdeeeebc76b876ae255f91dc5662c7270a125955c5f1be2d9bd30921d0` |
| Validation v1 annotation | `559811dd8166922c537e71457fecc0faf9ad0acc5522cafca7954273c0e2a67d` |
| Released `ctao_public.pth` | `e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c` |
| Training config, read-only | `d8ae926e540c327d4baa2ced8cb1338f70d6a67eb73d2fb02de4d79893cecc08` |
| Class-name source | `6f50d7b3f16eef96e15b6cfbf97a53b195e26dbc34df51e7679645c99e6937d9` |
| Metadata plan | `f37f1ecc52734953ea5269407ab0aa8013e650e97bed4dde3cdd399a11e13ba5` |
| Plan generator | `c391140fcea32b11b1bbcfe27c1e10f86bbdcd94701a5809c28fd6cc8e6dcfaa` |

All inventory observations in this report were read live in this session. No inference, training, evaluator rerun or GPU action was performed for this readiness task.
