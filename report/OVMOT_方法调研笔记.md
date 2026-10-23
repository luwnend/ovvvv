# OVMOT（开放词汇多目标跟踪）方法调研笔记

> 记录方式：每篇 **一句话定位 → 用了什么模块解决什么问题 → 代码怎么跑 → 实现/数据坑 → 相对基线提升**
> 代码位置：`/home/chenhc/ov/OVMOT_RESEARCH/repos/`
> 数据：`/raid/datasets/ring_data/data/tao`（TAO 原始 + OVTrack 转换标注 + C-TAO 连续标注）

---

## 0. 任务与评测口径（先统一，否则数字没法比）

- **OV-TAO 设定**：TAO 的 1203 个 LVIS 类别里，866 个 base（frequent）参与训练，337 个 novel（rare）只在测试出现。评测用 **TETA**，拆成 `TETA / LocA / AssocA / ClsA`，并分别报 base / novel。
- **TETA = (LocA + AssocA + ClsA)/3**，其中 LocA 是定位（检测+轨迹框质量）、AssocA 是关联、ClsA 是分类。**novel 的 ClsA 普遍只有 2~7**，这是 OVMOT 最真实的痛点：检测器能框出来，但分类几乎全靠 CLIP 文本对齐。
- **数据坑（影响所有方法）**：TAO 原始标注是**稀疏**的（每 30 帧标一次），OVTrack 的 `validation_ours_v1.json` 只有 36k 帧 / 988 视频，但视频有 ~1171 帧，中间大量帧无 GT → 评测只在这些稀疏帧上算，训练时"相邻帧"其实跨度很大。C-TAO 把训练集补成 490k 帧连续标注，这正是 COVTrack/VOVTrack 的出发点。
- **实现坑**：OVTrack 系代码是 mmdet 2.23 + mmcv 1.4.4 + torch1.10，现代环境直接跑会连环报错（`np.int`、`bdd100k` 可选依赖、`tools/test.py` 缺 `--work-dir`、CLIP 缺 `ftfy`、`lvis` 包）。我已在本机修复并归档，见 `work/ovtrack_base`。

---

## 1. OVTrack (CVPR 2023) — 本领域的开山基线

- **GitHub**: https://github.com/SysCV/ovtrack ｜ 本地 `repos/ovtrack`
- **核心网络**：Faster R-CNN(R50-FPN) 检测 + **CLIP 知识蒸馏分类头**（DetPro prompt，1203 类文本 embedding，图像/文本双分支集成）+ **QuasiDense 外观 embedding 头**（4conv+1fc → 256d）。**关联完全是手工的**：`bisoftmax`（行/列双向 softmax）与 cosine 各占一半，再按 tracklet 动量更新外观，阈值 0.5 贪心匹配。
- **模块→问题**：
  - 扩散模型生成"同一物体不同外观"的图像对 → 解决**没有视频跟踪标注**也能学外观；
  - CLIP 蒸馏 + DetPro prompt → 解决**开放词表分类**；
  - 动量外观 + memo bank → 解决**短时遮挡**。
- **怎么跑**（已在本机跑通）：
  ```bash
  # 环境：/home/chenhc/.conda/envs/ovtrack_ovmot (torch1.10+mmcv1.4.4+mmdet2.23)
  cd work/ovtrack_base
  CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/ovtrack-teta/ovtrack_r50.py \
      saved_models/ovtrack_detpro_prompt.pth --work-dir work_dirs/base \
      --eval track --eval-options resfile_path=results/base
  ```
- **本机 GPU 注意**：物理 A100 40G 用 `CUDA_VISIBLE_DEVICES=2`；`3` 是 4G DGX Display，不要按 `nvidia-smi` 序号直接假设。
- **实测**：3 序列子集 TETA 53.7（全 base，无 novel）；val200 子集已完成，Base TETA 36.07 / Novel 28.79，详见 `report/OVTrack_收尾总结.md`。
- **官方数字**（OV-TAO val）：novel TETA 27.8 / AssocA 33.6 / ClsA 1.5；base TETA 35.5 / AssocA 36.9 / ClsA 20.2。
- **弱点（我的改进切入点）**：① 完全不建模运动，遮挡后靠外观硬撑；② 分类只在**单帧 ROI** 上做，时序信息浪费；③ 关联阈值对类别数敏感（softmax 随 1203 类摊薄，分数被压到 1e-3 量级），需要 `dynamic_rcnn_thre` 这种 hack；④ 贪心匹配非全局最优。

## 2. TETer / TETA (ECCV 2022) — 指标与"类别无关关联"基线

- **论文**: Tracking Every Thing in the Wild；**代码**: https://github.com/SysCV/tet
- **核心**：把 TAO 的评测从"只算检测 AP"改成 **TETA**（LocA/AssocA/ClsA 三支），并提出 TETer：检测器 + **类别无关的关联**（外观 embedding 不依赖类别）+ 分类头。
- **解决什么问题**：证明"大词表下，**关联必须类别无关**"，否则 1203 类的分类噪声会毁掉匹配；同时给出可分解的诊断指标。
- **怎么跑**：`pip install git+https://github.com/SysCV/tet.git#subdirectory=teta`；所有方法的评测都用它（我已装进 `ovtrack_ovmot` 环境）。
- **坑**：TETA 的 `TETA[50]` 是按类别求平均再平均，**类别极不平衡时会被长尾类别主导**；`COMBINED_SEQ` 与 per-class 的聚合方式不同，跨论文对比要确认口径。
- **提升**：相对纯检测 AP，TETer 在 TAO val 上 TETA 33.3（vs QDTrack 30.0）。

## 3. SLAck (ECCV 2024) — 语义/位置/外观三线索 + 可学习图

- **论文**: SLAck: Semantic, Location, and Appearance Aware Open-Vocabulary Tracking ｜ **代码**: https://github.com/siyuanliii/SLAck
- **⚠️ 仓库现状**：截至抓取时**只有 README + teaser 图，没有代码**（`find` 仅 2 个文件）。这是复现的最大障碍，只能读论文。
- **核心**：把 **语义（CLIP 分类 logits）、位置（IoU/中心距）、外观（embedding）** 三种线索在**关联的早期**就融合，用一个轻量 **spatial-temporal object graph** 学习如何加权，替代 OVTrack 那种"先外观匹配、再手工后处理"的范式。
- **模块→问题**：语义图节点/边 → 解决 novel 类分类不稳导致的外观漂移；位置先验 → 解决大词表下运动模式复杂、纯外观失效；图注意力 → 解决"手工融合权重"不可学。
- **怎么跑**：**无代码**；论文给的结果需要用 OVTrack 框架自己实现（工作量大）。
- **官方数字**（OV-TAO val）：novel TETA 31.1 / LocA 54.3 / AssocA 37.8；base TETA 37.2 / LocA 55.0 / AssocA 37.6。相对 OVTrack novel **+3.3 TETA**。
- **启示**：**融合位置线索是性价比最高的一步**——我的改进方案正是沿这条线做，但用更轻的"运动+语义置信度门控"，不需要训练图网络。

## 4. OVTR (ICLR 2025) — 第一个端到端 Transformer 开放词表跟踪器

- **论文**: OVTR: End-to-End Open-Vocabulary Multiple Object Tracking with Transformer ｜ **代码**: https://github.com/jinyanglii/OVTR ｜ 本地 `repos/OVTR`
- **核心网络**：DINO 式 **dual-branch deformable DETR**（检测分支 + 跟踪分支），**900 个 track query** 在帧间传播；引入 **CIP（Category Information Propagation）** 把上一帧类别概率传给下一帧 query，**attention isolation** 把"开放词表感知"与"跟踪"解耦，避免文本 cross-attention 污染跟踪特征。
- **模块→问题**：track query 传播 → 解决"检测-关联两阶段"的误差累积与后处理；CIP → 解决 novel 类逐帧分类抖动；isolation mask → 解决文本分支与视觉分支互相干扰。
- **怎么跑**：
  ```bash
  # 需先编译 deformable attention CUDA op
  cd repos/OVTR/ovtr/models/ops && sh make.sh
  cd ../.. && sh tools/ovtr_multi_frame_lite_train.sh          # 训练 Lite
  sh tools/ovtr_ovmot_eval_lite_val.sh                          # 评测
  ```
  数据需要额外准备 `lvis_filtered_train_images.h5`、`lvis_clear_75_60.json`、`iou_neg5_ens.pth`、`clip_image_embedding_all.pt`。
- **官方数字**（TAO val）：novel TETA 31.4 / AssocA 34.5 / ClsA 5.4；base TETA 36.6 / AssocA 37.6 / ClsA 20.1；速度 3.4 FPS（Lite 12.4 FPS）。
- **坑**：① 端到端训练极吃显存与数据（900 query × 5 帧）；② 需要预生成 CLIP image embedding，**换数据集/换类别就要重跑预处理**，工程上很重；③ 论文承认 novel ClsA 仍只有 5.4，说明**端到端并不能自动解决开放词表分类**。

## 5. MASA (CVPR 2024 Highlight) — 通用实例外观模型

- **论文**: Matching Anything by Segmenting Anything ｜ **代码**: https://github.com/siyuanliii/masa ｜ 本地 `repos/masa`
- **核心**：用 **SAM 的密集分割**当"免费的目标区域提议"，对同一张图做**穷举式数据增广**（颜色、模糊、遮挡、几何变换），把增强前后同一区域的 embedding 拉近、不同区域推远 → **完全不需要跟踪标注**就能学"任意物体的实例对应"。
- **模块→问题**：SAM → 解决"没有框级标注也能构造正样本对"；exhaustive augmentation → 解决跨域外观变化；**MASA adapter** → 可插到任意检测/分割模型上，变成跟踪器。
- **怎么跑**：MMDetection 3.x 体系，`configs/masa-detic`、`masa-gdino`、`masa-sam`；推理走 `masa/models/tracker/masa_tao_tracker.py`（外观匹配 + IoU/分数门控）。
- **官方数字**（OV-TAO val，用 Detic-SwinB 检测）：MASA-R50 **base TETA 46.5 / novel 41.1**；MASA-GroundingDINO base 47.3 / novel **41.9**——**比 OVTrack 高 14+ 个点**，主要涨在 AssocA（42.7 vs 33.6）。
- **坑**：① 检测器换了（Detic/GDINO），**不是纯跟踪器的公平对比**；② 推理要跑 SAM/大模型，速度慢；③ 代码是 mmdet3.x，与 OVTrack 的 2.23 不兼容，**无法直接把这套 embedding 塞进 OVTrack**（除非只取权重做蒸馏）。
- **启示**：**关联的天花板由外观模型决定**。如果只能改跟踪器，最划算的是"更强的、跨域的外观特征"，而不是继续调 softmax。

## 6. TRACT (ICCV 2025) — 轨迹级关联 + 轨迹级分类

- **论文**: Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking ｜ **代码**: https://github.com/Nathan-Li123/TRACT ｜ 本地 `repos/TRACT`
- **核心**：两个模块
  - **TCR（Trajectory Consistency Reinforcement）**：用轨迹历史约束同一 ID 的身份与类别一致性；
  - **TraCLIP（plug-and-play 轨迹分类器）**：**TFA** 把轨迹内多帧 CLIP 特征聚合，**TSE** 用轨迹语义上下文增强，再和文本 embedding 对齐。
  - 关联侧用 **MASA** 作为通用外观模型（`TRACT/masa`），分类侧用 TraCLIP 离线重分类。
- **模块→问题**：TFA → 解决单帧分类噪声；TSE → 解决遮挡/类别歧义下语义漂移；TCR → 解决长轨迹 ID switch。
- **怎么跑**（两段式，工程上最友好）：
  ```bash
  # 1) 用任意跟踪器产出轨迹（OVTrack / YOLO-World+OCSORT / MASA）
  # 2) 轨迹- GT 匹配 → 抽 tracklet 数据集
  python TraCLIP/tools/match_tracks.py
  python TraCLIP/tools/extract_track_dataset_pred.py
  # 3) 训练/评测轨迹分类器
  bash TraCLIP/scripts/train_tao.sh
  bash TraCLIP/scripts/val.sh
  # 4) 回写 TETA 格式
  python TraCLIP/tools/transfer_tracklets_to_teta.py
  ```
- **官方数字**：用 YOLO-World 检测时，base/novel TETA = **39.4 / 33.7**（val）、**35.7 / 33.1**（test）；比 MASA 再 +1.9/+1.5（val）。
- **坑**：① **两段式**，轨迹分类器只能用"已跟踪出的轨迹"，**关联错误无法挽回**；② 代码里大量硬编码绝对路径（`/data3/InsCap/imgs`、`/data2/...`），迁移成本高；③ 依赖 `clip_len=10`，短轨迹（<10 帧）要 padding，TAO 稀疏标注下轨迹很短。
- **启示（对我最重要）**：**"离线轨迹重分类"是低风险高收益的改进**——不动检测/关联，只把每条轨迹的类别重新投票。我的方案会在**在线**版本里做类似的事（轨迹级 EMA 语义 + 置信度门控），从而同时改善 ClsA 和 AssocA。

## 7. VOVTrack (ICCV 2025) — 用原始视频做自监督

- **论文**: VOVTrack: Exploring the Potentiality in Raw Videos for Open-Vocabulary Multi-Object Tracking ｜ **代码**: https://github.com/zekunqian/VOVTrack ｜ 本地 `repos/VOVTrack`
- **核心**：不再依赖"生成的图像对"，而是**直接用原始视频**做自监督：
  - **tracking-state-aware prompt attention**（检测阶段把"跟踪状态"注入 prompt）；
  - **self-consistency association**（短-长采样、类别一致性、互一致性）在视频上自训练外观头。
- **模块→问题**：原始视频自监督 → 解决"扩散生成的假图像对分布不真实"；短-长采样 → 解决遮挡/快速运动下的正样本对构造；类别一致性 → 解决自训练伪标签噪声。
- **怎么跑**：
  ```bash
  # Stage1/2: DetPro 训练（LVIS）
  ./tools/dist_train.sh configs/lvis/detpro_ens_20e.py 4 ...
  ./tools/dist_train.sh configs/lvis/detpro_ens_20e_part_batch_prompt_reverse.py 4 ...
  # Stage3: pair pretrain
  ./tools/dist_train.sh configs/ovtrack-teta/ovtrack_r50_reverse_without_inference.py 4 ...
  # Stage4: self-supervised finetune
  ./tools/dist_train.sh configs/ovtrack-teta/adding_spatial/...ratio1.0.py 4 ...
  # 评测
  tools/dist_test_new.sh ... --eval track --eval-options resfile_path=results/vovtrack_iccv2025 \
    --cfg-options model.tracker.match_score_thr=0.33 model.test_cfg.rcnn.max_per_img=110 \
    model.tracker.memo_frames=30 model.tracker.momentum_embed=0.4
  ```
- **官方数字**（OV-TAO val）：base TETA **39.6** / LocA 58.9 / AssocA 40.9 / ClsA 19.1；novel TETA **35.3** / LocA 58.5 / AssocA 40.9 / ClsA 6.5。
- **坑**：① 四阶段训练，**算力/时间成本高**；② 关键超参（`match_score_thr=0.33`、`memo_frames=30`、`momentum_embed=0.4`）与 OVTrack 默认差异很大，**说明默认关联参数本就次优**；③ 依赖权重在 HuggingFace，国内下载不稳。
- **启示**：**memo_frames 30 + 动量 0.4** 这种"更长记忆、更快更新"的组合值得直接 ablation——我的方案也会显式建模"轨迹记忆长度"。

## 8. COVTrack (ICCV 2025) — 连续视频监督 + 自适应多线索融合

- **论文**: COVTrack: Continuous Open-Vocabulary Tracking via Adaptive Multi-Cue Fusion ｜ **代码**: https://github.com/zekunqian/COVTrack ｜ 本地 `repos/COVTrack`
- **核心**：① 发布 **C-TAO**（把 TAO 训练集补成 490k 帧 / 1.52M 框的连续标注）；② **自适应多线索融合**：外观 + 运动/位置 + 语义 + **帧内置信度** + **帧间置信度自校正**，用 `FeatureFusionModule` 学每个线索的权重，直接输出融合后的关联特征。
- **模块→问题**：C-TAO → 解决"稀疏标注学不到运动"；**intra-frame confidence** → 解决单帧检测/分类不可靠时该信谁；**inter-frame confidence** → 解决相邻帧线索跳变；fusion head → 替代手工加权。
- **怎么跑**：
  ```bash
  # 论文原命令：4 卡，C-TAO 训练 20 epoch（本机只有 1 张可用 A100，不能直接照搬）
  CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh \
    configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py 4 25001 \
    --work-dir work_dirs/ctao --cfg-options model.roi_head.feature_fusion_head.max_fusion_ratio=2.0 \
    data.train.dataset.dataset.extra_sample_ratio=8 model.roi_head.use_cyc_loss=False total_epochs=20

  # 本机单卡复现：先跑 test/baseline，不要直接开 20 epoch 训练
  CUDA_VISIBLE_DEVICES=2 tools/dist_test.sh \
    configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py \
    saved_models/ctao_public_res/ctao_public.pth 1 25001 \
    --eval track --eval-options resfile_path=results/cov_val60_base \
    --cfg-options model.tracker.match_score_thr=0.37 model.test_cfg.rcnn.max_per_img=80 \
    model.tracker.confused_features=True model.roi_head.only_validation_categories=True \
    model.tracker.memo_frames=50 model.tracker.momentum_embed=0.4
  ```
- **官方数字**（OV-TAO val）：novel TETA **34.3** / LocA 58.2 / AssocA 41.3；base TETA **39.6** / LocA 57.3 / AssocA 42.0。相对 OVTrack novel **+6.5**。
- **坑**：① C-TAO 只有 **base 类**的连续标注（`ctao_base.json`），novel 类仍无监督；② 依赖 `ovtrack_pair.pth` 预训练 + 自定义 RPN (`MyRPNHead`) + 大量新注册类，**直接迁移容易权重不匹配**；③ `run_later.py` 里写死了端口和路径。
- **启示**：**帧内/帧间置信度是核心**。我的方案会用一个**轻量的可靠性估计器**（不需要 C-TAO 全量重训）来门控"外观 vs 运动 vs 语义"。

**本地代码地图（2026-10-22 核对）**
- 仓库：`repos/COVTrack`，commit `9b0ced5`；环境可复用 `ovtrack_ovmot`。
- `ovtrack/models/roi_heads/ovtrack_roi_head.py:1393` `FeatureFusionModule`：`bbox(4)→256`、`cls(512)→256`、`gate_network(768→256→2)`；融合强度为 `max_fusion_ratio*(1-assoc_conf)^2`。
- `ovtrack/models/trackers/ovtracker.py:348` `OVTrackerUncertainty`，`:481` `match`；memo 只保留最近若干帧，轨迹过期即删除，没有跨段拼接。
- 配置：`configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py`；训练从 `ovtrack_pair.pth` 起，读 `ctao_base.json`，默认 10 epoch，`run_later.py` 覆盖为 20。
- 官方评测参数：`match_score_thr=0.37`、`max_per_img=80`、`confused_features=True`、`memo_frames=50`、`momentum_embed=0.4`。
- 与 OVTrack 的差异：多了 `FeatureFusionModule` 与 intra/inter confidence；但关联分数仍是 `bisoftmax+cosine` 平均，分类仍是逐帧+TCC，这两点正是可改进处。
- 运行前需从 HuggingFace 拉 `ctao_public_res/ctao_public.pth` 与 `ctao_dataset/ctao_base.json`；本地已落地并完成官方完整 val 复现（Base 39.554 / Novel 34.205，见 `report/项目进度.md`）。

## 9. OVT-B / OVTrack+ (NeurIPS 2024) — 新基准与强基线

- **论文**: OVT-B: A New Large-Scale Benchmark for Open-Vocabulary Multi-Object Tracking ｜ 代码我尝试从 GitHub 拉取失败（仓库名/权限问题），**未拿到官方实现**。
- **核心**：构建 **OVT-B**：1973 个视频、1048 类、**密集标注**（解决 TAO 稀疏的根本问题）；提出 **OVTrack+**：在 OVTrack 上**显式加运动特征**（Kalman/位置）+ 更好的检测，证明"运动线索对 OVMOT 至关重要"。
- **数字**：OVT-B 上 OVTrack+ TETA **47.0** vs OVTrack **11.5**、StrongSORT 12.4；OV-TAO-val 上 OVTrack+ base TETA 38.4 / novel 32.5（AssocA novel 38.7 vs OVTrack 33.6）。
- **坑/启示**：① **换数据集后 OVTrack 从 34.7 掉到 11.5**——说明现有方法对标注密度/域极度敏感，这是"数据集带来的问题"的最好证据；② **加运动就能 +4~5 AssocA**，再次印证我的方向；③ 仓库不可得，只能作为对照引用。

## 10. COVTrack++ (arXiv 2026) — 连续视频 + 检测/关联协同

- **论文**: COVTrack++: Learning Open-Vocabulary Multi-Object Tracking from Continuous Videos via a Synergistic Paradigm (arXiv 2603.24016)
- **核心**：针对"缺连续标注"和"检测与关联割裂"两大痛点，提出**协同范式**：检测端与关联端共享/互相增强（而不是先检测再关联），并用连续视频监督训练。
- **数字**：稀疏 TAO val/test 上 novel TETA **35.4 / 30.5**；另报 Dense C-TAO-val 上 COVTrack **33.0 / 38.4**、COVTrack++ **34.6 / 39.7**（novel/base TETA）。**Dense C-TAO-val 与官方主表的稀疏 TAO val 不是同一口径，不能混比。**
- **坑**：① 2026 年新工作，**代码未见公开**；② 协同训练需要检测器可微/可联合训练，与 OVTrack 的 `freeze_detector=True` 范式冲突，迁移成本极高；③ Dense C-TAO-val 标注未随官方仓库发布，目前无法本地复现该表。
- **启示**：**联合优化是终局**，但短期最可行的是"冻结检测器 + 改进关联/分类"——这也是我选择的路线（算力可控、可解释、可快速迭代）。

---

## 11. 横向对比表（OV-TAO val，官方报告数字）

| 方法 | 会议 | 检测器 | base TETA | base AssocA | base ClsA | novel TETA | novel AssocA | novel ClsA | 代码 |
|---|---|---|---|---|---|---|---|---|---|
| OVTrack | CVPR23 | R50-FPN | 35.5 | 36.9 | 20.2 | 27.8 | 33.6 | 1.5 | ✅ |
| TETer | ECCV22 | R101 | 33.3 | 35.0 | 13.2 | — | — | — | ✅(TETA) |
| SLAck | ECCV24 | R50 | 37.2 | 37.6 | — | 31.1 | 37.8 | — | ❌ 仅README |
| OVTR | ICLR25 | DINO-R50 | 36.6 | 37.6 | 20.1 | 31.4 | 34.5 | 5.4 | ✅ |
| MASA-R50 | CVPR24 | Detic-SwinB | 46.5 | 43.0 | — | 41.1 | 42.7 | — | ✅ |
| TRACT | ICCV25 | YOLO-World | 39.4 | — | — | 33.7 | — | — | ✅(两段式) |
| VOVTrack | ICCV25 | R50 | 39.6 | 40.9 | 19.1 | 35.3 | 40.9 | 6.5 | ✅ |
| COVTrack | ICCV25 | R50 | 39.6 | 42.0 | — | 34.3 | 41.3 | — | ✅ |
| OVTrack+ | NeurIPS24 | — | 38.4 | 41.0 | 18.9 | 32.5 | 38.7 | 1.8 | ❌ |
| COVTrack++ | arXiv26 | — | — | — | — | 35.4 | — | — | ❌ |

> 注：MASA 用了更强的开放词表检测器（Detic-SwinB），与 R50-FPN 的方法**不完全公平**；但即使只看关联，MASA 的 AssocA 也明显更高。

---

## 12. 我从中提炼的"可改进点"（技术深度版）

### 观察 1：所有方法的 **novel ClsA 都极低（1.5~6.5）**，而 LocA 已 50~59
→ 瓶颈不在"看见"，在"说对名字"和"别跟丢"。分类噪声又会污染外观匹配（同一物体不同帧被分到不同类，若关联用类别约束就会断轨）。

### 观察 2：**运动线索被系统性低估**
OVTrack 完全无运动；SLAck/OVTrack+/COVTrack 加了运动后 AssocA 普遍 +3~5。但大词表场景下**相机在动**（TAO 含大量手持/车载/YFCC），全局运动补偿比简单 IoU 更重要。

### 观察 3：**关联阈值与类别数强耦合**
OVTrack 的 `bisoftmax` 在 1203 类下分数被压到 1e-3，靠 `init_score_thr=1e-4` 这种魔法数。**分数尺度不稳定**导致跨数据集迁移极差（OVT-B 上掉到 11.5）。

### 我的方案（下一节详述）
**RAMS（Reliability-Aware Multi-cue Association with Semantic Momentum）**：
1. **运动补偿**：估计帧间全局仿射（稀疏光流/RANSAC 或相位相关），把 tracklet 预测框变换到当前帧再做 GIoU；
2. **可靠性门控**：用"分类置信度 p1、p1-p2 margin、熵、novel 概率质量"估计每条检测的语义可靠性，动态决定外观/运动/语义的权重（不训练大网络，轻量 MLP/规则）；
3. **语义动量**：轨迹级类别概率 EMA + 一致性约束，抑制单帧抖动（在线版 TraCLIP）；
4. **尺度无关的匹配分数**：用 sigmoid/rank-normalize 替代 softmax，消除类别数耦合。

这三条都能在**冻结检测器**的前提下实现，算力可控，且每一条都有清晰的 ablation。
