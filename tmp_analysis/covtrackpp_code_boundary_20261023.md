# COVTrack++ 官方代码边界与两个可学习方向的独立审计

审计日期：2026-10-23。范围：公开代码、论文机制、近邻文献与可证伪设计；未训练、未启动新实验、未修改 `report/`、冻结仓库或运行中的训练。本文件中的方向均是待验证假说，不能写成已经完成的方法或实验结论。

## 1. 官方代码的实时核实

**结论：论文官方链接可以访问，但当前公开的是与冻结 COVTrack 完全一致的旧代码树；尚未找到可明确对应 ++ 新增 MGA/TCP 的公开实现。不能把“有官方仓库链接”写成“++ 已完整开源”，也不能写成“作者没有官方仓库”。**

证据链：

1. [arXiv 2603.24016v2](https://arxiv.org/html/2603.24016v2)，2026-07-02，数据集章节明确链接 `https://github.com/zekunqian/covtrack`。本地对应 `papers/COVTrackpp_arxiv2603.24016.pdf`。
2. GitHub 实时 API 将该链接解析为 [`zekunqian/COVTrack`](https://github.com/zekunqian/COVTrack)，description 仍是 ICCV 2025 COVTrack；`pushed_at=2026-08-06T19:11:41Z`。
3. `git ls-remote --heads --tags` 和 branches API 均只给出 `main`，HEAD 为 **`9b0ced5779ee36f5dd73dbe39b5ae5d57abb4b3b`**；未返回公开 tag。
4. GitHub recursive tree 有 303 个条目、248 个 blob；**全部 248 个 path/blob SHA 与本地冻结 `repos/COVTrack` 的 HEAD 一致**。README、ROI head、tracker 和 cycle loss 的远端原文也分别与本地工作树逐字节一致。
5. 枚举公开树、作者公开仓库、检索模块名并检查当前融合/跟踪路径后，未识别出 MGA 的 parent-child 包含掩码/TAQ 门控和 TCP 的高低置信检测恢复实现。仅凭名称检索不能证明任意匿名实现不存在，因此结论限定为“当前官方公开树中未找到可核验的 ++ 新模块”。

原始证据、论文下载与 SHA256 清单存于 `tmp_analysis/covtrackpp_live_sources_20261023/`，入口为 `manifest.json`、`github_repository.json`、`github_branches.json`、`github_tree.json` 和 `github_author_repositories.json`。HF paper API 对该编号返回 404；随后直接读取 arXiv 正文成功，404 不代表论文或代码不存在。

**未创建 `repos/COVTrackpp`**：重新克隆同一个旧 HEAD 并将目录命名为 ++ 会制造错误来源。现有冻结代码足以审查已发布 MCF；++ 新增部分目前只能做论文级机制审计。

## 2. 具体代码与公式边界

### 2.1 已公开的 MCF 路径

| 机制 | 已核实位置 | 实际行为与边界 |
|---|---|---|
| 联合预测 intra-frame gates | `repos/COVTrack/ovtrack/models/roi_heads/ovtrack_roi_head.py:1393`，尤其 1420、1528–1529 | `FeatureFusionModule` 将 appearance、bbox、classification 特征拼接，用 MLP+sigmoid 输出两个门控；不是三个独立手设权重 |
| 融合及 residual 强度 | 同文件 1545–1570 | gates 再乘 cue confidence 与 0.5；辅助特征经 residual MLP；强度为 `max_fusion_ratio*(1-assoc_conf)^2`，最后加回 appearance |
| inter-frame confidence | `repos/COVTrack/ovtrack/models/losses/cyc_loss.py:102`，尤其 111–117 | 两个方向分别 row-softmax，再相乘取对角；不能误读成一个 row-softmax 与自身转置的乘积 |
| 推理中实际调用 | `repos/COVTrack/ovtrack/models/trackers/ovtracker.py:529`、637–640 | 当前 appearance/bbox/cls 与 memo 分别计算 consistency，再传入 fusion head，融合结果用于关联 |
| 默认配置 | `repos/COVTrack/configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py:59`、172 | `FeatureFusionModule(feature_dim=256,max_fusion_ratio=2.0)` 和 `OVTrackerUncertainty` |

注意：++ 论文 Eq. (8) 将 appearance 与 fused feature 按 `c_app` 线性重连；当前公开代码采用上表中的平方 residual 强度。二者功能相近，但**不能把公开 MCF 代码直接当成 ++ 论文各公式的逐式实现**。

### 2.2 ++ 论文的机制覆盖范围

| 模块 | 主要证据 | 已占据内容 | 与本轮分类问题的区别 |
|---|---|---|---|
| MCF | [Sec. IV-B](https://arxiv.org/html/2603.24016v2#S4.SS2)，Eqs. (1)–(9)，PDF pp. 6–8 | appearance/location/semantic 联合 gates，双向循环稳定性，confidence 调制特征融合，以关联损失学习 | 给 residual 增加语义/几何置信度，或学习 cue 权重，属于其直接邻域；没有显式估计全轨迹分类错误协方差 |
| MGA | [Sec. IV-C](https://arxiv.org/html/2603.24016v2#S4.SS3)，Eqs. (10)–(17)，PDF pp. 8–9 | IoC 空间包含掩码；TAQ 双向循环分数；child feature 硬门控；parent query、child key/value cross-attention 和 residual | 部件、嵌套框、空间上下文和可靠性门控都不能笼统声称为空白；它增强关联特征，没有定义分类残差依赖 |
| TCP | [Sec. IV-D](https://arxiv.org/html/2603.24016v2#S4.SS4)，Eqs. (18)–(23)，PDF pp. 9–10 | cosine 映射到 [0,1]，与 IoU 合成边；Eq. (22) 限同预测类别；Eq. (23) 传播该类别的标量 confidence，恢复低分检测 | **恢复检测可在后续帧继续作为 source，形成多帧链**。不能以“只构建相邻帧图”声称它没有长期传播；但其传播对象不是完整类别 posterior 或分类错误协方差 |

TCP 正文还明确承认：相邻帧可稳定预测成同一个错误类别。这是“自然相关误判”的直接动机，但**不是我们已经测量其出现率、能预测其影响、或已解决它的证据**。

### 2.3 TRACT 和 OVTR

- **TRACT TCR**：原始 [arXiv 2503.08145v1, Sec. 3.3](https://arxiv.org/html/2503.08145v1#S3.SS3) 已有 appearance EMA+feature bank、confidence 分段接纳和 category bank voting。不能把类别记忆、置信度替换或多数投票包装成新模块。
- **TRACT TFA**：同文 Sec. 3.4 已有 confidence-based frame sampling、CLIP visual features、self-attention+MLP、global average pooling；TSE 使用语言属性。因此“学一个 temporal attention 权重”直接重叠。当前本地 `repos/TRACT/TraCLIP/main.py:15` 导入的 `models.track_classifier.TraCLIP` 源码目录缺失；只能确认论文机制，不能声称已经逐行排除其全部实现。
- **OVTR CIP**：`repos/OVTR/ovtr/models/updater.py:186`，尤其 220–243；调用在 `repos/OVTR/ovtr/models/ovtr.py:796`。它用 image embedding+query position 作 q/k，对 active queries 做 multi-head attention，经过 MLP/residual 更新 `query_tgt`，随后由 query 状态跨帧传递。**这是可学习特征/状态传播，不是概率 EMA**；所查路径未显式建模 posterior error dependence 或一条 released ID 内的竞争身份 posterior。

## 3. 新补查的原始文献：新颖性边界比 OVMOT 内部更窄

下表均以原论文为依据，下载文本保存在同一 live-sources 目录；检索发现不是实验复现。

| 原始来源 | 明确已有的机制 | 对新方向的约束 |
|---|---|---|
| [Neural Aggregation Network, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Yang_Neural_Aggregation_Network_CVPR_2017_paper.html)，Sec. 2 | 两级 attention 对可变长人脸集合学习加权聚合；用分类/验证损失自动抑制模糊遮挡等低质量样本 | 端到端、质量自适应、可变长、attention pooling 都不是新意 |
| [Quality Aware Network, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Liu_Quality_Aware_Network_CVPR_2017_paper.html)，Sec. 3 | representation branch+quality branch，经 set pooling 并用 softmax/triplet supervision 学质量 | `MLP(feature)->quality->weighted pooling` 已有直接先例 |
| [Probabilistic Face Embeddings (PFE), ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Shi_Probabilistic_Face_Embeddings_ICCV_2019_paper.html)，Sec. 4.2，Eqs. (5)–(8) | 概率嵌入、精度加权融合；明确指出视频帧冗余使条件独立假设失效、不确定性受帧数偏置，采用逐维最小不确定性；标量方差退化为 quality pooling | “视频冗余导致重复计证”和“用不确定性加权”均不是首次提出；仅把权重命名为有效证据也不够 |
| [Bayesian Classifier Fusion with an Explicit Model of Correlation, AISTATS 2022](https://proceedings.mlr.press/v151/trick22a.html)，Sec. 3.3，Eq. (4)，Fig. 1 | 以共享 Gamma 潜变量构造 correlated Dirichlet；给真类别条件下的分类器概率输出联合建模，监督推断 bias、variance 和 correlation；支持多个分类器 | **显式学习类别 posterior 的相关性本身也已有先例**。固定专家/class-specific 参数迁移成视频长序列不是自动成立的新意；需证明类别共享、可变长、自然误差过程的技术必要性 |
| [Temporal-attentive Covariance Pooling, NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/70efdf2ec9b086079795c442636b55fb-Abstract.html) | temporal attention+second-order covariance representation 用于视频识别 | feature covariance pooling 与预测误差 covariance 不同；只引入 covariance、二阶统计或 attention 不能形成边界 |
| [Redundancy Reduction Attention, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Chen_Zhu_Fine-grained_Video_Categorization_ECCV_2018_paper.html) | 多轮时空 attention 及 feature-channel suppression，以减少冗余、关注互补细节 | “减少冗余、发掘互补帧”是既有目标，需明确估计对象和可验证机制 |
| [Uncertainty-Gated Graph, ICCV 2019](https://arxiv.org/abs/1905.02756)，Sec. 3 | 根据 body/context connections 与身份分布，动态更新边 gate，在 tracklets 间传播身份；可监督或半监督训练；使用 cannot-link 信息 | 身份分布、外观关系、不确定性 gate 和防错误传播已有先例；方向 B 不能仅是 UGG 式 gate 的 OVMOT 改名 |
| [CLIP-VIS](https://arxiv.org/abs/2403.12455)，Sec. III-B/C，Eqs. (9)–(12) | 用历史 top-K 匹配降低 noisy frame 干扰；用 objectness/mask IoU 乘 CLIP 分类分数 | 质量乘分和筛帧已有直接先例。其分类乘数对单 query 各类别相同，不能据此声称它改变该 query 的 top-1 类别 |

因此，下面只能称“有明确空缺假说的候选技术方向”，**不作已经通过充分新颖性检索的保证**。

## 4. 必须区分的三个量，以及循环定义问题

1. **视觉冗余/密度**：两帧 appearance embedding 很接近，或某一局部邻域采样很密。
2. **自然误差依赖**：给定对象身份、真实类别、质量等条件后，两帧是否仍倾向于朝相同错误类别偏移；它需要跨多个训练轨迹估计，不能从一条轨迹的单次错误直接识别统计 covariance。
3. **有用的独立增量**：加入一帧是否提供现有记忆没有的辨别信息。重复的清楚正确信号和重复的稳定误判都是“密集”，但二者的错误风险不同；新视角也可能独立但完全无用。

若直接定义 `K_ij=sim(cls_feats_i,cls_feats_j)`，再用 K 降权或以 class agreement 定义 purity，有三个问题：

- classification embedding 已携带类别/视角偏差；相似性可能只是同类别均值，未必是 error covariance。
- 稳定的错误类别可被定义成“高 purity/高可靠性”；少数正确新视角则可能被当异常移除，产生自我强化。
- 反过来，也可能把大量稳定正确证据降权，使偶然高分错误视角决定轨迹类别。inverse density 不自动等价于最优信息权重。

应比较 `raw track_feats`、`cls_feats`、两者组合与纯时间距离。`raw track_feats` 来自 appearance head，但仍与分类头共享 detector/backbone，**不是统计独立传感器**；detach 只切梯度，cross-fitting 只减少拟合泄漏，均不能证明独立性。

最低诊断要求：先在预测可见特征上定义分组/核，再用 GT **仅评分**，在同身份、类别/质量/时间间隔可比的条件下报告 error covariance、同错类别率、以及加入新视角的收益。不能先删掉未匹配框再聚合；未匹配 observation 仍是官方 readout 的输入，其身份未知。

人工复制同一观测的稳定性只是实现性质。复制两份相同证据却给二者独立 diagonal noise，会人为增加信息；通过 duplicate test 不能代替自然数据上的机制验证。所有帧共同朝同一个错误类别偏移且没有剩余辨别信息时，仅重加权不可能凭空恢复正确类别。

## 5. 方向 A：从 base 类监督学习可迁移的共同误判过程

**问题**：同一轨迹中哪些观测共享同一种分类错误；怎样让持续出现的错误视角不被反复当作新证据？目标是学习 error dependence，不是把 attention 或 density 权重重新命名。

**最小可学习结构**：冻结 detector、boxes、IDs、分支分类 logits 和文本词表；把两个分类分支的各帧视为 `i=(t,branch)`。输入 `h_i` 为 appearance、归一化几何、时间、branch ID、可选 detached semantic features。一个类别共享、可变长网络输出低秩误差载荷及噪声：

`Sigma_phi = L_phi(H) L_phi(H)^T + diag(d_phi(H)), d_phi >= 0`。

把 `e_i(c)=log(max(p_i(c),epsilon))` 或类别相对 margin 作为观测。在多个 **base training** 轨迹上，用真类别监督预测错误、同错类别/残差关系，并以适当的 track likelihood 或 centered-residual likelihood 训练 Sigma；单帧误差均值/偏差必须与 covariance 分开估计。均值中心化模型须在独立 training folds 拟合，防止把类均值/难度误当成依赖。只回归 `b_i*b_j` 的 co-error 不是完整 posterior covariance，不能混称；还需验证错误方向是否相同。

可首先用非负最小方差读出作为透明的下游：

`w = argmin_{w>=0, 1^T w=1} w^T Sigma_phi w;  s(c)=sum_i w_i e_i(c)`。

这个 QP/GLS **是已有工具，不是创新点**。若其无法描述 class-dependent error direction，应换成共享参数的条件 likelihood，而非给每个 LVIS 类拟合专属矩阵；参数必须按候选类别关系/文本描述共享，测试时不使用真实类别。模型输出的 covariance 需要在留出类别上校准，不能只靠分类 CE 间接命名成 covariance。

**与近邻方法的具体差异**：TFA/NAN/QAN 学任务加权，不监督共同错误过程；MCF 学用于 association 的 cue reliability；MGA 聚合空间子框；TCP 提升检测标量 confidence；AISTATS CFM 已有明确相关概率融合，本候选剩余问题仅是“可变长视频观测、类别共享、自然视角/分支共同误判过程及其迁移”。若只做固定专家 CFM 或相似度核+QP，技术贡献不足。

**可证伪预测与必要对照**：

- 在留出 base 类/视频的自然观测对上，学习的 Sigma 比时间距离、appearance cosine、classification cosine 和 inverse density 更能预测 **控制质量/类别后的残差共同偏移**；若不能，停止依赖机制主张。
- 固定质量分布、轨迹长度与证据预算，收益应主要随自然错误依赖强度增加，且在 matched-single-identity 子集仍存在；若仅与低置信度相关，则属于 quality pooling。
- 与相同容量、相同 token 数的 TFA 式 attention、QAN 式质量权重、sumlog、随机/置信度采样、view-component pooling、RBF-GLS、小规模 CFM 对照；不能只胜过 TCC。
- 保留 diagonal、置乱 off-diagonal；或在 quality/semantic 分布相近的轨迹间交换 error-kernel。真正依赖机制应在这些干预后失去对应收益；单纯温度变化不能改变 top-1，不计作类别改进。
- 重复同一错误视角不应继续提高错误置信度；加入互补且能辨别类别的新视角应改变 posterior。人工试验只验证边界，最终仍需自然数据和未见类的分类改进。

**首轮结论门槛**：先证明“可预测的自然误差依赖”及其超出 density/quality 的增量，再讨论训练完整 readout。当前没有这项完成证据；现有 sumlog 增益不能作为该机制的验证。

## 6. 方向 B：将身份污染路由与语义可靠性分开学习

**问题**：语义冲突是同一个对象的分类噪声，还是 released track 混入了别的对象？把两者都压成 semantic consistency 会掩盖原因。

**最小可学习结构**：每条 released ID 保留少量竞争身份状态 `u_m` 和各自类别 posterior `pi_m(c)`。路由只读 raw appearance、motion/geometry、时间和 class-blind association alternatives：

`r_tm = softmax_m g_theta(a_t,b_t,delta_t,{u_m})`；
`q_t(c) = sum_m r_tm pi_m(c)`。

base training 的 identity labels 监督 membership/pairwise same-ID，base category labels 监督各状态的分类。**类别概率、预测类别一致性和 classification-loss 梯度不进入 identity router**；分别学习身份归属与分类质量，避免用要修复的类别结论证明自身纯净。训练需要自然产生的 mixed-ID 片段和 hard same-class distractors；合成交换/插入仅作可控补充。冻结输出 ID 的初步诊断可以只改类别读出，但如果最终按状态拆分/重关联 ID，必须另立 association 实验，不能继续宣称 LocA/AssocA 保持不变。

这仍含普通 mixture/聚类/质量加权成分。可研究的剩余点是：**在开放词表下，把“身份被污染的概率”和“该身份观测的分类错误”作为不同变量训练、校准和推断，并验证其类别迁移**。UGG 已有身份分布与关系 gate；单纯加 gate、slot 或 appearance clustering 不能算充分技术新意。

**可证伪预测与必要对照**：

- 相同 identity、不同外观/类别预测的片段应被保留在同一 identity state；不同 identity、同类别的片段应可分开。只会按类别或视角聚类则失败。
- 在固定原 boxes/IDs、只改变语义概率的干预中，identity routing 应保持不变；在保持语义分布近似不变、替换 appearance/geometry 为异身份观测时，routing 应响应。
- 收益应主要出现在真实 mixed-ID 子集；单身份困难视角不能因误路由显著退化。必须比较单状态 quality pooling、简单 class-blind appearance clustering、几何切段、以及相同容量的自由 attention。
- 先算 identity purification oracle 上界，再比较可学习 router 的 oracle gap。如果纯净化上界不足或 class-blind signals 无法预测污染，不投入完整训练。
- GT 只用于 training supervision 或 diagnostic scoring；推理时不使用 GT matching/purity，不提前删掉 unmatched observations。遇到证据不足的身份歧义应保留 uncertainty，而非把 dominant category 当身份事实。

现有独立 Hungarian+IoU 诊断中，1,461/1,720 个 rare matched observations 位于 single-matched-identity tracks，259 个位于 mixed tracks；这提示 B 的适用范围可能较窄。该“single”只限定已匹配观测，不能证明包含未匹配框的整条轨迹完全纯净；它也不是 TETA evaluator 的身份划分。

## 7. 当前优先级与允许表达的结论

优先 A 的低成本自然误差诊断；B 先做 purification upper bound 和 class-blind 可辨识性审查。两者均应在 base training 的类别留出/视频留出上完成选择，保持 novel validation 作为冻结评估；当前尚未达到新训练所需证据条件。

目前可写：官方公开树尚未找到可核验的 ++ 新模块；MCF/TFA/CIP 已覆盖普通可学习融合与时序聚合；概率相关融合和视频冗余也已有明确先例；仍可审查的假说是自然共同误判过程的可迁移学习，以及身份污染与语义错误的分离。

目前不可写：++ 没有官方仓库；TCP 只能传播一帧；CIP 是概率 EMA；相似度就是错误 covariance；GLS/attention/mixture 本身构成新方法；两个方向已通过经验验证或已获得新颖性保证。
