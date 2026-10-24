# 几何一致的 ROI 局部证据：机制、排序与证伪边界

日期：2026-10-24。范围：冻结 COVTrack 的压缩前 ROI 网格、现有四帧 pilot 与相关原始工作。状态：`HOLD_COLLECT_GEOMETRY_EVIDENCE; NO_TRAINING`。本文给机制与实验排序，不是新颖性结论、收益结论或已训练方法。

## 1. 结论

**优先研究“几何证据”，但不要把普通 attention、flow warp、part pooling 或 object memory 改名。** 可行的研究问题应写成：当同一轨迹跨越视角、尺度或局部遮挡时，能否仅用当前检测框和**过去可得的局部几何**，把历史 ROI 的可见部位送到当前部位，并让每个历史 token 的贡献受该位置的可观测性约束；最终读出在固定身份候选中超过同类异身份硬负例。

建议排序：**A（几何一致的可见 ROI 证据传输）> C（空间关系因果分解）> B（压缩前 token 的类别条件对应）**。A 先做；B 只在 A 的几何门通过后作为条件增强，因为 B 的普通形式碰撞更密，而现有类别条件实现也尚未通过独立审计。C 可并入 A 的消融，不单独立项。

必须承认的直接近邻：FGFA 已有 flow warp、逐位置 cosine 权重和空间特征聚合；VRSTC 已有邻帧局部 patch 补全、遮挡训练和身份损失；MeMOTR 已有对象记忆与 memory attention；TRACT 已有轨迹 feature bank、confidence 分段与轨迹 attention；COVTrack++ MGA 已有同帧子框/部件证据门控与 cross-attention，TCP 已有同预测类别的跨帧标量 confidence 传播；FineTrack 已有无部件标签的局部 mask 与部件身份损失。几何一致、局部、时序、可靠性分别都已有先例。

**当前没有证据支持训练。** 四帧 pilot 有 241 行，但只有 15 个合法人工身份 anchor、11 条支持轨迹、4 个同身份跨帧对。它只证明 ROI 内部有非均匀空间结构，不能证明局部 token 提供超过 raw256 的身份信息。现有缓存没有可见性、对偶局部对应和原始未遮挡 teacher；不能从 `[N,256]` 反推 `7×7` 网格。

## 2. 代码接口与观测缺口

冻结路径是 R50/FPN → RPN → `RoIAlign [N,256,7,7]` → 四层 ConvModule → flatten(12,544) → FC1024 → `fc_embed 256`。真正的模块入口在 `repos/COVTrack/ovtrack/models/roi_heads/ovtrack_roi_head.py:815` 与 `track_heads/quasi_dense_embed_head.py:98–109`；后者目前没有额外局部 attention。

tracker 只保存 fused appearance EMA、最新 semantic、最近框与时间，见 `trackers/ovtracker.py:405–440`。没有历史 ROI token、flow、可见性或多个局部 part 的记忆。MCF 输入是全局 raw、绝对框和 semantic；它不定义同一轨迹内部跨时刻的 token 对应。

四帧观察保存 `roi_input` 和 `conv_spatial` 各 `[N,256,7,7]`，但不保存原图像素局部 patch、合法 visibility、对应或未遮挡 teacher。单帧完整 FIT 帧的 detector 采集已约 2.9 小时；全量一层 ROI float16 约 15.9 GiB，两层约 31.9 GiB，二者都要写到 `/raid`。所以下一步应选固定小队列做 source-held-out 信息门，不能先建全量网络缓存。

## 3. 候选 A：几何一致的可见局部证据传输

### 3.1 对象

当前 query 的 ROI token `Z_q={z_{q,r}}_{r=1..49}`，历史已关联 track `k` 的过去可见 token `Z_{k,tau,s}`，当前框 `b_q`、历史框 `b_k,tau`、原帧差 `dt`。历史只能来自当前时刻之前且已合法关联的观测；GT 身份只用于训练，不能选择部署 memory。

允许使用的“几何”必须明确是哪种：检测框相对几何、过去轨迹的局部仿射/单应估计，或独立 GT-free 光流。只使用 box 比例缩放与中心平移，不能宣称补偿了相机运动或三维视角。

### 3.2 一个可实现的离散对应

把每个 ROI 的 7×7 token 视作归一化网格上的胞元，胞元中心为 `u_r`。对每个当前 token `r` 和历史帧 `tau` 建立映射

```text
v_{r,tau} = Pi_{q,k,tau}(u_r),
p_r,tau = sum_s phi(v_{r,tau}-u_s) * m_k,tau,s * z_k,tau,s
a_r,tau = <W_q z_q,r, W_k p_r,tau> / sqrt(d)
rho_r,tau,1..L = softmax_tau( a_r,tau + gamma * log m_k,tau,r + b_vis )
U_q = sum_{tau,r} alpha_r,tau * rho_r,tau,r * norm(W_v p_r,tau)
Z_q' = Z_q + gain * reshape(U_q)
```

`phi` 是固定双线性核，`m` 只表示过去可观测/可用，不从当前 GT 反推。`alpha` 可由检测 score 或已关联轨迹的过去置信度产生，但普通 score/confidence gate 必须作为控制；`b_vis` 只编码过去帧中的 token 可用性，不宣称真实遮挡概率。`gain` 采用双路径差分初始化：冻结参考分支和可训练分支初值相同，输出 `gain*(f_train-f_ref)`；初始残差严格为 0，但训练分支从第一步就有非零梯度。这样避免零初始化 query/key 导致第一层 attention 完全不可启动的死区。首帧必须输出 `past + current`，全 mask 时传输项为 0。

这里的新意只能在**局部位移场与 token 支持同时约束证据选择**这一可测量对象上：对应不是作用于整张 ROI 特征图的均值 warp，也不是只给同帧子框增加 cross-attention；掩码和对应共同决定哪个位置的哪份历史证据可以进入身份读出。FGFA 的逐位置对齐加权是强近邻，因此必须直接比较，不能只和 raw256 比。

### 3.3 更简洁的可执行版本

若完整双线性传输的成本或实现风险过高，先用固定 `Pi` 的**位移对齐 token 选择**：对每个 query token，在当前框坐标系中按其中心位移选过去 1–2 个最近 token，乘以可见性与 cosine 权重，再平均。这个版本只需缓存稀疏 `49×49` 近邻和 local mask，可作为无学习上界/控制；只有它显著优于未对齐平均后，才值得实现可学习残差。不能把简单几何的选择器称为新神经网络。

## 4. 候选 C：空间关系的因果分解

候选 A 可能仍被“只是多用了框/位置”解释。C 在固定身份读出下比较：只有 `Z_q`；`Z_q` + 当前与历史框的标量相对几何；`Z_q` + 历史 token 对齐；以及完整 A。关键是**改变历史位置而保持外观不变**、**改变历史内容而保持几何不变**两种干预，检查身份相似度的变化能否分别归因于几何对应和视觉证据。

这不是独立的模型贡献，而是 A 的必要诊断。若完整 A 的增益在关闭局部对应、只保留相对框几何后仍然存在，说明收益来自普通位置 cue；若改变外观、保持正确对应没有作用，则局部证据没有进入读出。C 必须使用相同的 history 窗口、候选数、参数预算和 source fold，不能利用 GT 在当前时刻选择正确历史。

## 5. 候选 B：压缩前 token 的类别条件对应

B 把类别文本 `c` 条件化到 token 选择：

```text
a^c_{r,s,tau} = g_c(z_q,r, z_k,tau,s, v_r,tau,s, m_k,tau,s)
U_q^c = sum a^c * Pi_* z_k,tau,s
```

可写论文的窄问题不是“按类别做 attention”，而是：**类别条件是否改变局部证据的跨帧对应，并在固定身份读出上超过同参数 learned-class table、普通 spatial attention 和仅类别 scalar gate。**

必须把文本置空、换成无关类别、只改变类别 embedding、改变当前外观而保持类别不变作为干预。若 learned-class table 复制全部效果，文本不是信息来源；若改变类别只改变最终 logits 而没有改变 token 支持或对应，仍属于普通分类读出；若分类提升而身份检索不升，不能声称改善关联或局部身份证据。

现有 `text_conditioned_token_evidence_20261024.py` 仍处于实现审计阶段，不能进入此排序的训练。独立审计找到零初始化 attention 梯度死区、首帧滞后、mask 与无效类残差清理问题；先修契约并通过作者/独立测试三轮，才讨论 B。

## 6. 精确对照与判定

所有对照应在**同一 source-held-out fold、同一 video/GT trajectory 分母、同一候选集合、同一合法身份标签、同一历史预算、同一训练预算**下比较；unknown 不是负例，GT 不参与部署 memory 选择。

| 对照 | 必须证明什么 |
|---|---|
| 冻结 raw256、原始分类/身份读出 | A 的收益不是新数据或 backbone/训练带来的 |
| 当前帧 7×7 global mean/attention，同参数量 MLP | 局部选择不是单纯容量增加 |
| FGFA 式 flow/box warp + cosine 加权平均 | A 不是“跨帧对齐平均”的已有结果；若只胜未对齐平均，证据属于对齐 |
| 相等历史的算术平均、EMA、last raw | 时序容量本身没有解决身份问题 |
| 同预算未对齐 token cross-attention | A 不是普通空间 attention 的别名 |
| 同预算 learned-class table / class scalar gate（B） | 类别条件不是普通类别先验或权重缩放 |
| 错误时间、错误身份历史、随机 token、随机可见性 | 正确对应与可见性是必要因果因素；错配应退化 |
| MeMOTR/TRACT 式对象 memory、feature bank + temporal attention | 不是普通长期对象记忆或轨迹池化 |
| COVTrack++ 式 MGA 同帧子框聚合、TCP 标量置信传播 | 不把已有同帧部件/跨帧置信机制重命名 |
| FineTrack/PAFormer 式 part mask/visibility | 不把普通可见性部件选择算作新贡献 |

A 的最小离线读出应报告纯外观身份检索：同视频同预测类别中，已知同身份的跨帧对相对已知异身份硬负例的 ranking、source 宏平均、repair/break 和留出 source。还需要固定候选的 cycle/second-best margin、按 source 删除敏感性；候选 pair 数量不能当独立样本。通过后才进入完整 online tracking；TETA 要分开报 LocA、AssocA、ClsA 与 Base/Novel，分类改善不得冒充关联改善。

## 7. 最小实验与 STOP 线

### 7.1 先做无训练信息门

固定采样协议，不按检测结果、IoU 或身份收益选帧。建议最小版为 6 个 FIT source、12 个视频、每个视频 6 个原帧、保留所有行；预算约 4,320 行、0.28 GiB，按现有 pilot 69.4 KiB/行估计。采集需要 frozen R50/FPN/RPN/RCNN 和 ROI 网格；每帧另存 `roi_input`、`conv_spatial`、合法 box/time、局部配准输入与可见性 mask。先把行 locator、原帧映射、NMS/断行、缺帧和重复行做成可审计契约。

最小门必须同时满足：

1. 至少 3 个 source-held-out fold；每 fold 至少 30 个已知同身份跨帧对与 30 个同预测类别异身份对；unknown 保持 unknown。
2. A 的固定几何近邻在控制 raw256、类别、框尺度、历史长度与 token 平均预算后，对纯身份检索有正增量，且多数 fold 方向一致。
3. A 超过 FGFA 式 warp-average、未对齐 average 和同预算 spatial attention；至少一个 view/scale 分层的修复数大于破坏数。
4. 错误时间/错误身份/随机对应干预使优势消失；正确对应与当前观测同时保留时有增益。
5. B 若继续，文本置换相对 learned-class table 的增量还要单独为正；否则 B 退回普通分类读出诊断。

### 7.2 失败处理

每个失败至少先做三轮、每轮两项实现核验：行/框对齐、`[N,256,7,7]` 轴与 flatten 顺序、ROI 坐标与原图坐标、mask 广播、历史是否 causal、原始帧差是否被当成样本 ordinal、是否误把 unknown 当 negative、是否使用了 GT 选择 memory、是否把量化缓存与原 GPU 路径混用、source/video 聚合是否重复计数。先排除实现错误；三类失败都指向同一机制时，才写 STOP。

### 7.3 STOP

- 合法 source-held-out 支持不足，或只有 4 个同身份跨帧对：**停止训练**，只做描述性观察。
- 局部 token 的纯身份增量由 raw256、全局池化、同参数 spatial scorer 或类别先验解释：**停止 A 的模块主张**。
- A 只胜未对齐平均而不能胜 FGFA 式对应 warp：**停止“学习对应”的解释**，保留普通几何基线。
- 错误对应/错误时间没有退化，或完整 A 的增益在只保留 scalar box geometry 时仍存在：**停止局部证据因果解释**。
- B 的文本条件只改变 logits/权重，或 learned-class table 完全复制：**停止类别条件创新**。
- 只有重建损失或分类 ClsA 改善，身份检索和官方 AssocA 没改善：**不能开在线训练**。
- 完整 tracker 只在小视频、单一 source、单一种子或单个类别上改善，或更换 fold 后反号：**停止泛化/部署主张**，保留机制诊断。
- 实现审计未通过（零梯度死区、首帧缺 current、全 mask 非零、无效类残差、参数/契约 hash 不一致）：**禁止训练**，先修复并重跑独立测试。

## 8. 论文故事与 claim 边界

可以写成的问题：*When a track spans scale, viewpoint and partial occlusion, can local ROI evidence be transported through past-only geometry and admitted only where it remains observable, improving identity evidence beyond raw global features and generic temporal aggregation?*

可以主张（在实验通过之后）：

- 在固定冻结检测与身份读出下，几何一致的局部可见 token 相对 raw256、均值/EMA、未对齐 attention 和 FGFA 式对齐平均提供 source-held-out 身份检索增量；
- 错误对应和错误时间会去除该增量，说明局部对应而非单纯的时序容量起作用；
- 若 B 也通过，类别条件可改变局部证据选择，并对同一身份读出产生与 learned-class table 不同的效应。

不能写：“首次利用时空信息”“首次 ROI attention”“首次多帧/几何/可见性融合”“任意 novel 类普遍提升”“相机运动已被补偿”（除非有独立相机估计）、“raw256 没有运动信息”“局部 token 一定优于全局特征”“分类提升等同关联提升”“pair AUC 等同 tracker TETA”“未知身份可作为负例”“四帧 pilot 已证明新颖性”。也不能把 MGA 的同帧子框、TCP 的标量传播、TRACT 的 bank/attention、FGFA 的 warp-average 作为本方法的贡献。

## 9. 目前准确的证据状态

| 事项 | 证据 |
|---|---|
| ROI 网格真实接口 | 已核对源码 `roi_heads/ovtrack_roi_head.py:815`、`track_heads/quasi_dense_embed_head.py:98–109` |
| 四帧 pilot | 241 行、2 source、2 video、15 合法 anchor、11 支持轨迹、4 同身份跨帧对；v6 只读审计 PASS，但 `identity_evidence_sufficient=false` |
| 局部 token 与 raw 的关系 | 全局均值与 raw 余弦中位数接近 0，说明表示不同；局部中心化同身份余弦约 0.850，raw 约 0.847，不证明额外身份信息 |
| 训练数据 | C-TAO 500 视频、490,210 帧、1,489,637 框、2,588 轨迹；具备 schema/core/strong 合法人工队列，但尚无 ROI token cache |
| 现有类别条件模块 | 作者/独立测试存在契约分歧；零初始化梯度死区未解决，禁止训练 |
| 官方 TETA | 已有 COVTrack 与分类读出结果；本文件候选尚无新方法 TETA |

入口证据：[ROI 方向审计](roi_module_direction_audit_20261024.md)、[类别条件文献审计](region_text_attention_novelty_20261024.md)、[时序局部补全近邻](temporal_roi_completion_novelty_audit_20261024.md)、[缓存可行性](feature_cache_module_feasibility_20261024.md)、[四帧信息门](../results/roi_token_information_audit_20261024_v6/audit_report.json)。

### 9.1 近邻的精确边界

- **FGFA**：flow warp、逐位置余弦 softmax、时序聚合；因此 A 的核心必须验证“局部对应改变身份证据”，不能把 warp 本身当新意。
- **VRSTC**：邻帧 patch 补全、空间 generator、身份 guider；因此 A 不使用“遮挡补全+身份 loss”作独立新意。
- **MeMOTR / TRACT**：对象记忆、长期 memory attention、feature bank、confidence 与轨迹 attention；因此普通历史 token attention 已经占位。
- **COVTrack++**：MGA 是同帧 IoC/子框证据的 TAQ 门控与 parent-child cross-attention；TCP 是同类别的标量 confidence 传播。它们不等同于“跨时刻同物体局部几何对应”，但不能把部件可靠性或跨帧置信传播当作空白。
- **FineTrack**：无部件标签的局部 mask、部件身份/三元损失和多样性约束；不能把普通 part pooling 当新模块。
- **RegionCLIP / GLIP**：ROI/region 与文本对齐；不能把类别文本进入区域特征或文本查询本身当 B 的创新。

本文件的“几何证据”描述是基于源码与这些原始工作的**可测试机制假说**；未完成穷尽文献检索，也没有资格称“首次”。论文最终指标必须以完整官方 evaluator、固定 source/video/GT-trajectory 分母、独立 seed/折叠和冻结 baseline 为准。
