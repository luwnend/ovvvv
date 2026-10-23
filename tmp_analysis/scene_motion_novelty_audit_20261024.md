# 候选两端排除后的场景运动与外观传输：原文机制审计

审计日期：2026-10-24。结论限定于下列原论文与所查官方实现，不等同于穷尽检索。仅新增本报告和独立来源副本；没有读取新的 OOF 留出评估，没有改动 OOF pins，也没有训练或采集实验数据。论文全文获取使用 [Hugging Face Papers 技能](/home/chenhc/.codex/skills/huggingface-papers/SKILL.md)，公式页核对使用 [PDF 技能](/home/chenhc/.codex/skills/pdf/SKILL.md)。来源与哈希见 `scene_motion_sources_20261024/manifest.json`。

## 1. 判断

**宽泛新颖性主张应停止；只值得保留一个小型可辨识性诊断。** 共享运动补偿、其他目标的相对运动参考、含错配的稳健模型拟合、按观测及模型误差定义兼容度，以及互为近邻配对后的正交嵌入对齐，都有明确原文先例。不能把这些成分组合后直接称为新机制。

在本轮已读方法中，尚未核实如下完整规定：**对每个待评分候选，先从支持池移除两端及可追溯历史，再构造外观对应与拟合共享变换；以独立支持的预测残差/不确定性给位置兼容度，并在支持不可识别时输出缺失。** 这是有待查证和测试的程序边界，不是已证明的文献空白。删除观测、留一法和外部预测本身也不是新思想。

唯一建议继续检验的差异是：**严格候选排除后，其他实例是否还包含能泛化到被排除对象的运动增量。** 若支持只在候选参与配对/拟合时显得有效，则应停止该方向。

## 2. 被审计的候选及必要边界

对候选 `(当前检测 i, 历史参考 j)`，先移除当前端、参考端，以及由预测可见 lineage 能追溯到二者的历史，再在剩余 raw appearance 上建立对应。令支持为 `D_-ij = Match(A \ endpoints/history)`，共享运动为 `theta_-ij = Fit(D_-ij)`；最后评估被排除候选的位置残差。

- 移除必须早于 MNN、softmax/归一化、attention 和共识拟合。仅删去最终 `(i,j)` 匹配边，不能消除它对最近邻竞争、候选选择或历史状态的影响。
- 推理只能用可见的 observation/预测 ID 关系；released ID 可能已混入异身份，不能声称达到 GT 身份独立。若当前检测没有已知历史，要写清实际可排除范围。
- 当前帧其他检测可用于当前关联，属于因果可用信息；这不等于仅使用过去帧。
- 缺少支持、支持退化、匹配不稳定或模型失配时保留 missingness；不能默认将单位变换解释为“测得无场景运动”。缺失本身可能与场景难度相关，需单独报告 coverage。

## 3. 直接近邻与已占据机制

| 原文和已读位置 | 已有机制 | 对本候选的边界 |
|---|---|---|
| [RMOT，Yoon 等，WACV 2015](https://faculty.ucmerced.edu/mhyang/papers/wacv15_mot.pdf)，Secs.3–5、Eqs.(1)–(15)、Algorithm 1 | 以对象间位置/速度差建立 relative motion network；由其他已跟踪对象预测目标；相对模型权重随观测更新，联合运动/外观/大小做 Hungarian 关联 | **最直接的运动近邻**。“其他实例帮助预测目标”早已有之。它的相对模型显式含目标历史，并总保留 self-motion；Eq.(11)依据被评分观测选择相对模型。不能等同于本候选的两端历史均排除后，从其他对象之间独立匹配和拟合公共变换 |
| [BoT-SORT](https://arxiv.org/abs/2206.14651)，Sec.3.2、Eqs.(5)–(9)，Sec.3.3、Eqs.(10)–(13) | 图像关键点/光流对应、离群点处理、RANSAC 估计仿射运动；修正 KF 状态和 covariance；EMA appearance 与 IoU/cosine 关联融合 | 共享运动补偿、稳健对应和运动/外观融合已有。主要估计背景图像运动，与其他目标 embedding 对应不同；显示的 `P'=M P M^T` 传播已有状态 covariance，不能单独证明显式包含估计变换参数的不确定性 |
| [OC-SORT](https://arxiv.org/abs/2203.14360)，Sec.4.1–4.2、Eqs.(6)–(8)、实现说明 | 重关联后按两端观测生成虚拟轨迹以重更新 KF；用真实历史观测方向建立 OCM；OCR 从最后观测恢复目标 | 观测中心的历史、方向一致性及其噪声问题已有。它主要使用目标自身轨迹；ORU 的后验回填不是当前候选排除后的外部运动预测 |
| [Deep OC-SORT](https://arxiv.org/abs/2302.11813)，Sec.3.2、Eq.(1)，Secs.3.3–3.4 | 相似变换修正 OC-SORT 历史观测、位置/速度/covariance，并在 KF 外推前应用；置信度自适应 EMA 与行列可区分性控制 appearance 权重 | “修正整个历史后关联”“置信度自适应融合”均已有。本文使用图像运动估计，而非独立其他对象匹配；其 covariance 同样不能直接当作支持拟合参数误差的完整预测 covariance |
| [TransMOT，Chu 等，WACV 2023](https://openaccess.thecvf.com/content/WACV2023/papers/Chu_TransMOT_Spatial-Temporal_Graph_Transformer_for_Multiple_Object_Tracking_WACV_2023_paper.pdf)，Secs.4.1–4.4、Eq.(3) | IoU>0 构图并以 IoU 加权空间 attention；图卷积汇聚邻居；沿轨迹时间 attention；当前候选图与历史图交叉解码，输出扩展 assignment matrix | 跨目标空间上下文和历史学习已占据。已读公式没有构造候选排除后的公共变换或外部预测误差；其候选与历史是关联计算本身的输入。不能把普通图/时空 attention 当剩余技术贡献 |
| [COVTrack++](https://arxiv.org/html/2603.24016v2#S4.SS2)，Sec.IV-B、Eqs.(1)–(9) | appearance/location/semantic 联合 intra-frame gates；分别 row-normalized 的双向 temporal cycle；置信度控制融合与 residual 重连 | 多 cue、时序互证和可靠性融合已有。不能用“新增几何置信度”概括新意；该节未给出从其他独立对象拟合公共运动模型的过程 |
| [COVTrack++ MGA](https://arxiv.org/html/2603.24016v2#S4.SS3)，Sec.IV-C、Eqs.(10)–(17) | IoC 包含关系；TAQ 双向 cycle 质量门控；parent query 对 child key/value cross-attention；残差增强 parent | 部件/嵌套框与可靠性门控已有。它为同一 parent 身份聚合 contained child evidence；与从其他不同对象拟合公共变换在估计对象上不同。但这种区别不证明后者新颖 |
| [Fischler & Bolles，RANSAC，1981](https://www.sri.com/wp-content/uploads/2021/12/ransac-publication.pdf)，Sec.II、尤其 II-A，印刷 pp.383–384 | 最小子集拟合、共识检验、重拟合；兼容误差同时考虑 datum error 与 model error，可用解析或扰动来设容差 | **稳健对应+不确定性兼容度也不能泛称新颖。** 普通 RANSAC 可让被评分点参与抽样、共识或最终拟合，不提供严格候选排除保证 |
| [Cook，1977](https://doi.org/10.1080/00401706.1977.10489493)，Sec.2、印刷 pp.15–16；另见原文[镜像转录](https://www.scribd.com/document/249357207/Cooks) | 删除一个观测重新估计参数，衡量其影响；结合 residual 与预测/残差方差，讨论删点后 rank 不足 | deletion/influence 原理早已成熟。它不是 MOT 两端历史排除算法，也不能据此直接证明本候选有效。出版社/大学 PDF 获取失败；本轮读到原文正文转录但部分公式缺失，因此不援引其缺损公式作逐式证明 |

BoT-SORT 官方 [tracker/gmc.py](https://github.com/NirAharon/BoT-SORT/blob/main/tracker/gmc.py) 已另外逐行检查：`applyFeaures` 在关键点检测前将 detection boxes 置零；所查 `applySparseOptFlow` 则调用 `goodFeaturesToTrack(...,mask=None)`。各分支不能混写。初始化/支持不足时的单位矩阵返回是工程 fallback，不是可信预测证据；本地副本哈希随 manifest 保留，未将当前 main 当作永恒固定版本。

COVTrack++ 实现边界沿用本地原始来源快照核对：当前所查官方树可核验旧 COVTrack MCF，但尚未找到独立可识别的 ++ MGA/TCP 实现；不能写“没有官方仓库”。旧代码平方 residual 强度与 ++ Eq.(8) 线性重连不同，不能声称逐式实现一致。本轮重读了 IV-B/C 公式快照，未重新做整个远端仓库枚举。

## 4. 运动可辨识性与预测不确定性

从前景中心对应只能观测组合运动：`y_k = T_theta(x_k) + u_k + eps_k`。其中对象自身位移 `u_k` 可以与公共平移互相抵消；车流/人群同向运动、相机运动、深度视差和不同类别运动均可能给出相近对应。稳健共识只寻找占优势的可拟合成分，不能赋予它“相机自运动”的因果身份。应称“其他对象的共享图像运动预测”，除非另有背景/深度/位姿证据。

位置预测的不确定性至少要区分：候选自身运动变化、候选检测定位误差、支持匹配错误、有限支持造成的变换参数误差、公共模型失配。若采用局部线性化形式，`S_pred` 可含 `J_theta Cov(theta) J_theta^T`、参考/检测误差及模型残差；它的校准仍需外部预测验证。把 donor residual covariance 直接除以 donor 数，只估计公共均值的误差，会漏掉新目标自身运动的预测方差，易产生虚假的高置信度。

可使用 donor-out 残差检查模型是否能外推到未参与配对/拟合的对象；其删除同样必须早于配对。仅在同一 inlier 集上算小残差不能验证外推可靠性。不同 donors 共享图像和 backbone，不是独立样本；像素尺度、gap 和候选框尺度也必须明确。仿射/单应模型额外需要空间秩与覆盖，支持数量本身不充分。

## 5. 原始 embedding 的公共外观传输

该附加候选是：从其他对象 raw embedding 的 mutual matches 拟合共享正交或低秩变换，再用于被排除候选。**不能以此主张新的基础对齐机制。**

- [Conneau 等，ICLR 2018](https://arxiv.org/abs/1710.04087)，Sec.2、Eq.(2)、Sec.2.2 与 3.4：先建立高质量互为近邻词对，再用 orthogonal Procrustes/SVD 拟合共享映射；CSLS 减少 hubness。其数据域不是视频，但“互为近邻锚点→共享正交映射”是直接算法重叠。该论文同时明确正交变换保留内部 dot products/距离。
- [Grave、Joulin、Berthet，AISTATS 2019](https://proceedings.mlr.press/v89/grave19a/grave19a.pdf)，Secs.2.1–2.4、Eqs.(1)–(2)：共同求解点对应与正交变换，`min_(Q orthogonal,P permutation) ||XQ-PY||²`；还讨论非凸性和初始化。自动共同匹配和传输同样已有，不应包装成视频中特有的新思想。

**本数据条件下首先缺少可辨识证据，而不是缺少一个 SVD 实现：**

1. 若维度为 256、支持仅数十个对象，则 donor span 的秩远低于 256。存在多个正交映射在 donor span 上同样吻合、在被排除目标方向上不同；完整旋转不可由这些锚点唯一确定。单位映射补全、近单位正则或低秩 nuisance 子空间都是新加入的先验，需验证，不能叫观测到的公共外观漂移。
2. 视频 appearance 改变可来自每个对象各自姿态、遮挡和视角，未必存在类别共享线性变换。MNN 的正确性又依赖未经修复的 appearance，易只挑出原本稳定对象。必须测试拟合对被排除对象的预测价值，不能以 anchors 重建变好作为证据。
3. 对列向量应用**同一个**正交 `Q`，`(Qf_a)^T(Qf_b)=f_a^T f_b`。同参考时刻的整个 bank 若共享一个 `Q`，Gram 保留；不同参考时刻若使用不同 `Q_a,Q_b`，则出现 `f_a^T Q_a^T Q_b f_b`，没有相同保证。
4. 即使参考时刻相同，若每个候选因排除不同端点而拟合 `Q_-ij`，跨候选输出也不存在一个公共转换后的 bank；不能声称其整体 Gram 被保留。只有同一个变换同时作用于被比较向量时，恒等式才成立。
5. 单位 `Q=I` 也完全保留 Gram；这个代数性质不能证明传输改善关联。一般低秩线性映射还可能直接改变 Gram。保持 pairwise geometry 与预测某个跨时刻 query 的对应是不同要求。

因此 broad feature prediction 继续 HOLD；不以已经失败的 per-track angular prediction 作为公共 nuisance 存在或不存在的证明。最小可继续事项只是 GT-free donor-out transport 的可辨识性检查，先看跨对象预测与秩/覆盖，尚不支持投入可学习新模块。

## 6. 一个可证伪的差异实验

目标只检验“其他实例提供候选之外的可泛化运动支持”。固定候选集合、raw features、gap、支持预算、模型族和评分规则，比较：

| 对照 | 删除位置 | 要揭示的问题 |
|---|---|---|
| Full-pool | 不删除候选 | 表面上界可能包含自我确认 |
| Edge-only | 配对后只删候选边 | MNN 竞争/选择的残余影响 |
| Endpoints-first | 配对前删两端可见 observations | 当前候选直接影响 |
| History-first | 配对前进一步删可追溯历史 | 通过历史回流的影响 |
| Shuffled donors | 在相同支持数量、gap、空间覆盖等条件下置乱 donor 对应 | 结果是否只是支持规模/难度选择，是否真需要正确对应 |

先做无 GT 构造检查：任意扰动被排除端点的 appearance/geometry 后，支持集合及拟合参数必须不变；候选兼容度可因候选本身改变而变化。若可追溯历史仍进入配对、温度、归一化或 donor 可靠性计算，这个检查应失败。

再用预先隔离的开发数据，仅在构造完成后评分 donor-out 预测误差、uncertainty coverage 与候选排序增量；按 support/rank/空间覆盖/gap 分组，报告 missing 比率及全候选结果，不能只保留高质量 inliers。任何阈值、模型阶数和取舍都不依据当前新 OOF 留出结果选择。

停止条件：严格排除后收益消失；matched-support 的随机对应同样有效；预测 uncertainty 对实际误差不校准；适用 coverage 太低或只覆盖 appearance 已容易的候选；外推只能由候选自身参与拟合产生。即使通过，也只能支持“外部支持具有预测信息”，不能单独证明相机运动分离、外观 nuisance 可辨识、基准追踪指标增益或方法新颖性。

## 7. 尚未验证的内容

未穷尽 relation-based MOT、候选删除/交叉拟合匹配和 nuisance transport 全部文献；没有运行上述真实数据实验。Cook 官方/大学 PDF 端点不可达，镜像原文转录公式不完整，因此只确认删除影响的原理。TransMOT、RMOT 和 Procrustes 的方法原文及关键公式已读；RMOT PDF p.3、TransMOT p.4、Grave p.3、RANSAC p.3 已渲染并检查，contact sheet 位于来源目录。不存在“未找到完整先例所以确认新颖”的推论。
