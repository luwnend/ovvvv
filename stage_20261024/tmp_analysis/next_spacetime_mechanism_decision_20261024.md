# 下一步时空机制决策：先补全候选残差信息门

2026-10-24。状态：`DESIGN_ONLY_ORDINARY_CONTROL_NO_NOVELTY_CLAIM`。
本审计只读代码和既有设计，不读 tune/test/class-heldout targets，不改冻结协议或缓存，不启动采集、训练或全 FIT 计算。OOF 与 donor 最终方向由主审计传入，本文件未独立重算它们；最终数值及验收以主报告为准。采用既有 experimental-design 技能的成对条件、共同人口和 source 实验单位原则。

**四帧 metadata pilot 已完成；它只通过了导出/成本门。** 这是可切实推进的信息与成本门；现在不再训练 adapter、不扩展 ROI 补全网络，也不把共同位移、中位数、GMC 或残差头称作新方法。

## 1. 现有结果究竟还缺什么

主审计传入的结果为：纯 identity adapter cosine 在三 seed、两 lag 都下降；geometry 在 adapter 臂补救但仍低于 raw，raw 臂小增不稳。严格 donor-out 的源宏归一化 L1 gain 约为 lag1 `0.4465`、lag4 `0.5662`。这些结果只分别拒绝当前固定训练配置的表征改善主张、支持一个有限的伪对应位移预测性结论，不能推出“表征不能学”“改 residual ranking 就有效”或“需要 uncertainty”。

代码 `scene_translation_donorout_20261024.py:152` 首先调用 `fullpool_mnn()` 选择 h，再对这些 h 排端点/同 proposal/direct overlap、重做 donor 匹配。因此当前诊断只回答 **raw 已选出的严格 MNN h 的位移能否被其他伪对应预测**。它没有为同一 query 的任意竞争 reference 生成 residual，不能据此回答身份候选排序，也不能用 h 的 gain 给未知竞争候选补分。

目前最少缺的一项信息是：在相同 query、相同完整 retained reference 池、相同合法 P/N 投影下，严格 donor residual 是否对 raw 的实例区分提供增量。若它不能区分正负，进一步做更大网络只会把“输入无效”和“利用失败”继续混在一起。

四帧 ROI trace 仅证明真实 `[N,256,7,7]` 采集、行映射和 native 全 batch FC 重放；它没有补上局部对应、可见性或受控 masked-image 前向，也没有身份增益。`temporal_roi_completion_novelty_audit_20261024.md` 已停止泛化的 temporal completion＋teacher＋ID 创新说法，不能将四帧采集成功重新解释为该路线成立。

## 2. 当前 exporter 范围合理，契约必须保持窄

对帧对每一条 edge `e=(q,j)`，先由 GT-free locator 和框计算两侧端点、同 proposal、直接正面积 overlap 排除集合，再在**同一次完整 float64 canonical raw-cosine 矩阵**的剩余行列上重新做严格 MNN；至少三个 anchor 才计算逐坐标 median 像平移。随后用自己的两端框几何平均宽高归一 displacement 和 innovation。不得先 fullpool matching 再删除命中端点的 anchors。

| 最少保存内容 | 用途与边界 |
|---|---|
| immutable query/reference row、frame/原帧 gap、输入与规则 SHA | 完整 Q×M 和已有缓存的一一映射；不能只保存 known、MNN、raw-error 或可修复行 |
| raw cosine、static normalized L1、strict corrected normalized L1 | 分开外观、未补偿空间和补偿空间信息；gain 只是后两者差，不能代替二者 |
| strict available、anchor count、缺失原因、translation、normalized innovation | unavailable 留缺失与 count，不伪装零运动；最终网络若用零填充，必须另保留 mask 并声明只是编码 |
| unique exclusion sets 与 donor anchor locator hashes/可重建 receipt | 证明先排后匹配、可独立复算，不为每个 edge 重复巨量 JSON anchor 列表 |
| MAD、leave-one-anchor-out shift（可选诊断） | 支持分散/稳定性描述；不是匹配正确率、校准 uncertainty 或相机运动真值 |

memoize 只能复用同一帧对、同一 canonical matrix、同一精确两侧 exclusion locator 集合、同一拟合规则下的 fit/anchor 结果。即使 exclusion 相同，不同 edge 的自身 displacement、size scale、innovation 仍须重新计算。不能以相同 anchor count、相近 box 或 fullpool anchor 列表替代 key。benchmark 同时报总 edge、unique exclusion keys、实际 fit 次数、峰值内存和时间，不能只用 memoize 后最好情况外推。

本 exporter 使用独立的 metadata 选择规则：字典序首个 sealed FIT source 内，字典序首个至少有四个缓存帧的 video，取 prefix `0..3`，构造一个 video 的三组 past-adjacent 帧对。它不是前述 ROI 观测的两个 video 各两帧设计；两个 pilot 不混用人口或成本。相邻 sampled observations 的原帧 gap 仍须逐项记录，不能冒充同一物理时间间隔；四帧不提供 lag4 帧对或成本/支持结论。只做 shape/数值/退化/成本验证，不读这四帧 identity 后挑选“好例子”，不把 pilot 报成全 FIT 支持率。

**严格排除的独立性边界：** direct overlap 排除集合本来依赖 endpoint box。给定固定排除集合时，donor fit 不再直接读取已删端点的 cache 数值；但改变 endpoint box 可能改变排除集合，共享 backbone 也可能把 endpoint 视觉信息带入其他 ROI。这里不建立图像因果独立、统计独立或 physical-object 完整排除。

## 3. pilot 先做三轮各至少两项实现检查

1. 独立 scalar oracle 对完整 Q×M 数值/locator；将既有 h 投影到 full Q×M，核其 strict fit/innovation 与旧 donor 契约一致。
2. 端点竞争、同 proposal/direct-overlap alias 的先删后匹配反例；空帧、零 raw、严格 tie、少于三 donor 的缺失语义。
3. memoize on/off 的逐 edge 完全一致与实际 key 覆盖；输入内存布局/immutable-row 排列后映回原 locator 的不变性。

实现通过、真实进程 exit0、输入/产物/保护 SHA 链齐全后，pilot 仍只到“可导出”。先依据实测 unique-key 成本冻结全 FIT 资源上界，再进入下一独立协议；不因 CPU 时间长自动改变人口或只计算合法标签行。

## 4. 后续最小信息门与最强普通控制

固定 raw appearance，不加 identity adapter、CV history、semantic 或新 memory。下一阶段所有条件共用原完整 retained 候选与 source folds；在 GT-free 导出全局封存后才 join 合法 FIT 身份。主条件沿用原 hard-query 人口与跨 lag 固定共同 query，source→trajectory→query 聚合、完整 source 删除敏感性和 repair/break 均保留。unknown 不作负例，没有合法 positive 不等于 no-match。

最小对照为 raw cosine、未补偿 displacement、严格补偿 residual，以及只供 time/missing/anchor-count 等支持描述的同容量控制。方向失配可事先固定为 translation 取负、保持原 support/mask；实际数值变化必须报告，不能从多个 placebo 中选择最差的一个。各臂完整人口保留缺失；主读出不按某臂成功与否换分母，另报预声明共同可用 P/N 的成对信息读出。

先报告固定几何单 cue 的 P/N 顺序、相对 raw repair/break 和支持覆盖；这回答 residual 有无可观察的实例区分。单 cue 有少数 repair 不保证联合收益。若进入学习，最强普通参考应包含**同监督/同预算的简单 additive association head**和既有容量 residual MLP，分别使用未补偿/严格补偿信息。每臂重新匹配训练输入分布；把新 residual 塞进旧 geometry head 只测输入扰动。训练只用各 fold 的 train source，不选 seed、lag 或新 checkpoint；这仍是已看 FIT 后的探索性 OOF。

另一个有价值但不阻断当前 exporter 的普通诊断是：两竞争 references 比较时，统一排除 query 及两 references 的全部 aliases，只拟合一次 translation，再给两候选相同 fit。逐 edge 排除会使不同假设拥有不同 fit、不同可用性；共享 fit 检查“相对证据”是否主要来自 estimator 随候选变化。它不保证统计独立，也不是新颖性。必须在 GT-free 比较集合上定义，不能看哪条是 GT-positive 再定排除集合。

若真实 residual 不超过 support-only/固定失配控制，停止正确空间对齐解释。若只比静态位置好、仍不能给 raw 增量，结论停在共同像位移预测，不能宣称跟踪改善。即使正向，也还需独立闭环 tracker/memory/assignment 和官方评测，不能由 pair 检索替代。

## 5. 暂不引入 uncertainty 或新表征网络

OOF 负结果没有识别唯一原因：合法支持、pairwise surrogate、normalizer、优化/容量和输入几何都可能参与。MAD 小、donor 多、LOO 稳只证明共识；同向物体运动或稳定错配可以同时满足这些条件。若以后要让 confidence 控制何时改 raw 排序，需单独验证“可部署预测量能否在未训练 source 上预测修复相对破坏”，并与恒定权重/同容量 support-only 比较；不把 cycle consistency 或 MAD 直接命名为 calibrated uncertainty。

目前没有建立足够的新机制。**可继续推进的是全候选 scene residual 的普通信息门；可发表贡献仍须由明确的剩余失效对象、独立机制和强控制产生。** ROI 局部路线若重启，必须另补独立对应/可见性、真实遮挡前向和相同信息/预算的 warp-average 等近邻控制，不能靠 attention、past-only、teacher 替换或换名字越过既有 STOP。

本审计直接核对 `scene_translation_donorout_20261024.py`、`endpoint_excluded_scene_motion.py`、`relative_track_geometry.py`、`roi_track_spatial_trace_20261024.py` 和实际 work-copy MCF/cycle/tracker 路径；并沿用 `spacetime_next_mechanism_design_audit_20261024.md`、`network_spacetime_next_information_audit_20261024.md`、两份局部补全/外观输运近邻审计的明确边界。未做新的宽泛文献检索，未确立任何首创性。

## 6. 四帧 pilot 实测结论（2026-10-24）

修正后的 runner/protocol 经两位独立复核后执行一次；第一次入口失败只因 synthetic receipt 字段读取错误，worker未启动、缓存未读，已作为实现错误修正并保留历史。最终 protocol SHA `dfd21130…`，runner SHA `dd23845a…`；父进程实际等待 `4.422678s`，worker `actual_exit_code=0`、`timed_out=false`，7保护与18 pins保持不变。

固定 ArgoVerse source `...033e2`/video1271，prefix 0–3（retained 50/54/56/53），三组 past-adjacent pair 的完整 Q×M 为 2,700/3,024/2,968，共 **8,692 candidates**。三组均 100% `available`（每个 candidate 至少3 donor）；unique exclusion fit 数为 2,064/2,496/2,288，单组计算 0.796/1.081/0.881s，worker峰值 RSS 约 60.7MiB。数值 arrays 压缩后约 0.17–0.20MiB/组，但完整 fit provenance JSON 为 2.44/3.54/3.16MiB/组。

因此不能把四帧速度线性外推成全 FIT 可接受成本：现有全量 lag1/4 约 55.35M candidates，若保留每 edge 的完整 anchor tuples，存储会超过当前磁盘余量，Python逐候选 fit也会成为小时级任务。全 FIT 前必须另立**流式、轻量 provenance/hash**协议：保留全候选数值/availability/count与可重建 exclusion-key摘要，按 frame-pair 分块落盘，禁止复制全 anchor JSON；先做资源上界与磁盘门，再决定是否执行。当前不启动全 FIT。独立完成审计收据 `tmp_analysis/all_candidate_scene_residual_four_frame_cpu_independent_completion_audit_20261024.json` SHA `0f3d302a…`，简报 SHA `1d73458a…`；未读取 identity/GT。

本 pilot 没有打开身份、类别或 GT，也没有回答 residual 能否改善同一 query 的合法 P/N 排序；四帧也不能支持 lag4 或泛化结论。
