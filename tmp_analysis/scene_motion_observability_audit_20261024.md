# 场景共同运动的可观测性审计（2026-10-24）

状态：`OBSERVABLE_SHARED_TRANSLATION_DIAGNOSTIC_ONLY`。现有冻结缓存足以构造“由其他物体的原始外观匹配估计共同图像位移”的无 GT 诊断；**不能据此把该位移认定为相机运动，也尚无有效性或新颖性证据**。最小可行机制是端点排除后的稳健平移，而不是从目标框直接恢复一般单应性。本次没有训练、评分或改动既有源码、协议、运行文件及 pins；只新增本文件。

## 1. 实际读取范围与证据等级

- 显式读取冻结 COVTrack 源码、collection 配置、GT-free inference metadata、source roles、frame execution metadata，以及一个 FIT NPZ 的指定无 GT 字段。
- `full_collection_plan.json` 只 lexical 定位并解码 `video_records`，再按 `split=fit` 与原 roles 限制 source；没有解码该文件的 manual-anchor annotation IDs。
- NPZ 只加载框、raw track features、路径/尺寸和帧 provenance；只列出其他 ZIP member 的名称，没有解码 posterior、类别、released IDs 或身份标签。
- 没有读取新 OOF metrics、模型输出、tune/test/heldout identity values。一次范围过宽的 `rg` 检索触达 `manual_anchor_targets.json`，仅返回匹配 `supervision_source` 的键行，没有提取或解析其框/track identity values；随后收紧为显式文件 allowlist。不能把这次操作描述为“从未触达 supervision 文件”。
- 下文 population/时间/路径数字为本次实际 metadata 检查；单 NPZ schema/非零向量检查只覆盖示例，不能外推为全缓存特征质量审计。没有执行实际 donor matching，故 donor 数量、独立性、正确率和 residual 改善均未知。

## 2. 冻结接口确实提供什么

| 可观测量 | 冻结来源与源码位置 | 可作出的判断 |
|---|---|---|
| 原图路径、原始尺寸 | `work/covtrack_base_error_20261023/ovtrack/base_error_dump.py:154–161` 保存 `filename/canonical_filename/ori_shape/image_metadata_json` | 图像仍可访问；背景配准是另一个尚未实现的分支，而非现有缓存中的背景运动证据 |
| 原始帧索引与 sampled step | `tmp_analysis/prepare_base_error_full_collection.py:200–224` 保存原始 `frame_index`，将 `frame_id` 重编为采样步 | lag=1/4 不是原视频 1/4 帧，也不是秒 |
| 原图像素中的 xyxy 框 | `ovtrack/apis/inference.py:104–106` 使用 `rescale=True`；active ROI 的 `ovtrack_roi_head.py:3131–3140` 定义该坐标；`:2962–2980` 只为 ROI feature extraction 乘回 scale factor | 共同平移应先在原图像素中估计；不得把 resize 后坐标与原图混合 |
| 融合前 raw appearance | `ovtrack/models/mot/ovtrack.py:209–218,227–234,253–261` 将 ROI 原始 features 交给 tracker 并另行交给 dump；`base_error_dump.py:138–147,166` 保存 keep 对齐的 raw 256D | 可以另算纯 raw cosine；不能把这些特征当作与 detector/selection 独立的传感器 |
| 原始框、proposal 与 retained-row 映射 | `base_error_dump.py:125–153,199–211`；坐标在 tracker 前后相等，允许 score 被平滑 | 可排除同 proposal 的别名行；retained rows 不等价于不同物理物体 |
| released-ID 定位的历史 | `tmp_analysis/identity_spacetime_oof_data_20261024.py:177–191` 对唯一非负 released ID 拷贝过去框/时间/locator；重复与 -1 明确缺失 | 只可作过去位置 locator；不是 donor identity truth 或候选筛选条件 |
| 外部 warp 接口 | `tmp_analysis/relative_track_geometry.py:1–6,44–65,75–108` | 只接收 externally supplied warp，没有 camera estimator；使用 CV 时还必须有 previous→memory warp |

上表除 `tmp_analysis` 路径外，`ovtrack/...` 均相对 `work/covtrack_base_error_20261023/`。collection 实际类型为 `OVTrackRoIHeadUncertainty` 和 `OVTrackerUncertainty`（`full_collect_v1/collection_config.py:40,176`），不是同文件内其他 slack/Kalman 类。

现有 tracker 会先 remove distractor，再用 appearance/bbox/class 的跨帧 consistency 形成当前融合特征（`ovtrack/models/trackers/ovtracker.py:512–540,637–658`），并将融合后的 features 写入 momentum memory（`:405–440,640,688–690`）。因此本诊断必须从 dump 的 raw features 重算 donor，不能把既有 IDs、融合 memory、匹配分数当作“纯外观独立锚”。collection 的 `vis=False` 且禁用 filename2ann（`tmp_analysis/collect_base_error_pilot.py:139–140,174–178`；model `ovtrack.py:186–188`），GT visualisation branch 不是可用输入。

## 3. FIT metadata 的本次完整检查

按原 roles 限制得到 **341 sources、403 videos、13,791 frames、613,536 retained rows**。全部 13,791 个 FIT 原图路径当前存在；同视频尺寸未变化；inference 中这 403 个视频均没有 fps 字段。本次只验证路径存在，没有读取图像像素、核对图像内容或计算背景特征。不能以原始 frame gap 推断秒数。

| 精确 sampled lag | frame pairs | 全 Q×M candidates | 原始 frame gap min / Q25 / median / Q75 / max |
|---|---:|---:|---|
| 1 | 13,388 | 29,096,237 | 15 / 30 / 30 / 30 / 690 |
| 4 | 12,185 | 26,255,927 | 60 / 120 / 120 / 120 / 900 |

这些 FIT 帧均非空，但最小机制仍须保留空帧/缺失分支，不能改用 nearest-nonempty reference。

排除一个 query row 和一个 reference row 后，`min(Nq−1,Nr−1)` 只是 donor 数量的**乐观上界**。该上界小于 3 的 frame pairs 在两种 lag 各有 8；小于 4 的 frame pairs 分别有 26/27，对应 787/926 个候选 pair。多数帧有足够原始行数，不能推出存在多个正确、独立或静止 donor。proposal 别名、几何重叠、严格互选、外观错误等约束尚未应用。

示例 FIT NPZ `full_collect_v1/dump/v0_i0_f0.npz` 的 SHA 为 `e766c3cbad1d92f0720a290f65b5a56db45a9c24f2248f269023974c6f729320`；实际 `det_bboxes=[60,5] float32`、`track_feats=[60,256] float16`，`ori_shape=[480,640,3]`，路径存在、框宽高为正、raw 零向量为 0。ZIP 中没有背景 keypoints/flow/warp/camera/depth 参数。

## 4. 最小可行机制与严格端点排除

以下是可预注册的最小诊断定义，**没有实施，也不构成更改现有模型的建议**。

1. 对精确 reference→query 帧对，保留全部 Q×M 待评分 candidates。每个候选 `(q_i,r_j)` 单独构造 donor pool；先删除 query 的 `i` 与 reference 的 `j`，以及同帧 `roi_proposal_row` 相同的别名行，再计算任何匹配、归一化或 confidence。不能先在全矩阵算 softmax/MNN 后只删最后的 endpoint match；endpoint 对其他行的竞争仍会残留。
2. 为严格版本再排除与对应 endpoint 框有正面积交集的 donor rows，且在 matching **之前**做此操作；这会保守地删去拥挤但真实不同的物体，必须报告损失的支持率。该约束确保框区域不重叠，不能保证 CNN receptive field 或检测选择独立，也不能凭无 GT 观测保证排除了该物体所有潜在错误别名。
3. 对剩余有限、非零 raw 256D 向量做 L2 normalize，以 cosine 算 strict mutual nearest neighbour。双方 maxima 必须唯一，ties 明确标记缺失；不使用类别/posterior/released IDs/GT，也不使用 bbox motion 来选择匹配。不额外引入可调 cosine、IoU 或 residual 阈值。每个 donor pair 一对一，重复 proposal 不可贡献多份投票。
4. 至少 3 个 donor matches 才形成 translation estimate：先在原始像素中计算 `d_k=center(q_k)−center(r_k)`，再取两个坐标的中位数 `t=median_k d_k`。这是 robust **shared image-plane displacement**，不是已知 camera transform。记录 donor count、frame gap、donor spatial span、位移分散程度、别名/重叠排除计数和匹配 ties；这些是可观测支持，不能事后用正确率选择阈值。
5. 针对 `(i,j)` 的最小新观测是 `g_ij=(center(q_i)−center(r_j)−t_−ij)/sqrt(WH(q_i)·WH(r_j))`，分 x/y 归一化。先估计像素 translation，再用该候选自身框尺度归一化；不能先把不同尺度 donor 位移归一化再直接 median。translation 不提供 zoom/rotation/size-change correction。
6. 供体不足、零向量、非唯一匹配、无参考等返回显式 `available=False` 与原始几何 fallback，仍保留全部 candidates。不得以 correction availability、donor category、距离或估计的一致性来删候选。fallback 与“实测零共同运动”必须分开记录。当前 frozen normalizer/head 的输入语义/pins 不变，不能把新几何写入正在运行的 OOF。

仅 displacement 的最小版本无需 released-ID history。若以后使用 compensated CV，则还要为每个 memory 的 previous→memory 区间独立估计 warp，并在该区间排除该待评分轨迹的可定位历史 endpoints。memory 的过去时间可以不同，不能共享错误时间区间的 warp。`relative_track_geometry.py:97–108` 已明确拒绝缺失历史 warp 却称 compensated velocity 的用法；`:119–128` 的静态 predicted IoU 在无 CV history 时仍可有值，caller 必须显式 mask。现有 pair_geometry 只提供每 memory/shared frame warp，而上述严格 `t_−ij` 随 query 与 reference candidate 变化，**不能用一个全帧 H 一次调用来表示**；未来若实现，需显式新 pairwise 层或逐 query 对齐接口，不能悄悄复用不相容的 API。

## 5. 多锚、退化与不可识别边界

- 平移在数学上一个匹配就能算；多个 donor 的价值在于对异常匹配与共享性的检验。至少 3 个也不证明独立：三个框可能是一个物体的别名，或同一车队/人群。raw-MNN 的 mutual 性只约束外观排序，不证明 identity correctness。
- 所有目标共同运动时，观测 `d_k=camera_component+object_component_k` 无法唯一分解；给 camera component 加任意向量、同时从各 object component 减去该向量，目标框观测不变。静止背景、IMU、深度或其他外部约束缺失时，相机解释不可识别。
- 大间隔下的转弯、加速、遮挡/再现、切镜、detector jitter 与错误外观匹配都能改变共同位移。lag1 中位 30、lag4 中位 120 个原帧，不能假定是小运动或恒速；原图存在也不能自动补出中间帧与运动轨迹。
- 平移模型不能处理明显 zoom、roll、透视、不同深度视差；框中心也不是固定物体表面特征点。多个点拟合 affine/homography 的代数可解性不等于物理 camera 可识别。general affine 至少 3 个非共线对应，homography 至少 4 个一般位置对应，仍需独立检查几何 rank、空间覆盖和模型残差；本缓存没有此估计器，本次也没有验证这些条件。
- 同 proposal 排除可精确实施；按物理身份排除端点的所有别名无法在无标签接口中保证。严格框区域排除只能提供保守的图像空间隔离。应将结论写成“cached-row/proposal/overlap exclusion”，不能写成已证明的 leave-identity-out。
- donor-out residual 改善最多证明 raw 外观伪匹配下有可预测的共同位移；raw 错匹配之间也可能自洽。它不等价于 correct-ID ranking 改善、完整 tracking 提升、camera accuracy 或因果机制成立。
- 这类“其他目标提供运动上下文”有直接历史邻居：Yoon et al., *Bayesian Multi-Object Tracking Using Motion Context from Multiple Objects*, WACV 2015（RMOT，<https://faculty.ucmerced.edu/mhyang/papers/wacv15_mot.pdf>）。本条来自并行 novelty agent 的全文核查通知，本审计未重读论文。严格 endpoint exclusion 是防自我解释控制，本身不能承载新颖性主张。

## 6. 一个精确、可证伪的下一问

**在冻结 FIT 的精确 lag1 与 lag4 帧对上，仅按 raw appearance 得到的严格互选 donor，其位移能否被排除自身端点后重新匹配得到的其他至少 3 个 donor 的 median translation 预测，且两个 lag 的 source-macro 归一化残差改善均严格大于 0？**

可执行定义：先按 raw appearance 确定待预测 donor `h=(q_a,r_b)`，不看几何或身份；移除 `a,b` 及第 4 节规定的 proposal/overlap rows 后重新计算 donor matching，用其至少 3 个匹配估计 `t_−h`。对 held-out donor 记录 `gain_h=||(d_h)/s_h||_1−||(d_h−t_−h)/s_h||_1`，其中 `s_h=sqrt(WH(q_a)·WH(r_b))`。每 frame pair 对有效 h 取 median，再每 source 对 frame pairs 等权平均，最后对有支持的 FIT sources 等权平均，分别得到 `Δ_lag1,Δ_lag4`；成立要求两者均大于 0。显式报告所有 341 sources 的支持状态与两 lag 的可用比例，不把无支持补成 0，也不把条件结论推广到无支持对象。`Δ≤0` 或任一 lag 无支持即未通过这项最小预测性检查，不再用同批 residual 改阈值/删 source 来挽救命题。

该问题完全不需要身份标签或新 OOF readout。它只检验共同位移能否跨 donor 预测，不能回答上述不可识别的 camera 问题；通过后仍需另行固定机制和独立 identity evaluation 才能讨论 association 效益。本次仅提出问题，没有运行它。

## 7. reference 原型边界与复杂度（未实现优化）

root 新增的 `tmp_analysis/endpoint_excluded_scene_motion.py:109–118` 确实先删传入的 rows，再 normalize/match；`:67–85` 要求正 cosine 与 strict unique reciprocal maxima，`:125–140` 计算 median、MAD 和固定 donor 集合的 leave-one-anchor stability。`:144–170` 单独计算待评分端点 residual。原型支持外部 extra exclusion lists，但**本身没有识别 proposal/overlap aliases，也没有历史 warp 或 real-cache entrypoint**。`:134–137` 的 leave-one-anchor shift 是已匹配 donor 集合内的估计稳定性，不是第 6 节“排 held-out donor 后重新 MNN”的预测检验；两者不可混写为已通过同一控制。本审计没有改动原型，也没有运行它。

设一对帧有 `N` query、`M` reference、维数 `D=256`、实际 donor 数 `K≤min(N−1,M−1)`。当前逐 candidate 重算 cosine，在全部 `NM` 个候选上仅 dot products 就约为 `NM(N−1)(M−1)D`，即等尺寸下 `O(n^4 D)`；另有 unique maxima 扫描、normalization 与每候选 `K` 次排序的 LOO 稳定性成本 `O(K² log K)`。后者不是免费的。

仅用已读取的 FIT frame row counts 计算（未运行 matching），两 lag 的 naive dot-product 数分别为 **79,198,901,692 / 70,837,370,564**，乘 256 后约 **20.27 / 18.13 万亿 scalar multiply-accumulates**；这不含 unique/tie/margin/LOO/Python 开销，不是实测 wall time。max retained rows=77；最大单帧对 similarity=5,929 entries（float64 47,432 bytes，不含其他数组）。

数学上可复用原始 rowwise-normalized cosine `S=Qhat Rhatᵀ`：逐行 L2 normalization 不依赖其他行，因此先算 S 再依据候选的 exclusions 重建 nearest neighbours，与删 endpoints 后重算 dot products 等价。**缓存 S 可行；缓存全帧 MNN 后直接删匹配不等价。** 每个 surviving row/column 的最佳对象可能因删除竞争端点而改变。

- 仅排一个 q row 与一个 r row 时，可缓存每行/列最大的两个**不同分数层级**、各层 multiplicity，以及 singleton 的 row locator。按被删 row/column 所处分数组更新 count；原最大层被删空时再退到第二层，严格恢复 unique maxima。不能只存 `argmax` 或随便打破 ties 的 top2 indices。由此每 candidate 的 directed best/unique 检查可降到 `O(N+M)`，再组成 MNN 并取位移 median；与每 candidate 扫完整 `N×M` 不同。
- 若还要精确重现 prototype 的 margins，删除最大层后还需要下一个 surviving alternative；存前 **3** 个不同分数层级与 counts 可处理一次删除。若 donor-out 额外排一个 row/column，unique best 需要前 3 层，margins 需要前 4 层。阈值/原始分数关系不变。
- proposal/overlap 或历史 aliases 可一次删任意多个 rows，固定两/三层不再保证正确。需缓存完整有序分数层级及 membership，再对每个 exclusion set 跳过被删项、重建双向 unique maxima；最坏复杂度仍可退回整矩阵扫描。可按相同 exclusion set 缓存结果，但不能改变排除规则来省算力。
- 固定 donor 的所有 LOO coordinate medians 可由每坐标排序及删除元素 rank 直接恢复，避免每删一个 donor 都重新排序；这不等价于 donor-out **重新 matching**，后者仍需独立重建匹配。

缓存 cosine 后两 lag 的 dot-product 工作约 **7,448,636,672 / 6,721,517,312 scalar multiply-accumulates**；但全部候选的 `NM(N+M)` directed-row visits 尺度仍约 **3,035,967,602 / 2,729,485,264**，尚未计 LOO。故当前 reference 原型不能直接接全量 cache；应先有单独冻结的 runtime/等价性设计。本次只估算并说明等价优化条件，没有实现或 benchmark 优化。

## 8. 本次核对的关键 SHA-256

| 文件 | SHA-256 |
|---|---|
| `full_collection_plan.json` | `cbb14c6fd41135d887b46ce1b5869f0068b76922bf5c40f87f78ec2c2a715f9b` |
| `full_inference.json` | `375fa82b7ec603cb8781a63abf47c3c64f03f1f4c8b0e242cb7403fa0786abf2` |
| `full_collect_v1/frame_execution.jsonl` | `e0e327a887840fbbdf5acacc374a589f696ad529af62d7a9a535bc2dd6161c07` |
| `full_collect_v1/dump_inventory.json` | `5a7f94a531e754aef0cec1c78b82a080ee92d6e451f943a46be8c71c6a7676db` |
| `pair_context_v1/source_roles.json` | `afa61df067f392b84ba917e1c536a9b64b869dbda19a2236b0a05b6f5c51cb8b` |
| `tmp_analysis/relative_track_geometry.py` | `c5a085f8c56da990bc7f1cb2bedc7116c7ba32223e279f4ef678ed7447c4064a` |
| isolated `ovtrack/models/mot/ovtrack.py` | `5c410e560a2db91f28d1267bcba65c126f54f9ce73922098558a1ba25d4406b6` |
| isolated `ovtrack/models/roi_heads/ovtrack_roi_head.py` | `955d03011368fba48efde974b73bc4224e63a0d3add1028c17b97da54a2940d7` |
| isolated `ovtrack/base_error_dump.py` | `e0a934285b6966f4bf541a6ef40a7366ce186c608911cee9f1a624e36def2168` |
| isolated `ovtrack/models/trackers/ovtracker.py` | `0eaa4c96a09058f6f1b6ef73b1f860fb16aa0b6542ef43d0dc238de95a8f6db5` |
| isolated `ovtrack/apis/inference.py` | `1b9a2a9f0735d82dfb3d2e7083bafed25a5b8867a1bff47acc5e6f3982893012` |
| `full_collect_v1/collection_config.py` | `712ecc9486b5e1c4da527722939bbb2beea2bcbc7e6d5bc4b492cc7bd56bdb16` |

前五项相对 `results/base_error_full_collection_20261023/`；isolated 项相对 `work/covtrack_base_error_20261023/`。本次仅做只读检查，以上 key artifacts 与已提供冻结值吻合；未执行所有受保护文件的全量 hash 审计，不能替代 root 的保护验收。
