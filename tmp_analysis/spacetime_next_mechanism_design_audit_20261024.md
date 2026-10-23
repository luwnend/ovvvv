# 下一步时空机制对照：独立设计审计

状态：`DESIGN_ONLY_NO_NEW_OOF_EVALUATION_READ`。2026-10-24。本文件不启动实验、不修改当前 OOF 的任何 pinned 文件。当前 OOF 的有效/无效尚不在本审计证据内；下文的“若 G1 无增益”是条件问题，不能写成已观察结果。

本次使用 experimental-design 技能：明确实验单位、成对干预、对照与交互，避免把候选/重复观测当独立重复。没有借此进行功效计算或统计显著性检验。

## 1. 已核实证据和当前可解释边界

直接读取并 SHA256 核验：

- 冻结当前协议 `tmp_analysis/identity_spacetime_oof_protocol_20261024.json`：`a7169f8674d8e019782820bc7b4a99ff4b2082bdeaa0a731e60352a488ed9be1`。
- 历史支持 `results/fit_lag_identity_support_20261024_v1/summary.json`：`9764e2a6f35e4e0a5ce3ab319d15ca4642afda5383289710a5e9dd66e48974c1`。
- 历史角度外推诊断 `results/angular_history_assumption_20261024_v1/summary.json`：`b8ad568025b910f01cf4bcb7072f6ff175338cfcb7feb916d4a130606050ee19`。
- 冻结机制说明：`identity_spacetime_factorial_design_20261024.md`、`identity_spacetime_followup_recommendation_20261024.md`；几何接口说明 `covtrack_feature_spacetime_design_20261024.md`。

没有读取新 identity-spacetime OOF run 的评价、query SQLite、control brief 或中途指标。全部数值均来自上述历史文件。

历史主读出是 same-predicted-category/common-CV 的已知 P+N conditional ranking：

| sampled lag | hard query / source / GT trajectory | raw 错误 | CV 单 cue repair / break | unknown / candidate pairs |
|---|---:|---:|---:|---:|
| 1 | 2949 / 172 / 578 | 46 | 26 / 289 | 1758988 / 1775532 |
| 4 | 2142 / 160 / 453 | 149 | 60 / 533 | 1468371 / 1481343 |

raw 错误改善的 micro 上限本来只有 `46/2949` 与 `149/2142`；source macro 上限不同，不能直接由这两个数推出。55,352,164 全候选不是55m合法训练样本，unknown 也不是可用负例。两个 lag 共享 1331 hard query / 107 source；共享集合 raw 错误16/99，两边都错8。比较不同 lag 必须同时保留全人口与这个固定交集。

历史角度外推的“全部已知 positive history 同 GT”层仍显著受损：lag1 2576 query，raw2550、angular2457，repair5/break98；lag4 1912 query，raw1799、angular691，repair30/break1138。这反驳“角度外推失败完全由这些已标 positive history 的已知 ID 污染造成”这一强说法。它不能证明像素 CV 失败的原因，也不能证明所有其他 candidate/history 干净；未标注 history 仍未知。该 GT-clean 层是事后诊断，不能成为部署筛选器。

当前 frozen head 是逐 edge 的 `s/0.1 + MLP19`，不是集合 softmax/attention/assignment。训练的 known compact gathering 不改变单条 edge 前向；Gperm donor 已在 full-retained pool 构造后再 gather。故在 raw/geometry/normalizer 相同且 adapter 冻结时，full 与 known compact 对选中 edge 的 score 应数值相同。不能未验证就把负结果归因于“网络训练看小集合、推理看大集合”。真正差别是合法监督支持、pairwise surrogate 对 top1、query 权重对不同读出，以及未来在线 memory/assignment，需分别测试。

## 2. 四个解释可同时成立，当前零差值不能分解它们

| 解释 | 可观察、可反驳预测 | 最小新对照 | 正结果仍不能说明 |
|---|---|---|---|
| 稀疏低支持 | 合法支持缩减会不稳定地损害几何增量；新、事先选定的密集标注预算能增加训练与可评价 P/N | 固定 source/trajectory 分组的嵌套标签预算；后续独立合法密集子集 | 仅 dropout 变差不能证明当前已在“标签不足”区；加标后提升也可能是人口改变 |
| 主导像运动混淆 | 其他目标估计的共同位移经严格端点排除后，真实 residual 对齐超过未补偿、support-only 与失配 residual | 未补偿/端点排除共同平移/同支持 placebo，训练推理各臂匹配 | 无外部参考时不能归因相机；整群同向运动也满足同样预测 |
| released-history ID 污染 | 历史依赖臂对受控错链敏感，禁用历史与不依赖released-ID的共同位移有不同交互 | 原 locator / 固定错链 / 历史禁用；使用完整 GT-free pool | 人工错链敏感不证明真实错链频率足够大；GT-clean 描述层不能推广全候选 |
| 训练/推理粒度差异 | full/compact score 不变但 P×N loss 与 top1 的成对读出不同；在线 memory 改变可单独引发变化 | 前向恒等性门；固定 pairwise 与 listwise 已知条件目标；未来 memory-only replay | 已知集合 listwise 改善不能外推未知全候选、null或在线 tracker |

同一 query、多 lag、seed 和候选共享源与轨迹。成对读出以 source→GT trajectory→query 聚合；seed 是初始化重复，不能充当独立数据重复。source 删除是敏感性说明，不能删坏源再汇总。当前无预声明最小效应量/功效，不能将零差值说成几何信息不存在。

## 3. 先做无需新监督的可辨识门

先封存这份设计，再构造 GT-free receipt；不查看哪个新 OOF query 错误后决定做谁。

1. **几何能否被当前 retained pool 表达。** 逐 source/video/帧对报告目标数、严格 reciprocal appearance anchor 数、空间范围、位移向量秩/散布、history availability、原帧 gap、共同位移支持缺失。它们是观测支持，不是可靠性概率；小 residual/MAD 不能证明正确匹配。
2. **所改数值是否真正改变排序信息。** 报 mapping 改变、实际几何值改变、score 改变分别多少；same query 下若给所有 candidate 的最终 logit只加同常数，top1必不变。减共同二维位移在非线性距离/MLP前可改变排序，不能先验视为无效。无改变的 arm 对有效性没有信息。
3. **full/compact恒等门。** 任意含unknown donor、重复ID、空帧、zero raw、缺失history的合成输入上：完整 raw归一/adapter/geometry/normalizer/Gperm后 gather 与训练 compact 前向逐 edge相等。若失败先修实现，不能解释为科学差异。此门已在当前独立预审检验，未来改集合模块必须重做。
4. **支持不足保持缺失。** 共同位移无法估计则 `available=False`，不伪装 identity warp/零运动；所有臂在相同完整人口输出，另有事先定义的共同支持交集。不得只在某臂成功匹配的query换分母。

只有这些门成立后，训练新 head 才回答“利用某种新输入是否有效”；固定现有 head 上突然换输入只测分布扰动，不能区分输入无用与训练未适配。

## 4. 最小可解释新实验：固定 appearance，2×2共同位移×history

优先固定 A0 raw cosine，不同时改变 identity adapter、head容量、label预算或集合网络。保留当前 G0/Gmask 作为共同参考；新实验单独命名、独立产物，不替换当前协议。

因共同位移与 ID-history可能交互，采用完整四格，而非只做“补偿后最好的一臂”：

| 条件 | current←reference空间位移 | CV history |
|---|---|---|
| C0H0 | 原像素相对空间 | 禁用过去history，不提供CV innovation/IoU |
| C1H0 | 严格端点排除的共同平移 residual | 同样禁用 |
| C0H1 | 原像素空间 | 原 released-ID locator |
| C1H1 | 同一共同位移契约，并分别估计 previous→reference 方向所需的历史变换 | 同一原 locator |

**若没有为 previous→reference 单独获得合法 GT-free 支持，则不实现 C1H1 的“补偿CV”；只做 C0H0/C1H0 两格，保留原C0H1作不同机制的描述对照。** current←reference residual 本身不是CV补偿。不同reference帧不可共享一个H；端点特定平移也不可假作整帧H。

每个新输入至少还有两种固定控制：

- **support-only**：只供相同时间、availability、anchor_count和预定支持描述，空间 residual槽置零。检验收益是否只是支持/场景密度提示。支持描述本身由raw匹配产生，因此这是有appearance context的控制，不能称纯几何。
- **alignment placebo**：在同一帧对、相同GT-free支持组内按 immutable row 预定循环错位 residual 数值，保留自己time/masks/support，先在fullpool构造再投影known。报告实际改变；不保证异GT或彻底破坏空间信息。不要从若干随机置换里选最坏的一个。

新训练每臂从相同初始化出发、同 source folds/seed/优化/epoch/完整P×N/层级权重；normalizer只fit对应train sources，并在干预对照间共享预声明的原始新输入normalizer。禁止按新结果加迭代/调支持阈值/选lag。真实残差必须超过support-only和alignment placebo，且四格交互须报告，方支持这项输入对conditional ranking有独立贡献。

历史污染的最小可反驳扩展：完整候选的 `(predicted category,history-valid,原始history gap)` 合法组内做固定locator循环替换，不用GT/known/raw-error定义组；保留当前框、所有candidate、自己的gap/mask。由于past frame可能不同，优先只在**同一reference frame和同一past image**内替换past boxes，避免把时间尺度也混入；group<2不变，实际错位覆盖另报。原 locator与错链训练/推理分别匹配；原训练→错链推理只是敏感性probe，不是学习效果。另做history-off正交对照。人工污染与GT-clean描述层只提供互补证据，不能估计真实污染的因果占比。

## 5. “leave-endpoints-out”共同运动的精确条件

设帧对 query行集合Q、reference行集合R，待评分edge e=(i,j)。外部证据程序应是：

`E_e = fit(match(Q\excl_Q(e), R\excl_R(e)))`，再计算`r_e=(center(q_i)-center(r_j)-E_e.translation)/scale_e`。

### 5.1 可执行的无直接自证条件

足以保证**给定固定excluded locator集合下**端点数值不直接影响fit的条件是：先按locator删端点及已声明aliases，再做归一化、候选竞争/matching、筛选、拟合、支持统计；这些步骤和所有参数只读取剩余数据。排除集合不能由待评分残差、GT或其最终排名决定。端点只允许进入最终residual和自己的size归一。

这保证的是cache值层面的函数不变性：任意合法端点box/raw变化，不改其locator，`E_e`、anchor列表、support/MAD等必须完全不变。它不保证最终residual不变（该项本来就应使用端点），不保证候选外部证据与目标物理状态统计独立。

只做`match(fullpool)`后删命中端点的anchor不满足该条件：端点可夺走另一个互为最近邻的位置，删后也不会自动恢复match。即使最后anchor没有端点，端点仍通过竞争影响结果。先删再match是本原型的可审契约；对某个特殊算法匹配后删可能偶然相等，但不能推导普遍不变性。

必要的操作性测试是多组端点mutation、输入排列、重复alias及“竞争被端点夺走”的反例；有限测试通过不是任意算法的数学充分证明。由实现依赖路径排除端点，加上测试才形成可审证据。只排除行不是leave-physical-object-out：同一物体重复box、分裂proposal、同released-ID或其他相似观测仍可能留下。GT-free重复/locator闭包能删除已知aliases，无法证明所有物理同身份都被删掉；用GT作完整闭包只能是另标oracle诊断，不能混入实际 predictor。

缓存raw来自共享图像/backbone；哪怕删除端点行，其他raw仍可含其视觉感受野。端点cache值不变性不等于重新渲染图像/重跑backbone后的视觉因果独立。

### 5.2 reciprocal raw匹配不是纯位置证据

严格唯一、正cosine、reciprocal匹配可消除显式tie和行序破tie，但不能证明正确身份；同类相似物体或共同误匹配可形成完全稳定共识。raw matching把appearance间接带入几何估计，属于appearance-conditioned scene context。所有学到的改善须与“相同匹配/相同支持但residual置零”比较。不能把它宣传成不依赖appearance的新几何来源。

排除后至少3 anchor是数值契约，不是理论识别camera的充分条件：二维纯平移在理想已知对应下1个pair就可给出位移，3个只允许初步consensus/leave-one敏感性；多anchor同一物体也不是多个独立物体。严格reciprocal无需GT但仍可能只有1–2支持，此时保持缺失。

### 5.3 中位数识别的对象和充分/非必要条件

观测位移写作 `d_k = c + u_k + epsilon_k`，c为共同像平移，u为目标自身运动/模型剩余。componentwise median估计 `c + median_k(u_k+epsilon_k)`；对于采用的odd/even样本median定义，**精确等于c当且仅当每坐标median(u+epsilon)=0**。这不是可由观测本身验证的camera条件。

严格超过半数anchor在每坐标满足`|d_k-c|≤epsilon`，足以保证中位数误差每坐标≤epsilon，即二维误差≤sqrt(2)epsilon；是任意污染下的有限样本稳健保证的充分条件，不是特定数据中准确的必要条件。少于半数污染也不解决系统偏差；重复alias使“多数anchor”不等于多数独立物体。MAD低和leave-one稳定只说明样本共识，不说明bias小、匹配正确或可校准置信区间。componentwise median甚至不必等于任一真实anchor向量。

**相机/物体共同运动不可辨识证明：** 对任意向量a，`c'=c+a, u'_k=u_k-a`得到完全相同d。没有静态背景、独立相机传感器、已知stationary anchor或预声明约束，仅靠这些object centers无法分离二者。全部物体同向等速时可有零MAD、零leave-one shift，同时把物体运动全部吸收到c。它可作为主导像平移的control，不能成为真实camera motion、ego compensation或物理独立运动证据。

真实camera补偿另需：GT-free背景/图像对应（且目标ROI外的证据边界事先定义）、正确时间方向、足够非退化支持、相机模型适用性/视差诊断；仅单应可拟合也不意味着全景深物体服从同一H。当前缓存没有这些对应，需新独立image estimator/cache，不能从objectflow或semantic平均向量恢复。

### 5.4 per-edge warp与复杂度限制

每个(i,j)排除不同集合，所以`translation_e`通常随candidate改变。它是candidate-specific残差证据，不是整帧共享camera warp，不能直接塞给只收共享H的relative_track_geometry调用；不保持整bank的单一几何坐标系，也不能宣称Gram结构保留。不同edge残差可能互不满足cycle一致性。需要逐edge receipt绑定exclusions/anchor locators/translation/support。

朴素每edge重新算(Q-1)×(M-1)×D cosine并匹配，单帧对复杂度约O(Q²M²D)；Q=M=77、D=256时约9×10^9乘加量级，不能按“多一次Q×M”估计。可预存一次rawcos矩阵后每edge排除重做极值竞争，仍约O(Q²M²)扫描与O(QM)fit数；不能用预存fullpool matches后删代替。相同eligible exclusion集合可cache，cache key包含全部locator+原始raw版本+规则。跨lag/跨frame不可任意复用。

当前全候选55m若逐edge产生至少3anchor的JSON将额外巨大；先做合成复杂度门和全人口GT-free支持计数，冻结bounded streaming/数量上限/存储格式后才有真实入口。不能先按GT-known或raw错误子集计算来掩盖部署成本。

## 6. 稀疏支持与粒度：能区分什么，仍缺什么

**最小标签预算实验**保持全部GT-free候选/历史/normalizer人口定义，并只在训练loss合法标签projection内用预定source→trajectory整组hash嵌套子集；不按错误/几何可修复性删query。至少原合法预算与一个固定减半预算，同支持readout并列，报告P/N/hard支持变化及零source。若标签减少使几何差值下降，说明这项学习对该预算敏感；若不变不能证明监督充足，若下降不能证明加标会改善。稀疏标签可能非随机缺失；轨迹级dropout也不能模拟新identity覆盖。要确认密集支持，需要事先按GT-free场景/密度/时间分层选新的合法标注单位，封存所有候选的完整身份与可见性，再做固定预算比较。新增密集标注不得打开当前未授权tune/test/classheldout身份。

**粒度实验**先通过full/compact score恒等门；再比较相同输入/架构/预算下固定pairwise mean与事先定义的known-conditioned listwise目标。它只区分训练surrogate与known-top1，不解决未知池竞争。full pool不能把unknown自动纳入softmax负类。listwise若没有合法负例/positive就不训练，仍无null标签。

**在线memory实验**属于下一独立协议：只替换last-observation与真实raw EMA/reference memory形成规则，保持同一GT-free predictor、detector和assignment规则，报告首个状态/ID分歧与后续传播。当前帧对缓存没有未来某种新tracker形成的完整counterfactual memory；离线edge收益不能替代在线闭环、寿命、遮挡、未匹配或最终TETA/AssocA。要公平训练memory输入必须用train sources合法、因果构建的memory分布，不能以GT轨迹纯化推理memory。

## 7. 停止与允许的结论

若共同位移缺失/aliases使支持不足，停止在支持描述；不靠降anchor数或搜margin救结果。若端点mutation改变fit，先修依赖路径；若matched/shifted residual未实际改变值，不能据此推断几何有效性。若真实残差不超过support-only/placebo，拒绝这项空间对齐解释，保留所有arm和失败覆盖。

即使正向，也最多说“在预定稀疏合法known conditional ranking和此source OOF下，某GT-free候选context residual对这类head提供增量”。不能外推真实camera估计、因果独立运动、预测history身份正确、backbone身份表征改善、全未知池top1、无匹配判断、在线tracker收益、独立泛化、充分统计功效或方法新颖性。四种解释可能交互；任何单一事后分层或零/正差值都不能给出唯一因果归因。


## 8. 新参考原型的独立合成审查收据

`endpoint_excluded_scene_motion.py` 仅为无真实入口的普通共同像平移参考模块。本审计未修改它。独立审查首先发现原版有限巨大box的中心求和overflow会令`available=True`却返回NaN；已向作者报告，作者修复中点、宽高及位移/MAD/leave-one/innovation的finite fail-closed检查。本审计以修后SHA `4ba1517f42d7b60e109db1675c9c4b1d54948cc3a45aeec5031cf40cf9ed730b` 为准；不把作者自测作为独立证据。

新独立脚本 `tmp_analysis/audit_endpoint_excluded_scene_motion_20261024.py` SHA `81089a0e2ae359ab4c9d3c4cf341c33e0b56e3ca36e09c0c84a40c1a09bed689` 的三轮均实际退出0、各3项检查。收据 `endpoint_excluded_scene_motion_independent_round{1,2,3}_20261024.json` 状态均为 `PASS_INDEPENDENT_ENDPOINT_EXCLUDED_SCENE_MOTION_SYNTHETIC_AUDIT`。覆盖24组端点合法mutation、固定extra alias排除、先删后match的竞争反例、27组独立scalar cosine/median/MAD/leave-one oracle、空/零/严格tie、独立行排列、多数异常值、中位数零MAD共同物体运动反例、非法类型/框/数值overflow和subnormal尺度。没有读取真实cache、真实身份或任何新OOF评价。

该PASS只证明这些合成契约与数值边界；不证明真实匹配、物理alias完整排除、camera识别、真实支持率、全人口runtime、学习/跟踪效果或新颖性。当前OOF所有冻结文件保持不变。
