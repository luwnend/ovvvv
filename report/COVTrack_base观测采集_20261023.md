# COVTrack base观测采集与暂停续接 — 2026-10-23

当前已完成独立full-vocabulary导出接口、1,299帧pilot封存、pilot fit/tune自然错误支持及固定均值质量诊断。**全部fit/tune扩采已完成并独立终审通过：15,283帧/681,602输出，实际2.91小时；正在进行full支持集与更强均值质量控制。可学习依赖方法尚未成立，没有新tracker训练。** test保持关闭。full stage的`background_execution_receipt.json`和`independent_completion_review.json`为当前采集完成证据，下文pause交接保留历史状态。

## 已完成的可核验接口

独立源码副本为`work/covtrack_base_error_20261023`，原`work/covtrack_tces_v2`的101个源码文件逐项hash未变，新copy只修改ROI/MOT两文件并新增dump utility。冻结公开`ctao_public.pth` SHA为`e4d0b65798844280ea13943e580ce6233ae5d331c4aa1d38eb226c3208dd567c`；既有训练的config/launcher与正式baseline均受保护。

实际采集词表为LVIS-v1 1203前景+背景，保存`posterior_category_ids=[1..1203,-1]`、实际337列novel mask及text/image融合指数。完整词表改变softmax竞争、融合和检测/关联输出，不能等同历史val296后验或将裁剪结果声称为同缓存复现。`fix_bg=False`、ensemble开启、GT可视化关闭。阈值`.37`、max80、memo50、momentum`.4`、fusion2记录在每次run config；memo单位为选中帧的采样步，并非密集物理时间。

每帧保存原始track/cls特征、三路后验、proposal→NMS→tracker keep→emitted映射、原始/采样时间、video/image/local track ID。`id=-1`单列保留并用emitted flag区分。零观测帧也必须有NPZ和receipt；不能静默跳过异常。聚合键固定为`(video_id,local_track_id)`，GT只能在后续监督/评分层出现。

## 三轮接口核验与修复

| 轮次 | 核验一 | 核验二 |
|---|---|---|
| 1 元数据 | image/video负类别和非穷尽类别侧通道改为白名单 | categories的标注聚合计数删除；注入拒绝检查通过 |
| 2 CPU合约 | 真实4帧pipeline复现旧`KeyError: frame_id`，桥接后精确恢复`0,1,0,1` | 1204列/337mask/rowmap、空帧和负ID合成检查通过 |
| 3 真实小测 | 4帧150行与model实际emitted结果精确一致，两次tracker对象重置 | 独立NPZ/receipt/hash/融合审计实际exit0；71源码与7保护项通过 |

旧准备目录`results/base_error_collection_20261023`和`_v2`保留为实施偏离来源，不能作为合格缓存使用；`_v2/smoke_v1`实际exit1，未产生模型观测。合格准备目录是`results/base_error_collection_20261023_v3`，真实通过的run是其`smoke_v2`。

smoke融合重建最大绝对误差`1.8621592845491364e-7`。真实smoke没有空帧或负ID，这两项当前仅有CPU合成证据。完整谱系：`tmp_analysis/base_error_collection_deviations_20261023.json`、`base_collection_protocol_review_20261023.md`、`smoke_v2/independent_collection_audit.json`。

## pilot范围与历史暂停交接

pilot为41视频/28源组/1,299稀疏人工anchor帧，另存3,953精确人工anchor目标，fit可用标签上限2,138。选择这些帧依据标签可用性；每个选中帧保留全部GT-free模型输出，不能称密集自然错误总体。视频/源组切分不代表released detector从未见过这些数据；class holdout是readout的pseudo-novel。

启动的run：`results/base_error_collection_20261023_v3/pilot_collect_v1`。PID4166653，原工具session91862，日志`execution_logs/pilot_collect_v1.log`。耗时初估6–10分钟。设备固定UUID`GPU-2a55197e-2c19-45ff-cf33-13031951f9d0`，启动前无chenhc进程且约36GB空闲，原有另一用户进程按最新用户授权允许共卡。本账号既有训练PID3546537/3546538及4GB显示卡不使用。

已另启动独立后台CPU收尾程序`tmp_analysis/finish_base_error_collection.py`：等本run完成后运行两个已通过smoke的validator，将结果写入`pilot_collect_v1/background_completion_audit.json`。它不启动GPU工作、训练、GT匹配或重试。若原进程中断且没有完成回执，会明确记为不完整；不覆盖部分输出。原runner的真实exit须从session91862读取，后台程序无法凭收尾文件杜撰原进程exit。

稍后继续时先检查：

```bash
tail -n 5 results/base_error_collection_20261023_v3/execution_logs/pilot_collect_v1.log
cat results/base_error_collection_20261023_v3/pilot_collect_v1/collection_receipt.json
cat results/base_error_collection_20261023_v3/pilot_collect_v1/background_completion_audit.json
```

该次暂停时`tmp_analysis/build_base_error_support.py`尚未独立审查/运行，历史记录见`base_support_review_handoff_20261023.json`。恢复后已补强来源链、通过独立审查并完成真实support；当前结果见后文。匹配由所有原始TAO GT参与且不使用预测类别，再仅用精确manual-anchor交集生成fit/tune监督；未匹配为unknown，test自然错误不计算。全emitted GT-free轨迹聚合先于监督打开，不能用GT选择读出观测。

## 学习目标边界

已写`tmp_analysis/base_error_learning_objective_20261023.md`和精确合成检查。hard同错受agreement平凡约束；`p*(1-pi)`中心化还会导致同身份每坐标乘积恒非负。两者不能直接声称学到共同错误协方差。候选是参考条件均值直接回归与out-of-source-group soft residual；仍需平均错误、quality+agreement、校准、同容量attention/相关融合控制。当前未训练，也未证明其新颖性或指标收益。

旧TCES错误失败结论已从正式文档移除，历史来源保持撤回标记。当前接口修复与合成检查不能替代真实机制证据。

## 完整pilot封存与真实支持结果

原采集工具session91862已实际退出0，耗时394.595秒。41视频/1,299帧/54,924输出均有NPZ与receipt，41次真实tracker重置通过，最大融合重建误差`3.2515110459652163e-7`。独立双validator均实际退出0，既有训练、baseline及源码保护项不变。此批没有真实空帧或`id=-1`，不能将CPU合成证据改称真实覆盖。回执`execution_receipt.json`与`pilot_collect_v1/independent_collection_audit.json`。

支持层先在打开监督前封存全部54,924观测和12,876个`(video_id,local_track_id)`，再对fit/tune做不使用类别的全GT Hungarian匹配（IoU>=.5，先最大有效匹配数，再总IoU）。原始TAO旧1230词表按唯一synset映射；1,068个可映射旧ID与新数值ID不同，绝不能直接拼接数值。1,025个fit/tune帧及2,969个人工anchor的原图/原身份/C-TAO `is_ori`、annotation ID和框来源链复核通过。

| 范围 | 可监督观测 | 分类正确/错误 | 源组/GT轨迹 | 目标类别 | 同pred-track同GT pairs |
|---|---:|---:|---:|---:|---:|
| Fit | 1,067 | 452 / 615 | 14 / 113 | 15 | 6,876 |
| Tune | 315 | 87 / 228 | 7 / 41 | 12 | 1,581 |

这些监督只来自fit目标类别的精确人工anchor；fit/tune的可监督anchor覆盖为49.91%/48.69%。预测未匹配的31,484/9,821行保持unknown，既不能算负样本，也不能据此算误检率。训练身份诊断与可训练pair字段已分离：oracle单身份scope只作描述，禁止作训练/调参mask；主pairs只要求两端独立合法，不用全轨迹GT纯度选样。

fit/tune实际有监督的目标类仅重叠`{173,207,793}`。fit主pair中单源组最多占22.06%、单GT轨迹10.78%；tune为30.80%/20.56%。pairs为穷举重复观测，不能当独立样本数。14/7源组和15/12类不足以宣称共同残差过程可迁移；不能把tune高错误率归因为类迁移本身。

test的274帧/11,950输出仅保留GT-free结构，没有prediction-GT匹配或错误结果；预留类别也未计算正确率。真实support实际退出0，独立审计逐行核验全部索引、1,382监督行与8,457pairs，并对15个固定分层帧独立复算匹配。回执`support_v1/support_summary.json`、`independent_support_audit.json`。

## 固定均值质量控制

这是CPU统计校准控制，不是新tracker训练。B0为fit子集的平均错误；B1为固定`lambda=1`、14维GT-free摘要的类别共享ridge-logistic。按源组→GT轨迹→观测等权；fit内5折源组交叉拟合，标准化只使用每折训练部分。tune仅评估，未用于选特征、正则、校准或checkpoint。输入不含类别/身份ID、GT IoU或轨迹纯度。

| 同层级等权指标 | B0 Brier | B1 Brier | B0 BCE | B1 BCE |
|---|---:|---:|---:|---:|
| Fit 源组留出预测 | 0.258882 | 0.208137 | 0.711028 | 0.604656 |
| Tune（全部fit拟合） | 0.233728 | 0.191559 | 0.660503 | 0.568527 |

普通质量信息可解释部分错误，后续pair网络必须超过它。tune等权真实错误率0.687961，B1平均预测仅0.598346，校准截距/斜率0.220913/1.712482，不能视为已得到可靠中心化均值。条件于已有预测的1,000次源组bootstrap中，tune Brier差区间为[-0.056752,-0.027273]；没有重拟合模型，只有7个源组，不作训练不确定性或独立泛化结论。fit micro Brier差区间仍包含0，权重口径会改变结论。

该B1仅是低容量sanity控制；仍缺更强、内层source-CV的质量模型。它既不能证明quality已被充分控制，也不能让pair方法通过训练门槛。真实诊断退出0，输出`mean_quality_v1/diagnostic.json`；轻量摘要和独立复核另存同目录。

## 扩采决策

扩采只复用原sealed源组与类别split，选择全部fit/tune人工anchor帧，不根据tune错误挑类别/视频，不采test。计划15,283帧/46,908精确anchor框；allocation450视频中2个没有精确anchor，实际为448非空视频、383源组，用于缓解当前独立源组和类别覆盖不足。新stage为`results/base_error_full_collection_20261023`，与pilot完全隔离；full词表/稀疏步序协议保持一致。独立元数据审计状态为`PASS_INDEPENDENT_FULL_COLLECTION_METADATA_AUDIT`。

扩采预计1–2小时，仅冻结公开权重推理，完成后由独立后台程序记录真实子进程exit并做CPU校验，不自动训练或用test调参。最终状态应检查新stage的`background_execution_receipt.json`，启动回执不等于采集成功；若进程仍运行，下一会话先现场核验PID及日志，不能重复启动。

上次用户暂停发生在full启动之前；历史`pause_handoff.json`保留该时刻及已完成证据的hash。恢复后已于14:19（Asia/Shanghai）由supervisor4190240启动collector4190241，run=`full_collect_v1`，日志=`execution_logs/full_collect_v1.log`。`background_launch_receipt.json`已生成，新的`background_pause_handoff.json`记录暂停Codex时的真实进程与帧进度。准备manifest的未启动/PENDING字段是生成时快照，当前以独立metadata audit及运行回执为准。预计超过20分钟故依最新用户规则暂停对话；后台继续并自动执行双CPU验收。既有另一会话训练保持运行。恢复时先核查`background_execution_receipt.json`和进程日志，**不得重复启动已有run**；不要重跑已封存pilot/support/固定质量诊断。full support builder仍需独立适配及审计，不能直接强套pilot guard。

## Full采集完成验收

采集实际于17:13（Asia/Shanghai）完成，10464.512秒=2.9068小时，原1–2小时预估偏低。15,283帧/448视频/681,602条retained及emitted输出完整，448次真实tracker重置；1204后验列、337类mask，融合重建全帧最大误差`4.2475654860041345e-7`。没有真实零输出帧或未分配ID，本批不能新增这两项真实覆盖声明。

`background_execution_receipt.json`记录collector与两个CPU审计子进程均实际exit0。`independent_completion_review.json`重新核对30,566个当前NPZ/receipt的SHA（约10.45GB）、15,283个逐帧事件/身份/计数、7项保护文件、71份源码及13项准备输入；26固定帧/1,213观测另做内容复核。全帧语义证据仍由原sealed双full audit承担，不将26帧抽核写成第二次全帧语义重算。外层detached supervisor自身的terminal exit没有直接wait证据，仅核实了其子进程真实exit链、最终日志/回执和当前进程不存在。

这封存的是冻结模型、完整词表、稀疏人工anchor时序下的观察流；没有新方法训练、full自然错误支持或新颖性结论。下一步使用独立full support适配器并审计，再做fit-only nested source-group强质量控制；test/留出类正确性仍关闭。


## Full支持完成与上次用户暂停（历史断点）

Full support builder工具session51211由root直接poll确认actual exit0。独立输出审计session68926由evaluator agent直接poll确认actual exit0（95.71秒），状态`PASS_INDEPENDENT_FULL_SUPPORT_OUTPUT_AUDIT`；receipt SHA绑定一致。审计覆盖全部681,602观测、144,930轨迹、15,283帧Hungarian与排除计数、46,908人工anchor来源、23,887监督观测及193,405重复观测对。fit为21,661观测/12,694错误/338实际有监督源组/131类；tune为2,226/1,081/42组/38类。这些是描述性支持，不等同独立样本数、条件协方差或方法有效性。

上次用户暂停时，断点为完整support审计通过、强mean真实拟合未启动。该时刻mean实现/独立预审和合成审计已封存，pair sealer仍为WIP；历史状态/hash见`results/base_error_full_collection_20261023/dialogue_pause_receipt.json`。用户已恢复，下面记录后续实际完成结果；当前入口为`tmp_analysis/full_stage_handoff_20261023.md`。早先1–2小时估计、采集运行中及旧pause记录只表示历史时刻。


## 恢复后强mean与GT-free索引完成

用户恢复后，full mean真实运行265.523秒/exit0，独立审计139.852秒/exit0；完整指标与范围见[Full质量控制报告](COVTrack_full质量控制_20261023.md)。主B3等权Brier fit/tune=0.147018/0.136580；tune B1=0.203912，B3−B1 bootstrap区间[-0.083776,-0.052271]。它是强普通质量控制，不是新机制方法或TETA结果。

GT-free上下文索引独立预审先修复了上游manifest只自洽、未绑定accepted来源及时间/整数边界；三轮每轮两check实际通过后才实跑。Producer45.802秒/exit0，独立54.17秒/exit0；52,303条长度至少4的预测轨迹、623,878边。独立重算全部3,947,118候选SHA及全部边的reference补集hash、source role、时间/cap/计数；未读取NPZ或监督。源组角色172 mean/169 pair/42 tune，依据全部预选元数据，不能因合法标签缺少而重分组。reference只是完整emitted轨迹减去两端的索引，不是GT筛过的reference，也尚未做posterior pooling/分裂reference。

## Fit身份支持与路线收敛

Fit-only身份join实际33.543秒/exit0，fresh独立全量审计80.099秒/exit0。全部560,371条固定fit边中12,158同GT、819不同GT、9,388单端unknown、538,006双端unknown；不同GT涉及135源组/280条预测轨迹。只解码fit合法身份，不解码分类正确性/tune身份，不改原参考池。只有12,977边两端已知，不能算总体污染率或认定身份净化一定有分类收益。详见[身份支持报告](COVTrack_fit身份混入支持_20261023.md)。

冻结共同错误G公式已因conditioning不一致停止部署/协方差归因路线，三轮每轮两项精确检查通过；没有启动真实pair拟合。该决定不否定所有关系网络，也不影响已完成质量控制/索引/支持。下一步为独立fit-only稀疏observed-label oracle对照及class-blind可辨识性；GT快照不进入真实readout，尚无新方法TETA或训练就绪结论。
