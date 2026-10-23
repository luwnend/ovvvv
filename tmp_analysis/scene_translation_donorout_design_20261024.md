# GT-free共同位移的donor-out预测检验：固定设计

2026-10-24。只回答“其他raw外观对应是否含有可预测的共同像平移”，不回答相机运动、身份正确性、关联增益或新颖性。正在运行的15模型身份外观×时空OOF维持原协议；本诊断没有新训练或OOF评价输入。

## 人口、可观察量与精确时间

仅原sealed FIT的341source/403video/13,791frame/613,536retained，全部候选按原顺序保留。角色只作metadata，不按anchor支持、类别、released ID、OOF错误或运动残差筛源/帧/候选。精确sampled lag1/4，每帧每lag均receipt，包括prefix/空帧；原帧号定义间隔，非秒。两lag共有55,352,164原Q×M候选，诊断目标只是每帧fullpool raw互选对应；不能称55m身份评价。

NPZ只取det_bboxes[:,:4]、raw track_feats、roi_proposal_row、video/image/sample/original-frame定位，不取类别、track ID、posterior、GT或人工identity。plan只投影video_records，其人工支持计数字段不参与任何选择/权重/预测。metadata原frameexec/inventory逐帧SHA核对，不读取另一个OOF输出目录。

## canonical浮点与目标定义

原始row→float64 **C-contiguous**；每row先除max(abs)，再除L2 norm，零输入保持零并从matching剔除。每帧对只计算一次完整`query_normalized @ reference_normalized.T`，同一C-layout矩阵用于所有exclusion。strict MNN要求rawcos>0，双方max均唯一且互选，ties不破；没有可调cosine/motion阈值。严格tie是这条冻结浮点路径的结果，不要求sliced BLAS或不同layout/逐scalar乘法的最后bit相同。独立auditor重建同一canonical矩阵，另以scalar数学核对数值。

fullpool strict MNN的全部对应h=(qi,rj)为被预测对象，先确定，之后不因框运动、fit availability或gain删除h。它们只是伪对应，不是已知真身份；偏向raw外观稳定对象，不能外推hard/unknown身份。

## 三臂与缺失

1. **fullpool**：原fullpool MNN的全部对应拟合一次，含h，是自我解释控制。
2. **row_excluded**：在matching前只删除qi/rj，然后重新双向MNN和fit；不能全匹配后只删除h边。
3. **proposal_overlap_excluded（主读出）**：同帧删除端点、相同roi_proposal_row的所有行、与对应端点box有正面积交集的所有行，再重新matching/fit。边界接触且交集面积0不删。此排除是direct endpoint overlap，不做transitive closure；若proposal alias没有框重叠也删除。

每个pool至少3个重新MNN对应才available。其余保留所有h，available=0、anchor_count保留、九个连续fit槽NaN；reason唯一编码为insufficient_reciprocal_anchors。不把缺失变成identity warp/实测零共同运动。非有限几何整体fail-closed并保留run，不按结果删行。

使用其他anchors的原像素中心差逐坐标median作translation。先得到像素translation，再用h自身的`s=sqrt(query_width_height)*sqrt(reference_width_height)`按x/y归一。

`static=|| (center(qi)-center(rj))/s ||1`；`corrected=|| (center(qi)-center(rj)-translation)/s ||1`；`gain=static-corrected`。MAD和固定donor删除一次的median最大shift只作支持描述，不是校准uncertainty。真正donor-out是前两端先删、重新matching，而不是固定原anchor集算LOO。

strict mask由端点原框/原proposal生成，故“端点数值突变不影响fit”只对**给定固定excluded locator集合**成立；重建overlap mask后不能要求不变。行/proposal/overlap排除也不能保证所有相同物理身份、共享backbone感受野或统计依赖都排除。每h拟合不同translation，不是全帧共享camera H，不是CV补偿。

## 固定读出与否证

每lag、每arm：同帧pair取available h的gain中位数；同source对有支持framepairs等权平均；有支持source等权平均。所有341source显式列出，未支持不填0。记录完整holdouts/available/frame/source计数与原分母。

两项common读出先按同framepair、同h相交：row+strict共同available，以及三臂共同available；在相同h上分别算每臂frame median，再同source/source-macro层级。跨lag只取“两个lag均有支持”的source交集，报告其各lag宏值；不把不同reference的h硬作纵向同目标。

最小primary门固定为strict的两个lag source-macro gain均存在且严格>0；否则NOT_PASSED，不调minanchors/支持阈值/模型阶数/源组或改主读出挽救。即使PASS也仅说明这些raw伪对应的位移有外部预测性，不证明真实ID、相机分离、unknown池排序、tracking、跨域泛化、充分效应大小、功效或论文贡献。

## 存储、资源与完整验收

每video单独NPZ：values[N,37]，rowloc[N,3]=(framepair receipt序号,query row,reference row)，每臂anchor SHA和exclusion SHA各uint8[N,3,32]。SHA只绑定有序anchor row pairs和排序exclusion locators，独立auditor从原cache重建名单；不依赖浮点值转JSON打hash。

37列为4个base（rawcos、两向margin、staticL1），加每臂11槽（available,count,tx,ty,MADx,MADy,fixedLOOmax,innovx,innovy,correctedL1,gain）。frame receipt绑定全部prefix、offset、step/image/original-frame、Q/M行数与h数。所有403video完成后全局seal，再写无标签描述summary；fresh auditor从原raw重建全部数值/两hash/receipt/source与common汇总。最后由supervisor真实wait producer、auditor、finalizer，核pins/7保护后才有完成凭据。

metadata-only upper bound：lag1/4最多564,223/506,219个h，总1,070,442；NPZ原始array上界548,066,304bytes（每h512bytes），另receipts/manifest。无需给55m全部候选逐个重算。单线程77×77、30帧对/2,310h的cached v2完整合成1.7733秒，reference leaveout8.0705秒；这不是完整真实计时。为producer/IO/audit规划30–120分钟CPU，无GPU；实际时长以exit凭据为准。诊断本身预留3GiB；与当前OOF并行时采用更严格的**31GiB空盘启动门**，覆盖其未写完的评分、诊断array及元数据规划、额外2GiB和原OOF18GiB余量。启动现场重新读取free；这不是磁盘预留或实际peak测量。

真实执行前必须完成三轮producer/cached与独立synthetic、runner真实success/failure wait链、协议外部SHA绑定的独立preflight。禁止重启覆盖已有run；失败先查实现，至少三轮每轮两项或明确已实现/具体不成立。输入/源码/设计冻结后不改；新结果不回流到原OOF。
