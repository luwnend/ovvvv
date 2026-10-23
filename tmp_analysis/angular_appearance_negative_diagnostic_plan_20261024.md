# 普通角速度负收益：三轮固定诊断方案

2026-10-24。状态：`DESIGN_ONLY_NO_REAL_DIAGNOSTICS_OR_TRAINING_RUN`。本文件只新增诊断设计，不修改 pinned protocol、producer、verifier、候选、缓存或结果，不执行真实数据/训练。它回答普通恒角速度外推若出现负收益，怎样区分实现错误、past ID 污染和动力学假设失败；不把普通方法改名，也不按失败结果选人口救收益。

依据的固定文件及读取 SHA256：

- `angular_appearance_control_protocol_20261024.json`：`44d8bce20de4ed3fd358a083665e6e9d8b2744d29d86dcdc9b2d9efa2f07eb69`。
- producer `diagnose_angular_appearance_control_20261024.py`：`1e9fae320d2435a4560ffeeba3bfe441891f9d9db202296b269ba0419096828e`。
- 独立 verifier `verify_angular_appearance_control_output_20261024.py`：`ce73f26fe8566527794a2d58b0278ff39a41562b026d544899d13b326ee76259`。
- 数学核 `causal_appearance_transport.py`：`8b534c5f133880165e3e46abbd93e49584ebf8e20bec9c58b46ab425fa8f5924`。

上述是读取时快照。结束复核时 protocol/producer/core 三者一致；verifier 被并行开发更新为 `582330163e3d68134574fdb68fa4289cc19c34d4de545aee406f03ec382b2c51`，本任务未修改它，不能把旧读取 SHA 说成当前最终版本。实际执行须绑定当时独立预审通过的最终字节，本设计不取代此条件。

## 固定人口与共同输出

原 341 sealed FIT source、403 cached video、13791 frame、613536 retained observation；两个未采 allocation 1720/1974 单列，不扩角色。lag `{1,4}` 的全部 retained Q×M 共 55,352,164 行，两 scope `all_candidates / same_predicted_category`、两 population `all_known_population / common_CV_covered_population` 保留。CV共同人口按原几何历史可用定义，不能缩成角速度成功/身份已知历史人口。空帧、无历史、released ID=-1、重复历史、raw-zero、coincident、antipodal 和 unknown 都保留并解释分母。

首先原样呈现全量 raw→angular 四格、repair/break、两 lag 与共同 hard-query 交集（原基线1331）、source→GT trajectory→query macro、原角色/dataset block、每 source 含零支持。负收益以固定人口上的 `repair-break` 和配对 correctness 差呈现，不以最有利分层替代主结果。已看 FIT 的诊断不能改称新独立泛化验证；候选数和重叠三点不是独立重复。

只有 GT-free 全量 score、rowmap、frame receipt **全局封印并复核之后**，才读取 audited eligible FIT manual identity。unknown 仍 unknown，不生成 negative/null；released ID 仅作 locator。任何派生文件另存新的诊断目录，保留原输出/输入 SHA、执行回执、明确单位和原始分母。三轮全部进行并同时呈现，不能找到一种解释就停止检查其余解释。若第一轮发现错误，先停科学归因，错误输出留档；修复产生独立版本并重做全人口及独立审计，不覆盖原结果。

## 第一轮：实现与时序，至少两项独立核验

| 检查 | 固定操作与证据 | 能排除或定位什么 |
|---|---|---|
| 1A：独立端到端数值复算 | 使用 fresh verifier 全入口，独立解析 cache/identity投影、重建 released-ID history、按原帧号手写球面外推，不 import 新 producer/core/GT decoder。逐 block 对 raw col0 与 parent 完全一致；对 angular、六个 masks、rho/theta、past locator、全部人口/统计及 seal 逐项核验，记录最大差、容差与实际退出。 | 正反方向写反、单位/ρ错误、行列错位、零/退化分支、raw比较人口变更、统计repair/break错算。已有数学单元通过不能替代真实映射审计。 |
| 1B：history快照与 query隔离 | 以全部 sealed locator验证 `t_prev<t_ref<t_query`、video/source reset、唯一非负ID、current duplicate及next-unique失效规则，空帧不跳过。独立重放必须得到相同 previous image/row/time，后续 history mutation 不改变既有reference。保留 synthetic query-appearance/box改变而prediction不变、future帧/ID突变不回写past，以及tie最低ref dump_row等反例回执。 | past定位和可变引用错误；query/future泄漏。不能只比较分数，因为不同预测可能偶然得到同一分数。 |
| 1C：可核对的退化恒等与加权单位 | last-zero预测严格零；missing/previous-zero/antipodal回退last，coincident静态；在相同原始规范化下验证相应 angular score等于raw（浮点容差按实现声明），不能把无历史placeholders当ρ/θ观测。核对reference exposure、candidate-weight与distinct triple数不同。 | fallback制造收益/损害、零占位污染直方图、把同memory的Q次重复当独立变化。 |

若第一轮均通过，只能说本实现/映射的已核验边界内没有找到错误，不能证明ID真实或恒角速度合理。

## 第二轮：past ID 污染与干净三点，至少两项读出

**clean known same triplet** 固定定义：immutable `previous locator / reference row / query row` 同video，三个时刻严格递增，三个eligible manual identity均已知且相同。只用原 released-ID locator，不用GT搜一个更有利过去替换它；数学是否directed另列，不把退化clean三点删掉。单位同时报告candidate exposure、distinct `(video,prevImage/row,refImage/row,queryImage/row)`、source/video/GTtrajectory；不能把 `GT_ID_triplets` 聚合数误称独立时间三点。

| 检查 | 固定读出 | 解释边界 |
|---|---|---|
| 2A：全量三点身份分解 | 对所有原 endpoint-known-same正例按 `past_known_same / past_known_different / past_unknown` 三分，并在每组报告 locator与退化mask、raw和angular正例cos、配对预测误差差、ρ/θ全边界、source/trajectory覆盖；三组和恢复原正例总量。clean三点只作为预声明诊断分层，不换主分母。 | 已知错误past能说明locator可靠性问题；unknown不能推成污染。clean正例仍预测更差则不能把全部负收益归咎于past ID。组间身份覆盖不随机，因此不能把三组差当污染因果效应。 |
| 2B：正例与干扰候选分别归因 | 固定每个known P+N query的全部known候选、tie和原scores。保存raw top1、angular top1，以及全部known正例past状态的集合签名；按所有正例都clean / clean与其他混合 / 无clean（再分known-different、unknown、混合）穷尽分层。每组完整报告四格及分母，不按谁repair筛query。对导致break的known负例仅追加其reference与past为same/different/unknown诊断；不得删除坏history的负例重跑主排序。 | 防止把“正例干净、负例被错误预测靠近query”误判成正例动力学失败，也防止只保留有利正例的ranking。clean正例误差与完整ranking必须分开。 |
| 2C：污染敏感性反例 | 先保留现有synthetic wrong-but-unique-ID反例；若以后增加数据干预，另冻新派生诊断：对原clean三点按immutable顺序固定供体/时间规则置换past，输入不使用query appearance/score选供体，身份仅在封印后描述污染覆盖。全部原triplet及未实际改变者报告，禁止给主run做GT纠错。 | 说明普通角速度对污染是否脆弱；人工污染不是自然污染率，也不能给clean/dirty差提供无混杂因果结论。此项不是立即执行授权或更改原producer。 |

现有 `history_truth_support` 只给身份支持汇总，`support.sqlite` 只存query p/n与correct bits，**不足以直接给出上述clean分层和预测误差**。后续实现必须从已封印 `frame_pairs.jsonl` 的forecast、`shards/v*.rows.json` 的locators、immutable cache raw和合法identity join流式派生；不得从汇总反推逐triplet结论或声称当前已有这些结果。derived诊断也要先封印与identity无关的prediction/row映射，再做label分层。

## 第三轮：恒角速度方向/幅值假设，至少两项读出

在所有原clean三点上保留全部退化和缺失原因；directed且非退化分支才计算有定义的方向量。future/query raw只作为封印后诊断目标，绝不能回流到history、fallback或模型选择。

| 检查 | 固定量与比较 | 可证伪解释 |
|---|---|---|
| 3A：端点误差和局部方向/速度 | 令 `a=normalize(previous), b=normalize(reference), c=normalize(query)`。比较raw静态与原angular的 `1-cos` /球面距离，保留逐三点配对和macro。已定义时计算past前向切向 `v=(cosθ b-a)/sinθ`，observed切向 `w=(c-(b·c)b)/sinφ`，报告 `v·w`、past speed `θ/(t_ref-t_prev)` 与observed shortest-arc speed `φ/(t_query-t_ref)` 的差/比及不可定义原因。 | 方向经常反向说明局部恒向外推不成立；方向相近但幅值偏离支持速度不恒定/测量噪声解释；方向/幅值均合理而ranking坏，需要看负例及margin。只能观察最短弧，长gap的真实绕圈/路径不可辨识，不能声称恢复真实角速度。 |
| 3B：完整ρ×θ分层与退化归因 | 严格使用下面所有固定边界，给marginal和完整交叉表，两个lag、两scope/pop、全部source以及clean/known-different/unknown均报告。每格列reference/triplet/query单位、支持、raw/angular误差、可定义方向和repair/break读出；空格为0/NA，未知为单独missing。额外仅按数学结构列 `ρθ≥π`、`ρθ≥2π` 覆盖与误差（含其补集和总量），不据它截断预测。 | 若损害集中高ρ/θ，提示恒速外推/长gap不适合，但不能删除这些格救结果；若clean低ρ/低θ也坏，单靠长gap转圈解释不足。层间对比不等于同source内因果效应。 |
| 3C：静态与恒速假说的完整反证 | 所有clean三点同时列raw静态、原angular误差及固定两lag共同query（不强求同reference）身份ranking。引用synthetic已知恒速与方向改变、停转/加速的确定例子证明量的符号解释，再与真实clean分布比较。若要加入时间/历史顺序破坏，只能另冻一个固定破坏映射、保持端点及监督、全人口报告；不得扫衰减alpha/clip/阈值。 | raw静态在clean上仍更好，直接拒绝本人口上的普通恒角速度优势；它不构成“所有可学习外观动态都不可能”的定理，也不自动支持更复杂网络。 |

固定ρ边界：`[0,1,2,4,8,16,32,64,128]`；θ边界：`[0,.01,.05,.1,.25,.5,1,2,π]`。沿用producer `searchsorted(edges,value,side='right')`：underflow `<首边界`，中间 `[a,b)`，最后 `≥末边界`；因此θ恰为π属于最后bin。每一边界都报告，不能只报最佳格，不能合并空/坏格以增强叙事。原distribution排除missing/last-zero的placeholder，但这些排除项必须另列mask分母。ρ/θ属于memory-reference-time属性，query-level ranking若多memory落不同格，必须用固定signature或将其明确作为edge/exposure统计；**不能任选winning候选的bin给整个query贴标签**。同query多正例的cross-bin重复必须显式报告，禁止把格内query数直接相加冒充总query数。

## 若 clean history 仍坏：两种后续假说怎样区分

这时优先保留“普通恒速假设被拒绝”的结论。后续均是独立、有限、source OOF 的探索假说，不把已有adapter、共享变换、attention或正交指数改名作创新；近邻报告仍为 `HOLD_NOVELTY_NOT_ESTABLISHED`，TFLF/SFP全文仍缺。下列只提出判据，不启动训练或更改现有fixed follow-up。

| 假说 | 在看结果前能写出的可证伪预测与必要控制 | 拒绝条件 |
|---|---|---|
| H_scene：有可共享的场景扰动 | 同一video/相近时间，不同已知实例的**过去**变化有超出object-specific噪声的共同成分；仅从目标之外其他track的过去估计共享变化，应能预测该目标下一观测。先审计同步、跨至少两个身份且多source的支持。若训练，共用小容量past-only算子，固定source OOF；target track leave-one-out、过去time打乱、异video匹配时间错配、同预算per-track预测、ordinary adapter均作为控制。所有缺乏多实例支持的帧原样fallback并全量报告；不能只保留相机运动明显片段。 | 共同变化仅在同一身份内出现、跨实例转移不超过静态/adapter、正确时序不超过破坏对照、或增益由少source/单时间段主导，均拒绝共享场景机制。无camera/光照观测只能叫共同feature扰动，不归因真实相机/光照。 |
| H_identity：冻结raw的实例区分不足 | 使用已建议的普通source-OOF identity adapter，完全不含geometry/history，固定监督与预算。预测其**纯cosine**在相同全候选、两lag和全source中改善known身份ranking；同一adapter静态与动态再作独立控制。改善只在pair head出现不能支持表征本体改善。 | 纯adaptercos无双lag一致改善、对source/seed极敏感，或仅记住source，则不支持该假说；即便改善也只是缓存后adapter，不能称backbone/ROI改进或新动态机制。 |

两假说不是互斥真相，可在以后单独冻结的2×2中用同一个adapter/同一个共享算子作主效应与interaction比较，不能先搜最好的组合。共享正交算子对同一批同参考时刻memory保Gram只是数学恒等，不是预测证据；异reference时刻接受不同算子就不保同一Gram。identity adapter若改善而共享时序算子无增量，应保留普通表征改进解释；shared cross-object/time-correct增量超过adapter、静态和破坏对照才提供共享动态的有限支持。二者都失败就停止这两条解释，不改lag/source补救。

## 输出完成与报告模板

未来每轮至少两项以上检查均须有实际执行/独立复算结果，当前本文件不能计为三轮已完成。建议新增独立诊断manifest、全格机器表、逐source/trajectory覆盖和一页解释；每项列 `PASS / FAIL / NOT_IDENTIFIABLE / INSUFFICIENT_SUPPORT`，不得把unknown改为PASS。最终解释依次回答：实现是否排除已知错误；负收益有多少发生于clean三点/错误past/unknown；clean上方向、速度与完整候选干扰分别怎样。完整负结果保留，明确只验证conditional known ranking，不能冒称在线assignment或官方tracking指标变化。
