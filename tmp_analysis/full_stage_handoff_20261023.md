# Full阶段当前交接（2026-10-23）

用户已恢复，研究继续；整体目标未完成。目前没有本轮GPU任务或新detector/tracker/pair训练。旧pause JSON均为历史快照，不能据此重启已完成任务。

## 已完成且独立通过

| 阶段 | 真实执行及独立审计 | 当前结论 |
|---|---|---|
| Full采集 | collector及两个CPU审计子进程actual exit0；独立终审PASS | 15,283帧/448视频/681,602输出，10464.512秒=2.9068小时；原1–2小时估计偏低 |
| Full support | root session51211/exit0；独立session68926/exit0 | 全emitted681,602观测/144,930轨迹先封存；fit/tune监督21,661/2,226，全部匹配/来源/行/pairs核验 |
| 强mean质量控制 | root session21525/exit0，265.523秒；独立session96818/exit0，139.852秒 | B3主模型fit/tune Brier=0.147018/0.136580；B1=0.200349/0.203912；普通控制，不是新方法收益 |
| GT-free pair上下文 | root session5623/exit0，45.802秒；独立session90413/exit0，54.175秒 | 623,878边，全部3,947,118候选SHA重算；172 mean/169 pair/42 tune元数据源组固定 |
| Fit身份支持 | root session21762/exit0，33.543秒；独立session70485/exit0，80.099秒 | 560,371固定fit边中same12158/different819/单端unknown9388/双端unknown538006；分类正确性值未解码 |

Full采集外层detached supervisor自身terminal exit没有直接wait证据；不能将其子进程exit0改写成直接观测外层exit0。没有当前采集任务，勿重复采集、support、mean、索引或身份join。

强mean协议固定14维推理质量特征、源组5折outer/3折inner、11候选，预定B3为inner-OOF Platt后模型，tune仅最终评一次。Tune B3−B1 Brier的1000次paired源组bootstrap区间[-0.083776,-0.052271]；B3−B2区间含0，不据tune换主模型。区间条件于冻结预测、未重拟合。历史HGB权重由已审代码和合成合约承担，独立审计重建216模型对象及全部预测，未重拟合HGB。

身份different边涉及135源组/280预测轨迹；最大源组28条，删后791条。只有12,977边两端身份已知，不能把819/560,371当总体污染率。保留全部fit边和原reference补集，unknown不作负例，不能按whole-track纯度选样。混合监督文件确实被扫描/哈希，仅解码fit身份字段，tune身份与分类正确性值未解码。

## 当前路线决定

冻结`full_mean_pair_gate_protocol_20261023.json` SHA `fc10e338c0160e389008aa34de13dd384cc0466bdd9df08ba7f937104de00718`保持原样。三轮、每轮两项精确有限分布检查确认：`E[G|A,Y]`用`P(Y|p_i,p_j,A)`积分，一般不等于所需的`E[G|p_i,p_j,A]`；oracle均值/pi也有反例。统一conditioning后，该G最优预测由full p与共同标签后验确定，击败有限C129不能识别额外“内在协方差”。

**STOP这条具体部署公式及协方差新颖性归因，不启动其真实pair拟合。** 不是已实现pair代码失败，也不否定所有关系预测/attention。决定见`pair_route_decision.json`，精确证据见`tmp_analysis/pair_conditioning_identifiability_audit_20261023.md`和synthetic回执。

arXiv2609.22351v1全文方法/实验/参考文献及PDF关键页已核验，其所谓fully correlated聚合为max。它覆盖共享模型/词表重复错误的宽泛动机，并未证明与本候选精确实现撞车。早先“只读摘要”是历史状态。全文审计与来源SHA见`semantic_uncertainty_2609_22351_fulltext_audit_20261023.md`和source receipt。

## 权威回执与大文件

均在`results/base_error_full_collection_20261023/`：

- `fit_tune_execution_receipt.json`：采集/support/强mean实际exit、输入输出SHA和保护检查；mean独立audit SHA `f1f1241a0adc6c9d922373769bb3b535cb920a70661c0476363cb719ab8b2721`。
- `pair_context_execution_receipt.json`：GT-free索引实际exit及独立全查；audit SHA `df7c386a8f01d3dbfce928e58758f9c455f685e8485dbfa8b19a0cd4ac936e81`。
- `fit_identity_execution_receipt.json`及`independent_fit_identity_edge_support_audit_execution_receipt.json`：身份join和fresh独立verifier真实exit0。Audit SHA `8e0cafbb0dea4c9120119e62ff0bb9d5a5a3bd06855d5140117aa2481372beaf`。
- `base_fit_identity_edge_support_20261023/identity_support_brief.json`：精确派生all/per-role/固定24cells，绑定22.96MB原summary、逐边rows及audit SHA；不改变原输出。

2.63MB support receipt、22.96MB身份summary、原始逐帧NPZ/JSONL、模型对象及文献HTML/PDF仅本地保留，不直接Git-sync。监督来源、split及禁用边界继承已独立审计的固定输入，不可换新源冒充同run。

## 下一步

1. 在独立新目录做fit-only observed-label oracle分类对照：固定原context、角色、完整词表与uniform pooling；比较all-emitted / manual-known / same-GT / deterministic count-matched参考快照。主比较使用同covered分母，另外报告完整合法端点覆盖。按源组→GT轨迹→唯一端点等权，重复contexts分配端点权重。GT快照只诊断，不改实际cache/readout。
2. 稀疏oracle不是数学上界；收益不等于GT-free可辨识性、新颖性或TETA。分开标签覆盖/数量效应，再检查同类异身份、class-blind separator及一级来源tracklet split/可靠记忆边界。不得因819条different边直接训练router。
3. 新脚本先三轮每轮至少两项合成/实现检查及独立预审，root再执行真实CPU诊断并独立审计；不重复旧已封存任务。系统python3缺NumPy，科学计算用`/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python`。
4. 新方法有超过质量/agreement/共享标签/同容量attention控制的可解释信号后，才进入独立训练目录；val只作探索，独立泛化另验。

## 保护、Git与长任务

全emitted按`(video_id,local_track_id)`先封存再监督；GT不得筛实际readout/reference。test/留出类正确性关闭，tune不用于选模型或改超参。旧错误“TCES失败”结论已从正式文档撤下，历史归档只供溯源。

保护发布baseline、`repos/COVTrack`、原始数据、sealed pilot/full输出及另一会话`results/covtrack_ctao_train20_bs24_20261022`；不停止/修改其训练或配置。此次身份receipt再次核对7项保护文件未变。

研究目录`/home/chenhc/ov/OVMOT_RESEARCH`，托管副本`/home/chenhc/ov/ovvvv`，远端`https://github.com/luwnend/ovvvv`；既有SSH可用，不需新密钥。当前最终阶段用`sync_research_stage.py --stage base_full_20261023`，逐文件核对snapshot SHA、diff后commit/push；旧`base_full_pause_20261023`快照保留历史，不是当前状态。

未来GPU启动前现场核owner/UUID，非4GB且启动前没有chenhc进程；他人任务可按授权共卡。预计GPU超过20分钟：可靠detached启动、验证、保存交接后按用户规则暂停Codex。CPU长任务不自动触发此规则。用户已恢复，本次不会主动暂停研究或标整体完成。
