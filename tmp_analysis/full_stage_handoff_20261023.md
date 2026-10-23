# Full采集与support通过后的暂停交接（2026-10-23）

用户要暂停Codex对话、过会儿重新打开。研究目标未完成；当前停止启动后续任务。完整采集和full support均已完成，强mean真实拟合尚未启动。当前状态以`results/base_error_full_collection_20261023/dialogue_pause_receipt.json`及其绑定的独立审计为准。旧pause文件和准备manifest是历史快照。

## 已验收断点

- 全量采集15,283帧、448视频、383源组、681,602条输出；实际2.9068小时。collector和两个CPU审计子进程实际exit0，独立终审PASS。外层supervisor自身终端exit没有直接wait证据。
- Full support工具session51211由root直接poll确认exit0；独立全输出审计session68926由evaluator agent直接poll确认exit0，耗时95.71秒。`support_v1/independent_support_audit.json`状态`PASS_INDEPENDENT_FULL_SUPPORT_OUTPUT_AUDIT`，SHA `92174aaa954f6e66cd0e7f520f2f883f7b9e0017c84fbf9bf4509e734ba714e8`；绑定support receipt SHA `e206051d0809b8e20b24529254eacece96f997c72b7155e8264197e7c07aae1e`。
- 全部emitted先按`(video_id,local_track_id)`封存：681,602观测、144,930轨迹。独立复算全部15,283帧几何匹配/排除计数，全部46,908人工anchor来源、23,887监督行与193,405pairs。fit/tune可监督观测21,661/2,226，错误12,694/1,081；重复观测对不是独立样本数。
- 强mean实现、独立实现预审、完整候选合成检查已完成；真实full mean未实跑。`diagnose_full_mean_quality.py` SHA `464b9ad6dd551a6f813de1eded509d0fd0d00f26336cf90ee2cd54fb14e84e17`。
- Pair协议JSON冻结SHA `fc10e338c0160e389008aa34de13dd384cc0466bdd9df08ba7f937104de00718`。`build_full_pair_context_cache.py`仅WIP（CPU合成通过，真实索引未跑，尚待独立审查）。无真实pair训练、新detector/tracker训练或方法收益结论。

## 恢复次序

1. 读`report/项目进度.md`、本文件和`dialogue_pause_receipt.json`；核对receipt哈希和独立PASS。不要重复采集或重建已有support，不覆盖sealed目录。系统`python3`缺NumPy，科学计算用下方固定解释器。
2. 现有full mean执行门槛已满足，恢复后可运行以下CPU诊断（本次尚未执行），再独立审计输出。11候选：B0、6个ridge、4个HGB；5折外层/3折内层按源组选模，固定inner-OOF Platt；B3校准模型为预定主模型，tune只评估一次。它仅验证hard平均错误质量，不能开放pair训练门槛。

```bash
/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python -u \
  tmp_analysis/diagnose_full_mean_quality.py \
  --support results/base_error_full_collection_20261023/support_v1 \
  --out results/base_error_full_collection_20261023/mean_quality_v1 \
  --execute > results/base_error_full_collection_20261023/execution_logs/mean_quality_v1.log 2>&1
```

3. 必须记录真实工具exit；成功后运行独立`audit_full_mean_quality.py --diagnostic <mean输出目录>`，生成`independent_mean_quality_audit.json`。该verifier已用全候选合成样例校验；真实输出审计尚未执行。禁止仅凭producer PASS或安静日志判断成功/失败。
4. Pair sealer先独立审查，再决定实跑新目录。只读GT-free manifest/两份index、full metadata plan和协议；不读监督/原GT/NPZ，不用GT筛选reference。每轨迹长度至少4，每4个lag bin各4条hash边，reference为同轨全部其他输出并排除两端；源组元数据拆分mean/pair角色。
5. 后续再做向量均值/可识别性审计、强quality/agreement、同容量attention/校准控制与独立泛化。新文献lead `Hierarchical Aggregation of Semantic Uncertainty in 3D Scene Graphs` (arXiv2609.22351)当前只核对摘要，全文机制对比待办，不能写成已经精读。

## 关键科学边界

Hard同错=agreement×bothwrong。coarse均值中心化仍会残留共享未知Y引入的协方差；若pi是真实条件标签后验，协议旧plug-in超额残差条件期望恒为0，只能诊断pi误设，不能作为新颖性或训练gate。当前候选主目标改为类别共享的Y条件均值/残差矩，推理对1203类积分；它仍是条件矩，不能称appearance因果。C129同容量校准控制、teacher-forced及disjoint-reference证伪均待真实执行。

所有输出先聚合再监督；未匹配为unknown；test/留出类正确性关闭。已有旧“TCES失败”错误分析已从正式结论撤回，历史归档保留溯源，不可沿用。保护发布baseline、`repos/COVTrack`、数据、pilot封存物及另一会话`results/covtrack_ctao_train20_bs24_20261022`训练/配置/进程。

## Git与长任务

研究工作区`/home/chenhc/ov/OVMOT_RESEARCH`，托管副本`/home/chenhc/ov/ovvvv`，远端`https://github.com/luwnend/ovvvv`，既有SSH可用。本次用`sync_research_stage.py --stage base_full_pause_20261023`同步暂停快照，明确区分通过的采集/support、预审实现和WIP；大型receipt/索引/后验/模型保留本地并绑定SHA。全阶段最终同步仍待mean完成后的`base_full_20261023`，不能将本暂停快照当最终研究完成。

恢复后GPU启动前现场核对owner/UUID，非4GB且启动前没有chenhc进程；他人任务可按授权共卡。预计GPU超过20分钟时先可靠detached启动并验证、保存交接，再暂停Codex；CPU任务不自动触发该规则。本次不再启动mean或新长任务。
