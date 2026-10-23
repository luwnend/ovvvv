# Source OOF identity × spacetime 续接

当前：完整独立预审已通过，2026-10-24 02:42:27 Asia/Shanghai 已启动真实CPU后台任务。不要重跑已完成FIT支持/angular，也不要重启本OOF输出目录。
正式protocol SHA `a7169f8674d8e019782820bc7b4a99ff4b2082bdeaa0a731e60352a488ed9be1`；preflight SHA `9fb4633e125a6f646a9de55642ad376da4642df220ff3a87673fe0c9b2dc100c`。
启动不是完成；后续状态以实际 execution receipts 和最终fresh audit为准。

阶段补充（2026-10-24 03:03:57 Asia/Shanghai快照）：前两个worker均完成训练/完整评分并actual exit0（578.933/592.738秒），第三个运行；70 launch pins及7保护现场重核未变，尚无全局seal或新OOF评价。另已完成共同运动观测/文献/参考模块独审，入口`report/COVTrack_共同运动与独立几何参照_20261024.md`；此模块未接入当前OOF，不能改变本协议或把其普通控制计为新方法。快照凭据`identity_spacetime_scene_audit_live_checkpoint_20261024.json`，当前状态仍用下述只读命令，不重启。

## 固定目标与边界

- 工作目录 `/home/chenhc/ov/OVMOT_RESEARCH`；托管副本 `/home/chenhc/ov/ovvvv`。
- Python `/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python`（3.8）；CPU float64 单线程。
- 341 原 sealed FIT source，403 视频、13,791 帧；lag1/4 完整 Q×M 每 seed
  55,352,164 候选。5 source folds × 3 seeds，共 15 独立模型、十列评分。
- source fold counts 为 74/72/69/64/62，metadata-only 已独立重算，manifest 不能
  按结果调整。未知不负例；tune/test/class-heldout target/correctness 保持关闭。
- adapter50/head20固定最终epoch；普通 adapter/head 是控制，不计为论文新意。
- 全15模型 held-out 完整分数、rowmap、checkpoint、normalizer 必须先全局封存，
  再统一合法 FIT identity 评价，不能逐 fold 看指标后调下一 fold。
- 旧 role 只作 metadata；不复用 nuisance/calibrator，不将已看 FIT 称新确认性集。
- 冻结 upstream、baseline、旧 sealed 产物与另一会话训练目录，不能覆盖。

## 新代码与入口

- `identity_spacetime_oof_data_20261024.py`：GT-free metadata/frame/history 与仅
  train-source 安全身份投影。
- `train_identity_spacetime_oof_model_20261024.py`：单个 fold/seed 的隔离 worker。
- `evaluate_identity_spacetime_oof_20261024.py`：全局 seal 成功后统一评价。
- `run_identity_spacetime_oof_20261024.py`：pin/preflight门、15个实际 child wait、
  全局seal、评价、detached supervisor、新进程 output auditor、finalizer。
- `audit_identity_spacetime_oof_implementation_20261024.py`：独立完整实现预审。
- `verify_identity_spacetime_oof_output_20261024.py`：独立 checkpoint 数学重算。
- `finalize_identity_spacetime_oof_stage_20261024.py`：实际exit/audit之后才写完成凭据。
- `sync_identity_spacetime_oof_stage_20261024.py`：显式白名单同步轻量证据，不推大数据。

只读状态命令（正式协议存在后）：

```bash
cd /home/chenhc/ov/OVMOT_RESEARCH
/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python tmp_analysis/run_identity_spacetime_oof_20261024.py --status
```

预定输出 `results/identity_spacetime_oof_20261024_v1`；执行链
`results/identity_spacetime_oof_execution_20261024_v1`。禁止覆盖已有目录或自行重试。
进程状态必须同时比 PID、boot ID、start ticks 和 cmdline；不能只凭旧 PID 判断活着。

## 当前核验范围

纯模块三轮与独审、数据层三轮与独审、trainer/evaluator/coordinator基础合成已存在；
这些不能替代完整 worker、15模型 global seal 和输出审计。完整实际 worker 合成入口
已揭示一次 `np.int64` training trace JSON 序列化实现错误，已修复并重新独立检验；
该错误已修复并通过完整实际worker突变与15worker输出复核；真实任务现已启动，暂无方法性能结论。

最终同版纯合成计时按旧 sealed hard 人口估计训练约61.4分钟，完整评分约19.2分钟，另加真实
IO/hash、几何构建、评价和独立复核。实际worker训练人口预期五折约
4184/4619/4831/4342/4720 query-lags，但须现场核验；三knownrefs/query只是计时
假设。保守预留数小时，不能将 toy 线性外推当完整实测。score 原始约12.37GiB，
启动最低空盘18GiB。长CPU后台可独立于对话继续，完成必须看到实际exit与audit。

## 训练后判读

先确认每seed raw与parent逐候选和全部 source/union统计完全一致、主共同query1331，
并通过fresh独审；再看纯adaptercos−raw、G1−G0/Gmask/Gperm、2×2interaction、
每seed/source/role/block和source删除敏感性。所有arms同分母，不挑最好seed/lag。
负结果先排实现问题，按三轮每轮至少两项核验后限定具体停止范围；不以更名当创新。

进度文档在完整阶段交接/结束时更新，再allowlist同步、diff/hash核验、commit/push并
核remote。SSH已可用，无须新密钥，不覆盖Git的ssh配置。

## 现场启动凭据

- execution `launch_spec.json` SHA `818cecdd0c35249aaf1e91d1ef58b2bb0782d4e3675b4a07cd1cb4266d7a62b1`。
- supervisor PID358090/start_ticks574364699；coordinator PID358095/start_ticks574364815。这些是启动时绑定，不应单凭旧PID认为仍活着；用上面的只读status命令现场核对。
- 首个worker为fold0_seed20261024，GT-free preparation已完成；实际训练人口和后续进度以各worker stdout为准。
- 预计100–240分钟（含未测IO和audit），可关闭对话后续接；不停止、不重新launch。完成必须有execution_receipt、独立audit PASS、completion_receipt及finalization.exit.json实际exit0。
## 新并行诊断快照（2026-10-24 03:42:51）

主OOF已6/15模型actualexit0，第7个`fold1_seed20261025`身份匹配running；尚无global seal/统一评价，不能读取新指标或改pin。并行的GT-free donor-out位移预测诊断已于03:42:40启动，独立协议`7ba6cb5a…`及preflight`a18679f7…`，37pins/7保护通过，31GiB并行启动门通过。其仅用raw/box/proposal/frame定位，不读本OOF输出；入口`tmp_analysis/scene_translation_donorout_handoff_20261024.md`，输出`results/scene_translation_donorout_20261024_v1`。新诊断不可替代本OOF最终15模型seal、统一评价和fresh output audit。
