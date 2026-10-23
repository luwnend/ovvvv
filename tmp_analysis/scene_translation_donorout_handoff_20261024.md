# GT-free外部运动预测诊断续接

2026-10-24。**真实诊断已于03:42:40 Asia/Shanghai启动，未完成**。独立preflight实际exit0/9.07秒，37pins/7保护/人口核对通过。原15模型source-OOF继续运行，不能重启、改pin或提前读取留出指标；整体论文目标仍active。

## 固定范围

- 工作目录`/home/chenhc/ov/OVMOT_RESEARCH`，托管副本`/home/chenhc/ov/ovvvv`。
- Python`/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python`，Python3.8.20/NumPy1.24.4，CPU单线程，CUDA关闭。
- 协议`tmp_analysis/scene_translation_donorout_protocol_20261024.json` SHA`7ba6cb5a7151dd462ec2abde7cef5c8e2b310d3f9a3f6f1dd16f774217e78220`，37项input pins。
- producer SHA`1b1cb57ad43898b58ad9e54c39444f4bc7953ab5ec91fc2df5cab60bbb120aa7`，cached SHA`5eaf00fa1bd52ab0c5a3430fcdf90fd9609d4c17fdc3d81db06cf25bc41b0f32`。
- auditor SHA`fa6463b8809a0490cfad3fac608cf8ccbd712ef687b69048d672b183393f5d1c`，runner SHA`686c41b2e80c9e995a9cd45aac466371c5a015223ba351c67762095ce88d09ec`。
- 341原sealed FIT源组、403缓存视频、13,791帧、613,536行；lag1/4，27,582含prefix帧对receipt。全部raw严格互选h先固定，再作fullpool/row/strict三臂；strict先删端点、same-proposal、direct positive overlap，再重新matching。
- 只取raw/box/proposal/frame定位七键；禁止身份、类别、released ID、posterior、GT以及新OOF结果。原数据、baseline、upstream、旧sealed及另一会话训练保护不改。

## QA与运行

cached与producer各三轮及独审通过；runner真实child/wait成功和失败停止通过。最终auditor三份v2 QA均actualexit0：8组scalar、36case/139h、完整3源3视频20receipt28h CLI。C/F/strided成员和两hash精确一致；5类重封口篡改actualexit1。正式真实审计不会import producer/cached。

输出`results/scene_translation_donorout_20261024_v1`；执行链`results/scene_translation_donorout_execution_20261024_v1`。preflight SHA`a18679f7d7fa4f18070d17cc999519ebc0bf16e8875d4c500ff2521eafca1a86`，launch_spec SHA`90c7fb204ad38a28960abaac95b2daae9b5921e98ae936f7b1144c8d39c0de78`。禁止覆盖已有run或自动重试。启动空盘门31GiB，现场独审free42.01GB；预计30–120分钟CPU，非真实完整测量。

启动时supervisor PID387140/start_ticks574726111，producerPID387161/start_ticks574726169（boot见started/identity receipts）。03:42:51现场两者身份匹配running，首视频1604h/3.857秒，无failure。该旧PID快照不能替代现场status。同期主OOF 6/15 actualexit0，第7个fold1_seed20261025正常运行，尚无global seal/评价。

独立启动核验`scene_translation_donorout_launch_live_audit_20261024.json` SHA`67798aa5976855226c219db77eae7a1e5a9f3f36a3c0ace601e0e1dfcfd68b38`，03:44:02 actualexit0/0.51秒；39launch pins、37冻结pins、preflight/checker及7保护一致，两个进程身份/亲缘/CPU单线程环境和无GPU设备通过。仍不算完成。

```bash
cd /home/chenhc/ov/OVMOT_RESEARCH
/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python tmp_analysis/run_scene_translation_donorout_20261024.py --status
/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python tmp_analysis/run_identity_spacetime_oof_20261024.py --status
```

状态需要PID+boot+start_ticks+cmdline现场一致，不能单凭旧PID。两任务互不读新指标、互不修改协议。

## 完成与解释

全403视频seal后生成无身份描述汇总；fresh auditor从raw重建全部h、37列、两名单SHA、prefix/offset/source/common统计。最终必须producer/auditor/finalizer actualexit0，execution_receipt、independent audit PASS、completion_receipt、finalization.exit.json及supervisor_complete俱全，37pins/7保护一致才验收。

主读出为strict各lag帧内h增益median→源内有支持帧均值→有支持源等权；两lag均正才最小预测门PASS。所有341源显式保留，缺失不补零；共同支持先同帧同h相交。PASS也只支持raw伪对应的外部位移预测，不证明身份/相机/排序/tracking/泛化/创新。无论结果如何，不修改原OOF或调阈值挽救。

下一步必须验收真实诊断与主15模型OOF；identity-only adapter、原几何/掩码/置乱控制都完成后，才能决定网络机制。若检验失败，区分实现和具体机制，遵守至少三轮每轮两项的核验边界。

阶段交接更新`report/项目进度.md`、`report/COVTrack_外部运动预测诊断_20261024.md`和本文件；脚本`sync_scene_translation_stage_20261024.py`显式同步小文件，commit/push后核remote，数据/权重/逐帧缓存不上传。SSH已可用，无需新key。
