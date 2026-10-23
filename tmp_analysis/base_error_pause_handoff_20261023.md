# 2026-10-23 用户暂停交接（最新）

用户要求暂停Codex对话，稍后回来。研究目标未完成；不启动新的full采集。项目进度与本文件优先于历史pilot暂停交接。

## 已完成并封存

- `results/base_error_collection_20261023_v3/execution_receipt.json`：pilot实际exit0；41视频/1,299帧/54,924输出/12,876预测轨迹，双validator通过。
- `support_v1/independent_support_audit.json`：fit/tune监督1,067/315条，错误615/228；6,876/1,581对只是重复观测。全部emitted先封存，几何匹配全原始GT后仅使用允许的人工anchor监督；unmatched为unknown。
- `mean_quality_v1/independent_mean_quality_audit.json`：固定B0/B1质量诊断通过。层级等权tune Brier 0.233728→0.191559；仍低估错误率，不能作为可靠中心化或pair机制证据。没有新tracker训练。
- `results/base_error_full_collection_20261023/independent_metadata_audit.json`：15,283帧/448视频/383源组/46,908人工anchor，fit/tune eligible targets 39,903/4,557。无test帧；独立metadata PASS不等于采集完成。
- 全量采集尚未启动；本次`pause_handoff.json`确认没有launch/exec receipt和运行进程。旧TCES错误失败分析已从正式文档删除，历史来源仍撤回。

## 恢复顺序

1. 先读`report/项目进度.md`和本文件，查看full stage是否后来出现`background_launch_receipt.json`/`background_execution_receipt.json`、进程及日志。已有run时不重复启动。
2. 使用`/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python`。现场检查GPU UUID、空余显存和owner：非4GB且启动前无chenhc进程；其他用户进程按最新授权允许共卡。保护PID3546537/3546538及`results/covtrack_ctao_train20_bs24_20261022`，不停止或改动。
3. metadata审计与annotation SHA仍匹配、目标run尚不存在时，启动已准备的supervisor；预计1–2小时，先向用户报告耗时。命令如下（工作目录`/home/chenhc/ov/OVMOT_RESEARCH`）：

```bash
setsid nohup /usr/bin/python3 -u tmp_analysis/run_full_collection_background.py \
  --stage /home/chenhc/ov/OVMOT_RESEARCH/results/base_error_full_collection_20261023 \
  --run /home/chenhc/ov/OVMOT_RESEARCH/results/base_error_full_collection_20261023/full_collect_v1 \
  > /home/chenhc/ov/OVMOT_RESEARCH/results/base_error_full_collection_20261023/background_supervisor.log \
  2>&1 < /dev/null &
```

4. 启动后核对PID/初始帧与launch receipt；不要同步等待1–2小时。supervisor记录真实exit并运行双CPU校验，不重试、不匹配GT、不训练。
5. 收集完后核对真实exit、全帧receipt、双audit与保护hash，再独立适配full support builder。旧builder锁pilot，不能直接套用。继续nested source-CV强quality控制，再检验pair共同参考残差目标；test/heldout正确性保持关闭。

Git托管副本`/home/chenhc/ov/ovvvv`，远端`https://github.com/luwnend/ovvvv`。仅同步allowlist轻量脚本/报告/回执，大型数据/模型/原始后验留本地；既有GitHub SSH已可用，不需新密钥。
