# 2026-10-23 用户暂停交接（最新）

用户已恢复，并新增“预计GPU用时大于20分钟就暂停Codex对话”规则。full采集预计1–2小时，已启动独立后台；确认首批真实帧后保存本交接并暂停Codex，后台继续。研究目标未完成。项目进度与本文件优先于历史pilot暂停交接。

## 已完成并封存

- `results/base_error_collection_20261023_v3/execution_receipt.json`：pilot实际exit0；41视频/1,299帧/54,924输出/12,876预测轨迹，双validator通过。
- `support_v1/independent_support_audit.json`：fit/tune监督1,067/315条，错误615/228；6,876/1,581对只是重复观测。全部emitted先封存，几何匹配全原始GT后仅使用允许的人工anchor监督；unmatched为unknown。
- `mean_quality_v1/independent_mean_quality_audit.json`：固定B0/B1质量诊断通过。层级等权tune Brier 0.233728→0.191559；仍低估错误率，不能作为可靠中心化或pair机制证据。没有新tracker训练。
- `results/base_error_full_collection_20261023/independent_metadata_audit.json`：15,283帧/448视频/383源组/46,908人工anchor，fit/tune eligible targets 39,903/4,557。无test帧；独立metadata PASS不等于采集完成。
- 全量采集已于14:19（Asia/Shanghai）独立后台启动：supervisor4190240、collector4190241，run=`full_collect_v1`。`background_launch_receipt.json`已存在；最新暂停时快照为`background_pause_handoff.json`。旧`pause_handoff.json`只是上一次未启动时的历史记录，准备manifest也不代表当前运行状态。旧TCES错误失败分析已从正式文档删除，历史来源仍撤回。

## 恢复顺序

1. 先读`report/项目进度.md`和本文件，检查full stage的`background_execution_receipt.json`、真实PID4190240/4190241及`execution_logs/full_collect_v1.log`。已有run，**不得重复启动**；PID消失不等于成功，必须核对真实exit/全部帧/双audit/保护hash。日志短暂不更新也不等于进程终止。
2. 使用`/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python`。现场检查GPU UUID、空余显存和owner：非4GB且启动前无chenhc进程；其他用户进程按最新授权允许共卡。保护PID3546537/3546538及`results/covtrack_ctao_train20_bs24_20261022`，不停止或改动。
3. 下面是本次已执行的启动命令，仅供溯源，**不要直接重跑**。若之后实际失败，先诊断并保留部分输出，另选新run名和回执方案；不能覆盖本run。工作目录`/home/chenhc/ov/OVMOT_RESEARCH`：

```bash
setsid nohup /usr/bin/python3 -u tmp_analysis/run_full_collection_background.py \
  --stage /home/chenhc/ov/OVMOT_RESEARCH/results/base_error_full_collection_20261023 \
  --run /home/chenhc/ov/OVMOT_RESEARCH/results/base_error_full_collection_20261023/full_collect_v1 \
  > /home/chenhc/ov/OVMOT_RESEARCH/results/base_error_full_collection_20261023/background_supervisor.log \
  2>&1 < /dev/null &
```

4. supervisor会在采集结束后记录真实exit并运行双CPU校验，不重试、不匹配GT、不训练；即使Codex对话暂停也独立运行。本轮恢复无需盲等；先查权威运行/完成证据。
5. 收集完后核对真实exit、全帧receipt、双audit与保护hash，再独立适配full support builder。旧builder锁pilot，不能直接套用。继续nested source-CV强quality控制，再检验pair共同参考残差目标；test/heldout正确性保持关闭。

Git托管副本`/home/chenhc/ov/ovvvv`，远端`https://github.com/luwnend/ovvvv`。仅同步allowlist轻量脚本/报告/回执，大型数据/模型/原始后验留本地；既有GitHub SSH已可用，不需新密钥。
