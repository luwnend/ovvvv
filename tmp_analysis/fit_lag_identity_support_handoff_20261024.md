# 完整 FIT 时间间隔与身份支持续接（2026-10-24）

## 当前状态

**真实后台任务已启动，尚未完成。** 2026-10-24 01:04:00 Asia/Shanghai（UTC 2026-10-23 17:04:00）root启动命令实际exit0。supervisor PID305992、producer PID305994的start ticks/boot ID/cmdline均核实一致，ready状态`READY_PINNED_FIT_LAG_SUPPORT_PRODUCER`。完成producer后自动接fresh独立全输出审计；不要另起或重跑。

01:06:12现场检查：806个score shard已生成，全部55,352,164候选全局封存，seal SHA `d24932ab0885bd6b902374532770726cf5fcca1f24fbe9ff9b239c7c538a20a3`；producer仍运行、没有actualexit或audit完成。`progress_receipt.json`是此历史时刻快照；恢复时先查最新status，勿据它猜当前仍在生成。

上一特征阶段代码与文档已推送GitHub commit `8263c3bb7703c9d00609c9f6975f03d2875c2e94`，push实际exit0、remote `refs/heads/main`相同，工作副本clean。旧五模式memory和12源组相邻候选结果已封存，不重跑。

## 为什么做这项

五模式memory没有稀疏anchor身份收益；12源组的同预测类别CV-covered已知候选只有88个可比较query，5个rawcos错误全部来自单一源组。当前不能用该小集合选网络或运动权重。完整FIT审计固定短、长两个间隔，先检查跨源困难身份与正例覆盖，再决定appearance本体与geometry利用的2×2是否有可辨识空间。

## 固定输入与规模

- 全部原sealed FIT source 341个（mean172/pair169）；405个原FIT视频分配中403个有缓存，13,791帧、613,536 retained观测。未采video1720/1974及其source未进入sealed roles，单独保留未采记录，不扩大来源或重分角色。
- sampled-step lag固定`{1,4}`；相对时间另用原始frame_index差，不能当4原始帧或秒。每帧对全部retained Q×M，不按GT、旧ID、分类或结果筛候选；unknown/旧id=-1保留。
- lag1 29,096,237、lag4 26,255,927，共55,352,164候选；六score float64原始约2.474GiB。video×lag memmap，最多五特征帧和一个≤5,929候选块驻留，SQLite只存已知query/身份聚合，不把全部候选写SQLite。
- 全部GT-free score/rowmap/frame receipt全局封存之后才接eligible FIT146原manual identity；unknown不负例。不打开tune/test/heldout targets、GT类别或分类correctness。无positive的已知子集不等于实际unmatched，不能自动给未来null/dustbin监督。

## 实现与启动

producer：`tmp_analysis/audit_fit_lag_identity_support_20261024.py`

SHA `9beb8faf932d13f417d491ddd8f853de001d17a414eac0d59d016b0ffcf3af3b`。

protocol：`tmp_analysis/fit_lag_identity_support_protocol_20261024.json`

SHA `dd84da8fd028af36c223f42a30d19f9f23725eb790e60aaa668700c80a0806ad`。

producer三轮每轮2+最终凭据为`fit_lag_identity_support_synthetic_roundN_20261024_v3.json`。独立preflight为`fit_lag_identity_support_independent_preflight_20261024.json`，状态`PASS_INDEPENDENT_FIT_LAG_IDENTITY_SUPPORT_IMPLEMENTATION_PREFLIGHT`；其独立22帧/3视频/4源组、全部sixfeature/history、184query/48trajectory/108identitypair与100随机ranking scope均通过。

supervisor：`tmp_analysis/run_fit_lag_support_background_20261024.py`，SHA `68818258f50e0d937d00611af20171e17d02fc2386b281095c400edd41a8eae8`；最终synthetic v2实际exit0、17项通过（含launcher退出后后台存活）。禁止自动retry或清理已有out。

真实producer目录`results/fit_lag_identity_support_20261024_v1/`由producer独占创建。执行目录`results/fit_lag_identity_support_execution_20261024_v1/`保存launch spec/receipt、supervisor_ready、producer/verifier进程绑定与actualexit、日志、7保护前后及最终execution_receipt。独立输出审计器为`verify_fit_lag_identity_support_output_20261024.py --out <producer目录>`，最终SHA `74a5c59c957439411acdb7139822c8d83c583584ab07e9e21957e731e648d3ce`；合成完整audit入口actualexit0，receipt SHA `7724f0651c671580e08848217d4f512674366a48b388f30f0b79de207e7dc7a1`。

实际已执行的启动命令（仅供溯源，不要再执行）：

```bash
/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python \
  tmp_analysis/run_fit_lag_support_background_20261024.py launch \
  --producer-sha256 9beb8faf932d13f417d491ddd8f853de001d17a414eac0d59d016b0ffcf3af3b \
  --protocol-sha256 dd84da8fd028af36c223f42a30d19f9f23725eb790e60aaa668700c80a0806ad \
  --preflight-sha256 0344310cdf2ca904ac00cb64dd250db4a0b1b71f7f94904fb56f38d94ed36b0d \
  --verifier-sha256 74a5c59c957439411acdb7139822c8d83c583584ab07e9e21957e731e648d3ce
```

状态查询：

```bash
/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python \
  tmp_analysis/run_fit_lag_support_background_20261024.py status
```

只有PID、start ticks、boot ID、cmdline全部相同才认为原进程还在；最终child exit来自detached supervisor对自有子进程的`wait()`。不能把它描述为root直接观测外层terminal exit。

## 时间、结果与续接

尚无真实完成耗时；producer粗估15–45分钟CPU，fresh全输出重算另需相近预算，估计来自不同实现的旧小run，不是承诺。CUDA禁用。CPU长任务不触发用户的GPU>20分钟暂停规则；后台进程与交接便于对话关闭后续接。root已核ready与绑定进程，继续核首批shard/日志；尚无方法指标或训练结果。

报告全部source、两lag、同预测类别/全部候选、CV共同分母与未知覆盖、GT/source错误分布、同query跨lag支持和全部source held-out fold。支持门仅是必要逻辑条件，不能将55m相关候选当独立样本，也不自动转训练。真正新训练需支持通过后另行冻结并审查具体协议。

保护发布baseline、冻结上游、原数据、旧sealed结果与另一会话训练。磁盘当前约50GiB可用（启动前复查），本任务不清理旧结果。完整score/SQLite/逐帧JSONL/full summary仅本地；`sync_fit_lag_support_stage_20261024.py`显式同步轻量脚本/协议/阶段回执，phase running不冒称完成。全输出通过后派生`support_brief.json`绑定完整summary/SQLite/seal/hash，再更新报告、Git。
