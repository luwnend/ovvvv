# 完整 FIT 时间间隔与身份支持续接（2026-10-24）

## 当前状态

**全FIT真实计算、fresh独立全查与最终哈希复核均已完成；整体研究继续。** 2026-10-24 01:04:00启动，01:11:34 Asia/Shanghai结束。producer192.726秒、独立审计258.325秒；detached supervisor对自有producer305994/verifier307134的`wait()`实际记录两个exit0。执行状态`PASS_FIT_LAG_SUPPORT_EXECUTION_AND_INDEPENDENT_AUDIT`，7保护与全部pin未变。不要重跑旧任务或把启动时process.state当当前活着。

完成入口：`results/fit_lag_identity_support_20261024_v1/completion_receipt.json`，状态`PASS_FIT_LAG_IDENTITY_SUPPORT_AND_INDEPENDENT_OUTPUT_AUDIT`，SHA `5e70740d4940f2a4a7cbf4e94c538b152f277826834ebeb8a58bd70d494f14b8`。轻量`support_brief.json`1,033,840bytes，SHA `25b191940d57c4307f4dc1c4566f3e05f1bacb43c14d16d03ce459a93b2f1803`，包含全部341源组与8个union cells；full summary/SQLite/全局seal留本地。独立audit SHA `99e0d3467529f23c85926591799ae624dfa0802c8d0a597c675b8c89968c6606`，summary SHA `9764e2a6f35e4e0a5ce3ab319d15ca4642afda5383289710a5e9dd66e48974c1`，seal SHA `d24932ab0885bd6b902374532770726cf5fcca1f24fbe9ff9b239c7c538a20a3`。`progress_receipt.json`仍是01:06历史快照，不当当前状态。

前一特征阶段已推送`8263c3b`；本阶段running checkpoint `23e0d45af9483ecb07bf6baa26856ea93d3a6680` push实际exit0且remote SHA核同（一次banner timeout后默认配置重试成功，不改SSH命令、不需新钥匙）。最终阶段另作complete snapshot/commit；以现场Git与publication receipt为准。旧五模式memory和12源组相邻候选结果均不重跑。

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

旧15–45分钟CPU粗估已被实测替代：producer192.73秒、fresh全输出重算258.33秒，总启动至完成约7分34秒。CUDA禁用；未触发GPU>20分钟暂停规则。独立复算全部55,352,164行六score，最大误差CV 2.84e-14，其余0；SQLite query/trajectory/identitypair113,719/12,452/38,138逐行重建通过。没有官方新方法指标或训练结果。

同predcat/commonCV的lag1/4各2949/2142 hardquery，raw错46/149，分布29/50源组；CV修26/60、破289/533。固定共同1331query/107源组raw错16/99（both8），说明分母固定后仍有跨时刻困难；reference/history仍改变，不能作时间的因果归因。必要门通过，合并hard/error source176/55，bothlag fold156pass/185不满足，全部保留。unknown不负例，无positive不等于unmatched。不能把55m候选当独立样本；支持门仍HOLD、没有训练就绪承诺。

保护发布baseline、冻结上游、原数据、旧sealed结果与另一会话训练。磁盘当前约50GiB可用（启动前复查），本任务不清理旧结果。完整score/SQLite/逐帧JSONL/full summary仅本地；`sync_fit_lag_support_stage_20261024.py`显式同步轻量脚本/协议/阶段回执，phase running不冒称完成。全输出通过后派生`support_brief.json`绑定完整summary/SQLite/seal/hash，再更新报告、Git。

## 继续做什么

1. 先核本次完成snapshot与Git publication。旧producer/protocol/preflight/verifier/supervisor/identity_spacetime_factorial_design全pinned，继续只读，不改它们来准备新实验。
2. `identity_spacetime_followup_recommendation_20261024.md`另存新source-OOF方案。原mean/pair角色是旧nuisance职责，不能直接当新身份训练/untouched holdout。全部FIT支持已看，后续OOF为探索性；tune/test/留出target继续关闭。adapter本体与同一frozen appearance下geometry增量分开读出。
3. 新`causal_appearance_transport.py`与checker三轮实际exit0，fresh独立26项合成审计CPU session59647 exit0；receipt `causal_appearance_transport_independent_synthetic_20261024.json`，SHA `6f5000ab974130cba931135ebfaedfac0d14669ef1dc746ad6d962fce13b6dc9`。仅合成、未读真实数据或训练。固定单位球面恒角速度外推是普通动态外观控制；受约束低秩正交指数是候选数学原语，未实现history encoder或接tracker。当前已知反例：track-specific正交可消除身份间距、错误past-ID能把正确last特征外推至干扰物。未来调用者必须对重复/无ID提供无效mask，core不能检出唯一但错误ID。
4. `causal_appearance_transport_candidate_20261024.md`记录神经输运候选、碰撞边界与停止门。先冻结并审普通无训练动态外观控制的全FIT协议；不得直接将合成通过当作真实收益。文献已有2015动态外观与2023特征预测，2026 TFLF命中但全文访问受阻，广泛“预测外观”不算新意。
5. 整体研究active。需要长GPU训练时先落实可检验机制/强控制，再按已有授权现场检查GPU/owner；预计GPU>20分钟时可靠后台启动并交接后暂停对话，CPU不自动触发此规则。
