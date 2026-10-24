# 流式 full-candidate scene residual 协议与资源预检

2026-10-24。本文档和同名 JSON/脚本只冻结流式输出接口与资源上界，不启动 FIT 全量 worker，也不产生新的 residual 数值。

## 固定人口与块边界

- 只读 sealed full-collection 的 FIT 403 个 metadata-nonempty video、341 个 source group、13,791 帧；不读取 tune/test，也不使用 GT、类别、posterior、released ID、manual identity 或 OOF 分数。
- lag=1/4 只连接同 video、精确 `frame_id` 间隔的帧对。每个 `(video, lag, query_frame, reference_frame)` 独立成为一个 chunk，完整保留 `Q×M`，不按 MNN、availability、gain 或标签筛候选。
- 预检得到 13,388 个 lag1 pair、12,185 个 lag4 pair。准确候选总量为 lag1 `29,096,237`、lag4 `26,255,927`、合计 `55,352,164`；最大 chunk 是 video 383 lag1 的 `77×77=5,929` 候选。

## 输出与 provenance

每个 chunk 使用 exclusive-create 的 NPZ 和一行 append-only receipt。边级数组包含 raw cosine、static L1、三种 arm 的 availability、anchor count、translation、corrected L1、gain、`uint64 fit_key`、anchor SHA-256 和 `mask`；`rowloc` 用两个 `uint32` 保存 query/reference 行号。端点 exclusion 的完整 SHA-256 只在 `[Q,3,32]`/`[M,3,32]` 表保存一次，避免为每个 edge 复制完整 exclusion tuple。`fit_key` 是 arm 与两个 endpoint digest 的 SHA-256 截断摘要；auditor 仍可用完整 endpoint digest 检查碰撞。anchor SHA 对有序 `uint32` anchor row pairs 做域隔离的 SHA-256；缺失 arm 仍保留 candidate，值为 NaN，不能改成 0 或 identity warp。

receipt 必须绑定输入帧 locator、Q/M、候选数、数组 dtype/shape、chunk/output hash、endpoint-table hash、前一条 receipt hash和 protocol hash。完成所有 chunk 后，fresh auditor 从 pinned raw cache 重算三臂数值和两类 hash，global seal 之后才允许连接合法 FIT identity。

## 资源预检结果

`streaming_full_candidate_scene_residual_preflight_20261024.py` 只读 plan、frame execution、inventory、模块/缓存哈希与既有四帧 pilot receipt；没有打开 NPZ payload，没有创建 output/execution 目录，也没有启动 worker。结果在 `streaming_full_candidate_scene_residual_preflight_20261024.json`：

- candidate similarity-entry 工作量估计为 `155,746,372,958`；三臂按 pilot 标定的原始时间约 `14.18 h`，乘 2 倍规划安全系数约 `28.36 h`。这不是实际运行时长。
- 必要边级数组的未压缩上界是 `250 B/candidate`，约 `12.89 GiB`；加 chunk header、1 GiB 元数据和 2 GiB safety reserve 后约 `15.12 GB` 规划空间。
- 当前现场可用磁盘约 `30.5 GB`，按保留运行后至少 `31 GiB` 的冻结门，disk gate 未通过；runtime 12 h gate 也未通过。因此状态是 `HOLD_METADATA_STREAMING_RESOURCE_PREFLIGHT`，不得直接启动全 FIT。

下一步应先决定是否采用更低存储/更快的计算实现，例如只在通过信息门的固定 source 子集上运行，或实现不改变候选人口的流式/增量 MNN 计算；不能为了通过资源门删掉低分、unknown 或无支持候选。
