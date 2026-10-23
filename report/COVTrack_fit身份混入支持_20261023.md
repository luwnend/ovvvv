# Fit身份混入支持诊断（2026-10-23）

结论：固定预测轨迹中存在多个已知身份的端点支持，分散在多个源组；尚不能证明身份净化改善分类，也不足以启动router训练。全部真实连接与独立全量审计实际exit0。

## 固定范围与结果

复用已封存的GT-free边、源组角色与全emitted参考补集，不按GT重选边或净化实际reference。仅连接fit中21,661条独立合法、精确人工anchor身份；两端身份均已知才判断same/different。未匹配、无合法身份端点保持unknown，不当负例。不读取分类正确性值、tune身份或NPZ。

| 固定角色 | 全部保留fit边 | 同GT身份 | 不同GT身份 | 单端unknown | 双端unknown | 有不同GT边的源组 |
|---|---:|---:|---:|---:|---:|---:|
| mean | 297,409 | 6,516 | 428 | 4,984 | 285,481 | 65 |
| pair | 262,962 | 5,642 | 391 | 4,404 | 252,525 | 70 |
| 全fit | 560,371 | 12,158 | 819 | 9,388 | 538,006 | 135 |

源组角色固定为172 mean/169 pair，不能因缺少合法监督重分配。只有12,977条边两端身份已知，不能把819/560,371解释成总体污染率，也不能将稀疏已知子集比例外推为全轨迹污染率。

不同GT边涉及280条预测轨迹、135个源组；232条轨迹和134个源组同时支持same/different。最大单源组28/819=3.42%，删最大源组剩791条。源组集中度的有效数量71.12仅为集中度描述，不是独立样本量。重复端点/边也不是独立重复。

## 执行与独立核验

Producer工具session21762实际exit0，33.543秒；独立verifier工具session70485实际exit0，80.099秒。独立审计未导入producer，逐行核验全部21,661条fit身份来源及560,371条fit边、unknown状态和reference哈希，重算全部global/role/source/24个固定cell/source-cell及集中度。2,226条tune监督行仅解码split后跳过；63,507条GT-free tune边跳过。混合监督文件字节确实被扫描和hash，不能称“没有读取文件”；分类正确性与tune身份值未解码。

权威输出目录：`results/base_error_full_collection_20261023/base_fit_identity_edge_support_20261023/`。独立audit状态`PASS_INDEPENDENT_FIT_IDENTITY_EDGE_SUPPORT_OUTPUT_AUDIT`，SHA `8e0cafbb0dea4c9120119e62ff0bb9d5a5a3bd06855d5140117aa2481372beaf`。执行链见`fit_identity_execution_receipt.json`及`independent_fit_identity_edge_support_audit_execution_receipt.json`。7项保护文件再次核验未变。

原summary为22.96MB、逐边JSONL仅本地保留；可同步的`identity_support_brief.json`逐字段复制all/per-role/固定24cells并绑定完整summary/rows/audit哈希，不修改封存producer输出。

## 下一步及解释边界

先做独立fit-only的observed-label oracle分类对照：在相同端点和相同covered分母上，比较完整原参考、全部manual-known参考、同GT参考及预定的数量匹配控制。保持完整词表、统一固定pooling规则，分别报告实际emitted label与posterior argmax。按源组→GT轨迹→唯一端点等权，重复context分配端点权重。unknown不视为异身份；GT快照仅诊断，不进入实际readout/cache。

该稀疏标签oracle不是所有身份感知方法的数学上界。即使出现收益，也只说明特定fit覆盖子集、固定读出规则下有诊断空间；仍需class-blind可辨识性、同类异身份控制及一级文献边界，不能据此认定新颖性、部署收益或训练就绪。当前没有新方法TETA结果，没有新detector/tracker/pair训练。
