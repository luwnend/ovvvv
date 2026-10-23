# 三份 dependence 审计脚本的独立复核

日期：2026-10-23。工作方式：只读源码和已落盘证据，补充轻量手例及计数重算；未重复全量扫描/评测，未启动训练或修改运行脚本、正式报告。本记录只验收复核范围，**不等于两项主候选的独立官方全评已经完成**；后者须核验各自 receipt、metrics、退出状态及输出哈希。

## 审核版本

| 脚本 | SHA256 |
|---|---|
| `tmp_analysis/build_teta_classification_events.py` | `c390794cf8e3e743158840e399f923dbbbef66bae354f2bcc60b3593604939e6` |
| `tmp_analysis/diagnose_natural_dependence.py` | `c3eb25b4123fdeab49bc407996df0fbc8f69e6c6590f2ce78b910fdbafa50045` |
| `tmp_analysis/score_dependence_readouts.py` | `5e36843ad56e9cf79a8381d19af045d8dd969171ddf967c0a2e378cae2f48952` |

以上 SHA 分别与事件 receipt、natural 输出及 score summary 记录一致。安装 evaluator 位于 `/home/chenhc/.conda/envs/ovtrack_ovmot/lib/python3.8/site-packages/teta/`。

## 事件缓存与20项评分

- `build_teta_classification_events.py:51–58` 原样调用并返回 `super().compute_matches` 的结果，仅记录分类 alpha 对应的匹配；global-assignment 阶段不记录，也不改 Hungarian。
- installed `metrics/teta.py:150–163` 的 non-exhaustive 分类逻辑与 `replay_counts:85–93` 一致：正确匹配增加 GT 类 TP，错误匹配增加 GT 类 FN 和预测类 FP；该模式不额外计未匹配检测的分类 FP。现有296类标签均位于 evaluator 类别集合内。
- installed `datasets/tao.py:329–338,366–391` 的 overlap/ID 筛选和 `metrics/teta.py:179–218` 的匹配只依赖 GT 类别、几何及 IDs。当前 annotation 没有 category merge，默认 `MAX_DETECTIONS=0`；缓存324,639个 global IDs 唯一。结论要求 boxes、IDs、scores、输出顺序及 evaluator 固定，且每轨迹一个类别。
- 事件 receipt 为 `PASS_EXACT_FIXED_GEOMETRY_CLASSIFICATION_EVENT_REPLAY`：383,115条事件、988视频、296类；五个独立官方 readout 的全部 class/video/alpha TP/FN/FP 精确一致。独立核对事件脚本/输出 SHA，以及五份原始 counts 与各 receipt 的 SHA，均一致。
- `score_dependence_readouts.py:23–29` 正确先加总视频 counts，再计算十个分类 alpha 的均值，并用冻结 LocA/AssocA 更新 TETA；`91,113–118` 核对 class/sequence 顺序。六个官方控制均有 bitwise count parity；完整十项 per-class 指标最大误差为 `3.552713678800501e-14`。
- scorer 的 `55–70,109–115` 绑定旧缓存、新 sealed labels、事件、evaluator、六份 counts/metrics 的哈希，补足事件 builder 本身未封存所有参考数组来源的限制。`125–138` 只读已经封存的候选标签；GT 事件和 annotation 只用于评分与支持量，未反馈到预测。
- 独立从四份输出 counts（released/corrected × full_sumlog/dep_inverse_density）重算 ClsA，与 per-class 输出最大误差均为0；per-class 输出哈希一致。`143–147` 按每个 mask 自身 full_sumlog 计算差值，独立重算四个 base/novel 差值误差均为0：

| mask | ΔBase ClsA | ΔNovel ClsA | ΔBase TETA | ΔNovel TETA |
|---|---:|---:|---:|---:|
| released | -1.176959071 | +0.500750727 | -0.392319690 | +0.166916909 |
| corrected | -0.794812541 | +0.111721102 | -0.264937514 | +0.037240367 |

未发现实质匹配、计数、指标公式、顺序或 GT 隔离错误。上述源码证明加独立控制 parity，足以支持此固定条件下20项高效**诊断评分**；不能称为20项独立全 evaluator 运行，也不验证改变检测/关联或逐帧变化标签的方案。

## 自然误差诊断及实质限制

- `diagnose_natural_dependence.py:104–117` 正确以各端点自身 GT 定义错误；same-wrong 要求两端都错且预测同类。四个 identity/class scopes 互斥穷尽，lag/score-rank/class-pair 分层与协议相符；`310–354` 的每端类别半权重、paired near/far、equal-track 和 equal-supported-class 汇总未发现错误。分数 midrank 使用全部 emitted observations，未先按 GT 删除输入再计算 rank。
- **tie 退化边界**（95–101、608–614）：flat cosine 仍按稳定 pair 顺序强制划分 near/far。独立手例中4个相同 feature、预测 `[7,7,3,3]`、GT全3，far/near cosine均1，却得到 same-wrong 差 -0.5。已直接同步执行者。随后独立读取本次完整输出：24个 scope×variant×feature 条件的 flat strata 和 selected-boundary tie strata **全部为0**，因此此退化不影响本轮结果；仍是未来数据上的实现边界。
- **mixed composition**（587–592）：分层固定的是同/异身份 scope 和 GT 类别对，没有固定实际 GT identity pair；mixed scopes 仍有身份难度组成混杂。主 single-matched-identity scope 避开此项混杂，但“已匹配观测只有一个 GT identity”不证明整条含未匹配框的轨迹完全纯净。
- **统计量含义**（26–31、104–117、652–656）：结果描述同错率、错误基率和条件同错误类别集中度；没有估计中心化的 conditional residual covariance，不能直接当作可学习 Sigma 的证据。共享 backbone/分类路径、GT 匹配选择和粗分箱后的剩余混杂仍存在；相似度关联不证明因果依赖或候选有效性。

主要证据入口：`results/dependence_audit_20261023/teta_events/receipt.json`、`scores/score_summary.json`、`scores/per_class_metrics.npz`、`natural_error_dependence.json`。本记录不追加训练、论文采纳或方法新颖性结论。
