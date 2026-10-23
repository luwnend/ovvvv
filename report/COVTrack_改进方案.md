# COVTrack 改进方案索引

> 更新：2026-10-23。旧版本中的错误分析和推断已从正式文档删除，不再作为方法结论。

研究目标：以 COVTrack 为正式 baseline，发展有技术深度、可消融和可证伪的 OVMOT 方法；先验证再训练。

| 文档 | 用途 |
|---|---|
| [项目进度](项目进度.md) | 当前阶段、保护约束、GPU 规则与下一步 |
| [论文级改进设计](COVTrack_论文级改进设计.md) | 实现审计、与 COVTrack++/TRACT/OVTR 的边界、依赖与身份污染候选、三轮证伪 |
| [完整 val 读出报告](COVTrack_完整val轨迹读出_20261023.md) | 本轮正式指标、逐类支持、影响分析与复核产物 |
| [自然依赖核验](COVTrack_自然依赖核验_20261023.md) | 20项控制、自然误判、分箱补核、三轮证伪和具体停止范围 |
| [方法调研笔记](OVMOT_方法调研笔记.md) | 已调研方法及代码入口 |

冻结发布对照：COVTrack R50 / `ctao_public.pth` / 完整 TAO validation / A 参数 `0.37/80/50/0.4`。

- 发布 TCC：Base/Novel TETA **39.554/34.205**。
- 发布后验 sumlog：**39.774/35.771**，是普通聚合控制；后续新机制应与它直接比较。
- corrected-mask sumlog：**39.838/36.311**，已补独立官方控制；mask校准不算新方法贡献。
- 固定inverse-density相对自身sumlog的ΔBase/Novel TETA，发布为**−0.392/+0.167**、corrected为**−0.265/+0.037**；两mask均被普通分数加权同时超过Base与Novel，停止该具体读出。
- 自然同错集中是描述性机制信号，不等于可用covariance。可学习共同误判过程与class-blind身份污染模型仍未完成验证；下一步先核验base监督/词表/留出与采集条件，不把候选设计写成已成立贡献。

上游 `repos/COVTrack` 与发布 baseline 保持冻结；新实验只使用独立副本。
另一个会话的训练目录 `results/covtrack_ctao_train20_bs24_20261022` 不停止、不覆盖、不修改。
本轮没有新训练。
