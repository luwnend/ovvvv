# 历史条件外观输运：近邻原文查重

2026-10-24。只读文献审计；范围固定为下列 **5 篇**。比较对象是 `causal_appearance_transport_candidate_20261024.md`，读取 SHA256 为 `74a7f07e258805908f2161f890b49ecc566fbd0518e8e01f27144829406a4bb7`。不修改候选、已封印协议/缓存，不运行真实数据或训练。引用来源、读取边界及公式定位见同目录 `appearance_transport_literature_sources_20261024.json`。

**结论：不能批准新颖性主张。** 历史外观预测、predict-then-align、连续时间 ReID、历史条件的旋转/矩阵特征预测分别已有直接先例；把这些与低秩矩阵指数组合，不能据此认定新颖。最接近的 2026 TFLF/SFP 全文公式未取得，精确碰撞仍是 `NEEDS_FORMULA_SFP_FULLTEXT`。这些判断不否定执行已独立冻结的普通角速度控制，也不授权训练或替换方法。

| 文献与实际读取范围 | 已占据的机制 | 对候选的有范围判定 |
|---|---|---|
| Li / Chen / Bi，**Multiple object tracking with appearance feature prediction and similarity fusion**，IEEE Access 2023，[DOI:3279868](https://doi.org/10.1109/ACCESS.2023.3279868)。完整作者接受稿，III-D / III-F；非最终排版 VOR。 | 最近五帧外观经位置编码和 Transformer 预测当前外观，之后用于关联；Eqs.2–8 是预测器，Eqs.9–12 是位置/全局及局部外观相似度融合。 | `STOP_NOVELTY_HISTORY_APPEARANCE_PREDICTION`。所读预测器没有候选的低秩 skew 指数或跨历史更新路径约束；这只限定公式差异，不等于证明新颖。 |
| He / Han / Li / Li，**Focusing on temporal feature variations: Improving the adaptability of Re-identification in multi-object tracking**，Neurocomputing 685:133637，2026，[DOI:133637](https://doi.org/10.1016/j.neucom.2026.133637)。出版方索引预览的摘要/引言/章节片段；全文 403。 | TFLF 的 GRU 时间记忆与 SFP：从历史动态预测当前表征，并与当前外观对齐，控制特征漂移；另有时间一致采样/增强。 | `STOP_NOVELTY_PREDICT_THEN_ALIGN_OR_FEATURE_DRIFT_CONTROL`；精确 SFP 条件、stride、损失、因果输入及约束均 `NEEDS_FORMULA_SFP_FULLTEXT`，不能从摘要排除候选碰撞。 |
| Zhang / Huang / Duan，**Video person reidentification based on neural ordinary differential equations and graph convolution network**，EURASIP JASP 2021:58，[官方全文](https://link.springer.com/article/10.1186/s13634-021-00747-1)，Sec.2.2–2.3。 | Eqs.4–6 用 ODE 演化潜变量并解码特征；Eq.7 的中心时间图包含前后帧。 | `STOP_NOVELTY_CONTINUOUS_TIME_LATENT_REID`。这是视频 ReID，所读模型含未来邻居，不能当成已验证的 past-only 在线 MOT；未见候选式硬正交 generator。 |
| Fiquet / Simoncelli，**A polar prediction model for learning to represent visual transformations**，NeurIPS 2023，[官方记录](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f8e55d98b0c2569bd0aa25b076e6b3f8-Abstract-Conference.html)、[作者 arXiv v2 全文](https://arxiv.org/html/2303.03432v2)，Sec.3.1 / 3.3。 | Eq.3 从过去两帧系数的相位增量预测下一帧，Eq.5 明确为二维旋转；Eq.8 从两帧历史学习组内预测矩阵。 | `STOP_NOVELTY_HISTORY_CONDITIONED_ROTATIONAL_FEATURE_PREDICTION`。这是本次最直接的算子家族碰撞；研究对象是视觉变换/视频预测，尚非身份关联。 |
| Frion et al.，**Neural Koopman prior for data assimilation**，[arXiv:2309.05317v3](https://arxiv.org/html/2309.05317v3)，2024；Sec.II-B / III-A / III-B 原文。 | Eq.21 是全局 Koopman 矩阵的软正交惩罚；Eqs.22–24 用矩阵指数处理任意时间间隔。文中也讨论 skew 生成元与特殊正交动力学。 | `STOP_NOVELTY_ORTHOGONAL_CONTINUOUS_LATENT_FLOW`。其实际算子主要全局共享、软约束，不能写成已实现候选的每历史硬 skew 算子；但连续指数输运和正交稳定先验已非空白。 |

## 公式层面的差异与边界

候选是冻结 raw 特征上的 `T_m(Δ)=I+U[exp(ΔK_m)-I]U^T`，其中共享 `U^TU=I`，`K_m` 由过去观测产生且为 skew。这与 AFPSF 的 Transformer、Polar 的系数组旋转、Koopman 的全局软正交矩阵并非逐式相同。尤其 Polar 的 PP 在系数对内旋转，但分析/合成映射不因此成为候选的 raw-space 正交算子；QP 的历史条件矩阵也不保证正交。**有这些具体差异，不足以从本次五篇中推出首创。** 本次没有核实与五篇对应的完整官方训练实现，也没有据搜索未命中声称代码不存在。

可保留的待证对象应具体落在任务与可证伪约束上：跨历史更新后独立产生的算子是否保持真实输运一致性，并在未见 source 的同类干扰中，比普通角速度、等预算自由预测和 pair head 更好地保留身份区分。固定 generator 的半群/逆关系是结构恒等式；正交保范数同样不能替代身份效益。五篇的所读部分没有提供候选在稀疏合法 MOT 身份支持下的这一完整实证对象，但这只是**未核实到同一完整对象**，不是新颖性通行证。SFP 全文缺口仍阻止排除最近 MOT 近邻。

## 共享 nuisance 算子的收窄讨论

以下仅记录数学边界，不改候选或协议。每个 track 使用自己的正交算子，不能保证不同实例间距；若同一 video/time 对同一批 memory 使用完全相同的 past-only 正交矩阵 `Q`，则 `(Qx_i)^T(Qx_j)=x_i^Tx_j`，整批 Gram 精确保留。但这条恒等式不证明 `Q` 能预测当前 query，不能证明变化是相机/光照造成，也不证明它改善身份关联。

memory 若来自不同 reference time，分别施加 `Q(t_q,t_i)` 与 `Q(t_q,t_j)` 后就不再满足“同一矩阵”的前提。需要明确共同参考、per-time 共享 flow 及估计时序；把不同时间的特征对齐本身仍需证据。共享 nuisance 可作为可辨识性假说：在保持实例残差的同时估计共同变化，且估计器不能访问当前 query appearance。对象特有的非线性视角变化未必可共享。Polar 的共享分析基/变化表示、Koopman 的共享动力学已是相关近邻；本次未对更窄的共享 nuisance 专题另做文献排除，因此不能将这个收窄称为新方法或补救性结论。

## 来源完整性与停止范围

AFPSF 接受稿的标题、作者及页脚 DOI 与 Crossref 匹配；稿件头部保留占位 DOI，因此只称作者稿，不伪称最终 IEEE 排版。TFLF/SFP 的 DOI、卷号和文章号已核实，出版方扩展预览不是完整方法。2021 ReID、Polar、Koopman 均读到上述原文方法公式。Koopman 文中关于所有特殊正交多频系统严格周期性的表述不在本审计中采纳；此处只依赖正交/指数公式，不复述该普遍周期性结论。

停止的是上述宽泛新颖性说法。当前总门为 `HOLD_NOVELTY_NOT_ESTABLISHED`，另有 `NEEDS_FORMULA_SFP_FULLTEXT`；没有任何真实收益或训练就绪结论。优先补齐 SFP 精确方法，再判断最窄约束是否还提供额外可检验对象；普通角速度控制按其独立协议继续，不能用本报告给失败结果更换人口、lag 或解释。
