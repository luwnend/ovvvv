# Full mean-quality control and identifiable pair experiment — 2026-10-23

**本轮先执行强质量控制；配对科学目标改为 label-conditional wrong-evidence moment。`t−t0` 的可预测性只作为语义后验失配诊断，不能通过改名成为新依赖机制。** 精确配置在同名 JSON，最终 SHA256 为 `fc10e338c0160e389008aa34de13dd384cc0466bdd9df08ba7f937104de00718`。本设计没有查看 full tune 结果，也没有打开 test/class-holdout correctness。

Full cache 的已知采集范围为15,283帧、448视频、681,602 emitted observations。该数目不能替代尚待审计的可监督观测/错误/pair 数。输入仍为 full1203、稀疏 exact original-manual-anchor 时间轴；不迁移已有 validation296 的训练或选择结果。

## 1. 第一项可直接执行的完整实验：强 mean-quality

实现由 root 独立新增 `diagnose_full_mean_quality.py`；本协议不改旧 pilot runner、feature helper、objective 或正式报告。沿用封存14维单观测质量特征、主 emitted-label error 及 `source → eligible official GT trajectory → observation` 层级权重。GT类别、身份、IoU、source/video/track ID都不能进入数值预测器。原始检测置信度严格取 pretracker row，不以 tracker 平滑后的 score 替代。

| 候选族 | 固定配置 |
|---|---|
| B0 | 当前训练子集的层级加权平均错误 |
| Ridge logistic，6个 | λ=`10,1,.1,.01,.001,.0001`；weighted mean BCE+`.5λ‖β‖²`；截距不罚；训练子集内加权标准化 |
| HistGradientBoosting，4个 | `max_leaf_nodes ∈ {7,15}` × `l2_regularization ∈ {10,1}`；100轮、learning rate .05、min_samples_leaf40、max_bins128、early_stopping=False、seed20261023；sample_weight=`n_train*w_hierarchy` |

**5 outer × 3 inner source-group CV。** SHA256固定group顺序并round-robin；outer seed=`full_mean_outer_v1`，inner seed=`full_mean_inner_v1`。每次训练子集重算权重、标准化、B0；任何 outer-validation group 的标签都不能影响其模型选择、拟合或校准。内层拼接全部 OOF 预测后，用层级加权 Brier 选择；距离最小值≤`1e-5`时按 JSON 中确定顺序选择：B0→大λlogistic→小leaf/大l2树。

对选中 base 的**内层 OOF**预测拟合固定 Platt：`sigmoid(a+b*logit(p))`，`a∈[-20,20]`、`b∈[0,10]`，损失为加权BCE+`.5*1e-4*(a²+(b−1)²)`。随后将 base 在完整 outer train 重拟合，预测 outer-valid。不能在同一 inner-validation 标签上评价已用它拟合的校准，再把该值称为无偏结果；整个选择+Platt流程的诚实成绩来自 outer OOF。

始终输出 **B0_mean、B1_fixed_ridge_1、B2_nested_raw、B3_nested_Platt**。B3为预先指定主控制，B2同时报告，不能在 outer/tune 上二选一。最终用全部fit的3个inner folds选base、拟合OOF校准、refit all-fit，然后对tune评价一次。Tune不改网格、λ、特征、Platt或checkpoint。

报告micro与相同层级权重的Brier/BCE、平均预测/观察错误、校准截距/斜率、固定`.0,.1,…,1.0`可靠性bins。1000次paired source-group bootstrap只重采已固定预测，不能声称包含训练不确定性。常数预测的joint slope不可辨识，Platt的identity prior也不会改变这一统计事实。

**解释边界：hard error risk 的预测/校准改善，不代表 `E[W|V]` 向量均值准确，也不能自动批准pair模型或 covariance centering。** Mean候选共198个base inner fits，加固定B1/最终refit；实际监督上限为39,903 fit与4,557 tune anchors。提取缓存后单CPU线程约2–20分钟为预估，需用实际匹配行数/计时更新，不用681,602 emitted替代训练数。

## 2. 为什么共同真标签本身就是强null

对有共同GT类别的合法pair，`W_i,c=p_i,c*1[c≠Y]`。即使端点 `p_i,p_j` 在粗化context `V` 下相互独立、均值完全正确，只要共同 `Y` 未知，中心化后的正乘积仍会出现。若 `m_i,c=E[p_i,c|V]` 不随Y变化，则：

```text
trace Cov(W_i,W_j | V) = Σ_c m_i,c*m_j,c*π_c*(1−π_c)   (>0 可以成立)
```

一般情形的精确分解为：

```text
Cov(W_i,W_j|V)
  = E_Y[Cov(W_i,W_j|V,Y)]
    + Cov_Y(E[W_i|V,Y],E[W_j|V,Y]).
```

第二项不是“mean再校准一点”就能消失的估计错误；它可以在理想条件均值下存在。把它学习出来，只能说明共同标签/语义不确定性可预测。

一个可操作的共同标签posterior参照：在 disjoint mean-role 的合法pairs上，以全1203 CE拟合class-shared loglinear `π(Y|p_i,p_j,q_ref,V)`；六个共享系数对应 endpoint text/image/fused 的对称log-score和reference三分支log-score，固定非负边界/正则见JSON。没有绝对类别参数。由当前端点后验、该π及任何固定direct means `μ_i,μ_j`可精确计算：

```text
G = W_i·W_j
G0 = p_i·p_j − Σ_c π_c*p_i,c*p_j,c
mtilde_i = p_i*(1−π)
t = (W_i−μ_i)·(W_j−μ_j)
t0 = G0 − mtilde_i·μ_j − μ_i·mtilde_j + μ_i·μ_j.
```

**若π是真实 `P(Y|p_i,p_j,V)`，则 `E[t−t0|p_i,p_j,V]=0`。** 因此P5对`t−t0`的非零预测可以完全由π失配、语义校准不足或conditioning粗化导致；它不是 view-coupling 估计量。P5不作为 novelty 或机制重训练通行证。

## 3. 先把分组、边和reference封存

在任何full outcomes进入设计前，按source-group前缀block，组名以 `SHA256(pair_role_v1 + NUL + group)`排序，交替分为 **mean_role / pair_role**。赋值来自sealed metadata；没有合法监督的组也不因结果而重分配。Mean-role拟合质量、wrong-evidence均值和π；pair-role的标签不得参与任何nuisance训练、选择或校准。Pair-role内部再做5×3 source-group CV。这样不需要预先构造一个会把pair-validation标签泄入nuisance拟合的全局OOF target表。

边选择先对**全部 emitted GT-free 预测轨迹**执行，以 `(video_id,local_track_id)`为key。轨迹≥4观测；sample-step lag bins为1–2、3–5、6–10、≥11，每bin按固定hash保留最多4边，即每轨迹≤16边。边hash含两端排序后的observation ID与video/local ID。封存后才按 independently eligible manual-anchor endpoints、同GT identity 得到监督子集。不能用whole-track oracle purity、错误事件或某个类别筛边。

每条边的reference是同一预测轨迹的**全部其他 emitted 观测**，排除两端，不按GT、correctness或holdout过滤；三分支分别foreground normalize后做算术mean。Reference可有identity contamination；轨迹保留≥4只是输入可用性，不能宣称干净。无合法pair时保持quality-only fallback。采样lag/memo50均是稀疏sample steps，不是秒或密集帧间隔。

Pair训练和评价均 `equal source → equal eligible GT trajectory → equal retained edge`；均值拟合的重复endpoint使用 `equal source → trajectory → unique endpoint → context`，避免同一端点在多边中重复支配。权重和标准化随训练子集重算。

## 4. 低成本但明确的conditional-mean与pair函数

Q含端点14维质量的对称mean/absdiff/product、固定quality risk的对称摘要、预测标签agreement、fused posterior overlap/cosine/JS、reference entropy/top1/margin、reference数。A在Q上加两种**原始**appearance cosine及sample/original-index log lag。均值和关系head具有相同的A conditioning；不能只给关系head更多语义/时间信息再把增益归因于covariance。

Direct wrong-evidence mean采用6个共享basis：reference fused/text/image、归一化fused²、归一化sqrt(fused)、uniform1203。模型为：

```text
μ_i = sigmoid(αᵀz_i) * Σ_m softmax(B z_i)_m * Q_m.
```

只有一套共享参数，所有1203坐标进入平方损失，不含current endpoint的逐类`p_i,c/p_j,c`、GT坐标或绝对类别embedding。固定λ1e-3、两次确定初始化、L-BFGS最多300步。可缓存6×6 basis Gram、六个`Q_m·W_i`和`‖W_i‖²`，无需每轮物化edge×1203网络输出。

Mean-role内部3组OOF只诊断这一固定family；报告residual mass及每个L2-normalized basis projection，在固定mass bins、agreement和lag上分别检查。预声明equivalence margin为±.02；至少10个source groups的cell，95%source-bootstrap区间完全进入范围才称该projection通过；小cell为inconclusive，不看结果合并bins。**有限projection通过仍不证明完整条件均值；失败禁止covariance措辞。**

标量pair head采用固定128维Gaussian random Fourier features+截距，即129个活跃参数；Q/A各自同维函数、相同六个ridge λ、相同5×3选择预算。输入只在训练子集加权标准化。λ按inner OOF hierarchical MSE选，`1e-8`内取大λ。所有方法同样将trace预测clip到[-1,1]，另报告未clip敏感性。这里匹配参数和训练预算，不声称函数类完全等价。

P0=0/training mean，P1=Q→t，P2=A→t，P3=t0，P4=t0+Q→(t−t0)，P5=t0+A→(t−t0)。它们先解释“质量、agreement与共同标签参照能解释多少”，**不是主要新方法候选**。

## 5. 真正可推进的候选：Y条件moment，再对未知Y积分

在同一class-shared mean中加入candidate-y的reference probability/rank两个标量query，mask第y坐标为0并重新归一化basis，得到 `μ_i^y≈E[W_i|A,Y=y]`。已知Y只在合法监督行的训练/诊断query里出现；它不是部署时输入。固定disjoint mean-role后构造：

```text
t_Y = (W_i−μ_i^Y)·(W_j−μ_j^Y)
d(A,y) = E[t_Y|A,Y=y].
```

用同129参数的candidate-query RFF函数拟合d，输入A及`q_ref,y/rank_y`，不输入当前`p_i,y/p_j,y`或绝对class ID。对照为Q及相同y-query。若条件均值准确，这个目标对应within-Y moment，移除了上一节的between-Y covariance项；共享identity/source难度仍可能解释它，因此也不是appearance因果效应。

推理在全部1203个candidate-y上查询并使用固定π：

```text
D_cond = Σ_y π_y*d(A,y)
G_indY = Σ_y π_y*dot(μ_i^y,μ_j^y)
G_cond = Σ_y π_y*(dot(μ_i^y,μ_j^y)+d(A,y)).
```

不需要1203²物化。删y坐标后的Gram可由 `(Gram_mn−Q_m,y*Q_n,y)/((1−Q_m,y)*(1−Q_n,y))`获得；每y仅6个mixture weights与一个mass。RFF全部candidate查询和积分为`O(K*(36+128))`，每批≤256边，K=1203。**不能静默top-k；否则估计量已改变。**

需要一个同等强度的语义参照 **C129_shared_label_CE**：同pair-role、相同5×3 folds与六λ预算，129个class-shared系数，六种candidate log-score×`[1,context20]`共126项，加三种reference rank，共129项；输出π_C。以CE拟合，内层按其隐含 `G_C=p_i·p_j−Σπ_C,c*p_i,c*p_j,c` 对实际G的MSE选择，比较时所有方法必须使用**相同G目标**。低rank factorization和bounded logit batches避免edge×1203×129张量。

实际进入候选的预声明条件：pair-role outer OOF中，A-query在`t_Y`上优于同容量Q-query；可部署`G_cond`在同一G目标上，相对 **G_QA与G_C两者**均降低≥5%层级MSE，paired source-bootstrap差值95%upper<0，且逐个删除source后的平均差仍<0。5%是本次预声明的实际效果尺度选择，不是功效或独立样本证明。若C129匹配/优于它，停止机制重训练，不能改名采用P5语义修正。

## 6. 明确的证伪和一次tune

- **条件Y挑战**已成为主科学目标，不再只用unknown-Y的正残差论证机制；条件mean失败则inconclusive，不能把失败oracle当作成功对照。
- **Disjoint reference**：在原GT-free emitted长度≥6的预声明子集，将非端点reference按hash交错分为各≥2的A/B，两端分别用A/B；在mean-role重拟合对应nuisance后，重复固定Q/A比较。只有共享reference可见的增益不能归因为独立端点间新机制。
- **Shuffle**：训练标签`t−t0`在source×agreement内置换；每cell至少2训练边，报告覆盖；候选与shuffle比较相同evaluation scope。未见evaluation/tune标签不参与shuffle。
- **Appearance/time去除**各作为预先声明的描述性ablation，维持129参数和搜索预算，不能在tune挑最后模型。

所有比较定型后，tune一次完整评价，不从tune选择λ、模型、特征、reference、loss或readout penalty。Tune成功是developmental确认，不是一个新未触碰test；失败拒绝该冻结candidate。独立test和pseudo-novel correctness继续关闭。即使这些预测比较全部通过，generic attention/learned log-pool、真实correlated-fusion对照与正式downstream协议仍未完成，不能声称新颖性或跟踪收益。

## 7. 本轮已完成的精确合成验证与成本

规格先于测试写入。新脚本 `check_full_mean_pair_gate_synthetic.py` 未打开任何dataset；凭据 `full_mean_pair_gate_synthetic_20261023.json` 为 `PASS_EXACT_COMMON_LABEL_AND_REFERENCE_SYNTHETIC_ONLY`：

- 独立端点+共享Y，使用理想粗化mean仍有trace **.1104**；Y条件中心化为数值0。
- 总covariance `.0576 = within-Y 0 + between-Y .0576`。
- 100组精确Y积分验证全部cross terms、端点交换、类别置换；trueπ下`|E[t−t0]|≤1.3e−16`。
- 仅π失配就能产生 **−.0132** apparent excess。
- 共同reference估计误差产生`.04`乘积，分离reference误差为0。

这些是constructed mechanism checks，不能当实际数据支持、功效、性能或新颖性。

建议执行顺序：独立审计full support→已实现的强mean runner→封存GT-free edge/reference索引→mean-role basis sufficient statistics/条件均值OOF→固定scalar诊断→C129与Yconditional完整比较。基本nuisance+RFF估计15–60分钟；完整C129 nested CE和1203精确积分可能再需30–180分钟CPU。先计时一个fit-only candidate，若预估超过3小时，记录resource stop并优化缓存，不改变假设/数据或宣称完成；不预占GPU。全部raw三分支posterior float32负载约9.8GB，不应每fold复制。

本文件是可执行设计与CPU代数凭据。除root独立实现的mean runner外，pair训练实现与独立审查仍是后续工作；本次没有真实pair训练、GPU操作或既有文件改写。
