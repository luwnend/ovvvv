# EvidenceOwnershipLedger v4 独立审计（833a4bf6）

日期：2026-10-25  
冻结副本：`/raid/datasets/chc_data/partwhole_review_20261025_v1/evidence_ownership_head_833a4bf.py`  
冻结 SHA-256：`833a4bf6dcd7e6af2763b602f975ee9a0e7fea09e68dde9a5b7904f84305af6b`（20,835 bytes）  
对照主文件：`tmp_analysis/evidence_ownership_head.py`；本报告结束前 read-only hash 同为 `833a4bf6…`。  
环境：PyTorch `1.10.0+cu113`，CPU float32。未启动 CUDA、训练、遮挡采集，未修改主文件。

## 判定

**FAIL，唯一阻断项是实现问题：单行 batch 在 `train()` 模式下经过 `child_mask` 的 `GroupNorm(8, hidden)` 时抛出**

```text
ValueError: Expected more than 1 value per channel when training,
got input size [1, 8, 1, 1]
```

这是很现实的调用形态：一个 parent–child edge、一个事件批或一张 ROI 就是单行。评估模式不会触发，但训练一开始就会失败。三处 `GroupNorm` 都不能靠“调用方总拼大 batch”来规避，因为 `max_edges=64` 使合法小批仍可出现单行输入；也没有给 `N=1` 的显式 fallback 或禁止 train-mode 的接口契约。修成 `GroupNorm → InstanceNorm(affine=True)`、`LayerNorm`，或显式 `if x.shape[0]==1` 的不会破坏空间统计的 fallback，再重审。

因此：**不允许训练、接回 COVTrack、或把“训练可运行”作为通过结论。**  
**允许**进入不依赖训练模式指标的只读 GPU 遮挡特征采集，前提是用镜像固定 `833a4bf6` 的副本、`eval()`、新输出目录和完整 receipt；采集只验证 teacher/因果门槛，不能替代本 FAIL。采集若需要可训练 `child_mask` 的反向梯度，则当前版本仍**不允许**。

此前 v3 快照 `68fa8839…` 的旧结论只属于该快照，不能转给本版本。其记录中的空 geometry、时间类型、类宽和 owner 上限问题在本版本已有真实修复。

## 审计文件

- 探针：`tmp_analysis/owner_ledger_v4_audit_20261025.py`，SHA-256 `d00d6929df0e7a7b5a421ef18066589456a8e51694f89c156f8ebcc61d959448`
- 证据：`tmp_analysis/owner_ledger_v4_audit_evidence_20261025.json`，SHA-256 `6bc039495d40a817b7b42124c8c393a4acce0bbb491f346c127f7e0aae7f7335`
- 审计模式：CPU 合成输入；读取冻结副本和主文件哈希，执行只读导入。探针启动时校验冻结源 SHA，结束时再校验当前 SHA 没有漂移。

## 实际通过的项目

三轮共 16 项；15 项 PASS，1 项 FAIL。第二轮和第三轮都包含独立反例，不是只跑作者自带测试。

| 检查 | 结果 | 证据 |
|---|---|---|
| `python -O` 下非法空 overlap shape 仍拒绝 | PASS | 三种 `[0,2,3]`、`[0,3,2]`、`[1,2,2]` 全部 `ValueError: overlap must be [E,H,W]` |
| `class_width` 精确等于 posterior 列数 | PASS | 空 edge 和非空 edge 的增宽配置均拒绝；1203 类非空实际前向通过 |
| 时间向量形状、有限性和未来信息 | PASS | 非有限 child time 拒绝；`child_time > parent_time` 拒绝；同帧/过去接受 |
| 时间 dtype 一致性 | PASS | 仅一致 `int32`或`int64`接受；int32/int64 混合、int/float 混合、bool、float 拒绝 |
| per-parent `max_children_per_parent` | PASS | 两个 parent 各一 child 接受；同一 parent 两 child 拒绝；错误信息匹配 |
| 全局 `max_edges` | PASS | 分布在两个 parent 的 2 edge 在 `max_edges=1` 时拒绝；与 per-parent 计数独立 |
| 父正证据 / 全零身份 | PASS | 全零 parent 精确零；随机 20 组有限且非正 |
| 空 edge 精确恒等 | PASS | 返回 `[1,3]` 全零 |
| owner 局部最大 | PASS | 非连续 `[0,1,0]` edge 序列和不同分数下，每个 parent 由局部最高分 owner 拥有 |
| 严格 tie | PASS | 同分时 lower edge 稳定拥有；交换 child payload 后随 edge 反转 |
| 完全重叠不双扣 | PASS | 两个相同 child 与单独一个 child 的 correction 在 float32 容差内一致 |
| 部分重叠两 child 不同区域 | PASS | top-row A 与 bottom-row+shared-top B；单独 A/B 与合并 owner 的逐类 correction 精确和为合并结果，且各自非零、合并结果非正；共享格只扣一次 |
| 类别 relation 输出 | PASS | 4 个空间格 × 3 类，末端线性输出 `[4,3]`，forward 中 reshape 为 `[2,2,3]`；不是每格单一标量 |
| 打乱类文本 | PASS | shuffled embedding 改变 correction，差值 `5.73e-3` |
| 单个父类 embedding 扰动 | PASS | 控制 relation 下，目标类变化明显，其他类变化 `<1e-8` |
| child posterior detach | PASS | posterior 传入 hook 改变 `child_query` 输出，但 backward 后 `grad is None` |
| 梯度路径 | PASS | child_query、relation、child_mask 的末层梯度均有限且非零 |
| 随机初始化的类差异 | PASS（弱条件） | 数值差异非零，但 `mean_class_spread` 只有约 `6e-5`，不能拿初始化当“已学会类关系” |
| 1203 类非空前向 | PASS | shape `[1,1203]`，有限且非正 |
| 单行 `train()` | **FAIL** | `GroupNorm` 的 batch 内归一化要求导致 `ValueError`，见上 |

上述通过项足以说明：当前的输入契约和类别条件实现不再是 v2 的“共享 spatial scalar”，也不应因旧版失敗继续阻挠只读遮挡采集。

## 方法含义和口径

本头仍未训练，仍是合成契约原型。`relation` 现在是“父 ROI token × 类文本 / 子 query / posterior”组合后的逐类 logit，并非已证明学出了 `part-of`；`child_mask` 也只做归因前的空间分数。论文里不能把这些测试写成因果修复或 TETA 收益。

即使修好 `GroupNorm`，进入真实数据前仍需要固定的 source→video→GT trajectory 划分、真实 child/错 child/面积匹配 mask 的对照，并超过零参数 containment、class-agnostic gate、learned class table、随机 relation 等强控制。只有真实遮挡修复/破坏明显为正，且完整官方 TETA 的 ClsA 与类别宏平均同向，才可写成有效模块。当前不允许用 GT 选择 owner、阈值、关系或 memory。

## 最终结论

- **快照 `833a4bf6…`：FAIL，唯一阻断项为 `train()` 单行 `GroupNorm`。**
- 其余指定契约均通过；两轮中发现的测试脚本错误已改正后重跑，不计入模块缺陷。
- 修这个失败后需新的 SHA 和重新三轮审计。
- 当前放行**只读 eval 模式遮挡特征采集**，禁止训练、权重更新、修改类别输出或接回 COVTrack。
