# OVTrack 收尾总结

> 结论冻结日期：2026-10-22
> 代码状态：`work/ovtrack_base` 冻结，不再修改；后续改进转到 `work/covtrack_base`
> 协议：TAO OV-MOT，本地 `val200` 子集（245 视频 / 8,911 帧 / 41,858 框 / 166 个有标注类，其中 base 131、novel 35）

## 1. 这一阶段做完了什么

不是停留在“跑通 demo”，而是完成了三件事：

1. **官方基线可复现**：OVTrack 在本机 GPU 上跑完 `val200`，产物为 `results/base_val200/`。
2. **离线回放器与官方对齐**：自建 replay 复刻检测、embedding、贪心匹配与 TCC 后处理；`LocA/AssocA` 与 GPU 结果在四舍五入后一致，差异只来自 fp16 分类分数。
3. **改进方向做过同口径对照**：有稳定正收益的保留，有局部收益但全量掉点的明确证伪，避免后续重复踩坑。

## 2. 基线与评测口径

### 2.1 官方 OVTrack（`val200`，GPU）

| 组别 | TETA | LocA | AssocA | ClsA |
|---|---:|---:|---:|---:|
| COMBINED | 34.534 | 51.286 | 35.729 | 16.587 |
| Base | 36.068 | 51.313 | 36.265 | 20.625 |
| Novel | 28.793 | 51.181 | 33.722 | 1.477 |

产物：`results/base_val200/tao_track.json`、`results/base_val200/OVTrack/teta_summary_results.pth`。

### 2.2 离线回放对齐

| 路径 | LocA | AssocA | 说明 |
|---|---:|---:|---|
| 官方 GPU | 51.286 | 35.729 | fp16 推理 |
| replay + TCC | 51.245 | 35.725 | 与 GPU 差异 <0.05，工具可信 |

对齐的两个必要步骤，后续任何数据集都不能省：

- **per-video track id offset**：不同视频的 track id 会冲突，必须先做全局偏移；
- **TCC majority vote**：官方评测默认按轨迹做类别多数投票。

早期漏掉这两步时，base TETA 只有 34.166，会被误判为“实现有 bug”。

## 3. 已验证有效的改进

### 3.1 分数尺度校准（bi-softmax only）

OVTrack 原始关联分数是：

```text
(bi-softmax + cosine) / 2
```

cosine 与 1203-way softmax 的尺度不同，`match_score_thr=0.5` 实际是在两个不可比的分布之间取平均。改为**纯 bi-softmax** 后，阈值可以按同一尺度重扫。

`val200f` 全量结果：

| 配置 | Base TETA | Base LocA | Base AssocA | Base ClsA | Novel TETA | Novel LocA | Novel AssocA | Novel ClsA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 官方 baseline | 36.07 | 51.31 | 36.27 | 20.63 | 28.79 | 51.18 | 33.72 | 1.48 |
| bis t=0.15 | 36.54 | 52.97 | 36.14 | 20.52 | 28.42 | 52.54 | 31.64 | 1.06 |
| bis t=0.20 | 36.81 | 52.65 | 37.34 | 20.44 | 29.03 | 52.06 | 33.88 | 1.16 |
| **bis t=0.25** | **36.85** | 52.47 | **37.41** | **20.67** | **29.27** | 51.75 | **34.74** | 1.34 |
| bis t=0.30 | 36.78 | 52.16 | 37.65 | 20.55 | 29.16 | 51.41 | 34.54 | 1.53 |

**保留配置：`score_mode='bis', match_score_thr=0.25`。**

相对 baseline：Base TETA +0.78、Base AssocA +1.15；Novel TETA +0.48、Novel AssocA +1.02。`0.30` 的 AssocA 略高但 TETA 回落，不作为默认；`0.35` 未收，不影响“0.25 最稳”的结论。

产物：`results/sweep_val200_bis.jsonl`、`results/v200_bis0.25.json`、`results/eval_v200_bis0.25/`。

### 3.2 轨迹级标签证据聚合（高潜力，尚未全量定论）

官方 TCC 是“逐帧 argmax 后做多数投票”，会丢掉概率幅度。用完整 1203 类后验做轨迹级聚合，在 `val60` 上得到：

| 标签规则 | Base TETA | Base ClsA | Novel TETA | Novel ClsA |
|---|---:|---:|---:|---:|
| majority（官方 TCC） | 31.49 | 17.38 | 23.58 | 0.10 |
| sumprob | **32.77** | **21.21** | 23.83 | 0.86 |
| sumlog | 30.73 | 15.11 | **28.70** | **15.46** |
| ema | 31.23 | 16.61 | 23.83 | 0.86 |

解释：

- `sumprob` 保留概率质量，base 分类明显更好；
- `sumlog` 等价于“轨迹内证据相乘”，对 novel 的长尾类别极强，但会压掉 base；
- 因此合理方向不是二选一，而是**按轨迹可靠性做 base/novel 选择性路由**。

`val200` 上已用 p1-only 做过轻量对照：majority base ClsA 21.12、confidence-weighted 21.37，提升很小；`results/tl_*` 下的 full-posterior 结果目前只覆盖 **65 个类**，还不是完整 166 类口径。因此这条线只能写成“有强信号、待全量确认”，不能写成已成立的主贡献。

产物：`results/tl_majority/`、`results/tl_sumprob/`、`results/tl_sumlog/`、`results/tl_ema/`、`ovmot_improve/track_label.py`。

## 4. 已证伪的方向（不要重复）

### 4.1 全局运动融合（R1）

- `val3` 上 AssocA 46.9→51.9，看似 +5；
- `val200` 上 COMBINED TETA 33.608 vs baseline 34.166，AssocA 34.305 vs 35.724，**负收益**；
- 结论：小样本过拟合。运动线索本身有信息（见诊断），但“直接线性融合进匹配分数”不成立。

### 4.2 分数归一化（R4）

- 第一版 sigmoid 把 cosine 压到均值 0.09，所有候选低于阈值，AssocA 崩到 21.9；
- 第二版 row-wise min-max 仍 −17；
- 第三轮确认是方向错误，不是调参问题，弃用。

### 4.3 运动门控 / 匈牙利全局分配

- `val60` 上 motion gate 只有约 +0.3 base TETA；
- Hungarian 不如 greedy；
- 结论：只能作为分析工具，不作为主创新。

### 4.4 `rams3.py` 的已知实现 bug

`score_mode` 参数没有真正生效，始终使用 bi-softmax。任何引用 `r3_ovk_*` 的结果都不可信；若复用必须先修。

## 5. 诊断结论（决定 COVTrack 方向的依据）

### 5.1 线索有信息，但融合方式错了

在 `val200` 的 25,806 个 GT 对上：

| 线索 | Top-1 | AUC |
|---|---:|---:|
| appearance | 79.81% | 0.9467 |
| motion | 68.87% | 0.8475 |
| product | 81.65% | 0.9506 |
| linear fusion w=0.5 | 82.24% | 0.9540 |
| ambiguity-gated | 82.66% | — |

运动单独看有 68.9% Top-1，融合后还能到 82.7%；端到端却掉点，说明问题在**接受阈值、分数尺度、分配顺序**，不是运动没用。

### 5.2 主要误差是碎片化，不是身份交换

- 预测轨迹中有 GT 的 4,471 条；
- purity 均值 0.960，89.4% 的轨迹纯度 ≥0.9；
- 每个 GT id 平均对应 3.10 条预测轨迹，中位数 2。

这意味着：轨迹内部很少“串人”，更多是“断成多段”。后续应优先做轨迹复活/拼接，而不是继续加复杂外观模型。

### 5.3 novel 分类是排序问题，不是完全不可分

- `val60` 中 579 个 novel GT 被检测到，但几乎全部被分到 base 类；
- novel GT 的 top-1 只有 3.8%，top-10 52.8%，top-50 85.9%，top-100 91.7%；
- 正确类别已经在候选里，问题在 base 先验过强、逐帧证据太弱。

典型混淆：`army_tank → truck/car/train/boat`、`barbell → skateboard`、`lawn_mower → motorcycle`、`bow → garden_hose/kite`、`hockey_stick → ski_pole`。

## 6. 可复用资产

### 6.1 代码

| 文件 | 用途 |
|---|---|
| `ovmot_improve/replay.py` | 从缓存复刻 OVTracker，秒级迭代关联策略 |
| `ovmot_improve/eval_teta.py` | replay JSON → TAO 格式 → 官方 TETA；默认 TCC |
| `ovmot_improve/track_label.py` | 轨迹级标签聚合（majority/sumprob/sumlog/ema/select） |
| `ovmot_improve/diag_pairs.py` | 外观/运动/融合的 pair 级可分性 |
| `ovmot_improve/diag_purity.py` | 轨迹纯度与碎片化统计 |
| `ovmot_improve/diag_trackcls.py` | 逐帧 vs 轨迹级分类准确率 |

### 6.2 缓存与数据

- `work/cache_val200f`：8,911 帧，含 `cls_full` 50×1204 后验，2.3G；
- `work/cache_val60`：2,305 帧，含 full posterior，608M；
- `work/ovtrack_base/data/tao/annotations/ovmot_subset/val60.json`：63 视频 / 2,305 帧；
- `work/ovtrack_base/data/tao/annotations/ovmot_subset/val200.json`：245 视频 / 8,911 帧；
- C-TAO：`work/ovtrack_base/data/tao/ctao_dataset/ctao_base.json`，500 视频 / 490,210 帧 / 1,521,559 框，仅 base 类。

### 6.3 关键命令

```bash
cd /home/chenhc/ov/OVMOT_RESEARCH/ovmot_improve

# bi-softmax 回放
/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python replay.py \
  --cache ../work/cache_val200f --tracker base --score_mode bis \
  --match_score_thr 0.25 --out ../results/v200_bis0.25.json

# 回放结果转 TETA（含 per-video offset + TCC）
/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python eval_teta.py \
  --replay ../results/v200_bis0.25.json \
  --ann ../work/ovtrack_base/data/tao/annotations/ovmot_subset/val200.json \
  --out ../results/eval_v200_bis0.25
```

## 7. 边界与后续

- 本阶段所有结论只对 **本地 val200 子集 + 官方 OVTrack R50 checkpoint** 成立，不能直接外推到完整 TAO val/test。
- `val200` 只覆盖 166 个有标注类（base 131 / novel 35），novel 类数量少于官方全量协议，数字不可与论文表直接横比。
- OVTrack 代码与实验到此冻结；需要新实验时只复用 `replay/eval_teta/track_label` 三个工具，不再改 `work/ovtrack_base`。
- 下一阶段在 COVTrack 上验证同样两条假设：分数尺度校准、轨迹级证据聚合；见 `report/COVTrack_改进方案.md`。
