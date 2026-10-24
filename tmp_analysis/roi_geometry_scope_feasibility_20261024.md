# ROI 空间证据的几何可行性审计（2026-10-24）

**结论：HOLD。** 现有四帧可以精确恢复 RoIAlign 7×7 每个 bin 的图像连续坐标和裁剪归一化坐标；
但末层四层 3×3 卷积后的 token 是重叠感受野聚合，不能当作独立像素 patch 或单一位置。
现有观测因此足够构造几何作用域与 mask 原型，不足以训练跨帧局部证据模块：只有 15 个合法
manual-anchor 行、4 个同身份跨帧对。GT 身份只可用于评价，不能选择样本或部署记忆。

## 1. 口径与公式

保留的 `det_bboxes` 是原图像素 xyxy。送入 track head 前乘 preprocessing `scale_factor`，
得到模型输入坐标 `(x1',y1',x2',y2')`。FPN level 由 mmdet `SingleRoIExtractor` 决定：

```text
s = sqrt((x2'-x1')*(y2'-y1'));  level = clamp(floor(log2(s/56 + 1e-6)), 0, 3)
stride = 2^(2+level)
```

mmcv `RoIAlign(aligned=True)` 的 bin 中心在模型输入坐标中为

```text
cx_model(i,j) = x1' + (j+0.5)*(x2'-x1')/7 - 0.5*stride
cy_model(i,j) = y1' + (i+0.5)*(y2'-y1')/7 - 0.5*stride
```

回到原图坐标时只需除以 `scale_factor`。由于 dense sampling 在一个 bin 内均匀取点，
其平均值恰在该 bin 中心；若关心整个 bin 的覆盖范围，则用 `j/7` 到 `(j+1)/7` 的区间。
稳定裁剪归一化坐标为 `(j+0.5)/7,(i+0.5)/7`；严格考虑了 aligned 位移后应减去
`0.5*stride/(scale_factor*box_width)`（x）或对应高度项（y）。

四层 3×3、padding=1 的卷积之后，一个输出 token 的 nominal 支持是 9×9 个 pooled 单元。
裁剪到 7×7 后中心 token 看全部 49 个单元，边角 token 的支持因卷积 padding 被截断且不对称。
因此可报告“该 token 的网格锚点/近似中心”，不能声称它对应独立可见区域；跨帧局部比较必须
使用可学习的感受野作用域或显式 warp/visibility，而不能把 token 当 49 个物理 patch。

## 2. 合成核对

在解析场 `f(x,y)=3x+5y+7` 上，线性函数的 bin 均值等于 bin 中心值。
用真实 `mmcv.ops.roi_align` CPU 算子核对三组 box/stride，最大绝对误差如下：

| case | box (original) | stride | max abs error |
|---:|---|---:|---:|
| 0 | `(8.0, 10.0, 72.0, 58.0)` | 1 | 6.539e-05 |
| 1 | `(16.0, 20.0, 88.0, 76.0)` | 2 | 1.417e-05 |
| 2 | `(32.0, 24.0, 152.0, 120.0)` | 4 | 4.360e-06 |

四层卷积的合成支持掩码检查：

| token | 非零 pooled 单元数 | 占 49 |
|---|---:|---:|
| center | 49 | 1.000 |
| corner | 25 | 0.510 |
| edge | 35 | 0.714 |

## 3. 四帧真实 NPZ

| file | rows | source | ori | img/pad | FPN level 0/1/2/3 | box median (orig) |
|---|---:|---|---|---|---:|---:|
| v1271_f0.npz | 50 | ArgoVerse | [1200, 1920] | [800, 1280] / [800, 1280] | 42/4/3/1 | 39.5×69.3 |
| v1271_f1.npz | 54 | ArgoVerse | [1200, 1920] | [800, 1280] / [800, 1280] | 43/9/1/1 | 70.1×62.8 |
| v801_f0.npz | 69 | BDD | [720, 1280] | [750, 1333] / [768, 1344] | 61/5/3/0 | 26.3×29.7 |
| v801_f1.npz | 68 | BDD | [720, 1280] | [750, 1333] / [768, 1344] | 61/4/3/0 | 23.3×32.9 |

所有 241 个保留 ROI 的原图 xyxy 都在图像范围内（`boxes_outside_original=0`）；
但 aligned 的半特征像素偏移仍使 21/11809 个 pooled bin 中心略落到原图外，在模型坐标中会由 RoIAlign 边界 clamp 处理。按缩放后坐标比较，额外 pad 暴露为 21/11809。
四层卷积又在裁剪网格边界使用零 padding，所以 border token 的依赖仍被截断；
这些边界效应必须在 mask/训练中显式记录，不能因为原框合法就把所有 token 当作完整观测。

## 4. 现有观测能支持什么

**可以支持：** 对每个保留行构造 7×7 的原始坐标 field、裁剪归一化坐标、FPN level、
边界 mask 和几何作用域；可用于离线检查空间先验、mask 实现和 warp 公式。

**不能支持：** 从 `conv_spatial` 反推每个 token 的独立图像 patch；从 box 几何恢复遮挡、
flow 或跨帧对应；用当前 4 个同身份对训练或验证局部注意力。最终 token 的局部性应表述为
“ROI 裁剪内的卷积聚合”，而不是“49 个物理局部块”。

## 5. 必须排除的泄漏与资源

- GT identity、GT category、posterior correctness 只能在离线评价读出后打开；不得参与采样、
  memory 选择、阈值/超参数选择或训练批次构造。缺失身份是 UNKNOWN，不能作负例。
- 跨帧身份对必须以 `(source, video, gt trajectory)` 为统计单位；预测 local ID 只能作为
  GT-free 对照，不能直接叫身份。
- 不要把同一行被多个候选复用的 ROI 当成独立样本；先按 source 留出，再按 video 聚合。
- 现有四帧 241 行、15 个合法 anchor、4 个同身份对，仅适合公式/实现审计，不能做方法结论。
- 本轮纯 CPU：无模型加载、无 GPU、无训练；输入 SHA 在读取前后均绑定到 observation receipt。

## 6. 后续实现门槛

若继续做局部时空模块，必须先采到每个 source-held-out fold 至少 30 个同身份跨帧对和
30 个同预测类别异身份对，保存原始 ROI、裁剪/预处理、FPN level、box、frame index 与
可见性/对应元数据；再将几何中心、uniform pooling、global attention、感受野作用域 MLP
放在同一身份读出下比较。只有稳定超过 raw256/均值/固定空间先验，才允许进入完整训练。

文件与运行信息写入同名 JSON；结论是**可行性边界**，不是网络收益或论文结果。
