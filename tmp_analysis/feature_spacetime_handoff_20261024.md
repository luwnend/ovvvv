# 特征与时空阶段续接（2026-10-24）

用户要求改善网络特征，或更好利用时间、空间、外观与位置。整体研究目标仍在进行。此阶段只用原 TAO train / C-TAO 原人工 anchor 交集中的固定 fit 部分；没有新训练、GPU、官方 TETA 或新方法有效性结论。

## 已完成，不重跑

- `results/context_feature_memory_smoke_20261024_v1/`：固定12源组/20视频/697帧、28,082 retained行，五模式预测先封存，再投影924合法fit146身份anchor。CPU session85118实际exit0、80.935秒；独立session21264实际exit0、11.083秒。
- 首要raw_confidence与量化released的source-macro稀疏anchor B-cubed F1均0.824266037648786；raw_only/one_sided/symmetric分别0.8160414511/0.8222221286/0.8163623002。没有正收益，不改参数或扩大子集挽救这个结果。
- 924anchor的两模式分区完全等价：6132个same-pair均相同。全retained观测有5183个pair关系不同（2606 split、2577 merge），其中未知身份部分不能判好坏。量化released仅18/20视频分区与原GPU相同，1280个raw-ID差不能当错误数；必须用同量化released作对照。
- `completion_receipt.json`绑定audit SHA `632818fa8027fc57d138295a2810da032f2fe102bd233e68eb8feb51d94ab1ed`；`independent_primary_partition_supplement.json` SHA `1cabf1ddae3477ad957991956b140bd0f8fec6ba92d09b1c825904be421b605c`，旧audit不改。
- `relative_track_geometry.py`最终SHA `c5a085f8c56da990bc7f1cb2bedc7116c7ba32223e279f4ef678ed7447c4064a`：尺度归一位移、log尺寸比、原始帧差、past-only CV残差、history mask及外部相机H接口。producer11项和独立6项合成通过；尚未估计真实相机运动或接入tracker评分。

## 相邻时空候选诊断已完成，不重跑

`results/spacetime_candidate_features_20261024_v1/`：1,210,939个完整current×previous retained候选先封存，再连接同一924合法身份anchor；1,832个候选双端身份已知（608positive/1224negative），unknown不作负例。最终三轮v2及独立preflight通过；root session21963实际exit0、20.484秒，独立session22828实际exit0、15.675秒。全六特征/行映射/history/身份/AUROC/top1/3×3表独立重算通过，audit SHA `7d051e9722ebd176e3490073d7653c9bfb4ee3c1b411a6e7480c39b668b84559`。

同一CV-covered已知候选子集，391个至少一正一负query中rawcos正确384个；CV替代纠正4、破坏63，relative displacement纠正3、破坏56。该排序过滤unknown，仅是条件诊断，不能当完整候选assignment收益。同预测类别88query/6源组中rawcos83对，5个错全部集中于一个ArgoVerse源组；semantic84对（纠正4/破坏3）也只来自该组，不据此训练或择优。相邻原帧差15–90、中位25，history15–1140、中位30；无相机补偿，不能外推逐帧CV。

冻结producer SHA `c6fef39442dd476f918c326210ace407cdcb4e8b4700544f34454e29ba7665aa`、protocol SHA `d38beac2839a6473ff6a973e41a7c194febb818e40e433df63793e3349f086a1`。`completion_receipt.json`绑定实际exit与审计，保护7文件再次未变。

## 当前下一项

先审完整既有FIT缓存、预定lag `{1,4}` sampled steps的跨源支持，再考虑identity-only appearance adapter × geometry输入2×2。全部341源组/403实际视频/13,791帧/613,536retained，元数据计算55,352,164个完整候选；六score float64原始约2.474GiB。计划按video×lag分块memmap，生成/评分只驻留有限帧与单帧对，先全部封存再接合法fit146身份。原帧间隔、unknown、缺失CV、同预测类别与每源组错误覆盖均单列。不按现12源组错误或收益选样，不开tune/test/heldout targets。

该支持脚本/协议正在准备，尚未真实执行；CPU暂估15–45分钟，来自旧实现线性外推而非新实现实测。需三轮每轮2+synthetic、独立预审后root再启动；可靠后台启动与交接后可在其他工作期间等待。当前不启动网络训练。2×2的辨识原则：geometry关闭时adapter纯cosine改善才支持表征改善；同一冻结appearance下geometry增量才支持更好利用时空。普通adapter/融合本身不能宣称新颖。

## 已核实的代码边界

- 推理confidence比较当前raw与历史fused EMA，训练在raw key/ref上算confidence后融合双方；这是数据流差异，不是已证明性能bug。
- 1×1候选的cycle confidence恒1，不能替代绝对外观相似或运动相容性。
- 原绝对xyxy cosine不具共同平移不变性；共同均匀缩放下不变。新pair几何不能直接替换预训练bbox_transform的node输入单位。
- 缓存raw256和平均cls512是float16；text/image独立向量和ROI张量未缓存。离线adapter可更新自身，不能回传更新ROI/backbone。
- 新网络/注意力/adapter/几何门控的有效性与新颖性仍待验证。下一项由真实时空区分和排序结果决定，不依据AUC直接启动训练或声称跟踪提升。

## 保护与操作

不改`repos/COVTrack`、发布baseline、原数据、sealed pilot/full结果及另一会话`results/covtrack_ctao_train20_bs24_20261022`。unknown不作负例；test/留出类correctness关闭，tune不用于改参数。旧错误TCES分析已撤下，不沿用。

科学解释器：`/home/chenhc/.conda/envs/ovtrack_ovmot/bin/python`；系统python3无NumPy。研究目录`/home/chenhc/ov/OVMOT_RESEARCH`不是Git；托管副本`/home/chenhc/ov/ovvvv`，远端`git@ssh.github.com:luwnend/ovvvv.git`。SSH已可用，无需新密钥。使用显式轻量allowlist，单文件≤2MB、总≤20MB，NPZ/JSONL/模型/日志不上传。同步后核逐文件hash、diff，再显式add/commit/push并核remote SHA与clean。

预计GPU超过20分钟时，可靠detached启动、验证并保存交接后按用户规则暂停Codex；CPU不自动触发。既有full采集/support/强mean/上下文/身份支持已完成，勿重跑；溯源见`full_stage_handoff_20261023.md`。
