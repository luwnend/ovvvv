# 动态外观阶段：完成后的续接

2026-10-24。**普通角速度完整对照、fresh独立全输出复算与主人口只读诊断均完成，整体研究仍active。不要重跑任何本阶段任务。**

## 当前权威与结论

- `results/angular_appearance_control_20261024_v1/completion_receipt.json`：`PASS_ANGULAR_APPEARANCE_CONTROL_AND_INDEPENDENT_OUTPUT_AUDIT`，SHA `0dda118132cfd885fbbe1778fb39b13d1a9d3674428bdc2385a361499f3deaa8`。
- `control_brief.json` SHA `1cce578853cc479b87b8e97f4cd598ffb7b69dd8115042ff47d49afa733138f9`，含全341源组、全部scope/pop/lag的轻量统计；raw NPY/rowmap/JSONL/SQLite/full summary不上传。
- 执行目录`results/angular_appearance_control_execution_20261024_v1`，最终status `PASS_ANGULAR_APPEARANCE_CONTROL_EXECUTION_AND_INDEPENDENT_AUDIT`；2026-10-24 01:49:16 Asia/Shanghai完成。producer181.403秒、fresh verifier243.139秒，实际两个child由detached supervisor wait()记录exit0；不是root直接wait这两个子进程。
- 全55,352,164 score rows：raw/angular独立最大误差均0；113719 querySQLite逐行一致，raw对parent所有2736source/unioncells与25573frame blocks精确一致。7保护和所有pin未变，CUDA关闭、未训练。
- 主same-predcat/common-CV/known P+N：lag1 query2949，raw/angular正确2903/2783，repair/break16/136；lag4 query2142，1993/777，46/1262。共同1331query/107source，raw错误16/99→angular60/869。
- `results/angular_history_assumption_20261024_v1/summary.json`与`execution_receipt.json`：只读主两lag共5091条query-lag记录的诊断actual tool session92916 exit0，summary输入/输出已再验。clean query2576/1912，仍repair/break5/98、30/1138；前后切向点积宏−0.2268809/−0.1895197，speed比宏1.38565/.47085。不是物理运动真值，shared reference测量噪声亦可造成负增量相关。
- 停止本人口的普通恒角速度优势/直接接入，不定义所有可学习动态理论失败，不根据结果加alpha/clip/换lag/source。

完整数字和三轮范围见`report/COVTrack_动态外观对照_20261024.md`。真实第2/3轮只限primary hard；更广负结果设计的两scope×两pop全部triplet任务未全执行，不能把设计或synthetic当真实完成。

## 最终实现与审计 pins

- protocol `angular_appearance_control_protocol_20261024.json` SHA `44d8bce20de4ed3fd358a083665e6e9d8b2744d29d86dcdc9b2d9efa2f07eb69`。
- producer `diagnose_angular_appearance_control_20261024.py` SHA `1e9fae320d2435a4560ffeeba3bfe441891f9d9db202296b269ba0419096828e`；三轮v2实际exit0。
- preflight SHA `fed48008427ce84c0a0732f9b6de25694cc2d199549dab84b2ad259e68b97824`；verifier SHA `c12bcd2ad7024a559cf0fa5c22ea7af9d3f6d2a18df8641cd379820fd139babd`，其final full-temp实际exit0。
- supervisor SHA `dbfa9eb345493c91c2356fc1154b6d01f53bd55faa194826dcea997be41167a2`；17项selftest通过。当前只是历史执行工具，不重launch。
- postseal diagnostic `diagnose_angular_history_assumption_20261024.py` SHA `2b48a706035e673527adc6563d2c38338ce3ba77960b24375dbe5d03ffc582f5`；三轮v4实际exit0(2/4/3项)，独立9项含raw_loader真实temp入口/strictclean/cross空格实际exit0。修过antipodal诊断turn字段，未改主预测；v2/v3本地保留。
- `appearance_transport_literature_audit_20261024.md`及sources JSON：5篇查重，`HOLD_NOVELTY_NOT_ESTABLISHED`，SFP精确公式`NEEDS_FORMULA_SFP_FULLTEXT`。角速度是强控制；历史/旋转/指数流的宽泛新意已被先例占据。

## 下一步（未执行）

按 `identity_spacetime_followup_recommendation_20261024.md`（SHA `23a1361129af172e201e10c0fefa82cf3e76c78ce640d824329652785f40f173`）完成source-OOF runner和严格训练/封存/heldout解码边界。不能把旧mean/pair角色当新干净train/test；新5fold metadata划分、CPU seeds3、adapter50epoch/head20epoch均已建议固定，不按结果扫。全部已看FIT上的OOF仍探索性。

纯adapter cosine必须改善才支持缓存后表征改善；同一adapter训练一次后freeze用于G0/G1/Gmask/Gperm，真实geometry须超过time/missing及GTfree错位控制。同query全部candidate始终保留，unknown不负例，不造null。所有fold/seed预测全局seal后再评价；每个模型f仅解码其train-source身份，heldout label mutation只约束该f。

目前最小`identity_spacetime_modules_20261024.py`已实现（普通控制），三轮v2纯合成与独立审计actual exit0（最终module SHA `2171db60d075039ba85c8b6b7389adc573793df0c1477e45390733b95b8c0bc3`，独立回执`identity_spacetime_modules_independent_audit_20261024.json`）；没有runner、真实训练、官方新收益。下一步完成资源估算、逐fold legal projector和完整streaming score封存，再执行训练。若GPU预计>20分钟，按用户要求可靠后台交接后暂停；CPU不触发此规则。

冻结`repos/COVTrack`、发布baseline、原数据、旧sealed结果和另一会话`results/covtrack_ctao_train20_bs24_20261022`。不改本账号已有训练；新GPU启动前现场查owner/UUID且禁4GB。SSH已有配置可用，不覆盖core.sshCommand，不需新key。Git工作副本`/home/chenhc/ov/ovvvv`；本阶段按显式白名单同步代码/文档/轻量回执；实际commit与remote核验以本地`tmp_analysis/angular_appearance_git_publication_20261024.json`为准，缺凭据不能假定已push。
