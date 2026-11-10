# 实验状态 — 2026-11-10

## 新增：仓库约定、候选目录重组与检查脚本

- 候选源码按候选 ID 重组：`kernels/<candidate-id>/`、`src/mnn/overlays/<candidate-id>/`、`experiments/<candidate-id>/`；内容与 SHA-256 不变，旧路径映射见 [CONVENTIONS.md 第 7 节](CONVENTIONS.md)。`scripts/patches/manifest.json` 升为 schema 2，`prepare_mnn.py --candidate-set tile4|v2` 可切换候选集。
- 新增 [`docs/CONVENTIONS.md`](CONVENTIONS.md)、根目录 `AGENTS.md`、`workload.json`、`schemas/`、`experiments/index.json`（候选登记）、`benchmarks/history/index.json` schema 2（含 v2 候选的 planned 运行记录）、`tools/verify/check_repo.py`；GitHub Actions 工作流 `check.yml` 放在 `main` 分支，随 `git merge main` 进入各工作负载分支（当前推送 token 缺少 `workflow` 权限，待补）。
- 新增仓库内热点排名脚本 [`tools/analysis/rank_hotspots.py`](../tools/analysis/rank_hotspots.py)，不再依赖外部工具包即可从 `OMNI_PROFILE_PATH` 的 JSONL 复算算子排名。
- `README.html` 已于 2026-11-10 用 Pandoc 3.13 重新导出（`scripts/export_readme.py`），与当前 README.md 哈希一致；以后改 README.md 后同样重新导出，`check_repo.py` 会检查两者是否匹配。
- 共享工具将移到 `main` 分支，工作负载分支通过 `git merge main` 同步。

## 新增：候选集 v2（源码交付，待板端验收）

- **bigvgan-deconv1d-v2**：通用 1-D ConvTranspose OpenCL kernel。
  - 相位分解；寄存器分块为 TT 个时间位置 × OCB 个输出通道块；每个 ic block 只读一次输入窗口；默认 LWS 沿时间维展开。
  - 覆盖 BigVGAN `/ups.0~5` 全部 6 个反卷积（热点表合计 18.2 s），任意 T，不再有 T108 回退。
  - 每个输出的 mad 序列与 MNN `deconv_2d` 相同。
- **rknn-audio-prefix-v2**：预分配 IO，可选多核和多 tile 并发，首条请求位级自检并自动回退。

交付前只做了非目标设备检查：tap 次序穷举证明（316 万个输出位置）、MNN 7e3c6fb 集成语法检查、Apple M4 FP32 位级冒烟。**尚无 RK3588 FP16 性能或正确性结果。** 部署和测试步骤见 [V2_DEPLOY.md](V2_DEPLOY.md)。各候选的优化原理见 [principles/](../principles/README.md)(2026-11-10 新增，覆盖 tile4、RKNN 前缀子图、PreDiT 同步和本轮两个 v2 候选)。

## 以下为 2026-11-09 状态


## 新增：组合路径随机10条 A/B

2026-11-09 原板裸机运行已结束，10条输入均尝试、20次正式请求和2次预热完整归档。原 MNN/OpenCL 完整音频编码器 + tile0，与既有 RKNN FP16 前缀 + CPU tail + tile4 比较。**9对有效、1对严格同输出、8对有效但输出变化，第10对两侧编码失败；组合输出保持验收未通过。**

有效9对累计耗时1263.105123413→1158.949073020秒，降低8.246%，仅系统路径耗时描述。不能作为同输出收益、答案质量或两个改动各自贡献。原runner的success误报与最终修正均保留；第5、9条的退化也保留。详见 [结果摘要](../reports/20261109-omni-combined-random10/README.md)、[运行档案](../benchmarks/history/20261109-omni-combined-random10/README.md)、[候选设计](../experiments/rknn-prefix-bigvgan-tile4/design.md)。

本次发布仅整理文件并离线复算；没有连接板子、构建镜像、重新推理或操作其他工作负载。板端实验结束并已确认本任务进程退出。新增runner源码与实测原件字节一致；本次补入的CMake构建入口尚未重新编译。模型、音频、波形与完整HTML按约定留在本地，身份见 [资产清单](../assets/runs/20261109-112115-omni-combined-e2e-random10.json)。

## 以下为2026-11-06单算子历史结果

当前候选针对 BigVGAN 第一个 ConvTranspose。以下是先前 RK3588/Mali-G610 板端记录核对后的历史结果；本次 GitHub 代码迁移没有连接板子或启动新的推理。

现在可从Git直接查看并复算 [小型历史证据包](../benchmarks/history/20261106-bigvgan-convtranspose/README.md)，查阅 [热点片段报告](../reports/20261106-bigvgan-convtranspose/README.md) 与 [候选设计](../experiments/bigvgan-convtranspose-tile4/design.md)。新增Docker配方尚未构建或板端运行；不改变下列历史结果的适用范围。

## 已通过的范围

输入 `[1,1536,T]`，输出 `[1,768,5T]`，权重 `[1536,768,11]`，stride5/pad3/dilation1/groups1。三种 T 来自真实语音请求：120 首块、136 稳态块、52 原生末块，未通过裁剪伪造 shape。

| T | 原 kernel 中位数 ms | tile4 候选中位数 ms | 原/新中位数 |
|---:|---:|---:|---:|
| 52 | 369.157901 | 30.474510 | 12.1137× |
| 120 | 850.644925 | 64.446955 | 13.1991× |
| 136 | 975.302268 | 72.878692 | 13.3825× |

口径：关闭 profiling 的 host enqueue→clFinish；驻留输入/权重/输出 buffer；每种 shape 预热 3 次、15 个交替顺序 A/B 对。编译、数据打包、输入输出拷贝和正确性读回不在此计时内。因此这是局部算子收益，不是完整 BigVGAN 或 Omni 加速倍数。另有打开 profiling 的 OpenCL event 时间，仅用于分析，不与 host 时间相加。

- exact packed weight/bias 对应部署值转换为 FP16 RNE，未改权重或精度。
- 三种 shape 的微基准旧输出与部署捕获输出一致，候选与旧输出 bitwise equal；9 个请求的比较门通过。
- MNN 集成中三种 shape 的输入、目标算子输出及三个完整 BigVGAN 波形均 bitwise equal；另外 9 个检查通过。
- 集成 dispatch 确认为候选 tile4、precision0。不是只凭配置猜测 GPU 执行。
- 同有效权重的 FP32 参考相对 RMSE 约 0.73%–1.24%，是已有 FP16 累加误差的诊断，候选没有新增此误差。

## 默认异步版本未通过：完整 Omni 端到端 A/B

完整请求使用两个独立常驻 LLM 实例、相同输入和种子。当前样本中 text token 和 226 个 speech token 相同，两个输出各有 106,560 个 float 样本；波形不同：

- max absolute error：`0.6606618762`
- relative RMSE：`0.3390519365`
- `measured_pairs_completed = 0`

差异集中在波形前段。T108 出现在该请求末块，按合同回退到原 kernel。问题根因尚未确定；同一 mel 的三形状回放通过不能证明完整请求通过，也不能据此断定 kernel 数学本身有错。暖机的旧/新耗时不作为加速证据。

后续独立构建加入可选PreDiT输出等待后，已在同一真实样本通过3对完整A/B，所有8次暖机/测量波形位级一致，完整请求中位94.060697120→91.055044558秒（降低3.19544%）。这一结果属于独立同步候选，不归给本仓库默认异步实现；源码补丁、限制和操作见 [SYNC_VARIANT.md](SYNC_VARIANT.md)。新板和其他输入仍需验证。

## 代码迁移的变更与局限

- 替换原机器绝对路径与 `flags.make` 依赖，提供固定 MNN 版本的独立源码构建。
- 保留现有 kernel、dispatch 和最新 `omni_ab.cpp` 算法。
- `vcb_eval` 的 `dit_s` 改为 `null`，并保留原 `vision_us` 差值在 `talker_vision_counter_delta_s`，因为该 Qwen2.5 路径没有有效的独立 DiT 计时。`omni_ab` 已采用相同口径。
- `compare_raw.py` 增加 `--bitwise` 严格开关，避免把 `+0.0/-0.0` 数值相等误说成位级相等。
- VCB 选择清单移除机器绝对路径，内容身份必须通过数据快照、排序规则和源文件哈希核对，不能只看 seed。
- 迁移版未在新板编译链接/运行。源码准备、语法检查不等同于新板通过。
- 更早的正式 VCB 50 样本任务按要求停在完成 29 条处（23 文本、6 语音），其余 21 条未运行；不宣称完成 50 条答案质量评测。

## 结果来源身份

统计来自原始 `T*_profile*/results.jsonl`，分析入口为 `tools/analysis/analyze_results.py`。原始张量、音频和完整结果已通过Release附加，见 [ASSETS.md](ASSETS.md)。公开元数据路径作了可移植整理，随包记录整理前后摘要；下列SHA对应原始本地日志：

| 未 profiling 原始日志 | SHA-256 |
|---|---|
| T52 | `f5ee95b9724f6092b756d839436a49b308ef0b06adb64b4d59dee4635ddbf777` |
| T120 | `102138d5c37c6b0837536fce5198abe89595db5e8bb6f0bd873d6212a909d696` |
| T136 | `03c34b6e3e0ef4f574ad8a76610d3d9678f95cc0f6095e1169447ba60d93363a` |

下载 `benchmark-evidence` 后，可对 `evidence/bench` 和 `evidence/e2e/paired_sync_20261106` 运行分析器复算。仅克隆代码不下载Release附件。上述测量是历史板端结果，不代表已在新板重新测试。

## 本地任务管理代码交付

已新增 [Task / Candidate / Run 管理CLI](SANDBOX.md)、资产引用清单、候选/输入快照、显式精选文件归档、派生索引与查询。默认工作区 `.sandbox/` 被Git忽略。另附图片与样式内嵌的 [README.html](../README.html) 和可重复导出脚本。

本轮仅代码、配置与文档交付，未运行任务管理流程测试、模型测试、Docker构建或板端操作。该模块不执行记录中的命令，也不提供设备锁、调度或自动Git发布。
