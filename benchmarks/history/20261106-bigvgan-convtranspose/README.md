# BigVGAN ConvTranspose：历史 benchmark 证据

本目录归档 **2026-11-06、RK3588 / Mali-G610 原生 Linux 环境**下已有实验的小型文本证据。它不是 Docker 测量，也不表示新板已经通过。候选实现与工作合同见 [kernel](../../../kernels/bigvgan_deconv_s5_tiled.cl)、[算子合同](../../../docs/OPERATOR.md)、[实验状态](../../../docs/STATUS.md)；后续同步变体见 [SYNC_VARIANT.md](../../../docs/SYNC_VARIANT.md)。

原始 JSONL、有效配置、正确性记录、统计 CSV、参考图映射和环境遥测已经纳入 Git。权重、输入/输出张量、音频、模型与可执行文件保持外置；文本中只保留其身份、哈希和历史验证结论。

## 三级 benchmark 的结果

| 层级 | 比较范围 | 已保存结果 | 不能据此声称 |
|---|---|---|---|
| L1 单算子 | 同一真实输入、部署 packed 权重、FP16 及累加顺序；旧 kernel 对 tile4 | T52 / T120 / T136 的位级正确性记录通过；每种 shape 3 次预热、15 对交替测量 | 不能把局部速度比当作完整模型速度比 |
| L2 BigVGAN 集成 | MNN 中同一真实 mel；目标输入、目标输出和完整 vocoder 波形 | 三种 shape 共 9 项检查位级一致，dispatch 命中候选 | 本目录没有可报告的独立、干净 BigVGAN 总延迟结论 |
| L3 完整请求 | 同一语音样本、相同生成工作量；2 次暖机 + 3 对正式 A/B | **独立 PreDiT 同步构建**的 token 与波形一致；完整请求中位降低 3.19544% | 一个样本、3 对不能证明稳定 p95、统计显著性或所有输入收益 |

未开启 profiling 的单算子计时为 **host enqueue → clFinish**，buffer 常驻。编译、数据打包、传输和验证读回不计入该区间：

| T | 旧中位数 / ms | tile4 中位数 / ms | 旧/新中位数 |
|---:|---:|---:|---:|
| 52 | 369.157901 | 30.474510 | 12.1137× |
| 120 | 850.644925 | 64.446955 | 13.1991× |
| 136 | 975.302268 | 72.878692 | 13.3825× |

独立同步构建的完整请求中位数为 **94.060697120 → 91.055044558 秒**。旧/新两侧都使用 `OMNI_PREDIT_SYNC=1`，只在允许的目标 shape 上比较 tile0 / tile4；这不是把“异步旧版”直接与“同步新版”比较。完整请求中的 T120 / T136 命中候选，T108 按合同回退。

默认异步构建在暖机阶段波形门失败，**0 个正式 A/B 对**，记录保留于 [e2e/paired_20261106](e2e/paired_20261106)。失败请求的耗时不进入收益结论。同步成功记录位于 [e2e/paired_sync_20261106](e2e/paired_sync_20261106)，二者不合并。

## 文件导航

| 路径 | 内容 |
|---|---|
| [summary.json](summary.json) | 三级结果、计时范围、shape、工作量、指标与限制的机器可读摘要 |
| [manifest.json](manifest.json) | 每个公开副本的大小/SHA256、整理前原始 SHA、来源包、代码身份 |
| [validation.json](validation.json) | 本次文本归档与统计审计范围；明确未运行板子/容器、未重比波形字节 |
| `bench/T*_profile*/results.jsonl` | 6 组正式微基准：3 种 shape × profiling 开/关；保留逐次记录 |
| `bench/ablation_*/results.jsonl` | tile1 / tile2 / tile4 和旧 LWS 的消融记录；用于解释设计选择 |
| `e2e/` | 异步失败和同步成功各自的原始请求、比较门、有效配置及暖机 dispatch |
| `correctness/` | 部署权重转换、FP32 诊断、微基准/集成位级比较及 8 次波形哈希 |
| `statistics/` | 统计表、逐对变化及历史遥测摘要；时间单位见表内 `unit` |
| `records/` | 独立同步构建命令、执行命令、退出状态及温度/频率/负载遥测 |
| [mapping/reference_origin.json](mapping/reference_origin.json) | 参考 ONNX 的来源、原节点、张量边界与运行时结构候选证据 |
| [mapping/bigvgan_ups_0_0_reference.contract.json](mapping/bigvgan_ups_0_0_reference.contract.json) | 已提取参考节点的静态接口合同；不包含 ONNX 权重 |

原始记录中的 `files_ok`、`passed` 等字段是当时采集程序和校验程序的输出，不是复制到 Git 后重新对张量执行的验证。`correctness/e2e_file_integrity.json` 记录 8 份波形的相同 SHA256；要再次证明位级一致，需要取得对应外置字节。

## 仅克隆代码后复算统计

在仓库根目录运行；分析器使用 Python 标准库，不执行推理：

```bash
python3 tools/analysis/analyze_results.py \
  --bench-root benchmarks/history/20261106-bigvgan-convtranspose/bench \
  --e2e benchmarks/history/20261106-bigvgan-convtranspose/e2e/paired_sync_20261106 \
  --out runs/history-text-reanalysis
```

这里复算逐次计时、交替顺序、配置差异和已记录的正确性门。它不会读取或重新比较缺省外置的波形/张量文件；通过表示文本协议审计通过。替换为异步失败目录时，分析器应拒绝把失败记录算成合格收益。

`statistics/` 保留原公开统计副本；重新计算的结果写入新的 `runs/` 目录，不修改历史记录。profiling 开启的 GPU event 和宿主时间属于重叠计时域，不能相加；不开启 profiling 的 host 时间是此处主要比较口径。

## 参考图与真实 shape 的来源

`mapping/reference_origin.json` 对应重新从原 HF 权重导出的 BigVGAN 参考图，**不是历史 MNN 转换输入**。参考节点 `n000022` 为 `/ups.0.0/ConvTranspose`；合同中的 `explicit_selection` 与 `uncertain_mapping=false` 只表示选中了该参考节点，不证明从部署 runtime 到原始编译图的血缘。

以下 shape 属于不同记录，不能混为一条请求：

- 参考 ONNX 固定输入 T16，用于结构检查。
- 早期热点诊断样本的观察为 T120 / T136 / T46。
- 本轮微基准真实采集输入为 T52 / T120 / T136。
- 同步端到端样本使用 T120 / T136 / T108，T108 回退。

`reference_origin.json` 保留的是当时的源码候选和 `kernel_execution_confirmed=false`。后续集成中的候选 dispatch 证据独立保存在正确性与 E2E 记录里；不回写历史文件来提高旧证据等级。

## 来源及知识复用

文本来自已公开的 [历史资产 Release](https://github.com/shunwendongan/kernel_bench_opt/releases/tag/qwen2.5-omni-3b-vcb-bench-assets-20261106) 中 `benchmark-evidence` 与 `reference-onnx` 包，复制时保持公开副本逐字节一致。旧机器路径已在该公开包中整理为 `${TASK}`、`${BOARD_PROJECT}` 等占位符。`manifest.json` 同时保留整理前、公开后和本次复制的身份，避免把两种不同字节的哈希混用。

候选 `.cl` 与微基准 `.cpp` 对照历史源码逐字节相同；当前 MNN candidate execution 文件只做过尾部空白整理，原/新哈希均保留。此身份核对不代替新平台构建与运行。

这些数据适合沉淀“观察 → 假设 → 变更 → 正确性 → 分层指标 → 结论”的样本。训练或评价时应保留失败路径、数值/计时单位、设备/构建身份和小样本限制；同一请求的暖机、正式重复与不同摘要不能拆成互相独立的泛化样本。数值波形一致性也不等于 ASR/MOS 或语义答案质量通过。
