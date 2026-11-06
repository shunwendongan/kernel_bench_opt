# 历史 profile、映射与评测数据

这里保存原有实验的 **52 份小型文本证据**，供后续分析、复现与经验检索。历史 HTML 报告和两份独立优化 SOP 按要求未入库；本轮仅整理文件，没有重新运行模型或 benchmark。

## 从哪些数据开始

| 内容 | 入口 | 如何解释 |
|---|---|---|
| Omni / VCB 请求结果 | [results.csv](evidence/results.csv)、[summary.json](evidence/summary.json) | 从全类别抽取50条；任务按要求停止，实际完成29条，含23条文本与6条语音请求，不能写成50条完成 |
| 样本身份与评分参考 | [manifest.jsonl](evidence/manifest.jsonl)、[references.jsonl](evidence/references.jsonl) | 仅小型文本索引与参考，音频外置；参考答案不进入被测prompt |
| MNN 热点 | [算子排名](evidence/mnn_operator_rankings.csv)、[完整分析JSON](evidence/mnn_toolkit_analysis.json) | 宿主同步callback口径，含诊断开销；不是干净GPU kernel计时 |
| 来源映射 | [mapping_audit.json](evidence/mapping_audit.json)、[source_map.json](evidence/source_map.json)、[实现候选](evidence/kernel_source_candidates.json) | 保留候选与未确认关系；参考ONNX不等于部署MNN的历史转换输入 |
| NPU 编译单元 | [profile](evidence/npu_tile0.profile.json)、[mapping](evidence/npu_tile0.mapping.json) | 编译/融合单元与原ONNX节点不必一一对应 |
| 子图边界 | [BigVGAN合同](evidence/subgraphs/bigvgan_ups_0_0_reference.contract.json)、[Conv2+GELU合同](evidence/subgraphs/conv2_gelu.contract.json)、[来源记录](evidence/subgraphs/reference_origin.json) | ONNX二进制外置，保留shape、边界、来源和既有验证 |
| ConvTranspose微基准 | [统计与审计](convtranspose/analysis/microbenchmark_summary.json)、[汇总CSV](convtranspose/microbenchmark_summary.csv) | T52/120/136；干净计时与profiling分组解释 |
| 集成 / E2E | [集成正确性](convtranspose/resumed_board/records/integrated_correctness_20261106.json)、[同步批次](convtranspose/analysis/e2e_summary.json)、[异步失败](convtranspose/analysis/failed_e2e_audit/e2e_summary.json) | 同步批次一个样本3对；异步波形失败批次0正式对，分开保留 |
| 张量与环境身份 | [capture清单](convtranspose/capture_manifest.json)、[环境](convtranspose/resumed_board/records/final_environment_20261106.json)、[遥测](convtranspose/analysis/telemetry_summary.json) | 大张量外置；保留部署权重/程序身份与条件 |

逐次L1/L3原始记录及统计复算入口位于 [benchmarks/history](../../benchmarks/history/20261106-bigvgan-convtranspose/README.md)；当前候选见 [设计与适用范围](../../experiments/bigvgan-convtranspose-tile4/design.md)。本目录与这些入口有部分重合，是同一批来源的不同视图，不能作为独立实验样本重复计数。

## 来源与整理规则

[manifest.json](manifest.json) 为每个文件记录来源标识、原始字节SHA256、发布副本SHA256和整理动作。原文件保持原样；公开副本将机器绝对路径替换为 `${TASK_ROOT}`、`${BOARD_PROJECT}` 等符号根目录。与本实验无关的其他工作负载进度/控制信息已从副本中移除；模型自身的vision字段、实验指标与失败结果保留。

`convtranspose/code/` 是历史源码快照，便于对应当时的程序身份；其中构建脚本含符号路径，仅供溯源。当前可迁移入口使用仓库 [scripts](../../scripts)、[src](../../src) 和 [容器指南](../../docs/CONTAINERS.md)。历史文件中的validation字段、图表验收和哈希属于原实验/原报告，并非本轮重新验证。

模型、音频、完整trace、运行库与二进制见 [外置资产索引](external-assets.md)。保留原始许可：[VCB-Bench归属与CC-BY-4.0](../../licenses/VCB-Bench/ATTRIBUTION.md)、[第三方来源](../../docs/THIRD_PARTY.md)。公开评分参考和历史结果属于分析材料，训练/评测划分时应按来源分组。
