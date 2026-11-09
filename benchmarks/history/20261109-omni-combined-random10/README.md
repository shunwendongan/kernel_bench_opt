# Omni 组合路径 · 随机 10 条原始记录

本目录归档 `20261109-112115-omni-combined-e2e-random10` 的精选文本证据。Task 为 `omni-combined-audio-route-e2e`，Candidate 为 `rknn-prefix-bigvgan-tile4`。这是原板裸机测量，不是容器测量；本次整理没有重新连接设备或运行推理。

**数据完整，组合的同输出验收未通过。** 共 10 条输入、20 次正式请求、2 次预热；9 对音频编码及输出有效，1 对严格同输出，8 对有效但输出有差异，第 10 对两侧音频编码失败。有效 9 对总耗时 1263.105123413 → 1158.949073020 秒，降低 8.246%；工作量有变化，只作为系统路径耗时描述。

| 材料 | 入口 | 用途 |
|---|---|---|
| 任务与运行身份 | [task.json](task.json)、[run.json](run.json) | 目标、候选、环境、输入、决策与来源组 |
| 原始正式及预热记录 | [e2e/results.jsonl](e2e/results.jsonl)、[e2e/run.json](e2e/run.json) | 全部逐次耗时、token、原程序状态 |
| 原程序对拍结论 | [e2e/comparisons.json](e2e/comparisons.json) | 保留原始声明，不能替代后续运行日志审查 |
| 输入与协议 | [selection](inputs/selection/selection_summary.json)、[manifest](inputs/bundle/manifest.jsonl)、[references](inputs/bundle/references.jsonl)、[protocol](inputs/bundle/protocol_evidence.json) | 全类别不放回抽样；问题和历史不改；评分参考与输入分开 |
| 配置与路径 | [config.json](config.json)、[old](e2e/route_old.json)、[new](e2e/route_new.json) | 原 MNN/OpenCL 完整音频编码器，与 RKNN 前缀 + CPU 尾部的区别 |
| 命令与环境 | [command](records/command.json)、[compile](records/compile.json)、[environment](records/environment_before.json)、[model](records/model_before.json) | 实测二进制、源码、模型、库身份和运行参数 |
| 编码失败原文 | [process.txt](records/process.txt) | 第 82–91 行：shape 错误、无音频特征；覆盖外层 success 声明 |
| 环境遥测 | [telemetry.jsonl](records/telemetry.jsonl) | 温度、频率、负载等运行条件 |
| 最终审查与统计 | [final_acceptance](records/final_acceptance.json)、[summary](analysis/summary.json)、[CSV](analysis/per_sample.csv)、[answers](analysis/answers.json) | 有效性修正、逐样本差异；未进行答案/听感质量评分 |
| 发布来源 | [manifest.json](manifest.json)、[publication_index.json](publication_index.json) | 原件/公开副本哈希、路径规范化、发布文件索引 |
| 大资产 | [外置资产清单](../../../assets/runs/20261109-112115-omni-combined-e2e-random10.json) | 输入音频、输出波形和外部 HTML 的身份；模型身份另见 model 记录 |

每个 `e2e/sample_<n>/` 保留输入、两侧 metrics、answer 和 stream；预热单独存于 `e2e/warmup_*`，不混入正式统计。仅预热 dispatch 日志用于证明候选命中；正式阶段关闭详细 profiling、capture 和 dispatch 记录，不能声称有正式逐请求 kernel trace。

## 只用仓库文本复算

在仓库根目录运行：

```bash
python3 tools/analysis/analyze_combined.py \
  --root benchmarks/history/20261109-omni-combined-random10 \
  --out runs/combined-random10-reanalysis \
  --recorded-waveform-checks \
    benchmarks/history/20261109-omni-combined-random10/analysis/recorded_waveform_checks.json
```

这会重算耗时、token、运行错误和有效性，并核对绑定文本的 SHA256；波形使用已保存的历史校验，明确标记 `recorded_evidence_not_rehashed`，不会重新读取外置波形。预期 `record_integrity_passed=true`、`audit_passed=false`（保留真实编码失败），有效 9 对、严格同输出 1 对。程序返回成功只表示完成分析，不表示候选验收成功。

需要重新校验原始波形时，省略记录模式，提供 `--waveform-root /path/to/local/archive/board/e2e`；目录内须保留原来的请求相对路径。这仍是离线数据分析，不是新设备推理。

## 构建与证据边界

实测 runner 的原始字节位于 [src/apps/omni_batch_ab.cpp](../../../src/apps/omni_batch_ab.cpp)，SHA256 为 `84d75273ab1e25cad8b2786dbdd400003c3d0a483c9670e0f7f112f9d6435fa0`。历史编译复用了独立同步静态库；本次将 runner 加入已有 CMake 和构建脚本，但没有重新编译。两侧历史运行均启用 `OMNI_PREDIT_SYNC=1`，引擎构建需使用 [同步变体](../../../docs/SYNC_VARIANT.md)。

原 runner 对“音频编码器返回空特征后继续生成文本”的情况缺少失败判断，因此原始 `success=true` 不可信；保留原源码有助于复查测量语义。可移植分析器通过 process 日志识别本轮失败，未来更换日志或 runner 时应重新核对诊断规则。

公开副本把机器目录替换为 `${RUN_ROOT}`、`${BOARD_RUN}`、`${BOARD_PROJECT}`、`${TOOLKIT_ROOT}`、`${TASK_ROOT}` 和 `${ASSET_ROOT}`。它们是来源占位符，历史命令不会自动展开后执行。原数据哈希仍对应规范化前原件；`manifest.json` 同时记录两份身份。`original_bundle.SHA256SUMS` 是原始完整 bundle 的校验记录，包含未入 Git 的音频，不能直接当作本目录的完整性检查。

数据集文字记录保留 [VCB-Bench 署名和许可](../../../licenses/VCB-Bench/ATTRIBUTION.md)。同一输入的重复运行与派生摘要共用来源组，未经质量审查不要直接当作优质训练样本。完整 HTML 报告、音频、波形、模型和缓存按项目约定留在本地。

[结果图文摘要](../../../reports/20261109-omni-combined-random10/README.md) · [候选设计与后续方向](../../../experiments/rknn-prefix-bigvgan-tile4/design.md)
