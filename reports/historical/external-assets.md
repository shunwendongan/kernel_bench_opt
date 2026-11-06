# 外置资产索引

大型模型、数据集音频、张量、波形、完整profile与板端二进制保留在本地资产库。当前任务不配置台式机存储，不复制这些文件到Git。

[external-assets.json](external-assets.json) 记录历史证据引用的12个外置文件及大小。可从已有清单确定的SHA256注明其来源；未取得已记录身份的字段保留null，本轮没有重新读取全部大文件作校验。`workspace/outputs/...` 是历史来源标识，不是其他机器上可以直接使用的路径。

| 类型 | 对应信息 |
|---|---|
| 部署MNN模型、NPU前缀与真实capture | 仓库 [资产清单](../../assets/manifest.json) 和 [来源说明](../../docs/ASSETS.md)；未来使用端侧可见的本地目录 |
| BigVGAN参考ONNX、提取子图 | 本目录 `evidence/subgraphs/` 的合同与来源；参考图的作用是结构分析 |
| 完整MNN诊断trace | 当前仓库保留 [14条热点片段](../20261106-bigvgan-convtranspose/profile_excerpt.jsonl) 与完整解析结果；完整压缩trace外置 |
| 历史正式请求语音输出 | `evidence/results.csv`、`summary.json` 保留指标；六份WAV外置 |
| 同步版ARM64程序 | 历史构建/程序哈希保留；新设备按 [容器指南](../../docs/CONTAINERS.md) 构建与验收 |

此前发布的 [GitHub Release](https://github.com/shunwendongan/kernel_bench_opt/releases/tag/qwen2.5-omni-3b-vcb-bench-assets-20261106) 仍然保留，可作为已有资产的迁移来源。它不保证包含这里每个文件；以Release资产清单为准。今后的默认流程是“本地大资产 + GitHub代码/配置 → 端侧容器”。
