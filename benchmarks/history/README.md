# 历史运行索引

原始观测、派生分析和候选决策分开保存。换设备或精度时创建新 run，不覆盖历史结果；大资产以清单和哈希连接到本地档案。

| 运行 | 范围与结论 | 候选 / 报告 |
|---|---|---|
| [2026-11-06 BigVGAN ConvTranspose](20261106-bigvgan-convtranspose/README.md) | 三种真实 shape 的 L1/L2；独立同步构建中单样本三对 L3 通过；默认异步失败也保留 | [tile4 设计](../../experiments/bigvgan-convtranspose-tile4/design.md)、[热点片段](../../reports/20261106-bigvgan-convtranspose/README.md) |
| [2026-11-09 Omni 组合随机 10 条](20261109-omni-combined-random10/README.md) | 系统路径 A/B：9 对有效、1 对严格同输出、8 对输出变化、1 对编码失败；同输出验收未通过 | [组合设计](../../experiments/rknn-prefix-bigvgan-tile4/design.md)、[结果摘要](../../reports/20261109-omni-combined-random10/README.md) |

[机器可读索引](index.json) 只索引这些档案，不执行实验；来源与大资产见每个 run 的清单。
