# RKNN 音频前缀 + BigVGAN tile4：组合路径设计与决策

这份候选把既有 RKNN 音频前缀与已有 BigVGAN ConvTranspose tile4 同时接回推理系统，比较完整请求的表现。**本轮组合未通过输出保持验收**；9 对音频特征有效的请求总耗时降低 8.25%，只能描述两条系统路径的耗时差异。

优化原理：[principles/02-rknn-audio-prefix-conv-gelu.md](../../principles/02-rknn-audio-prefix-conv-gelu.md)。

| 身份 | 内容 |
|---|---|
| Task | `omni-combined-audio-route-e2e` |
| Candidate | `rknn-prefix-bigvgan-tile4` |
| 父基线 | `mnn-opencl-audio-encoder-deconv-tile0` |
| 复用候选 | [bigvgan-convtranspose-tile4](../bigvgan-convtranspose-tile4/design.md) |
| Run | `20261109-112115-omni-combined-e2e-random10` |
| 决策 | `output_equivalence_rejected`，保留数据与实现供后续定位 |

## 观察与假设

已有定位表明，音频编码前缀与 BigVGAN 的 `/ups.0.0/ConvTranspose_output_0` 是可研究的计算入口。BigVGAN tile4 在历史限定 shape 下已有独立对拍；音频侧已有包含 `Conv2 → GELU` 的 RKNN `Conv1 → GELU → Conv2 → GELU` 连续前缀。这里没有另一个独立优化的 Conv2+GELU kernel。

假设是：把这两条既有路径同时启用，可能降低请求延迟；需要重新核对模型输出与工作量，才能判断组合是否成立。历史单算子正确性不会自动覆盖后端迁移、编码器尾部与生成链路。

## 实际变换与保持项

| 项目 | old | new |
|---|---|---|
| 音频编码 | 原完整 MNN `audio.mnn`，配置保持 OpenCL | RKNN FP16 前缀 + 既有 CPU 编码器尾部 |
| BigVGAN ConvTranspose | tile0 原实现 | tile4；T=52/120/136 命中，否则原实现回退 |
| PreDiT 同步 | 开启 | 开启 |
| DiT | FP32，5 steps、solver=1 | 相同 |
| 模型/数据来源 | 固定同一快照与抽样清单 | 相同 |
| 生成配置 | CPU 线程4，Thinker/Talker 上限各1024，seed=20261109 | 相同 |

模型快照22个文件在运行前核对为一致，沿用部署权重来源；RKNN FP16 与 CPU tail 相对于原 OpenCL 编码器的数值执行路径发生变化，不能把它表述为“全链路精度保持不变”。代码、模型、库与前缀的身份见 [candidate.json](candidate.json) 和 [运行证据](../../benchmarks/history/20261109-omni-combined-random10/README.md)。

## 本轮评测协议

从 VCB-Bench 全类别的11,362条记录、63个 JSONL 中以 `random.Random(20261109).sample(..., 10)` 不放回抽样，保留抽取顺序：8条 audio2text、2条 audio2audio，其中3条多轮记录。保持原问题与历史，参考答案与模型输入分离。

两个模型实例常驻，串行执行请求；每条一对正式 A/B，相邻样本交替先后顺序。每侧用第3条语音任务预热一次，不计入正式结果。正式计时从 response 开始，到文本返回或完整语音生成结束；不含模型加载、预热和输出文件落盘。详细 profiling 与 capture 关闭，dispatch 只在预热记录，因此不能把预热命中当成每条正式请求的逐 kernel 证据。其他动态 shape 首次成本仍包含在正式耗时中。

这是原板裸机的系统路径 L3 对比；未构建或测量 Docker，也未形成两个改动各自的消融归因。

## 反馈与接受条件

| 检查 | 实际结果 | 决策含义 |
|---|---|---|
| 原始记录完整性 | 10对输入、20条正式请求、2条预热均保留 | 可以审查完整尝试，不等于正确性通过 |
| 音频编码与执行有效 | 9/10对 | 第10条两侧编码失败，剔出有效耗时与严格一致性统计 |
| 严格同输出 | 1/10对；有效集合中1/9对 | 8对有效请求有 token 或波形差异，组合输出保持失败 |
| 有效9对累计耗时 | 1263.105123413 → 1158.949073020 s | 降低8.246%，仅系统路径耗时描述 |
| 第5/9条 | 候选分别173.496/91.147 s，均慢于基线 | 候选两条达到1024文本 token 上限，不能隐藏退化 |

第10条外层程序返回0并报告 success，但日志包含 `Broad cast error, dim1 = 1500, dim2 = 2251`、`Compute Shape Error for /Add_2_output_0` 和 `Media encoder returned no features`。最终审查以运行日志覆盖原 success 标记，原始误报仍保留，未裁剪或替换该样本。

第2条是唯一严格同输出的正式配对，14.698678489 → 6.055010139 s；它不能代表全部输入。第7条文本 token 数同为84，但语音/波形仍不同；相同 token 数不构成完整输出一致。第3条波形长度不同，未对齐截断来制造一致性。

## 下一轮方向与可复用经验

1. **先对齐音频路径。** 文本任务也有差异，BigVGAN 不参与这些输出；应先比较 RKNN 前缀输出、CPU tail 边界与原 OpenCL 编码器，核对 layout、dtype、mask、动态长度和数值误差。
2. **分离两个改动。** 保持同一编码路线比较 tile0/tile4；再固定 BigVGAN 比较原编码器与 RKNN+CPU 路径。没有消融不能分别归因。
3. **修正失败判定。** 将音频特征有效性纳入 runner 的成功条件；新实现建立新候选/新 Run，保留本次精确测量源码与误报证据。
4. **限定工作量后再比较性能。** 先处理输出差异及1024上限，再增加重复次数、shape覆盖与第二设备验证；不能只保留更快的样本。

以上是后续设计建议，本次归档没有执行这些新增实验。保存的是可审阅的观察、假设与决策，不包含 Agent 隐藏推理过程。

## 来源与阅读入口

[结果摘要](../../reports/20261109-omni-combined-random10/README.md) · [逐次数据与来源清单](../../benchmarks/history/20261109-omni-combined-random10/README.md) · [精确 runner](../../src/apps/omni_batch_ab.cpp) · [离线分析器](../../tools/analysis/analyze_combined.py)
