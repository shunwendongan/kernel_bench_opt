# BigVGAN ConvTranspose：从热点到候选的分析报告

## 原始热点证据

早期诊断请求 `en_qingganhua_000493` 中，`/ups.0.0/ConvTranspose_output_0` 的14次同步宿主callback合计 **13,110,591 µs**，观察到T120/136/46。该值带profiling开销，属于单节点多次调用的宿主区间总和，不是端到端时间或纯GPU event。

本目录 [profile_excerpt.jsonl](profile_excerpt.jsonl) 保留这14条事件及原始行号，完整源文件的压缩字节哈希见 [manifest.json](manifest.json)。这是明确筛选的热点片段，不是完整profile，也不能用来计算全模型热点占比；完整日志继续外置。

## 来源与实现

运行时节点与重新导出的BigVGAN参考ONNX零起算节点22 `/ups.0.0/ConvTranspose` 结构相符。参考图SHA及张量边界见 [来源记录](../../benchmarks/history/20261106-bigvgan-convtranspose/mapping/reference_origin.json) 和 [合同](../../benchmarks/history/20261106-bigvgan-convtranspose/mapping/bigvgan_ups_0_0_reference.contract.json)。缺少历史MNN转换输入，来源等级仍是 `reference_structure_candidate`。

可修改入口是MNN OpenCL Buffer执行器的 `DeconvBufExecution::onEncode`；候选以源码集成重新编译，显式匹配节点、结构、精度和shape后选择tile4。后续集成记录确认T52/120/136的候选dispatch；它与早期“只定位源码候选”的记录属于不同证据阶段。

## 优化与结果

tile4复用同stride相位的权重载入，保持权重、FP16累加和输出次序不变。独立真实capture的T52/120/136在L1位级正确性通过，局部中位速度比12.11–13.38×；L2同mel的完整BigVGAN输出位级一致。

默认异步完整请求波形门失败，0正式对。后续在PreDiT交接处同步的独立构建，旧/新两侧均使用同步设置，完整请求中位94.0607→91.0550秒，下降3.19544%。只有一个样本3对，T108回退；不能据此宣称全部输入或容器环境相同收益。

可复算的原始统计与错误记录见 [历史证据包](../../benchmarks/history/20261106-bigvgan-convtranspose/README.md)。[设计与下一轮方向](../../experiments/bigvgan-convtranspose-tile4/design.md) 给出真实末块覆盖、新板复现和优化后重新profile的顺序。

本报告由已有证据整理，新增profile片段已核对父文件SHA、14条事件数量和总耗时。本次没有新的板端采集或容器测试。大张量/模型不随本报告提交Git。
