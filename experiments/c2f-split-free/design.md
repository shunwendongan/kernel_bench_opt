# 候选 ID：c2f-split-free

原理：[principles/05-c2f-split-free.md](../../principles/05-c2f-split-free.md)。身份：[candidate.json](candidate.json)。

## 观察与来源

最终模型 PERF_DETAIL：8 个 `Split:/model.N/Split` 合计 2.859 ms（0.652/0.330/0.371/0.239/0.426/0.325/0.382/0.134），单核执行，`model.2/Split` RW 1,600 KB。首轮热点组 `model.2`、`model.4`、`model.15`、`model.6` 都是含 Split 的 C2f。

## 可检验的假设

删除 Split 后对应模块的节点时间之和下降，且 `cv1a + cv1b` 的时间不超过原 `cv1 + Split`；全模型 `rknn_run` 减少 1～2.5 ms。否定条件：模块节点时间不降或上升（输入重复读取代价大于 Split）。

## 实现与不变量

- 变换：`rewrite_yolov8n_graph.py --split-free`：对每个 Split 的生产链 `Conv → Sigmoid → Mul` 生成 `cv1a/cv1b`（权重、偏置按输出通道切片），输出名沿用 `Split_output_0/1`，删除原 `cv1`/SiLU/Split。
- 不变量：ORT 9 路输出逐位相同；转换配置与 CPU Sigmoid 的 `op_target` 不变；可与 `--logits6` 叠加。
- 回退：使用未改写模型。

## 评测计划

- L1：`tests/test_rewrite_graph.py`（结构 + 数值），`compare_onnx_variants.py`。
- L2：转换后按 `l2_compare_bridges.py` 跑 40 图（数值门槛、检测匹配、`run_ms`）；详细模式对比各 `/model.N/` 分组的节点时间。
- 停止条件：数值门槛失败即判 `rejected_numeric`。

## 结果与决策

计划中；未转换、未上板。

## 可复用经验与后续方向

“按输出通道切片卷积”可以无代价地消除紧随卷积的 Split；类似地可以消除 Slice。Concat→多路卷积求和则会引入 Add，账面不划算，已记录为不采用。
