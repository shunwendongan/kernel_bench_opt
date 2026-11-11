# 候选 ID：head-logits6

原理：[principles/01-head-logits6.md](../../principles/01-head-logits6.md)。身份：[candidate.json](candidate.json)。

## 观察与来源

- 通用轮最终模型 `yolov8n_fp16.rknn`（sha256 `1b213f80…`）的 PERF_DETAIL（`benchmarks/yolov8n_general/evidence/model/final_cpu_placement.perf.txt`，单图 312192）：`Sigmoid:/model.22/Sigmoid` 21.208 ms、`Sigmoid_1` 2.740 ms、`Sigmoid_2` 0.559 ms，三者 24.507 ms，占单帧节点总时间 81.632 ms 的 30.0%；设备列 CPU，数据类型 FLOAT16。
- 首轮模型（NPU Sigmoid）详细轮节点和平均 61.42 ms；差额与 CPU Sigmoid 一致。
- 口径：PERF_DETAIL 节点耗时，含 profiling 开销；不是正常轮墙钟。
- 未确认：6 输出模型的原生输出布局；RKNN 编译器是否因输出数变化改变检测头切分。

## 可检验的假设

删除三个尾部子图后，`rknn_run` 的墙钟（正常模式）比最终模型减少接近 PERF_DETAIL 的 24.5 ms，且宿主侧 Sigmoid 只增加 ≤ 0.5 ms（8,400 次）；还原后的 canonical9 通过 `quality_protocol.json` 的数值门槛（RMSE ≤ 0.001、MAE ≤ 0.005）。否定条件：L2 中 `run_ms` 下降少于 10 ms，或任一数值门槛失败。

## 实现与不变量

- 基线：最终模型及其转换配置；父候选 `1b213f80…`。
- 变换：`tools/npu/rewrite_yolov8n_graph.py --logits6`，输出 `(318, cls_logits_80x80, 338, cls_logits_40x40, 357, cls_logits_20x20)`；`tools/npu/convert_variant.py` 以 FP16、mean 0 / std 255、rk3588、无 `op_target` 转换。
- 宿主：`post-dfl-nms-neon` 的 logits 模式（`cls_is_logits=1`）；验收导出用 `yolo_sigmoid_sum`。
- 不变量：权重、回归分支、预处理字节、后处理语义、质量门槛；ORT 还原后最大绝对差 3.85e-6。
- 回退：若数值门槛失败或 `run_ms` 无收益，继续使用 CPU Sigmoid 模型，保留记录。

## 评测计划

- L1：`tests/test_post_kernel.py`（logits6 路径与 `model_adapter.adapt_outputs` + `yolo_common.postprocess` 一致）；`tests/test_rewrite_graph.py`。
- L2（板端）：`scripts/run_yolo_board.sh l2 <logits6.rknn> <out>`，40 张冻结输入，每配置 3 次计时重复、3 次预热；对比 `run_ms`、`post_ms`，`--onnx` 时逐图执行 `compare_canonical` 门槛与检测匹配。详细模式另采 PERF_DETAIL 确认 Sigmoid 节点消失。
- L3：`scripts/run_yolo_board.sh l3 <logits6.rknn> <out>`，40 图 × 10 次，e2e 口径与首轮正常轮相同。
- 停止条件：L2 数值门槛失败即停止，不调整门槛。

## 结果与决策

计划中；尚未转换、尚未在板端运行。所有指标为 null。

## 可复用经验与后续方向

适用于任何把激活放到 CPU 的 RKNN 部署：运行库的 CPU FP16 激活是标量路径，把激活移到宿主并利用单调性只算必要元素更划算。下一个不同假设：若 NPU 上的 Sigmoid 下限问题源于 FP16 LUT，可测试 `Sigmoid` 前插入缩放/偏移是否改变下限（本轮不做）。
