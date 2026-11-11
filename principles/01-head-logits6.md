# 01 · head-logits6：把 24.5 ms 的 CPU Sigmoid 从 `rknn_run` 里拿出来

- 候选：[candidate.json](../experiments/head-logits6/candidate.json) · 过程：[design.md](../experiments/head-logits6/design.md)
- 源码：[rewrite_yolov8n_graph.py `--logits6`](../tools/npu/rewrite_yolov8n_graph.py) · 转换：[convert_variant.py](../tools/npu/convert_variant.py) · 宿主侧配套：[post-dfl-nms-neon](03-post-dfl-nms-neon.md)
- 父候选：通用轮最终模型 `yolov8n_fp16.rknn`（`1b213f80…`，三个最终 Sigmoid 以 `op_target` 放到 CPU）

## 1. 算子与位置

检测头每个尺度的分类分支以 `cv3.x.2 Conv(1×1, 80→80) → Sigmoid → ReduceSum(axis=1) → Clip(0,1)` 结束，输出概率（80 通道）和裁剪后的概率和（1 通道）。首轮模型让 Sigmoid 跑在 NPU 上，FP16 实现出现 0.00183 的概率下限，分类/求和输出没有通过相对 L2 门槛；通用轮把三个 Sigmoid 指定到 CPU 后数值通过，但最终模型的 PERF_DETAIL 里这三个节点耗时 21.208 + 2.740 + 0.559 = **24.507 ms，占单帧节点总时间 81.63 ms 的 30%**（单图 312192）。

## 2. 原实现的成本

672,000 个 FP16 元素的 Sigmoid 用了 24.5 ms，约 27 M 元素/s：运行库的 CPU FP16 Sigmoid 是单线程标量路径，并且它处在 `rknn_run` 内部，NPU 在这段时间里空等。随后的 `ReduceSum_2conv`/ConvClip 还要把 80 通道概率再读一遍（0.36 ms），`OutputOperator` 再把三路概率和三路求和写出（约 2.3 MB）。而宿主侧后处理真正需要的只是每个 anchor 的 **最大类别概率和它的类别号**：8,400 个 anchor 各做一次 sigmoid。

## 3. 优化原理

1. **图改写**：对三个尺度删除 `Sigmoid`、`ReduceSum`、`Clip` 及其常量，把 `cv3.x.2` 的输出直接作为图输出 `cls_logits_{80,40,20}x{…}`。输出顺序变为 `(reg64, logits80) × 3`，与 `model_adapter.adapt_outputs` 的 logits6 契约相同。模型内不再有任何 CPU 节点（InputOperator/OutputOperator 除外），`rknn_run` 只剩 NPU 工作。
2. **单调性**：Sigmoid 单调递增，所以 `argmax_c sigmoid(x_c) = argmax_c x_c`，`max_c sigmoid(x_c) = sigmoid(max_c x_c)`。宿主扫描在 logits 上做 max/argmax，只对每个 anchor 的最大值算一次 float32 稳定 Sigmoid（`x ≥ 0 → 1/(1+e^-x)`，否则 `e^x/(1+e^x)`，与 `model_adapter.stable_sigmoid` 相同），再与 confidence 比较。工作量从 672,000 次 Sigmoid 降到 8,400 次。
3. **验收导出**：质量门槛定义在 canonical9 上。桥接 v2 的 `yolo_sigmoid_sum` 用 NEON 对整张 logits 张量做 float32 Sigmoid 并累加、裁剪，得到概率和求和两路，供 `prepare_model.compare_canonical` 直接使用；这一步只在验收/导出时执行，不在计时路径内。
4. **数值更好而不是更差**：CPU FP16 Sigmoid 的输出被舍入到 FP16，在 p≈0.5 附近 ulp 为 4.9e-4，这正是通用轮 RMSE 5.5e-4 的来源；宿主侧 float32 Sigmoid 只承受 logits 本身的 FP16 舍入（|x|<1 时 ulp ≤ 9.8e-4，乘以 sigmoid′ ≤ 0.25），预期 RMSE 更小。

## 4. 不变量

- 权重、卷积、回归分支字节不变；只删除三组尾部节点并改输出列表。ORT 对比：6 路经 `adapt_outputs` 还原后与原图 9 路的最大绝对差 3.85e-6（6 张合成图），检测结果完全一致。
- 候选筛选的比较基准仍是 `max 概率 ≥ confidence`，NMS、排序、坐标还原不变（见 03）。
- 质量门槛（RMSE ≤ 0.001、MAE ≤ 0.005 等）不变，用还原后的 canonical9 评估。
- RKNN 转换配置不变（FP16、mean 0 / std 255、rk3588），只是 `op_target` 为空。

## 5. 证据边界

| 检查 | 结果 | 说明 |
|---|---|---|
| 图结构 | 228 → 213 个节点，6 路输出 | `tests/test_rewrite_graph.py` |
| ORT 数值 | max abs 3.85e-6（还原后） | 开发机 float32，非板端 |
| 宿主 Sigmoid/求和 | 与 `model_adapter.adapt_outputs` 差 ≤ 3e-7 / 2e-5 | `tests/test_post_kernel.py` |
| RKNN 转换 | **未执行**（需 Linux x86 + toolkit2 2.3.2） | `convert_variant.py` |
| 板端时间 | **未测量** | 由 L2/L3 给出 |

## 6. 局限与下一步

- 6 输出模型的原生输出布局要由板端 `rknn_query(NATIVE_OUTPUT_ATTR)` 确认，桥接 v2 已支持 NCHW 与 NC1HWC2 两种。
- 若 RKNN 编译器因为输出数量变化改变了检测头卷积的切分，节点时间会变；比较时看整段 `rknn_run` 和 PERF_DETAIL 两种口径。
- 若板端 RMSE 门槛仍失败，保留结果并回到 CPU Sigmoid 模型，不修改门槛。
