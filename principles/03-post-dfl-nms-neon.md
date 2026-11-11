# 03 · post-dfl-nms-neon：9.9 ms 的 numpy 后处理变成一次 NEON 扫描

- 候选：[candidate.json](../experiments/post-dfl-nms-neon/candidate.json) · 过程：[design.md](../experiments/post-dfl-nms-neon/design.md)
- 源码：[yolo_post.h](../kernels/post-dfl-nms-neon/yolo_post.h) · [yolo_post.c](../kernels/post-dfl-nms-neon/yolo_post.c) · 对拍/微基准：[bench_post.c](../kernels/post-dfl-nms-neon/bench_post.c) · Python 封装：[tools/yolo_kernels.py](../tools/yolo_kernels.py)
- 父候选：`benchmarks/yolov8n_general/scripts/yolo_common.py::postprocess` 与 `model_adapter.py::adapt_outputs`

## 1. 算子与位置

首轮正常轮 DFL + NMS + 坐标还原平均 9.92 ms（中位数 7.85，P95 18.08），全部是 numpy：对 9 路 float32 输出按尺度做 `argmax/max`、`np.where`、DFL softmax、按类别循环 NMS。它处在 `rknn_outputs_get` 之后，与 NPU 串行。

## 2. 原实现的成本

每个尺度先对 80×H×W 的概率做 `argmax` 和 `max`（两次全张量遍历），布尔索引再产生临时数组；DFL 在候选上用 `exp`/`sum` 产生多次临时数组；NMS 按 `np.unique` 的类别循环，每轮都做 Python 级别的数组切片。真正的算术量很小（8,400 个 anchor × 80 类的比较，几百个候选的 64 维 softmax），时间都花在解释器与临时内存上。

## 3. 优化原理

1. **一次扫描完成 max/argmax**：NCHW 布局下以 4 个 anchor 为一组，沿 80 个通道做 `vcgtq`/`vbslq` 更新最大值与下标；严格大于保证并列时取最小类别号，与 numpy `argmax` 相同。NC1HWC2 布局（原生输出）下每个 anchor 的 8 个通道连续，按通道块扫描再做一次横向归约。fp16 用 `vcvt_f32_f16` 精确展开，不需要先转 float32。
2. **精确的预筛选**：canonical9 的第三路是裁剪到 [0,1] 的概率和。因为所有概率非负，`sum ≥ max`，所以 `sum < confidence ⇒ max < confidence`，该 anchor 可以跳过 80 通道的读取而不会漏掉任何候选；结果与不预筛选逐位相同（测试 `test_sum_prefilter_is_exact`）。logits6 模型没有这一路，改为在 logits 上做 max 再取 Sigmoid。
3. **DFL 复现 numpy 的归约顺序**：16 个 bin 的 `exp(d − max)`、归一化、与 `arange(16)` 的点积按 numpy `pairwise_sum` 的 8 累加器树形顺序求和，框坐标像 numpy 一样在 float64 中算 `((x+0.5) − l)·stride` 再舍入到 float32。剩余差异只来自 `expf` 与 numpy `exp` 的末位，框坐标差 ≤ 2e-3 px（测试门槛）。
4. **NMS 复现 numpy 的次序规则**：候选按尺度优先、行优先编号；排序键为（类别升序，分数降序，编号升序），对应 `np.unique` + `argsort(kind='stable')`；抑制条件 `iou > thr`，面积不加 1，分母 `max(union, 1e-12)`，全部用 float32；保留列表再按（分数降序，保留顺序）稳定排序后截断 `max_det`，最后按 `restore_boxes` 的 float32 公式还原并裁剪。
5. **零分配**：工作区由调用方一次分配（8,400 个候选 × 48 字节 + 两个索引数组），每帧不 malloc。

## 4. 不变量

- 输入：与 numpy 参考看到同样的数值（fp16 时参考先 `astype(float32)`），类别与分数必须逐位相同，框坐标允许 ≤ 2e-3 px。
- 非有限值：参考会抛出 `ValueError`；kernel 统计 `nonfinite` 计数，桥接 v2 据此报错，不静默。
- 输出布局 `(N, 6) float32 [x1,y1,x2,y2,score,cls]` 与参考相同，顺序相同。

## 5. 证据边界（开发机 Apple M4，仅说明方向）

| 场景 | NEON | 参考 C 路径 | numpy 参考一致 |
|---|---:|---:|---|
| canonical9 NCHW fp16，conf 0.25（555 候选） | 0.30 ms | 0.97 ms | 是（fp32/fp16/logits6/NC1HWC2，`tests/test_post_kernel.py` 7 项） |
| canonical9 NCHW + 求和预筛选 | 0.23 ms | 0.97 ms | 是 |
| canonical9 NC1HWC2 fp16 | 0.27 ms | 1.12 ms | 是 |
| logits6 NCHW fp16 | 0.27 ms | 0.94 ms | 是 |
| conf 0.001、8,400 个候选（评估模式） | 2.8 ms | 3.5 ms | 是 |

ORT 真实模型输出（合成图像）在 conf 0.25/0.001 两组阈值下与 `yolo_common.postprocess` 一致。板端 Cortex-A76 的时间**未测量**，首轮 9.92 ms 是对照基线。

## 6. 局限与下一步

- NC1HWC2 路径假定 `C2 == 8`（fp16）；其他块大小退回标量参考路径，功能正确但更慢。
- NMS 是 O(n²)/类别；评估模式（conf 0.001）下 1.5 ms 可接受，若未来需要更低可改为按分数桶预排序。
- 若板端需要 `confidence` 以外的筛选语义（例如 objectness），要同时改参考实现，kernel 不单独改语义。
