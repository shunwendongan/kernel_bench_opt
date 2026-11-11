# 06 · stem-space-to-depth：把 3 通道 3×3/s2 茎部改写成 12 通道 2×2/s1

- 候选：[candidate.json](../experiments/stem-space-to-depth/candidate.json) · 过程：[design.md](../experiments/stem-space-to-depth/design.md)
- 源码：[rewrite_yolov8n_graph.py `--s2d-stem graph|host`](../tools/npu/rewrite_yolov8n_graph.py) · 宿主打包：[letterbox.c `yolo_input_pack`](../kernels/pre-letterbox-neon/letterbox.c)（NC1HWC2/NCHW 路径可扩展为 s2d 打包）
- 父候选：`Conv:/model.0/conv/Conv`

## 1. 算子与位置

`/model.0/conv/Conv`（3→16，3×3，stride 2，pad 1，输入 640×640）在最终模型 PERF_DETAIL 中耗时 **4.113 ms**，MacUsage 仅 0.70%，DDR 周期 970,032 > NPU 周期 921,600，RW 2,402 KB；首轮热点组 `model.0` 4.07 ms。它是单个节点中最慢的卷积，却几乎没有算力利用率。

## 2. 原实现的成本

3 通道 FP16 输入在 NPU 上要补到通道对齐（原生 NC1HWC2 的 C2 = 8），有效带宽利用率不到 40%；stride-2 卷积的输入访问模式对 NPU 的行缓存也不友好。时间几乎全部是数据搬运。

## 3. 优化原理

空间到深度（block 2）把 640×640×3 重排成 320×320×12：输出通道 `(pr·2 + pc)·3 + c` 存放行相位 `pr`、列相位 `pc` 的像素。原卷积输出 `y[i,j]` 读取输入行 `2i−1, 2i, 2i+1`，对应 s2d 块行 `i−1`（相位 1）和 `i`（相位 0、1），即块行上的 2 tap 卷积，pad 只在上/左各 1。于是 `Conv3×3/s2/p1(3ch) ≡ Conv2×2/s1/pads[1,1,0,0](12ch)`，新权重 `W′[o, (pr·2+pc)·3+c, kr, kc] = W[o, c, 2kr+pr−1, 2kc+pc−1]`（越界的 tap 为 0）。乘加次数相同（16 × 12 × 4 = 768 权重中 432 个非零），但输入变成 12 通道（补到 16 的利用率 75%）、320×320、stride 1。

两种落地方式：
- `graph`：图内插入 ONNX `SpaceToDepth`，输入契约不变；RKNN 可能把它编译成一次重排节点，收益要扣掉这次重排。
- `host`：图的输入直接是 `[1,12,320,320]`，由宿主在打包时按相位写入（与 letterbox 融合后不增加遍历）。输入契约改变，桥接 v2 元数据会报告新的输入属性。

## 4. 不变量

- 权重只重排，不改值；ORT 对比 9 路输出 max abs 4.0e-5（累加顺序不同导致的 float32 差异），检测完全一致。
- 其余 227 个节点不变；可与 01、05 叠加（`--logits6 --split-free --s2d-stem host` 已验证）。
- 宿主 s2d 打包的通道顺序由 `compare_onnx_variants.space_to_depth` 与 ONNX `SpaceToDepth` 的对比测试固定。

## 5. 证据边界

| 检查 | 结果 |
|---|---|
| 图结构（graph/host 两种） | 通过 `tests/test_rewrite_graph.py` |
| ORT 数值 | max abs 4.0e-5，检测一致 |
| RKNN 是否支持 2×2 核与 12 通道输入的高效路径 | **未知**，需转换 + PERF_DETAIL |
| 宿主 s2d 打包 kernel | 未实现 NEON 快速路径（可在 host 变体确认收益后补） |

## 6. 局限与下一步

- 若 RKNN 把 `SpaceToDepth` 编译成比原卷积差额更贵的重排，`graph` 变体判为 `rejected_perf`，再测 `host` 变体。
- 若 2×2 核在 NPU 上退化为低效路径（例如补到 3×3），该候选整体放弃，记录原因。
- `model.1`（2.245 ms，MacUsage 3.4%）同样 DDR 受限，但没有等价的免费重写，留作后续。
