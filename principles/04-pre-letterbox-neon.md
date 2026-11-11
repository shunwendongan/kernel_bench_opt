# 04 · pre-letterbox-neon：与 cv2 逐字节一致的 letterbox，直接写进 fp16 输入缓冲

- 候选：[candidate.json](../experiments/pre-letterbox-neon/candidate.json) · 过程：[design.md](../experiments/pre-letterbox-neon/design.md)
- 源码：[letterbox.h](../kernels/pre-letterbox-neon/letterbox.h) · [letterbox.c](../kernels/pre-letterbox-neon/letterbox.c) · 微基准：[bench_letterbox.c](../kernels/pre-letterbox-neon/bench_letterbox.c)
- 父候选：`yolo_common.py::letterbox`（`cv2.resize INTER_LINEAR` + `copyMakeBorder(114)` + `cvtColor(BGR2RGB)`）与运行库内的 uint8→fp16 转换

## 1. 算子与位置

首轮“读取解码 + 预处理”平均 8.61 ms，其中 JPEG 解码本候选不触及；resize/补边/颜色转换由 OpenCV 单线程完成，然后 numpy 数组交给桥接，运行库再做一次 uint8→FP16 的转换与复制（13.19 ms，计在 `inputs_set`）。

## 2. 原实现的成本

三次独立遍历（resize 写一张中间图、copyMakeBorder 再复制一次、cvtColor 第三次），加上运行库的第四次遍历转 fp16。640×640×3 = 1.2 M 像素，每次遍历都受 Python/OpenCV 调度和内存带宽影响。

## 3. 优化原理

1. **几何完全复现**：`ratio = min(640/w, 640/h)`、`round(w·ratio)`、`round(dw ∓ 0.1)` 全部按 IEEE double 与 Python `round`（四舍六入五成双，C `nearbyint`）计算，transform 字典与参考逐字段相等（318 组尺寸测试）。
2. **复现 OpenCV 的定点双线性**：INTER_LINEAR 对 8 位图像使用 11 位定点系数（`saturate_cast<short>(coef·2048)`），横向以 int32 累加，纵向用 OpenCV `VResizeLinear<uchar,int,short,…>` 特化的规则 `((b0·(S0>>4))>>16 + (b1·(S1>>4))>>16 + 2) >> 2` 并饱和到 8 位；横向边界把 `fx` 置零并以单 tap 处理，纵向只裁剪行号而保留原始系数（这一点决定了首末行的结果）；`dsize == ssize` 时直接复制。坐标按 `scale = 1/(dsize/ssize)` 计算，与 OpenCV 的 `inv_scale` 一致。结果与 `cv2.resize` 在 58 组尺寸（含放大、缩小、随机尺寸）上逐字节相同。
3. **补边与颜色转换合并**：114 填充用 `memset` 直接写在目标行上，BGR→RGB 用 `vld3q_u8/vst3q_u8` 原地交换，不再有中间图。
4. **直接打包到模型输入**：`yolo_input_pack` 把 RGB uint8 转成模型要求的格式写进 DMA 缓冲：fp16 NHWC（NEON：16 字节 → 4×`vcvt_f16_f32`）、uint8 NHWC、NC1HWC2 fp16（3 通道补到 C2）、NCHW fp16/f32（供 ORT 参考，`divide=255` 时逐位等于 `np.float32(x)/255`）。uint8 到 fp16 的整数转换精确，无舍入。
5. **保留 uint8 中间图**：桥接 v2 保存 letterbox 后的 RGB uint8，用它核对冻结输入的 SHA-256。这是预处理一致性的直接证据，代价是一次 1.2 MB 的额外写入。

## 4. 不变量

- 输出 RGB uint8 字节与 `yolo_common.letterbox` 相同，因此与首轮冻结的 `dataset/inputs/*.u8` 哈希相同（板端 L2 逐图核对）。
- `rounding` 参数保留两种诊断规则（OpenCV 通用 22 位 castOp、尾部 16 元素用 castOp），只用于定位板端 cv2 构建差异，默认规则为 OpenCV 8 位特化。

## 5. 证据边界（开发机 Apple M4，仅说明方向）

| 场景 | NEON | 标量构建 |
|---|---:|---:|
| 640×480（等尺寸，纯复制 + 补边） | 0.10 ms | 0.34 ms |
| 500×375 → 640×480 | 0.90 ms | 1.18 ms |
| 1920×1080 → 640×360 | 0.64 ms | — |
| 打包 NHWC fp16 | 0.08 ms | — |

与 cv2 4.11（arm64 NEON 构建）逐字节一致：`tests/test_letterbox_kernel.py` 5 项通过。板端 OpenCV 版本与 CPU 的结果**未测量**；首轮 8.61 ms 中解码与预处理的拆分也未单独测量，L3 脚本把 `decode_ms` 与 `letterbox_ms` 分列。

## 6. 局限与下一步

- 横向插值目前是标量 int32 代码（gather 访问），1280×720 以上的大图可再向量化；A76 上预计 1～2 ms。
- NC1HWC2/NCHW 打包是标量路径（约 1.5～1.8 ms），只在原生输入属性要求这些格式时使用。
- 若板端 cv2 版本的舍入路径不同导致哈希不一致，用 `rounding=1/2` 定位并记录，不改冻结输入。
