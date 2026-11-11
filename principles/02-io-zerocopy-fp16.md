# 02 · io-zerocopy-fp16：让运行库不再替我们转换 2.4 MB 输入和 2.4 MB 输出

- 候选：[candidate.json](../experiments/io-zerocopy-fp16/candidate.json) · 过程：[design.md](../experiments/io-zerocopy-fp16/design.md)
- 源码：[rknn_bridge_v2.cpp](../src/apps/rknn_bridge_v2.cpp) · [rknn_bridge_v2.py](../src/apps/rknn_bridge_v2.py) · [build_bridge_v2.sh](../src/apps/build_bridge_v2.sh)
- 父候选：首轮桥接 `benchmarks/yolov8n_general/scripts/rknn_bridge.cpp`（`rknn_inputs_set` + `rknn_outputs_get(want_float=1)`）

## 1. 算子与位置

首轮正常轮的“推理调用整体”71.12 ms 里，`rknn_run` 只有 43.98 ms；`rknn_inputs_set` 13.19 ms、`rknn_outputs_get` 12.00 ms、Python 复制 1.72 ms 合计 **26.9 ms**，是端到端 89.65 ms 的 30%。

## 2. 原实现的成本

运行库元数据说明了为什么：模型输入属性是 `NHWC FP16 [1,640,640,3]`（2,457,600 字节），输出全部是 `FP16`，而 20×20 和单通道输出的 `size_with_stride` 大于 `size`（通道补 8、宽度补 8 的原生布局）。桥接 v1 提交 UINT8 NHWC，运行库在 CPU 上把 1,228,800 个像素转成 FP16 再复制进 DMA 内存；取输出时又把 1,218,000 个 FP16 转成 float32 并重排成 NCHW，然后再被 numpy 复制一次。两段转换都是运行库的单线程标量代码，吞吐约 100 M 元素/s。

## 3. 优化原理

1. **一次绑定，反复使用**：创建会话时用 `rknn_create_mem` 为输入和每路输出分配 DMA 内存（大小取 `size_with_stride`），用 `rknn_set_io_mem` 绑定一次。之后每帧只需写输入缓冲、调 `rknn_run`、读输出缓冲，没有 `inputs_set`/`outputs_get`/`outputs_release`。
2. **按模型的原生格式写输入**：`rknn_query(NATIVE_INPUT_ATTR)` 告知运行库期望的格式。宿主 kernel [pre-letterbox-neon](04-pre-letterbox-neon.md) 直接把 letterbox 结果打包为 FP16 NHWC（或原生 NC1HWC2）写进输入缓冲，`pass_through=1`，运行库不再转换。uint8→fp16 是精确转换，归一化 `/255` 已由编译器折进第一层卷积（PERF_DETAIL 中 InputOperator 只有 7 µs，没有归一化节点），所以写入的是 0～255 的原始像素值。为防这一判断在别的模型上不成立，桥接保留 `input_mode=1`：声明 UINT8 NHWC、让运行库在 `rknn_run` 内转换，可在 L2 中与原生路径逐位对比。
3. **按原生布局读输出**：用 `rknn_query(NATIVE_OUTPUT_ATTR)` 绑定输出，fp16 数据保持 NC1HWC2（或带步长的 NCHW）布局；[post-dfl-nms-neon](03-post-dfl-nms-neon.md) 的张量视图直接在这两种布局上做扫描和解码，不需要先转成 float32 NCHW。质量验收时才用 `yolo_view_to_nchw_f32` 导出 float32 数组（fp16→fp32 精确，因此与 v1 的 `want_float` 结果应逐位相同，这是 L2 的硬检查）。
4. **可配置缓存属性**：`mem_flags=1` 时输入用 `rknn_create_mem2(..., NON_CACHEABLE)` 分配，宿主顺序写入时可省掉运行库的 cache flush；输出保持可缓存以便 NEON 读取。两种配置都在 L2 扫描中。
5. **核掩码参数化**：`core_mask` 接受 0/1/2/3/4/7，为 [07](07-npu-core-mask-sweep.md) 提供手段；默认 7 与首轮相同。

## 4. 不变量

- 模型文件、输入字节（letterbox 后的 RGB uint8）、输出语义不变。零拷贝路径导出的 float32 输出必须与 v1 `want_float` 逐位相同（同一 .rknn、同一输入）。
- 保留 `io_mode=0`：同一二进制里复现 v1 的调用序列，作为 A/B 的对照。
- 计时口径：`letterbox_ms`、`pack_ms`、`run_ms`、`post_ms` 分别记录；`perf_run_us` 仍来自 `RKNN_QUERY_PERF_RUN`。详细模式与 v1 一样把 PERF_DETAIL 原始字节写到 `perf_path`。

## 5. 证据边界

| 检查 | 结果 |
|---|---|
| C++ 对厂商头文件 `rknn_api.h`（2.3.2）语法检查 | 通过（clang -std=c++17） |
| ctypes 结构与 C 结构布局 | `tests/test_abi_layout.py` 通过 |
| 板端编译、`rknn_set_io_mem` 行为、原生属性 | **未执行** |
| 时间收益 | **未测量**；首轮口径下的上限是 26.9 ms |

## 6. 局限与下一步

- 若 `NATIVE_INPUT_ATTR` 报告的格式不在支持列表（NHWC fp16/u8、NC1HWC2 fp16、NCHW fp16/f32）内，会话创建会失败并给出格式名；此时改用 `input_mode=1`。
- 输出缓存失效由运行库负责；如需 `RKNN_FLAG_DISABLE_FLUSH_OUTPUT_MEM_CACHE` 一类的进一步优化，必须配合 `rknn_mem_sync`，本轮不做。
- 零拷贝对多线程/多上下文吞吐场景的收益另行评估，本轮只看单请求延迟。
