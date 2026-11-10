# RKNN 音频前缀适配器 v2：设计与交付

候选身份：[candidate.json](candidate.json)。源码：[overlays/rknn-audio-prefix-v2/omni_rknn_prefix.hpp](../../src/mnn/overlays/rknn-audio-prefix-v2/omni_rknn_prefix.hpp)。它覆盖部署树中的 `omni_rknn_prefix.hpp`，类名和接口不变，`omni.cpp` 不需要改动。

优化原理：[principles/04-rknn-audio-prefix-v2.md](../../principles/04-rknn-audio-prefix-v2.md)。

## 观察

历史 random10 运行记录了每条请求的 `OMNI_RKNN_REQUEST`。前缀总耗时为 15–54 ms：

- `run_us` 为 12–35 ms，是主体。
- `outputs_get_us` 为 1–8 ms，每次都分配缓冲并转换为 FP32。
- 冷启动时 `inputs_set_us` 可达 16 ms。
- 音频超过 512 帧 mel 时会切成 2 个 tile，在同一 NPU 核上串行执行（例如 frames=701、calls=2、run 28–35 ms）。

RK3588 有 3 个 NPU 核，v1 没有绑定核心，也没有并行。

## 变换

| 选项 | 作用 | 数值影响 |
|---|---|---|
| 预分配 IO（默认开启） | tile/mask/输出缓冲常驻，`is_prealloc=1`，每个 tile 不再 malloc/free | 无 |
| `OMNI_RKNN_PARALLEL_TILES=2/3` | 用 `rknn_dup_context` 复制上下文，第 i 个上下文绑定 NPU core i，多个 tile 并发执行 | 每个 tile 仍是同一图的一次独立 `rknn_run`；用 SELFCHECK 确认 |
| `OMNI_RKNN_CORE_MASK=012` 等 | 单上下文的多核/指定核心 | 多核切分可能改变累加次序，必须用 SELFCHECK 确认 |
| `OMNI_RKNN_INPUT_FP16=1` | CPU 一次完成转置和 RNE 舍入，传 FP16 输入，减半输入字节 | 舍入可能与运行时自身的转换不同，默认关闭 |
| `OMNI_RKNN_SELFCHECK=1` | 首条请求逐 tile 用 v1 设置重算一次并做位级比较；不一致时打印结果，并在之后一直回退到 v1 设置 | 输出始终采用 v1 设置的结果 |

## 边界

- v2 **不解决** RKNN 路线和原 MNN OpenCL 编码器之间的输出不一致问题。RKNN FP16 与 OpenCL 低内存卷积的累加方式本来就不同。该路线在 random10 上严格同输出只有 1/10，这个结论不变。
- 历史第 10 条样本两侧都在编码器内报 `Broad cast error, dim1 = 1500, dim2 = 2251`。原因是音频超出编码器 1500 个位置的上限，与本适配器无关，需要在上游分段。
- 板端先用 `OMNI_RKNN_SELFCHECK=1` 跑一条样本，确认输出 `result=PASS`，再比较 `OMNI_RKNN_REQUEST ... adapter=v2` 的 total_us。
