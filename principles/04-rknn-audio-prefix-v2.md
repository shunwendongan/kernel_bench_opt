# 04 · rknn-audio-prefix-v2:宿主侧开销与 NPU 核利用率

- 候选:[candidate.json](../experiments/rknn-audio-prefix-v2/candidate.json) · 过程:[design.md](../experiments/rknn-audio-prefix-v2/design.md)
- 源码:[omni_rknn_prefix.v2.hpp](../src/mnn/omni_rknn_prefix.v2.hpp)(覆盖原 `omni_rknn_prefix.hpp`,类名和接口不变,`omni.cpp` 无改动)
- 父候选:[02 · rknn-audio-prefix-conv-gelu](02-rknn-audio-prefix-conv-gelu.md)

## 1. 位置

NPU 上的 `.rknn` 图、tile 合同(516 帧窗口 → 256 帧输出)、掩码构造都不变。v2 只改宿主侧的适配器:怎么准备输入、怎么取输出、用几个 NPU 核。

## 2. 原实现的成本

random10 运行中每条请求记录了 `OMNI_RKNN_REQUEST` 的阶段时间,前缀总耗时 15–54 ms:

| 阶段 | 观测 | 原因 |
|---|---|---|
| `run_us` | 12–35 ms | NPU 执行,单核;conv2 占一个 tile 的 82% |
| `outputs_get_us` | 1–8 ms | 每个 tile `want_float` 由运行时分配缓冲并做 FP16→FP32 + 布局转换,再 `release` |
| `inputs_set_us` | 暖态 ~0.6 ms,冷启动 16 ms | FP32 输入由运行时转 FP16 并重排 |
| `copy_release_us` | 0.3–1.2 ms | 把 `[1280,256]` 的 tile 输出 memcpy 进 `[1280,s]` |
| 多 tile | frames=701 → calls=2,run 28–35 ms | 两个 tile 在同一个核上**串行** |

RK3588 有 3 个 NPU 核,v1 没有绑定核心,也没有并发。

## 3. 优化原理

### 3.1 预分配 IO(默认开启,数值无关)

tile/mask/输出缓冲做成上下文成员常驻,`rknn_outputs_get` 用 `is_prealloc=1` 写进我们的缓冲,去掉每个 tile 的 malloc/free 和 release 时的释放。输出仍由运行时转 FP32,所以数值不变。

### 3.2 多 tile 并发:一核一上下文

`rknn_dup_context` 复制出的上下文共享权重但各自独立执行;第 i 个上下文用 `rknn_set_core_mask` 绑到 NPU core i,tile i 交给第 i % N 个上下文,各线程写 `dst` 中互不重叠的列区间。长音频的 2 个 tile 从串行 28–35 ms 变成并行,理论上接近单 tile 的时间。每个 tile 仍是同一张图在一个核上的一次完整 `rknn_run`,计算本身没变。

### 3.3 单上下文多核(可选)

`OMNI_RKNN_CORE_MASK=012` 让运行时把一张图切到 3 个核上。卷积按输出通道切分理论上不改变每个输出的累加,但这取决于运行时实现,所以必须用自检确认。

### 3.4 FP16 输入(可选)

宿主在构造 516×128 窗口时本来就要逐元素转置,v2 顺手做 RNE 舍入直接写成 FP16,输入字节减半,运行时不再转换。舍入规则若与运行时的转换不同就会改变 bit,因此默认关闭,由自检把关。

### 3.5 位级自检与自动回退

`OMNI_RKNN_SELFCHECK=1` 时,首条请求的每个 tile 先用 v1 设置(FP32 输入、上下文 0)算一次作为 golden,再用 v2 设置算一次,逐 float 比较 bit;对外输出 golden。任何不一致都打印 `OMNI_RKNN_SELFCHECK ... result=FALLBACK_V1`,之后整个进程生命周期回到 v1 设置。这是把"可能改变数值的加速选项"变成"要么位级一致、要么不用"的闸门。

## 4. 边界

- v2 不解决父候选的根本问题:RKNN 路线与原 MNN OpenCL 编码器输出不一致。两边的 FP16 累加方式本来就不同。
- 各选项全部不设时,运行时行为与 v1 相同(只多了缓冲复用)。
- 所有改动都没有在板上运行;预期收益是 `outputs_get`/拷贝的 1–9 ms 和多 tile 请求的 run 时间减半,以 `OMNI_RKNN_REQUEST ... adapter=v2` 日志为准。
