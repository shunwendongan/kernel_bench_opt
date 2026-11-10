# 候选集 v2：部署与板端测试

本轮交付两个候选的源码：

- [bigvgan-deconv1d-v2](../experiments/bigvgan-deconv1d-v2/design.md)：通用 1-D ConvTranspose OpenCL kernel，覆盖 BigVGAN 全部 `/ups.*`，任意 T，与原 kernel 位级一致。
- [rknn-audio-prefix-v2](../experiments/rknn-audio-prefix-v2/design.md)：RKNN 音频前缀适配器的工程优化。

**两者都还没有在板上运行。** 默认配置下行为与基线相同，只有设置环境变量后才启用。

## A. 本仓库固定的 MNN 7e3c6fb（推荐）

按 [BUILD.md](BUILD.md) 正常构建即可。`scripts/prepare_mnn.py` 默认选择候选集 `v2`（`--candidate-set tile4` 回到历史候选集），会覆盖以下文件：

| 仓库文件 | MNN 树中的位置 |
|---|---|
| `src/mnn/overlays/bigvgan-deconv1d-v2/DeconvBufExecution.cpp` | `source/backend/opencl/execution/buffer/DeconvBufExecution.cpp` |
| `src/mnn/overlays/bigvgan-deconv1d-v2/DeconvBufExecution.v2.hpp` | `source/backend/opencl/execution/buffer/DeconvBufExecution.v2.hpp` |
| `kernels/bigvgan-deconv1d-v2/bigvgan_deconv1d_v2_source.hpp` | `source/backend/opencl/execution/buffer/bigvgan_deconv1d_v2_source.hpp` |
| `kernels/bigvgan-convtranspose-tile4/bigvgan_deconv_source.hpp` | 同目录（tile4，保留用于 A/B） |
| `src/mnn/overlays/rknn-audio-prefix-v2/omni_rknn_prefix.hpp` | `transformers/llm/engine/src/omni_rknn_prefix.hpp` |

已用旧候选准备过的 MNN 树可以直接升级，脚本接受上一版 overlay 的哈希。构建时还会生成 `bench_deconv_v2`。

```bash
scripts/run_v2_board.sh l1            # 位级门 + TT/OCB/LWS 扫描，输出 runs/v2-l1-*/SUMMARY.md
scripts/run_v2_board.sh l1-capture    # ups.0 真实 capture（需要先按 REPRODUCE.md 第2–3步采集）
BIGVGAN_MNN=/path/bigvgan.mnn scripts/run_v2_board.sh l2
scripts/run_v2_board.sh env           # 打印完整请求 A/B 新侧的环境变量
```

## B. 移植到其他 MNN 版本

改动集中在 `DeconvBufExecution::onEncode` 中 `useV2` 这个分支，只依赖以下几样：

- `runtime->buildKernelFromSource`
- `localWS3DDefault`
- `mResource->mFilter` / `mResource->mBias`（`CONV2D_FILTER` 打包）
- `openCLBuffer(...)`

如果目标版本的 Deconv buffer 权重打包仍是 `[Ci, Co/4, K, 4]`(由 `convertToNC4HW4Buffer(..., CONV2D_FILTER)` 生成)，可以直接合入这个分支和头文件。移植后先跑 `bench_deconv_v2`，用该版本的 `deconv_2d.cl` 作为 baseline，位级门通过后再上线。

## C. 其他引擎（直接调用 kernel）

`kernels/bigvgan-deconv1d-v2/bigvgan_deconv1d_v2.cl` 是独立文件。

- **编译宏**：`-DFLOAT=half -DFLOAT4=half4 -DMNN_SUPPORT_FP16`（FP32 时用 `float`），以及 `-DDC_K -DDC_S -DDC_PAD -DDC_CI4 -DDC_CO4 -DDC_TT -DDC_OCB`，可选 `-DRELU/-DRELU6`。
- **参数**：`(int gws0, int gws1, int gws2, input, weights, bias, output, int T)`。
- **全局尺寸**：`{CO4/OCB, ceil(T/TT), S}`，需要向上取整到 LWS 的整数倍，kernel 内有边界检查。默认 LWS 为 `{4,8,1}`。
- **数据布局**：
  - 输入 `[CI4, T, 4]`
  - 权重 `[CI4*4, CO4, K, 4]`，元素 `(ci, co, k)` 位于 `((ci*CO4 + co/4)*K + k)*4 + co%4`
  - 输出 `[CO4, S*T, 4]`
  - bias `[CO4*4]`
- **约束**：batch=1、groups=1、`K−2P==S`。

## 环境变量一览

| 变量 | 默认 | 说明 |
|---|---|---|
| `OMNI_BIGVGAN_DECONV_V2` | 未设置 | `1` 启用 v2（`OMNI_BIGVGAN_DECONV_TILE=4` 时 tile4 优先） |
| `OMNI_BIGVGAN_DECONV_V2_NODES` | `ups` | `ups` / `all` / 逗号分隔的精确节点名 |
| `OMNI_DECONV_V2_TT` / `_OCB` | 4 / 2 | 时间分块 / 输出通道块数 |
| `OMNI_DECONV_V2_LWS` | `4,8,1` | 固定 LWS，或 `tune` 使用 MNN 调优器 |
| `OMNI_DECONV_V2_K<k>` | — | 按 kernel 尺寸覆盖，格式 `TT:OCB[:x,y,z]`，由 l1 SUMMARY.md 给出 |
| `BIGVGAN_DISPATCH_LOG` | — | 记录每个节点、每种 T 实际使用的 kernel |
| `OMNI_RKNN_PARALLEL_TILES` | 1 | 2/3：多 tile 请求在 NPU core 0/1/2 上并发 |
| `OMNI_RKNN_CORE_MASK` | 未设置 | `0/1/2/01/012/all/auto` |
| `OMNI_RKNN_INPUT_FP16` | 0 | `1` 使用 FP16 输入 |
| `OMNI_RKNN_SELFCHECK` | 0 | `1` 首条请求与 v1 设置做位级比较，不一致时自动回退 |
