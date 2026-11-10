# rknn-audio-prefix-conv-gelu：音频编码器卷积前端的 NPU 子图

候选身份：[candidate.json](candidate.json)。源码：[宿主适配器](../../src/mnn/overlays/rknn-audio-prefix-conv-gelu/omni_rknn_prefix.hpp)、[路由](../../src/mnn/omni.cpp)、导出脚本 [prepare_conv_frontend.py](../../tools/npu/prepare_conv_frontend.py) / [prepare_mnn_prefix.py](../../tools/npu/prepare_mnn_prefix.py) / [convert_rknn.py](../../tools/npu/convert_rknn.py)。

优化原理：[principles/02-rknn-audio-prefix-conv-gelu.md](../../principles/02-rknn-audio-prefix-conv-gelu.md)。

这份记录是 2026-11-10 补建的：此前该适配器只作为组合候选 [rknn-prefix-bigvgan-tile4](../rknn-prefix-bigvgan-tile4/design.md) 的"复用部分"出现，没有自己的候选身份。所有测量都早于本记录，没有新增实验。

## 观察与假设

音频编码器前端两层卷积通道数固定、计算稠密，适合 NPU 的静态图；Transformer 部分长度动态，留在 MNN。假设把前端搬到 NPU 能降低请求延迟。

## 变换与不变量

- 图：Conv1→GELU→Conv2→GELU，权重取 MNN 部署图解码后的值，GELU 用 tanh 近似，RKNN FP16 编译。
- 分块：输出 256 帧 / tile，输入窗口 516 帧（2 帧 halo），`true_valid` 掩码把语句范围外的 conv1 输出压为 0 以重现原图零填充；seam 测试 max_abs < 2e-4（FP32）。
- 数值**不**保持：NPU FP16 累加与 OpenCL 低内存卷积不同，编码器尾部迁到 CPU。

## 结果与决策

只在组合路径 random10 中评测：有效 9 对总耗时 −8.25%，严格同输出 1/10。输出保持验收未通过，保留实现与数据。详见组合候选的 [design.md](../rknn-prefix-bigvgan-tile4/design.md) 与 [运行证据](../../benchmarks/history/20261109-omni-combined-random10/README.md)。

## 后续方向

分离消融（同一编码路线比较 tile0/tile4；固定 BigVGAN 比较原编码器与 RKNN+CPU 路径）；宿主侧工程改进见 [rknn-audio-prefix-v2](../rknn-audio-prefix-v2/design.md)。
