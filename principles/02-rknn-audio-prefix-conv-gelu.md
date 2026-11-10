# 02 · rknn-audio-prefix-conv-gelu:音频编码器卷积前端卸载到 NPU

- 候选:[rknn-prefix-bigvgan-tile4/candidate.json](../experiments/rknn-prefix-bigvgan-tile4/candidate.json)(组合路径) · 过程:[design.md](../experiments/rknn-prefix-bigvgan-tile4/design.md)
- 子图导出:[prepare_conv_frontend.py](../tools/npu/prepare_conv_frontend.py)、[prepare_mnn_prefix.py](../tools/npu/prepare_mnn_prefix.py)、[convert_rknn.py](../tools/npu/convert_rknn.py)
- 宿主适配器:[omni_rknn_prefix.hpp](../src/mnn/overlays/rknn-audio-prefix-conv-gelu/omni_rknn_prefix.hpp) · 路由:[omni.cpp](../src/mnn/omni.cpp)(`OMNI_RKNN_AUDIO`)
- 子图合同:[conv2_gelu.contract.json](../reports/historical/evidence/subgraphs/conv2_gelu.contract.json)

## 1. 子图与位置

Qwen2.5-Omni 的音频编码器(Whisper 结构)以两层一维卷积开头:

| 层 | 结构 | 每输出帧 MAC |
|---|---|---|
| conv1 + GELU | Conv1d 128→1280, K=3, stride 1, pad 1 | 1280×128×3 ≈ 0.49 M |
| conv2 + GELU | Conv1d 1280→1280, K=3, **stride 2**, pad 1 | 1280×1280×3 ≈ 4.9 M |

之后是 permute、位置编码和 32 层 Transformer。MNN 部署图里这个前端的权重是 4-bit 量化的,编码器整体在 OpenCL 上运行。历史热点表中 `media_encoder` 模块单次请求 4.73 s(宿主 callback 口径),但前端两层卷积在算子排名里**没有被单独列出**(排名中的 Convolution 行都是 fc 的 matmul 转换),所以没有"原 OpenCL 卷积 kernel vs NPU"的隔离数据;这一候选的依据是结构(稠密卷积、通道数固定、激活可融合)和系统路径 A/B。

## 2. 优化原理

### 2.1 为什么是 NPU

RK3588 的 NPU 对稠密卷积有专用的 MAC 阵列,并把 Conv+GELU 编译成一个融合单元(`ConvExGelu`)。conv2 每 256 帧输出需要 1.26 G MAC,GPU 上要和 LLM、BigVGAN 争同一块 Mali;搬到 NPU 相当于给请求链路加了一个此前空闲的执行单元。Transformer 部分长度动态、算子种类多,不适合 NPU 的静态图,所以**只切前端两层**,边界放在 permute 和位置编码之前(MNN 图中 `/Mul_3_output_0`),编码器尾部用 CPU 的 MNN 模块继续执行。

### 2.2 固定形状分块:256 帧输出 + 2 帧 halo

RKNN 图需要静态 shape,而 mel 长度随音频变化。做法是按**输出 256 帧**切 tile:

- conv2 输出第 j 帧依赖 conv1 输出 2j−1…2j+1,conv1 输出第 i 帧依赖 mel i−1…i+1,所以 conv2 输出 j 的感受野是 mel 2j−2…2j+2,半径 2。
- 一个 tile 的输入窗口因此是 2×256+4 = **516 帧**,起点 `2·start − 2`。
- 图内 conv2 输出 258 帧,取 `[1:257]`,丢掉两端受窗口人工零填充影响的帧。

这样相邻 tile 的输出拼起来与整段计算一致,不需要任何跨 tile 状态。

### 2.3 有效性掩码:在 tile 内重现原图的零填充

原图里 conv2 对 conv1 的输出做 pad=1 的零填充,也就是说**语句范围之外的 conv1 输出必须恰好是 0**。但 tile 窗口里语句之外的位置虽然 mel 是 0,conv1(0)+bias 经过 GELU 后并不是 0。所以图里加了第二个输入 `true_valid`(516 维 0/1),在 conv1 的 GELU 之后乘一次,把语句外的位置压成 0,再送 conv2。这一步让 tile 计算在语句首尾也与原图等价。

seam 测试(FP32,n∈{1,2,3,199,200,201,511,512,513,1027} 及 3 段真实音频)的 tile 拼接误差 max_abs < 2e-4。

### 2.4 与部署 MNN 对齐的权重和激活

有两份导出:`prepare_conv_frontend.py` 用原始 HF FP32 权重和精确 GELU;实际部署的 `audio_conv_prefix_mnn4bit` 用 `prepare_mnn_prefix.py`,**权重取 MNN 部署图解码出来的 4-bit 权重,GELU 用 tanh 近似**,使 NPU 看到的有效权重和激活定义与部署 MNN 相同。RKNN 以 FP16 编译(`do_quantization=False`,optimization_level 3,RKNN 2.3.2)。

### 2.5 NPU 内的时间分布(tile_0 逐层 profile)

| 层 | 时间 |
|---|---:|
| conv1 + GELU(ConvExGelu) | 1.90 ms |
| 掩码 Mul | 1.07 ms |
| conv2 + GELU(ConvExGelu) | 16.32 ms |
| Slice | 0.41 ms |
| 合计 | 19.9 ms |

conv2 占 82%;掩码乘法占 5%,是可以再省的(例如改成 conv1 偏置侧的处理或在宿主侧清零)。运行时宿主侧每条请求的阶段:`run` 12–35 ms、`inputs_set` 暖态 ~0.6 ms(冷启动 16 ms)、`outputs_get` 1–8 ms、拷贝 0.3–1.2 ms;超过 511 帧 mel 的音频切成 2 个 tile 串行执行。

## 3. 数值:为什么输出会变

这个候选**不是**位级保持的优化,有三处数值路径变化:

1. NPU 的 FP16 卷积累加顺序和精度,与 MNN OpenCL 低内存 4-bit 卷积路径不同;
2. 编码器尾部从 OpenCL 迁到 CPU(`Precision_Normal`, `Memory_Low`);
3. 前端的量化权重经过"解码→FP16 重量化"两次舍入。

embedding 的微小差异经过 LLM 解码的 argmax 会放大成不同的 token,再传导成不同的语音 token 和波形。所以"模型文件相同"不等于"输出相同",这类候选的验收不能用位级比较,应当用容差 + 答案质量,或者先做消融。

## 4. 证据边界

random10 组合路径 A/B(与 tile4 一起启用,原板裸机,每样本 1 对):10 对中 9 对音频特征有效,**有效 9 对总耗时 1263.1 → 1158.9 s(−8.25%)**;严格同输出只有 1/10 对;第 5、9 条候选更慢,因为输出分叉后生成到 1024 token 上限;第 10 条两侧编码器都因音频超过 1500 个位置报 `Broad cast error`,与本候选无关。没有把 RKNN 前缀和 tile4 的贡献分开,也没有做答案质量评测。

结论:耗时差异只是两条系统路径的描述性比较,**输出保持验收未通过**。

## 5. 局限与引出的下一步

- 多 tile 串行跑在一个 NPU 核上,RK3588 另外两个核空闲;输出每次 `want_float` 分配并转换;输入 FP32 由运行时再转 FP16。这些是 [04 · rknn-audio-prefix-v2](04-rknn-audio-prefix-v2.md) 的改动点。
- 要重新定位到底是哪一处数值变化造成了 token 分叉,需要固定其他两处分别比较:RKNN 前缀输出 vs 原 OpenCL 前端输出、CPU 尾部 vs OpenCL 尾部。
- 超过 1500 位置的长音频需要在上游分段,这是两条路线共有的问题。
