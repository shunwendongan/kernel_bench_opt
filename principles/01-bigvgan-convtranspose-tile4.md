# 01 · bigvgan-convtranspose-tile4:同相位时间分块复用权重

- 候选:[candidate.json](../experiments/bigvgan-convtranspose-tile4/candidate.json) · 过程:[design.md](../experiments/bigvgan-convtranspose-tile4/design.md)
- 源码:[bigvgan_deconv_s5_tiled.cl](../kernels/bigvgan_deconv_s5_tiled.cl) · 集成:[DeconvBufExecution.candidate.cpp](../src/mnn/DeconvBufExecution.candidate.cpp)
- 基线:MNN 7e3c6fb OpenCL Buffer `deconv_2d`(USE_BUFFER, BIAS)

## 1. 算子与位置

BigVGAN 第一个上采样 `/ups.0.0/ConvTranspose_output_0`:Ci=1536、Co=768、K=11、stride=5、pad=3、groups=1,输入 `[1,1536,T]`,输出 `[1,768,5T]`。真实请求里 T 按 mel 分块出现 120(首块)、136(稳态)、52/108/46(末块)。

数学定义:`y[o,t] = b[o] + Σ_i Σ_k x[i,(t+3−k)/5] · w[i,o,k]`,只累计 `(t+3−k)/5` 为非负整数且小于 T 的项。

在早期诊断请求的热点表里(宿主同步 callback 口径、含 profiling 开销),它 14 次调用合计 13.11 s,是 BigVGAN 模块(232 s)中单节点最高的一项,也是整个 talker 侧最高的算子。

## 2. 原实现 `deconv_2d` 的成本

MNN 的 `deconv_2d` 给**每个输出位置、每个 4 通道输出块**一个 work item:GWS = {Co/4=192, 5T, 1}。每个 work item 沿 384 个输入通道块循环,对每个有效 tap 读 4 个权重 half4(对应 4 个输入通道)和 1 个输入 half4,做 4 次 mad。

每个输出位置只有 2–3 个 tap 有效(见下节),所以一个 work item 的账是:384 × (2~3) × (4 次权重 load + 1 次输入 load + 4 次 mad),**每次 load 只换来 0.8 次 mad**。

更关键的是权重流量。5 个相位的有效 tap 合起来正好是全部 11 个 tap,所以**每一个输入时间位置 q 都要把 26 MB 的权重完整流一遍**(1536×768×11×2 B = 25.9 MB),T=136 时所有 work item 的权重读取总量约 3.5 GB。Mali-G610 的 L2 远小于 26 MB,这些读取大部分落到 DRAM 或 L2 反复换入。板端实测 T=136 时 975 ms,与这个结构是一致的。

## 3. 优化原理

### 3.1 相位分解

输出位置 `o = 5q + p`。把 `(o+3−k)/5` 为整数的条件代入,相位 p 的有效 tap 固定为:

| p | tap k | 输入位置 |
|---|---|---|
| 0 | 8, 3 | q−1, q |
| 1 | 9, 4 | q−1, q |
| 2 | 10, 5, 0 | q−1, q, q+1 |
| 3 | 6, 1 | q, q+1 |
| 4 | 7, 2 | q, q+1 |

这与 MNN 原 kernel 内部 `kernel_start/deal_kernel` 的循环推出来的 tap 集合完全一致,只是把"哪个 tap 有效"从运行期判断变成了按相位可预知。

### 3.2 同相位的 4 个时间位置共用权重

对固定的 (oc 块, 相位 p),位置 q、q+1、q+2、q+3 用的是**同一组权重** `w[·, oc, k∈taps(p)]`,只是输入位置依次平移。tile4 让一个 work item 算这 4 个位置:每个 ic block、每个 tap 读 4 个权重 half4 **一次**,然后对 4 个位置各读 1 个输入 half4、各做 4 次 mad。

每个 ic block、每个 tap 的账从 4 × (4W + 1X → 4 mad) 变成 (4W + 4X → 16 mad),**每次 load 换 2 次 mad**,权重总流量降为原来的 1/4(T=136 时约 880 MB)。GWS 相应变为 {192, ⌈T/4⌉, 5},LWS 固定 {8,4,1}。

### 3.3 其他保持不变的部分

输入/输出布局 `[C/4,T,4]`、权重的 `CONV2D_FILTER` 打包字节、bias、FLOAT/FLOAT4 宏(precision0 下 half 存储、half 累加)全部沿用 MNN;没有把累加改成 FP32,也没有融合激活(节点本身没有激活)。

## 4. 不变量:位级一致的依据

FP16 加法不满足结合律,累加次序一变 bit 就变。tile4 对每个输出保留了和 `deconv_2d` 相同的运算序列:bias → ic block 升序 → tap k 降序 → x,y,z,w 四次 mad。两个细节照抄:

- `ix < 0` 的 tap(只出现在 q=0 的 lead 相位)**跳过**,因为 MNN 从 `max(0, …)` 开始循环;
- `ix ≥ T` 的 tap **执行一次乘 0 的 mad**(MNN 的 outBoundry 分支),它影响 `−0.0` 的符号。

因此 tile4 与原 kernel 的差别只在"同一个权重什么时候读、被几个输出用",不在算术上。

## 5. 证据边界

| 层级 | 结果 | 口径 |
|---|---|---|
| L1 单算子(板端) | T52 369.16→30.47 ms,T120 850.64→64.45 ms,T136 975.30→72.88 ms,**12.11–13.38×**;三种 T 输出与原 kernel 位级一致 | 关闭 profiling 的 host enqueue→clFinish,驻留 buffer,预热 3 次、15 对交替 |
| L2 子图(板端) | 同一 mel 回放,目标输入/输出及完整 BigVGAN 波形共 9 项位级一致 | capture 模式,不计时 |
| L3 完整请求(板端) | 默认异步构建波形门失败;PreDiT 同步构建(两侧都同步)中位 94.06→91.06 s,−3.20% | 1 个样本、3 对;T108 回退原 kernel |

局部十几倍和端到端 3% 是不同的计时范围:ConvTranspose 只是 BigVGAN 的一部分,BigVGAN 又只是请求的一部分,其余模块、串行依赖、拷贝和同步仍占用时间。

## 6. 局限与引出的下一步

1. **输入仍按 tap 重复读**:4 个位置 × 2–3 个 tap 要读 8–12 个输入 half4,实际只有 4+ntaps−1 = 5–6 个不同的值。
2. **形状写死**:`ic < 384`、`192*11`、`BIGVGAN_TILE∈{1,2,4}`,只能用于 Ci1536/Co768/K11/S5。
3. **dispatch 白名单** T∈{52,120,136}:真实末块 T108 回退到原 kernel;`/ups.1.0`(热点表第二个反卷积,4.33 s)及 `/ups.2~5` 未覆盖。六个 ups 合计 18.2 s,tile4 只覆盖其中 13.1 s。
4. LWS {8,4,1} 是固定值,没有扫描过"沿时间维展开"的 work group 形状。

这四点正是 [03 · bigvgan-deconv1d-v2](03-bigvgan-deconv1d-v2.md) 的出发点。
