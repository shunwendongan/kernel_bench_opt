# 03 · bigvgan-deconv1d-v2:只改数据搬运,不改一步算术

- 候选:[candidate.json](../experiments/bigvgan-deconv1d-v2/candidate.json) · 过程:[design.md](../experiments/bigvgan-deconv1d-v2/design.md) · 部署:[V2_DEPLOY.md](../docs/V2_DEPLOY.md)
- 源码:[bigvgan_deconv1d_v2.cl](../kernels/bigvgan-deconv1d-v2/bigvgan_deconv1d_v2.cl) · 集成:[overlays/bigvgan-deconv1d-v2/DeconvBufExecution.cpp](../src/mnn/overlays/bigvgan-deconv1d-v2/DeconvBufExecution.cpp) · 对拍/调参:[bench_deconv_v2.cpp](../kernels/bigvgan-deconv1d-v2/bench_deconv_v2.cpp)
- 父候选:[01 · tile4](01-bigvgan-convtranspose-tile4.md);基线仍是 MNN `deconv_2d`

## 1. 算子与位置

同 tile4,但目标扩大到 BigVGAN 全部 6 个上采样反卷积,它们在热点表中合计 18.2 s:

| 节点 | Ci→Co | K / S / P | 热点表 |
|---|---|---|---:|
| /ups.0.0 | 1536→768 | 11 / 5 / 3 | 13.11 s |
| /ups.1.0 | 768→384 | 7 / 3 / 2 | 4.33 s |
| /ups.2.0 | 384→192 | 4 / 2 / 1 | 0.49 s |
| /ups.3.0 ~ /ups.5.0 | 192→96→48→24 | 4 / 2 / 1 | 0.27 s |

三种配置都满足 `K − 2P = S`,输出长度恰为 `S·T`。

## 2. 瓶颈:权重流量,不是算力

ups.0 的权重 25.9 MB 装不进任何 GPU cache。每个 work item 负责 (oc 块, 相位, 一段 q),要沿 1536 个输入通道把自己那几个 tap 的权重流一遍;所有 work item 加起来,

> 权重读取总量 ≈ 25.9 MB × (q-tile 数量) ÷ (同一 work group 内共享同一份权重的 q-tile 数)

T=136、每个 work item 算 4 个 q 时是 34 遍、约 880 MB;原 `deconv_2d` 是 136 遍、3.5 GB。输入只有 1536×136×2 B ≈ 418 KB,能常驻 L2,它影响的是 load 指令数而不是 DRAM。tile4 每个 ic block、每个 tap 读 4 个权重 + 4 个输入换 16 次 mad,**2 mad/load**,Mali 上基本是 load 单元在等。

## 3. 优化原理

### 3.1 通用相位分解

输出 `o = S·q + p`。相位 p 的有效 tap 为

`k = P + p + S·d`, `d ∈ [ −⌊(P+p)/S⌋ , ⌊(K−1−P−p)/S⌋ ]`, 输入 `ix = q − d`

K/S/P 任意,所以三种 ups 配置共用一份 kernel;tile4 中 `lead/first_k` 的写法是它在 K11/S5/P3 下的特例。每个 ic block 内按 d 从大到小(k 降序)累加。

### 3.2 输入窗口复用

同一 ic block 的 ntaps 个 tap 用的是相邻的输入位置:位置 t、tap j 要的是 `ix0 + t + j`。一个 work item 一次把 `TT + ntaps − 1` 个连续输入读进寄存器窗口 `win[]`,所有 tap 都从窗口取。TT=4、3 tap 时输入读取从 tile4 的 12 次降到 6 次。

### 3.3 输出通道分块 OCB

一个 work item 同时算 OCB 个输出通道块,输入窗口再被摊薄 OCB 倍。权重每个通道块独有,这一项不省权重流量,但提高每次 load 换到的 mad 数。

### 3.4 时间分块 TT 可调

一个权重 half4 被 TT 个时间位置复用,权重流量与 TT 成反比;代价是 OCB×TT 个 half4 累加器和并行度下降。几种组合每个 ic block 的账(以 3 tap 相位为例):

| 配置 | 权重 load | 输入 load | mad | mad / load | 主要寄存器(half4) |
|---|---:|---:|---:|---:|---:|
| tile4 | 12 | 12 | 48 | 2.0 | 4 acc + 4 w |
| v2 TT4 OCB1 | 12 | 6 | 48 | 2.7 | 4 acc + 6 win + 4 w |
| **v2 TT4 OCB2(默认)** | 24 | 6 | 96 | 3.2 | 8 + 6 + 8 |
| v2 TT8 OCB1 | 12 | 10 | 96 | 4.4 | 8 + 10 + 4 |
| v2 TT8 OCB2 | 24 | 10 | 192 | 5.6 | 16 + 10 + 8 ≈ 68 个 32 位寄存器 |

TT8 OCB2 账面最好,但在 Mali Valhall 上超过 64 个寄存器会把每核线程数砍半,实际快慢必须在板上扫。所以 TT/OCB 是编译宏,`run_v2_board.sh l1` 扫描后用 `OMNI_DECONV_V2_K<k>=TT:OCB:lws` 按 kernel 尺寸固定。

### 3.5 work group 沿时间维展开(本地冒烟中影响最大)

同一 work group 里的 work item 几乎同时执行。若它们是**同一 oc、不同 q**(LWS `{4,8,1}`),要的权重地址完全相同:一个取回,其余命中 cache,DRAM 上的权重流量再除以 8。反过来 LWS `{32,1,1}` 让 32 个 work item 各读各的权重,M4 上慢 3–8 倍。tile4 的 `{8,4,1}` 介于两者之间。这是公式里的分母项。

### 3.6 内部 / 边缘双路径 + 全编译期常量

第一块要跳过负索引,最后一块要补零 tap,中间块什么检查都不需要。v2 把两种情况拆成两段代码:中间块是一条完全展开、没有分支谓词的直线代码;边缘块只出现在首尾。K/S/P/Ci/Co/TT/OCB 全是宏,循环全展开,地址运算成常量;只有 T 是运行期参数,因此**不再需要 T 白名单**,T108 这类真实末块不会回退。MNN 侧按编译配置缓存 KernelWrap,resize 不重新 JIT。

## 4. 不变量:位级一致的依据

每个输出的运算序列与 `deconv_2d` 相同:bias → ic block 升序 → tap k 降序 → x,y,z,w。`ix<0` 跳过、`ix≥T` 乘 0 再加(保留 −0.0 行为),与 MNN 一致。权重打包字节、布局、FLOAT/FLOAT4 宏全部沿用,所以 precision0/1/2 都适用。

[prove_tap_order.py](../tools/verify/prove_tap_order.py) 对 3 种层配置 × 7 种 TT × T=1..300 的 **3,160,500 个输出位置**逐个比对 (k, ix, 跳过/补零) 序列,全部一致。这是与精度、设备无关的结构证明;FP16 位级一致由板端 L1/L2 再验。

## 5. 证据边界

| 检查 | 结果 | 说明 |
|---|---|---|
| tap 次序穷举 | 通过 | 结构证明 |
| MNN 7e3c6fb 集成 | `prepare_mnn.py` PREPARE_OK;`DeconvBufExecution.cpp`、`omni.cpp` 语法检查通过 | 未在板上编译 |
| Apple M4 FP32 位级 | ups.0 T136/T37/T52、ups.1 T680/T37、ups.2 T2040 全部变体与 deconv_2d 位级一致 | M4 无 fp16,**非目标设备** |
| Apple M4 FP32 速度 | ups.0 T136:deconv_2d 229 ms,tile4 9.85 ms,v2 `tt4_ocb2_lws4,8,1` 6.49 ms;T52:tile4 2.56 ms,v2 2.58 ms | 只说明方向 |
| RK3588 FP16 | **尚未运行** | 由 `scripts/run_v2_board.sh l1/l2` 给出 |

第 2 节的流量模型是推断;Mali-G610 的 cache 更小、带宽更低,输入复用和 LWS 的收益按理会比 M4 上更明显,但以板端扫描为准。

## 6. 局限与下一步

1. 如果板上仍受权重带宽限制,可把整段 T 放进一个 work group 并用 local memory 缓存权重作为 v3;Mali 的 local memory 走普通 cache,收益需实测。
2. 热点表中 `DeconvolutionDepthwise`(抗混叠上采样)合计 31.2 s,超过全部 ups 反卷积,是下一个值得定位的算子。
3. 位级门若在某个形状失败,保留记录,该形状回退 tile4 或基线。
