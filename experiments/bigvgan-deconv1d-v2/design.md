# BigVGAN ConvTranspose v2：设计与交付

候选身份：[candidate.json](candidate.json)。源码：[OpenCL kernel](../../kernels/v2/bigvgan_deconv1d_v2.cl)、[MNN执行器](../../src/mnn/DeconvBufExecution.v2.cpp)、[对拍/调优工具](../../kernels/v2/bench_deconv_v2.cpp)、[板端脚本](../../scripts/run_v2_board.sh)。父候选：[tile4](../bigvgan-convtranspose-tile4/design.md)。

优化原理：[principles/03-bigvgan-deconv1d-v2.md](../../principles/03-bigvgan-deconv1d-v2.md)。

## 从 tile4 出发的观察

tile4 每个 work item 计算同一相位的 4 个时间位置，并共享权重。它仍有三处限制：

1. **每个 tap 都重新读输入**。一个 ic block 有 2–3 个 tap，每个 tap 读 4 个输入 half4，共 8–12 次；实际只用到 4+ntaps−1 个不同的输入。
2. **只覆盖 `/ups.0.0`，并且只覆盖 T=52/120/136**。T108 等真实末块会回退到原 kernel。`/ups.1.0` 在热点表中排第 2 个反卷积（4.33 s / 14 次调用），也没有覆盖。六个 ups 反卷积合计 18.2 s，`/ups.0.0` 只占其中 13.1 s。
3. **Ci/Co/K/S 写死**（`ic < 384`、`192*11`），无法复用到其他层或其他模型。

本地 sweep 还表明（数据见下表），这个算子的瓶颈是**权重流量**：同一权重 quad 在每个 q tile 都要读一次。work group 跨越的时间维越大，同一 oc block 的权重在 cache 中被复用得越多。选错 LWS 会慢 3–8 倍，这比 kernel 内部的写法影响更大。

## 变换

对通用 1-D 转置卷积做相位分解：输出 `o = S*q + p`，相位 p 的有效 tap 为 `k = P + p + S*d`，其中 `d ∈ [−⌊(P+p)/S⌋, ⌊(K−1−P−p)/S⌋]`，对应输入 `ix = q − d`。

- 每个 work item 计算 **TT 个连续 q × OCB 个输出通道块**，属于同一相位。
- 每个 ic block 只读一次长度为 `TT+ntaps−1` 的输入窗口，所有 tap 和所有 OCB 共享它。
- 每个 tap、每个 oc block 读一次权重 quad，由 TT 个位置共享。
- K/S/P/Ci/Co/TT/OCB 是编译期宏，循环全部展开。T 在运行期传入，因此**不需要 T 白名单**。
- 分两条路径：interior（无越界检查）和 edge（处理首块的负索引和末块的零 tap），二者 mad 序列相同。
- 默认 `TT=4, OCB=2, LWS={4,8,1}`，可以按 kernel 尺寸用环境变量覆盖。MNN 内按编译配置缓存 KernelWrap，resize 时不会重新 JIT。

## 不变量（与 MNN `deconv_2d` 位级一致的依据）

- 输入/输出布局 `[C/4,T,4]`、权重 `CONV2D_FILTER` 打包字节、bias 都沿用 MNN 原有的。FLOAT/FLOAT4 由运行时决定（precision0 下为 half 存储、half 累加），没有改为 FP32。
- 每个累加器的运算序列与原 kernel 完全相同：先 bias，然后 ic block 升序 → k 降序 → x,y,z,w。
- `ix<0` 的 tap 跳过，因为 MNN 从 `max(0,…)` 开始。`ix≥T` 的 tap 执行一次乘以 0 的 mad，与 MNN 的 outBoundry 相同，所以 `-0.0` 的行为也一致。
- [prove_tap_order.py](../../tools/verify/prove_tap_order.py) 对 3 组 kernel 配置 × 7 种 TT × T=1..300 做了穷举，共 3,160,500 个输出位置，(k, ix, load/zero) 序列逐项相同。
- 只有满足以下条件才启用 v2：batch=1、groups=1、dilation=1、无 output padding、单一时间轴、`K−2P==S`、输出长度为 `S*T`。其他情况保持原 kernel。

## 交付前检查（非目标设备）

开发机 Apple M4 的 OpenCL 1.2 **不支持 fp16**，所以只能用 FP32 做次序一致性的冒烟测试。下面的数字不代表 Mali-G610 的表现，也不能作为板端收益。

| 检查 | 结果 |
|---|---|
| 位级（FP32） | ups.0 T136/T37/T52、ups.1 T680/T37、ups.2 T2040：v2 全部变体和 tile4 都与 deconv_2d 位级一致 |
| MNN 集成 | 在 MNN 7e3c6fb 上 `prepare_mnn.py` 输出 PREPARE_OK；DeconvBufExecution.cpp 与 omni.cpp 语法检查通过 |
| M4 FP32 ups.0 T136 | deconv_2d 229 ms；tile4 9.85 ms；v2 `tt4_ocb2_lws4,8,1` 6.49 ms |
| M4 FP32 ups.0 T52 | deconv_2d 87 ms；tile4 2.56 ms；v2 `tt4_ocb1_lws8,8,1` 2.58 ms |
| LWS 敏感度（M4, T136, tt4_ocb1） | `8,4,1` 9.2 ms；`32,1,1` 55.7 ms |

## 板端验收（按顺序执行）

```bash
scripts/run_v2_board.sh l1           # 6个ups形状 × T∈{52,108,120,136}：位级门 + TT/OCB/LWS 扫描，生成推荐环境变量
scripts/run_v2_board.sh l1-capture   # ups.0 真实 capture 输入/权重
BIGVGAN_MNN=/path/bigvgan.mnn scripts/run_v2_board.sh l2   # 同一 mel 回放：完整波形位级比较 + dispatch 覆盖
scripts/run_v2_board.sh env          # 完整请求 A/B 的新侧环境
```

接受条件：

- L1 所有候选的 bit_mismatches 均为 0。
- L2 所有 waveform 位级相同，且 dispatch 显示每个 `/ups.*` 节点、每种 T 都是 `v2=true`。
- L3 按 [REPRODUCE.md](../../docs/REPRODUCE.md) 的完整请求协议执行，两侧都使用同步构建，工作量保持一致。

任一项失败都保留记录，并回退到 tile4 或基线。

## 下一步可验证方向

1. 如果板上扫描显示仍受权重带宽限制，可以尝试在 `OCB` 之外再沿时间维把整段 T 放进一个 work group，并用 local memory 缓存权重，作为 v3 候选。Mali 的 local memory 走普通缓存，是否有收益要以实测为准。
2. 热点表中 `DeconvolutionDepthwise`（抗混叠上采样）合计 31.2 s，超过全部 ups 反卷积（18.2 s）。它是下一个值得定位的算子。
