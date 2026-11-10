# 优化原理文档

这个目录为分支 `Qwen2.5-Omni-3B_VCB-Bench` 上的每一个候选算子 / kernel / 子图保存一份**原理说明**:它改了什么、为什么这样改会更快、哪些东西必须保持不变、哪些结论是实测、哪些是推断。

`experiments/<candidate>/design.md` 记录的是实验过程(观察、假设、评测计划、结果与决策),`candidate.json` 记录身份与哈希,`benchmarks/`、`reports/` 记录证据。此前这些文件只在"观察与假设"里用一两句话带过优化动机,没有专门讲原理的文档;2026-11-10 起统一补齐在这里,新候选也按同样格式新增。

## 索引

| # | 候选 | 类型 | 一句话原理 | 状态 | 过程与证据 |
|---|---|---|---|---|---|
| 01 | [bigvgan-convtranspose-tile4](01-bigvgan-convtranspose-tile4.md) | OpenCL kernel | stride-5 转置卷积按相位分解,同相位的 4 个时间位置共用一次权重读取 | 板端 L1/L2 通过,同步构建 L3 通过(1 样本 3 对);限 `/ups.0.0`、T∈{52,120,136} | [design](../experiments/bigvgan-convtranspose-tile4/design.md) · [证据](../benchmarks/history/20261106-bigvgan-convtranspose/README.md) |
| 02 | [rknn-audio-prefix-conv-gelu](02-rknn-audio-prefix-conv-gelu.md) | NPU 子图 | 音频编码器前端 Conv1→GELU→Conv2→GELU 以固定 256 帧 tile + 2 帧 halo + 有效性掩码搬到 RK3588 NPU,其余编码器留在 CPU | 系统路径 A/B 9 对耗时 −8.25%,但严格同输出只有 1/10,未通过输出保持验收 | [design](../experiments/rknn-prefix-bigvgan-tile4/design.md) · [证据](../benchmarks/history/20261109-omni-combined-random10/README.md) |
| 03 | [bigvgan-deconv1d-v2](03-bigvgan-deconv1d-v2.md) | OpenCL kernel | 通用 1-D 转置卷积:时间×输出通道寄存器分块、每个 ic block 只读一次输入窗口、work group 沿时间维展开以复用权重;乘加次序与原 kernel 相同 | 源码交付,待板端验收 | [design](../experiments/bigvgan-deconv1d-v2/design.md) · [部署](../docs/V2_DEPLOY.md) |
| 04 | [rknn-audio-prefix-v2](04-rknn-audio-prefix-v2.md) | NPU 宿主适配器 | 预分配 IO、多 tile 在 3 个 NPU 核并发、可选 FP16 输入,首条请求与 v1 位级自检 | 源码交付,待板端验收 | [design](../experiments/rknn-audio-prefix-v2/design.md) |
| 05 | [predit-sync](05-predit-sync.md) | 系统级(非算子) | PreDiT 输出交给 FP32 DiT 前显式等待 GPU 完成,消除跨命令队列的可见性竞争 | 同一样本 8 次波形一致;根因未完全确认 | [SYNC_VARIANT](../docs/SYNC_VARIANT.md) · [patch](../experiments/predit-sync/predit-sync.patch) |

## 每份文档的结构

1. **算子与位置**:结构、shape、在热点表中的位置。
2. **原实现的成本**:数据搬运和计算的账,指出瓶颈。
3. **优化原理**:逐条说明变换及其为什么有效。
4. **不变量**:为了结果可比,哪些东西一个字节都不能动;如果数值会变,说明为什么会变。
5. **证据边界**:哪些是板端实测、哪些是非目标设备的冒烟、哪些只是推断。
6. **局限与引出的下一步**。

## 写新候选时

- 先写原理再写实验,实验设计应能否定原理中的假设。
- 数字都标注计时口径(host enqueue→finish / 设备 event / 宿主 callback / 完整请求)和设备;非目标设备上的数字只能说明方向。
- 热点表的来源是 [mnn_operator_rankings.csv](../reports/historical/evidence/mnn_operator_rankings.csv)(宿主同步 callback 口径,含诊断开销),引用时注明。
