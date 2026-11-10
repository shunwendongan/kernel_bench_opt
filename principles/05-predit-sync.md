# 05 · predit-sync:跨命令队列的输出可见性

- 变更:[predit-sync.patch](../experiments/predit-sync/predit-sync.patch)(`src/mnn/omni.cpp` 中 8 行) · 说明:[SYNC_VARIANT.md](../docs/SYNC_VARIANT.md)
- 类型:系统级候选,不是算子优化;放在这里是因为它决定了 tile4 完整请求 A/B 能否成立。

## 1. 位置

token2wav 链路里,PreDiT 在 OpenCL(FP16)后端执行,DiT 单独使用 FP32 后端。PreDiT 的三个输出(code_embeds、rope、mask)直接作为 DiT 的输入。

## 2. 观察到的问题

默认异步构建的完整请求 A/B 中,旧 tile0 与新 tile4 的 text token 和 226 个 speech token 完全相同,但 106,560 个波形样本不同(max abs 0.66,relative RMSE 0.34),差异集中在波形前段;正式对数为 0。同一 mel 的三形状 BigVGAN 回放却位级一致,说明差异不在 ConvTranspose 的数学里。

## 3. 原理

MNN 的 OpenCL 后端是异步提交的:一次 `forward` 返回时,命令可能还在队列里。PreDiT 和 FP32 DiT 若使用不同的运行时/命令队列,DiT 侧读取 PreDiT 输出时没有隐式的队列间依赖,**可能读到尚未写完的数据**;kernel 快慢不同(tile0 vs tile4 改变了 GPU 上的时序)会改变竞争的结果,表现为两侧波形不同。

补丁在 PreDiT 返回后对每个输出调用 `tensor->wait(MAP_TENSOR_READ, true)`,强制等待该张量的设备写入完成并对宿主/其他队列可见,再交给 DiT。这是最小的、语义明确的同步点;它不改变任何算子的数值。

## 4. 证据边界

- 同步构建、同一样本、相同种子:8 次暖机/测量的 text/speech token 与波形全部一致;3 对正式请求中位 94.06 → 91.06 s(−3.20%),该收益属于 tile4,两侧都开启同步。
- 只验证了一个样本和一台板。同步"消除了观察到的差异"不等于"证明了根因":要确认,需要在异步构建上复现差异并用 event 时间线证明读写重叠。
- 开关按环境变量**是否存在**判断,`OMNI_PREDIT_SYNC=0` 仍启用,禁用需 `unset`。

## 5. 代价与下一步

显式等待会在 PreDiT→DiT 交接处引入一次宿主同步,可能损失一点流水线重叠。下一步是用 MNN 的跨后端依赖(或把 PreDiT/DiT 放到同一队列)替代宿主等待,并在多样本上确认异步版本的差异是否全部由此造成。
