# 07 · npu-core-mask-sweep：三核模式是否真的让卷积并行

- 候选：[candidate.json](../experiments/npu-core-mask-sweep/candidate.json) · 过程：[design.md](../experiments/npu-core-mask-sweep/design.md)
- 手段：[rknn_bridge_v2.cpp](../src/apps/rknn_bridge_v2.cpp) 的 `core_mask` 参数 + [l2_compare_bridges.py](../tools/verify/l2_compare_bridges.py) 的 `--config name:io,in,out,core`
- 父候选：首轮桥接固定 `core_mask = 7`（`RKNN_NPU_CORE_0_1_2`）

## 1. 观察

最终模型 PERF_DETAIL 的 `WorkLoad(0/1/2)` 列：57 个 `ConvExSwish`、8 个 `Split`、3 个 `MaxPool`、2 个 `Resize` 全部为 `100%/0%/0%`；而 `Add`、`Concat`、无激活的 `Conv`（检测头 1×1）、`ConvClip` 为约 `33%/33%/33%`。占节点时间 53% 的卷积只在核 0 上执行。

## 2. 假设与反例

假设 H1：对本模型，`core_mask=7` 相比单核 `core_mask=1` 的 `rknn_run` 时间差异只来自 Add/Concat/1×1 Conv/ConvClip 等约 8.5 ms 的节点，卷积主体不受影响。
假设 H2：`RKNN_NPU_CORE_AUTO` 不会改变卷积的分配。
否定 H1 的证据：单核下 ConvExSwish 节点时间显著高于三核，或三核下 WorkLoad 出现非 100/0/0 的卷积。

## 3. 原理与意义

如果 H1 成立，模型本身的 NPU 时间不随核数变化，三核的意义在于**吞吐**（三个独立 context 各绑一核并发处理三张图）而不是单帧延迟；延迟优化只能靠减少节点工作量（01/05/06）和宿主侧开销（02/03/04）。如果 H1 不成立，说明融合卷积在某些条件下可以多核拆分，值得查找条件。这个实验不改模型、不改数值，只改运行时参数。

## 4. 不变量

同一 .rknn、同一输入、同一桥接二进制；输出必须逐位相同（多核拆分不改变 FP16 数值），L2 脚本会检查。

## 5. 证据边界

尚无任何板端测量；本候选只有观察（第 1 节）和实验设计。

## 6. 下一步

L2 默认扫描已包含 `zerocopy_core0`（core_mask 1）与 `zerocopy_auto`（core_mask 0）；在详细模式下再各采一份 PERF_DETAIL 比较 WorkLoad 列。
