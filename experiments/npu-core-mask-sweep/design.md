# 候选 ID：npu-core-mask-sweep

原理：[principles/07-npu-core-mask-sweep.md](../../principles/07-npu-core-mask-sweep.md)。身份：[candidate.json](candidate.json)。

## 观察与来源

最终模型 PERF_DETAIL 的 `WorkLoad(0/1/2)`：全部 ConvExSwish/Split/MaxPool/Resize 为 100%/0%/0%，Add/Concat/Conv/ConvClip 为约 33/33/33。首轮与通用轮都固定 `core_mask=7`。

## 可检验的假设

H1：`core_mask=1` 与 `7` 的 `rknn_run` 差异只来自非卷积节点（≤ 9 ms）；H2：`auto` 不改变卷积分配。否定证据：单核下卷积节点时间显著上升，或三核下出现非 100/0/0 的卷积 WorkLoad。

## 实现与不变量

桥接 v2 的 `core_mask` 参数；`l2_compare_bridges.py --config` 的 `zerocopy_core0:1,2,0,1`、`zerocopy_auto:1,2,0,0`。同一模型、同一输入、输出必须逐位相同。

## 评测计划

L2 默认扫描；详细模式各采一份 PERF_DETAIL 比较 WorkLoad 列与节点时间。

## 结果与决策

计划中。

## 可复用经验与后续方向

若 H1 成立，RK3588 三核对本模型的价值是吞吐（三 context 并发）而非延迟；延迟优化应集中在节点工作量与宿主开销。
