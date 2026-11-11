# 05 · c2f-split-free：用两个半宽 1×1 卷积代替 C2f 里的 Split 搬运

- 候选：[candidate.json](../experiments/c2f-split-free/candidate.json) · 过程：[design.md](../experiments/c2f-split-free/design.md)
- 源码：[rewrite_yolov8n_graph.py `--split-free`](../tools/npu/rewrite_yolov8n_graph.py) · 转换：[convert_variant.py](../tools/npu/convert_variant.py)
- 父候选：首轮/通用轮 RKNN 模型中的 8 个 `Split:/model.N/Split` 节点

## 1. 算子与位置

YOLOv8 的 C2f 模块：`cv1(1×1, c→2c′) + SiLU → Split([c′, c′]) → 半 a 直接进 Concat，半 b 经 Bottleneck → Concat([a, b, m…]) → cv2(1×1)`。八个 C2f（model.2/4/6/8/12/15/18/21）的 Split 在最终模型 PERF_DETAIL 里合计 **2.859 ms**：0.652、0.330、0.371、0.239、0.426、0.325、0.382、0.134 ms。首轮热点组 `model.2`（8.0 ms）、`model.4`（6.8 ms）、`model.15`（5.2 ms）里都含一个 Split。

## 2. 原实现的成本

RKNN 把 Split 编译成真实的搬运节点：`model.2/Split` 的 RW 为 1,600 KB（读 160×160×32 fp16，写两个 160×160×16），运行在单核（WorkLoad 100/0/0）。它不做任何算术，只是把一个张量按通道切成两个。

## 3. 优化原理

卷积的每个输出通道只依赖自己的权重行：`Conv(W, b)(x)[o] = Σ W[o,·,·,·]·x + b[o]`。因此 `Split(Conv(W, b)(x)) ≡ (Conv(W[:c′], b[:c′])(x), Conv(W[c′:], b[c′:])(x))`，SiLU 是逐元素的，可以分别作用。改写把 `cv1 → SiLU → Split` 替换为 `cv1a → SiLU → a` 和 `cv1b → SiLU → b`，输出张量沿用 `Split_output_0/1` 的名字，所以 Concat、Bottleneck、残差 Add 全部不用改。算术总量不变（两个 c′ 输出的 1×1 卷积 = 一个 2c′ 输出的 1×1 卷积），Split 的一读一写消失；代价是 `cv1` 的输入要被两个卷积各读一次。对 `model.2`（输入 160×160×32 fp16 = 1.6 MB）这意味着多读 1.6 MB、少读写 3.2 MB，净节省约一次张量搬运。编译器仍能把每个新卷积与 SiLU 融合为 ConvExSwish。

“已分析未采用”：把 `Concat → cv2(1×1)` 改为各路 1×1 卷积相加，可以同样精确地消去 Concat，但要引入 `Add` 节点（本模型中 Add 每个 0.23～0.41 ms，编译器未把残差 Add 融进 Conv），且 SiLU 必须放在求和之后，账面上得不偿失，不作候选。

## 4. 不变量

- 权重数值不变，只是按输出通道切片；ORT 对比 9 路输出与原图**逐位相同**（6 张合成图，max abs 0）。
- 节点名沿用 `/model.N/cv1a|cv1b/…`，PERF_DETAIL 的 FullName 仍可按 `/model.N/` 分组。
- 其他节点、输出契约、转换配置（含 CPU Sigmoid 的 `op_target`）不变；可与 01 叠加（`--logits6 --split-free`，ORT 已验证）。

## 5. 证据边界

| 检查 | 结果 |
|---|---|
| 图结构 | Split 0 个，Conv 63 → 71，节点 228 → 244 |
| ORT 数值 | 逐位相同 |
| RKNN 转换、节点时间 | **未执行 / 未测量** |

## 6. 局限与下一步

- 收益上限 2.86 ms 节点时间，实际取决于编译器对两个半宽卷积的调度与输入重复读取的代价；如果 L2 显示 `model.N/cv1a+cv1b` 的时间之和超过原 `cv1 + Split`，判为 `rejected_perf` 并保留记录。
- 若编译器已把某些 Split 做成零拷贝视图（PERF_DETAIL 时间接近 0），该模块不改也无妨；本模型八个 Split 都有可观时间。
