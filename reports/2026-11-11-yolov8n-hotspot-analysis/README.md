# RK3588 YOLOv8n 热点深度分析（2026-11-11，基于分支内已有证据）

本报告只使用分支里已经入库的证据：首轮诊断（`yolov8n_home_20261106_01`，模型 `56808e7b…`，NPU 内 Sigmoid）的正常轮/详细轮统计，以及通用轮最终模型（`1b213f80…`，CPU Sigmoid）的单图 PERF_DETAIL 报告。所有数字都注明口径；没有任何一项是本轮新采集的板端数据。结论用于指导 [候选算子](../../principles/README.md) 的设计，候选的收益必须由板端 L1/L2/L3 实测确认。

## 1. 证据与口径

| 证据 | 文件 | 口径 |
|---|---|---|
| 首轮正常轮 | `benchmarks/yolov8n_home/summary.json` → `baseline` | 40 图 × 10 次混合分布；主机 `perf_counter_ns` 分段墙钟；API 阶段为 C++ `steady_clock` 包围时间 |
| 首轮详细轮 | 同上 → `detail`、`presentation/report_data.json` → `hotspots` | 40 份 PERF_DETAIL 的编译节点耗时之和，不是独立子图墙钟 |
| 最终模型详细报告 | `benchmarks/yolov8n_general/evidence/model/final_cpu_placement.perf.txt` | 单图 312192、PERF_DETAIL 模式，Time(us) 为节点耗时 |
| 运行库属性 | `summary.json` → `baseline.runtime.metadata` | `rknn_query` 的输入/输出属性（fmt、type、size_with_stride） |

未锁频、未绑核，详细轮含 profiling 开销；下表中的差值和占比都是在这些条件下的描述统计。

## 2. 端到端时间去了哪里（首轮正常轮，单位 ms）

| 阶段 | 平均 | 中位数 | P95 | 说明 |
|---|---:|---:|---:|---|
| 读取解码 + letterbox/RGB | 8.61 | 6.50 | 19.90 | `cv2.imdecode` + `cv2.resize` + `copyMakeBorder` + `cvtColor`，单线程 |
| 推理调用整体 | 71.12 | 65.18 | 92.82 | 下面四项之和 |
| ├ `rknn_inputs_set` | 13.19 | — | — | 运行库把 UINT8 NHWC 转成模型输入 **FP16 NHWC**（1,228,800 个元素）并复制到 DMA 内存 |
| ├ `rknn_run` | 43.98 | — | — | 运行库入口时间，含 NPU 执行与内部 CPU 节点 |
| ├ `rknn_outputs_get(want_float)` | 12.00 | — | — | 运行库把 9 路 **FP16** 输出转成 float32（1,218,000 个元素） |
| └ 输出复制到 numpy | 1.72 | — | — | Python 侧 `.copy()` |
| DFL + NMS + 坐标还原（numpy） | 9.92 | 7.85 | 18.08 | 纯 Python/numpy，单线程 |
| **端到端** | **89.65** | **88.65** | **120.88** | 不含模型加载、保存、评估 |

结论 1：**约 37 ms（41%）不是 NPU 计算**，而是输入转换 13.2、输出转换 13.7、Python 后处理 9.9 这三类宿主开销。它们都由运行库或 numpy 的标量路径完成，是零拷贝 IO 与 NEON kernel 可以直接替换的部分。

元数据证实了转换的必要性：输入属性 `fmt=NHWC, type=FP16, size=2,457,600`，输出属性全部 `type=FP16, fmt=NCHW`，且 20×20 与 1 通道输出的 `size_with_stride` 大于 `size`（例如 `331`：12,800 → 102,400；`369`：800 → 7,680），说明运行库内部保存的是通道补到 8、宽度补到 8 的原生布局，`want_float` 还包含一次重排。

## 3. 最终模型新引入的 CPU Sigmoid 热点

通用轮为修正分类概率的 FP16 下限（0.00183）把三个最终 Sigmoid 指定到 CPU。该模型的 PERF_DETAIL（单图）：

| 节点 | 形状 | 时间 ms | 占单帧节点总时间 81.63 ms |
|---|---|---:|---:|
| `Sigmoid:/model.22/Sigmoid` (CPU, FLOAT16) | 1×80×80×80 | 21.208 | 26.0% |
| `Sigmoid:/model.22/Sigmoid_1` | 1×80×40×40 | 2.740 | 3.4% |
| `Sigmoid:/model.22/Sigmoid_2` | 1×80×20×20 | 0.559 | 0.7% |
| 三者合计 | 672,000 个元素 | **24.507** | **30.0%** |

吞吐只有约 27 M 元素/s，说明运行库的 CPU FP16 Sigmoid 是单线程标量实现。首轮详细轮节点和平均 61.42 ms，最终模型单图 81.63 ms，差额与这 24.5 ms 一致：**数值修正把正常轮大约 20 ms 的时间加回了 `rknn_run`**。这是目前整条链路上单项最大的可消除开销，对应候选 [01 head-logits6](../../principles/01-head-logits6.md)：让图直接输出 logits，Sigmoid 由宿主 NEON kernel 对每个 anchor 的最大 logit 做一次（候选筛选只需要 argmax 和 sigmoid(max)），全量概率只在质量验收导出时计算。

## 4. NPU 节点的构成（最终模型单图）

| 类型 | 节点数 | 时间 ms | 占比 | 备注 |
|---|---:|---:|---:|---|
| ConvExSwish | 57 | 43.513 | 53.3% | Conv+SiLU 融合 |
| Sigmoid (CPU) | 3 | 24.507 | 30.0% | 第 3 节 |
| Concat | 13 | 5.114 | 6.3% | C2f 内 3.094，颈部跨尺度 1.846，SPPF 0.174 |
| Split | 8 | 2.859 | 3.5% | 全部来自 C2f 的 `cv1 → Split` |
| Add | 6 | 1.783 | 2.2% | 骨干 Bottleneck 残差 |
| Conv (无激活) | 6 | 1.280 | 1.6% | 检测头最后的 1×1 |
| Resize / MaxPool / ConvClip | 2 / 3 / 3 | 1.008 / 0.950 / 0.358 | 2.8% | 上采样、SPPF、score-sum |

### 4.1 茎部两层占 ConvExSwish 的 14.6%

`/model.0/conv/Conv`（3→16，3×3/s2，640×640 输入）耗时 4.113 ms，MacUsage 只有 **0.70%**，DDR 周期 970,032 大于 NPU 周期 921,600，RW 2,402 KB；`/model.1` 2.245 ms、MacUsage 3.42%。3 通道 FP16 输入在 NPU 上既不能填满通道对齐，也让访存模式很差。候选 [06 stem-space-to-depth](../../principles/06-stem-space-to-depth.md) 把 3×3/s2 重写成 12 通道 320×320 上的 2×2/s1 卷积，数学等价（ORT 最大绝对差 4.0e-5），是否更快取决于 RKNN 编译器对 12 通道输入和 2×2 核的处理，必须实测。

### 4.2 C2f 的 Split 与 Concat

八个 C2f 模块都含 `cv1(1×1) → Split(两半) → … → Concat → cv2(1×1)`。Split 在 NPU 上是一次真实的读写（`model.2` 的 Split RW 1,600 KB、0.652 ms），而且它只是把 `cv1` 的输出通道一分为二。候选 [05 c2f-split-free](../../principles/05-c2f-split-free.md) 把 `cv1` 拆成两个各产一半通道的 1×1 卷积，Split 节点消失，消费者不变，ORT 验证逐位相同。可消除上限是 8 个 Split 的 2.859 ms，代价是 `cv1` 的输入被读两次，净收益需实测。

Concat 占 5.1 ms，但把 `Concat → cv2(1×1)` 改成多路 1×1 卷积求和会引入 `Add` 节点（本模型 Add 每个 0.23～0.41 ms，且编译器没有把残差 Add 融进 Conv），账面不划算，本轮不作候选，记录在原理文档的“已分析未采用”。

### 4.3 多核分配的观察

PERF_DETAIL 的 `WorkLoad(0/1/2)` 列显示：所有 `ConvExSwish`、`Split`、`MaxPool`、`Resize` 节点都是 `100%/0%/0%`，而 `Add`、`Concat`、无激活 `Conv`、`ConvClip` 节点是约 `33%/33%/33%`。也就是说在 `core_mask=7` 下，占 53% 时间的卷积只跑在一个核上。这可能是运行库对带 SiLU 的融合卷积不做多核拆分，也可能是编译期决定。它直接决定“三核”是否对本模型有意义，所以列为可证伪假设 [07 npu-core-mask-sweep](../../principles/07-npu-core-mask-sweep.md)：用桥接 v2 的 `core_mask` 参数在 1、7、auto 之间对比 `rknn_run` 时间与节点 WorkLoad，不改模型。

### 4.4 首轮热点分组回看

首轮按模块连通分量选出的热点（节点时间和，入选图平均）：`model.2` 8.01 ms、`model.4` 6.85、`model.15` 5.19、`model.6` 4.65、`model.22#2` 4.34、`model.0` 4.07。`model.2`/`model.4`/`model.15`/`model.6` 都是 C2f，其中 1×1 卷积在 160×160、80×80 上 DDR 受限（`model.2/cv1` DDR 周期 554,443 对 NPU 周期 102,400），Split/Concat/Add 合计约占这些模块时间的 20%～25%。`model.22#2` 是 P3 分类分支（节点 66/72/73/74：1.418+1.801+0.462 ms 的卷积加 CPU Sigmoid）。

## 5. 候选与预期（全部待板端实测）

| 候选 | 杠杆 | 预期（口径） | 开发机证据 | 主要风险 |
|---|---|---|---|---|
| 01 head-logits6 | 删除 3 个 CPU Sigmoid、3 个 ReduceSum/Clip、3 路输出 | 单帧节点时间 −24.5 ms（最终模型 PERF_DETAIL）；对首轮模型则是数值修正不再付出 CPU 代价 | ORT 等价（还原后 max abs 3.8e-6），检测一致 | 运行库对 6 输出模型的原生布局可能不同；由桥接 v2 元数据确认 |
| 02 io-zerocopy-fp16 | `rknn_create_mem`/`rknn_set_io_mem`，fp16 原生布局直读 | 消除 inputs_set 13.2 + outputs_get 12.0 + 复制 1.7 ms（首轮正常轮均值） | C++ 对厂商头文件语法检查通过；布局/dtype 全部由 kernel 视图支持 | `pass_through` 原生格式与归一化位置需 L2 的逐位对比证明 |
| 03 post-dfl-nms-neon | NEON 扫描/DFL/NMS 替代 numpy | 9.92 ms → 约 0.5～1 ms（A76 推测；M4 0.23～0.30 ms） | 与 numpy 参考在 fp32/fp16/logits6/NC1HWC2 下结果一致 | 极少数并列分数的次序；已按 numpy 规则复现 |
| 04 pre-letterbox-neon | NEON letterbox + 直接打包 fp16 输入 | 8.61 ms 中的 resize/pad/cvtColor 部分（JPEG 解码不变） | 与 cv2 INTER_LINEAR 逐字节一致（58 组尺寸） | 板端 cv2 构建若用不同舍入路径，用 `rounding` 参数验证 |
| 05 c2f-split-free | 删除 8 个 Split | ≤ 2.86 ms 节点时间 | ORT 逐位相同 | `cv1` 输入重复读取可能吃掉部分收益 |
| 06 stem-space-to-depth | 3→12 通道、3×3/s2→2×2/s1 | 4.1 ms 茎部的一部分 | ORT max abs 4.0e-5 | 编译器可能插入布局转换；宿主打包版本改变输入契约 |
| 07 npu-core-mask-sweep | `core_mask` 1/7/auto | 假设检验，不预设收益 | — | 无 |

把 01～04 叠加在首轮正常轮均值上的粗算：89.65 − 24.5（若以最终模型为基线则另加）− 26.9 − 9.4 − 约 3 ≈ 推理链路有机会降到 50 ms 以下，其中 `rknn_run` 约 44 ms 成为新的瓶颈。这是推断，不是测量。

## 6. 不变量与数值风险

- 质量门槛不变：`quality_protocol.json` 的 relative L2 ≤ 0.02、RMSE ≤ 0.001、MAE ≤ 0.005、检测匹配 ≥ 0.95 等继续适用；logits6 模型用 `model_adapter.adapt_outputs`（float32 Sigmoid + 求和裁剪）还原 canonical9 后再进门槛。
- 预处理字节不变：桥接 v2 的 letterbox 必须与冻结的 `dataset/inputs/*.u8` 哈希一致，`l2_compare_bridges.py` 逐图检查。
- 后处理语义不变：confidence 比较、按类 NMS、稳定排序、坐标还原与裁剪全部按 `yolo_common.postprocess` 复现；允许的差异只有 `expf` 与 numpy `exp` 的最后一位。
- 图改写不改变权重数值；只有 s2d 茎部因累加顺序不同出现 1e-5 量级差异。

## 7. 未测量项

内部算子库 API、设备 kernel 时间线、能耗、跨后端成本矩阵仍未采集；原生 NC1HWC2 输出的实际 C2/W 对齐、`RKNN_QUERY_NATIVE_INPUT_ATTR` 对 FP16 图像输入的报告、多核 WorkLoad 的成因都要由板端桥接 v2 的元数据给出。本报告不替代任何板端运行记录。
