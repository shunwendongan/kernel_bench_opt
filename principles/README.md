# 优化原理文档（yolov8n-COCO）

这个目录为分支 `yolov8n-COCO` 上的每一个候选算子 / 适配器 / 子图改写保存一份**原理说明**：它改了什么、为什么会更快、哪些东西必须保持不变、哪些结论是实测、哪些是推断。格式与 `Qwen2.5-Omni-3B_VCB-Bench` 分支相同（`main` 的 `docs/CONVENTIONS.md`）。

过程记录在 `experiments/<candidate>/design.md`，身份与哈希在 `candidate.json`，热点分析在 [reports/2026-11-11-yolov8n-hotspot-analysis](../reports/2026-11-11-yolov8n-hotspot-analysis/README.md)，板端运行登记在 `benchmarks/history/`。

截至 2026-11-11，所有候选都是 **源码交付、待板端验收**（`source_delivered`/`proposed`）。开发机（Apple M4）上的数字只说明方向。

## 索引

| # | 候选 | 类型 | 一句话原理 | 预期杠杆（已有证据口径） | 状态 |
|---|---|---|---|---|---|
| 01 | [head-logits6](01-head-logits6.md) | NPU 子图改写 | 删除检测头的 Sigmoid/ReduceSum/Clip，输出 logits；宿主只对每个 anchor 的最大 logit 做一次 float32 Sigmoid | 最终模型单帧节点时间中 24.5 ms 的 CPU Sigmoid（30%） | 源码交付，待转换与板端验收 |
| 02 | [io-zerocopy-fp16](02-io-zerocopy-fp16.md) | 宿主适配器 | `rknn_create_mem`/`rknn_set_io_mem` 一次绑定；按原生格式写 fp16 输入、就地读 fp16 输出 | 首轮正常轮 inputs_set 13.2 + outputs_get 12.0 + 复制 1.7 ms | 源码交付（语法检查、ABI 测试通过），待板端编译验收 |
| 03 | [post-dfl-nms-neon](03-post-dfl-nms-neon.md) | 宿主 kernel | 一次 NEON 扫描做 max/argmax 与精确预筛选，DFL/NMS 按 numpy 规则复现 | 首轮正常轮 9.92 ms 的 numpy 后处理 | 源码交付，开发机 L1 通过 |
| 04 | [pre-letterbox-neon](04-pre-letterbox-neon.md) | 宿主 kernel | 与 cv2 INTER_LINEAR 逐字节一致的定点 letterbox，直接打包为模型输入格式 | 首轮预处理 8.61 ms 中的 resize/pad/cvtColor 部分 | 源码交付，开发机 L1 通过 |
| 05 | [c2f-split-free](05-c2f-split-free.md) | NPU 子图改写 | 把 `cv1 → Split` 换成两个半宽 1×1 卷积，Split 搬运消失 | 8 个 Split 共 2.86 ms 节点时间 | 源码交付，ORT 逐位一致，待转换 |
| 06 | [stem-space-to-depth](06-stem-space-to-depth.md) | NPU 子图改写 | 3×3/s2 茎部等价改写为 12 通道 2×2/s1 卷积 | `model.0` 4.1 ms（MacUsage 0.7%） | 源码交付，ORT 等价，收益不确定 |
| 07 | [npu-core-mask-sweep](07-npu-core-mask-sweep.md) | 系统级（假设检验） | 验证 ConvExSwish 在 core_mask=7 下只跑单核的观察 | 不预设收益 | 设计完成，待板端 |

## 每份文档的结构

1. **算子与位置**：结构、shape、在热点表中的位置。
2. **原实现的成本**：数据搬运和计算的账，指出瓶颈。
3. **优化原理**：逐条说明变换及其为什么有效。
4. **不变量**：为了结果可比，哪些东西一个字节都不能动；如果数值会变，说明为什么会变。
5. **证据边界**：哪些是板端实测、哪些是开发机检查、哪些只是推断。
6. **局限与引出的下一步**。

## 写新候选时

- 先写原理再写实验，实验设计应能否定原理中的假设。
- 数字都标注计时口径（主机 `perf_counter_ns` 分段 / C++ `steady_clock` API 包围 / PERF_DETAIL 节点耗时 / PERF_RUN）和设备；非目标设备上的数字只能说明方向。
- 热点表的来源是首轮 `summary.json`、`report_data.json` 与最终模型 `final_cpu_placement.perf.txt`，引用时注明。
