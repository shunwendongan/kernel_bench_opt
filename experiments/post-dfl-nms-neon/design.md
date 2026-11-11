# 候选 ID：post-dfl-nms-neon

原理：[principles/03-post-dfl-nms-neon.md](../../principles/03-post-dfl-nms-neon.md)。身份：[candidate.json](candidate.json)。

## 观察与来源

首轮正常轮 `postprocess_ms` 均值 9.92、中位数 7.85、P95 18.08（numpy，单线程，主机 `perf_counter_ns`）。参考实现 `yolo_common.postprocess`：按尺度 `argmax/max` → `score ≥ confidence` → DFL softmax 期望 → 按类别稳定排序贪心 NMS → 分数稳定排序截断 `max_det` → float32 坐标还原与裁剪。

## 可检验的假设

NEON 实现在 Cortex-A76 上 ≤ 1 ms（conf 0.25）且与参考结果一致（类别/分数逐位，框 ≤ 2e-3 px）；在 fp16 原生布局上直接运行无需先转 float32。否定条件：任一冻结输入的检测与参考不一致，或板端耗时 ≥ 5 ms。

## 实现与不变量

- 文件：`kernels/post-dfl-nms-neon/yolo_post.{h,c}`（C11，NEON 与可移植路径同源）、`bench_post.c`、`tools/yolo_kernels.py`。
- 输入视图：NCHW fp16/fp32（可带宽度步长）、NC1HWC2 fp16（C2 = 8）；canonical9 或 logits6。
- 不变量：numpy 的候选次序、`argmax` 并列规则、`pairwise_sum` 归约顺序、float64 框计算后一次舍入、NMS 比较与分母、稳定排序、`restore_boxes` 公式。非有限值计数并报错。
- 回退：Python 参考实现保留在 `benchmarks/yolov8n_general/scripts/yolo_common.py`。

## 评测计划

- L1：`tests/test_post_kernel.py`（合成张量 + ORT 真实模型输出，fp32/fp16/logits6/NC1HWC2，conf 0.25 与 0.001）；`build/bench_post` 微基准（NEON 与参考路径结果逐位一致）。
- L2：在 `l2_compare_bridges.py` 中逐图与 numpy 后处理对比（同一 float32 输出）。
- L3：`post_ms` 进入 e2e 统计。

## 结果与决策

开发机 Apple M4（仅说明方向）：conf 0.25、555 候选时 NEON 0.23～0.30 ms，参考 C 路径 0.94～1.12 ms；conf 0.001、8,400 候选时 2.8 ms。7 项测试通过。板端未测量。

## 可复用经验与后续方向

把“筛选”放在扫描阶段并利用 score-sum 的上界做精确预筛选，是对 Rockchip 九输出头的直接利用；logits 模式下靠 Sigmoid 单调性。后续：评估模式的 NMS 可按分数分桶减少 O(n²)。
