# post-dfl-nms-neon

YOLOv8n 检测后处理（类别扫描 → DFL 解码 → 按类 NMS → 坐标还原），直接读取 RKNN 输出缓冲（fp16 NCHW / fp16 NC1HWC2 / fp32 NCHW），支持九路 canonical 输出（含 score-sum 精确预筛选）与六路 logits 输出（Sigmoid 单调性，只对最大 logit 做一次 float32 Sigmoid）。

- 契约与数值规则：`yolo_post.h` 头部注释、[原理文档](../../principles/03-post-dfl-nms-neon.md)。
- 参考实现：`benchmarks/yolov8n_general/scripts/yolo_common.py::postprocess` + `model_adapter.py::adapt_outputs`。
- 测试：`tests/test_post_kernel.py`；微基准：`bench_post.c`（NEON 与可移植路径结果逐位一致才返回 0）。
- 辅助函数：`yolo_view_to_nchw_f32`（导出 float32 NCHW）、`yolo_sigmoid_sum`（logits → 概率 + 裁剪求和，用于质量验收）。
