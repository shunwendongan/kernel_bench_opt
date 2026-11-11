# tools — 候选算子的配套工具

| 路径 | 作用 | 运行环境 |
|---|---|---|
| `yolo_kernels.py` | `kernels/` 的 ctypes 封装（测试、桥接封装共用） | 任意（numpy） |
| `npu/rewrite_yolov8n_graph.py` | ONNX 图改写：`--logits6`、`--split-free`、`--s2d-stem graph|host`，可叠加；校验源模型哈希，写改写记录 | onnx |
| `npu/convert_variant.py` | 变体 → RKNN（FP16、mean 0/std 255、rk3588；九路变体保留 CPU Sigmoid 的 `op_target`） | Linux x86 + rknn-toolkit2 2.3.2 |
| `verify/compare_onnx_variants.py` | 原图与变体的 ORT 等价性（logits6 先还原 canonical9） | onnxruntime |
| `verify/l2_compare_bridges.py` | 板端 L2：首轮桥接 vs 桥接 v2，40 张冻结输入，逐位/检测/哈希/数值门槛 | RK3588 |
| `verify/l3_e2e_v2.py` | 板端 L3：桥接 v2 快速路径 40 图 × 10 次端到端 | RK3588 |

开发机已生成并验证的变体与报告见 `experiments/<id>/evidence/`（ORT 口径，非板端）。
