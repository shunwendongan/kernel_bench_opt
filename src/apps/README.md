# src/apps — RKNN 桥接 v2（零拷贝 + 宿主 NEON 前后处理）

| 文件 | 作用 |
|---|---|
| `rknn_bridge_v2.cpp` | C ABI：`rknn_create_mem`/`rknn_set_io_mem` 零拷贝会话；`io_mode=0` 复现首轮桥接的调用序列用于 A/B；内置 `kernels/` 的 letterbox 与后处理 |
| `rknn_bridge_v2.py` | ctypes 封装 `RknnSessionV2`：`run(rgb)` 与首轮 `RknnSession.run` 契约兼容；`infer_bgr(bgr)` 为快速路径 |
| `build_bridge_v2.sh` | 板端构建（链接 `/lib/librknnrt.so`）；`--syntax-only` 用于无运行库的主机/CI |

原理：[02 io-zerocopy-fp16](../../principles/02-io-zerocopy-fp16.md)。板端入口：`scripts/run_yolo_board.sh build|l1|l2|l3`。首轮桥接 `benchmarks/yolov8n_general/scripts/rknn_bridge.cpp` 保持不变，作为对照。
