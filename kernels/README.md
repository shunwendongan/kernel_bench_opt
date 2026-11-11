# kernels/ — 宿主侧候选 kernel 源码（yolov8n-COCO）

每个子目录对应一个 `candidate_id`（见 `experiments/index.json`），内含 C11 源码、头文件和 `bench_*.c` 对拍/微基准程序。NEON 路径与可移植标量路径同源，`-DYOLO_FORCE_SCALAR` 可强制标量构建用于自检。没有任何 RKNN 依赖，可在板端、Apple Silicon 和 x86 CI 上编译。

| 目录 | 候选 | 入口 |
|---|---|---|
| `common/` | 共享头：`fp16.h`（精确 fp16↔fp32）、`tensor_view.h`（NCHW / NC1HWC2 视图） | — |
| `post-dfl-nms-neon/` | [03 post-dfl-nms-neon](../principles/03-post-dfl-nms-neon.md) | `yolo_post.h`、`bench_post.c` |
| `pre-letterbox-neon/` | [04 pre-letterbox-neon](../principles/04-pre-letterbox-neon.md) | `letterbox.h`、`bench_letterbox.c` |

```bash
bash scripts/build_host_kernels.sh --bench          # build/libyolo_kernels.{so,dylib} + build/bench_post + build/bench_letterbox
bash scripts/build_host_kernels.sh --bench --scalar # 标量自检构建
./build/bench_post 200 0.02 0.25                    # repeats hot_fraction confidence
./build/bench_letterbox 100
python -m pytest tests -q                           # 与 numpy / cv2 参考逐项对拍（需要 numpy、opencv；ORT 测试需要冻结 ONNX）
```

Python 侧通过 `tools/yolo_kernels.py`（ctypes）调用；板端推理路径通过 `src/apps/rknn_bridge_v2.cpp` 直接链接这些源码。
