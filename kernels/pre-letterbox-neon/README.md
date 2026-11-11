# pre-letterbox-neon

与 `yolo_common.letterbox`（`cv2.resize INTER_LINEAR` + `copyMakeBorder(114)` + `BGR→RGB`）逐字节一致的 letterbox，以及把 RGB uint8 打包为模型输入格式（fp16 NHWC / uint8 NHWC / NC1HWC2 fp16 / NCHW fp16·f32）的例程。

- 几何与舍入规则：`letterbox.h` 头部注释、[原理文档](../../principles/04-pre-letterbox-neon.md)。
- `rounding` 参数：默认 OpenCV 8 位特化规则；另两种只用于板端 cv2 构建差异的诊断。
- 测试：`tests/test_letterbox_kernel.py`（几何 318 组、像素 58 组、放大/缩小、打包格式）；微基准：`bench_letterbox.c`。
