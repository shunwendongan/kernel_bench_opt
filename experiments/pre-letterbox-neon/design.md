# 候选 ID：pre-letterbox-neon

原理：[principles/04-pre-letterbox-neon.md](../../principles/04-pre-letterbox-neon.md)。身份：[candidate.json](candidate.json)。

## 观察与来源

首轮正常轮 `preprocess_ms` 均值 8.61（含 `cv2.imdecode`），中位数 6.50，P95 19.90；运行库 `inputs_set` 另耗 13.19 ms 做 uint8→FP16。参考实现 `yolo_common.letterbox`：`cv2.resize INTER_LINEAR` → `copyMakeBorder(114)` → `cvtColor(BGR2RGB)`。冻结输入 `dataset/inputs/*.u8` 的哈希由板端 `check_bundle` 核对，跨平台一致。

## 可检验的假设

NEON letterbox 在 A76 上 ≤ 2 ms，输出字节与冻结输入哈希相同；`pack_ms` ≤ 0.5 ms。否定条件：任一冻结输入哈希不一致（则用 `rounding` 诊断模式定位板端 cv2 的舍入路径，不改冻结输入）。

## 实现与不变量

- 文件：`kernels/pre-letterbox-neon/letterbox.{h,c}`、`bench_letterbox.c`。
- 几何：IEEE double、`nearbyint`（与 Python `round` 相同）。
- 插值：OpenCV 8 位 INTER_LINEAR 定点规则（11 位系数、横向 int32、纵向 `((b0·(S0>>4))>>16 + (b1·(S1>>4))>>16 + 2)>>2`、横向边界置零 fx、纵向只裁剪行号、同尺寸直接复制、`scale = 1/(dsize/ssize)`）。
- 打包：fp16 NHWC（NEON）、uint8 NHWC、NC1HWC2 fp16、NCHW fp16/f32（`divide` 复现 `/255`）。
- 不变量：RGB uint8 字节与参考相同；transform 字典逐字段相同。

## 评测计划

- L1：`tests/test_letterbox_kernel.py`（318 组几何、58 组像素、放大/缩小、打包格式）；`build/bench_letterbox`。
- L2：`l2_compare_bridges.py --images` 时逐图比对哈希（`letterbox_hash_matches`）。
- L3：`letterbox_ms`、`pack_ms`、`decode_ms` 分列统计。

## 结果与决策

开发机 Apple M4（仅说明方向）：640×480 等尺寸 0.10 ms，500×375→640×480 0.90 ms，1920×1080→640×360 0.64 ms，fp16 NHWC 打包 0.08 ms；与 cv2 4.11 逐字节一致。板端未测量。

## 可复用经验与后续方向

OpenCV 的 8 位双线性不是“数学上的双线性”，复现它需要定点系数、首末行的原始系数和特化的纵向舍入；这些规则对任何需要与 cv2 预处理哈希对齐的部署都适用。后续：横向插值 NEON 化、与 s2d 打包融合。
