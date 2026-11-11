# 候选 ID：stem-space-to-depth

原理：[principles/06-stem-space-to-depth.md](../../principles/06-stem-space-to-depth.md)。身份：[candidate.json](candidate.json)。

## 观察与来源

最终模型 PERF_DETAIL：`Conv:/model.0/conv/Conv` 4.113 ms，MacUsage 0.70%，DDR 周期 970,032 > NPU 周期 921,600，RW 2,402 KB；首轮热点组 `model.0` 4.07 ms。

## 可检验的假设

12 通道 320×320 的 2×2/s1 卷积比 3 通道 640×640 的 3×3/s2 卷积至少快 1.5 ms，且 `graph` 变体中 SpaceToDepth 节点的代价小于该差额。否定条件：PERF_DETAIL 中新茎部（含重排）时间 ≥ 4.1 ms，或编译器不支持 2×2 核/12 通道的高效路径。

## 实现与不变量

- 变换：`rewrite_yolov8n_graph.py --s2d-stem graph|host`；权重按相位重排到 `[16,12,2,2]`，pads `[1,1,0,0]`。
- `graph`：输入契约不变；`host`：输入 `images_s2d [1,12,320,320]`，宿主打包时按 `(pr·2+pc)·3+c` 写入（NEON 快速路径未实现，先用 ORT/标量验证）。
- 不变量：ORT 9 路 max abs 4.0e-5，检测一致；其余节点不变；可叠加 01/05。
- 回退：使用未改写模型。

## 评测计划

- L1：`tests/test_rewrite_graph.py`（含 ONNX `SpaceToDepth` 通道顺序测试）。
- L2：先转换 `graph` 变体采 PERF_DETAIL；若重排代价高再转换 `host` 变体并在桥接 v2 中打包（需要新增 s2d 打包格式）。
- 停止条件：转换失败或节点时间不降即 `rejected_perf`，保留日志。

## 结果与决策

计划中。

## 可复用经验与后续方向

对 NPU 上利用率极低的 3 通道输入卷积，空间到深度是零成本的等价重写；是否有效取决于编译器对小核/多通道的支持，必须实测。
