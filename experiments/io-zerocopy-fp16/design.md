# 候选 ID：io-zerocopy-fp16

原理：[principles/02-io-zerocopy-fp16.md](../../principles/02-io-zerocopy-fp16.md)。身份：[candidate.json](candidate.json)。

## 观察与来源

- 首轮正常轮（`benchmarks/yolov8n_home/summary.json` → `baseline`，40 图 × 10 次）：`inputs_set_ms` 均值 13.19，`run_ms` 43.98，`outputs_get_ms` 12.00，Python 输出复制 1.72；推理调用整体 71.12 ms。口径：C++ `steady_clock` 包围 API；Python `perf_counter_ns`。
- 运行库属性：输入 `NHWC FP16 [1,640,640,3]`，输出 9 路 `FP16 NCHW`，20×20 与单通道输出的 `size_with_stride` 大于 `size`（原生布局有通道/宽度补齐）。
- 未确认：`RKNN_QUERY_NATIVE_INPUT_ATTR` 对该模型报告的格式；`pass_through=1` 下归一化是否已折进第一层卷积（PERF_DETAIL 中 InputOperator 7 µs，无归一化节点，推断为已折入）。

## 可检验的假设

零拷贝路径下，每帧的输入提交降为 `pack_ms`（预期 ≤ 1 ms），输出获取为 0，`run_ms` 不增加，端到端减少 20 ms 以上；导出的 float32 输出与 v1 `want_float` 逐位相同。否定条件：输出不逐位相同（说明格式/归一化判断错误），或 `run_ms` 增加吞掉收益（说明运行库在 run 内做了转换）。

## 实现与不变量

- 文件：`src/apps/rknn_bridge_v2.cpp`（C ABI）、`src/apps/rknn_bridge_v2.py`、`src/apps/build_bridge_v2.sh`。
- 模式：`io_mode` 0 旧路径 / 1 零拷贝；`input_mode` 0 自动 / 1 声明 UINT8 NHWC 由运行库转换 / 2 原生直通；`output_mode` 0 原生 / 1 普通 fp16 NCHW；`mem_flags` 1 输入不可缓存；`core_mask` 可配置。
- 不变量：同一 .rknn、同一输入字节；详细模式 PERF_DETAIL 写法与 v1 相同；`run()` 返回值契约与 v1 兼容（float32 NCHW 数组列表 + 指标）。
- 回退：`io_mode=0` 即 v1 行为。

## 评测计划

- L1：`tests/test_abi_layout.py`（结构布局）、`src/apps/build_bridge_v2.sh --syntax-only`。
- L2：`scripts/run_yolo_board.sh l2 <model> <out>` 默认扫描六个配置（原生直通、声明 uint8、普通 fp16 NCHW、旧路径、单核、auto），逐图检查 v2 导出与 v1 输出逐位相同、NEON 检测与 numpy 后处理一致、letterbox 哈希与冻结输入一致；记录 `pack_ms/run_ms/post_ms/wall_ms` 的汇总统计。
- L3：`l3_e2e_v2.py` 40 图 × 10 次。
- 停止条件：任一配置创建失败或输出不一致时记录错误文本并排除该配置，不修改对照。

## 结果与决策

计划中；板端尚未编译与运行。

## 可复用经验与后续方向

零拷贝的收益来自“让运行库不转换”，前提是宿主能产出原生格式；本分支的 kernel 视图同时支持 NCHW/NC1HWC2 fp16，可直接移植到其他 RKNN 模型。后续：`RKNN_FLAG_DISABLE_FLUSH_*` + `rknn_mem_sync` 的细粒度缓存控制。
