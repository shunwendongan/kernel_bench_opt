# 候选状态（yolov8n-COCO）

更新：2026-11-11。状态枚举见 `main` 的 `docs/CONVENTIONS.md` 第 4 节。板端未运行任何候选；所有性能预期来自已有证据的推断。

| 候选 | 类型 | 状态 | 开发机证据 | 下一步 |
|---|---|---|---|---|
| head-logits6 | subgraph | source_delivered | ORT 还原后 max abs 3.85e-6，检测一致；6 项图改写测试通过 | Linux 主机转换 → 板端 L2（数值门槛）→ L3 |
| io-zerocopy-fp16 | adapter | source_delivered | C++ 对 rknn_api.h 2.3.2 语法检查、ABI 布局测试通过 | 板端 `build_bridge_v2.sh` → L2 六配置扫描 |
| post-dfl-nms-neon | kernel | source_delivered | 7 项对拍测试通过；M4 0.23～0.30 ms（方向） | 板端 L1 微基准 → L2 |
| pre-letterbox-neon | kernel | source_delivered | 与 cv2 4.11 逐字节一致（5 项测试）；M4 0.04～0.9 ms（方向） | 板端 L2 哈希核对 |
| c2f-split-free | subgraph | source_delivered | ORT 逐位相同 | 转换 → L2 PERF_DETAIL 分组对比 |
| stem-space-to-depth | subgraph | source_delivered | ORT max abs 4.0e-5 | 转换 graph 变体 → PERF_DETAIL |
| npu-core-mask-sweep | system | proposed | 仅观察（WorkLoad 列） | L2 `zerocopy_core0`/`zerocopy_auto` |

计划中的运行见 `benchmarks/history/index.json`；完成后在对应目录补 `summary.json`、`validation.json`、`manifest.json` 并更新本表。

既有基准状态不变：首轮严格数值门槛失败（诊断基线）；通用轮 40 图开发验证通过，独立 4960 图验收与 80 图性能轮未完成。
