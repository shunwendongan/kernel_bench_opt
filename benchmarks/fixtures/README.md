# 真实报告回归夹具

这些文件来自首轮 YOLOv8n FP16 实测，仅用于解析器与图映射回归。它们不代表通用轮新模型的性能。

| 文件 | 原始位置 | SHA-256 |
| --- | --- | --- |
| `yolov8n_home_bridge_smoke.perf.txt` | `yolov8n_home/evidence/bridge_smoke_detail.perf.txt` | `b175f53f42accc3ff577fe78991bb960c91a4aecd155d3365f6c7fdc8e40a192` |
| `yolov8n_home_native_graph.json` | `yolov8n_home/graphs/native_graph.json` | `829323acd92ed1df943fd5dc1e93213e2ebfd1731934342cd26ca6cda1109713` |

对应编译模型 SHA-256：`56808e7b2d85de3cc59e953711d158f15b36bb4b33f12a05d8bd27e0a70d5813`。

保留原始节点、张量索引和报告字节。测试通过不表示重新执行了板端推理。容器解析另有独立构造的合成 RKNN 测试。

## 采集时的源码快照

`yolov8n_home_sources/<SHA-256>/run_board.py` 保存首轮实际使用的采集脚本。其哈希为 `b5ff49b3ebf6a665568a8030ce695243ca734d13097d977491ad182528f53df4`，与三个原始 `start.json` 的记录一致。

该快照只供历史证据核验，保留当时的实现，不作为新采集入口。当前脚本修复后，汇报生成器按采集哈希读取快照，不改写旧运行记录，也不把修复后的代码当成旧实测的来源。
