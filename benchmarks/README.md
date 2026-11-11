# RK3588 YOLO 检测基准

本目录保存 YOLOv8n 数据准备、数值验收、板端采集和热点分析代码。

| 目录 | 当前状态 | 入口 |
| --- | --- | --- |
| `yolov8n_home` | 首轮实测完成。严格数值门槛失败，性能属于诊断结果。 | [SOP](yolov8n_home/README.md)、[实测简报](yolov8n_home/首轮实测简报.md) |
| `yolov8n_general` | 通用轮开发验证完成。完整数据准备、独立验收和性能轮尚未完成。 | [当前状态](yolov8n_general/README.md)、[预注册协议](yolov8n_general/quality_protocol.json) |
| `fixtures` | 首轮真实报告和图的回归夹具。 | [来源与哈希](fixtures/README.md) |
| `history` | 2026-11-11 候选算子的板端运行登记（planned）。 | [运行索引](history/index.json)、[候选原理](../principles/README.md) |
| `presentation` | 13 页离线汇报，含选型依据、真实模块依赖和可展开节点图。 | [展示与复建](presentation/README.md) |

## 本机测试

使用 Python 3.12。安装依赖并分别运行两轮测试，避免同名模块的导入缓存混用：

```bash
python -m pip install -r benchmarks/requirements.txt
python -m pytest benchmarks/yolov8n_home/tests -q --ignore=benchmarks/yolov8n_home/tests/test_bridge.py
python -m pytest benchmarks/yolov8n_general/tests -q
python -m pytest benchmarks/presentation/tests -q
```

桥接契约测试依赖 Linux、g++ 和匹配的 RKNN Runtime，在 RK3588 环境执行：

```bash
python benchmarks/yolov8n_home/tests/test_bridge.py -v
```

干净检出未包含模型和完整数据。依赖真实 ONNX 或已完成下载审计的测试会明确跳过。普通单元测试包含合成夹具与固定实测报告回归，不能替代硬件验收。

## 数据和证据

图片、输入张量、模型、完整运行目录和复现压缩包保留在本地，按 `.gitignore` 独立归档。版本控制仅包含源码、协议、冻结 ID、测试和少量已核验摘要。原始 COCO 标注及模型来源见各轮 SOP 和模型 provenance。

首轮完整复现包的文件名为 `RK3588_YOLOv8n_首轮诊断复现包_20261106.zip`。其中的原始失败结果保持不变。重新运行时使用新输出目录。

水印项目不属于本基准范围。
