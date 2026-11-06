# 通用场景第二轮：开发状态

本轮已冻结 COCO val2017 的 40 图开发集、4960 图独立验收集和 80 图性能子集。正式验收尚未执行，不能标为质量通过。

## 已有实现

- `scripts/prepare_general.py`：固定分层、断点续传、原图与 RGB 输入哈希、全标注保留和完整回读审计。
- `scripts/prepare_model.py`：从原始 ONNX 的真实 producer 查找最终分类 Sigmoid，将其指定到 CPU，再转换 RKNN。
- `scripts/model_adapter.py`：校验九输出契约。另有六输出激活适配的单元测试，本轮模型没有采用六输出方案。
- `scripts/quality_gate.py`：读取冻结协议，检查张量误差、记录覆盖、输入与源码哈希、双向检测匹配和 COCO 总体/尺度/类别指标。
- `scripts/graph_tools.py`：真实张量连边、严格报告绑定与最多四个不重叠热点。

模型开发使用旧 40 图。CPU Sigmoid 候选通过本轮开发门槛，官方 Lite2 与 C API 的指定样本输出一致。详见 [模型开发交接](evidence/model/交接说明.md)。旧轮统一相对 L2 门槛仍有失败，未追认旧轮通过。

## 待完成

1. 恢复下载并完成全部 5000 图回读。中断时增量日志有 4740 条成功记录，尚未生成最终数据审计。
2. 人工审查性能子集的场景标签。当前标签仅来自 COCO 目标类别。
3. 集成通用轮 runner，绑定模型、协议、样本和源码哈希，保存逐图质量与预测 JSONL。
4. 对独立 4960 图执行正式质量验收。
5. 质量通过后运行 80 图正常性能与详细 profiling，生成本轮热点索引和复现包。

本轮 `legacy_runtime.py` 和 `analysis_common.py` 仅保留首轮可复用函数，尚含首轮路径和诊断语义，不能直接作为本轮正式入口运行。

## 复跑入口

依赖及通用测试见 [基准入口](../README.md)。数据流程需要首轮固定哈希的原始 COCO 标注和 40 图清单，模型转换需要固定 ONNX 文件以及 RKNN Toolkit2 环境。

```bash
python benchmarks/yolov8n_general/scripts/prepare_general.py --workers 16 --attempts 5
python benchmarks/yolov8n_general/scripts/prepare_model.py --help
python benchmarks/yolov8n_general/scripts/quality_gate.py --help
```

正式汇总需要 `config.json`、`dataset/acceptance_samples.json`、完整标注及运行记录。当前没有产出正式运行目录，不能直接执行验收汇总并宣称通过。

### 正式汇总的身份字段

- `config.json` 必须为 ONNX/RKNN 分别声明一个模型路径和 SHA-256。路径字段使用 `<backend>_path` 或 `<backend>_model`，二者不能同时存在。
- `start.model_sha256` 必须将各后端绑定到同一配置路径和哈希。不得通过扫描目录寻找“恰好同哈希”的其他模型代替配置模型。
- `config.development_manifest_path` 必须指向基准根目录内的相对路径。该文件应为首轮原始 40 图清单的原字节副本，哈希必须匹配预注册协议。
- `start.split_sha256` 与验收清单的 `split_sha256` 必须匹配冻结 `dataset/split.json`。验收 ID 必须严格等于全集减去开发 ID，也必须等于冻结验收 ID。
- 每条验收清单记录必须保存 `input_sha256`。成功质量记录必须匹配该值。

汇总开始时先发布失败占位报告，再读取证据。读取、哈希或日志失败时保持失败状态。输出位置不可写时退出码为 2。正式 runner 集成时必须遵守这些字段，不能据缺少字段的旧测试记录宣称通过。

已有冻结文件的机器定位字段 `development_source` 不参与身份比较。更换检出目录时仍保留原文件字节和哈希。旧开发图缓存缺失时可以下载同一 ID，但仍按原清单核对原图与 RGB 哈希；损坏缓存不会静默替换。

协议见 [quality_protocol.json](quality_protocol.json)，修改前计划见 [实现计划](实现计划.md)。硬门槛失败后须保留结果，不得修改门槛追认通过。
