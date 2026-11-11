# YOLOv8n · COCO · RK3588

本分支保存 YOLOv8n 的 RK3588 检测基准、证据摘要和小组汇报。分支名为 `yolov8n-COCO`。

## 展示入口

- [13 页小组汇报 HTML](汇报材料/2026-11-06_YOLOv8n_RK3588_小组汇报.html)：下载后用浏览器打开。包含模型与数据选型、预处理、热点类型、模块图、可展开节点图和分段延迟。
- [展示说明与构建方法](benchmarks/presentation/README.md)：说明图边来源、延迟口径与复建方法。
- [基准总览与测试命令](benchmarks/README.md)：定位两轮实验、测试依赖与证据范围。

## 候选算子与优化原理（2026-11-11 新增）

| 入口 | 内容 |
| --- | --- |
| [优化原理文档](principles/README.md) | 7 个候选的原理、不变量与证据边界：head-logits6、io-zerocopy-fp16、post-dfl-nms-neon、pre-letterbox-neon、c2f-split-free、stem-space-to-depth、npu-core-mask-sweep |
| [热点深度分析](reports/2026-11-11-yolov8n-hotspot-analysis/README.md) | 基于首轮与最终模型 PERF_DETAIL 的时间去向、运行库转换开销、CPU Sigmoid 热点、多核分配观察 |
| [候选登记与设计记录](experiments/index.json) | `experiments/<id>/candidate.json`（源码哈希）、`design.md`（观察→假设→评测→结果） |
| [候选源码](kernels/README.md) | `kernels/`（NEON 前后处理）、`src/apps/`（零拷贝桥接 v2）、`tools/npu`（ONNX 图改写与 RKNN 转换）、`tools/verify`（板端 L2/L3） |
| [计划中的板端运行](benchmarks/history/index.json) | 7 条 planned 记录；完成后补 summary/validation/manifest |
| [状态表](docs/STATUS.md) · [小组汇报（原理）](汇报材料/2026-11-11_YOLOv8n_RK3588_算子优化原理汇报.md) | 当前全部为“源码交付、待板端验收” |

板端复跑：`scripts/run_yolo_board.sh build|l1|l2|l3`（见脚本头部说明）。图改写变体先在装有 rknn-toolkit2 2.3.2 的 Linux 主机上用 `tools/npu/rewrite_yolov8n_graph.py` 与 `tools/npu/convert_variant.py` 生成 .rknn。原有 `benchmarks/` 基准目录保持不变。

## 当前结论

| 实验 | 状态 | 说明 |
| --- | --- | --- |
| 首轮 YOLOv8n 诊断 | 已采集板端性能，严格数值门槛失败 | 延迟与热点属于诊断证据，不能作为正式验收通过的结果。 |
| 通用 COCO 轮 | 开发验证已完成 | 完整数据准备、独立验收和性能轮尚未完成。 |

正常计时与详细 profiling 使用独立采集。节点耗时之和不代表独立子图的墙钟延迟。现有热点没有可用于独立重放的内部边界张量。

## 获取与同步

```bash
git clone --branch yolov8n-COCO --single-branch https://github.com/shunwendongan/kernel_bench_opt.git
cd kernel_bench_opt
git pull --ff-only
```

源码、协议、冻结 ID、测试、少量证据摘要和离线 HTML 纳入 Git。模型、原始图片、完整运行目录与复现压缩包仍独立归档。下载本分支不会自动取得这些资产。

## 迁移来源

本分支沿用 `yousizaitianqiong/kernel-operator-optimization` 的提交历史。原 `main` 与 `feat/rk3588-yolo-quality-baseline` 的提交均可从本分支追溯。

原仓库 PR 与 CI 的状态快照见 [迁移前 GitHub 记录](benchmarks/migration/source-github-records.json)。该记录保留原始状态和提交 ID，不代表目标仓库已重新运行 CI。GitHub 的 PR 编号和运行记录不能直接并入另一个已有仓库。

上游模型、数据和代码的许可与来源继续按各轮 SOP、模型 provenance 和 vendor 文件保存。
