# kernel_bench_opt · yolov8n-COCO

RK3588 上 YOLOv8n（COCO val2017）的端侧基准、热点分析与候选算子优化分支。`main` 分支只保存共享的 SOP、约定与 schema；本分支保存这个工作负载的全部内容。分支入口页：[YOLOv8n-COCO.md](YOLOv8n-COCO.md)。

## 当前状态（2026-11-11）

| 项目 | 状态 |
|---|---|
| 首轮诊断基准（40 图，NPU 内 Sigmoid 模型） | 板端已采集；严格数值门槛失败，性能为诊断结果 |
| 通用轮（CPU Sigmoid 最终模型） | 40 图开发验证通过；独立 4960 图验收与 80 图性能轮未完成 |
| 候选算子（7 个） | 源码交付，开发机测试通过；**板端未运行，RKNN 变体未转换** |

端到端约 90 ms 中约 37 ms 是运行库格式转换与 Python 后处理；最终模型的 3 个 CPU Sigmoid 又占单帧节点时间 24.5 ms。候选围绕这两点与 NPU 图结构展开，详见[热点深度分析](reports/2026-11-11-yolov8n-hotspot-analysis/README.md)。

## 目录

```text
YOLOv8n-COCO.md              分支入口：展示入口、当前结论、获取方式、迁移来源
workload.json                分支元数据（branch_kind: workload）
benchmarks/                  首轮与通用轮基准代码、协议、冻结 ID、测试、少量证据摘要（原有布局，保持不变）
  yolov8n_home/              首轮 SOP、实测简报、分析脚本
  yolov8n_general/           通用轮脚本（桥接 v1、质量门槛、模型准备）、vendor/rknn_api.h
  presentation/              13 页离线汇报的数据生成器与模板
  history/                   候选的板端运行登记（当前 7 条 planned）
experiments/                 候选登记：index.json、<id>/candidate.json（源码哈希）、<id>/design.md、ORT 证据
principles/                  每个候选一份优化原理（README 为索引）
kernels/                     宿主侧 NEON kernel：post-dfl-nms-neon、pre-letterbox-neon、common
src/apps/                    零拷贝 RKNN 桥接 v2（C ABI + Python 封装 + 板端构建脚本）
tools/                       npu/（ONNX 图改写、RKNN 转换）、verify/（ORT 等价性、板端 L2/L3）、yolo_kernels.py、make_records.py
scripts/                     build_host_kernels.sh、run_yolo_board.sh
tests/                       kernel 对拍、ABI 布局、图改写测试（CI 执行）
reports/                     热点深度分析
docs/STATUS.md               候选状态表
汇报材料/                     2026-11-06 基准汇报 HTML、2026-11-11 优化原理汇报
```

## 候选一览

| 候选 | 类型 | 一句话 |
|---|---|---|
| [head-logits6](principles/01-head-logits6.md) | ONNX 图改写 | 删除检测头 Sigmoid/ReduceSum/Clip，输出 logits；宿主只对每个 anchor 的最大 logit 做一次 float32 Sigmoid |
| [io-zerocopy-fp16](principles/02-io-zerocopy-fp16.md) | 桥接 v2 | `rknn_set_io_mem` 一次绑定，fp16 原生布局直读直写，去掉运行库的 uint8→fp16 与 fp16→fp32 转换 |
| [post-dfl-nms-neon](principles/03-post-dfl-nms-neon.md) | NEON kernel | 一次扫描完成 max/argmax 与精确预筛选；DFL/NMS 按 numpy 规则复现 |
| [pre-letterbox-neon](principles/04-pre-letterbox-neon.md) | NEON kernel | 与 cv2 INTER_LINEAR 逐字节一致的定点 letterbox，直接打包为模型输入格式 |
| [c2f-split-free](principles/05-c2f-split-free.md) | ONNX 图改写 | `cv1 → Split` 换成两个半宽 1×1 卷积，Split 搬运消失，数值逐位相同 |
| [stem-space-to-depth](principles/06-stem-space-to-depth.md) | ONNX 图改写 | 3×3/s2 茎部等价改写为 12 通道 2×2/s1 卷积 |
| [npu-core-mask-sweep](principles/07-npu-core-mask-sweep.md) | 假设检验 | 验证 ConvExSwish 在 core_mask=7 下只跑单核的观察 |

## 获取与运行

```bash
git clone --branch yolov8n-COCO --single-branch https://github.com/shunwendongan/kernel_bench_opt.git
cd kernel_bench_opt
```

本机测试（Python 3.12，依赖见 `benchmarks/requirements.txt`，与 CI 相同）：

```bash
python -m pip install -r benchmarks/requirements.txt
python -m pytest benchmarks/yolov8n_home/tests -q --ignore=benchmarks/yolov8n_home/tests/test_bridge.py
python -m pytest benchmarks/yolov8n_general/tests -q
python -m pytest benchmarks/presentation/tests -q
bash scripts/build_host_kernels.sh --bench && python -m pytest tests -q   # 候选 kernel 与图改写（ORT 测试需要冻结 ONNX 放在 models/yolov8n.onnx）
```

RK3588 板端（需要首轮板端目录中的冻结 40 图输入与模型，环境变量见脚本头部说明）：

```bash
scripts/run_yolo_board.sh build                        # kernels + 桥接 v2，链接 /lib/librknnrt.so
scripts/run_yolo_board.sh l1                           # 微基准 + 主机测试
scripts/run_yolo_board.sh l2 <模型.rknn> <输出目录>     # 首轮桥接 vs 桥接 v2：逐位输出、检测、letterbox 哈希、数值门槛
scripts/run_yolo_board.sh l3 <模型.rknn> <输出目录>     # 40 图 × 10 次端到端
```

图改写候选需要先在装有 rknn-toolkit2 2.3.2 的 Linux x86 主机上生成 `.rknn`：

```bash
python tools/npu/rewrite_yolov8n_graph.py --source models/yolov8n.onnx --out models/variants/yolov8n_logits6.onnx --logits6 --record models/variants/rewrite_logits6.json
python tools/verify/compare_onnx_variants.py --reference models/yolov8n.onnx --variant models/variants/yolov8n_logits6.onnx
python tools/npu/convert_variant.py --source models/variants/yolov8n_logits6.onnx --output models/variants/yolov8n_logits6.rknn --record models/variants/convert_logits6.json --rewrite-record models/variants/rewrite_logits6.json
```

运行完成后把 `summary.json`、`validation.json`、`manifest.json` 放入对应的 `benchmarks/history/<run-id>/`，并更新 `experiments/<id>/candidate.json` 的状态与 `docs/STATUS.md`。门槛失败保留记录，不修改门槛。

## 证据边界

- 性能数字的口径分三类：主机 `perf_counter_ns` 分段墙钟、C++ `steady_clock` API 包围时间、RKNN PERF_DETAIL 节点耗时（含 profiling 开销，不是独立子图墙钟）。引用时注明。
- 开发机（Apple M4）上的 kernel 时间只说明方向；板端 Cortex-A76 时间以 `benchmarks/history/` 的运行记录为准。
- 模型、原始图片、完整运行目录与复现压缩包不入库（见 `.gitignore`）；模型来源、哈希与许可见各轮 SOP 与 `models/model_provenance.json`。

## 约定

目录布局、候选命名、状态枚举、每轮必须提交的文件以 `main` 分支的 `docs/CONVENTIONS.md` 与 `docs/SOP.md` 为准；记录格式见 `main` 的 `schemas/`。`tools/make_records.py` 根据源码哈希重新生成 `experiments/*/candidate.json`、`experiments/index.json`、`workload.json` 与 `benchmarks/history/`。
