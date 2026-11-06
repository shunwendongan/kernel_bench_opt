# YOLOv8n 检测首轮 SOP

本轮已完成检测部署、质量对照及分轮实测。实现前的计划见 [实现计划.md](实现计划.md)，结果见 [首轮实测简报.md](首轮实测简报.md)。严格输出数值门槛失败，检测匹配与子集 AP 门槛通过。性能结果仅作为诊断基线，不能称为正式验收通过。

## 1. 冻结模型与数据

- 模型：Rockchip Model Zoo 的 YOLOv8n 检测导出，batch=1、640×640、opset 12。
- ONNX 来自[官方下载脚本所列文件](https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov8/yolov8n.onnx)。参考仓库提交为 `837db8e992815ff8647eadb709f07067f4bc3a71`。
- ONNX SHA-256：`0c8716701f471067932b797eeb67c8e5db47c693c2557c881d7679ec12e21bc5`。
- RKNN 使用板端 Toolkit2 2.3.2 转换，`do_quantization=false`，mean=`[0,0,0]`，std=`[255,255,255]`。
- RKNN SHA-256：`56808e7b2d85de3cc59e953711d158f15b36bb4b33f12a05d8bd27e0a70d5813`。
- 原始标注：[COCO trainval2017 标注 ZIP](https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip)，成员 `annotations/instances_val2017.json`。JSON SHA-256：`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。

`dataset/samples.json` 保存 40 张图片、来源位置、原图哈希、预处理输入哈希和标注 ID。`dataset/instances_home40.json` 保留这些图片的全部类别标注与 crowd 信息，共 436 条实例。

抽样按瓶、杯、椅、沙发、床、餐桌、电视、笔记本电脑的顺序。每类候选按图片 ID 排序，排除已选图片，用同一个 `Random(42)` 各取 5 张。入选主类别须有非 crowd 且宽高大于零的框。图片许可筛为 COCO `license=4`，即 CC BY 2.0。

这只是家庭相关物品子集。缩略图审查见 `dataset/scene_review.json`：35 张室内，5 张室外。室内含餐饮、教室和办公空间，未确认真实家庭域。保留固定抽样结果，不因预测表现替换图片。

## 2. 前后处理与参考

RGB 输入使用等比例缩放及居中 letterbox。填充值为 114，OpenCV `INTER_LINEAR`。保留实际缩放宽高和四侧 padding。

- ONNX：NCHW、FP32、除以 255。
- RKNN：提交 NHWC UINT8。归一化由转换配置处理，不再次除以 255。
- 三个尺度各有三路输出：64 通道框分布、80 通道类别概率、1 通道 score-sum。
- 类别概率已经过 Sigmoid。score-sum 不作为 objectness。
- 公共后处理执行 DFL、置信度筛选、分类别 NMS 和原图坐标还原。坐标面积不加 1，按实际缩放比例还原。
- 正常检测阈值：confidence=0.25、NMS IoU=0.45、max_det=300。
- 子集 AP：confidence=0.001、NMS IoU=0.65。COCOeval 显式指定全部 40 个 image_id，包含零预测图片，保留全部 80 类。

同一 ONNX 的 ONNX Runtime CPU 输出是部署数值参考。COCO 原始标注是任务质量参考。两者分别验收，不能互相替代。

事先固定的门槛见 `config.json`：所有输出有限，每输出张量相对 L2≤0.02；同类别且 IoU≥0.90 的参考框匹配率≥95%；本子集 AP 的绝对下降≤0.01。该门槛用于本轮部署一致性，不是产品精度验收。

## 3. 当前可复现环境

本机独立环境位于项目根目录 `.build/yolo-rk3588-venv`。实际依赖锁定记录见 `evidence/host_requirements.lock.txt`。

板端实验目录：`/home/firefly/work/benchmarks/yolov8n_home_20261106_01`。
板端 Python：`/home/firefly/work/venvs/yolo-pose/bin/python`。
板端 Runtime：`/lib/librknnrt.so`，实际 SDK 和驱动版本须以每轮 `runtime.json` 为准。

没有复制登录凭据。板端登录使用操作者已有的 SSH 方式。

## 4. 复跑步骤

先把本目录复制到板端新目录，包含 models、dataset、scripts、vendor 和 config.json。不要复制本机虚拟环境。在副本的 config.json 中修改 run_id，以区分新实验。已有输出目录拒绝覆盖，复跑应换目录名。

在板端目录编译桥接库：

```bash
bash scripts/build_bridge.sh
```

设置线程环境。NPU mask=7，申请三核。未锁频，未绑 CPU 核。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/firefly/work/venvs/yolo-pose/bin/python
$PY scripts/run_board.py --mode verify --out runs/verify_repro
```

`verify` 逐样本检查图片、输入、标注和模型哈希。它会重新预处理，要求与本机冻结输入字节相同。随后比较九路 ONNX/RKNN 输出及检测结果。没有通过完整质量检查前，不采正式验收性能。

在安装了 pycocotools 的分析环境中计算子集 AP：

```bash
python scripts/finalize_verification.py --run runs/verify_repro
```

退出码 0 才表示本轮部署质量门槛通过。失败原因和原始结果仍保留。

本轮 `verify_01` 采集程序的退出码为 0，质量汇总程序 `finalize_verification.py` 的退出码为 2。分类概率及分数求和张量未通过相对 L2 门槛。官方 Lite2 单图复测与桥接程序九路输出完全相同。未修改模型或放宽门槛，后续采集范围调整为诊断。调整依据见 `evidence/diagnostic_policy.json`。

分别采正常轮和详细轮：

```bash
$PY scripts/run_board.py --mode baseline --out runs/baseline_repro
$PY scripts/run_board.py --mode detail --out runs/detail_repro
```

本轮实际结果目录为 `runs/baseline_diagnostic_01` 和 `runs/detail_diagnostic_01`。复跑后须保留质量状态。诊断结果不得标为正式验收通过。

两轮分别加载模型并预热 3 次。正常轮每图 10 次，每轮打乱 40 图顺序，共 400 次正式记录。详细轮每图一次，共 40 份正式节点报告。预热记录单列，模型加载不摊入单图耗时。

每次正常端到端计时包含读图/解码、letterbox/RGB、输入提交、run、输出获取与复制释放，以及 DFL/NMS/坐标还原。哈希校验、结果保存、绘图和质量计算在计时外。详细查询和写盘会扰动详细轮，不用其端到端值替代正常轮性能。

## 5. 证据和热点

- `start.json`：环境、配置与脚本哈希、后台进程快照。
- `runtime.json`：输入输出属性、运行库/驱动版本、内存查询与初始化。
- `warmup.json`：首轮及后续预热时间。
- `measurements.jsonl`：正式调用、主机时间戳、阶段时间、输出索引与状态。
- `resources.jsonl`：频率、温度、负载、CPU affinity 和进程峰值 RSS。
- `outputs/*.npz`：每样本首次正式调用的九路输出。
- `perf/*.perf.txt`：详细轮的原始节点报告。warmup 文件不进入样本统计。
- `graphs/`：实际 `.rknn` 的共享节点、张量和依赖图。
- `RK3588Bench.jsonl`：一行一个原始样本，每样本最多 4 个热点。

热点按真实编译图的 tensor_id 建边。名称只用于候选模块分组。候选拆成无向连通分量，再按节点时间之和排序。四组节点集合不得重叠。每份报告都核对 ID、类型和完整名称。

热点时间是详细轮的节点累计值，不是独立子图墙钟时间。尚未抓取热点内部边界张量值，因此 `subgraph_io_available=false`、`replayable=false`。

在完整实验目录执行以下命令，复核已保存的结果并重建统计与索引：

```bash
python scripts/check_saved_outputs.py
python scripts/analyze_run.py
```

`check_saved_outputs.py` 对比三个模式保留的首次正式输出。`analyze_run.py` 校验模型、输入、输出、配置与报告哈希，再重建 `summary.json`、共享图和 `RK3588Bench.jsonl`。`evidence/artifact_audit.json` 的 passed 仅表示证据完整性，质量状态另列。

分析新采集的目录时，显式指定三个模式。新目录中也要保存本轮质量判定和诊断范围说明。

```bash
python scripts/analyze_run.py --verify runs/verify_repro --baseline runs/baseline_repro --detail runs/detail_repro
```

测试按平台执行。本机运行 `python -m pytest tests -q --ignore=tests/test_bridge.py`。板端运行 `python tests/test_bridge.py -v`。桥接契约依赖 Linux、g++ 和配套头文件，不能在当前缺少 g++ 的 Windows 环境执行。

## 6. 使用边界与来源

没有锁定 CPU/NPU/DDR 频率。记录当时状态，但不宣称排除了调频和后台负载影响。未采功耗、设备 kernel 时间线、闭源内部算子函数耗时或跨后端成本矩阵，缺失项保留 null。

本轮是基线准备与热点采集。未据此宣称算子加速收益。今后的优化须保持输入、工作量和质量门槛一致，再用正常轮复测。

参考资料：用户提供的 RK3588Bench 构建 SOP、Qwen/YOLO11s-pose 交接文档、RKNN 脚本包、异构 Profile 规范。仅沿用通用方法，未使用 Qwen 的节点数量或历史性能作为本轮结果。

许可：Model Zoo 代码采用 Apache-2.0；不能将其自动套用到 YOLO 权重。Ultralytics 模型许可另行遵循 AGPL-3.0/企业许可约定。COCO 标注采用 CC BY 4.0，图片许可按各图元数据处理。原始 Flickr 来源已保存；图片元数据未提供完整作者署名，对外再分发前需补齐署名。
