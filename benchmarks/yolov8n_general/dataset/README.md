# COCO val2017 通用检测数据

本目录使用 COCO val2017 原始 5000 张图片和完整实例标注。下载与读回的实际结果以 `../evidence/data_audit.json` 为准；`download_progress.jsonl` 是增量记录，下载未完成时不会被当作完整清单。

- `split.json`：下载前冻结的开发、验收与性能子集 ID，以及分层规则、种子、原始标注与旧清单哈希。已有内容不一致时脚本拒绝覆盖。
- `dev_samples.json`：首轮 40 图，保持其原图和 RGB 输入 SHA-256 一致。
- `acceptance_samples.json`：余下 4960 图，完整保留所有类别、crowd 与无目标图；不按类别或许可筛除。
- `samples.json`：80 图性能子集，来自验收集，按冻结顺序排列。
- `instances_val2017.json`：原始标注字节副本。
- `instances_dev40.json`、`instances_acceptance.json`、`instances_profile80.json`：对应图片的全部原始标注，保留完整 categories、licenses 和 info。
- `image_input_hashes.json`：按 image ID 排序的 5000 图原图/RGB 输入哈希索引。
- `images/`：原始图片；`inputs/`：只有性能子集的 80 个 HWC RGB uint8 640×640 张量。

分层使用原始 COCO supercategory，按下表顺序从排除开发集和已选图片的 ID 升序候选中调用 `random.Random(20261106).sample(..., 10)`。候选要求该层至少一个非 crowd 且框宽高大于零的实例。不限制 license=4。

| 分层 | COCO supercategory |
| --- | --- |
| person | person |
| traffic_vehicles | vehicle，包括 bicycle/car/motorcycle/airplane/bus/train/truck/boat |
| animals | animal |
| sports | sports |
| food_kitchen | food、kitchen |
| furniture | furniture |
| electronics | electronic |
| other_objects | outdoor、accessory、appliance、indoor |

八层类别定义覆盖全部 80 类；冻结的 80 张性能图片实际包含 65 类、599 条标注。4960 图验收集覆盖全部 80 类，并包含 48 张零标注图片。类别不能证明真实拍摄场景，所有图片的 `scene_review` 初值为“未审图”，`scene_label_source` 为 null。

每条成功记录保留原始 COCO 图像元数据与许可 URL，以及原图 SHA、RGB 输入 SHA、letterbox transform、类别/标注 ID、crowd/非 crowd 实例数和原图尺度。尺度统计用 segmentation area 的互斥区间：small <1024，medium ≥1024 且 <9216，large ≥9216；COCOeval 仍应使用其标准评价区间。

不保存 `.u8` 的图片可按 `image` 路径读取，用 `scripts/yolo_common.py:letterbox` 重建，并核对 `input_sha256` 与 `transform`。输入为 RGB、HWC、uint8、640×640，线性缩放、114 补边，无归一化。预处理源码和 Python/OpenCV/NumPy/Pillow 版本见审计文件。

在工作区根目录运行：

```bash
".build/yolo-rk3588-venv/Scripts/python.exe" -B -u benchmarks/yolov8n_general/scripts/prepare_general.py --workers 16 --attempts 5
".build/yolo-rk3588-venv/Scripts/python.exe" -B -m pytest benchmarks/yolov8n_general/tests/test_prepare_general.py -q -p no:cacheprovider --basetemp=C:/Users/Hhhfg/AppData/Local/hermes/cache/scratch/yolov8n_general_pytest
```

下载 URL 固定为 `https://s3.amazonaws.com/images.cocodataset.org/val2017/{file_name}`。每个线程复用 HTTPS session，失败重试同一 ID，不换图。每条 JSONL 记录写入后 flush/fsync；重跑校验成功缓存，末行未写完的内容保存为 `*.interrupted_tail.*` 后恢复，日志中间损坏会报错。最后再次从磁盘逐图严格 Pillow 解码、OpenCV 解码与尺寸验证、重建 RGB 哈希和 transform，并核对性能张量。失败条目仍出现在其原定清单和审计文件中，进程返回非零。

原始标注 SHA-256：`e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`。
冻结拆分 SHA-256：`545d1f0fd07cccc2169d954b3e85d81dec6cd08ceb3ffae39b68ca73d8492cff`。
