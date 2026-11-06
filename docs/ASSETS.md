# 模型、数据与复现资产

当前SOP改为大型模型、数据集和张量由本地主机管理，端侧同步/共享后只读挂载。以下是此前已发布的资产归档和下载方法，继续保留用于历史复现；本次SOP更新没有新增大文件上传，也没有配置台式机存储。新流程见 [SOP.md](SOP.md) 与 [CONTAINERS.md](CONTAINERS.md)。

大文件保存在本仓库的 [GitHub Release](https://github.com/shunwendongan/kernel_bench_opt/releases/tag/qwen2.5-omni-3b-vcb-bench-assets-20261106)，代码保存在 `Qwen2.5-Omni-3B_VCB-Bench` 分支。GitHub 普通 Git 仓库不适合直接提交数 GB 权重；Release 附件是同一仓库的版本资产，单片小于 2 GiB。本项目分片上限为 1 GiB。

## 下载

```bash
git clone --branch Qwen2.5-Omni-3B_VCB-Bench \
  https://github.com/shunwendongan/kernel_bench_opt.git
cd kernel_bench_opt

python3 scripts/fetch_assets.py --list
python3 scripts/fetch_assets.py --output /data/kernel-bench-assets
```

默认下载模型、NPU 前缀、随机50样本、真实 capture 和原始对比证据。脚本支持断点续传，校验每片大小/SHA-256，按顺序解压，拒绝覆盖不同内容。需要同时留出压缩下载和解压文件空间；默认组建议至少约 9 GiB 可用空间。下载脚本仅需 Python 标准库。

默认组压缩总量 **3,514,539,760 bytes（约3.51 GB）**，解压文件总量 **4,223,542,360 bytes**；全部8包压缩总量 **3,823,540,635 bytes（约3.82 GB）**，解压文件总量 **4,856,572,553 bytes**。完整分片和文件SHA见 [manifest.json](../assets/manifest.json)。

| package ID | 默认下载 | 展开位置 | 内容 |
|---|---|---|---|
| `model-mnn` | 是 | `models/Qwen2.5-Omni-3B-MNN/` | 22个部署模型/权重/embedding/tokenizer/config文件，全部对齐原板SHA |
| `npu-prefix` | 是 | `models/rknn/` | 已转换前缀ONNX/RKNN、合同与验证记录 |
| `vcb-random50` | 是 | `data/vcb_random50/bundle/` | 混合类别50条样本、64段音频、输入/参考/来源清单 |
| `convtranspose-captures` | 是 | `captures/bigvgan/`、`captures/integrated/` | 真实 T52/120/136 输入输出、部署权重、packed buffer、mel/noise/wave |
| `benchmark-evidence` | 是 | `evidence/`、`source/sync_snapshot/` | 原始微基准、旧异步失败与后续同步版本成功的E2E记录、同步源码快照 |
| `reference-onnx` | 否 | `models/reference/` | 重新导出的完整BigVGAN FP32参考ONNX及目标子图，非历史MNN转换输入 |
| `runtime-rk3588` | 否 | `runtime/rk3588/` | 旧板aarch64可执行文件、MNN静态库、Mali/OpenCL用户态库及许可 |
| `cache-snapshots` | 否 | `cache_snapshots/` | 历史设备/驱动缓存，只作诊断，不安装到新板运行缓存目录 |

可单独取需要的包：

```bash
python3 scripts/fetch_assets.py \
  --packages reference-onnx runtime-rk3588 cache-snapshots \
  --output /data/kernel-bench-assets
```

`git clone` 本身只下载代码；上面的 `fetch_assets.py` 才下载 Release 大文件。全量12GB原HF checkpoint与全量VCB数据集未包含在本次上传中；当前已部署MNN模型、此次抽取的50条数据和优化所需真实张量均已提供。

## 模型身份与许可证

部署模型固定到 `taobao-mnn/Qwen2.5-Omni-3B-MNN@00dc2e9131a4bb325b43a47f4210dd6450116687`。先从原板只读计算22个文件SHA，再在GitHub Actions从固定公开上游下载，逐个匹配后打包。没有重新量化或用原始HF权重替换部署权重。

每包带来源、原文件摘要及许可。模型保留 Qwen RESEARCH LICENSE 和指定Notice；上游MNN模型卡的Apache标签不用于覆盖原模型条款。VCB数据附CC-BY-4.0、Tencent来源及16k/mono/PCM16转换说明。公开副本中的机器绝对路径改为占位路径，元数据内保留原摘要和公开副本摘要；二进制音频、张量、模型值不改动。

## 运行库

优先使用新板固件已有、与内核匹配的Mali/OpenCL和RKNN驱动。`runtime-rk3588` 是可选历史用户态快照，不能替代固件或作为通用系统库安装包；不要覆盖 `/lib`。包含的Mali及OpenCL库保留各自许可。

RKNN SDK文件未在本仓库再分发。可复用已有SDK，或直接从官方固定版本获取并核对原板SHA：

```bash
python3 scripts/fetch_rknn_vendor.py \
  --output /data/kernel-bench-assets/runtime/vendor-rknn
```

它从 `airockchip/rknn-toolkit2` 的 `v2.3.2` 对应commit获取 `librknnrt.so`、`rknn_api.h` 和厂商LICENSE，不更改系统文件。默认构建参数可以使用这里的 `include/` 和 `lib/librknnrt.so`，OpenCL仍指向新板实际驱动。

## 在新板接上代码

```bash
python3 scripts/make_config.py \
  --model-dir /data/kernel-bench-assets/models/Qwen2.5-Omni-3B-MNN \
  --cache-dir "$PWD/runs/cache" --output "$PWD/runs/config.json"

export OMNI_RKNN_AUDIO=/data/kernel-bench-assets/models/rknn/audio_conv_prefix_mnn4bit_fp16.rknn
```

将 [REPRODUCE.md](REPRODUCE.md) 中的 bundle 替换成 `/data/kernel-bench-assets/data/vcb_random50/bundle`，旧capture替换成 `/data/kernel-bench-assets/captures/bigvgan/operators`。已有真实数据可直接做单算子测试，不必再占用原板采集。

后续同步候选单独说明在 [SYNC_VARIANT.md](SYNC_VARIANT.md)。资产下载、解压和旧日志复算不运行模型；新板实际推理应由你显式执行对应命令。

下载后可先复算已有结果，无需板端推理：

```bash
python3 tools/analysis/analyze_results.py \
  --bench-root /data/kernel-bench-assets/evidence/bench \
  --e2e /data/kernel-bench-assets/evidence/e2e/paired_sync_20261106 \
  --out runs/downloaded-evidence-analysis
```
