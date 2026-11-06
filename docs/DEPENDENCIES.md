# 依赖与外部资产

版本身份见机器可读的 [versions.json](../dependencies/versions.json)。代码分支保存源码与说明，大文件通过同仓库Release提供，见 [ASSETS.md](ASSETS.md)。安装命令需要在新板按实际已有环境执行。

| 组件 | 原实验身份 / 要求 | 用途 |
|---|---|---|
| MNN | 3.6.2，commit `7e3c6fb5225d3ccf22807ee641332669df34650d` | C++ 引擎、OpenCL 原 kernel、public headers |
| RKNN Runtime / API headers | 2.3.2，原板 NPU driver 0.9.8 | NPU 音频前缀；复用厂商 SDK/runtime |
| OpenCL | 原板 Mali-G610 r0p0，报告 driver 3.0 | GPU 执行；新板需要支持 FP16 的兼容驱动 |
| 构建工具 | CMake、Git、C++17 编译器、pthread/dl | Linux/aarch64 源码构建 |
| Python | 核心脚本 Python3.10+，已固定分析环境用 Python3.11 | 构建、配置、数据准备、分析 |
| NumPy | 本机分析环境 2.4.6 | float/half 位比较和参考计算 |
| ffmpeg | 从新板 PATH 或 `--ffmpeg` 指定 | VCB 音频无裁剪转 mono16k PCM16 |
| VCB 源码 | `Tencent/VCB-Bench` commit `542990571e5b6e3fbce4a318492aec5607aaee3c` | 数据协议和 prompt 函数，工具会校验相关源文件 SHA |

只构建和运行 MNN C++ 不需要 PyTorch、Transformers 或 Python MNN binding。常规日志统计 `analyze_results.py` 仅需标准库；`analysis_correctness.py` / `compare_raw.py` 需要 NumPy。

`dependencies/requirements-analysis.txt` 记录本机已核对的分析环境直接依赖；`requirements-reference-export.txt` 记录此前参考 ONNX 导出所用的另一环境。两个环境的 NumPy/ONNX 版本不同，应分开使用。导出依赖的特定 wheel 是否支持目标系统需在安装时确认；它们不要求安装到推理板上。复用已有匹配环境优先。

## 从 Release 获取模型与测试数据

MNN 模型来源：[taobao-mnn/Qwen2.5-Omni-3B-MNN](https://huggingface.co/taobao-mnn/Qwen2.5-Omni-3B-MNN/tree/00dc2e9131a4bb325b43a47f4210dd6450116687)。需保留同一快照的完整依赖关系，包括：

- `llm.mnn/.weight`、`llm_config.json`、tokenizer、embedding 等语言资产；
- `audio.mnn/.weight`、`talker.mnn/.weight`、Talker embeddings/配置；
- `predit.mnn/.weight`、`dit.mnn/.weight`、`bigvgan.mnn/.weight`、speaker 资产。

配置检查只核实若干必需文件存在，完整资产由 MNN 加载时检查。为了复现三设备分工，另需已转换的 `audio_conv_prefix_mnn4bit_fp16.rknn`，通过 `OMNI_RKNN_AUDIO` 指定。该模型来自部署 MNN 4bit 解码权重与 tanh GELU，不能用原 HF FP32/BF16 权重生成的新图冒充同一部署前缀。

VCB 的原始数据根目录应含 `data/downloaded_datasets/vcb_bench/`，每条 `audio_path` 从数据根目录解析。也可复用已有准备完成的 bundle（含 `manifest.jsonl`、音频、references 和 SHA256SUMS）。**克隆后运行 `scripts/fetch_assets.py` 即可取得这些资产；仅 `git clone` 不下载Release附件。**

## 额外分析工具

`tools/profiling/` 和 `match_bigvgan_reference.py` 可独立运行。`bind_npu_profile.py` 需要显式 `--toolkit` 指向已有 `RK3588_Mac_Analysis_Toolkit` 和 `--source-map` 指向人工审阅过的映射；该 toolkit 不作为当前仓库的隐式依赖。没有它也可完成 kernel/MNN A/B。

`tools/npu/` 保存此前音频前缀工作源码；`prepare_mnn_prefix.py` 依赖 PyTorch、NumPy、ONNX 和已解码权重/三份 mel，`convert_rknn.py` 依赖已有 RKNN Toolkit2 环境。它们不在默认 BigVGAN 构建/测试流程内，不会自动运行。MNN 节点索引、GELU 近似和前后处理合同限定于原快照，换模型后需要重新核对。
