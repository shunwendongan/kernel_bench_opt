# 性能分析工具包公共附录（Mac / Windows）

本附录提供 2026-11-06 的 Mac 和 Windows 性能分析工具包快照，供下载工具、阅读调用说明与 Codex Skill 说明，并参考离线 profile 分析及历史分析产物。

**[打开下载页面](https://github.com/shunwendongan/kernel_bench_opt/releases/tag/performance-analysis-toolkit-20261106)**。两个 ZIP 均作为本仓库的 GitHub Release 附件提供；`git clone` 不会自动下载它们。

## 下载文件

| 平台 | 完整压缩包 | 文件大小 | ZIP 内文件的解压大小之和 |
|---|---|---:|---:|
| Mac | [performance-analysis-toolkit-macos-20261106.zip](https://github.com/shunwendongan/kernel_bench_opt/releases/download/performance-analysis-toolkit-20261106/performance-analysis-toolkit-macos-20261106.zip) | 542,649,594 字节（517.51 MiB） | 2,043,975,545 字节（约 1.90 GiB） |
| Windows | [performance-analysis-toolkit-windows-20261106.zip](https://github.com/shunwendongan/kernel_bench_opt/releases/download/performance-analysis-toolkit-20261106/performance-analysis-toolkit-windows-20261106.zip) | 534,862,613 字节（510.08 MiB） | 2,033,444,482 字节（约 1.89 GiB） |

ZIP 内同时包含工具源码、使用说明、环境快照、参考 ONNX、样例及历史分析记录。请为压缩包、解压目录及重建环境预留额外磁盘空间。

附件仅改用便于命令行下载的英文文件名，ZIP 内容保持原样。原文件名、准确大小和 SHA-256 记录在 [manifest.json](manifest.json)；校验文件为 [SHA256SUMS](SHA256SUMS)。Windows 原始文件名中的 `Sklll` 拼写按来源保留在清单中。

已安装 GitHub CLI 的用户可选择下载对应平台：

```bash
# Mac
gh release download performance-analysis-toolkit-20261106 \
  --repo shunwendongan/kernel_bench_opt \
  --pattern 'performance-analysis-toolkit-macos-20261106.zip' \
  --pattern SHA256SUMS --dir toolkit-downloads
```

```powershell
# Windows PowerShell
gh release download performance-analysis-toolkit-20261106 --repo shunwendongan/kernel_bench_opt --pattern 'performance-analysis-toolkit-windows-20261106.zip' --pattern SHA256SUMS --dir toolkit-downloads
```

也可直接点击上表的 ZIP 链接下载。

## 校验下载

macOS：

```bash
cd toolkit-downloads
shasum -a 256 performance-analysis-toolkit-macos-20261106.zip
```

Windows PowerShell：

```powershell
Get-FileHash .\toolkit-downloads\performance-analysis-toolkit-windows-20261106.zip -Algorithm SHA256
```

将结果与同目录 `SHA256SUMS` 中对应文件名的值比较。

## 解压后从哪里开始

Mac 包：先阅读顶层的 `01_性能分析工具包_调用使用说明.md` 和 `02_Codex_性能分析_SKILL.md`，再进入 `RK3588_Mac_Analysis_Toolkit_副本/` 阅读 `README_zh_CN.md`。安装入口为 `install_mac.sh`，图形入口为 `launch_dashboard.command`。

Windows 包：先阅读外层的 `开始使用_Windows.md`，按说明执行 `install_windows.cmd -SkipPipUpgrade` 和 `install_codex_skill.cmd`。工具入口包括 `start_dashboard.cmd`、`launch_netron.cmd` 和 `run_toolkit.cmd python -m hotspot_pipeline --help`。这是 Windows 原生适配包。

两个包均带有既有虚拟环境、历史机器路径和少量历史私网设备标识，作为原始快照的溯源元数据保留。跨机器使用时请根据包内指南重建本机环境；这些环境快照不能保证直接迁移。Mac 底层命令行模块可在工具包目录用新环境检查：

```bash
python3.11 -m venv .venv-public
.venv-public/bin/python -m pip install -r requirements.txt
.venv-public/bin/python -m hotspot_pipeline --help
```

完整 Codex 报告流程还依赖 Skill 的脚本与安装位置，请按包内 Skill 说明配置。仅存在 `02_Codex_性能分析_SKILL.md` 说明副本，不代表所有外部 Skill 脚本已在当前安装位置可用。

## 分析范围与证据边界

工具围绕已有 profile 的离线解析、热点汇总、ONNX 来源候选映射，以及审阅后的子图提取。已核对的输入类型包括 RKNN profile、采集器标准 CSV（MNN/RKLLM）、TensorRT `trtexec` 的 `exportProfile` JSON、ONNX Runtime trace，以及导出的 Nsight Systems SQLite / Nsight Compute CSV。原生 `.nsys-rep`、`.ncu-rep` 需要先用对应工具导出。

包内历史记录和验证报告保留其原有适用范围。发布本附录只完成文件打包检查、上传与完整性校验，没有重新进行 Mac/Windows 安装验收或目标设备推理，也不据此增加性能、正确性或质量结论。profile 中的运行时单元、ONNX 来源候选和实际设备 kernel 应分别解释。

## 来源与许可

这是工具包快照，不是独立依赖齐备的源码发行版。主仓库的 [Apache-2.0 LICENSE](../../LICENSE) 不替代包内第三方软件、模型派生文件和数据各自的许可。

参考本仓库已有的 [第三方来源说明](../../docs/THIRD_PARTY.md)、[Qwen 许可](../../licenses/Qwen/LICENSE) 与 [Notice](../../licenses/Qwen/Notice)、[VCB-Bench 数据许可](../../licenses/VCB-Bench/LICENSE-Dataset.txt) 及 [来源说明](../../licenses/VCB-Bench/ATTRIBUTION.md)。使用和再分发时保留适用的原始版权、许可和来源记录。
