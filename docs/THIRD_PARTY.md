# 许可与修改来源

仓库保留原有 Apache-2.0 许可。模型权重、选中数据和可选GPU运行库通过Release分发，各自适用随包提供的原项目条款；RKNN SDK通过官方固定版本获取，完整MNN源码仍从上游取得。

## MNN

源项目：[alibaba/MNN](https://github.com/alibaba/MNN/tree/7e3c6fb5225d3ccf22807ee641332669df34650d)，固定 commit `7e3c6fb5225d3ccf22807ee641332669df34650d`。原版权为 Alibaba Group Holding Limited，Apache-2.0，许可副本位于 `dependencies/MNN_LICENSE.txt`。

下列源码包含本实验修改（2026-11-05/06），保留原文件头：

| 文件/补丁 | 修改 |
|---|---|
| `src/mnn/DeconvBufExecution.candidate.cpp` | 当前部署权重/dispatch 捕获，guarded tile4 分派和原 kernel 回退 |
| `src/mnn/DeconvBufExecution.task.hpp` | 每 execution 的候选 KernelWrap 缓存 |
| `src/mnn/omni.cpp` | CPU/NPU/GPU 分工、DiT 高精度、可选 profile/capture、固定噪声种子 |
| `src/mnn/sampler.repeat.cpp` | 可选种子与逐请求采样器重置 |
| `scripts/patches/01-llm.hpp.patch`、`02-omni.hpp.patch` | Talker context 接口和音频前缀成员 |
| `scripts/patches/03-llm.cpp.patch` | 可关闭的 profile hook |
| `scripts/patches/04-CMakeLists.txt.patch` | 实验构建目标入口 |

OpenCL 候选复用原 `deconv_2d` 的参数接口、buffer 合同和逐输出累加次序；基准原 kernel 在新板从固定 MNN checkout 读取。仓库保存候选源码与测试程序。MNN 的 `schema/private/`、`source/internal/` 不属于此交付范围。

## VCB-Bench

`tools/data/` 为任务适配工具，在用户指定的 [Tencent/VCB-Bench](https://github.com/Tencent/VCB-Bench/tree/542990571e5b6e3fbce4a318492aec5607aaee3c) checkout 中读取 dataset 常量与纯 prompt 函数，记录和校验其来源。代码分支没有复制VCB源码；Release随机50包包含标注与音频，并附CC-BY-4.0许可、来源和变更说明。

## 外部模型和驱动

模型与可选OpenCL/Mali用户态快照已附在Release，并保留其原许可证。RKNN runtime/API headers由新板已有SDK或官方固定版本获取。本仓库Apache许可不覆盖这些外部资产。参考ONNX导出工具只使用本地模型，默认设置离线模式。
