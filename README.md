# kernel_bench_opt

RK3588 上 Qwen2.5-Omni-3B 的异构部署、BigVGAN ConvTranspose 优化与新旧实现对比实验代码。用于在另一台 RK3588 上重新构建、采集真实输入、验证正确性并继续优化。

**当前状态：三种真实 shape 的单算子和 MNN BigVGAN 集成正确性已通过；完整 Omni 端到端 A/B 的暖机波形不一致，正式计时未开始。候选默认关闭。** 见 [实验状态与指标](docs/STATUS.md)。

本次迁移只包含代码、配置模板、依赖版本和操作说明。模型权重、VCB 音频/标注、采集张量、缓存及编译二进制需要另行提供。仓库工具不连接旧板，不管理其他人的测试进程。

## 目录

| 路径 | 内容 |
|---|---|
| `kernels/` | 同精度 tile4 OpenCL kernel、内嵌源码和新旧单算子微基准 |
| `src/mnn/` | MNN 执行器候选、原实现回退、Omni capture/profile、RKNN prefix 适配 |
| `src/apps/` | `bigvgan_replay`、`vcb_eval`、完整端到端 `omni_ab` |
| `scripts/` | 固定版本 MNN 的源码准备、全新构建、配置生成 |
| `tools/benchmark/` | 串行微基准和本次启动进程的环境记录 |
| `tools/analysis/` | FP32 参考诊断、bitwise 比较、原始 A/B 日志统计 |
| `tools/data/` | VCB 全类别不放回随机抽样与输入准备；不含数据 |
| `tools/profiling/`、`tools/mapping/` | profile 标准化、参考 ONNX 导出与结构候选匹配 |
| `tools/npu/` | 音频 Conv1→GELU→Conv2→GELU 的历史导出、转换和验证源码 |
| `dependencies/` | 外部依赖身份、可选 Python 依赖、MNN 许可 |

## 在新 RK3588 上开始

需要 Linux/aarch64、C++17 编译器、CMake、Git、Python 3.10+、可用的 Mali OpenCL 驱动，以及 RKNN Runtime 2.3.2 和对应 `rknn_api.h`。优先复用板上已有 SDK 和 Python 环境。完整依赖和外部资产清单见 [DEPENDENCIES.md](docs/DEPENDENCIES.md)。

```bash
git clone https://github.com/shunwendongan/kernel_bench_opt.git
cd kernel_bench_opt

# 使用专属于本实验的 MNN checkout，构建脚本会修改其中的任务文件。
mkdir -p vendor
git clone https://github.com/alibaba/MNN.git vendor/MNN
git -C vendor/MNN checkout 7e3c6fb5225d3ccf22807ee641332669df34650d

# 将 SDK/库路径替换为新板已有的实际位置。
python3 scripts/build_rk3588.py \
  --mnn-source "$PWD/vendor/MNN" \
  --build-dir "$PWD/build/rk3588" \
  --rknn-include /path/to/rknn/include \
  --rknn-lib /path/to/librknnrt.so \
  --opencl-lib /path/to/libOpenCL.so.1 \
  --jobs 4

export LD_LIBRARY_PATH="$PWD/build/rk3588/runtime-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python3 scripts/make_config.py \
  --model-dir /data/Qwen2.5-Omni-3B-MNN \
  --cache-dir "$PWD/runs/cache" \
  --output "$PWD/runs/config.json"
```

构建产物在 `build/rk3588/bin/`。`runtime-lib/` 中的 `libOpenCL.so` 是指向显式指定库的本地别名，供 MNN 动态加载；不会改动系统库。

源码准备可先单独检查：

```bash
python3 scripts/prepare_mnn.py --mnn-source "$PWD/vendor/MNN" --dry-run
```

新脚本已经通过源码准备、Python 静态检查和应用 C++ 语法检查；**本次代码迁移没有执行新板 CMake 配置、ARM 编译链接或推理**。历史板端验证使用的是旧板隔离构建，不能代替新板验收。

## 测试顺序

1. 按 [复现步骤](docs/REPRODUCE.md) 准备独立模型/数据，并采集真实 mel、算子输入和部署权重。
2. 运行单算子 `bench_deconv`，先检查正确性，再比较关闭 profiling 的计时。
3. 用同一份真实 mel 回放完整 BigVGAN，核对输入、目标算子输出和波形。
4. 运行完整 `omni_ab`。所有暖机与测量请求必须通过 text token、speech token 和 float 波形一致性检查；失败会停止正式比较。

```bash
# 此步骤会在执行命令的板子上进行推理；只在准备好的新板执行。
export OMNI_RKNN_AUDIO=/data/audio_conv_prefix_mnn4bit_fp16.rknn
export OMNI_TEST_SEED=20261105
build/rk3588/bin/omni_ab \
  runs/config.json /data/vcb-bundle/manifest.jsonl /data/vcb-bundle \
  runs/e2e-trial-001 14 3 4
```

`14` 是当前 bundle 中的 `original_row_number`，必须确认它对应语音输出样本；更换 bundle 后不能假定相同行号还是相同内容。程序使用两个常驻模型实例，需留足内存。当前完整 E2E 波形失败尚待定位，不能把该命令视为已通过的优化验收。

## 替换机制与继续优化

沿用 MNN `Deconvolution` 注册，在 `DeconvBufExecution::onEncode` 中按明确条件选择候选 kernel。这里是源码集成并重新编译 MNN，不是通用动态插件注册。

- `OMNI_BIGVGAN_DECONV_TILE=4`：仅目标节点、匹配结构、precision0、`T∈{52,120,136}` 时启用候选。
- 未设置或设为 `0`：使用原 kernel；不匹配条件的节点/shape 同样回退。
- FP16 存储和累加顺序、packed 权重及偏置保持不变。DiT 单独保持 FP32。

修改 `kernels/bigvgan_deconv_s5_tiled.cl` 后运行 `python3 tools/embed_kernel.py`，再从新的独立 MNN checkout 构建。准备脚本会拒绝覆盖未知的本地改动。验收合同见 [OPERATOR.md](docs/OPERATOR.md)。

## 许可与来源

保留仓库原有 [Apache-2.0 LICENSE](LICENSE)。MNN 衍生文件保留原版权声明与 [MNN 许可副本](dependencies/MNN_LICENSE.txt)，修改清单见 [THIRD_PARTY.md](docs/THIRD_PARTY.md)。MNN、VCB、模型和 SDK 的来源版本见依赖清单；未重新分发其模型、数据或厂商运行库。
