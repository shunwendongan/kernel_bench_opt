# RK3588 容器运行入口

本目录提供 Linux/arm64 的**开发与评测环境配方**。当前验收状态：Dockerfile、启动参数和源码准备流程已做静态/本地检查；当前主机没有 Docker，本次未构建镜像、未在新板编译或推理。历史性能数字来自原板裸机进程，不能直接当作容器性能。

## 1. 机器与数据的关系

- 本地主机：维护大型模型、数据集、capture，运行分析工具包和 agent。
- RK3588：运行 Docker daemon、容器和真实 CPU/GPU/NPU 推理，使用板上内核驱动。
- GitHub：保存代码、设计、体积受控的 profile/评测记录、报告和资产身份。

Docker bind mount 读取 **daemon 所在机器**的文件。台式机的数据要先同步到板上的缓存目录，或由板子挂载共享目录，再传给 `--assets`；直接在 Mac 上指定一个路径不会把它自动挂到远端板子。本次不配置台式机存储、网络共享或数据同步服务。

## 2. 构建镜像

在准备好的 RK3588 上执行；也可以在支持 arm64 构建的主机生成镜像后转交端侧。性能测试必须在真实板子运行。

```bash
git clone --branch Qwen2.5-Omni-3B_VCB-Bench \
  https://github.com/shunwendongan/kernel_bench_opt.git
cd kernel_bench_opt
TASK_REVISION=$(git rev-parse HEAD)
docker build --platform linux/arm64 \
  --build-arg SOURCE_REVISION="$TASK_REVISION" \
  -f docker/Dockerfile.rk3588 \
  -t "kernel-bench-omni:$TASK_REVISION" .
```

Dockerfile 固定 Ubuntu 22.04 基础镜像内容摘要，MNN 源码固定 commit；APT 包使用构建时的 Ubuntu 仓库，尚未冻结到逐包快照。实际版本表保存在镜像 `/opt/image-metadata/packages.tsv`。比较实验应使用同一镜像 ID，并记录镜像、编译和驱动身份。

镜像只包含构建工具、代码和分析所需 NumPy。模型、数据、设备缓存、板载驱动和 RKNN 专有库从外部提供。`.dockerignore` 采用白名单，不把本地大文件送进构建上下文。

## 3. 准备四个板端路径

| 参数 | 内容 |
|---|---|
| `--assets` | 板上可读资产目录，模型/数据/采集张量，容器内 `/assets` 只读 |
| `--work` | 本次独立构建和结果目录，容器内 `/work` 可写 |
| `--rknn-sdk` | 内含 `include/rknn_api.h` 与 `lib/librknnrt.so`，容器内只读 |
| `--mali-lib` | 本固件的实际 `libmali` 文件，须提供 OpenCL API，容器内只读 |

资产目录可沿用此前发布包的布局：

```text
assets/
  models/Qwen2.5-Omni-3B-MNN/   当前部署模型
  models/rknn/audio_conv_prefix_mnn4bit_fp16.rknn
  data/vcb_random50/bundle/     manifest.jsonl、音频、references
  captures/bigvgan/operators/  T52/120/136输入、输出、packed权重
  captures/bigvgan/native/     mel_sequence.jsonl及对应mel
```

设备节点由用户确认后显式传入 `--device`，可重复。下面的 `/dev/mali0`、`/dev/dri/renderD129` **只是示例**；不同内核/固件的 NPU 设备路径可能不同，请以当前 SDK 和板子为准。启动器不自动猜测设备，不使用 `--privileged`。

```bash
# 以下路径需替换为这台板子上的实际文件。
mkdir -p /data/kernel-work/omni-sync-001
python3 docker/run_rk3588.py \
  --image "kernel-bench-omni:$TASK_REVISION" \
  --assets /data/kernel-assets \
  --work /data/kernel-work/omni-sync-001 \
  --rknn-sdk /data/sdk/rknn \
  --mali-lib /usr/lib/aarch64-linux-gnu/libmali.so.1 \
  --device /dev/mali0 --device /dev/dri/renderD129 \
  --interactive -- /bin/bash
```

可在 `--interactive` 前加 `--dry-run`，仅检查已有路径并打印 Docker 命令，不启动容器。真实启动前要求 Linux/arm64、已存在的字符设备和镜像；每次调用记录 image ID 和参数到工作目录。

容器共享板上内核；用户态 Mali/RKNN 库仍需与固件/ABI 匹配。容器内可先用 `ldd /opt/board-gpu/libmali.so.1`、`ldd /opt/board-rknn/lib/librknnrt.so` 检查依赖。该配方提供常见 EGL/GBM/X11/Wayland 动态库；具体固件缺少的依赖须按实际错误补充，不能据此宣称所有 RK3588 镜像都已兼容。示例默认以容器 root 写工作目录，输出归属需按团队环境处理。

## 4. 在容器内编译

```bash
# 显式选择历史完整A/B通过的PreDiT同步候选。
bash /opt/kernel-bench/docker/build_inside.sh sync 4
```

脚本将当前镜像源码复制到新的 `/work/source`，只在副本应用 `predit-sync.patch`；MNN 固定为 `7e3c6fb5225d3ccf22807ee641332669df34650d`。如果 `/work/MNN` 已是此版本专用 checkout，则复用；否则需要联网获取源码。完全离线时可预先在工作目录放好该 checkout。

`sync` 构建完成后，推理还需设置 `OMNI_PREDIT_SYNC=1`。`async` 构建保留原路径，用不同 `--work` 目录隔离。脚本拒绝覆盖已有 `/work/source` 和 `/work/build`；这让每个构建绑定一个源码/镜像版本。编译只准备二进制，不运行模型。

```bash
python3 /work/source/scripts/make_config.py \
  --model-dir /assets/models/Qwen2.5-Omni-3B-MNN \
  --cache-dir /work/cache --output /work/config.json
export OMNI_RKNN_AUDIO=/assets/models/rknn/audio_conv_prefix_mnn4bit_fp16.rknn
export OMNI_PREDIT_SYNC=1
export OMNI_TEST_SEED=20261105
```

DiT 的 FP32 设置由现有实现保留。新板不得为通过性能测试而更换权重、精度、输入、解码长度或步数。

## 5. 分级执行；不自动跑满测试集

下面命令会进行真实推理，仅在准备好、没有其他性能工作负载干扰的目标板执行。

```bash
python3 /work/source/tools/benchmark/run_micro.py \
  --binary /work/build/bin/bench_deconv \
  --captures /assets/captures/bigvgan/operators \
  --mnn-source /work/MNN --output /work/micro-001 --phase formal
```

然后按 [REPRODUCE.md](REPRODUCE.md) 第3节执行同mel的旧/新BigVGAN capture和bitwise比较；模型替换为 `/assets/models/Qwen2.5-Omni-3B-MNN/bigvgan.mnn`，已有mel目录为 `/assets/captures/bigvgan/native`，二进制为 `/work/build/bin/bigvgan_replay`。这一层通过后再做完整A/B：

```bash
unset OMNI_PROFILE_PATH OMNI_RKNN_PERF_DIR OMNI_CAPTURE_DIR DECONV_CAPTURE_DIR BIGVGAN_DISPATCH_LOG
/work/build/bin/omni_ab /work/config.json \
  /assets/data/vcb_random50/bundle/manifest.jsonl \
  /assets/data/vcb_random50/bundle /work/e2e-001 14 3 4

python3 /work/source/tools/analysis/analyze_results.py \
  --bench-root /work/micro-001/bench --e2e /work/e2e-001 \
  --out /work/analysis-001
```

`14` 只对应原bundle中的语音样本，换数据先核对。完整A/B使用两个常驻实例，内存不足时应记录失败并调整实验设计，不能以不同工作量对比。诊断采集另开run，不能把带详细profiling的时延当干净性能。

官方依据：[设备透传](https://docs.docker.com/engine/containers/run/)、[bind mount所在主机](https://docs.docker.com/engine/storage/bind-mounts/)、[arm64镜像构建](https://docs.docker.com/build/building/multi-platform/)。
