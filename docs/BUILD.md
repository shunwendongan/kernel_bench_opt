# 在 RK3588 原生构建

容器入口见 [CONTAINERS.md](CONTAINERS.md)。本页保留不使用Docker时的构建方法，复用板上的工具链、模型和驱动。

需要 Linux/aarch64、C++17、CMake、Git、Python 3.10+、可用的Mali OpenCL，以及RKNN 2.3.2 runtime/header。版本见 [DEPENDENCIES.md](DEPENDENCIES.md)。

```bash
git clone --branch Qwen2.5-Omni-3B_VCB-Bench \
  https://github.com/shunwendongan/kernel_bench_opt.git
cd kernel_bench_opt

# 若要复现完整A/B通过的同步候选，在构建前选择它。
git apply --check experiments/predit-sync.patch
git apply experiments/predit-sync.patch

mkdir -p vendor
git clone https://github.com/alibaba/MNN.git vendor/MNN
git -C vendor/MNN checkout 7e3c6fb5225d3ccf22807ee641332669df34650d

python3 scripts/build_rk3588.py \
  --mnn-source "$PWD/vendor/MNN" --build-dir "$PWD/build/rk3588" \
  --rknn-include /path/to/rknn/include \
  --rknn-lib /path/to/librknnrt.so \
  --opencl-lib /path/to/libOpenCL.so.1 --jobs 4

export LD_LIBRARY_PATH="$PWD/build/rk3588/runtime-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 scripts/make_config.py \
  --model-dir /data/kernel-assets/models/Qwen2.5-Omni-3B-MNN \
  --cache-dir "$PWD/runs/cache" --output "$PWD/runs/config.json"
export OMNI_RKNN_AUDIO=/data/kernel-assets/models/rknn/audio_conv_prefix_mnn4bit_fp16.rknn
export OMNI_PREDIT_SYNC=1
export OMNI_TEST_SEED=20261105
```

构建脚本修改专属的MNN checkout，拒绝覆盖未知改动。`runtime-lib/libOpenCL.so` 只是在构建目录创建的别名。构建产物位于 `build/rk3588/bin/`，编译成功仍需完成 [多级复现](REPRODUCE.md) 的正确性与性能验证。

跳过同步补丁则是原默认异步版本，该版本的历史完整A/B波形门失败。同步版本的证据和适用范围见 [SYNC_VARIANT.md](SYNC_VARIANT.md)。`OMNI_PREDIT_SYNC=0` 仍启用，禁用应删除环境变量。

仅检查源码准备：

```bash
python3 scripts/prepare_mnn.py --mnn-source "$PWD/vendor/MNN" --dry-run
```

迁移版和新Docker配方尚未在第二台板上编译/推理；历史结果不替代新板验收。
