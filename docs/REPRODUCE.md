# 在第二台 RK3588 上继续实验

先按 [README](../README.md) 编译并生成配置。本页所有推理命令都只在准备好的新板运行；工具不会连接或调度原板。输出放入被 Git 忽略的 `runs/`，每次使用新目录，避免覆盖原始证据。

## 1. 准备 VCB 输入

已有 bundle 时直接复用，并校验其 `SHA256SUMS`。重新准备时，使用原数据快照与固定的 VCB 源码 checkout：

```bash
python3 tools/data/prepare_random50.py \
  --source-root /data/vcb-assets \
  --output runs/selection \
  --seed 20261105 --count 50

sha256sum runs/selection/selection.jsonl

python3 tools/data/prepare_bundle.py \
  --selection runs/selection/selection.jsonl \
  --selection-sha256 REPLACE_WITH_THE_HASH_ABOVE \
  --source-root /data/vcb-assets \
  --vcb-source /path/to/VCB-Bench \
  --output runs/bundle
```

抽样范围是整个 VCB-Bench 各类别混合 50 条，不放回。`original_row_number` 是抽样后的顺序，并非原文件行号；原文件位置另存。原实验已按要求停止，不需要在迁移时自动重新跑满50条。协议适配不向当前用户 prompt 混入参考答案或用户音频转录，多轮历史助手内容按原协议保留。

新清单不含主机绝对路径，因此不能用旧序列化格式的 SHA 与它直接比较；应核对源文件哈希、源记录位置、种子和排序规则。

## 2. 在真实请求中采集 BigVGAN 输入

以下示例要求 bundle 的第44条仍是原语音样本；否则选择新的语音输出行，实际 chunk shape 以采集文件为准。

```bash
export LD_LIBRARY_PATH="$PWD/build/rk3588/runtime-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMNI_RKNN_AUDIO=/data/audio_conv_prefix_mnn4bit_fp16.rknn
export OMNI_TEST_SEED=20261105
mkdir -p runs/capture/native

OMNI_BIGVGAN_DECONV_TILE=0 \
OMNI_CAPTURE_DIR="$PWD/runs/capture/native" \
build/rk3588/bin/vcb_eval \
  runs/config.json runs/bundle/manifest.jsonl "$PWD/runs/bundle" \
  runs/capture/request 44 1 0
```

`native/` 包含 initial noise、逐块 mel、wave 和 `mel_sequence.jsonl`。真实 shape 应从这里取得，不能人为裁剪输入后称为真实末块。

## 3. 用相同 mel 采集旧/新 MNN 的算子数据

```bash
mkdir -p runs/capture/old runs/capture/new

OMNI_BIGVGAN_DECONV_TILE=0 \
DECONV_CAPTURE_DIR="$PWD/runs/capture/old" \
build/rk3588/bin/bigvgan_replay \
  /data/Qwen2.5-Omni-3B-MNN/bigvgan.mnn \
  runs/capture/native runs/capture/old capture

OMNI_BIGVGAN_DECONV_TILE=4 \
DECONV_CAPTURE_DIR="$PWD/runs/capture/new" \
BIGVGAN_DISPATCH_LOG="$PWD/runs/capture/new/dispatch.jsonl" \
build/rk3588/bin/bigvgan_replay \
  /data/Qwen2.5-Omni-3B-MNN/bigvgan.mnn \
  runs/capture/native runs/capture/new capture
```

`capture` 模式按每种实际 T 选一块，保存目标输入/输出和整个 BigVGAN 的 waveform。`DECONV_CAPTURE_DIR` 另保存 decoded weight 和实际 uploaded packed weight/bias。采集包含同步/读回，不能作为性能结果。

比较三种目标输入/输出：

```bash
for T in 52 120 136; do
  for tensor in input output; do
    python3 tools/analysis/compare_raw.py \
      "runs/capture/old/${tensor}_T${T}.f32" \
      "runs/capture/new/${tensor}_T${T}.f32" \
      --atol 0 --rtol 0 --bitwise || exit 1
  done
done

for oldwave in runs/capture/old/wave_*.f32; do
  python3 tools/analysis/compare_raw.py \
    "$oldwave" "runs/capture/new/$(basename "$oldwave")" \
    --atol 0 --rtol 0 --bitwise || exit 1
done
```

同时检查 dispatch 的三种 T 是否 `candidate=true, precision=0, tile=4`。其他 T 的回退是预期行为，不算候选覆盖验证。

## 4. 原始 kernel 与候选 kernel 微基准

```bash
python3 tools/benchmark/run_micro.py \
  --binary build/rk3588/bin/bench_deconv \
  --captures runs/capture/old \
  --mnn-source vendor/MNN \
  --output runs/micro-001 --phase formal

python3 tools/analysis/analyze_results.py \
  --bench-root runs/micro-001/bench --out runs/micro-001/analysis
```

`--dry-run` 只打印将执行的命令。正式配置是三种 shape × 两种 profiling 状态，预热3次、15对交替顺序。输入、权重、偏置、原输出都来自同一份 capture。分析器检查位级比较、重复次数、计时域和样本覆盖，拒绝把失败结果算成通过。

可附加同有效权重的 FP32 诊断：

```bash
python3 tools/analysis/analysis_correctness.py \
  --captures runs/capture/old --weights runs/capture/old \
  --candidate-captures runs/capture/new \
  --micro-old 52=runs/micro-001/bench/T52_profile0/old.nc4hw4.f16 \
  --micro-new 52=runs/micro-001/bench/T52_profile0/candidate.nc4hw4.f16 \
  --micro-old 120=runs/micro-001/bench/T120_profile0/old.nc4hw4.f16 \
  --micro-new 120=runs/micro-001/bench/T120_profile0/candidate.nc4hw4.f16 \
  --micro-old 136=runs/micro-001/bench/T136_profile0/old.nc4hw4.f16 \
  --micro-new 136=runs/micro-001/bench/T136_profile0/candidate.nc4hw4.f16 \
  --output runs/micro-001/correctness.json
```

## 5. 完整端到端 A/B

先阅读 [当前失败状态](STATUS.md)。目前暖机波形不一致是未解决问题，不能跳过正确性门。

```bash
export OMNI_TEST_SEED=20261105
export OMNI_RKNN_AUDIO=/data/audio_conv_prefix_mnn4bit_fp16.rknn
unset OMNI_PROFILE_PATH OMNI_RKNN_PERF_DIR OMNI_CAPTURE_DIR DECONV_CAPTURE_DIR BIGVGAN_DISPATCH_LOG

build/rk3588/bin/omni_ab \
  runs/config.json runs/bundle/manifest.jsonl "$PWD/runs/bundle" \
  runs/e2e-001 14 3 4

# 仅在 A/B 成功完成后执行；失败/不足3对会被分析器拒绝。
python3 tools/analysis/analyze_results.py \
  --bench-root runs/micro-001/bench --e2e runs/e2e-001 \
  --out runs/e2e-001/analysis
```

每个实例一次完整暖机，随后3对按 old/new、new/old 交替。计时从 `response` 开始，直到完整波形回调完成且 `generateWavform` 返回。记录加载耗时、TTFA、完整请求、音频时长及多个阶段。DiT 未单独计时的字段保留 `null`，不将零值视为 DiT 无开销。

诊断 capture 与干净计时必须分开。当前 runner 有意关闭完整 A/B 中的 capture；要定位未解决差异，可在独立诊断版本增加 old/new 各自的 native mel/noise 目录，并保留诊断构建身份。只有重新通过完整一致性后，才产生可解释的端到端性能结论。

## 6. 可选 profile 与参考 ONNX

独立诊断请求可设置 `OMNI_PROFILE_PATH` 与 `OMNI_RKNN_PERF_DIR`，采集后用 `tools/profiling/convert_mnn_profile.py` / `normalize_rknn_perf.py` 标准化。MNN callback 是包含等待的宿主耗时，不是纯 GPU event；RKNN 编译融合层不等同于一个原 ONNX 节点。

`tools/mapping/export_bigvgan_reference_draft.py --help` 提供本地原 HF 权重的参考导出入口。此前使用 `--fix-export-filter-channels` 成功导出。它输出的 FP32 ONNX 用于结构核对，不能冒充部署 MNN 的历史转换输入。ONNX 匹配脚本始终保留候选与未映射状态。
