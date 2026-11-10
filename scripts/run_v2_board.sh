#!/usr/bin/env bash
# Board-side gates for candidate set v2 (bigvgan-deconv1d-v2 + rknn-audio-prefix-v2).
# Run on the RK3588 after scripts/build_rk3588.py. Every stage writes a new
# directory under runs/ and never overwrites earlier evidence.
#
#   scripts/run_v2_board.sh l1          # kernel: bit-pattern gate + variant/LWS sweep, all ups.* shapes
#   scripts/run_v2_board.sh l1-capture  # same for ups.0 using real captured input/packed weights
#   scripts/run_v2_board.sh l2          # same-mel BigVGAN replay: baseline vs v2, bitwise waveforms
#   scripts/run_v2_board.sh env         # print the environment for full-request A/B
#
# Variables (defaults in brackets):
#   BUILD [build/rk3588]  MNN_SOURCE [vendor/MNN]  BIGVGAN_MNN [required for l2]
#   CAPTURE_NATIVE [runs/capture/native]  CAPTURE_OLD [runs/capture/old]
#   T_LIST ["52 108 120 136"]  VARIANTS ["2:2,4:1,4:2,4:4,8:1,8:2"] (TT:OCB)
#   LWS ["4,8,1;4,16,1;8,8,1;2,16,1;2,32,1;1,32,1;8,4,1;16,4,1;0,0,0"]  REPS [15]  WARMUP [3]
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${BUILD:-build/rk3588}
MNN_SOURCE=${MNN_SOURCE:-vendor/MNN}
T_LIST=${T_LIST:-"52 108 120 136"}
VARIANTS=${VARIANTS:-"2:2,4:1,4:2,4:4,8:1,8:2"}
LWS=${LWS:-"4,8,1;4,16,1;8,8,1;2,16,1;2,32,1;1,32,1;8,4,1;16,4,1;0,0,0"}
REPS=${REPS:-15}
WARMUP=${WARMUP:-3}
STAMP=$(date +%Y%m%d-%H%M%S)
BENCH="$BUILD/bin/bench_deconv_v2"
BASE_CL="$MNN_SOURCE/source/backend/opencl/execution/cl/deconv_2d.cl"
export LD_LIBRARY_PATH="$PWD/$BUILD/runtime-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# BigVGAN upsamplers of Qwen2.5-Omni token2wav: K S P Cin Cout time-multiplier-vs-mel-T
SHAPES=(
  "ups0 11 5 3 1536 768 1"
  "ups1 7 3 2 768 384 5"
  "ups2 4 2 1 384 192 15"
  "ups3 4 2 1 192 96 30"
  "ups4 4 2 1 96 48 60"
  "ups5 4 2 1 48 24 120"
)

bench() {  # out_dir extra-args...
  local out=$1; shift
  mkdir -p "$(dirname "$out")"
  "$BENCH" --baseline "$BASE_CL" --v2 kernels/v2/bigvgan_deconv1d_v2.cl \
    --tile4 kernels/bigvgan_deconv_s5_tiled.cl --precision fp16 --axis h \
    --variants "$VARIANTS" --lws "$LWS" --reps "$REPS" --warmup "$WARMUP" --profile 0 \
    --out "$out" "$@"
}

case "${1:-}" in
  l1)
    OUT="runs/v2-l1-$STAMP"; mkdir -p "$OUT"
    for spec in "${SHAPES[@]}"; do
      read -r name K S P CI CO MUL <<<"$spec"
      for T0 in $T_LIST; do
        T=$((T0 * MUL))
        echo "== $name K$K S$S P$P $CI->$CO T=$T"
        bench "$OUT/${name}_T${T}.jsonl" --K "$K" --S "$S" --P "$P" --ci "$CI" --co "$CO" --T "$T" \
          | tee "$OUT/${name}_T${T}.txt"
      done
    done
    python3 tools/verify/summarize_v2.py "$OUT" | tee "$OUT/SUMMARY.md"
    ;;
  l1-capture)
    CAPTURE_OLD=${CAPTURE_OLD:-runs/capture/old}
    OUT="runs/v2-l1capture-$STAMP"; mkdir -p "$OUT"
    for f in "$CAPTURE_OLD"/input_T*.f32; do
      T=$(basename "$f" .f32); T=${T#input_T}
      echo "== ups0 real capture T=$T"
      bench "$OUT/ups0_T${T}.jsonl" --K 11 --S 5 --P 3 --ci 1536 --co 768 --T "$T" \
        --input "$f" --input-layout nchw-f32 \
        --weights "$CAPTURE_OLD/packed_weight.bin" --bias "$CAPTURE_OLD/packed_bias.bin" \
        | tee "$OUT/ups0_T${T}.txt"
    done
    python3 tools/verify/summarize_v2.py "$OUT" | tee "$OUT/SUMMARY.md"
    ;;
  l2)
    : "${BIGVGAN_MNN:?set BIGVGAN_MNN=/path/to/Qwen2.5-Omni-3B-MNN/bigvgan.mnn}"
    CAPTURE_NATIVE=${CAPTURE_NATIVE:-runs/capture/native}
    OUT="runs/v2-l2-$STAMP"; mkdir -p "$OUT/old" "$OUT/new"
    OMNI_BIGVGAN_DECONV_TILE=0 \
      "$BUILD/bin/bigvgan_replay" "$BIGVGAN_MNN" "$CAPTURE_NATIVE" "$OUT/old" capture
    OMNI_BIGVGAN_DECONV_TILE=0 OMNI_BIGVGAN_DECONV_V2=1 OMNI_BIGVGAN_DECONV_V2_NODES=ups \
    BIGVGAN_DISPATCH_LOG="$PWD/$OUT/new/dispatch.jsonl" \
      "$BUILD/bin/bigvgan_replay" "$BIGVGAN_MNN" "$CAPTURE_NATIVE" "$OUT/new" capture
    status=0
    for oldwave in "$OUT"/old/wave_*.f32; do
      python3 tools/analysis/compare_raw.py "$oldwave" "$OUT/new/$(basename "$oldwave")" \
        --atol 0 --rtol 0 --bitwise || status=1
    done
    echo "dispatch (expect v2=true for every /ups.* node and T):"
    sort -u "$OUT/new/dispatch.jsonl" | tee "$OUT/dispatch_unique.jsonl"
    exit $status
    ;;
  env)
    cat <<'EOF'
# Full-request A/B: old side keeps everything unset; new side adds:
export OMNI_BIGVGAN_DECONV_TILE=0          # tile4 takes priority when =4; keep 0 for v2
export OMNI_BIGVGAN_DECONV_V2=1
export OMNI_BIGVGAN_DECONV_V2_NODES=ups    # all /ups.* ConvTranspose nodes
# Built-in default is TT=4, OCB=2, LWS=4,8,1. Paste the board-tuned lines from l1 SUMMARY.md, e.g.:
# export OMNI_DECONV_V2_K11=4:2:4,8,1
# export OMNI_DECONV_V2_K7=4:2:4,8,1
# export OMNI_DECONV_V2_K4=4:2:4,8,1
# RKNN audio prefix v2 (only if the RKNN route is under test):
export OMNI_RKNN_SELFCHECK=1               # first request: v1 vs v2 bit check, auto-fallback
export OMNI_RKNN_PARALLEL_TILES=2          # concurrent tiles on NPU core 0/1 for >256-frame audio
# export OMNI_RKNN_CORE_MASK=012           # single-context alternative; check numerics with SELFCHECK
# export OMNI_RKNN_INPUT_FP16=1            # opt-in; SELFCHECK falls back if bits differ
export OMNI_PREDIT_SYNC=1                  # both sides, as in the accepted historical E2E
EOF
    ;;
  *)
    sed -n '2,16p' "$0"; exit 1;;
esac
