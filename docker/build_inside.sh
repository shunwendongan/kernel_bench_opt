#!/usr/bin/env bash
set -euo pipefail

variant="${1:-sync}"
jobs="${2:-4}"
if [[ "$variant" != sync && "$variant" != async ]]; then
  echo 'Usage: bash build_inside.sh [sync|async] [jobs]' >&2
  exit 2
fi
case "$(uname -s):$(uname -m)" in
  Linux:aarch64|Linux:arm64) ;;
  *) echo 'Build in a Linux/arm64 environment.' >&2; exit 2 ;;
esac
for file in /opt/board-gpu/libOpenCL.so /opt/board-rknn/include/rknn_api.h /opt/board-rknn/lib/librknnrt.so; do
  test -f "$file" || { echo "Missing mounted dependency: $file" >&2; exit 2; }
done
# A work directory belongs to one source image/variant. Reuse it for benchmark
# outputs, but never overwrite a prior source tree or build under a new image.
for path in /work/source /work/build; do
  if [[ -e "$path" ]]; then
    echo "Existing $path; select a new --work directory for a new build." >&2
    exit 2
  fi
done
cp -a /opt/kernel-bench /work/source
if [[ "$variant" == sync ]]; then
  (cd /work/source && git apply --check experiments/predit-sync.patch && git apply experiments/predit-sync.patch)
fi
mnn_commit=7e3c6fb5225d3ccf22807ee641332669df34650d
if [[ ! -d /work/MNN ]]; then
  git init /work/MNN
  git -C /work/MNN remote add origin https://github.com/alibaba/MNN.git
  git -C /work/MNN fetch --depth 1 origin "$mnn_commit"
  git -C /work/MNN checkout --detach FETCH_HEAD
fi
test "$(git -C /work/MNN rev-parse HEAD)" = "$mnn_commit"
python3 /work/source/scripts/build_rk3588.py \
  --mnn-source /work/MNN --build-dir /work/build \
  --rknn-include /opt/board-rknn/include \
  --rknn-lib /opt/board-rknn/lib/librknnrt.so \
  --opencl-lib /opt/board-gpu/libOpenCL.so --jobs "$jobs"
printf '%s\n' "$variant" > /work/build/omni-variant.txt
echo "Build complete ($variant). Inference has not been run."
