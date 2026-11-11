#!/usr/bin/env bash
# Build the host-side YOLOv8n kernels (post-dfl-nms-neon, pre-letterbox-neon)
# into one shared library for the Python tests and the ctypes wrapper.
# Works on the RK3588 board (gcc, aarch64 NEON), on an Apple Silicon Mac
# (clang, NEON) and on x86 CI (portable scalar paths). No RKNN dependency.
#
#   bash scripts/build_host_kernels.sh            # -> build/libyolo_kernels.{so,dylib}
#   bash scripts/build_host_kernels.sh --scalar   # force the portable paths (self-check)
#   bash scripts/build_host_kernels.sh --bench    # also build build/bench_post and build/bench_letterbox
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${YOLO_KERNELS_BUILD_DIR:-$root/build}"
mkdir -p "$out_dir"
defines=()
suffix=""
bench=0
for arg in "$@"; do
    case "$arg" in
        --scalar) defines+=(-DYOLO_FORCE_SCALAR); suffix="_scalar" ;;
        --bench) bench=1 ;;
        *) printf 'unknown argument: %s\n' "$arg" >&2; exit 2 ;;
    esac
done
case "$(uname -s)" in
    Darwin) ext="dylib"; shared=(-dynamiclib) ;;
    *) ext="so"; shared=(-shared -Wl,-z,defs) ;;
esac
cc="${CC:-cc}"
output="$out_dir/libyolo_kernels$suffix.$ext"
"$cc" -std=c11 -O2 -fPIC -Wall -Wextra "${defines[@]+"${defines[@]}"}" "${shared[@]}" \
    "$root/kernels/post-dfl-nms-neon/yolo_post.c" \
    "$root/kernels/pre-letterbox-neon/letterbox.c" \
    -lm -o "$output"
printf 'built=%s\n' "$output"
if [[ $bench == 1 ]]; then
    "$cc" -std=c11 -O2 -Wall -Wextra "${defines[@]+"${defines[@]}"}" \
        "$root/kernels/post-dfl-nms-neon/bench_post.c" "$root/kernels/post-dfl-nms-neon/yolo_post.c" -lm \
        -o "$out_dir/bench_post$suffix"
    "$cc" -std=c11 -O2 -Wall -Wextra "${defines[@]+"${defines[@]}"}" \
        "$root/kernels/pre-letterbox-neon/bench_letterbox.c" "$root/kernels/pre-letterbox-neon/letterbox.c" -lm \
        -o "$out_dir/bench_letterbox$suffix"
    printf 'built=%s\nbuilt=%s\n' "$out_dir/bench_post$suffix" "$out_dir/bench_letterbox$suffix"
fi
