#!/usr/bin/env bash
# Build librknn_bridge_v2.so on the RK3588 board against the installed /lib/librknnrt.so.
#   bash src/apps/build_bridge_v2.sh [output.so]
#   bash src/apps/build_bridge_v2.sh --syntax-only     # hosts without librknnrt (CI / dev machines)
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$script_dir/../.." && pwd)"
vendor="$root/benchmarks/yolov8n_general/vendor"
output="$script_dir/librknn_bridge_v2.so"
syntax_only=0
for arg in "$@"; do
    case "$arg" in
        --syntax-only) syntax_only=1 ;;
        *) output="$arg" ;;
    esac
done
test -r "$vendor/rknn_api.h"
cxx="${CXX:-g++}"
cc="${CC:-gcc}"
kernels=("$root/kernels/post-dfl-nms-neon/yolo_post.c" "$root/kernels/pre-letterbox-neon/letterbox.c")
if [[ $syntax_only == 1 ]]; then
    "$cxx" -std=c++17 -fsyntax-only -Wall -Wextra -I "$vendor" "$script_dir/rknn_bridge_v2.cpp"
    for k in "${kernels[@]}"; do "$cc" -std=c11 -fsyntax-only -Wall -Wextra "$k"; done
    printf 'syntax-only check passed\n'
    exit 0
fi
test -r /lib/librknnrt.so
command -v "$cxx" >/dev/null
command -v "$cc" >/dev/null
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
objs=()
for k in "${kernels[@]}"; do
    obj="$tmp/$(basename "${k%.c}").o"
    "$cc" -std=c11 -O2 -fPIC -Wall -Wextra -Werror -c "$k" -o "$obj"
    objs+=("$obj")
done
mkdir -p -- "$(dirname -- "$output")"
"$cxx" -std=c++17 -O2 -shared -fPIC -Wall -Wextra -Werror -Wl,-z,defs -Wl,-rpath,/lib \
    -I "$vendor" "$script_dir/rknn_bridge_v2.cpp" "${objs[@]}" /lib/librknnrt.so -lm -o "$output"
printf 'built=%s\n' "$output"
sha256sum "$output"
ldd "$output"
