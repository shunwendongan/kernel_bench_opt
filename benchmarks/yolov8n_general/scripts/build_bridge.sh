#!/usr/bin/env bash
# 在板端用已有编译器构建，显式链接现有 /lib/librknnrt.so，不安装软件。
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
output="${1:-$script_dir/librknn_bridge.so}"
if [[ $# -gt 1 ]]; then
    printf 'usage: bash build_bridge.sh [output.so]\n' >&2
    exit 2
fi
test -r "$project_dir/vendor/rknn_api.h"
test -r /lib/librknnrt.so
command -v "${CXX:-g++}" >/dev/null
mkdir -p -- "$(dirname -- "$output")"
"${CXX:-g++}" -std=c++17 -O2 -shared -fPIC -Wall -Wextra -Werror \
    -Wl,-z,defs -Wl,-rpath,/lib \
    -I "$project_dir/vendor" "$script_dir/rknn_bridge.cpp" \
    /lib/librknnrt.so -o "$output"
printf 'built=%s\n' "$output"
sha256sum "$output"
ldd "$output"
