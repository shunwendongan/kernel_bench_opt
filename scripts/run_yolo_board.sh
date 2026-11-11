#!/usr/bin/env bash
# Board-side entry points for the YOLOv8n candidates (RK3588, Linux aarch64).
#
#   scripts/run_yolo_board.sh build                    # kernels + bridge v2 against /lib/librknnrt.so
#   scripts/run_yolo_board.sh l1                       # kernel micro-benchmarks + host tests
#   scripts/run_yolo_board.sh l2 <model.rknn> <out>    # bridge v1 vs v2 on the frozen 40 inputs (+ONNX gates)
#   scripts/run_yolo_board.sh l3 <model.rknn> <out>    # 40 images x 10 repeats e2e with the fast path
#
# Environment (defaults match the first-round board layout):
#   YOLO_DEV_ROOT   first-round root with dataset/samples.json, dataset/inputs, dataset/images, models/
#                   default /home/firefly/work/benchmarks/yolov8n_home_20261106_01
#   YOLO_PY         python with numpy/cv2/onnxruntime, default /home/firefly/work/venvs/yolo-pose/bin/python
#   YOLO_LIB_V1     first-round librknn_bridge.so (optional, enables the v1 comparison)
#   YOLO_CONFIGS    extra "--config name:io,in,out,core[,mem]" arguments for l2 (space separated)
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dev_root="${YOLO_DEV_ROOT:-/home/firefly/work/benchmarks/yolov8n_home_20261106_01}"
py="${YOLO_PY:-/home/firefly/work/venvs/yolo-pose/bin/python}"
lib_v2="$root/src/apps/librknn_bridge_v2.so"
cmd="${1:-}"
case "$cmd" in
    build)
        bash "$root/scripts/build_host_kernels.sh" --bench
        bash "$root/src/apps/build_bridge_v2.sh" "$lib_v2"
        ;;
    l1)
        test -x "$root/build/bench_post" || bash "$root/scripts/build_host_kernels.sh" --bench
        "$root/build/bench_post" 200 0.02 0.25
        "$root/build/bench_post" 50 0.3 0.001 | head -4
        "$root/build/bench_letterbox" 100
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$py" -m pytest "$root/tests" -q -p no:cacheprovider
        ;;
    l2)
        model="${2:?model.rknn}"; out="${3:?output directory}"
        test -r "$lib_v2" || bash "$root/src/apps/build_bridge_v2.sh" "$lib_v2"
        extra=()
        if [[ -n "${YOLO_LIB_V1:-}" ]]; then extra+=(--library-v1 "$YOLO_LIB_V1" --reference-model "$dev_root/models/yolov8n_fp16.rknn"); fi
        if [[ -r "$dev_root/models/yolov8n.onnx" ]]; then extra+=(--onnx "$dev_root/models/yolov8n.onnx"); fi
        # default sweep: zero-copy native, zero-copy with declared uint8 input, plain fp16 NCHW outputs, legacy, single core
        configs=(--config zerocopy_native:1,2,0,7 --config zerocopy_u8in:1,1,0,7 --config zerocopy_nchw16:1,2,1,7
                 --config legacy:0,0,0,7 --config zerocopy_core0:1,2,0,1 --config zerocopy_auto:1,2,0,0)
        if [[ -n "${YOLO_CONFIGS:-}" ]]; then read -r -a user_configs <<< "$YOLO_CONFIGS"; configs=("${user_configs[@]}"); fi
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$py" -B "$root/tools/verify/l2_compare_bridges.py" \
            --samples "$dev_root/dataset/samples.json" --inputs-root "$dev_root" --model "$model" --library-v2 "$lib_v2" \
            "${configs[@]}" "${extra[@]+"${extra[@]}"}" --out "$out"
        ;;
    l3)
        model="${2:?model.rknn}"; out="${3:?output directory}"
        test -r "$lib_v2" || bash "$root/src/apps/build_bridge_v2.sh" "$lib_v2"
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$py" -B "$root/tools/verify/l3_e2e_v2.py" \
            --samples "$dev_root/dataset/samples.json" --inputs-root "$dev_root" --model "$model" --library-v2 "$lib_v2" \
            --repeats 10 --warmup 3 --out "$out"
        ;;
    *)
        sed -n '2,16p' "$0"; exit 2 ;;
esac
