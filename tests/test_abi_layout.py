"""The ctypes structures used by tools/yolo_kernels.py and src/apps/rknn_bridge_v2.py must
match the C structs byte for byte. This compiles a tiny probe with the real headers and
compares sizeof/offsetof; it needs a C compiler (skipped otherwise)."""
import ctypes as C
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import yolo_kernels  # noqa: E402

PROBE = r'''
#include <stdio.h>
#include <stddef.h>
#include "kernels/post-dfl-nms-neon/yolo_post.h"
#include "kernels/pre-letterbox-neon/letterbox.h"
struct V2Timing { double letterbox_ms, pack_ms, inputs_set_ms, run_ms, outputs_get_ms, outputs_release_ms,
                  perf_query_ms, post_ms, export_ms; long long perf_run_us; };
struct V2OutputView { const void* data; unsigned long long bytes; int layout, dtype, channels, height, width, w_stride, c2; };
#define P(T, F) printf("\"%s.%s\": %zu, ", #T, #F, offsetof(T, F))
#define S(T) printf("\"sizeof %s\": %zu, ", #T, sizeof(T))
int main(void) {
  printf("{");
  S(yolo_tensor_view); P(yolo_tensor_view, data); P(yolo_tensor_view, layout); P(yolo_tensor_view, c2);
  S(yolo_post_scale); P(yolo_post_scale, reg); P(yolo_post_scale, cls); P(yolo_post_scale, sum); P(yolo_post_scale, side);
  S(yolo_post_params); P(yolo_post_params, confidence); P(yolo_post_params, max_det); P(yolo_post_params, pad_left); P(yolo_post_params, original_height);
  S(yolo_post_stats); P(yolo_post_stats, scan_ms); P(yolo_post_stats, total_ms);
  S(yolo_detection); S(yolo_letterbox_transform); P(yolo_letterbox_transform, ratio); P(yolo_letterbox_transform, pad_left); P(yolo_letterbox_transform, fill);
  S(struct V2Timing); P(struct V2Timing, perf_run_us);
  S(struct V2OutputView); P(struct V2OutputView, bytes); P(struct V2OutputView, layout); P(struct V2OutputView, c2);
  printf("\"end\": 0}\n");
  return 0;
}
'''


def load_wrapper():
    spec = importlib.util.spec_from_file_location('rknn_bridge_v2', ROOT / 'src/apps/rknn_bridge_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(shutil.which('cc') is None, 'needs a C compiler')
class AbiLayoutTests(unittest.TestCase):
    def test_c_and_ctypes_layouts_agree(self):
        wrapper = load_wrapper()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'probe.c'
            src.write_text(PROBE)
            exe = Path(tmp) / 'probe'
            subprocess.run(['cc', '-std=c11', '-I', str(ROOT), str(src), '-o', str(exe)], check=True)
            layout = json.loads(subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout)
        expected = {
            'sizeof yolo_tensor_view': C.sizeof(yolo_kernels.TensorView),
            'yolo_tensor_view.data': yolo_kernels.TensorView.data.offset,
            'yolo_tensor_view.layout': yolo_kernels.TensorView.layout.offset,
            'yolo_tensor_view.c2': yolo_kernels.TensorView.c2.offset,
            'sizeof yolo_post_scale': C.sizeof(yolo_kernels.PostScale),
            'yolo_post_scale.reg': yolo_kernels.PostScale.reg.offset,
            'yolo_post_scale.cls': yolo_kernels.PostScale.cls.offset,
            'yolo_post_scale.sum': yolo_kernels.PostScale.sum.offset,
            'yolo_post_scale.side': yolo_kernels.PostScale.side.offset,
            'sizeof yolo_post_params': C.sizeof(yolo_kernels.PostParams),
            'yolo_post_params.confidence': yolo_kernels.PostParams.confidence.offset,
            'yolo_post_params.max_det': yolo_kernels.PostParams.max_det.offset,
            'yolo_post_params.pad_left': yolo_kernels.PostParams.pad_left.offset,
            'yolo_post_params.original_height': yolo_kernels.PostParams.original_height.offset,
            'sizeof yolo_post_stats': C.sizeof(yolo_kernels.PostStats),
            'yolo_post_stats.scan_ms': yolo_kernels.PostStats.scan_ms.offset,
            'yolo_post_stats.total_ms': yolo_kernels.PostStats.total_ms.offset,
            'sizeof yolo_detection': 24,
            'sizeof yolo_letterbox_transform': C.sizeof(yolo_kernels.Transform),
            'yolo_letterbox_transform.ratio': yolo_kernels.Transform.ratio.offset,
            'yolo_letterbox_transform.pad_left': yolo_kernels.Transform.pad_left.offset,
            'yolo_letterbox_transform.fill': yolo_kernels.Transform.fill.offset,
            'sizeof struct V2Timing': C.sizeof(wrapper._Timing),
            'struct V2Timing.perf_run_us': wrapper._Timing.perf_run_us.offset,
            'sizeof struct V2OutputView': C.sizeof(wrapper._OutputView),
            'struct V2OutputView.bytes': wrapper._OutputView.bytes.offset,
            'struct V2OutputView.layout': wrapper._OutputView.layout.offset,
            'struct V2OutputView.c2': wrapper._OutputView.c2.offset,
        }
        for key, value in expected.items():
            self.assertEqual(layout[key], value, key)


if __name__ == '__main__':
    unittest.main()
