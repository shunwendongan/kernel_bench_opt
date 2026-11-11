"""Python wrapper for librknn_bridge_v2.so (zero-copy RKNN session + host NEON kernels).

Drop-in replacement for benchmarks/yolov8n_general/scripts/rknn_bridge.RknnSession:
``run(rgb_uint8_hwc)`` still returns float32 NCHW arrays plus a metrics dict, so
legacy_runtime.py / quality_gate.py can consume either bridge. The fast path is
``infer_bgr`` (letterbox + run + post-processing inside the shared library).

Modes (all recorded in ``metadata``):
  io_mode     0 legacy (rknn_inputs_set / rknn_outputs_get want_float, like v1)
              1 zero-copy (rknn_create_mem + rknn_set_io_mem)
  input_mode  0 auto, 1 declare uint8 NHWC (runtime converts), 2 native pass-through
  output_mode 0 native layout (NC1HWC2 when the runtime reports it), 1 plain fp16 NCHW
  core_mask   0 auto, 1/2/4 single core, 3 two cores, 7 three cores
"""
from __future__ import annotations

import ctypes as C
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.yolo_kernels import PostParams, PostStats, Transform  # noqa: E402

LAYOUT_NAMES = {0: 'NCHW', 1: 'NC1HWC2'}
DTYPES = {0: np.float16, 1: np.float32}


class _Timing(C.Structure):
    _fields_ = [("letterbox_ms", C.c_double), ("pack_ms", C.c_double), ("inputs_set_ms", C.c_double),
                ("run_ms", C.c_double), ("outputs_get_ms", C.c_double), ("outputs_release_ms", C.c_double),
                ("perf_query_ms", C.c_double), ("post_ms", C.c_double), ("export_ms", C.c_double),
                ("perf_run_us", C.c_int64)]

    def as_dict(self):
        return {f[0]: getattr(self, f[0]) for f in self._fields_}


class _OutputView(C.Structure):
    _fields_ = [("data", C.c_void_p), ("bytes", C.c_uint64), ("layout", C.c_int32), ("dtype", C.c_int32),
                ("channels", C.c_int32), ("height", C.c_int32), ("width", C.c_int32), ("w_stride", C.c_int32),
                ("c2", C.c_int32)]


class RknnSessionV2:
    def __init__(self, model_path, library_path, detail=False, core_mask=7, io_mode=1, input_mode=0,
                 output_mode=0, mem_flags=0, rounding=0):
        self._detail = bool(detail)
        self._handle = C.c_void_p()
        self._rounding = int(rounding)
        lib = self._library = C.CDLL(os.fspath(library_path))
        lib.rknn_bridge_v2_create.argtypes = [C.c_char_p, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32,
                                              C.c_uint32, C.POINTER(C.c_void_p), C.c_char_p, C.c_size_t]
        lib.rknn_bridge_v2_create.restype = C.c_int
        lib.rknn_bridge_v2_metadata_json.argtypes = [C.c_void_p, C.POINTER(C.c_char_p), C.c_char_p, C.c_size_t]
        lib.rknn_bridge_v2_metadata_json.restype = C.c_int
        lib.rknn_bridge_v2_destroy.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t]
        lib.rknn_bridge_v2_destroy.restype = C.c_int
        lib.rknn_bridge_v2_prepare_bgr.argtypes = [C.c_void_p, C.c_void_p, C.c_int32, C.c_int32, C.c_int64, C.c_int32,
                                                   C.POINTER(Transform), C.POINTER(_Timing), C.c_char_p, C.c_size_t]
        lib.rknn_bridge_v2_prepare_bgr.restype = C.c_int
        lib.rknn_bridge_v2_prepare_rgb.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(_Timing), C.c_char_p, C.c_size_t]
        lib.rknn_bridge_v2_prepare_rgb.restype = C.c_int
        lib.rknn_bridge_v2_letterboxed_rgb.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t), C.c_char_p, C.c_size_t]
        lib.rknn_bridge_v2_letterboxed_rgb.restype = C.c_int
        lib.rknn_bridge_v2_run.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(_Timing), C.c_char_p, C.c_size_t]
        lib.rknn_bridge_v2_run.restype = C.c_int
        lib.rknn_bridge_v2_output_count.argtypes = [C.c_void_p]
        lib.rknn_bridge_v2_output_count.restype = C.c_int
        lib.rknn_bridge_v2_output_view.argtypes = [C.c_void_p, C.c_uint32, C.POINTER(_OutputView), C.c_char_p, C.c_size_t]
        lib.rknn_bridge_v2_output_view.restype = C.c_int
        lib.rknn_bridge_v2_output_f32.argtypes = [C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint64, C.POINTER(_Timing), C.c_char_p, C.c_size_t]
        lib.rknn_bridge_v2_output_f32.restype = C.c_int
        lib.rknn_bridge_v2_postprocess.argtypes = [C.c_void_p, C.POINTER(PostParams), C.c_void_p, C.c_int32, C.POINTER(C.c_int32),
                                                   C.POINTER(PostStats), C.POINTER(_Timing), C.c_char_p, C.c_size_t]
        lib.rknn_bridge_v2_postprocess.restype = C.c_int
        lib.rknn_bridge_v2_release.argtypes = [C.c_void_p, C.POINTER(_Timing), C.c_char_p, C.c_size_t]
        lib.rknn_bridge_v2_release.restype = C.c_int
        error = C.create_string_buffer(4096)
        status = lib.rknn_bridge_v2_create(os.fsencode(model_path), int(detail), int(core_mask), int(io_mode),
                                           int(input_mode), int(output_mode), int(mem_flags), C.byref(self._handle),
                                           error, len(error))
        if status:
            raise RuntimeError(error.value.decode('utf-8', errors='replace'))
        try:
            text = C.c_char_p()
            status = lib.rknn_bridge_v2_metadata_json(self._handle, C.byref(text), error, len(error))
            if status or not text.value:
                raise RuntimeError(error.value.decode('utf-8', errors='replace') or 'empty metadata')
            self.metadata = json.loads(text.value)
            self.n_output = int(self.metadata['n_output'])
            self.sides = (80, 40, 20)
            self._validate_metadata()
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------ contract
    def _validate_metadata(self):
        meta = self.metadata
        if meta['n_input'] != 1 or meta['n_output'] not in (6, 9):
            raise RuntimeError('YOLOv8n requires one input and 6 or 9 outputs')
        per = 3 if meta['n_output'] == 9 else 2
        channels = (64, 80, 1) if per == 3 else (64, 80)
        for i, attr in enumerate(meta['outputs']):
            side = 80 >> (i // per)
            dims = tuple(attr['dims'])
            if attr['fmt'] == 'NHWC':
                dims = (dims[0], dims[3], dims[1], dims[2])
            if dims != (1, channels[i % per], side, side):
                raise RuntimeError(f'output {i} dims {attr["dims"]} do not match the YOLOv8n head contract')

    @property
    def logits6(self):
        return self.n_output == 6

    def _check(self, status, error):
        if status:
            raise RuntimeError(error.value.decode('utf-8', errors='replace'))

    # ------------------------------------------------------------------ inputs
    def prepare_bgr(self, bgr):
        if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise TypeError('input must be a BGR uint8 HWC numpy array')
        bgr = np.ascontiguousarray(bgr)
        t, timing, error = Transform(), _Timing(), C.create_string_buffer(4096)
        self._check(self._library.rknn_bridge_v2_prepare_bgr(self._handle, bgr.ctypes.data, bgr.shape[1], bgr.shape[0],
                                                             bgr.strides[0], self._rounding, C.byref(t), C.byref(timing),
                                                             error, len(error)), error)
        return t.as_dict(), timing.as_dict()

    def prepare_rgb(self, rgb):
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (640, 640, 3) or not rgb.flags.c_contiguous:
            raise ValueError('input must be C-contiguous RGB HWC (640, 640, 3) uint8')
        timing, error = _Timing(), C.create_string_buffer(4096)
        self._check(self._library.rknn_bridge_v2_prepare_rgb(self._handle, rgb.ctypes.data, rgb.nbytes, C.byref(timing),
                                                             error, len(error)), error)
        return timing.as_dict()

    def letterboxed_rgb(self):
        ptr, n, error = C.c_void_p(), C.c_size_t(), C.create_string_buffer(4096)
        self._check(self._library.rknn_bridge_v2_letterboxed_rgb(self._handle, C.byref(ptr), C.byref(n), error, len(error)), error)
        return np.ctypeslib.as_array(C.cast(ptr, C.POINTER(C.c_uint8)), shape=(n.value,)).reshape(640, 640, 3).copy()

    # ------------------------------------------------------------------ execution
    def run_prepared(self, perf_path=None):
        if self._detail != (perf_path is not None and bool(os.fspath(perf_path))):
            raise ValueError('detail requires perf_path; baseline forbids perf_path')
        timing, error = _Timing(), C.create_string_buffer(4096)
        self._check(self._library.rknn_bridge_v2_run(self._handle, os.fsencode(perf_path) if perf_path else None,
                                                     C.byref(timing), error, len(error)), error)
        return timing.as_dict()

    def postprocess(self, transform, confidence=0.25, nms_iou=0.45, max_det=300, restore=True):
        params = PostParams(cls_is_logits=int(self.logits6), num_classes=80, reg_max=16, input_size=640,
                            confidence=confidence, nms_iou=nms_iou, max_det=max_det, restore=int(restore),
                            pad_left=float(transform['pad_left']), pad_top=float(transform['pad_top']),
                            scale_x=float(transform['scale_x']), scale_y=float(transform['scale_y']),
                            original_width=float(transform['original_width']),
                            original_height=float(transform['original_height']))
        dets = np.empty((8400, 6), np.float32)
        count, stats, timing, error = C.c_int32(), PostStats(), _Timing(), C.create_string_buffer(4096)
        self._check(self._library.rknn_bridge_v2_postprocess(self._handle, C.byref(params), dets.ctypes.data, 8400,
                                                             C.byref(count), C.byref(stats), C.byref(timing),
                                                             error, len(error)), error)
        return dets[:count.value].copy(), {f[0]: getattr(stats, f[0]) for f in stats._fields_}, timing.as_dict()

    def output_views(self):
        views = []
        for i in range(self.n_output):
            v, error = _OutputView(), C.create_string_buffer(4096)
            self._check(self._library.rknn_bridge_v2_output_view(self._handle, i, C.byref(v), error, len(error)), error)
            views.append({'layout': LAYOUT_NAMES.get(v.layout, v.layout), 'dtype': DTYPES[v.dtype].__name__,
                          'channels': v.channels, 'height': v.height, 'width': v.width, 'w_stride': v.w_stride,
                          'c2': v.c2, 'bytes': v.bytes})
        return views

    def raw_output(self, index):
        """Copy of the bound output buffer as stored (fp16 native layout or fp32 NCHW)."""
        v, error = _OutputView(), C.create_string_buffer(4096)
        self._check(self._library.rknn_bridge_v2_output_view(self._handle, index, C.byref(v), error, len(error)), error)
        dtype = DTYPES[v.dtype]
        count = v.bytes // np.dtype(dtype).itemsize
        flat = np.ctypeslib.as_array(C.cast(v.data, C.POINTER(C.c_uint8)), shape=(v.bytes,)).view(dtype)[:count].copy()
        if v.layout == 1:
            c1 = -(-v.channels // v.c2)
            return flat[:c1 * v.height * v.w_stride * v.c2].reshape(1, c1, v.height, v.w_stride, v.c2)
        return flat[:v.channels * v.height * v.w_stride].reshape(1, v.channels, v.height, v.w_stride)

    def outputs_f32(self, timing=None):
        """Dense float32 NCHW copies of all outputs (6 or 9), outside the timed inference path."""
        result = []
        t = _Timing()
        per = 3 if self.n_output == 9 else 2
        channels = (64, 80, 1) if per == 3 else (64, 80)
        for i in range(self.n_output):
            side = 80 >> (i // per)
            arr = np.empty((1, channels[i % per], side, side), np.float32)
            error = C.create_string_buffer(4096)
            self._check(self._library.rknn_bridge_v2_output_f32(self._handle, i, arr.ctypes.data, arr.size, C.byref(t),
                                                                error, len(error)), error)
            result.append(arr)
        if timing is not None:
            timing['export_ms'] = t.export_ms
        return result

    def release(self):
        timing, error = _Timing(), C.create_string_buffer(4096)
        self._check(self._library.rknn_bridge_v2_release(self._handle, C.byref(timing), error, len(error)), error)
        return timing.as_dict()

    # ------------------------------------------------------------------ convenience
    def run(self, rgb_uint8_hwc, perf_path=None):
        """v1-compatible: letterboxed RGB in, float32 NCHW arrays (6 or 9) and metrics out."""
        started = time.perf_counter_ns()
        metrics = self.prepare_rgb(rgb_uint8_hwc)
        metrics.update(self.run_prepared(perf_path))
        try:
            copy_start = time.perf_counter_ns()
            outputs = self.outputs_f32(metrics)
            metrics['output_copy_ms'] = (time.perf_counter_ns() - copy_start) / 1e6
        finally:
            metrics.update({k: v for k, v in self.release().items() if k == 'outputs_release_ms'})
        metrics['wall_ms'] = (time.perf_counter_ns() - started) / 1e6
        return outputs, metrics

    def infer_bgr(self, bgr, confidence=0.25, nms_iou=0.45, max_det=300, perf_path=None):
        """Fast path: letterbox + pack + run + NEON post-processing, detections in image coordinates."""
        started = time.perf_counter_ns()
        transform, metrics = self.prepare_bgr(bgr)
        metrics.update(self.run_prepared(perf_path))
        try:
            dets, stats, post = self.postprocess(transform, confidence, nms_iou, max_det)
        finally:
            metrics.update({k: v for k, v in self.release().items() if k == 'outputs_release_ms'})
        metrics['post_ms'] = post['post_ms']
        metrics['post_stats'] = stats
        metrics['wall_ms'] = (time.perf_counter_ns() - started) / 1e6
        return dets, transform, metrics

    def __enter__(self):
        if not self._handle.value:
            raise RuntimeError('RknnSessionV2 is closed')
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        if self._handle.value:
            error = C.create_string_buffer(4096)
            status = self._library.rknn_bridge_v2_destroy(self._handle, error, len(error))
            self._handle = C.c_void_p()
            if status:
                raise RuntimeError(error.value.decode('utf-8', errors='replace'))
