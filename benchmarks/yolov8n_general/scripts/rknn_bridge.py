"""常驻 RKNN context 包装；单线程同步，厂商 ABI 由 C++ 真头文件负责。

C++ 按运行时返回的数量查询所有输出；本项目 Python 层限定官方 9 输出
YOLOv8n、batch=1、RGB UINT8 HWC640。输出保持属性给出的维度，不猜测转置。
模型内 mean=0/std=255 由转换配置负责，这里不再次归一化。
API 阶段时间由 C++ steady_clock 计毫秒；output_copy_ms/wall_ms 使用
Python perf_counter_ns。PERF_RUN 原值为微秒；perf_query_ms 含报告文件写入。
init_ms 只含 rknn_init，create_ms 到属性查询完成，不含 Python 导入/验证。
detail 必须提供 perf_path，baseline 不查询 PERF_DETAIL。
"""
from __future__ import annotations

import ctypes as C
import json
import os
import time
import numpy as np


class _Output(C.Structure):
    _fields_ = [("data", C.POINTER(C.c_float)), ("n_elems", C.c_uint64),
                ("size_bytes", C.c_uint64)]


class _Timing(C.Structure):
    _fields_ = [("inputs_set_ms", C.c_double), ("run_ms", C.c_double),
                ("outputs_get_ms", C.c_double), ("outputs_release_ms", C.c_double),
                ("perf_query_ms", C.c_double), ("perf_run_us", C.c_int64)]


class RknnSession:
    def __init__(self, model_path, library_path, detail=False, core_mask=7):
        if core_mask != 7:
            raise ValueError("core_mask must be 7")
        self._detail = bool(detail)
        self._handle = C.c_void_p()
        self._library = C.CDLL(os.fspath(library_path))
        self._library.rknn_bridge_create.argtypes = [
            C.c_char_p, C.c_uint32, C.c_uint32, C.POINTER(C.c_void_p),
            C.c_char_p, C.c_size_t,
        ]
        self._library.rknn_bridge_create.restype = C.c_int
        self._library.rknn_bridge_destroy.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t]
        self._library.rknn_bridge_destroy.restype = C.c_int
        self._library.rknn_bridge_metadata_json.argtypes = [
            C.c_void_p, C.POINTER(C.c_char_p), C.c_char_p, C.c_size_t]
        self._library.rknn_bridge_metadata_json.restype = C.c_int
        self._library.rknn_bridge_run.argtypes = [
            C.c_void_p, C.c_void_p, C.c_size_t, C.c_char_p,
            C.POINTER(C.POINTER(_Output)), C.POINTER(C.c_size_t),
            C.POINTER(_Timing), C.c_char_p, C.c_size_t]
        self._library.rknn_bridge_run.restype = C.c_int
        self._library.rknn_bridge_release.argtypes = [
            C.c_void_p, C.POINTER(_Timing), C.c_char_p, C.c_size_t]
        self._library.rknn_bridge_release.restype = C.c_int
        error = C.create_string_buffer(4096)
        status = self._library.rknn_bridge_create(
            os.fsencode(model_path), int(detail), core_mask, C.byref(self._handle),
            error, len(error))
        if status:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))

        try:
            metadata = C.c_char_p()
            status = self._library.rknn_bridge_metadata_json(
                self._handle, C.byref(metadata), error, len(error))
            if status:
                raise RuntimeError(error.value.decode("utf-8", errors="replace"))
            if not metadata.value:
                raise RuntimeError("empty metadata")
            self.metadata = json.loads(metadata.value)
            self._validate_metadata()
        except BaseException:
            self.close()
            raise

    def _validate_metadata(self):
        meta = self.metadata
        if meta["n_input"] != 1 or meta["n_output"] != 9:
            raise RuntimeError("YOLOv8n requires one input and 9 outputs")

        def nchw(attr):
            dims = tuple(attr["dims"])
            if len(dims) != 4:
                raise RuntimeError("YOLOv8n tensor dims must have rank 4")
            if attr["fmt"] == "NCHW":
                return dims
            if attr["fmt"] == "NHWC":
                return dims[0], dims[3], dims[1], dims[2]
            raise RuntimeError(f"unsupported YOLOv8n tensor format: {attr['fmt']}")

        if nchw(meta["inputs"][0]) != (1, 3, 640, 640):
            raise RuntimeError("YOLOv8n input dims must describe RGB 640x640 batch=1")
        for i, attr in enumerate(meta["outputs"]):
            size = 80 >> (i // 3)
            expected = (1, (64, 80, 1)[i % 3], size, size)
            if nchw(attr) != expected:
                raise RuntimeError(f"YOLOv8n output {i} dims {attr['dims']} != {expected}")

    def run(self, rgb_uint8_hwc, perf_path=None):
        """返回独立 FP32 数组；C++ steady_clock 计 API，Python 计复制及总墙钟。"""
        started = time.perf_counter_ns()
        if not self._handle.value:
            raise RuntimeError("RknnSession is closed")
        if not isinstance(rgb_uint8_hwc, np.ndarray) or rgb_uint8_hwc.dtype != np.uint8:
            raise TypeError("input must be a numpy uint8 array")
        if rgb_uint8_hwc.shape != (640, 640, 3) or not rgb_uint8_hwc.flags.c_contiguous:
            raise ValueError("input must be C-contiguous RGB HWC (640, 640, 3)")
        if self._detail != (perf_path is not None and bool(os.fspath(perf_path))):
            raise ValueError("detail requires perf_path; baseline forbids perf_path")
        views = C.POINTER(_Output)()
        count = C.c_size_t()
        timing = _Timing()
        error = C.create_string_buffer(4096)
        status = self._library.rknn_bridge_run(
            self._handle, C.c_void_p(rgb_uint8_hwc.ctypes.data), rgb_uint8_hwc.nbytes,
            os.fsencode(perf_path) if perf_path is not None else None,
            C.byref(views), C.byref(count), C.byref(timing), error, len(error))
        if status:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        try:
            copy_start = time.perf_counter_ns()
            if count.value != len(self.metadata["outputs"]):
                raise RuntimeError("output count mismatch")
            outputs = []
            for index, attr in enumerate(self.metadata["outputs"]):
                view = views[index]
                if (view.n_elems != attr["n_elems"] or
                        view.size_bytes != attr["n_elems"] * 4 or not view.data):
                    raise RuntimeError(f"output buffer mismatch at {index}")
                outputs.append(np.ctypeslib.as_array(view.data, shape=(view.n_elems,))
                               .reshape(attr["dims"]).copy())
            copy_ms = (time.perf_counter_ns() - copy_start) / 1e6
        finally:
            status = self._library.rknn_bridge_release(
                self._handle, C.byref(timing), error, len(error))
            if status:
                raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        metrics = {field[0]: getattr(timing, field[0]) for field in timing._fields_}
        metrics.update(output_copy_ms=copy_ms, wall_ms=(time.perf_counter_ns() - started) / 1e6)
        return outputs, metrics

    def __enter__(self):
        if not self._handle.value:
            raise RuntimeError("RknnSession is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self):
        if self._handle.value:
            error = C.create_string_buffer(4096)
            status = self._library.rknn_bridge_destroy(self._handle, error, len(error))
            self._handle = C.c_void_p()
            if status:
                raise RuntimeError(error.value.decode("utf-8", errors="replace"))
