"""ctypes wrapper for the host-side YOLOv8n kernels (post-dfl-nms-neon, pre-letterbox-neon).

Used by the tests and by the v2 bridge wrapper. The shared library is built by
scripts/build_host_kernels.sh; set YOLO_KERNELS_LIB to point at a specific build.
Arrays are numpy; views describe RKNN buffers (NCHW fp16/fp32 or NC1HWC2 fp16).
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAYOUT_NCHW, LAYOUT_NC1HWC2 = 0, 1
DTYPE_F16, DTYPE_F32 = 0, 1
INPUT_NHWC_U8, INPUT_NHWC_F16, INPUT_NCHW_F16, INPUT_NCHW_F32, INPUT_NC1HWC2_F16 = 0, 1, 2, 3, 4
ROUND_OPENCV, ROUND_SCALAR22, ROUND_OPENCV_TAIL22 = 0, 1, 2


class TensorView(C.Structure):
    _fields_ = [("data", C.c_void_p), ("layout", C.c_int32), ("dtype", C.c_int32),
                ("channels", C.c_int32), ("height", C.c_int32), ("width", C.c_int32),
                ("w_stride", C.c_int32), ("c2", C.c_int32)]


class PostScale(C.Structure):
    _fields_ = [("reg", TensorView), ("cls", TensorView), ("sum", TensorView), ("side", C.c_int32)]


class PostParams(C.Structure):
    _fields_ = [("cls_is_logits", C.c_int32), ("num_classes", C.c_int32), ("reg_max", C.c_int32),
                ("input_size", C.c_int32), ("confidence", C.c_float), ("nms_iou", C.c_float),
                ("max_det", C.c_int32), ("restore", C.c_int32), ("pad_left", C.c_float),
                ("pad_top", C.c_float), ("scale_x", C.c_float), ("scale_y", C.c_float),
                ("original_width", C.c_float), ("original_height", C.c_float)]


class PostStats(C.Structure):
    _fields_ = [("candidates", C.c_int32), ("kept", C.c_int32), ("nonfinite", C.c_int32),
                ("prefiltered", C.c_int32), ("scan_ms", C.c_double), ("decode_ms", C.c_double),
                ("nms_ms", C.c_double), ("total_ms", C.c_double)]


class Transform(C.Structure):
    _fields_ = [("original_width", C.c_int32), ("original_height", C.c_int32), ("ratio", C.c_double),
                ("scale_x", C.c_double), ("scale_y", C.c_double), ("pad_left", C.c_int32),
                ("pad_top", C.c_int32), ("pad_right", C.c_int32), ("pad_bottom", C.c_int32),
                ("resized_width", C.c_int32), ("resized_height", C.c_int32), ("size", C.c_int32),
                ("fill", C.c_int32)]

    def as_dict(self):
        return {'original_width': self.original_width, 'original_height': self.original_height,
                'ratio': self.ratio, 'scale_x': self.scale_x, 'scale_y': self.scale_y,
                'pad_left': self.pad_left, 'pad_top': self.pad_top, 'pad_right': self.pad_right,
                'pad_bottom': self.pad_bottom, 'resized_width': self.resized_width,
                'resized_height': self.resized_height, 'size': self.size, 'fill': self.fill}


def default_library_path(scalar=False):
    override = os.environ.get("YOLO_KERNELS_LIB")
    if override and not scalar:
        return Path(override)
    ext = "dylib" if sys.platform == "darwin" else "so"
    name = "libyolo_kernels_scalar" if scalar else "libyolo_kernels"
    return Path(os.environ.get("YOLO_KERNELS_BUILD_DIR", ROOT / "build")) / f"{name}.{ext}"


def ensure_library(scalar=False):
    path = default_library_path(scalar)
    if not path.is_file():
        args = ["bash", str(ROOT / "scripts/build_host_kernels.sh")] + (["--scalar"] if scalar else [])
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
    return path


class Kernels:
    def __init__(self, library_path=None, scalar=False):
        path = Path(library_path) if library_path else ensure_library(scalar)
        self.path = path
        lib = self._lib = C.CDLL(os.fspath(path))
        lib.yolo_post_workspace_bytes.argtypes = [C.c_int32]
        lib.yolo_post_workspace_bytes.restype = C.c_size_t
        for name in ("yolo_post_run", "yolo_post_run_reference"):
            fn = getattr(lib, name)
            fn.argtypes = [C.POINTER(PostScale), C.c_int32, C.POINTER(PostParams), C.c_void_p, C.c_int32,
                           C.c_void_p, C.c_size_t, C.POINTER(PostStats)]
            fn.restype = C.c_int32
        lib.yolo_view_to_nchw_f32.argtypes = [C.POINTER(TensorView), C.c_void_p]
        lib.yolo_view_to_nchw_f32.restype = C.c_int64
        lib.yolo_sigmoid_sum.argtypes = [C.POINTER(TensorView), C.c_void_p, C.c_void_p]
        lib.yolo_sigmoid_sum.restype = C.c_int64
        lib.yolo_post_has_neon.restype = C.c_int32
        lib.yolo_letterbox_has_neon.restype = C.c_int32
        lib.yolo_letterbox_geometry.argtypes = [C.c_int32, C.c_int32, C.c_int32, C.POINTER(Transform)]
        lib.yolo_letterbox_workspace_bytes.argtypes = [C.c_int32, C.c_int32]
        lib.yolo_letterbox_workspace_bytes.restype = C.c_size_t
        lib.yolo_letterbox_rgb_u8.argtypes = [C.c_void_p, C.c_int32, C.c_int32, C.c_int64, C.c_void_p, C.c_int32,
                                              C.c_int32, C.POINTER(Transform), C.c_void_p, C.c_size_t]
        lib.yolo_letterbox_rgb_u8.restype = C.c_int32
        lib.yolo_resize_linear_u8c3.argtypes = [C.c_void_p, C.c_int32, C.c_int32, C.c_int64, C.c_void_p, C.c_int32,
                                                C.c_int32, C.c_int64, C.c_int32, C.c_void_p, C.c_size_t]
        lib.yolo_resize_linear_u8c3.restype = C.c_int32
        lib.yolo_input_pack.argtypes = [C.c_void_p, C.c_int32, C.c_int32, C.c_int32, C.c_int32, C.c_float, C.c_int32,
                                        C.c_void_p, C.c_size_t]
        lib.yolo_input_pack.restype = C.c_int64
        self.has_neon = bool(lib.yolo_post_has_neon()) and bool(lib.yolo_letterbox_has_neon())
        self._workspace = None

    # ---- views -------------------------------------------------------------
    @staticmethod
    def make_view(array, channels=None, width=None):
        """NCHW [1,C,H,W] (fp16/fp32, W may be padded) or NC1HWC2 [1,C1,H,Wp,C2] fp16."""
        array = np.ascontiguousarray(array)
        view = TensorView()
        view.data = array.ctypes.data
        if array.ndim == 4:
            _, c, h, wp = array.shape
            view.layout = LAYOUT_NCHW
            view.channels = int(channels or c)
            view.height = h
            view.width = int(width or wp)
            view.w_stride = wp
            view.c2 = 0
            view.dtype = DTYPE_F16 if array.dtype == np.float16 else DTYPE_F32
            if array.dtype not in (np.float16, np.float32):
                raise TypeError("NCHW view needs float16/float32")
        elif array.ndim == 5:
            _, c1, h, wp, c2 = array.shape
            if array.dtype != np.float16:
                raise TypeError("NC1HWC2 view needs float16")
            if channels is None:
                raise ValueError("NC1HWC2 view needs the logical channel count")
            view.layout = LAYOUT_NC1HWC2
            view.dtype = DTYPE_F16
            view.channels = int(channels)
            view.height = h
            view.width = int(width or wp)
            view.w_stride = wp
            view.c2 = c2
        else:
            raise ValueError("unsupported array rank")
        return view, array

    @staticmethod
    def to_nc1hwc2(nchw_fp16, c2=8, w_align=8):
        """Pack an NCHW fp16 array into NC1HWC2 with channel and width padding (test helper)."""
        _, c, h, w = nchw_fp16.shape
        c1 = -(-c // c2)
        wp = -(-w // w_align) * w_align
        out = np.zeros((1, c1, h, wp, c2), np.float16)
        padded = np.zeros((1, c1 * c2, h, wp), np.float16)
        padded[:, :c, :, :w] = nchw_fp16
        out[0] = padded[0].reshape(c1, c2, h, wp).transpose(0, 2, 3, 1)
        return out

    def to_nchw_f32(self, array, channels=None, width=None):
        view, keep = self.make_view(array, channels, width)
        out = np.empty((1, view.channels, view.height, view.width), np.float32)
        n = self._lib.yolo_view_to_nchw_f32(C.byref(view), out.ctypes.data)
        assert n == out.size, (n, out.size)
        return out

    def sigmoid_sum(self, logits, channels=None, width=None):
        view, keep = self.make_view(logits, channels, width)
        probs = np.empty((1, view.channels, view.height, view.width), np.float32)
        total = np.empty((1, 1, view.height, view.width), np.float32)
        self._lib.yolo_sigmoid_sum(C.byref(view), probs.ctypes.data, total.ctypes.data)
        return probs, total

    # ---- post-processing ---------------------------------------------------
    def postprocess(self, outputs, meta, confidence=0.25, nms_iou=0.45, max_det=300, logits=False,
                    use_sum=True, reference=False, restore=True, channels=None, widths=None, return_stats=False):
        """outputs: 9 arrays (reg, probs, sum) x 3 scales or 6 arrays (reg, logits) x 3 scales."""
        per_scale = 2 if logits else 3
        if len(outputs) != 3 * per_scale:
            raise ValueError("expected 6 (logits) or 9 (canonical) outputs")
        scales = (PostScale * 3)()
        keep = []
        for s in range(3):
            group = outputs[s * per_scale:(s + 1) * per_scale]
            width = widths[s] if widths else None
            reg_view, a = self.make_view(group[0], 64, width); keep.append(a)
            cls_view, b = self.make_view(group[1], 80, width); keep.append(b)
            scales[s].reg = reg_view
            scales[s].cls = cls_view
            if per_scale == 3 and use_sum:
                sum_view, c = self.make_view(group[2], 1, width); keep.append(c)
                scales[s].sum = sum_view
            else:
                scales[s].sum = TensorView()
            scales[s].side = reg_view.height
        params = PostParams(cls_is_logits=int(logits), num_classes=80, reg_max=16, input_size=int(meta['size']),
                            confidence=confidence, nms_iou=nms_iou, max_det=max_det, restore=int(restore),
                            pad_left=float(meta['pad_left']), pad_top=float(meta['pad_top']),
                            scale_x=float(meta['scale_x']), scale_y=float(meta['scale_y']),
                            original_width=float(meta['original_width']), original_height=float(meta['original_height']))
        total = sum(int(scales[s].side) ** 2 for s in range(3))
        need = self._lib.yolo_post_workspace_bytes(total)
        if self._workspace is None or len(self._workspace) < need:
            self._workspace = C.create_string_buffer(need)
        dets = np.empty((total, 6), np.float32)
        stats = PostStats()
        fn = self._lib.yolo_post_run_reference if reference else self._lib.yolo_post_run
        n = fn(scales, 3, C.byref(params), dets.ctypes.data, total, self._workspace, need, C.byref(stats))
        if n < 0:
            raise RuntimeError(f"yolo_post_run failed with {n}")
        result = dets[:n].copy()
        if return_stats:
            return result, {f[0]: getattr(stats, f[0]) for f in stats._fields_}
        return result

    # ---- pre-processing ----------------------------------------------------
    def letterbox(self, bgr, size=640, rounding=ROUND_OPENCV):
        bgr = np.ascontiguousarray(bgr)
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError("letterbox needs BGR uint8 HWC")
        h, w = bgr.shape[:2]
        rgb = np.empty((size, size, 3), np.uint8)
        t = Transform()
        need = self._lib.yolo_letterbox_workspace_bytes(size, w)
        ws = C.create_string_buffer(need)
        rc = self._lib.yolo_letterbox_rgb_u8(bgr.ctypes.data, w, h, bgr.strides[0], rgb.ctypes.data, size,
                                             rounding, C.byref(t), ws, need)
        if rc:
            raise RuntimeError(f"yolo_letterbox_rgb_u8 failed with {rc}")
        return rgb, t.as_dict()

    def geometry(self, w, h, size=640):
        t = Transform()
        self._lib.yolo_letterbox_geometry(w, h, size, C.byref(t))
        return t.as_dict()

    def resize_linear(self, bgr, dw, dh, rounding=ROUND_OPENCV):
        bgr = np.ascontiguousarray(bgr)
        h, w = bgr.shape[:2]
        out = np.empty((dh, dw, 3), np.uint8)
        need = self._lib.yolo_letterbox_workspace_bytes(max(dw, dh), w)
        ws = C.create_string_buffer(need)
        rc = self._lib.yolo_resize_linear_u8c3(bgr.ctypes.data, w, h, bgr.strides[0], out.ctypes.data, dw, dh,
                                               out.strides[0], rounding, ws, need)
        if rc:
            raise RuntimeError(f"yolo_resize_linear_u8c3 failed with {rc}")
        return out

    def pack_input(self, rgb, fmt=INPUT_NHWC_F16, w_stride=0, c2=8, scale=1.0, divide=False):
        rgb = np.ascontiguousarray(rgb)
        size = rgb.shape[0]
        ws = w_stride or size
        if fmt == INPUT_NHWC_U8:
            out = np.empty((1, size, ws, 3), np.uint8)
        elif fmt == INPUT_NHWC_F16:
            out = np.empty((1, size, ws, 3), np.float16)
        elif fmt == INPUT_NCHW_F16:
            out = np.empty((1, 3, size, ws), np.float16)
        elif fmt == INPUT_NCHW_F32:
            out = np.empty((1, 3, size, ws), np.float32)
        elif fmt == INPUT_NC1HWC2_F16:
            out = np.empty((1, 1, size, ws, c2), np.float16)
        else:
            raise ValueError("unknown input format")
        n = self._lib.yolo_input_pack(rgb.ctypes.data, size, fmt, w_stride, c2, float(scale), int(divide),
                                      out.ctypes.data, out.nbytes)
        if n != out.nbytes:
            raise RuntimeError(f"yolo_input_pack returned {n}, expected {out.nbytes}")
        return out
