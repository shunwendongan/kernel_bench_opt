"""pre-letterbox-neon vs the frozen Python/OpenCV reference (yolo_common.letterbox).

Synthetic images only; equality here is a host-side L1 check, not board evidence.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmarks/yolov8n_general/scripts'))

from tools import yolo_kernels  # noqa: E402

try:
    import cv2  # noqa: E402
    import yolo_common  # noqa: E402
except ImportError:  # pragma: no cover
    cv2 = None

TYPICAL_SIZES = [(640, 480), (480, 640), (640, 427), (427, 640), (500, 375), (375, 500), (612, 612),
                 (1024, 768), (1280, 720), (333, 500), (640, 360), (720, 1280), (300, 200), (200, 300),
                 (641, 481), (999, 1001), (64, 48), (1920, 1080)]


def random_sizes(count, seed):
    rng = np.random.default_rng(seed)
    return [(int(w), int(h)) for w, h in zip(rng.integers(50, 1400, count), rng.integers(50, 1400, count))]


@unittest.skipIf(cv2 is None, 'opencv is required for the reference')
class LetterboxKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernels = yolo_kernels.Kernels()
        cls.scalar = yolo_kernels.Kernels(scalar=True)
        cls.rng = np.random.default_rng(20261111)

    def image(self, w, h):
        return self.rng.integers(0, 256, (h, w, 3), dtype=np.uint8)

    def test_geometry_matches_python_arithmetic(self):
        for w, h in TYPICAL_SIZES + random_sizes(300, 1):
            with self.subTest(w=w, h=h):
                _, ref = yolo_common.letterbox(np.zeros((h, w, 3), np.uint8))
                self.assertEqual(self.kernels.geometry(w, h), ref)

    def _compare(self, kernels, sizes, rounding, label):
        mismatched_images = []
        for w, h in sizes:
            bgr = self.image(w, h)
            ref_rgb, ref_meta = yolo_common.letterbox(bgr)
            rgb, meta = kernels.letterbox(bgr, rounding=rounding)
            self.assertEqual(meta, ref_meta)
            if not np.array_equal(rgb, ref_rgb):
                diff = np.abs(rgb.astype(np.int16) - ref_rgb.astype(np.int16))
                mismatched_images.append((w, h, int((diff > 0).sum()), int(diff.max())))
        return mismatched_images

    def test_letterbox_pixels_match_opencv_rule(self):
        sizes = TYPICAL_SIZES + random_sizes(40, 2)
        mism = self._compare(self.kernels, sizes, yolo_kernels.ROUND_OPENCV, 'neon')
        self.assertEqual(mism, [], f'{len(mism)} images differ from cv2 INTER_LINEAR (w,h,pixels,max): {mism[:10]}')

    def test_scalar_build_matches_neon_build(self):
        for w, h in TYPICAL_SIZES[:8] + random_sizes(12, 3):
            bgr = self.image(w, h)
            a, _ = self.kernels.letterbox(bgr)
            b, _ = self.scalar.letterbox(bgr)
            self.assertTrue(np.array_equal(a, b), (w, h))

    def test_resize_matches_cv2_for_up_and_down_scaling(self):
        rng = np.random.default_rng(5)
        cases = [(640, 480, 640, 480), (640, 480, 320, 240), (300, 200, 640, 427), (37, 91, 640, 211),
                 (1280, 720, 640, 360), (123, 456, 640, 236), (640, 640, 320, 320), (99, 101, 640, 647)]
        for _ in range(30):
            cases.append(tuple(int(v) for v in rng.integers(16, 1300, 4)))
        for sw, sh, dw, dh in cases:
            bgr = rng.integers(0, 256, (sh, sw, 3), dtype=np.uint8)
            ref = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_LINEAR)
            out = self.kernels.resize_linear(bgr, dw, dh)
            with self.subTest(sw=sw, sh=sh, dw=dw, dh=dh):
                diff = np.abs(out.astype(np.int16) - ref.astype(np.int16))
                self.assertEqual(int((diff > 0).sum()), 0, f'max abs diff {int(diff.max())}')

    def test_input_pack_formats(self):
        rgb = self.rng.integers(0, 256, (640, 640, 3), dtype=np.uint8)
        nhwc16 = self.kernels.pack_input(rgb, yolo_kernels.INPUT_NHWC_F16)
        self.assertTrue(np.array_equal(nhwc16[0], rgb.astype(np.float16)))
        nhwc16s = self.kernels.pack_input(rgb, yolo_kernels.INPUT_NHWC_F16, w_stride=656)
        self.assertEqual(nhwc16s.shape, (1, 640, 656, 3))
        self.assertTrue(np.array_equal(nhwc16s[0, :, :640], rgb.astype(np.float16)))
        nchw32 = self.kernels.pack_input(rgb, yolo_kernels.INPUT_NCHW_F32, scale=255.0, divide=True)
        ref = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255.)
        self.assertTrue(np.array_equal(nchw32, ref))
        nchw16 = self.kernels.pack_input(rgb, yolo_kernels.INPUT_NCHW_F16)
        self.assertTrue(np.array_equal(nchw16[0], rgb.transpose(2, 0, 1).astype(np.float16)))
        blk = self.kernels.pack_input(rgb, yolo_kernels.INPUT_NC1HWC2_F16, c2=8)
        self.assertEqual(blk.shape, (1, 1, 640, 640, 8))
        self.assertTrue(np.array_equal(blk[0, 0, :, :, :3], rgb.astype(np.float16)))
        self.assertEqual(float(np.abs(blk[0, 0, :, :, 3:].astype(np.float32)).max()), 0.0)
        u8 = self.kernels.pack_input(rgb, yolo_kernels.INPUT_NHWC_U8)
        self.assertTrue(np.array_equal(u8[0], rgb))
        scalar = self.scalar.pack_input(rgb, yolo_kernels.INPUT_NHWC_F16)
        self.assertTrue(np.array_equal(scalar, nhwc16))


if __name__ == '__main__':
    unittest.main()
