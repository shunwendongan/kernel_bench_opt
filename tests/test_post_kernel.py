"""post-dfl-nms-neon vs the frozen Python reference (yolo_common.postprocess, model_adapter).

Synthetic tensors and ONNX Runtime outputs on synthetic images; host-side L1
checks only. Board evidence comes from tools/verify/l2_compare_bridges.py.
"""
import hashlib
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmarks/yolov8n_general/scripts'))

from tools import yolo_kernels  # noqa: E402
import model_adapter  # noqa: E402
import yolo_common  # noqa: E402

ONNX_SHA256 = '0c8716701f471067932b797eeb67c8e5db47c693c2557c881d7679ec12e21bc5'
META = {'original_width': 500, 'original_height': 375, 'ratio': 1.28, 'scale_x': 1.28, 'scale_y': 1.28,
        'pad_left': 0, 'pad_top': 80, 'pad_right': 0, 'pad_bottom': 80, 'resized_width': 640,
        'resized_height': 480, 'size': 640, 'fill': 114}


def onnx_path():
    candidates = [os.environ.get('YOLO_ONNX_PATH'), ROOT / 'models/yolov8n.onnx',
                  ROOT / 'benchmarks/yolov8n_general/models/yolov8n.onnx']
    for c in candidates:
        if c and Path(c).is_file():
            if hashlib.sha256(Path(c).read_bytes()).hexdigest() == ONNX_SHA256:
                return Path(c)
    return None


def synthetic_outputs(rng, logits=False, fp16=True, hot_fraction=0.02, tie_anchors=4):
    """9 canonical (or 6 logits) outputs with ties and duplicated anchors to exercise ordering."""
    outputs = []
    for side in (80, 40, 20):
        reg = rng.normal(0.0, 2.0, (1, 64, side, side)).astype(np.float32)
        raw = rng.normal(-6.0, 1.5, (1, 80, side, side)).astype(np.float32)
        hot = rng.random((1, 1, side, side)) < hot_fraction
        raw = np.where(hot, raw + rng.uniform(4.0, 10.0, raw.shape).astype(np.float32), raw)
        # exact duplicates: same class vector at two anchors, two classes sharing the max
        for _ in range(tie_anchors):
            y0, x0, y1, x1 = rng.integers(0, side, 4)
            raw[0, :, y1, x1] = raw[0, :, y0, x0]
            reg[0, :, y1, x1] = reg[0, :, y0, x0]
            c = int(raw[0, :, y0, x0].argmax())
            raw[0, (c + 7) % 80, y0, x0] = raw[0, c, y0, x0]
        if fp16:
            reg = reg.astype(np.float16).astype(np.float32)
            raw = raw.astype(np.float16).astype(np.float32)
        if logits:
            outputs += [reg, raw]
        else:
            probs = model_adapter.stable_sigmoid(raw)
            if fp16:
                probs = probs.astype(np.float16).astype(np.float32)
            summed = np.clip(probs.sum(axis=1, keepdims=True, dtype=np.float32), 0, 1)
            if fp16:
                summed = summed.astype(np.float16).astype(np.float32)
            outputs += [reg, probs, summed]
    return outputs


def assert_same_detections(test, got, ref, box_atol=2e-3, score_atol=0.0, label=''):
    test.assertEqual(got.shape, ref.shape, f'{label}: shape {got.shape} vs {ref.shape}')
    if len(ref) == 0:
        return
    np.testing.assert_array_equal(got[:, 5], ref[:, 5], err_msg=f'{label}: classes')
    if score_atol == 0.0:
        np.testing.assert_array_equal(got[:, 4], ref[:, 4], err_msg=f'{label}: scores')
    else:
        np.testing.assert_allclose(got[:, 4], ref[:, 4], atol=score_atol, rtol=0, err_msg=f'{label}: scores')
    np.testing.assert_allclose(got[:, :4], ref[:, :4], atol=box_atol, rtol=0, err_msg=f'{label}: boxes')


class PostKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.k = yolo_kernels.Kernels()
        cls.scalar = yolo_kernels.Kernels(scalar=True)

    def run_all(self, outputs, meta, logits=False, **kw):
        ref_inputs = model_adapter.adapt_outputs(list(outputs)) if logits else outputs
        ref = yolo_common.postprocess(ref_inputs, meta, kw.get('confidence', 0.25), kw.get('nms_iou', 0.45),
                                      kw.get('max_det', 300))
        got = self.k.postprocess(outputs, meta, logits=logits, **kw)
        got_ref = self.k.postprocess(outputs, meta, logits=logits, reference=True, **kw)
        got_scalar = self.scalar.postprocess(outputs, meta, logits=logits, **kw)
        return ref, got, got_ref, got_scalar

    def test_canonical9_fp32_and_fp16_values(self):
        rng = np.random.default_rng(1)
        for fp16 in (False, True):
            for conf in (0.25, 0.05, 0.001):
                outputs = synthetic_outputs(rng, fp16=fp16)
                arrays = [a.astype(np.float16) if fp16 else a for a in outputs]
                ref, got, got_ref, got_scalar = self.run_all(arrays, META, confidence=conf)
                label = f'fp16={fp16} conf={conf} n={len(ref)}'
                self.assertGreater(len(ref), 0, label)
                assert_same_detections(self, got, ref, label=label)
                np.testing.assert_array_equal(got, got_ref, err_msg=label)
                np.testing.assert_array_equal(got, got_scalar, err_msg=label)

    def test_sum_prefilter_is_exact(self):
        rng = np.random.default_rng(2)
        outputs = [a.astype(np.float16) for a in synthetic_outputs(rng, fp16=True, hot_fraction=0.05)]
        with_sum, stats = self.k.postprocess(outputs, META, use_sum=True, return_stats=True)
        without, _ = self.k.postprocess(outputs, META, use_sum=False, return_stats=True)
        np.testing.assert_array_equal(with_sum, without)
        self.assertGreater(stats['prefiltered'], 0)
        self.assertEqual(stats['nonfinite'], 0)

    def test_logits6_matches_adapter_plus_reference(self):
        rng = np.random.default_rng(3)
        for conf in (0.25, 0.001):
            outputs = [a.astype(np.float16) for a in synthetic_outputs(rng, logits=True, fp16=True)]
            ref, got, got_ref, got_scalar = self.run_all(outputs, META, logits=True, confidence=conf)
            label = f'logits conf={conf} n={len(ref)}'
            self.assertGreater(len(ref), 0, label)
            # sigmoid via libm expf vs numpy exp: allow 1 ulp-ish on the probability
            assert_same_detections(self, got, ref, score_atol=2e-7, label=label)
            np.testing.assert_array_equal(got, got_ref, err_msg=label)
            np.testing.assert_array_equal(got, got_scalar, err_msg=label)

    def test_nc1hwc2_layout_equals_nchw(self):
        rng = np.random.default_rng(4)
        for logits in (False, True):
            outputs = [a.astype(np.float16) for a in synthetic_outputs(rng, logits=logits, fp16=True)]
            per = 2 if logits else 3
            packed = []
            for i, arr in enumerate(outputs):
                packed.append(self.k.to_nc1hwc2(arr, c2=8, w_align=8))
            widths = [outputs[s * per].shape[-1] for s in range(3)]
            a = self.k.postprocess(outputs, META, logits=logits, confidence=0.01)
            b = self.k.postprocess(packed, META, logits=logits, confidence=0.01, widths=widths)
            c = self.scalar.postprocess(packed, META, logits=logits, confidence=0.01, widths=widths)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(a, c)
            # de-layout helper reproduces the dense tensors
            for arr, blk in zip(outputs, packed):
                dense = self.k.to_nchw_f32(blk, channels=arr.shape[1], width=arr.shape[-1])
                np.testing.assert_array_equal(dense, arr.astype(np.float32))

    def test_sigmoid_sum_adapter(self):
        rng = np.random.default_rng(5)
        logits = rng.normal(-3, 4, (1, 80, 40, 40)).astype(np.float16)
        probs, total = self.k.sigmoid_sum(logits)
        ref = model_adapter.adapt_outputs([np.zeros((1, 64, 80, 80), np.float32), np.zeros((1, 80, 80, 80), np.float32),
                                           np.zeros((1, 64, 40, 40), np.float32), logits.astype(np.float32),
                                           np.zeros((1, 64, 20, 20), np.float32), np.zeros((1, 80, 20, 20), np.float32)])
        np.testing.assert_allclose(probs, ref[4], atol=3e-7, rtol=0)
        np.testing.assert_allclose(total, ref[5], atol=2e-5, rtol=0)
        self.assertTrue(np.all(probs >= 0) and np.all(probs <= 1))
        sp, st = self.scalar.sigmoid_sum(logits)
        np.testing.assert_allclose(sp, probs, atol=3e-7, rtol=0)

    def test_restore_disabled_keeps_letterbox_coordinates(self):
        rng = np.random.default_rng(6)
        outputs = [a.astype(np.float16) for a in synthetic_outputs(rng, fp16=True)]
        raw = self.k.postprocess(outputs, META, restore=False)
        restored = self.k.postprocess(outputs, META, restore=True)
        self.assertEqual(len(raw), len(restored))
        if len(raw):
            manual = yolo_common.restore_boxes(raw[:, :4], META)
            np.testing.assert_allclose(manual, restored[:, :4], atol=1e-4, rtol=0)


@unittest.skipIf(onnx_path() is None, 'frozen YOLOv8n ONNX not available (set YOLO_ONNX_PATH)')
class PostKernelOnnxRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import onnx
        import onnxruntime as ort
        cls.k = yolo_kernels.Kernels()
        path = onnx_path()
        model = onnx.load(str(path))
        heads = ['/model.22/cv3.0/cv3.0.2/Conv_output_0', '/model.22/cv3.1/cv3.1.2/Conv_output_0',
                 '/model.22/cv3.2/cv3.2.2/Conv_output_0']
        for name in heads:
            model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        cls.session = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
        cls.input_name = cls.session.get_inputs()[0].name

    def images(self):
        rng = np.random.default_rng(7)
        yield 'noise', rng.integers(0, 256, (375, 500, 3), dtype=np.uint8)
        yy, xx = np.mgrid[0:480, 0:640]
        gradient = np.stack([(xx * 255 // 639), (yy * 255 // 479), ((xx + yy) * 255 // 1118)], axis=2).astype(np.uint8)
        yield 'gradient', gradient
        blobs = np.full((427, 640, 3), 90, np.uint8)
        for _ in range(12):
            x0, y0 = rng.integers(0, 500), rng.integers(0, 300)
            w, h = rng.integers(40, 140, 2)
            blobs[y0:y0 + h, x0:x0 + w] = rng.integers(0, 256, 3)
        yield 'blobs', blobs

    def test_kernel_matches_reference_on_ort_outputs(self):
        total_detections = 0
        for name, bgr in self.images():
            rgb, meta = yolo_common.letterbox(bgr)
            tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255.)
            outs = self.session.run(None, {self.input_name: tensor})
            canonical = outs[:9]
            logits6 = [canonical[0], outs[9], canonical[3], outs[10], canonical[6], outs[11]]
            for conf, iou in ((0.25, 0.45), (0.001, 0.65)):
                ref = yolo_common.postprocess(canonical, meta, conf, iou, 300)
                # float32 NCHW as produced by the legacy want_float path
                got = self.k.postprocess(canonical, meta, confidence=conf, nms_iou=iou)
                assert_same_detections(self, got, ref, label=f'{name} f32 conf={conf} n={len(ref)}')
                # fp16 NCHW as produced by zero-copy outputs (reference sees the same rounded values)
                half = [a.astype(np.float16) for a in canonical]
                ref16 = yolo_common.postprocess([a.astype(np.float32) for a in half], meta, conf, iou, 300)
                got16 = self.k.postprocess(half, meta, confidence=conf, nms_iou=iou)
                assert_same_detections(self, got16, ref16, label=f'{name} f16 conf={conf} n={len(ref16)}')
                # logits6 in fp16 vs adapter(float32 sigmoid) + reference
                half6 = [a.astype(np.float16) for a in logits6]
                ref6 = yolo_common.postprocess(model_adapter.adapt_outputs([a.astype(np.float32) for a in half6]),
                                               meta, conf, iou, 300)
                got6 = self.k.postprocess(half6, meta, logits=True, confidence=conf, nms_iou=iou)
                assert_same_detections(self, got6, ref6, score_atol=2e-7, label=f'{name} logits6 conf={conf} n={len(ref6)}')
                total_detections += len(ref)
        self.assertGreater(total_detections, 0, 'synthetic images produced no candidates at all')


if __name__ == '__main__':
    unittest.main()
