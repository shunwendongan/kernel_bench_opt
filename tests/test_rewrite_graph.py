"""Graph rewrite tool: structural checks on the frozen ONNX (skipped when the model is absent)
and ONNX Runtime equivalence of every variant on synthetic images."""
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/npu'))
sys.path.insert(0, str(ROOT / 'tools/verify'))

import rewrite_yolov8n_graph as rewrite  # noqa: E402
import compare_onnx_variants as compare  # noqa: E402

ONNX_SHA256 = rewrite.SOURCE_SHA256


def onnx_path():
    for c in (os.environ.get('YOLO_ONNX_PATH'), ROOT / 'models/yolov8n.onnx',
              ROOT / 'benchmarks/yolov8n_general/models/yolov8n.onnx'):
        if c and Path(c).is_file() and hashlib.sha256(Path(c).read_bytes()).hexdigest() == ONNX_SHA256:
            return Path(c)
    return None


def test_space_to_depth_channel_order():
    """Our numpy SpaceToDepth must match the ONNX definition used by the rewrite."""
    import onnx
    import onnxruntime as ort
    x = np.arange(1 * 3 * 4 * 6, dtype=np.float32).reshape(1, 3, 4, 6)
    node = onnx.helper.make_node('SpaceToDepth', ['x'], ['y'], blocksize=2)
    graph = onnx.helper.make_graph([node], 's2d', [onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 3, 4, 6])],
                                   [onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1, 12, 2, 3])])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 13)])
    model.ir_version = 8
    sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
    y = sess.run(None, {'x': x})[0]
    np.testing.assert_array_equal(y, compare.space_to_depth(x))


@unittest.skipIf(onnx_path() is None, 'frozen YOLOv8n ONNX not available (set YOLO_ONNX_PATH)')
class RewriteGraphTests(unittest.TestCase):
    def variant(self, tmp, tag, **kw):
        out = Path(tmp) / f'{tag}.onnx'
        record = rewrite.rewrite(onnx_path(), out, **kw)
        return out, record

    def test_logits6_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, record = self.variant(tmp, 'logits6', logits6=True)
            import onnx
            g = onnx.load(str(out)).graph
            self.assertEqual([o.name for o in g.output], ['318', 'cls_logits_80x80', '338', 'cls_logits_40x40', '357', 'cls_logits_20x20'])
            ops = [n.op_type for n in g.node]
            self.assertNotIn('ReduceSum', ops)
            self.assertNotIn('Clip', ops)
            self.assertEqual(ops.count('Sigmoid'), 57)   # 60 SiLU/Sigmoid minus the 3 head Sigmoids
            self.assertEqual(record['node_count_after'], 228 - 15)

    def test_split_free_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, record = self.variant(tmp, 'splitfree', split_free=True)
            import onnx
            g = onnx.load(str(out)).graph
            ops = [n.op_type for n in g.node]
            self.assertEqual(ops.count('Split'), 0)
            self.assertEqual(ops.count('Conv'), 63 + 8)
            self.assertEqual(len(record['splits']), 8)
            names = {n.name for n in g.node}
            for module in ('/model.2', '/model.4', '/model.6', '/model.8', '/model.12', '/model.15', '/model.18', '/model.21'):
                self.assertIn(f'{module}/cv1a/conv/Conv', names)
                self.assertIn(f'{module}/cv1b/conv/Conv', names)

    def test_s2d_structures(self):
        import onnx
        with tempfile.TemporaryDirectory() as tmp:
            out, record = self.variant(tmp, 's2d_graph', s2d_stem='graph')
            g = onnx.load(str(out)).graph
            self.assertIn('SpaceToDepth', [n.op_type for n in g.node])
            self.assertEqual([d.dim_value for d in g.input[0].type.tensor_type.shape.dim], [1, 3, 640, 640])
            out, record = self.variant(tmp, 's2d_host', s2d_stem='host')
            g = onnx.load(str(out)).graph
            self.assertNotIn('SpaceToDepth', [n.op_type for n in g.node])
            self.assertEqual(g.input[0].name, 'images_s2d')
            self.assertEqual([d.dim_value for d in g.input[0].type.tensor_type.shape.dim], [1, 12, 320, 320])

    def test_variants_are_numerically_equivalent(self):
        cases = {'logits6': dict(logits6=True), 'splitfree': dict(split_free=True),
                 's2d_host': dict(s2d_stem='host'), 'combined': dict(logits6=True, split_free=True, s2d_stem='host')}
        with tempfile.TemporaryDirectory() as tmp:
            for tag, kw in cases.items():
                out, _ = self.variant(tmp, tag, **kw)
                report = compare.compare(onnx_path(), out, compare.synthetic_images(3, 3))
                with self.subTest(tag=tag):
                    self.assertLessEqual(report['worst']['max_abs'], 1e-4, report['worst'])
                    self.assertTrue(all(r['same_count'] and r['same_classes'] for r in report['detections']), report['detections'])
                    if tag == 'splitfree':
                        self.assertEqual(report['worst']['max_abs'], 0.0)

    def test_rejects_unverified_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            bogus = Path(tmp) / 'bogus.onnx'
            bogus.write_bytes(b'not a model')
            with self.assertRaises(ValueError):
                rewrite.rewrite(bogus, Path(tmp) / 'out.onnx', logits6=True)


if __name__ == '__main__':
    unittest.main()
