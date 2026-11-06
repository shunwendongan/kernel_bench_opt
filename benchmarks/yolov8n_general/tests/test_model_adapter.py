"""Adapter contract tests; synthetic arrays here are never board evidence."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class ModelAdapterTests(unittest.TestCase):
    def adapter(self):
        self.assertTrue((ROOT / 'scripts/model_adapter.py').is_file(), 'adapter is not implemented')
        import model_adapter
        return model_adapter

    def test_stable_sigmoid_extreme_float32_has_no_overflow(self):
        adapter = self.adapter()
        x = np.array([-1000, -100, -30, -8, -1, 0, 1, 8, 30, 100, 1000], np.float32)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            result = adapter.stable_sigmoid(x)
        ref = np.array([0, 3.720075976e-44, 9.357622969e-14, 0.0003353501305,
                        0.2689414214, .5, .7310585786, .9996646499, 1, 1, 1], np.float64)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(result, ref, rtol=1e-6, atol=1e-45)
        np.testing.assert_array_equal(x, np.array([-1000,-100,-30,-8,-1,0,1,8,30,100,1000], np.float32))

    def test_six_outputs_reconstruct_canonical_probabilities_and_sum(self):
        adapter = self.adapter()
        raw = []
        for size in (80, 40, 20):
            raw += [np.zeros((1, 64, size, size), np.float32),
                    np.full((1, 80, size, size), -8, np.float32)]
        # Exact sigmoid must not reproduce the old ~.001831 probability floor.
        canonical = adapter.adapt_outputs(raw)
        self.assertEqual(len(canonical), 9)
        for i, size in enumerate((80, 40, 20)):
            reg, probs, summed = canonical[3*i:3*i+3]
            self.assertIs(reg, raw[2*i])
            self.assertEqual(probs.shape, (1, 80, size, size))
            self.assertEqual(summed.shape, (1, 1, size, size))
            self.assertEqual(summed.dtype, np.float32)
            np.testing.assert_allclose(probs, 0.0003353501305, rtol=1e-6)
            np.testing.assert_allclose(summed, 80 * 0.0003353501305, rtol=2e-6)
        raw[1].fill(0)
        canonical = adapter.adapt_outputs(raw)
        np.testing.assert_array_equal(canonical[1], .5)
        np.testing.assert_array_equal(canonical[2], 1)
        self.assertIs(adapter.adapt_outputs(canonical), canonical)
    def test_rejects_wrong_layout_dtype_count_nonfinite_and_probability_range(self):
        adapter = self.adapter()
        raw = [np.zeros((1, c, s, s), np.float32) for s in (80, 40, 20) for c in (64, 80)]
        invalid = [raw[:-1], raw + [raw[0]]]
        for replacement in (np.zeros((1, 80, 80, 80), np.float32),
                            np.zeros(raw[0].shape, np.uint8),
                            np.full(raw[0].shape, np.nan, np.float32),
                            np.full(raw[0].shape, np.inf, np.float32)):
            invalid.append([replacement, *raw[1:]])
        for bad in invalid:
            with self.subTest(count=len(bad), shape=bad[0].shape):
                with self.assertRaises((ValueError, TypeError)):
                    adapter.adapt_outputs(bad)
        canonical = adapter.adapt_outputs(raw)
        for index in (1, 2, 4, 5, 7, 8):
            bad = list(canonical)
            bad[index] = np.full(bad[index].shape, 1.1, np.float32)
            with self.assertRaises(ValueError):
                adapter.adapt_outputs(bad)

    def test_conversion_targets_true_sigmoid_output_tensors(self):
        self.assertTrue((ROOT / 'scripts/prepare_model.py').is_file(), 'model preparation is not implemented')
        import prepare_model
        if not (ROOT / 'models/yolov8n.onnx').is_file():
            self.skipTest('真实 ONNX 定位测试需要另行准备固定哈希的模型文件')
        result = prepare_model.inspect_final_heads(ROOT / 'models/yolov8n.onnx')
        self.assertEqual([h['node_name'] for h in result],
                         ['/model.22/Sigmoid', '/model.22/Sigmoid_1', '/model.22/Sigmoid_2'])
        self.assertEqual([h['probability'] for h in result],
                         ['onnx::ReduceSum_326', 'onnx::ReduceSum_346', 'onnx::ReduceSum_365'])
        self.assertEqual([h['regression'] for h in result], ['318', '338', '357'])
        self.assertTrue(all(h['sum_axes'] == [1] and h['clip_bounds'] == [0.0, 1.0] for h in result))

    def test_metric_gates_keep_old_strict_failure_and_include_zero_reference(self):
        import prepare_model
        self.assertTrue(hasattr(prepare_model, 'compare_canonical'), 'numeric gates are not implemented')
        ref = [np.ones((1, c, s, s), np.float32) * (1 if c == 64 else 0)
               for s in (80, 40, 20) for c in (64, 80, 1)]
        out = [a.copy() for a in ref]
        for i in range(9):
            out[i] += (.01, .0005, .002)[i % 3]
        stats = prepare_model.compare_canonical(ref, out)
        self.assertTrue(stats['passed'])
        self.assertFalse(stats['strict_2pct_passed'])
        self.assertAlmostEqual(stats['tensors'][0]['relative_l2'], .01, places=6)
        self.assertAlmostEqual(stats['tensors'][1]['rmse'], .0005, places=7)
        self.assertAlmostEqual(stats['tensors'][2]['mae'], .002, places=7)
        for i, delta in enumerate((.021, .0011, .0051)):
            bad = [a.copy() for a in ref]
            bad[i] += delta
            self.assertFalse(prepare_model.compare_canonical(ref, bad)['passed'])
        out[0][0,0,0,0] = np.nan
        self.assertFalse(prepare_model.compare_canonical(ref, out)['passed'])
        with self.assertRaises(ValueError):
            prepare_model.compare_canonical(ref, out[:-1])


if __name__ == '__main__':
    unittest.main()
