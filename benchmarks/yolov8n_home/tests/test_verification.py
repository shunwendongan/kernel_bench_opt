"""检查数值验收不会放过 NaN、形状差异和类别错配。"""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_quality_checks_reject_nonfinite_and_class_mismatch():
    assert importlib.util.find_spec('verification') is not None, '尚未实现数值验收'
    from verification import compare_outputs, match_detections
    ref = [np.array([1., 2.], np.float32)]
    assert compare_outputs(ref, ref)['passed']
    assert not compare_outputs(ref, [np.array([np.nan, 2.])])['passed']
    assert not compare_outputs(ref, [np.array([1., 3.])])['passed']
    with pytest.raises(ValueError):
        compare_outputs(ref, [np.array([[1., 2.]])])
    boxes = np.array([[0, 0, 100, 100, .9, 0], [200, 0, 300, 100, .8, 1]], np.float32)
    changed = boxes.copy()
    changed[1, 5] = 2
    result = match_detections(boxes, changed)
    assert result['matched'] == 1 and result['reference_count'] == 2
    assert result['reference_match_fraction'] == 0.5
