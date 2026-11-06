"""抽样与预处理的行为测试；合成图像只用于单元测试。"""
import importlib.util
from pathlib import Path
import sys
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))


def test_letterbox_preserves_rgb_and_inverse_geometry():
    spec = importlib.util.find_spec('yolo_common')
    assert spec is not None, '尚未实现公共预处理模块'
    from yolo_common import letterbox, restore_boxes
    bgr = np.zeros((240, 320, 3), dtype=np.uint8)
    bgr[:] = (10, 20, 30)
    rgb, meta = letterbox(bgr)
    assert rgb.shape == (640, 640, 3)
    assert rgb.dtype == np.uint8 and rgb.flags.c_contiguous
    assert list(rgb[0, 0]) == [114, 114, 114]
    assert list(rgb[80, 0]) == [30, 20, 10]
    assert meta['pad_left'] == 0 and meta['pad_top'] == 80
    actual = restore_boxes(np.array([[0, 80, 640, 560]], dtype=np.float32), meta)
    np.testing.assert_allclose(actual, [[0, 0, 320, 240]], atol=1e-5)


def test_nine_head_decoder_ignores_score_sum_and_has_no_extra_sigmoid():
    import yolo_common
    assert hasattr(yolo_common, 'postprocess'), '尚未实现九路输出解码'
    outputs = []
    for size in (80, 40, 20):
        outputs.extend([np.zeros((1, 64, size, size), np.float32),
                        np.zeros((1, 80, size, size), np.float32),
                        np.zeros((1, 1, size, size), np.float32)])
    meta = {'pad_left': 0, 'pad_top': 0, 'scale_x': 1, 'scale_y': 1,
            'original_width': 640, 'original_height': 640, 'size': 640}
    assert yolo_common.postprocess(outputs, meta).shape == (0, 6)
    outputs[1][0, 0, 40, 40] = 0.9
    outputs[1][0, 0, 40, 41] = 0.8
    result = yolo_common.postprocess(outputs, meta)
    assert result.shape == (1, 6)
    np.testing.assert_allclose(result[0], [264, 264, 384, 384, 0.9, 0], atol=1e-5)
    outputs[1][0, 1, 40, 41] = 0.95
    assert yolo_common.postprocess(outputs, meta).shape == (2, 6)
