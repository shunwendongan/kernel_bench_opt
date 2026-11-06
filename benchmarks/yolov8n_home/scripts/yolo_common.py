"""ONNX 与 RKNN 共用的检测前后处理。"""
import cv2
import numpy as np


def letterbox(bgr, size=640):
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError('输入必须是非空 BGR uint8 图像')
    h, w = bgr.shape[:2]
    ratio = min(size / w, size / h)
    new_w, new_h = round(w * ratio), round(h * ratio)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    left, right = round(dw - 0.1), round(dw + 0.1)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = np.ascontiguousarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
    return rgb, {'original_width': w, 'original_height': h, 'ratio': ratio,
                 'scale_x': new_w / w, 'scale_y': new_h / h,
                 'pad_left': left, 'pad_top': top, 'pad_right': right, 'pad_bottom': bottom,
                 'resized_width': new_w, 'resized_height': new_h, 'size': size, 'fill': 114}


def restore_boxes(boxes, meta):
    result = np.asarray(boxes, dtype=np.float32).copy()
    result[:, [0, 2]] = (result[:, [0, 2]] - meta['pad_left']) / meta['scale_x']
    result[:, [1, 3]] = (result[:, [1, 3]] - meta['pad_top']) / meta['scale_y']
    result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, meta['original_width'])
    result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, meta['original_height'])
    return result


def postprocess(outputs, meta, confidence=0.25, nms_iou=0.45, max_det=300):
    if len(outputs) != 9:
        raise ValueError('本模型必须有九路输出')
    candidates = []
    for branch, side in enumerate((80, 40, 20)):
        reg, probs, summed = [np.asarray(a, dtype=np.float32) for a in outputs[branch * 3:branch * 3 + 3]]
        expected = [(1, 64, side, side), (1, 80, side, side), (1, 1, side, side)]
        if [a.shape for a in (reg, probs, summed)] != expected:
            raise ValueError('输出形状不符合已冻结的 YOLOv8n 契约')
        if not all(np.isfinite(a).all() for a in (reg, probs, summed)):
            raise ValueError('模型输出包含非有限值')
        # 类别分数已做 Sigmoid。score-sum 不是 objectness。
        classes = probs[0].argmax(axis=0)
        scores = probs[0].max(axis=0)
        ys, xs = np.where(scores >= confidence)
        if not len(xs):
            continue
        dist = reg[0, :, ys, xs].reshape(-1, 4, 16)
        dist = np.exp(dist - dist.max(axis=2, keepdims=True))
        dist /= dist.sum(axis=2, keepdims=True)
        dist = (dist * np.arange(16, dtype=np.float32)).sum(axis=2)
        stride = meta['size'] / side
        center = np.stack([xs + 0.5, ys + 0.5], axis=1)
        boxes = np.concatenate([center - dist[:, :2], center + dist[:, 2:]], axis=1) * stride
        candidates.append(np.column_stack([boxes, scores[ys, xs], classes[ys, xs]]))
    if not candidates:
        return np.empty((0, 6), dtype=np.float32)
    det = np.concatenate(candidates).astype(np.float32)
    kept = []
    # 使用连续坐标面积，不加 1。不同类别分别抑制。
    for cls in np.unique(det[:, 5]):
        group = np.flatnonzero(det[:, 5] == cls)
        order = group[np.argsort(-det[group, 4], kind='stable')]
        while len(order):
            idx = order[0]
            kept.append(idx)
            rest = order[1:]
            if not len(rest):
                break
            xy1 = np.maximum(det[idx, :2], det[rest, :2])
            xy2 = np.minimum(det[idx, 2:4], det[rest, 2:4])
            inter = np.maximum(xy2 - xy1, 0).prod(axis=1)
            area0 = np.maximum(det[idx, 2:4] - det[idx, :2], 0).prod()
            areas = np.maximum(det[rest, 2:4] - det[rest, :2], 0).prod(axis=1)
            iou = inter / np.maximum(area0 + areas - inter, 1e-12)
            order = rest[iou <= nms_iou]
    kept = np.array(kept, dtype=np.int64)
    kept = kept[np.argsort(-det[kept, 4], kind='stable')][:max_det]
    result = det[kept]
    result[:, :4] = restore_boxes(result[:, :4], meta)
    return result
