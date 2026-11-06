"""部署数值一致性、检测匹配与 COCO 子集精度。"""
import numpy as np


def compare_outputs(reference, actual, relative_l2_max=0.02):
    if len(reference) != len(actual):
        raise ValueError('输出数量不一致')
    rows = []
    for idx, (ref, out) in enumerate(zip(reference, actual)):
        if ref.shape != out.shape:
            raise ValueError(f'输出 {idx} 形状不一致')
        finite = bool(np.isfinite(ref).all() and np.isfinite(out).all())
        if finite:
            # float64 只用于误差统计，避免归约误差掩盖差异。
            r, a = ref.astype(np.float64), out.astype(np.float64)
            delta = a - r
            relative = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(r.ravel()), 1e-12))
            maximum = float(np.max(np.abs(delta)))
            mean = float(np.mean(np.abs(delta)))
        else:
            relative = maximum = mean = None
        rows.append({'index': idx, 'shape': list(ref.shape), 'finite': finite,
                     'relative_l2': relative, 'max_abs': maximum, 'mean_abs': mean,
                     'passed': finite and relative is not None and relative <= relative_l2_max})
    return {'passed': all(row['passed'] for row in rows), 'tensors': rows}


def match_detections(reference, actual, iou_min=0.9):
    used = set()
    matches = []
    for ri in np.argsort(-reference[:, 4], kind='stable') if len(reference) else []:
        ref = reference[ri]
        candidate = [i for i, out in enumerate(actual) if i not in used and out[5] == ref[5]]
        if not candidate:
            continue
        other = actual[candidate]
        overlap = np.maximum(0, np.minimum(ref[2:4], other[:, 2:4]) - np.maximum(ref[:2], other[:, :2])).prod(axis=1)
        area = np.maximum(0, ref[2:4] - ref[:2]).prod()
        areas = np.maximum(0, other[:, 2:4] - other[:, :2]).prod(axis=1)
        ious = overlap / np.maximum(area + areas - overlap, 1e-12)
        best = int(np.argmax(ious))
        if ious[best] >= iou_min:
            ai = candidate[best]
            used.add(ai)
            matches.append({'reference_index': int(ri), 'actual_index': ai,
                            'iou': float(ious[best]),
                            'coordinate_max_abs_px': float(np.max(np.abs(actual[ai, :4] - ref[:4]))),
                            'score_abs': float(abs(actual[ai, 4] - ref[4]))})
    return {'reference_count': len(reference), 'actual_count': len(actual), 'matched': len(matches),
            'reference_match_fraction': len(matches) / len(reference) if len(reference) else 1.0,
            'actual_match_fraction': len(matches) / len(actual) if len(actual) else (1.0 if not len(reference) else 0.0),
            'matches': matches}


def coco_predictions(image_id, detections, category_ids):
    return [{'image_id': image_id, 'category_id': category_ids[int(row[5])],
             'bbox': [float(row[0]), float(row[1]), float(row[2] - row[0]), float(row[3] - row[1])],
             'score': float(row[4])} for row in detections]


def evaluate_coco(annotation_path, predictions, image_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    truth = COCO(str(annotation_path))
    if predictions:
        detected = truth.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = {'images': truth.dataset['images'], 'categories': truth.dataset['categories'], 'annotations': []}
        detected.createIndex()
    evaluator = COCOeval(truth, detected, 'bbox')
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {'AP': float(evaluator.stats[0]), 'AP50': float(evaluator.stats[1]),
            'AP75': float(evaluator.stats[2]), 'AR100': float(evaluator.stats[8]),
            'image_count': len(image_ids), 'scope': '所选40张、全部80类，不代表完整COCO或真实家庭分布',
            'all_stats': evaluator.stats.tolist()}
