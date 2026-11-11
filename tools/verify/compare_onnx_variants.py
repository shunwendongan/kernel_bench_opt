#!/usr/bin/env python3
"""ONNX Runtime equivalence check between the frozen YOLOv8n ONNX and a rewritten variant.

For a logits6 variant the six outputs are adapted to canonical9 with
model_adapter.adapt_outputs (float32 sigmoid + clipped sum) before comparison.
For a host-s2d variant the 12x320x320 input is produced from the same image with
the ONNX SpaceToDepth channel order. Synthetic images are used unless --images
points at a directory of real pictures. This is a float32 CPU check: it proves
the rewrite preserves the mathematics, it says nothing about RK3588 timing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks/yolov8n_general/scripts'))
import model_adapter  # noqa: E402
import yolo_common  # noqa: E402


def session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def space_to_depth(tensor, bs=2):
    n, c, h, w = tensor.shape
    t = tensor.reshape(n, c, h // bs, bs, w // bs, bs).transpose(0, 3, 5, 1, 2, 4)
    return np.ascontiguousarray(t.reshape(n, c * bs * bs, h // bs, w // bs))


def synthetic_images(count, seed):
    rng = np.random.default_rng(seed)
    sizes = [(500, 375), (640, 480), (427, 640), (1024, 768), (333, 500), (640, 640)]
    for i in range(count):
        w, h = sizes[i % len(sizes)]
        kind = i % 3
        if kind == 0:
            img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        elif kind == 1:
            img = np.full((h, w, 3), 90, np.uint8)
            for _ in range(10):
                x0, y0 = rng.integers(0, max(1, w - 60)), rng.integers(0, max(1, h - 60))
                bw, bh = rng.integers(30, 150, 2)
                img[y0:y0 + bh, x0:x0 + bw] = rng.integers(0, 256, 3)
        else:
            yy, xx = np.mgrid[0:h, 0:w]
            img = np.stack([(xx * 255 // max(1, w - 1)), (yy * 255 // max(1, h - 1)), ((xx + yy) % 256)], axis=2).astype(np.uint8)
        yield f'synthetic_{i}_{w}x{h}', img


def real_images(folder):
    import cv2
    for path in sorted(Path(folder).glob('*')):
        if path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield path.name, img


def compare(reference_path, variant_path, images, confidence=0.25, nms_iou=0.45):
    ref = session(reference_path)
    var = session(variant_path)
    ref_in = ref.get_inputs()[0].name
    var_in = var.get_inputs()[0]
    host_s2d = list(var_in.shape) == [1, 12, 320, 320]
    per_output = {}
    detection_rows = []
    worst = {'max_abs': 0.0, 'relative_l2': 0.0}
    timings = {'reference_ms': [], 'variant_ms': []}
    for name, bgr in images:
        rgb, meta = yolo_common.letterbox(bgr)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255.)
        t0 = time.perf_counter()
        expected = ref.run(None, {ref_in: tensor})
        t1 = time.perf_counter()
        feed = space_to_depth(tensor) if host_s2d else tensor
        actual = var.run(None, {var_in.name: feed})
        t2 = time.perf_counter()
        timings['reference_ms'].append((t1 - t0) * 1e3)
        timings['variant_ms'].append((t2 - t1) * 1e3)
        if len(actual) == 6:
            actual = model_adapter.adapt_outputs(list(actual))
        if len(actual) != 9:
            raise ValueError(f'variant produced {len(actual)} outputs')
        for index, (e, a) in enumerate(zip(expected, actual)):
            if e.shape != a.shape:
                raise ValueError(f'output {index} shape {a.shape} != {e.shape}')
            delta = a.astype(np.float64) - e.astype(np.float64)
            max_abs = float(np.max(np.abs(delta)))
            rel = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(e.astype(np.float64).ravel()), 1e-12))
            row = per_output.setdefault(index, {'role': ('regression', 'probability', 'clip_sum')[index % 3],
                                                'shape': list(e.shape), 'max_abs': 0.0, 'relative_l2': 0.0,
                                                'bit_identical_images': 0})
            row['max_abs'] = max(row['max_abs'], max_abs)
            row['relative_l2'] = max(row['relative_l2'], rel)
            row['bit_identical_images'] += int(np.array_equal(e, a))
            worst['max_abs'] = max(worst['max_abs'], max_abs)
            worst['relative_l2'] = max(worst['relative_l2'], rel)
        ref_det = yolo_common.postprocess(expected, meta, confidence, nms_iou)
        var_det = yolo_common.postprocess(actual, meta, confidence, nms_iou)
        same_count = len(ref_det) == len(var_det)
        box_diff = float(np.max(np.abs(ref_det[:, :4] - var_det[:, :4]))) if same_count and len(ref_det) else 0.0
        score_diff = float(np.max(np.abs(ref_det[:, 4] - var_det[:, 4]))) if same_count and len(ref_det) else 0.0
        class_same = bool(np.array_equal(ref_det[:, 5], var_det[:, 5])) if same_count else False
        detection_rows.append({'image': name, 'reference_detections': int(len(ref_det)), 'variant_detections': int(len(var_det)),
                               'same_count': same_count, 'same_classes': class_same,
                               'max_box_abs_px': box_diff, 'max_score_abs': score_diff})
    return {'per_output': per_output, 'worst': worst, 'detections': detection_rows,
            'images': len(detection_rows), 'host_space_to_depth_input': host_s2d,
            'timing_ms_mean': {k: float(np.mean(v)) for k, v in timings.items()},
            'variant_output_count': len(var.get_outputs())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--images', help='directory with real images (otherwise synthetic)')
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--max-abs', type=float, default=1e-4, help='gate on the worst max-abs difference')
    parser.add_argument('--out', help='write the JSON report here')
    args = parser.parse_args()
    images = real_images(args.images) if args.images else synthetic_images(args.count, 20261111)
    report = compare(args.reference, args.variant, images)
    report.update({'schema_version': 1, 'record_type': 'onnx_variant_comparison',
                   'reference_sha256': hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
                   'variant_sha256': hashlib.sha256(Path(args.variant).read_bytes()).hexdigest(),
                   'gate_max_abs': args.max_abs, 'passed': report['worst']['max_abs'] <= args.max_abs
                   and all(r['same_count'] and r['same_classes'] for r in report['detections']),
                   'scope': 'float32 ONNX Runtime CPU equivalence only; not an RK3588 measurement'})
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + '\n', encoding='utf-8')
    print(text)
    sys.exit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
