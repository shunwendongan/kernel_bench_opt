"""在完整采集记录上执行事先固定的部署质量门槛。"""
import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
from verification import evaluate_coco


# Same fixed batch=1, NCHW export contract as RknnSession._validate_metadata.
OUTPUT_SHAPES = [[1, channels, size, size]
                 for size in (80, 40, 20) for channels in (64, 80, 1)]


def _nonnegative_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _tensor_rows(rows):
    tensors = []
    for row in rows:
        report = row.get('tensors')
        items = report.get('tensors') if isinstance(report, dict) else None
        if not isinstance(items, list) or len(items) != len(OUTPUT_SHAPES):
            raise ValueError(f'image {row["image_id"]}: exactly nine tensor records required')
        for index, (tensor, expected_shape) in enumerate(zip(items, OUTPUT_SHAPES)):
            label = f'image {row["image_id"]}, tensor {index}'
            if (not isinstance(tensor, dict) or type(tensor.get('index')) is not int
                    or tensor['index'] != index):
                raise ValueError(f'{label}: invalid tensor index')
            shape = tensor.get('shape')
            if (not isinstance(shape, list) or any(type(dim) is not int for dim in shape)
                    or shape != expected_shape):
                raise ValueError(f'{label}: shape must be {expected_shape}')
            for metric in ('relative_l2', 'mean_abs', 'max_abs'):
                if not _nonnegative_number(tensor.get(metric)):
                    raise ValueError(f'{label}: {metric} must be finite and nonnegative')
        tensors.extend(items)
    return tensors


def _sample_coverage(config, manifest, rows):
    samples = manifest['samples']
    count = config['sampling']['count']
    if (type(count) is not int or count <= 0 or not isinstance(samples, list)
            or len(samples) != count or any(not isinstance(s, dict)
            or type(s.get('id')) is not int or not isinstance(s.get('sample_id'), str)
            or not s['sample_id'] for s in samples)):
        raise ValueError('Manifest must contain the declared nonempty sample population')
    expected = {s['id']: s['sample_id'] for s in samples}
    if len(expected) != count or len(set(expected.values())) != count:
        raise ValueError('Manifest sample IDs must be unique')
    if (not isinstance(rows, list) or len(rows) != count
            or any(not isinstance(r, dict) or type(r.get('image_id')) is not int for r in rows)
            or {r['image_id'] for r in rows} != set(expected)
            or any(r.get('sample_id') != expected[r['image_id']] for r in rows)):
        raise ValueError('数值校验没有逐一覆盖全部选中样本')
    return set(expected)


def _evaluate(root, run_path):
    config_bytes = (root / 'config.json').read_bytes()
    manifest_bytes = (root / 'dataset/samples.json').read_bytes()
    start_bytes = (run_path / 'start.json').read_bytes()
    start = json.loads(start_bytes)
    identity = {'config.json': hashlib.sha256(config_bytes).hexdigest(),
                'samples.json': hashlib.sha256(manifest_bytes).hexdigest(),
                'start.json': hashlib.sha256(start_bytes).hexdigest()}
    for name, key in (('config.json', 'config_sha256'), ('samples.json', 'samples_sha256')):
        if start.get(key) != identity[name]:
            raise ValueError(f'{key}: missing or mismatched collection hash')
    config = json.loads(config_bytes)
    manifest = json.loads(manifest_bytes)
    rows = json.loads((run_path / 'verification_samples.json').read_text(encoding='utf-8'))
    expected = _sample_coverage(config, manifest, rows)
    gates = config['quality_gate_declared_before_inference']
    for key in ('per_tensor_relative_l2_max', 'reference_detections_match_fraction_min',
                'subset_AP_drop_absolute_max'):
        if not _nonnegative_number(gates.get(key)):
            raise ValueError(f'{key}: threshold must be finite and nonnegative')
    finite_rows = _tensor_rows(rows)
    relative = [t['relative_l2'] for t in finite_rows]
    reference_count = actual_count = matched = 0
    for row in rows:
        matching = row['detection_matching']
        counts = [matching.get(key) for key in ('reference_count', 'actual_count', 'matched')]
        if (any(type(value) is not int or value < 0 for value in counts)
                or counts[2] > min(counts[:2])):
            raise ValueError(f'image {row["image_id"]}: invalid detection counts')
        reference_count += counts[0]
        actual_count += counts[1]
        matched += counts[2]
    fraction = matched / reference_count if reference_count else 1.0
    evaluations = {}
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        for backend in ('onnx', 'rknn'):
            predictions = json.loads((run_path / f'predictions_{backend}.json').read_text(encoding='utf-8'))
            evaluations[backend] = evaluate_coco(root / manifest['subset_annotation'], predictions, expected)
            ap = evaluations[backend].get('AP')
            if not _nonnegative_number(ap) or ap > 1:
                raise ValueError(f'{backend}: AP must be a finite number in [0, 1]')
    (run_path / 'coco_eval.log').write_text(log.getvalue(), encoding='utf-8')
    drop = evaluations['onnx']['AP'] - evaluations['rknn']['AP']
    passed = {'all_outputs_finite': all(t.get('finite') is True for t in finite_rows),
              'per_tensor_relative_l2': all(t.get('finite') is True
                                            and t['relative_l2'] <= gates['per_tensor_relative_l2_max']
                                            for t in finite_rows),
              'detection_agreement': fraction >= gates['reference_detections_match_fraction_min'],
              'subset_AP_drop': drop <= gates['subset_AP_drop_absolute_max']}
    result = {'status': 'passed' if all(passed.values()) else 'failed',
              'sample_count': len(rows), 'tensor_count': len(finite_rows),
              'max_relative_l2': max(relative) if relative else None,
              'reference_detection_count': reference_count, 'rknn_detection_count': actual_count,
              'matched_detections': matched, 'reference_match_fraction': fraction,
              'actual_match_fraction': matched / actual_count if actual_count else None,
              'evaluations': evaluations, 'AP_drop_absolute': drop,
              'gates': gates, 'checks': passed,
              'evidence': {**identity, **{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [run_path / 'verification_samples.json', run_path / 'predictions_onnx.json', run_path / 'predictions_rknn.json']}}}
    result['checks']['complete_validation'] = True
    return result


def main(root, run_path):
    """Invalidate the prior verdict before reading evidence or evaluating COCO."""
    report = run_path / 'verification.json'
    report.unlink(missing_ok=True)
    try:
        result = _evaluate(root, run_path)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    except Exception as exc:
        result = {'status': 'failed', 'checks': {'complete_validation': False},
                  'errors': [f'{type(exc).__name__}: {exc}']}
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    # An interrupted/failed write leaves no prior pass and no partial verdict.
    pending = report.with_suffix('.json.tmp')
    try:
        pending.write_text(payload, encoding='utf-8')
        pending.replace(report)
    finally:
        pending.unlink(missing_ok=True)
    print(payload)
    return 0 if result['status'] == 'passed' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(main(args.root.resolve(), args.root.resolve() / args.run))
