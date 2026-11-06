"""Frozen per-tensor gates and complete COCO acceptance for this benchmark.

Model identity contract: config.json must supply exactly one of <backend>_path
or <backend>_model for each of onnx/rknn, plus <backend>_sha256. Paths must name
files inside the benchmark root (relative paths resolve from that root).
start.model_sha256 must contain exactly one entry for each backend: its backend
name or the configured model path (a bare filename resolves under models/).
The configured path, recorded path, configured/recorded hashes and file bytes
must agree; model discovery by hash is intentionally unsupported.

Dataset contract: start.split_sha256 and acceptance_samples.json.split_sha256
must both lock dataset/split.json. config.development_manifest_path is required:
a nonempty root-relative path, resolving inside root, to the original development
manifest whose bytes match protocol.dataset.development_manifest_sha256 and the
split's declaration. split.development_source is historical metadata only.
Development sample IDs must exactly match split.dev_ids; acceptance IDs must
exactly equal both split.acceptance_ids and all annotation IDs minus development
IDs. All three populations must have unique integer IDs and frozen counts.
Every acceptance sample must declare a lowercase 64-hex input_sha256, including
failed samples; every successful quality record must match that frozen hash.

Reports are atomically replaced in the run directory. A failed placeholder
invalidates any prior pass before evidence is read. Evidence and log I/O errors
produce a failed report; if the output cannot be written, the CLI exits 2 with
an error (an inaccessible destination cannot be repaired by the gate).
"""
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _at_most(value, limit):
    return _number(value) and value <= limit


def _rms(array):
    """Scale before float64 reduction to avoid squaring overflow/underflow."""
    scale = float(np.max(np.abs(array)))
    if scale == 0:
        return 0.0
    if not math.isfinite(scale):
        return float('inf')
    return float(scale * np.sqrt(np.mean(np.square(array / scale), dtype=np.float64)))


def compare_tensors(reference, actual, protocol):
    """Compare nine outputs; a zero reference norm accepts only zero error.

    Undefined/non-finite metrics are JSON null and always fail their own gate.
    Probability and auxiliary outputs must both stay in the declared range.
    """
    gates = protocol['tensor_gates']
    shapes = gates['expected_shapes']
    if len(shapes) != 9:
        raise ValueError('The frozen output contract must contain nine shapes')
    rows = []
    kinds = ('regression', 'probability', 'score_sum')
    metrics = ('relative_l2', 'rmse', 'mean_abs')
    limits = (gates['regression_relative_l2_max'], gates['probability_rmse_max'],
              gates['score_sum_mae_max'])
    for index, expected in enumerate(shapes):
        ref = np.asarray(reference[index]) if index < len(reference) else None
        out = np.asarray(actual[index]) if index < len(actual) else None
        row = {'index': index, 'kind': kinds[index % 3], 'shape': list(ref.shape) if ref is not None else None,
               'reference_shape': list(ref.shape) if ref is not None else None,
               'actual_shape': list(out.shape) if out is not None else None,
               'expected_shape': list(expected), 'shape_matches': False, 'finite': False,
               'relative_l2': None, 'rmse': None, 'mean_abs': None, 'max_abs': None,
               'reference_min': None, 'reference_max': None, 'actual_min': None, 'actual_max': None,
               'range_passed': False, 'metric': metrics[index % 3], 'limit': limits[index % 3],
               'passed': False, 'failures': []}
        row['shape_matches'] = bool(ref is not None and out is not None
                                    and list(ref.shape) == list(out.shape) == list(expected))
        if not row['shape_matches']:
            row['failures'].append('shape_or_output_count')
        else:
            try:
                r, a = ref.astype(np.float64), out.astype(np.float64)
                row['finite'] = bool(np.isfinite(r).all() and np.isfinite(a).all())
            except (TypeError, ValueError, OverflowError):
                row['finite'] = False
            if not row['finite']:
                row['failures'].append('non_finite')
            else:
                row.update(reference_min=float(r.min()), reference_max=float(r.max()),
                           actual_min=float(a.min()), actual_max=float(a.max()))
                with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
                    delta = a - r
                    rmse, reference_rms = _rms(delta), _rms(r)
                    relative = rmse / reference_rms if reference_rms else (0.0 if rmse == 0 else float('inf'))
                    errors = {'relative_l2': relative, 'rmse': rmse,
                              'mean_abs': float(np.mean(np.abs(delta), dtype=np.float64)),
                              'max_abs': float(np.max(np.abs(delta)))}
                row.update({key: value if math.isfinite(value) else None for key, value in errors.items()})
                lower, upper = gates['probability_range']
                row['range_passed'] = index % 3 == 0 or (
                    lower <= row['reference_min'] <= row['reference_max'] <= upper
                    and lower <= row['actual_min'] <= row['actual_max'] <= upper)
                if not row['range_passed']:
                    row['failures'].append('probability_range')
                if not _at_most(row[row['metric']], row['limit']):
                    row['failures'].append(row['metric'])
                row['passed'] = not row['failures']
        rows.append(row)
    count_matches = len(reference) == len(actual) == len(shapes)
    return {'passed': count_matches and all(row['passed'] for row in rows),
            'output_count_matches': count_matches, 'reference_count': len(reference),
            'actual_count': len(actual), 'tensors': rows}


METRIC_INDICES = {'AP': 0, 'AP50': 1, 'AP75': 2, 'AP_small': 3,
                  'AP_medium': 4, 'AP_large': 5, 'AR100': 8}


def evaluate_coco(annotation_path, predictions, image_ids, protocol):
    """Evaluate all original categories and retain original crowd annotations."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    truth = COCO(str(annotation_path))
    ids = sorted(image_ids)
    if len(ids) != len(set(ids)) or not set(ids) <= set(truth.imgs):
        raise ValueError('Evaluation image IDs must be unique original image IDs')
    category_ids = sorted(truth.getCatIds())
    if predictions:
        detected = truth.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = {'images': truth.dataset['images'],
                            'categories': truth.dataset['categories'], 'annotations': []}
        detected.createIndex()
    evaluator = COCOeval(truth, detected, 'bbox')
    evaluator.params.imgIds = ids
    evaluator.params.catIds = category_ids
    evaluator.params.maxDets = list(protocol['evaluation']['coco_max_dets'])
    if evaluator.params.maxDets != [1, 10, 100]:
        raise ValueError('COCO standard maxDets [1,10,100] is required')
    evaluator.params.useCats = 1
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    selected = set(ids)
    instances, crowds, images = Counter(), Counter(), {cid: set() for cid in category_ids}
    for ann in truth.dataset['annotations']:
        if ann['image_id'] in selected:
            cid = ann['category_id']
            if ann.get('iscrowd', 0):
                crowds[cid] += 1
            else:
                instances[cid] += 1
                images[cid].add(ann['image_id'])
    per_category = []
    precision = evaluator.eval['precision']
    for index, cid in enumerate(category_ids):
        values = precision[:, :, index, 0, -1]
        valid = values[values >= 0]
        per_category.append({'category_id': cid, 'name': truth.cats[cid]['name'],
                             'AP': float(valid.mean(dtype=np.float64)) if valid.size else None,
                             'non_crowd_instances': instances[cid], 'image_count': len(images[cid]),
                             'crowd_instances': crowds[cid]})
    result = {name: float(evaluator.stats[index]) if evaluator.stats[index] >= 0 else None
              for name, index in METRIC_INDICES.items()}
    result.update(all_stats=evaluator.stats.tolist(), image_count=len(ids), image_ids=ids,
                  category_ids=category_ids, max_dets=evaluator.params.maxDets,
                  per_category=per_category,
                  scope='Frozen acceptance images; all original categories and crowd annotations retained')
    return result


def _sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _sha_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _reject_constant(value):
    raise ValueError(f'Non-finite JSON literal: {value}')


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=_reject_constant)


def _read_jsonl(path):
    rows, errors = [], []
    try:
        with path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line, parse_constant=_reject_constant)
                    if not isinstance(row, dict):
                        raise ValueError('Expected a JSON object')
                    rows.append(row)
                except (ValueError, TypeError) as exc:
                    errors.append(f'{path.name}:{line_number}: {exc}')
    except OSError as exc:
        errors.append(str(exc))
    return rows, errors


def _under_root(root, name, folder=None):
    path = Path(name)
    if folder and len(path.parts) == 1:
        path = Path(folder) / path
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Evidence path escapes benchmark root: {name}')
    return path


def _verify_identity(root, run, protocol, config, start, evidence):
    checks, errors = {}, []

    def verify(label, path, expected):
        try:
            actual = _sha256(path)
            evidence.add(path)
            good = _sha_string(expected) and actual == expected
        except OSError as exc:
            good = False
            errors.append(f'{label}: {exc}')
        checks[label] = bool(good)
        if not good:
            errors.append(f'{label}: missing or mismatched SHA-256 for {path.name}')

    for label, filename, key in (
        ('protocol_lock', 'quality_protocol.json', 'protocol_sha256'),
        ('config_lock', 'config.json', 'config_sha256'),
        ('manifest_lock', 'dataset/acceptance_samples.json', 'manifest_sha256'),
    ):
        verify(label, root / filename, start.get(key))
    verify('annotation_lock', root / 'dataset/instances_val2017.json',
           protocol['dataset']['annotation_sha256'])
    sources = start.get('source_sha256', {})
    checks['source_lock'] = isinstance(sources, dict) and bool(sources)
    if checks['source_lock']:
        for name, expected in sources.items():
            path = _under_root(root, name, 'scripts')
            verify('source:' + name, path, expected)
            checks['source_lock'] &= checks['source:' + name]
    else:
        errors.append('source_lock: source_sha256 mapping is required')
    models = start.get('model_sha256', {})
    checks['model_lock'] = isinstance(models, dict) and bool(models)
    for backend in ('onnx', 'rknn'):
        try:
            keys = [key for key in (backend + '_path', backend + '_model') if key in config]
            if len(keys) != 1 or not isinstance(config[keys[0]], str) or not config[keys[0]].strip():
                raise ValueError('exactly one explicit config model path is required')
            path = _under_root(root, config[keys[0]])
            expected = config.get(backend + '_sha256')
            if path.suffix != '.' + backend or not _sha_string(expected):
                raise ValueError('config model extension or SHA-256 is invalid')
            candidates = []
            if isinstance(models, dict):
                for name, recorded in models.items():
                    if name == backend:
                        candidates.append((path, recorded))
                    elif Path(name).suffix == '.' + backend:
                        candidates.append((_under_root(root, name, 'models'), recorded))
            if len(candidates) != 1 or candidates[0] != (path, expected):
                raise ValueError('exactly one start model entry must match the config path and SHA-256')
            verify('model:' + backend, path, expected)
            checks['model_lock'] &= checks['model:' + backend]
        except (OSError, ValueError, TypeError) as exc:
            checks['model_lock'] = False
            errors.append(f'model_lock: {backend}: {exc}')
    return checks, errors


def _verify_dataset_partition(root, protocol, config, start, manifest, image_ids, evidence):
    checks, errors = {}, []

    def check(label, passed, reason):
        checks[label] = bool(passed)
        if not passed:
            errors.append(f'{label}: {reason}')

    def ids(values, label):
        if (not isinstance(values, list) or any(type(value) is not int for value in values)
                or len(values) != len(set(values))):
            raise ValueError(f'{label}: unique integer image IDs are required')
        return set(values)

    split_path = root / 'dataset/split.json'
    evidence.add(split_path)
    split_hash = _sha256(split_path)
    check('split_lock', _sha_string(start.get('split_sha256')) and start['split_sha256'] == split_hash,
          'start split SHA-256 is missing or mismatched')
    check('manifest_split_lock', manifest.get('split_sha256') == split_hash,
          'acceptance manifest split SHA-256 is missing or mismatched')
    split = _read_json(split_path)
    dataset = protocol['dataset']
    check('split_annotation_lock', split.get('annotation_sha256') == dataset['annotation_sha256'],
          'split annotation SHA-256 differs from protocol')
    name = config.get('development_manifest_path')
    if not isinstance(name, str) or not name.strip() or Path(name).anchor:
        raise ValueError('development_manifest_path: explicit root-relative path is required')
    development_path = _under_root(root, name)
    evidence.add(development_path)
    declared = dataset.get('development_manifest_sha256')
    check('development_manifest_lock', _sha_string(declared)
          and _sha256(development_path) == declared == split.get('development_manifest_sha256'),
          'development manifest SHA-256 must match protocol and split')
    development = _read_json(development_path)['samples']
    if not isinstance(development, list) or any(not isinstance(s, dict) for s in development):
        raise ValueError('development manifest: samples must be objects')
    dev_ids = ids([sample.get('id') for sample in development], 'development manifest')
    all_ids = ids(image_ids, 'annotation')
    split_dev = ids(split.get('dev_ids'), 'split development')
    split_acceptance = ids(split.get('acceptance_ids'), 'split acceptance')
    acceptance = ids([sample['id'] for sample in manifest['samples']], 'acceptance manifest')
    check('development_ids', dev_ids == split_dev and dev_ids <= all_ids
          and len(dev_ids) == dataset['development_count'],
          'development IDs must match the preregistered source, split and frozen count')
    check('acceptance_partition', acceptance == split_acceptance == all_ids - dev_ids
          and len(acceptance) == dataset['acceptance_count'],
          'acceptance IDs must equal split.acceptance_ids and all IDs minus development IDs')
    return checks, errors


def _coverage(rows, expected, label):
    ids = [row.get('image_id') for row in rows]
    if any(type(image_id) is not int for image_id in ids):
        return False, [f'{label}: every image_id must be an integer']
    counts = Counter(ids)
    duplicate = sorted(image_id for image_id, count in counts.items() if count > 1)
    missing, unknown = sorted(set(expected) - counts.keys()), sorted(counts.keys() - set(expected))
    errors = []
    for kind, values in (('duplicate', duplicate), ('missing', missing), ('unknown', unknown)):
        if values:
            errors.append(f'{label}: {kind} image IDs: {values}')
    return not errors, errors


def _prediction_records(rows, expected, categories, protocol, label):
    valid, errors = _coverage(rows, expected, label)
    predictions = []
    for row in rows:
        items = row.get('predictions')
        if not isinstance(items, list):
            errors.append(f'{label}: image {row.get("image_id")} has no predictions list')
            continue
        if len(items) > protocol['evaluation']['max_detections']:
            errors.append(f'{label}: too many predictions for image {row.get("image_id")}')
        for item in items:
            if not isinstance(item, dict):
                errors.append(f'{label}: prediction is not an object')
                continue
            box = item.get('bbox')
            good = (item.get('image_id') == row.get('image_id')
                    and type(item.get('category_id')) is int and item['category_id'] in categories
                    and isinstance(box, list) and len(box) == 4 and all(_number(v) for v in box)
                    and box[2] > 0 and box[3] > 0
                    and _number(item.get('score'))
                    and protocol['evaluation']['confidence'] <= item['score'] <= 1)
            if not good:
                errors.append(f'{label}: invalid COCO prediction for image {row.get("image_id")}')
            else:
                predictions.append(item)
    return valid and not errors, predictions, errors


def _fraction(matched, count, other_count):
    return matched / count if count else (1.0 if other_count == 0 else 0.0)


def _tensor_record_errors(report, protocol):
    gates = protocol['tensor_gates']
    expected = gates['expected_shapes']
    if not isinstance(report, dict) or not isinstance(report.get('tensors'), list):
        return ['missing tensor comparison'], 0, False
    tensors = report['tensors']
    errors = []
    if (len(tensors) != len(expected) or report.get('reference_count') != len(expected)
            or report.get('actual_count') != len(expected) or report.get('output_count_matches') is not True):
        errors.append('output count does not match the frozen nine-output contract')
    if report.get('passed') is not True:
        errors.append('tensor comparison reported a failure')
    finite = len(tensors) == len(expected)
    metric_names = ('relative_l2', 'rmse', 'mean_abs')
    limits = (gates['regression_relative_l2_max'], gates['probability_rmse_max'], gates['score_sum_mae_max'])
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, dict) or index >= len(expected):
            errors.append(f'tensor {index}: invalid tensor record')
            finite = False
            continue
        if tensor.get('index') != index:
            errors.append(f'tensor {index}: incorrect index')
        if (any(tensor.get(key) != expected[index] for key in ('shape', 'reference_shape', 'actual_shape'))
                or tensor.get('shape_matches') is not True):
            errors.append(f'tensor {index}: incorrect shape')
        if tensor.get('finite') is not True:
            finite = False
            errors.append(f'tensor {index}: non-finite values')
        metric = metric_names[index % 3]
        value = tensor.get(metric)
        if not _at_most(value, limits[index % 3]) or value < 0:
            errors.append(f'tensor {index}: {metric}={value} exceeds {limits[index % 3]} or is invalid')
        for key in ('rmse', 'mean_abs', 'max_abs'):
            value = tensor.get(key)
            if not _number(value) or value < 0:
                errors.append(f'tensor {index}: invalid {key}')
        if index % 3:
            lower, upper = gates['probability_range']
            extrema = [tensor.get(key) for key in ('reference_min', 'reference_max', 'actual_min', 'actual_max')]
            if (not all(_number(v) for v in extrema) or not (
                    lower <= extrema[0] <= extrema[1] <= upper and lower <= extrema[2] <= extrema[3] <= upper)
                    or tensor.get('range_passed') is not True):
                errors.append(f'tensor {index}: probability range violation')
        if tensor.get('passed') is not True:
            errors.append(f'tensor {index}: reported failure')
    return errors, len(tensors), finite


def _matching_record_errors(matching, protocol):
    if not isinstance(matching, dict):
        return ['missing detection matching'], None
    numbers = [matching.get(key) for key in ('reference_count', 'actual_count', 'matched')]
    if (not all(type(value) is int and 0 <= value <= protocol['detection']['max_detections'] for value in numbers)
            or numbers[2] > min(numbers[:2])):
        return ['invalid detection counts'], None
    reference_count, actual_count, matched = numbers
    matches = matching.get('matches')
    if not isinstance(matches, list) or len(matches) != matched:
        return ['detection matching count differs from match records'], numbers
    errors, reference_indices, actual_indices = [], set(), set()
    for match in matches:
        if not isinstance(match, dict):
            errors.append('invalid match record')
            continue
        ri, ai, iou = (match.get(key) for key in ('reference_index', 'actual_index', 'iou'))
        if (type(ri) is not int or type(ai) is not int or not 0 <= ri < reference_count
                or not 0 <= ai < actual_count or ri in reference_indices or ai in actual_indices):
            errors.append('match indices must be valid and one-to-one')
        else:
            reference_indices.add(ri)
            actual_indices.add(ai)
        if not _number(iou) or not protocol['detection']['matching_iou_min'] <= iou <= 1:
            errors.append('match IoU below frozen threshold or invalid')
    # Use verification.match_detections conventions for an empty per-image side.
    fractions = {'reference_match_fraction': matched / reference_count if reference_count else 1.0,
                 'actual_match_fraction': _fraction(matched, actual_count, reference_count)}
    for key, expected in fractions.items():
        value = matching.get(key)
        if not _number(value) or value != expected:
            errors.append(f'inconsistent {key}')
    return errors, numbers


def _sample_records(rows, expected, protocol):
    counts = {'tensor_count': 0, 'reference_detection_count': 0,
              'actual_detection_count': 0, 'matched_detections': 0, 'execution_failed_count': 0}
    checks = {'sample_records': True, 'per_tensor_quality': True, 'all_outputs_finite': True,
              'detection_matching_valid': True}
    failed = []
    for row in rows:
        reasons = []
        image_id = row.get('image_id')
        sample = expected.get(image_id, {})
        if row.get('sample_id') != sample.get('sample_id'):
            checks['sample_records'] = False
            reasons.append('sample_id does not match frozen manifest')
        if row.get('status') == 'failed':
            counts['execution_failed_count'] += 1
            reasons.append('execution_failed: ' + str(row.get('error', 'missing error')))
            if not isinstance(row.get('error'), str) or not row['error']:
                checks['sample_records'] = False
        elif row.get('status') != 'success':
            checks['sample_records'] = False
            counts['execution_failed_count'] += 1
            reasons.append('invalid execution status')
        else:
            for key in ('input_sha256', 'reference_output_sha256', 'actual_output_sha256'):
                must_match = key == 'input_sha256' or key in sample
                if not _sha_string(row.get(key)) or (must_match and row.get(key) != sample.get(key)):
                    checks['sample_records'] = False
                    reasons.append('missing or mismatched ' + key)
            tensor_errors, tensor_count, finite = _tensor_record_errors(row.get('tensors'), protocol)
            counts['tensor_count'] += tensor_count
            if tensor_errors:
                checks['per_tensor_quality'] = False
                reasons.extend(tensor_errors)
            if not finite:
                checks['all_outputs_finite'] = False
            matching_errors, numbers = _matching_record_errors(row.get('detection_matching'), protocol)
            if matching_errors:
                checks['detection_matching_valid'] = False
                reasons.extend(matching_errors)
            if numbers is not None:
                for key, value in zip(('reference_detection_count', 'actual_detection_count', 'matched_detections'), numbers):
                    counts[key] += value
        if reasons:
            failed.append({'sample_id': row.get('sample_id'), 'image_id': image_id, 'reasons': reasons})
    checks['runtime_failures'] = counts['execution_failed_count'] <= protocol['runtime']['failed_samples_max']
    ref, actual, matched = (counts[k] for k in ('reference_detection_count', 'actual_detection_count', 'matched_detections'))
    counts['reference_match_fraction'] = _fraction(matched, ref, actual)
    counts['actual_match_fraction'] = _fraction(matched, actual, ref)
    checks['reference_detection_agreement'] = counts['reference_match_fraction'] >= protocol['detection']['reference_match_fraction_min']
    checks['actual_detection_agreement'] = counts['actual_match_fraction'] >= protocol['detection']['actual_match_fraction_min']
    return checks, counts, failed


def _ap(value):
    return _number(value) and 0 <= value <= 1


def _drop(reference, actual):
    # Subtract the serialized decimal values so an exact .05 boundary stays .05.
    return float(Decimal(str(reference)) - Decimal(str(actual))) if _ap(reference) and _ap(actual) else None


def evaluate_metric_gates(evaluations, protocol):
    """Apply all frozen absolute, relative, size and supported-category gates."""
    reference, actual = evaluations['onnx'], evaluations['rknn']
    gates = protocol['evaluation']
    checks = {}
    for name, limit in gates['minimum'].items():
        value = actual.get(name)
        checks['minimum_' + name] = _ap(value) and value >= limit
    drops = {name: _drop(reference.get(name), actual.get(name)) for name in METRIC_INDICES}
    for name, limit in gates['maximum_drop'].items():
        checks['maximum_drop_' + name] = _at_most(drops.get(name), limit)
    ref_cats = {c['category_id']: c for c in reference['per_category']}
    act_cats = {c['category_id']: c for c in actual['per_category']}
    category_valid = (len(ref_cats) == len(reference['per_category'])
                      and len(act_cats) == len(actual['per_category']) and ref_cats.keys() == act_cats.keys())
    comparisons = []
    category_gate = gates['category_gate']
    for cid, ref in sorted(ref_cats.items()):
        out = act_cats.get(cid, {})
        category_valid &= (ref['non_crowd_instances'] == out.get('non_crowd_instances')
                           and ref['image_count'] == out.get('image_count'))
        supported = (ref['non_crowd_instances'] >= category_gate['minimum_non_crowd_instances']
                     and ref['image_count'] >= category_gate['minimum_images'])
        drop = _drop(ref.get('AP'), out.get('AP'))
        comparisons.append({'category_id': cid, 'name': ref['name'],
                            'non_crowd_instances': ref['non_crowd_instances'], 'image_count': ref['image_count'],
                            'reference_AP': ref.get('AP'), 'actual_AP': out.get('AP'), 'AP_drop': drop,
                            'hard_gate': supported, 'limit': category_gate['maximum_ap_drop'],
                            'passed': _at_most(drop, category_gate['maximum_ap_drop']) if supported else None})
    checks['category_population_consistent'] = bool(category_valid)
    checks['category_AP_drop'] = category_valid and all(c['passed'] for c in comparisons if c['hard_gate'])
    return {'checks': checks, 'AP_drop': drops, 'category_comparisons': comparisons}


def _atomic_write_text(path, text):
    """Stage and flush beside the destination, then replace it atomically."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)


def finalize(root, run_path):
    """Write quality.json and coco_eval.log; errors always prevent acceptance."""
    root = Path(root).resolve()
    run = (root / Path(run_path)).resolve()
    if not run.is_relative_to(root):
        raise ValueError('Run directory must be inside the benchmark root')
    result = {'status': 'failed', 'formal_acceptance': False, 'checks': {}, 'gates': {},
              'failed_samples': [], 'sample_count': 0, 'tensor_count': 0,
              'reference_detection_count': 0, 'actual_detection_count': 0,
              'matched_detections': 0, 'reference_match_fraction': None, 'actual_match_fraction': None,
              'evaluations': {}, 'AP_drop': {}, 'category_comparisons': [],
              'errors': ['Validation did not complete'], 'evidence': []}
    run.mkdir(parents=True, exist_ok=True)
    report_path = run / 'quality.json'
    result['checks']['complete_validation'] = False
    # Invalidate a prior pass before any evidence I/O or evaluation can fail.
    _atomic_write_text(report_path, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    result['errors'].clear()
    evidence = {root / name for name in ('quality_protocol.json', 'config.json',
                 'dataset/acceptance_samples.json', 'dataset/instances_val2017.json')}
    evidence.update(run / name for name in ('start.json', 'quality_samples.jsonl',
                    'predictions_onnx.jsonl', 'predictions_rknn.jsonl'))
    log = io.StringIO()
    try:
        protocol, config, manifest, start = (_read_json(path) for path in (
            root / 'quality_protocol.json', root / 'config.json',
            root / 'dataset/acceptance_samples.json', run / 'start.json'))
        if not all(isinstance(value, dict) for value in (protocol, config, manifest, start)):
            raise ValueError('Protocol, config, manifest and start must be JSON objects')
        result['protocol_id'] = protocol['protocol_id']
        result['gates'] = {key: protocol[key] for key in ('tensor_gates', 'detection', 'evaluation', 'runtime')}
        checks, errors = _verify_identity(root, run, protocol, config, start, evidence)
        result['checks'].update(checks)
        result['errors'].extend(errors)
        samples = manifest['samples']
        if (not isinstance(samples, list) or any(not isinstance(s, dict) or type(s.get('id')) is not int
                or not isinstance(s.get('sample_id'), str) or not s['sample_id'] for s in samples)):
            raise ValueError('Invalid frozen sample IDs')
        invalid_inputs = [s['id'] for s in samples if not _sha_string(s.get('input_sha256'))]
        result['checks']['manifest_input_hashes'] = not invalid_inputs
        if invalid_inputs:
            result['errors'].append(f'manifest input_sha256 missing or invalid for image IDs: {invalid_inputs}')
        expected = {sample['id']: sample for sample in samples}
        result['checks']['manifest_coverage'] = (len(samples) == len(expected) == protocol['dataset']['acceptance_count']
                                                  and len({s['sample_id'] for s in samples}) == len(samples))
        annotation = _read_json(root / 'dataset/instances_val2017.json')
        image_ids = [image['id'] for image in annotation['images']]
        category_ids = [cat['id'] for cat in annotation['categories']]
        result['checks']['annotation_scope'] = (len(image_ids) == len(set(image_ids)) == protocol['dataset']['all_image_count']
                                                and set(expected) <= set(image_ids)
                                                and len(category_ids) == len(set(category_ids)) == 80)
        del annotation
        checks, errors = _verify_dataset_partition(root, protocol, config, start, manifest, image_ids, evidence)
        result['checks'].update(checks)
        result['errors'].extend(errors)
        rows, errors = _read_jsonl(run / 'quality_samples.jsonl')
        result['sample_count'] = len(rows)
        covered, coverage_errors = _coverage(rows, expected, 'quality_samples')
        result['checks']['quality_coverage'] = covered and not errors
        result['errors'].extend(errors + coverage_errors)
        checks, counts, failed = _sample_records(rows, expected, protocol)
        result['checks'].update(checks)
        result.update(counts, failed_samples=failed)
        for backend in ('onnx', 'rknn'):
            rows, errors = _read_jsonl(run / f'predictions_{backend}.jsonl')
            valid, predictions, prediction_errors = _prediction_records(rows, expected, set(category_ids), protocol, backend)
            result['checks'][backend + '_prediction_coverage'] = valid and not errors
            result['errors'].extend(errors + prediction_errors)
            if valid and not errors and result['checks']['annotation_scope']:
                with contextlib.redirect_stdout(log):
                    result['evaluations'][backend] = evaluate_coco(
                        root / 'dataset/instances_val2017.json', predictions, list(expected), protocol)
            del predictions, rows
        result['checks']['coco_evaluation'] = len(result['evaluations']) == 2
        if result['checks']['coco_evaluation']:
            metrics = evaluate_metric_gates(result['evaluations'], protocol)
            result['checks'].update(metrics.pop('checks'))
            result.update(metrics)
        else:
            for name in protocol['evaluation']['minimum']:
                result['checks']['minimum_' + name] = False
            for name in protocol['evaluation']['maximum_drop']:
                result['checks']['maximum_drop_' + name] = False
            result['checks']['category_AP_drop'] = False
    except Exception as exc:
        # This is the acceptance boundary: unexpected evidence/evaluator errors also fail closed.
        result['checks']['complete_validation'] = False
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    else:
        result['checks']['complete_validation'] = True
    try:
        _atomic_write_text(run / 'coco_eval.log', log.getvalue())
        evidence.add(run / 'coco_eval.log')
        result['checks']['evaluation_log'] = True
    except Exception as exc:
        result['checks']['evaluation_log'] = False
        result['errors'].append(f'coco_eval.log: {type(exc).__name__}: {exc}')
    result['checks']['evidence_hashes'] = True
    for path in sorted(evidence):
        try:
            result['evidence'].append({'path': path.relative_to(root).as_posix(), 'sha256': _sha256(path)})
        except Exception as exc:
            result['checks']['evidence_hashes'] = False
            result['errors'].append(f'evidence {path.name}: {type(exc).__name__}: {exc}')
    accepted = bool(result['checks']) and all(result['checks'].values()) and not result['errors']
    result['status'] = 'passed' if accepted else 'failed'
    result['formal_acceptance'] = accepted
    _atomic_write_text(report_path, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = finalize(args.root, args.run)
    except (OSError, ValueError) as exc:
        print(json.dumps({'status': 'failed', 'formal_acceptance': False, 'error': str(exc)}))
        return 2
    print(json.dumps({'status': result['status'], 'formal_acceptance': result['formal_acceptance'],
                      'failed_checks': [key for key, passed in result['checks'].items() if not passed],
                      'errors': result['errors']}, ensure_ascii=False))
    return 0 if result['formal_acceptance'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
