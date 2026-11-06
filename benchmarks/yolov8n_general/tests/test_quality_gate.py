"""Synthetic fixtures test acceptance logic; they are never run evidence."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / 'scripts/quality_gate.py'
    assert path.is_file(), 'quality_gate.py is not implemented yet'
    spec = importlib.util.spec_from_file_location('quality_gate_under_test', path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def protocol():
    return json.loads((ROOT / 'quality_protocol.json').read_text(encoding='utf-8'))


def outputs(p):
    return [np.full(shape, 1.0 if index % 3 == 0 else 0.5, dtype=np.float32)
            for index, shape in enumerate(p['tensor_gates']['expected_shapes'])]


def test_tensor_semantics_shapes_finite_range_and_float64():
    gate, p = module(), protocol()
    ref = outputs(p)
    actual = [a.copy() for a in ref]
    actual[0] += 0.01  # Regression uses relative L2, not probability RMSE.
    actual[1] += 0.0005
    actual[2] += 0.004  # Auxiliary clipped sum uses MAE, not RMSE.
    checked = gate.compare_tensors(ref, actual, p)
    assert checked['passed'] is True
    assert len(checked['tensors']) == 9
    assert checked['tensors'][0]['relative_l2'] == pytest.approx(0.01, abs=1e-7)
    assert checked['tensors'][2]['mean_abs'] == pytest.approx(0.004, abs=1e-7)
    assert gate.compare_tensors(ref, actual[:-1], p)['passed'] is False
    assert gate.compare_tensors(ref + [ref[0]], actual + [actual[0]], p)['passed'] is False
    for index, value in [(0, 1.03), (1, 0.502), (2, 0.506), (4, np.nan), (8, np.inf)]:
        changed = [a.copy() for a in ref]
        changed[index].fill(value)
        report = gate.compare_tensors(ref, changed, p)
        assert report['passed'] is False, index
        assert report['tensors'][index]['passed'] is False
        json.dumps(report, allow_nan=False)
    for index in (1, 2):
        outside = [a.copy() for a in ref]
        outside[index].fill(1.001)
        assert gate.compare_tensors(outside, outside, p)['passed'] is False
    wrong = [a.copy() for a in ref]
    wrong[3] = wrong[3].reshape(1, 64, 20, 80)
    assert gate.compare_tensors(wrong, wrong, p)['passed'] is False
    assert gate.compare_tensors(ref, wrong, p)['passed'] is False
    zero = [np.zeros_like(a) for a in ref]
    assert gate.compare_tensors(zero, zero, p)['passed'] is True
    tiny = [a.copy() for a in zero]
    tiny[0].fill(1e-30)
    assert gate.compare_tensors(zero, tiny, p)['passed'] is False
    changed_protocol = copy.deepcopy(p)
    changed_protocol['tensor_gates']['regression_relative_l2_max'] = 0.005
    assert gate.compare_tensors(ref, actual, changed_protocol)['passed'] is False
    # A float16 norm over these full shapes would overflow before the division.
    half = [a.astype(np.float16) for a in ref]
    half[0].fill(1000)
    half_actual = [a.copy() for a in half]
    half_actual[0] += np.float16(10)
    assert gate.compare_tensors(half, half_actual, p)['tensors'][0]['relative_l2'] == pytest.approx(0.01)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def test_coco_evaluation_keeps_crowd_all_categories_and_zero_prediction_images(tmp_path):
    gate = module()
    assert hasattr(gate, 'evaluate_coco'), 'complete COCO evaluation is not implemented'
    categories = [{'id': cid, 'name': f'category-{cid}'} for cid in range(1, 81)]
    annotations = [
        {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 8, 8], 'area': 64, 'iscrowd': 0},
        {'id': 2, 'image_id': 1, 'category_id': 1, 'bbox': [50, 50, 150, 150], 'area': 22500, 'iscrowd': 1},
        {'id': 3, 'image_id': 2, 'category_id': 1, 'bbox': [0, 0, 40, 40], 'area': 1600, 'iscrowd': 0},
        {'id': 4, 'image_id': 2, 'category_id': 1, 'bbox': [60, 60, 120, 120], 'area': 14400, 'iscrowd': 0},
        {'id': 5, 'image_id': 3, 'category_id': 1, 'bbox': [0, 0, 8, 8], 'area': 64, 'iscrowd': 0},
        {'id': 6, 'image_id': 4, 'category_id': 1, 'bbox': [0, 0, 8, 8], 'area': 64, 'iscrowd': 0},
    ]
    annotation_path = tmp_path / 'instances.json'
    save_json(annotation_path, {'info': {}, 'images': [{'id': i} for i in range(1, 5)],
                                'categories': categories, 'annotations': annotations})
    predictions = [{'image_id': a['image_id'], 'category_id': a['category_id'],
                    'bbox': a['bbox'], 'score': 0.9} for a in annotations[:4]]
    evaluated = gate.evaluate_coco(annotation_path, predictions, [1, 2, 3], protocol())
    assert evaluated['image_ids'] == [1, 2, 3]
    assert evaluated['category_ids'] == list(range(1, 81))
    assert evaluated['max_dets'] == [1, 10, 100]
    assert len(evaluated['all_stats']) == 12
    assert len(evaluated['per_category']) == 80
    assert 0 < evaluated['AP'] < 1  # The missed third image is included.
    assert evaluated['AP_medium'] == pytest.approx(1)
    assert evaluated['AP_large'] == pytest.approx(1)
    first = evaluated['per_category'][0]
    assert first['non_crowd_instances'] == 4
    assert first['image_count'] == 3
    assert first['crowd_instances'] == 1
    assert evaluated['per_category'][1]['AP'] is None
    empty = gate.evaluate_coco(annotation_path, [], [1, 2, 3], protocol())
    assert empty['AP'] == 0
    assert empty['AR100'] == 0


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows), encoding='utf-8')


def frozen_run(tmp_path):
    """Small independent protocol copy; production protocol is never edited."""
    root, run = tmp_path / 'fixture', tmp_path / 'fixture/runs/acceptance_01'
    run.mkdir(parents=True)
    p = protocol()
    p['dataset'].update(acceptance_count=4, all_image_count=5, development_count=1)
    p['tensor_gates']['expected_shapes'] = [[1, channels, 1, 1] for channels in [64, 80, 1] * 3]
    annotations = [
        {'id': i, 'image_id': i, 'category_id': 1, 'bbox': [0, 0, side, side],
         'area': side * side, 'iscrowd': 0} for i, side in enumerate([8, 40, 120], 1)]
    save_json(root / 'dataset/instances_val2017.json', {
        'info': {}, 'images': [{'id': i} for i in range(1, 6)], 'annotations': annotations,
        'categories': [{'id': i, 'name': f'category-{i}'} for i in range(1, 81)]})
    p['dataset']['annotation_sha256'] = digest(root / 'dataset/instances_val2017.json')
    save_json(root / 'dataset/development_source.json', {
        'samples': [{'id': 5, 'sample_id': 'fixture-5', 'input_sha256': 'a' * 64}]})
    p['dataset']['development_manifest_sha256'] = digest(root / 'dataset/development_source.json')
    save_json(root / 'dataset/split.json', {
        'dev_ids': [5], 'acceptance_ids': [1, 2, 3, 4],
        'annotation_sha256': p['dataset']['annotation_sha256'],
        'development_manifest_sha256': p['dataset']['development_manifest_sha256'],
        'development_source': 'Z:/obsolete-machine/development.json'})
    save_json(root / 'quality_protocol.json', p)
    save_json(root / 'dataset/acceptance_samples.json', {
        'split_sha256': digest(root / 'dataset/split.json'),
        'samples': [{'id': i, 'sample_id': f'fixture-{i}', 'input_sha256': 'a' * 64}
                    for i in range(1, 5)]})
    (root / 'models').mkdir()
    (root / 'scripts').mkdir()
    for name in ('synthetic.onnx', 'synthetic.rknn'):
        (root / 'models' / name).write_bytes(b'SYNTHETIC TEST MODEL - NOT INFERENCE EVIDENCE')
    (root / 'scripts/runner.py').write_bytes(b'# synthetic test source\n')
    config = {f'{backend}_sha256': digest(root / f'models/synthetic.{backend}')
              for backend in ('onnx', 'rknn')}
    config.update({f'{backend}_path': f'models/synthetic.{backend}' for backend in ('onnx', 'rknn')})
    config['development_manifest_path'] = 'dataset/development_source.json'
    save_json(root / 'config.json', config)
    start = {key: digest(root / path) for key, path in {
        'protocol_sha256': 'quality_protocol.json', 'config_sha256': 'config.json',
        'split_sha256': 'dataset/split.json',
        'manifest_sha256': 'dataset/acceptance_samples.json'}.items()}
    start['model_sha256'] = {f'models/synthetic.{b}': digest(root / f'models/synthetic.{b}')
                             for b in ('onnx', 'rknn')}
    start['source_sha256'] = {'runner.py': digest(root / 'scripts/runner.py')}
    save_json(run / 'start.json', start)
    arrays = outputs(p)
    tensors = module().compare_tensors(arrays, arrays, p)
    rows = []
    predictions = []
    for i in range(1, 5):
        count = int(i < 4)
        rows.append({'sample_id': f'fixture-{i}', 'image_id': i, 'status': 'success',
                     'tensors': copy.deepcopy(tensors),
                     'input_sha256': 'a' * 64, 'reference_output_sha256': 'b' * 64,
                     'actual_output_sha256': 'c' * 64,
                     'detection_matching': {'reference_count': count, 'actual_count': count,
                         'matched': count, 'reference_match_fraction': 1.0, 'actual_match_fraction': 1.0,
                         'matches': [{'reference_index': 0, 'actual_index': 0, 'iou': 1.0,
                                      'score_abs': 0.0, 'coordinate_max_abs_px': 0.0}] if count else []}})
        predictions.append({'image_id': i, 'predictions': [
            {'image_id': i, 'category_id': a['category_id'], 'bbox': a['bbox'], 'score': 0.9}
            for a in annotations if a['image_id'] == i]})
    save_jsonl(run / 'quality_samples.jsonl', rows)
    for backend in ('onnx', 'rknn'):
        save_jsonl(run / f'predictions_{backend}.jsonl', predictions)
    return root, run, rows, predictions


def test_finalize_complete_run_and_real_cli(tmp_path):
    gate = module()
    assert hasattr(gate, 'finalize'), 'acceptance finalization is not implemented'
    root, run, _, _ = frozen_run(tmp_path)
    result = gate.finalize(root, run)
    assert result['status'] == 'passed'
    assert result['formal_acceptance'] is True
    assert all(result['checks'].values())
    assert result['sample_count'] == 4
    assert result['tensor_count'] == 36
    assert result['reference_detection_count'] == result['actual_detection_count'] == 3
    assert result['reference_match_fraction'] == result['actual_match_fraction'] == 1
    assert result['failed_samples'] == []
    assert len(result['evaluations']['rknn']['per_category']) == 80
    assert json.loads((run / 'quality.json').read_text(encoding='utf-8')) == result
    for evidence in result['evidence']:
        assert digest(root / evidence['path']) == evidence['sha256']
    process = subprocess.run([sys.executable, str(ROOT / 'scripts/quality_gate.py'), '--root', str(root),
                              '--run', 'runs/acceptance_01'], capture_output=True, text=True,
                             env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert process.returncode == 0, process.stderr
    save_jsonl(run / 'predictions_rknn.jsonl', [])
    process = subprocess.run([sys.executable, str(ROOT / 'scripts/quality_gate.py'), '--root', str(root),
                              '--run', 'runs/acceptance_01'], capture_output=True, text=True,
                             env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert process.returncode == 2
    assert json.loads((run / 'quality.json').read_text(encoding='utf-8'))['formal_acceptance'] is False


def test_finalize_zero_predictions_fails_quality_without_losing_coverage(tmp_path):
    root, run, _, predictions = frozen_run(tmp_path)
    save_jsonl(run / 'predictions_rknn.jsonl', [{'image_id': row['image_id'], 'predictions': []}
                                               for row in predictions])
    result = module().finalize(root, run)
    assert result['status'] == 'failed'
    assert result['formal_acceptance'] is False
    assert result['checks']['rknn_prediction_coverage'] is True
    assert result['evaluations']['rknn']['AP'] == 0
    assert result['checks']['minimum_AP'] is False
    assert result['checks']['maximum_drop_AP_small'] is False


def test_metric_category_population_and_size_gates_are_exact():
    gate, p = module(), protocol()
    assert hasattr(gate, 'evaluate_metric_gates'), 'metric and category gates are not implemented'
    metrics = {name: 0.8 for name in ('AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large', 'AR100')}
    categories = [
        {'category_id': 1, 'name': 'eligible-boundary', 'AP': 0.8, 'non_crowd_instances': 100, 'image_count': 20},
        {'category_id': 2, 'name': 'too-few-instances', 'AP': 0.8, 'non_crowd_instances': 99, 'image_count': 100},
        {'category_id': 3, 'name': 'too-few-images', 'AP': 0.8, 'non_crowd_instances': 100, 'image_count': 19},
        {'category_id': 4, 'name': 'absent', 'AP': None, 'non_crowd_instances': 0, 'image_count': 0},
    ]
    evaluations = {'onnx': {**metrics, 'per_category': categories},
                   'rknn': {**metrics, 'per_category': copy.deepcopy(categories)}}
    for cat in evaluations['rknn']['per_category'][:3]:
        cat['AP'] = 0.75 if cat['category_id'] == 1 else 0.0
    result = gate.evaluate_metric_gates(evaluations, p)
    assert all(result['checks'].values())
    assert [c['hard_gate'] for c in result['category_comparisons']] == [True, False, False, False]
    assert result['category_comparisons'][0]['AP_drop'] == 0.05
    for metric in p['evaluation']['maximum_drop']:
        changed = copy.deepcopy(evaluations)
        changed['rknn'][metric] = 0.7
        checked = gate.evaluate_metric_gates(changed, p)
        assert checked['checks']['maximum_drop_' + metric] is False
    for metric, minimum in p['evaluation']['minimum'].items():
        changed = copy.deepcopy(evaluations)
        changed['rknn'][metric] = minimum - 0.01
        assert gate.evaluate_metric_gates(changed, p)['checks']['minimum_' + metric] is False
    changed = copy.deepcopy(evaluations)
    changed['rknn']['per_category'][0]['AP'] = 0.749
    assert gate.evaluate_metric_gates(changed, p)['checks']['category_AP_drop'] is False
    changed['rknn']['per_category'][0]['AP'] = None
    assert gate.evaluate_metric_gates(changed, p)['checks']['category_AP_drop'] is False


@pytest.mark.parametrize('stream', ['quality_samples', 'predictions_onnx', 'predictions_rknn'])
@pytest.mark.parametrize('defect', ['duplicate', 'missing_zero_prediction_image', 'unknown', 'truncated'])
def test_coverage_rejects_duplicate_missing_unknown_and_partial_records(tmp_path, stream, defect):
    root, run, rows, predictions = frozen_run(tmp_path)
    records = copy.deepcopy(rows if stream == 'quality_samples' else predictions)
    if defect == 'duplicate':
        records[-1] = copy.deepcopy(records[0])
    elif defect == 'missing_zero_prediction_image':
        records.pop()
    elif defect == 'unknown':
        records[-1]['image_id'] = 999
    path = run / (stream + '.jsonl')
    save_jsonl(path, records)
    if defect == 'truncated':
        with path.open('a', encoding='utf-8') as handle:
            handle.write('{"image_id":')
    result = module().finalize(root, run)
    assert result['formal_acceptance'] is False
    assert result['errors']


@pytest.mark.parametrize('defect', ['metric', 'null_metric', 'shape', 'range', 'output_count', 'finite', 'index',
                                   'matches_missing', 'duplicate_match', 'iou', 'hash', 'sample_id', 'runtime'])
def test_record_errors_cannot_hide_behind_passed_flag(tmp_path, defect):
    root, run, rows, _ = frozen_run(tmp_path)
    row = rows[0]
    tensor = row['tensors']['tensors'][1]
    if defect == 'metric':
        tensor['rmse'] = 0.1
    elif defect == 'null_metric':
        tensor['rmse'] = None
        tensor['mean_abs'] = None
        tensor['max_abs'] = None
    elif defect == 'shape':
        tensor['actual_shape'] = [1, 80, 1, 2]
    elif defect == 'range':
        tensor['actual_max'] = 1.01
    elif defect == 'output_count':
        row['tensors']['actual_count'] = 10
    elif defect == 'finite':
        tensor['finite'] = False
    elif defect == 'index':
        tensor['index'] = 0
    elif defect == 'matches_missing':
        row['detection_matching']['matches'] = []
    elif defect == 'duplicate_match':
        matching = row['detection_matching']
        matching.update(reference_count=2, actual_count=2, matched=2)
        matching['matches'].append(copy.deepcopy(matching['matches'][0]))
    elif defect == 'iou':
        row['detection_matching']['matches'][0]['iou'] = 0.89
    elif defect == 'hash':
        del row['input_sha256']
    elif defect == 'sample_id':
        row['sample_id'] = 'fixture-2'
    else:
        row.clear()
        row.update(image_id=1, sample_id='fixture-1', status='failed', error='synthetic execution failure')
    save_jsonl(run / 'quality_samples.jsonl', rows)
    result = module().finalize(root, run)
    assert result['formal_acceptance'] is False, defect
    assert any(item['image_id'] == 1 for item in result['failed_samples']), defect
    assert set(result['evaluations']) == {'onnx', 'rknn'}  # Numerical failures retain predictions.


@pytest.mark.parametrize('backend', ['reference', 'actual'])
def test_detection_matching_requires_both_aggregate_fractions(tmp_path, backend):
    root, run, rows, _ = frozen_run(tmp_path)
    matching = rows[0]['detection_matching']
    matching[backend + '_count'] = 2
    matching[backend + '_match_fraction'] = 0.5
    save_jsonl(run / 'quality_samples.jsonl', rows)
    result = module().finalize(root, run)
    assert result['formal_acceptance'] is False
    assert result['checks'][backend + '_detection_agreement'] is False
    other = 'actual' if backend == 'reference' else 'reference'
    assert result['checks'][other + '_detection_agreement'] is True


@pytest.mark.parametrize('target', ['quality_protocol.json', 'config.json', 'dataset/acceptance_samples.json',
                                    'dataset/instances_val2017.json', 'models/synthetic.onnx',
                                    'models/synthetic.rknn', 'scripts/runner.py'])
def test_frozen_identity_changes_fail_acceptance(tmp_path, target):
    root, run, _, _ = frozen_run(tmp_path)
    with (root / target).open('ab') as handle:
        handle.write(b' ')
    result = module().finalize(root, run)
    assert result['formal_acceptance'] is False
    assert any('SHA-256' in error for error in result['errors'])


def update_config(root, run, config):
    save_json(root / 'config.json', config)
    start = json.loads((run / 'start.json').read_text(encoding='utf-8'))
    start['config_sha256'] = digest(root / 'config.json')
    save_json(run / 'start.json', start)


@pytest.mark.parametrize('backend', ['onnx', 'rknn'])
@pytest.mark.parametrize('defect', ['different_path', 'same_bytes_different_path', 'missing_path',
                                   'ambiguous_path', 'missing_file', 'escaped_path', 'empty_path'])
def test_model_lock_binds_configured_file(tmp_path, backend, defect):
    root, run, _, _ = frozen_run(tmp_path)
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    key = backend + '_path'
    if defect in ('different_path', 'same_bytes_different_path', 'ambiguous_path'):
        other = root / f'models/other.{backend}'
        other.write_bytes((root / config[key]).read_bytes() if defect == 'same_bytes_different_path'
                          else b'DIFFERENT SYNTHETIC MODEL')
        if defect == 'ambiguous_path':
            config[backend + '_model'] = other.relative_to(root).as_posix()
        else:
            config[key] = other.relative_to(root).as_posix()
    elif defect == 'missing_path':
        del config[key]
    elif defect == 'missing_file':
        config[key] = f'models/missing.{backend}'
    elif defect == 'escaped_path':
        config[key] = f'../outside.{backend}'
    else:
        config[key] = ''
    update_config(root, run, config)
    result = module().finalize(root, run)
    assert result['formal_acceptance'] is False
    assert result['checks']['model_lock'] is False
    assert any('model_lock' in error for error in result['errors'])


@pytest.mark.parametrize('key_style', ['path', 'basename', 'backend'])
@pytest.mark.parametrize('config_key', ['path', 'model'])
def test_model_lock_explicit_paths_support_recorded_key_styles(tmp_path, key_style, config_key):
    root, run, _, _ = frozen_run(tmp_path)
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    if config_key == 'model':
        for backend in ('onnx', 'rknn'):
            config[backend + '_model'] = config.pop(backend + '_path')
    update_config(root, run, config)
    start = json.loads((run / 'start.json').read_text(encoding='utf-8'))
    start['model_sha256'] = {
        (b if key_style == 'backend' else f'synthetic.{b}' if key_style == 'basename'
         else f'models/synthetic.{b}'): config[b + '_sha256'] for b in ('onnx', 'rknn')}
    save_json(run / 'start.json', start)
    result = module().finalize(root, run)
    assert result['formal_acceptance'] is True, result['errors']


def update_locked_json(root, run, name, value, start_key):
    save_json(root / name, value)
    start = json.loads((run / 'start.json').read_text(encoding='utf-8'))
    start[start_key] = digest(root / name)
    save_json(run / 'start.json', start)


def update_split(root, run, split):
    update_locked_json(root, run, 'dataset/split.json', split, 'split_sha256')
    manifest = json.loads((root / 'dataset/acceptance_samples.json').read_text(encoding='utf-8'))
    manifest['split_sha256'] = digest(root / 'dataset/split.json')
    update_locked_json(root, run, 'dataset/acceptance_samples.json', manifest, 'manifest_sha256')


@pytest.mark.parametrize('defect', [
    'split_hash_missing', 'split_hash_wrong', 'split_bytes_changed', 'split_file_missing',
    'manifest_split_missing', 'manifest_split_wrong', 'development_path_missing',
    'development_path_absolute', 'development_path_escape', 'development_file_missing',
    'development_bytes_changed', 'protocol_development_hash', 'split_development_hash',
    'split_annotation_hash', 'split_dev_ids', 'split_acceptance_ids', 'split_duplicate_dev',
    'split_duplicate_acceptance', 'development_ids', 'development_duplicate_ids'])
def test_dataset_locks_reject_unbound_partition(tmp_path, defect):
    root, run, _, _ = frozen_run(tmp_path)
    if defect.startswith('split_hash'):
        start = json.loads((run / 'start.json').read_text(encoding='utf-8'))
        if defect == 'split_hash_missing':
            del start['split_sha256']
        else:
            start['split_sha256'] = '0' * 64
        save_json(run / 'start.json', start)
    elif defect in ('split_bytes_changed', 'development_bytes_changed'):
        name = 'split.json' if defect.startswith('split') else 'development_source.json'
        with (root / 'dataset' / name).open('ab') as handle:
            handle.write(b' ')
    elif defect in ('split_file_missing', 'development_file_missing'):
        name = 'split.json' if defect.startswith('split') else 'development_source.json'
        (root / 'dataset' / name).unlink()
    elif defect.startswith('manifest_split'):
        manifest = json.loads((root / 'dataset/acceptance_samples.json').read_text(encoding='utf-8'))
        if defect == 'manifest_split_missing':
            del manifest['split_sha256']
        else:
            manifest['split_sha256'] = '0' * 64
        update_locked_json(root, run, 'dataset/acceptance_samples.json', manifest, 'manifest_sha256')
    elif defect.startswith('development_path'):
        config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
        if defect == 'development_path_missing':
            del config['development_manifest_path']
        elif defect == 'development_path_absolute':
            config['development_manifest_path'] = str(root / 'dataset/development_source.json')
        else:
            outside = root.parent / 'outside.json'
            outside.write_bytes((root / 'dataset/development_source.json').read_bytes())
            config['development_manifest_path'] = '../outside.json'
        update_config(root, run, config)
    elif defect == 'protocol_development_hash':
        p = json.loads((root / 'quality_protocol.json').read_text(encoding='utf-8'))
        p['dataset']['development_manifest_sha256'] = '0' * 64
        update_locked_json(root, run, 'quality_protocol.json', p, 'protocol_sha256')
    else:
        split = json.loads((root / 'dataset/split.json').read_text(encoding='utf-8'))
        if defect in ('development_ids', 'development_duplicate_ids'):
            ids = [4] if defect == 'development_ids' else [5, 5]
            save_json(root / 'dataset/development_source.json', {'samples': [{'id': i} for i in ids]})
            p = json.loads((root / 'quality_protocol.json').read_text(encoding='utf-8'))
            p['dataset']['development_manifest_sha256'] = digest(root / 'dataset/development_source.json')
            update_locked_json(root, run, 'quality_protocol.json', p, 'protocol_sha256')
            split['development_manifest_sha256'] = p['dataset']['development_manifest_sha256']
        else:
            key, value = {
                'split_development_hash': ('development_manifest_sha256', '0' * 64),
                'split_annotation_hash': ('annotation_sha256', '0' * 64),
                'split_dev_ids': ('dev_ids', [4]),
                'split_acceptance_ids': ('acceptance_ids', [1, 2, 3, 5]),
                'split_duplicate_dev': ('dev_ids', [5, 5]),
                'split_duplicate_acceptance': ('acceptance_ids', [1, 2, 3, 3]),
            }[defect]
            split[key] = value
        update_split(root, run, split)
    result = module().finalize(root, run)
    assert result['formal_acceptance'] is False
    assert result['errors'], defect


@pytest.mark.parametrize('rewrite_split', [False, True])
def test_dataset_partition_excludes_actual_development_ids(tmp_path, rewrite_split):
    root, run, rows, predictions = frozen_run(tmp_path)
    manifest = json.loads((root / 'dataset/acceptance_samples.json').read_text(encoding='utf-8'))
    manifest['samples'][-1].update(id=5, sample_id='fixture-5')
    rows[-1].update(image_id=5, sample_id='fixture-5')
    predictions[-1]['image_id'] = 5
    update_locked_json(root, run, 'dataset/acceptance_samples.json', manifest, 'manifest_sha256')
    save_jsonl(run / 'quality_samples.jsonl', rows)
    for backend in ('onnx', 'rknn'):
        save_jsonl(run / f'predictions_{backend}.jsonl', predictions)
    if rewrite_split:
        split = json.loads((root / 'dataset/split.json').read_text(encoding='utf-8'))
        split.update(dev_ids=[4], acceptance_ids=[1, 2, 3, 5])
        update_split(root, run, split)
    result = module().finalize(root, run)
    assert result['formal_acceptance'] is False
    assert result['errors']


def test_dataset_development_source_is_relocatable(tmp_path):
    root, run, _, _ = frozen_run(tmp_path)
    relocated = root / 'frozen/preregistered.json'
    relocated.parent.mkdir()
    (root / 'dataset/development_source.json').rename(relocated)
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    config['development_manifest_path'] = 'frozen/preregistered.json'
    update_config(root, run, config)
    result = module().finalize(root, run)
    assert result['formal_acceptance'] is True, result['errors']
    assert {'dataset/split.json', 'frozen/preregistered.json'} <= {e['path'] for e in result['evidence']}


@pytest.mark.parametrize('hash_value', ['missing', None, '', 'A' * 64, 'g' * 64, 'a' * 63, 123])
@pytest.mark.parametrize('status', ['success', 'failed'])
def test_manifest_requires_input_hash_for_every_sample(tmp_path, hash_value, status):
    root, run, rows, _ = frozen_run(tmp_path)
    manifest = json.loads((root / 'dataset/acceptance_samples.json').read_text(encoding='utf-8'))
    if hash_value == 'missing':
        del manifest['samples'][-1]['input_sha256']
    else:
        manifest['samples'][-1]['input_sha256'] = hash_value
    update_locked_json(root, run, 'dataset/acceptance_samples.json', manifest, 'manifest_sha256')
    if status == 'failed':
        rows[-1].update(status='failed', error='synthetic failure')
        save_jsonl(run / 'quality_samples.jsonl', rows)
    result = module().finalize(root, run)
    assert result['formal_acceptance'] is False
    assert result['checks'].get('manifest_input_hashes') is False
    assert any('input_sha256' in error for error in result['errors'])


@pytest.mark.parametrize('defect', ['manifest_missing', 'record_different'])
def test_successful_records_require_matching_manifest_input_hash(tmp_path, defect):
    root, run, rows, _ = frozen_run(tmp_path)
    manifest = json.loads((root / 'dataset/acceptance_samples.json').read_text(encoding='utf-8'))
    if defect == 'manifest_missing':
        del manifest['samples'][0]['input_sha256']
    else:
        rows[0]['input_sha256'] = 'd' * 64
    gate = module()
    p = json.loads((root / 'quality_protocol.json').read_text(encoding='utf-8'))
    checks, _, failed = gate._sample_records(rows, {s['id']: s for s in manifest['samples']}, p)
    assert checks['sample_records'] is False
    assert any('input_sha256' in reason for sample in failed for reason in sample['reasons'])
    update_locked_json(root, run, 'dataset/acceptance_samples.json', manifest, 'manifest_sha256')
    save_jsonl(run / 'quality_samples.jsonl', rows)
    assert gate.finalize(root, run)['formal_acceptance'] is False


@pytest.mark.parametrize('fault', ['metadata_read', 'stream_read', 'identity_hash',
                                  'final_hash', 'unexpected_hash', 'disappearing_evidence', 'log_write'])
def test_io_faults_overwrite_prior_pass_with_failed_report(tmp_path, monkeypatch, capsys, fault):
    root, run, _, _ = frozen_run(tmp_path)
    gate = module()
    assert gate.finalize(root, run)['formal_acceptance'] is True
    if fault == 'metadata_read':
        original = gate._read_json

        def read_metadata(path):
            if path.name == 'config.json':
                raise PermissionError('synthetic metadata read failure')
            return original(path)
        monkeypatch.setattr(gate, '_read_json', read_metadata)
    elif fault == 'stream_read':
        def read_stream(path):
            raise OSError('synthetic stream read failure')
        monkeypatch.setattr(gate, '_read_jsonl', read_stream)
    elif fault == 'log_write':
        (run / 'coco_eval.log').unlink()
        (run / 'coco_eval.log').mkdir()
    else:
        original = gate._sha256

        def sha(path):
            target = 'synthetic.onnx' if fault == 'identity_hash' else 'predictions_rknn.jsonl'
            if path.name == target:
                if fault == 'disappearing_evidence':
                    path.unlink(missing_ok=True)
                    return original(path)
                error = RuntimeError if fault == 'unexpected_hash' else PermissionError
                raise error('synthetic evidence hash failure')
            return original(path)
        monkeypatch.setattr(gate, '_sha256', sha)
    code = None
    try:
        code = gate.main(['--root', str(root), '--run', str(run)])
    except Exception as exc:
        pytest.fail(f'CLI did not contain evidence fault: {type(exc).__name__}: {exc}')
    stdout = json.loads(capsys.readouterr().out)
    assert code == 2
    assert stdout['formal_acceptance'] is False
    report = json.loads((run / 'quality.json').read_text(encoding='utf-8'))
    assert report['formal_acceptance'] is False
    assert report['status'] == 'failed'
    assert report['errors']


def test_report_invalidates_old_pass_before_reading_evidence(tmp_path, monkeypatch):
    root, run, _, _ = frozen_run(tmp_path)
    gate = module()
    assert gate.finalize(root, run)['formal_acceptance'] is True
    original = gate._read_json
    observed = []

    def read(path):
        report = json.loads((run / 'quality.json').read_text(encoding='utf-8'))
        observed.append(report['formal_acceptance'])
        assert report['formal_acceptance'] is False
        return original(path)
    monkeypatch.setattr(gate, '_read_json', read)
    assert gate.finalize(root, run)['formal_acceptance'] is True
    assert observed and not any(observed)


def test_quality_reports_are_replaced_atomically(tmp_path, monkeypatch):
    root, run, _, _ = frozen_run(tmp_path)
    gate = module()
    assert gate.finalize(root, run)['formal_acceptance'] is True
    replace = os.replace
    swaps = []

    def replace_report(source, target, *args, **kwargs):
        source, target = Path(source), Path(target)
        if target == run / 'quality.json':
            assert source != target and source.parent == target.parent
            staged = json.loads(source.read_text(encoding='utf-8'))
            prior = json.loads(target.read_text(encoding='utf-8'))
            swaps.append((prior['formal_acceptance'], staged['formal_acceptance']))
        return replace(source, target, *args, **kwargs)
    monkeypatch.setattr(os, 'replace', replace_report)
    assert gate.finalize(root, run)['formal_acceptance'] is True
    assert swaps == [(True, False), (False, True)]
    assert not list(run.glob('.*.tmp'))


def test_final_report_publish_failure_keeps_failed_state_and_nonzero_cli(tmp_path, monkeypatch, capsys):
    root, run, _, _ = frozen_run(tmp_path)
    gate = module()
    assert gate.finalize(root, run)['formal_acceptance'] is True
    replace = os.replace

    def fail_publish(source, target, *args, **kwargs):
        if Path(target) == run / 'quality.json':
            staged = json.loads(Path(source).read_text(encoding='utf-8'))
            if staged['formal_acceptance']:
                raise PermissionError('synthetic final report publish failure')
        return replace(source, target, *args, **kwargs)
    monkeypatch.setattr(os, 'replace', fail_publish)
    assert gate.main(['--root', str(root), '--run', str(run)]) == 2
    stdout = json.loads(capsys.readouterr().out)
    assert stdout['formal_acceptance'] is False
    assert 'publish failure' in stdout['error']
    assert json.loads((run / 'quality.json').read_text(encoding='utf-8'))['formal_acceptance'] is False
    assert not list(run.glob('.*.tmp'))


def test_cli_reports_blocked_output_directory(tmp_path):
    root, run, _, _ = frozen_run(tmp_path)
    blocked = root / 'runs/blocked'
    blocked.write_bytes(b'FILE BLOCKS OUTPUT DIRECTORY')
    process = subprocess.run([sys.executable, str(ROOT / 'scripts/quality_gate.py'), '--root', str(root),
                              '--run', str(blocked)], capture_output=True, text=True,
                             env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert process.returncode == 2
    stdout = json.loads(process.stdout)
    assert stdout['status'] == 'failed' and stdout['formal_acceptance'] is False
    assert stdout['error']


@pytest.mark.parametrize('target', ['start.json', 'config.json'])
def test_malformed_metadata_overwrites_prior_success_with_failed_report(tmp_path, target):
    root, run, _, _ = frozen_run(tmp_path)
    gate = module()
    assert gate.finalize(root, run)['formal_acceptance'] is True
    path = run / target if target == 'start.json' else root / target
    save_json(path, [])
    result = gate.finalize(root, run)
    assert result['formal_acceptance'] is False
    assert json.loads((run / 'quality.json').read_text(encoding='utf-8'))['status'] == 'failed'


def test_match_detections_contract_uses_classes_and_unique_pairs():
    path = ROOT / 'scripts/verification.py'
    spec = importlib.util.spec_from_file_location('verification_contract', path)
    assert spec is not None and spec.loader is not None
    verification = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verification)
    ref = np.array([[0, 0, 10, 10, .9, 0], [30, 30, 40, 40, .8, 1]], dtype=np.float32)
    actual = np.array([[0, 0, 10, 10, .9, 0], [0, 0, 10, 10, .8, 0],
                       [30, 30, 40, 40, .8, 2]], dtype=np.float32)
    matched = verification.match_detections(ref, actual, protocol()['detection']['matching_iou_min'])
    assert matched['matched'] == 1
    assert matched['reference_count'] == 2
    assert matched['actual_count'] == 3
    errors, _ = module()._matching_record_errors(matched, protocol())
    assert errors == []
    empty = np.empty((0, 6), dtype=np.float32)
    for reference, outputs in ((empty, empty), (empty, actual), (ref, empty)):
        matched = verification.match_detections(reference, outputs, protocol()['detection']['matching_iou_min'])
        assert module()._matching_record_errors(matched, protocol())[0] == []
