"""Finalizer regressions use only synthetic evidence in pytest temporary dirs."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import finalize_verification


SHAPES = [[1, channels, size, size] for size in (80, 40, 20) for channels in (64, 80, 1)]


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'benchmark'
    run = root / 'runs' / 'synthetic'
    run.mkdir(parents=True)
    (root / 'dataset').mkdir()
    config = {'quality_gate_declared_before_inference': {
        'finite_all_outputs': True, 'per_tensor_relative_l2_max': .02,
        'detection_matching_iou_min': .9, 'reference_detections_match_fraction_min': .95,
        'subset_AP_drop_absolute_max': .01}, 'sampling': {'count': 1}}
    write_json(root / 'config.json', config)
    write_json(root / 'dataset/samples.json', {
        'samples': [{'id': 1, 'sample_id': 'synthetic-1'}],
        'subset_annotation': 'dataset/annotations.json'})
    write_json(root / 'dataset/annotations.json', {
        'info': {}, 'images': [{'id': 1, 'width': 100, 'height': 100}],
        'categories': [{'id': 1, 'name': 'synthetic', 'supercategory': 'synthetic'}],
        'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                         'bbox': [0, 0, 10, 10], 'area': 100, 'iscrowd': 0}]})
    write_json(run / 'start.json', {
        key: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for key, name in [('config_sha256', 'config.json'), ('samples_sha256', 'dataset/samples.json')]})
    rows = [{'image_id': 1, 'sample_id': 'synthetic-1', 'tensors': {
        'passed': True, 'tensors': [
            {'index': index, 'shape': shape, 'finite': True, 'relative_l2': .01,
             'mean_abs': .001, 'max_abs': .002, 'passed': True}
            for index, shape in enumerate(SHAPES)]},
        'detection_matching': {'reference_count': 1, 'actual_count': 1, 'matched': 1}}]
    write_json(run / 'verification_samples.json', rows)
    for backend in ('onnx', 'rknn'):
        write_json(run / f'predictions_{backend}.json', [
            {'image_id': 1, 'category_id': 1, 'bbox': [0, 0, 10, 10], 'score': .9}])
    return root, run


def test_relative_l2_is_recomputed_despite_saved_passed(bundle, capsys):
    root, run = bundle
    rows = read_json(run / 'verification_samples.json')
    rows[0]['tensors']['tensors'][0]['relative_l2'] = .05
    write_json(run / 'verification_samples.json', rows)

    code = finalize_verification.main(root, run)
    result = read_json(run / 'verification.json')
    diagnostic = capsys.readouterr().out
    assert code == 2, f'finalizer exit={code}; diagnostic={diagnostic}'
    assert result['status'] == 'failed'
    assert result['checks']['per_tensor_relative_l2'] is False
    assert result['max_relative_l2'] == .05
    assert result['gates']['per_tensor_relative_l2_max'] == .02


@pytest.mark.parametrize('filename', ['config.json', 'dataset/samples.json'])
def test_changed_frozen_file_is_rejected(bundle, filename):
    root, run = bundle
    path = root / filename
    # Even a byte-only edit must not reuse the original collection identity.
    path.write_text(path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    write_json(run / 'verification.json', {'status': 'passed'})

    assert finalize_verification.main(root, run) == 2
    result = read_json(run / 'verification.json')
    assert result['status'] == 'failed'
    assert 'sha256' in ' '.join(result['errors'])


@pytest.mark.parametrize('key', ['config_sha256', 'samples_sha256'])
def test_missing_start_hash_is_rejected(bundle, key):
    root, run = bundle
    start = read_json(run / 'start.json')
    del start[key]
    write_json(run / 'start.json', start)

    assert finalize_verification.main(root, run) == 2
    assert read_json(run / 'verification.json')['status'] == 'failed'


@pytest.mark.parametrize('filename', ['start.json', 'verification_samples.json', 'predictions_onnx.json'])
def test_malformed_evidence_replaces_stale_pass(bundle, filename):
    root, run = bundle
    write_json(run / 'verification.json', {'status': 'passed'})
    (run / filename).write_text('{broken', encoding='utf-8')

    assert finalize_verification.main(root, run) == 2
    result = read_json(run / 'verification.json')
    assert result['status'] == 'failed'
    assert result['checks']['complete_validation'] is False
    assert result['errors']


def test_evaluator_exception_replaces_stale_pass(bundle, monkeypatch):
    root, run = bundle
    write_json(run / 'verification.json', {'status': 'passed'})

    def unavailable(*args, **kwargs):
        raise RuntimeError('synthetic evaluator failure')

    monkeypatch.setattr(finalize_verification, 'evaluate_coco', unavailable)
    assert finalize_verification.main(root, run) == 2
    result = read_json(run / 'verification.json')
    assert result['status'] == 'failed'
    assert 'synthetic evaluator failure' in ' '.join(result['errors'])


def test_final_write_failure_cannot_preserve_stale_pass(bundle, monkeypatch):
    root, run = bundle
    report = run / 'verification.json'
    write_json(report, {'status': 'passed'})
    original = Path.write_text

    def deny_report_write(path, *args, **kwargs):
        if path.parent == run and path.name.startswith('verification.json'):
            raise OSError('synthetic report write failure')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'write_text', deny_report_write)
    with pytest.raises(OSError, match='synthetic report write failure'):
        finalize_verification.main(root, run)
    assert not report.exists() or read_json(report)['status'] == 'failed'


@pytest.mark.parametrize('damage', [
    'missing_report', 'missing_tensors', 'null_tensors', 'empty', 'missing_one', 'extra',
    'non_object', 'duplicate_index', 'wrong_index', 'boolean_index', 'missing_shape',
    'empty_shape', 'wrong_shape', 'boolean_dimension', 'zero_dimension', 'float_dimension'])
def test_incomplete_tensor_contract_cannot_pass(bundle, damage):
    root, run = bundle
    rows = read_json(run / 'verification_samples.json')
    report = rows[0]['tensors']
    tensors = report['tensors']
    if damage == 'missing_report':
        del rows[0]['tensors']
    elif damage == 'missing_tensors':
        del report['tensors']
    elif damage == 'null_tensors':
        report['tensors'] = None
    elif damage == 'empty':
        tensors.clear()
    elif damage == 'missing_one':
        tensors.pop()
    elif damage == 'extra':
        tensors.append(dict(tensors[0]))
    elif damage == 'non_object':
        tensors[0] = None
    elif damage == 'duplicate_index':
        tensors[1]['index'] = 0
    elif damage == 'wrong_index':
        tensors[0]['index'] = -1
    elif damage == 'boolean_index':
        tensors[0]['index'] = False
    elif damage == 'missing_shape':
        del tensors[0]['shape']
    else:
        tensors[0]['shape'] = {
            'empty_shape': [], 'wrong_shape': [1, 1, 1, 1],
            'boolean_dimension': [True, 64, 80, 80], 'zero_dimension': [0, 64, 80, 80],
            'float_dimension': [1., 64, 80, 80]}[damage]
    write_json(run / 'verification_samples.json', rows)

    assert finalize_verification.main(root, run) == 2
    assert read_json(run / 'verification.json')['status'] == 'failed'


@pytest.mark.parametrize('metric', ['relative_l2', 'mean_abs', 'max_abs'])
@pytest.mark.parametrize('value', [None, -0.01, float('nan'), float('inf'), float('-inf'), True, '0.01'])
def test_invalid_tensor_metric_cannot_pass(bundle, metric, value):
    root, run = bundle
    rows = read_json(run / 'verification_samples.json')
    rows[0]['tensors']['tensors'][0][metric] = value
    write_json(run / 'verification_samples.json', rows)

    assert finalize_verification.main(root, run) == 2
    assert read_json(run / 'verification.json')['status'] == 'failed'


@pytest.mark.parametrize('finite', [False, None, 1, 'true'])
def test_each_tensor_must_explicitly_be_finite(bundle, finite):
    root, run = bundle
    rows = read_json(run / 'verification_samples.json')
    rows[0]['tensors']['tensors'][4]['finite'] = finite
    write_json(run / 'verification_samples.json', rows)

    assert finalize_verification.main(root, run) == 2
    result = read_json(run / 'verification.json')
    assert result['status'] == 'failed'
    assert result['checks']['all_outputs_finite'] is False


@pytest.mark.parametrize('saved_passed', [False, None, 'true'])
def test_valid_metrics_at_threshold_pass_without_using_saved_labels(bundle, saved_passed):
    root, run = bundle
    rows = read_json(run / 'verification_samples.json')
    rows[0]['tensors']['passed'] = saved_passed
    for tensor in rows[0]['tensors']['tensors']:
        tensor['passed'] = saved_passed
        tensor['relative_l2'] = .02
    write_json(run / 'verification_samples.json', rows)

    assert finalize_verification.main(root, run) == 0
    result = read_json(run / 'verification.json')
    assert result['status'] == 'passed'
    assert result['tensor_count'] == 9
    assert result['max_relative_l2'] == .02


def relock(bundle):
    root, run = bundle
    start = read_json(run / 'start.json')
    for key, name in [('config_sha256', 'config.json'), ('samples_sha256', 'dataset/samples.json')]:
        start[key] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    write_json(run / 'start.json', start)


@pytest.mark.parametrize('damage', [
    'empty', 'duplicate_manifest', 'declared_count', 'boolean_id', 'boolean_row_id',
    'missing_row', 'duplicate_row', 'unknown_row', 'sample_id'])
def test_sample_coverage_must_be_nonempty_and_exact(bundle, damage):
    root, run = bundle
    manifest = read_json(root / 'dataset/samples.json')
    config = read_json(root / 'config.json')
    rows = read_json(run / 'verification_samples.json')
    if damage == 'empty':
        manifest['samples'] = []
        config['sampling']['count'] = 0
        rows.clear()
    elif damage == 'duplicate_manifest':
        manifest['samples'].append(dict(manifest['samples'][0]))
    elif damage == 'declared_count':
        config['sampling']['count'] = 2
    elif damage == 'boolean_id':
        manifest['samples'][0]['id'] = True
    elif damage == 'boolean_row_id':
        rows[0]['image_id'] = True
    elif damage == 'missing_row':
        rows.clear()
    elif damage == 'duplicate_row':
        rows.append(dict(rows[0]))
    elif damage == 'unknown_row':
        rows[0]['image_id'] = 999
    else:
        rows[0]['sample_id'] = 'other-sample'
    write_json(root / 'config.json', config)
    write_json(root / 'dataset/samples.json', manifest)
    write_json(run / 'verification_samples.json', rows)
    relock(bundle)

    assert finalize_verification.main(root, run) == 2
    assert read_json(run / 'verification.json')['status'] == 'failed'


@pytest.mark.parametrize('key', ['per_tensor_relative_l2_max',
                                'reference_detections_match_fraction_min', 'subset_AP_drop_absolute_max'])
@pytest.mark.parametrize('value', [True, -0.01, float('nan'), float('inf'), None, '.02'])
def test_invalid_frozen_threshold_cannot_pass(bundle, key, value):
    root, run = bundle
    config = read_json(root / 'config.json')
    config['quality_gate_declared_before_inference'][key] = value
    write_json(root / 'config.json', config)
    relock(bundle)

    assert finalize_verification.main(root, run) == 2
    assert read_json(run / 'verification.json')['status'] == 'failed'


@pytest.mark.parametrize('counts', [(-1, -1, -1), (0, 0, 1), (1, 1, 2), (1, -1, 1),
                                   (1, 1, True), (1, 1, 1.0), (1, 1, float('inf'))])
def test_invalid_detection_counts_cannot_pass(bundle, counts):
    root, run = bundle
    rows = read_json(run / 'verification_samples.json')
    rows[0]['detection_matching'] = dict(zip(('reference_count', 'actual_count', 'matched'), counts))
    write_json(run / 'verification_samples.json', rows)

    assert finalize_verification.main(root, run) == 2
    assert read_json(run / 'verification.json')['status'] == 'failed'


@pytest.mark.parametrize('ap', [-1., 1.1, True, None, float('nan'), float('inf')])
def test_invalid_evaluated_ap_cannot_pass(bundle, monkeypatch, ap):
    root, run = bundle
    # Inject an invalid external evaluator result; other tests use real synthetic COCO.
    monkeypatch.setattr(finalize_verification, 'evaluate_coco', lambda *args: {'AP': ap})
    assert finalize_verification.main(root, run) == 2
    assert read_json(run / 'verification.json')['status'] == 'failed'
