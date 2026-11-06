"""检查统计不会静默丢样本、重复记录或把失败计成零耗时。"""
import importlib.util
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_measurement_summary_preserves_coverage_and_failure():
    assert importlib.util.find_spec('analyze_run') is not None, '尚未实现原始记录统计'
    from analyze_run import validate_measurements, stats
    rows = [
        {'image_id': 11, 'repeat': 0, 'status': 'success'},
        {'image_id': 12, 'repeat': 0, 'status': 'success'},
        {'image_id': 11, 'repeat': 1, 'status': 'success'},
        {'image_id': 12, 'repeat': 1, 'status': 'failed', 'error': '实测失败'},
    ]
    assert validate_measurements(rows, [11, 12], 2) == {'expected': 4, 'success': 3, 'failed': 1}
    with pytest.raises(ValueError):
        validate_measurements(rows[:-1] + [rows[0]], [11, 12], 2)
    with pytest.raises(ValueError):
        validate_measurements(rows[:-1], [11, 12], 2)
    result = stats([1., 2., 3.])
    assert result['count'] == 3 and result['mean'] == 2. and result['median'] == 2.
    assert result['min'] == 1. and result['max'] == 3.
    assert stats([]) is None
    with pytest.raises(ValueError):
        stats([1., float('nan')])


def test_evidence_hash_rejects_changed_file(tmp_path):
    import hashlib
    import analyze_run
    assert hasattr(analyze_run, 'check_artifact'), '尚未实现证据哈希核对'
    p = tmp_path / 'sample.bin'
    p.write_bytes(b'real saved evidence')
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    result = analyze_run.check_artifact(tmp_path, 'sample.bin', expected)
    assert result['sha256'] == expected and result['path'] == 'sample.bin'
    p.write_bytes(b'changed')
    with pytest.raises(ValueError):
        analyze_run.check_artifact(tmp_path, 'sample.bin', expected)
    with pytest.raises(ValueError):
        analyze_run.check_artifact(tmp_path, '../outside.bin', expected)


def test_sample_record_keeps_failed_quality_and_unique_hotspot_ids():
    import analyze_run
    assert hasattr(analyze_run, 'assemble_sample'), '尚未实现样本索引'
    sample = {'sample_id': 'sample_1', 'id': 1, 'image': 'one.jpg', 'image_sha256': 'image_hash',
              'input': 'one.u8', 'input_sha256': 'input_hash', 'input_shape': [640, 640, 3],
              'transform': {'fill': 114}, 'annotation_ids': [12]}
    config = {'model': 'model', 'rknn_sha256': 'model_hash', 'run_id': 'run'}
    graph = {'path': 'graph.json', 'sha256': 'graph_hash'}
    hotspot = {'rank': 1, 'node_ids': [3, 4], 'subgraph_io_available': False, 'replayable': False}
    executions = [{'mode': 'baseline', 'status': 'success'}, {'mode': 'detail', 'status': 'success'}]
    result = analyze_run.assemble_sample(sample, config, graph, executions, [hotspot], 'failed')
    assert result['status'] == 'success' and result['quality']['status'] == 'failed'
    assert result['run']['performance_scope'] == 'diagnostic_only'
    assert result['hotspots'][0]['id'] == 'run:sample_1:h1'
    assert result['hotspots'][0]['replayable'] is False
    assert 'id' not in hotspot
    repeated = analyze_run.assemble_sample(sample, config, graph, executions, [hotspot], 'failed',
                                           quality_evidence='runs/verify_repro/verification.json')
    assert repeated['quality']['evidence'] == 'runs/verify_repro/verification.json'
    executions[1] = {'mode': 'detail', 'status': 'failed', 'error': '报告失败'}
    result = analyze_run.assemble_sample(sample, config, graph, executions, [], 'failed')
    assert result['status'] == 'failed' and result['hotspots'] == []
