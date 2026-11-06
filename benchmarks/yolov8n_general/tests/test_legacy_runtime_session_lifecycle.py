"""Runner lifecycle fault injection only; no board measurements or evidence."""
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


@pytest.fixture
def harness(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    # Isolate same-named helpers from the other benchmark round.
    for name in ('yolo_common', 'verification', 'legacy_runtime'):
        spec = importlib.util.spec_from_file_location(name, scripts / f'{name}.py')
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    runner = module
    state = SimpleNamespace(stage=None, error=None, constructors=0, created=0,
                            closes=0, runs=0, clock_calls=0)

    def fail(stage):
        if state.stage == stage:
            raise state.error

    output = [np.zeros((1,), dtype=np.float32)]

    class SessionStub:
        def __init__(self, model, library, *, detail, core_mask):
            state.constructors += 1
            assert core_mask == 7
            fail('constructor')
            state.created += 1
            self.detail = detail

        @property
        def metadata(self):
            fail('metadata')
            return {'test_only': True}

        def run(self, rgb, perf_path=None):
            assert state.closes == 0
            assert rgb.shape == (640, 640, 3)
            assert self.detail == (perf_path is not None)
            state.runs += 1
            if state.runs == 1:
                fail('warmup')
            if state.runs == 4:
                fail('measurement')
            if perf_path is not None:
                Path(perf_path).write_text('TEST_ONLY_PERF', encoding='utf-8')
            return output, {'test_only': True}

        def close(self):
            state.closes += 1
            fail('close')

    def ort_options():
        fail('ort_options')
        return SimpleNamespace()

    def ort_session(*args, **kwargs):
        fail('ort_init')
        port = SimpleNamespace(name='test_input', shape=[1, 3, 640, 640], type='tensor(float)')
        return SimpleNamespace(get_inputs=lambda: [port], get_outputs=lambda: [port],
                               run=lambda *args: output)

    monkeypatch.setitem(sys.modules, 'rknn_bridge', SimpleNamespace(RknnSession=SessionStub))
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        __version__='TEST_ONLY', SessionOptions=ort_options, InferenceSession=ort_session))
    monkeypatch.setattr(runner, 'hardware_snapshot', lambda: {'test_only': True})
    monkeypatch.setattr(runner, 'check_bundle', lambda *args: None)
    monkeypatch.setattr(runner, 'importlib', SimpleNamespace(
        metadata=SimpleNamespace(version=lambda name: 'TEST_ONLY')))
    monkeypatch.setattr(runner, 'subprocess', SimpleNamespace(check_output=lambda *args, **kwargs: 'TEST_ONLY'))
    monkeypatch.setattr(runner, 'postprocess', lambda *args: np.empty((0, 6), dtype=np.float32))
    write_json = runner.write_json

    def injected_write(path, data):
        fail(path.name)
        write_json(path, data)

    monkeypatch.setattr(runner, 'write_json', injected_write)
    fromfile = np.fromfile

    def injected_read(*args, **kwargs):
        fail('warm_read')
        return fromfile(*args, **kwargs)

    monkeypatch.setattr(np, 'fromfile', injected_read)
    ticks = itertools.count(step=1_000_000)

    def clock():
        state.clock_calls += 1
        if state.clock_calls == 2:
            fail('init_clock')
        return next(ticks)

    monkeypatch.setattr(runner, 'time', SimpleNamespace(perf_counter_ns=clock))
    root = tmp_path / 'test_only_bundle'
    (root / 'dataset').mkdir(parents=True)
    samples = [{'sample_id': f'test_{index}', 'id': index,
                'input': 'warm.rgb', 'image': 'image.png'} for index in (1, 2)]
    config = {'run_id': 'TEST_ONLY', 'baseline': {'repeats_per_sample': 2,
              'sample_order_seed': 7, 'confidence': .25, 'nms_iou': .45},
              'quality_gate_declared_before_inference': {
                  'per_tensor_relative_l2_max': .02, 'detection_matching_iou_min': .9},
              'evaluation': {'confidence': .001, 'nms_iou': .45}}
    write_json(root / 'config.json', config)
    write_json(root / 'dataset/samples.json', {'samples': samples, 'categories': [{'id': 1}]})
    (root / 'warm.rgb').write_bytes(bytes(640 * 640 * 3))
    ok, encoded = cv2.imencode('.png', np.zeros((2, 2, 3), dtype=np.uint8))
    assert ok
    (root / 'image.png').write_bytes(encoded.tobytes())
    out = root / 'test_only_output'
    previous_threads = cv2.getNumThreads()
    yield SimpleNamespace(state=state, out=out,
                          run=lambda mode='baseline': runner.run(root, mode, out, root / 'unused.so'))
    cv2.setNumThreads(previous_threads)


@pytest.mark.parametrize(('stage', 'mode'), [
    ('init_clock', 'baseline'),
    ('metadata', 'baseline'),
    ('runtime.json', 'baseline'),
    ('ort_options', 'verify'),
    ('ort_init', 'verify'),
    ('reference_runtime.json', 'verify'),
    ('warm_read', 'baseline'),
    ('warmup', 'baseline'),
    ('warmup.json', 'baseline'),
    ('completion.json', 'baseline'),
])
def test_post_creation_failure_closes_once(harness, stage, mode):
    state = harness.state
    state.stage, state.error = stage, OSError(f'TEST_ONLY: {stage}')
    with pytest.raises(OSError) as raised:
        harness.run(mode)
    assert raised.value is state.error
    assert state.constructors == state.created == 1
    assert state.closes == 1
    assert not (harness.out / 'completion.json').exists()


def test_interruption_during_runtime_write_closes_once(harness):
    state = harness.state
    state.stage, state.error = 'runtime.json', KeyboardInterrupt('TEST_ONLY interruption')
    with pytest.raises(KeyboardInterrupt) as raised:
        harness.run()
    assert raised.value is state.error
    assert state.created == state.closes == 1


@pytest.mark.parametrize('mode', ['baseline', 'detail', 'verify'])
def test_success_closes_once_after_all_measurements(harness, mode):
    harness.run(mode)
    state = harness.state
    repeats = 2 if mode == 'baseline' else 1
    assert state.constructors == state.created == state.closes == 1
    assert state.runs == 3 + 2 * repeats
    runtime = json.loads((harness.out / 'runtime.json').read_text(encoding='utf-8'))
    assert runtime['init_wall_ms'] == 1.0
    warmups = json.loads((harness.out / 'warmup.json').read_text(encoding='utf-8'))
    assert [row['wall_ms'] for row in warmups] == [1.0] * 3
    records = [json.loads(line) for line in (harness.out / 'measurements.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(records) == 2 * repeats
    assert sorted((row['repeat'], row['image_id']) for row in records) == [
        (repeat, image_id) for repeat in range(repeats) for image_id in (1, 2)]
    for row in records:
        assert row['status'] == 'success'
        assert [row[key] for key in ('preprocess_ms', 'inference_wall_ms', 'postprocess_ms', 'e2e_ms')] == [1.0, 1.0, 1.0, 3.0]
    completion = json.loads((harness.out / 'completion.json').read_text(encoding='utf-8'))
    assert completion['expected_measurements'] == 2 * repeats
    assert completion['failed'] == 0
    assert completion['failures'] == []
    assert completion['quality_samples'] == (2 if mode == 'verify' else 0)


def test_constructor_failure_does_not_close_uncreated_session(harness):
    state = harness.state
    state.stage, state.error = 'constructor', RuntimeError('TEST_ONLY constructor')
    with pytest.raises(RuntimeError) as raised:
        harness.run()
    assert raised.value is state.error
    assert state.constructors == 1
    assert state.created == state.closes == state.runs == 0
    assert not (harness.out / 'runtime.json').exists()


def test_sample_failure_keeps_records_and_continues_before_closing(harness):
    state = harness.state
    state.stage, state.error = 'measurement', OSError('TEST_ONLY measurement')
    with pytest.raises(RuntimeError, match='1 次执行失败，原始记录已保留'):
        harness.run()
    assert state.created == state.closes == 1
    assert state.runs == 7
    records = [json.loads(line) for line in (harness.out / 'measurements.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [row['status'] for row in records] == ['failed', 'success', 'success', 'success']
    assert records[0]['error'] == str(state.error)
    completion = json.loads((harness.out / 'completion.json').read_text(encoding='utf-8'))
    assert completion['expected_measurements'] == 4
    assert completion['failed'] == 1
    assert completion['failures'] == [records[0]]


def test_close_failure_is_not_retried(harness):
    state = harness.state
    state.stage, state.error = 'close', OSError('TEST_ONLY close')
    with pytest.raises(OSError) as raised:
        harness.run()
    assert raised.value is state.error
    assert state.created == state.closes == 1
    assert (harness.out / 'completion.json').is_file()
