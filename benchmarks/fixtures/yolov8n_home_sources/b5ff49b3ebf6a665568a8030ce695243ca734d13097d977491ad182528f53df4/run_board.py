"""在 RK3588 上按独立模式验证输出、采集正常计时或详细报告。"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import cv2
import numpy as np
from yolo_common import letterbox, postprocess
from verification import compare_outputs, match_detections, coco_predictions


def measurement_schedule(samples, repeats, seed):
    rng = random.Random(seed)
    schedule = []
    for repeat in range(repeats):
        order = list(samples)
        rng.shuffle(order)
        schedule.extend((repeat, sample) for sample in order)
    return schedule


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def hardware_snapshot():
    import resource
    values = {}
    patterns = ['/sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq',
                '/sys/devices/system/cpu/cpufreq/policy*/scaling_governor',
                '/sys/class/devfreq/*/cur_freq', '/sys/class/devfreq/*/governor',
                '/sys/class/thermal/thermal_zone*/temp', '/sys/class/thermal/thermal_zone*/type',
                '/proc/loadavg', '/proc/meminfo']
    import glob
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                values[path] = Path(path).read_text().strip()
            except OSError:
                values[path] = None
    return {'wall_clock_utc': datetime.now(timezone.utc).isoformat(),
            'monotonic_ns': time.perf_counter_ns(), 'sysfs': values,
            'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'process_max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'power_watts': None, 'energy_joules': None, 'frequency_locked': False}


def check_bundle(root, config, manifest):
    assert len(manifest['samples']) == config['sampling']['count']
    assert len({s['id'] for s in manifest['samples']}) == len(manifest['samples'])
    for kind, filename in [('onnx', 'yolov8n.onnx'), ('rknn', 'yolov8n_fp16.rknn')]:
        assert digest(root / 'models' / filename) == config[kind + '_sha256'], f'{kind} 模型哈希不一致'
    assert digest(root / manifest['subset_annotation']) == manifest['subset_annotation_sha256']
    for sample in manifest['samples']:
        assert digest(root / sample['image']) == sample['image_sha256'], '原图哈希不一致'
        assert digest(root / sample['input']) == sample['input_sha256'], '冻结输入哈希不一致'
        image = cv2.imdecode(np.frombuffer((root / sample['image']).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        rgb, transform = letterbox(image)
        assert transform == sample['transform'], '跨平台预处理参数不一致'
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == sample['input_sha256'], '跨平台预处理字节不一致'


def run(root, mode, out, library):
    from rknn_bridge import RknnSession
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    manifest = json.loads((root / 'dataset/samples.json').read_text(encoding='utf-8'))
    cv2.setNumThreads(1)
    check_bundle(root, config, manifest)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'outputs').mkdir()
    (out / 'perf').mkdir()
    samples = manifest['samples']
    category_ids = [c['id'] for c in sorted(manifest['categories'], key=lambda c: c['id'])]
    before = hardware_snapshot()
    environment = {'system': platform.platform(), 'python': platform.python_version(),
                   'numpy': np.__version__, 'opencv': cv2.__version__, 'opencv_threads': cv2.getNumThreads(),
                   'packages': {name: importlib.metadata.version(name) for name in ('rknn-toolkit2', 'onnxruntime')},
                   'background_processes': subprocess.check_output(['ps', '-eo', 'comm,pcpu,pmem', '--sort=-pcpu'], text=True).splitlines()[:20],
                   'thread_environment': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}}
    write_json(out / 'start.json', {'run_id': config['run_id'], 'mode': mode, 'config_sha256': digest(root / 'config.json'),
                                   'samples_sha256': digest(root / 'dataset/samples.json'), 'environment': environment,
                                   'resources': before,
                                   'source_sha256': {p.name: digest(p) for p in sorted((root / 'scripts').glob('*.py'))}})
    t0 = time.perf_counter_ns()
    session = RknnSession(str(root / 'models/yolov8n_fp16.rknn'), str(library), detail=(mode == 'detail'), core_mask=7)
    init_wall_ms = (time.perf_counter_ns() - t0) / 1e6
    write_json(out / 'runtime.json', {'init_wall_ms': init_wall_ms, 'metadata': session.metadata})
    reference_session = None
    if mode == 'verify':
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        reference_session = ort.InferenceSession(str(root / 'models/yolov8n.onnx'), options, providers=['CPUExecutionProvider'])
        write_json(out / 'reference_runtime.json', {'backend': 'ONNX Runtime CPUExecutionProvider', 'version': ort.__version__,
                  'inputs': [{'name': p.name, 'shape': p.shape, 'type': p.type} for p in reference_session.get_inputs()],
                  'outputs': [{'name': p.name, 'shape': p.shape, 'type': p.type} for p in reference_session.get_outputs()]})
    warm = np.fromfile(root / samples[0]['input'], dtype=np.uint8).reshape(640, 640, 3)
    warmups = []
    failures = []
    verify_rows, ort_predictions, rknn_predictions = [], [], []
    try:
        for index in range(3):
            begin = time.perf_counter_ns()
            _, metrics = session.run(warm, perf_path=str(out / 'perf' / f'warmup_{index}.perf.txt') if mode == 'detail' else None)
            warmups.append({'index': index, 'wall_ms': (time.perf_counter_ns() - begin) / 1e6, 'api': metrics})
        write_json(out / 'warmup.json', warmups)
        repeats = config['baseline']['repeats_per_sample'] if mode == 'baseline' else 1
        schedule = measurement_schedule(samples, repeats, config['baseline']['sample_order_seed'])
        current_round = None
        with (out / 'measurements.jsonl').open('w', encoding='utf-8') as stream, (out / 'resources.jsonl').open('w', encoding='utf-8') as resources:
            for repeat, sample in schedule:
                if repeat != current_round:
                    resources.write(json.dumps({'round': repeat, **hardware_snapshot()}, ensure_ascii=False) + '\n')
                    resources.flush()
                    current_round = repeat
                record = {'sample_id': sample['sample_id'], 'image_id': sample['id'], 'repeat': repeat, 'mode': mode}
                try:
                    started = time.perf_counter_ns()
                    bgr = cv2.imdecode(np.frombuffer((root / sample['image']).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                    rgb, transform = letterbox(bgr)
                    prepared = time.perf_counter_ns()
                    perf_path = out / 'perf' / f'{sample["id"]:012d}.perf.txt' if mode == 'detail' else None
                    outputs, api_metrics = session.run(rgb, perf_path=str(perf_path) if perf_path else None)
                    inferred = time.perf_counter_ns()
                    detections = postprocess(outputs, transform, config['baseline']['confidence'], config['baseline']['nms_iou'])
                    finished = time.perf_counter_ns()
                    record.update({'status': 'success', 'clock_domain': 'host_perf_counter_ns',
                                   'start_ns': started, 'preprocessed_ns': prepared, 'inference_complete_ns': inferred, 'end_ns': finished,
                                   'preprocess_ms': (prepared - started) / 1e6, 'inference_wall_ms': (inferred - prepared) / 1e6,
                                   'postprocess_ms': (finished - inferred) / 1e6, 'e2e_ms': (finished - started) / 1e6,
                                   'api': api_metrics, 'detection_count': len(detections)})
                    if repeat == 0:
                        output_path = out / 'outputs' / f'{sample["id"]:012d}.npz'
                        np.savez_compressed(output_path, **{f'output_{idx}': a for idx, a in enumerate(outputs)})
                        detection_path = out / 'outputs' / f'{sample["id"]:012d}.detections.json'
                        write_json(detection_path, detections.tolist())
                        record.update({'output': output_path.relative_to(root).as_posix(), 'output_sha256': digest(output_path),
                                       'detections': detection_path.relative_to(root).as_posix(), 'detections_sha256': digest(detection_path),
                                       'perf': perf_path.relative_to(root).as_posix() if perf_path else None,
                                       'perf_sha256': digest(perf_path) if perf_path else None})
                    if reference_session is not None:
                        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255.)
                        expected = reference_session.run(None, {reference_session.get_inputs()[0].name: tensor})
                        tensor_check = compare_outputs(expected, outputs, config['quality_gate_declared_before_inference']['per_tensor_relative_l2_max'])
                        reference_dets = postprocess(expected, transform)
                        matching = match_detections(reference_dets, detections, config['quality_gate_declared_before_inference']['detection_matching_iou_min'])
                        verify_rows.append({'sample_id': sample['sample_id'], 'image_id': sample['id'], 'tensors': tensor_check,
                                            'detection_matching': matching, 'reference_detections': reference_dets.tolist(),
                                            'rknn_detections': detections.tolist(),
                                            'reference_output_sha256': [hashlib.sha256(a.tobytes()).hexdigest() for a in expected]})
                        ort_eval = postprocess(expected, transform, config['evaluation']['confidence'], config['evaluation']['nms_iou'])
                        rknn_eval = postprocess(outputs, transform, config['evaluation']['confidence'], config['evaluation']['nms_iou'])
                        ort_predictions.extend(coco_predictions(sample['id'], ort_eval, category_ids))
                        rknn_predictions.extend(coco_predictions(sample['id'], rknn_eval, category_ids))
                    if mode == 'detail':
                        resources.write(json.dumps({'sample_id': sample['sample_id'], **hardware_snapshot()}, ensure_ascii=False) + '\n')
                    print(f'{mode} round={repeat} image={sample["id"]} infer_ms={record["inference_wall_ms"]:.3f} detections={len(detections)}', flush=True)
                except Exception as exc:
                    record.update({'status': 'failed', 'error': str(exc)})
                    failures.append(record)
                    print(f'FAILED {mode} {sample["id"]}: {exc}', flush=True)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
        if reference_session is not None:
            write_json(out / 'verification_samples.json', verify_rows)
            write_json(out / 'predictions_onnx.json', ort_predictions)
            write_json(out / 'predictions_rknn.json', rknn_predictions)
        write_json(out / 'completion.json', {'mode': mode, 'expected_measurements': len(schedule),
                                           'failed': len(failures), 'failures': failures,
                                           'quality_samples': len(verify_rows), 'resources': hardware_snapshot()})
    finally:
        session.close()
    if failures:
        raise RuntimeError(f'{len(failures)} 次执行失败，原始记录已保留')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--mode', choices=['verify', 'baseline', 'detail'], required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--library', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    run(root, args.mode, (root / args.out).resolve(), args.library or root / 'scripts/librknn_bridge.so')
