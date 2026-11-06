"""Prepare the frozen YOLOv8n model and verify only the original 40 development images.

The selected deployment keeps the original ONNX nine-output graph. RKNN config
op_target keys are OUTPUT TENSOR NAMES, not ONNX node names. Final Sigmoid runs
inside RKNN on CPU; that time is part of rknn_run, not application adaptation.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

SOURCE_SHA256 = '0c8716701f471067932b797eeb67c8e5db47c693c2557c881d7679ec12e21bc5'
DEVELOPMENT_MANIFEST_SHA256 = '5049d85d8c2feaa243b54585cc081eb6e3822ce197235833c78aa11381ab9bb8'
PROTOCOL_SHA256 = '51aa18692e188a958c7a6d8bf98c1839e5cb73ade8a445061acd2ac47d2b9943'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_final_heads(source):
    """Trace actual final-head producers and validate sum/clip math."""
    import onnx
    graph = onnx.load(str(source)).graph
    if len(graph.output) != 9:
        raise ValueError('source must have 9 canonical outputs')
    producers = {o: n for n in graph.node for o in n.output}
    initializers = {n.name: n for n in graph.initializer}

    def attributes(node):
        return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

    def scalar(name):
        value = initializers.get(name)
        if value is None:
            node = producers[name]
            if node.op_type != 'Constant':
                raise ValueError('clip bound must be constant')
            value = attributes(node)['value']
        return float(onnx.numpy_helper.to_array(value).item())

    heads = []
    for scale, side in enumerate((80, 40, 20)):
        outputs = graph.output[3 * scale:3 * scale + 3]
        for value, channels in zip(outputs, (64, 80, 1)):
            shape = [d.dim_value for d in value.type.tensor_type.shape.dim]
            if shape != [1, channels, side, side]:
                raise ValueError(f'unsupported output {value.name}: {shape}')
        reg, probability, clip_sum = [o.name for o in outputs]
        sigmoid = producers[probability]
        if sigmoid.op_type != 'Sigmoid' or len(sigmoid.input) != 1:
            raise ValueError('classification output must be produced by Sigmoid')
        clip = producers[clip_sum]
        if clip.op_type != 'Clip' or len(clip.input) != 3:
            raise ValueError('auxiliary output must be produced by Clip')
        reduction = producers[clip.input[0]]
        attr = attributes(reduction)
        bounds = [scalar(clip.input[1]), scalar(clip.input[2])]
        if (reduction.op_type != 'ReduceSum' or list(reduction.input) != [probability]
                or attr.get('axes') != [1] or attr.get('keepdims', 1) != 1 or bounds != [0, 1]):
            raise ValueError('expected clip(sum(sigmoid(logits), axis=1, keepdims=True), 0, 1)')
        heads.append({'scale': scale, 'side': side, 'regression': reg,
                      'probability': probability, 'clip_sum': clip_sum,
                      'node_name': sigmoid.name, 'logits': sigmoid.input[0],
                      'sum_axes': attr['axes'], 'clip_bounds': bounds})
    return heads


def compare_canonical(reference, actual):
    """Predeclared per-tensor gates plus unchanged historical strict 2% L2."""
    if len(reference) != 9 or len(actual) != 9:
        raise ValueError('numeric comparison requires nine canonical outputs')
    rows = []
    for index, (ref, out) in enumerate(zip(reference, actual)):
        side = 80 >> (index // 3)
        shape = (1, (64, 80, 1)[index % 3], side, side)
        if ref.shape != shape or out.shape != shape:
            raise ValueError(f'output {index} shape mismatch')
        finite = bool(np.isfinite(ref).all() and np.isfinite(out).all())
        r, a = ref.astype(np.float64), out.astype(np.float64)
        if finite:
            delta = a - r
            square_sum = float(np.sum(delta * delta))
            relative = np.sqrt(square_sum) / max(float(np.sqrt(np.sum(r * r))), 1e-12)
            rmse = np.sqrt(square_sum / delta.size)
            mae = float(np.mean(np.abs(delta)))
            maximum = float(np.max(np.abs(delta)))
        else:
            relative = rmse = mae = maximum = None
        in_range = bool(finite and (index % 3 == 0 or (np.all(a >= 0) and np.all(a <= 1))))
        threshold_met = finite and (relative <= .02 if index % 3 == 0 else
                                    rmse <= .001 if index % 3 == 1 else mae <= .005)
        rows.append({'index': index, 'role': ('regression','probability','clip_sum')[index % 3],
                     'shape': list(shape), 'finite': finite, 'actual_in_range': in_range,
                     'relative_l2': relative, 'rmse': rmse, 'mae': mae, 'max_abs': maximum,
                     'reference_min': float(r.min()) if finite else None,
                     'reference_max': float(r.max()) if finite else None,
                     'actual_min': float(a.min()) if finite else None,
                     'actual_max': float(a.max()) if finite else None,
                     'passed': bool(threshold_met and in_range),
                     'strict_2pct_passed': bool(finite and relative <= .02)})
    return {'passed': all(r['passed'] for r in rows),
            'strict_2pct_passed': all(r['strict_2pct_passed'] for r in rows), 'tensors': rows}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def convert(root, output, record_path):
    """Real toolkit2 conversion; refuse to replace an existing compiled model."""
    from datetime import datetime, timezone
    import importlib.metadata
    from rknn.api import RKNN
    source = root / 'models/yolov8n.onnx'
    if digest(source) != SOURCE_SHA256:
        raise ValueError('source ONNX hash differs from frozen original')
    heads = inspect_final_heads(source)
    if output.exists() or record_path.exists():
        raise FileExistsError('use new output/record paths to preserve existing evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {'target_platform': 'rk3588', 'mean_values': [[0, 0, 0]],
              'std_values': [[255, 255, 255]],
              'op_target': {h['probability']: 'cpu' for h in heads}}
    record = {'recorded_utc': datetime.now(timezone.utc).isoformat(),
              'toolkit2': importlib.metadata.version('rknn-toolkit2'),
              'source_sha256': SOURCE_SHA256, 'source_graph_unchanged': True,
              'source_path': str(source), 'output_path': str(output),
              'script_sha256': digest(__file__), 'config': config,
              'do_quantization': False, 'calibration_dataset_used': False,
              'output_contract': 'canonical9', 'final_heads': heads}
    runtime = RKNN(verbose=True)
    try:
        for stage, action in (
            ('config', lambda: runtime.config(**config)),
            ('load_onnx', lambda: runtime.load_onnx(model=str(source))),
            ('build', lambda: runtime.build(do_quantization=False)),
            ('export_rknn', lambda: runtime.export_rknn(str(output))),
        ):
            record['stage'] = stage
            code = action()
            record.setdefault('return_codes', {})[stage] = code
            if code != 0:
                raise RuntimeError(f'{stage} returned {code}')
        record.update(status='compiled', rknn_sha256=digest(output), bytes=output.stat().st_size)
    except BaseException as exc:
        record.update(status='failed', error=str(exc))
        raise
    finally:
        runtime.release()
        write_json(record_path, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return record


def development_probe(root, development_root, out, model, library):
    """Run exactly the hash-frozen old40; no independent acceptance data access."""
    from datetime import datetime, timezone
    import importlib.metadata
    import os
    import time
    started = time.perf_counter_ns()
    manifest_path = development_root / 'dataset/samples.json'
    if digest(manifest_path) != DEVELOPMENT_MANIFEST_SHA256:
        raise ValueError('this command only accepts the frozen original 40-image manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    samples = manifest['samples']
    if len(samples) != 40 or len({s['id'] for s in samples}) != 40:
        raise ValueError('expected exactly 40 unique development images')
    if digest(root / 'models/yolov8n.onnx') != SOURCE_SHA256:
        raise ValueError('original ONNX reference hash mismatch')
    protocol_path = root / 'quality_protocol.json'
    if not protocol_path.exists():
        protocol_path = root / 'evidence/model/quality_protocol.json'
    if digest(protocol_path) != PROTOCOL_SHA256:
        raise ValueError('quality protocol hash mismatch')
    if out.exists():
        raise FileExistsError('choose a new evidence path; prior runs are immutable')
    out.mkdir(parents=True)
    (out / 'outputs').mkdir()
    import sys
    # Isolated probe may use frozen dependency snapshots before main deployment.
    sys.path.append(str(root / 'evidence/model/dependencies'))
    import cv2
    import onnxruntime as ort
    import yolo_common
    import verification
    from model_adapter import adapt_outputs
    from rknn_bridge import RknnSession
    from yolo_common import letterbox, postprocess
    from verification import match_detections, coco_predictions
    cv2.setNumThreads(1)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    reference = ort.InferenceSession(str(root / 'models/yolov8n.onnx'), options,
                                    providers=['CPUExecutionProvider'])
    runtime = RknnSession(model, library, detail=False, core_mask=7)
    environment = {'recorded_utc': datetime.now(timezone.utc).isoformat(),
                   'manifest_sha256': digest(manifest_path), 'protocol_sha256': PROTOCOL_SHA256,
                   'onnx_sha256': SOURCE_SHA256, 'rknn_sha256': digest(model),
                   'bridge_sha256': digest(library),
                   'runtime_library_sha256': digest('/lib/librknnrt.so'),
                   'versions': {n: importlib.metadata.version(n) for n in
                                ('rknn-toolkit2', 'rknn-toolkit-lite2', 'onnxruntime', 'numpy')},
                   'opencv_threads': cv2.getNumThreads(), 'ort_intra_threads': 1, 'ort_inter_threads': 1,
                   'thread_environment': {n: os.environ.get(n) for n in
                                          ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                   'script_sha256': {**{n: digest(root / 'scripts' / n) for n in
                                         ('prepare_model.py', 'model_adapter.py', 'rknn_bridge.py')},
                                     'yolo_common.py': digest(yolo_common.__file__),
                                     'verification.py': digest(verification.__file__)},
                   'runtime_metadata': runtime.metadata, 'core_mask': 7,
                   'scope': 'development only; no independent acceptance images accessed',
                   'cpu_sigmoid_timing_location': 'inside RKNN rknn_run; adapter validation is application postprocess',
                   'reference_probability_note': 'ORT may return sub-ulp negative sigmoid values near zero; reference bytes remain untouched'}
    write_json(out / 'runtime.json', environment)
    rows = []
    reference_predictions, actual_predictions = [], []
    category_ids = [c['id'] for c in sorted(manifest['categories'], key=lambda c: c['id'])]
    crosscheck = None
    warmup = []
    try:
        warm = np.fromfile(development_root / samples[0]['input'], dtype=np.uint8).reshape(640, 640, 3)
        for _ in range(3):
            _, timing = runtime.run(warm)
            warmup.append(timing)
        write_json(out / 'warmup.json', warmup)
        loop_start = time.perf_counter_ns()
        with (out / 'samples.jsonl').open('w', encoding='utf-8') as stream:
            for sample_index, sample in enumerate(samples):
                sample_start = time.perf_counter_ns()
                row = {'image_id': sample['id'], 'sample_id': sample['sample_id'],
                       'input_sha256': sample['input_sha256'], 'status': 'failed'}
                stages = {}
                try:
                    t = time.perf_counter_ns()
                    payload = (development_root / sample['input']).read_bytes()
                    image_bytes = (development_root / sample['image']).read_bytes()
                    if (hashlib.sha256(payload).hexdigest() != sample['input_sha256'] or
                            hashlib.sha256(image_bytes).hexdigest() != sample['image_sha256']):
                        raise ValueError('input/image hash mismatch')
                    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                    rgb, transform = letterbox(image)
                    if transform != sample['transform'] or rgb.tobytes() != payload:
                        raise ValueError('decoded image does not recreate frozen preprocessing')
                    stages['input_validation_ms'] = (time.perf_counter_ns() - t) / 1e6
                    t = time.perf_counter_ns()
                    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)
                    expected = reference.run(None, {reference.get_inputs()[0].name: tensor})
                    stages['reference_inference_ms'] = (time.perf_counter_ns() - t) / 1e6
                    t = time.perf_counter_ns()
                    raw, api = runtime.run(rgb)
                    stages['rknn_inference_ms'] = (time.perf_counter_ns() - t) / 1e6
                    t = time.perf_counter_ns()
                    actual = adapt_outputs(raw)
                    stages['adapter_postprocess_ms'] = (time.perf_counter_ns() - t) / 1e6
                    t = time.perf_counter_ns()
                    numeric = compare_canonical(expected, actual)
                    stages['numeric_statistics_ms'] = (time.perf_counter_ns() - t) / 1e6
                    t = time.perf_counter_ns()
                    ref_det = postprocess(expected, transform, .25, .45, 300)
                    actual_det = postprocess(actual, transform, .25, .45, 300)
                    matching = match_detections(ref_det, actual_det, .90)
                    stages['normal_nms_and_matching_ms'] = (time.perf_counter_ns() - t) / 1e6
                    t = time.perf_counter_ns()
                    ref_eval = postprocess(expected, transform, .001, .65, 300)
                    stages['reference_low_threshold_nms_ms'] = (time.perf_counter_ns() - t) / 1e6
                    t = time.perf_counter_ns()
                    actual_eval = postprocess(actual, transform, .001, .65, 300)
                    stages['rknn_low_threshold_nms_ms'] = (time.perf_counter_ns() - t) / 1e6
                    reference_predictions.extend(coco_predictions(sample['id'], ref_eval, category_ids))
                    actual_predictions.extend(coco_predictions(sample['id'], actual_eval, category_ids))
                    t = time.perf_counter_ns()
                    with np.load(development_root / 'runs/verify_01/outputs' / f'{sample["id"]:012d}.npz') as z:
                        previous = [z[f'output_{i}'] for i in range(9)]
                    old_numeric = compare_canonical(expected, previous)
                    stages['old_output_comparison_ms'] = (time.perf_counter_ns() - t) / 1e6
                    if sample_index == 0:
                        from rknnlite.api import RKNNLite
                        t = time.perf_counter_ns()
                        lite = RKNNLite()
                        try:
                            if lite.load_rknn(str(model)) != 0 or lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
                                raise RuntimeError('Lite2 initialization failed')
                            official = lite.inference(inputs=[rgb[None]], data_format=['nhwc'])
                        finally:
                            lite.release()
                        if official is None or len(official) != 9:
                            raise RuntimeError('Lite2 output count mismatch')
                        equality = [bool(np.array_equal(a, b)) for a, b in zip(raw, official)]
                        crosscheck = {'image_id': sample['id'], 'rknn_sha256': digest(model),
                                      'input_sha256': sample['input_sha256'],
                                      'elementwise_equal': equality, 'all_equal': all(equality),
                                      'max_abs': [float(np.max(np.abs(a - b))) for a, b in zip(raw, official)],
                                      'bridge_output_sha256': [hashlib.sha256(a.tobytes()).hexdigest() for a in raw],
                                      'lite_output_sha256': [hashlib.sha256(a.tobytes()).hexdigest() for a in official]}
                        np.savez_compressed(out / 'lite_bridge_crosscheck.npz',
                                            **{f'bridge_{i}': a for i, a in enumerate(raw)},
                                            **{f'lite_{i}': a for i, a in enumerate(official)},
                                            **{f'reference_{i}': a for i, a in enumerate(expected)})
                        crosscheck['arrays_sha256'] = digest(out / 'lite_bridge_crosscheck.npz')
                        write_json(out / 'lite_bridge_crosscheck.json', crosscheck)
                        stages['lite_crosscheck_and_save_ms'] = (time.perf_counter_ns() - t) / 1e6
                    t = time.perf_counter_ns()
                    saved = out / 'outputs' / f'{sample["id"]:012d}.npz'
                    np.savez_compressed(saved, **{f'output_{i}': a for i, a in enumerate(actual)})
                    row.update(status='success', numeric=numeric, previous_model_numeric=old_numeric,
                               matching=matching, reference_detections=ref_det.tolist(), actual_detections=actual_det.tolist(),
                               reference_eval_count=len(ref_eval), actual_eval_count=len(actual_eval),
                               reference_output_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in expected],
                               actual_output_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in actual],
                               saved_output=saved.relative_to(root).as_posix(), saved_output_sha256=digest(saved), api=api)
                    stages['save_outputs_and_hashes_ms'] = (time.perf_counter_ns() - t) / 1e6
                except Exception as exc:
                    row['error'] = str(exc)
                stages['sample_total_ms'] = (time.perf_counter_ns() - sample_start) / 1e6
                row['timing_ms'] = stages
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                print(json.dumps({'sample': sample_index + 1, 'id': sample['id'], 'status': row['status'],
                                  'numeric_passed': row.get('numeric', {}).get('passed'),
                                  'elapsed_ms': stages['sample_total_ms']}, ensure_ascii=False), flush=True)
        loop_ms = (time.perf_counter_ns() - loop_start) / 1e6
    finally:
        runtime.close()
    # Aggregate the persisted records, retaining failed/zero-detection images.
    rows = [json.loads(line) for line in (out / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    good = [r for r in rows if r['status'] == 'success']
    tensor_rows = [dict(image_id=r['image_id'], **t) for r in good for t in r['numeric']['tensors']]
    ref_count = sum(r['matching']['reference_count'] for r in good)
    actual_count = sum(r['matching']['actual_count'] for r in good)
    matched = sum(r['matching']['matched'] for r in good)
    timing_keys = sorted({k for r in rows for k in r['timing_ms']})
    timing_summary = {k: {'count': sum(k in r['timing_ms'] for r in rows),
                          'total_ms': sum(r['timing_ms'].get(k, 0) for r in rows),
                          'mean_ms': sum(r['timing_ms'].get(k, 0) for r in rows) / sum(k in r['timing_ms'] for r in rows)}
                      for k in timing_keys}
    summary = {'scope': 'development40 only; this is not independent acceptance',
               'rknn_sha256': digest(model), 'onnx_sha256': SOURCE_SHA256,
               'manifest_sha256': digest(manifest_path), 'protocol_sha256': PROTOCOL_SHA256,
               'expected_count': 40, 'sample_count': len(rows), 'success_count': len(good),
               'failed_count': len(rows) - len(good), 'image_ids': [r['image_id'] for r in rows],
               'failed_samples': [r for r in rows if r['status'] != 'success'],
               'finite_all': all(t['finite'] for t in tensor_rows) and len(tensor_rows) == 360,
               'numeric_passed_images': sum(r['numeric']['passed'] for r in good),
               'numeric_passed_tensors': sum(t['passed'] for t in tensor_rows),
               'numeric_failures': [t for t in tensor_rows if not t['passed']],
               'strict_2pct_passed_images': sum(r['numeric']['strict_2pct_passed'] for r in good),
               'strict_2pct_passed_tensors': sum(t['strict_2pct_passed'] for t in tensor_rows),
               'strict_2pct_failures': [t for t in tensor_rows if not t['strict_2pct_passed']],
               'previous_model_numeric_passed_images': sum(r['previous_model_numeric']['passed'] for r in good),
               'previous_model_strict_2pct_passed_images': sum(r['previous_model_numeric']['strict_2pct_passed'] for r in good),
               'per_role': {role: {'tensor_count': sum(t['role'] == role for t in tensor_rows),
                                   'max_relative_l2': max((t['relative_l2'] for t in tensor_rows if t['role'] == role and t['finite']), default=None),
                                   'max_rmse': max((t['rmse'] for t in tensor_rows if t['role'] == role and t['finite']), default=None),
                                   'max_mae': max((t['mae'] for t in tensor_rows if t['role'] == role and t['finite']), default=None)}
                            for role in ('regression','probability','clip_sum')},
               'detection_matching': {'reference_count': ref_count, 'actual_count': actual_count, 'matched': matched,
                                      'reference_fraction': matched/ref_count if ref_count else 1,
                                      'actual_fraction': matched/actual_count if actual_count else (1 if not ref_count else 0)},
               'zero_reference_detection_ids': [r['image_id'] for r in good if not r['reference_detections']],
               'zero_actual_detection_ids': [r['image_id'] for r in good if not r['actual_detections']],
               'zero_reference_eval_ids': [r['image_id'] for r in good if r['reference_eval_count'] == 0],
               'zero_actual_eval_ids': [r['image_id'] for r in good if r['actual_eval_count'] == 0],
               'lite_bridge_crosscheck': crosscheck, 'timing': timing_summary, 'loop_total_ms': loop_ms,
               'total_ms': (time.perf_counter_ns() - started) / 1e6,
               'timing_scope': 'one execution per development image; includes validation, ORT, statistics, normal+low NMS, old-output check and saving; not a benchmark',
               'cpu_timing': 'Sigmoid is inside RKNN run; application adapter validation is postprocess'}
    summary['passed'] = (len(good) == 40 and summary['numeric_passed_images'] == 40
                         and summary['finite_all'] and crosscheck is not None and crosscheck['all_equal']
                         and summary['detection_matching']['reference_fraction'] >= .95
                         and summary['detection_matching']['actual_fraction'] >= .95)
    write_json(out / 'predictions_reference.json', reference_predictions)
    write_json(out / 'predictions_rknn.json', actual_predictions)
    write_json(out / 'development_results.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('convert')
    build.add_argument('--output', type=Path, default=Path('models/yolov8n_fp16.rknn'))
    build.add_argument('--record', type=Path, default=Path('evidence/model/conversion.json'))
    probe = commands.add_parser('develop')
    probe.add_argument('--development-root', type=Path, required=True)
    probe.add_argument('--out', type=Path, default=Path('evidence/model/development40'))
    probe.add_argument('--model', type=Path, default=Path('models/yolov8n_fp16.rknn'))
    probe.add_argument('--library', type=Path, default=Path('scripts/librknn_bridge.so'))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'convert':
        convert(root, root / args.output, root / args.record)
    else:
        result = development_probe(root, args.development_root.resolve(), root / args.out,
                                   root / args.model, root / args.library)
        if not result['passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
