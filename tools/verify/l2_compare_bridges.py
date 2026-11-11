#!/usr/bin/env python3
"""L2 on the RK3588 board: old bridge vs bridge v2 on the frozen 40 development inputs.

Inputs are the frozen letterboxed RGB files (dataset/inputs/*.u8 of the first round,
sha256 listed in dataset/samples.json). For every configuration the script records
per-image timings and compares:
  * canonical outputs: v2 (zero-copy fp16 -> float32) against v1 (want_float) - expected
    bit-identical for the same .rknn; for a logits6 model the six outputs are adapted to
    canonical9 with model_adapter.adapt_outputs and compared with the ONNX reference
    gates of prepare_model.compare_canonical when --onnx is given;
  * detections: NEON post-processing (bridge v2) against yolo_common.postprocess on the
    same float32 outputs (class/score exact, boxes within --box-atol pixels);
  * letterbox: bridge v2 letterbox of the original JPEG against the frozen .u8 bytes
    (sha256) when --images points at dataset/images.
Writes summary.json / validation.json / per_image.jsonl into --out. Nothing here is a
performance acceptance by itself; the e2e numbers feed benchmarks/history/<run-id>/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'benchmarks/yolov8n_general/scripts'
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))
import model_adapter  # noqa: E402
import yolo_common  # noqa: E402
import verification  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def stats(values):
    values = [float(v) for v in values]
    if not values:
        return None
    values.sort()
    return {'count': len(values), 'mean': statistics.fmean(values), 'median': values[len(values) // 2],
            'p95': values[min(len(values) - 1, int(round((len(values) - 1) * 0.95)))], 'min': values[0], 'max': values[-1]}


def parse_config(text):
    """name:io_mode,input_mode,output_mode,core_mask[,mem_flags]"""
    name, rest = text.split(':', 1)
    parts = [int(p) for p in rest.split(',')]
    while len(parts) < 5:
        parts.append(0 if len(parts) == 4 else 7 if len(parts) == 3 else 0)
    return {'name': name, 'io_mode': parts[0], 'input_mode': parts[1], 'output_mode': parts[2], 'core_mask': parts[3], 'mem_flags': parts[4]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--samples', required=True, help='frozen dataset/samples.json of the first round')
    parser.add_argument('--inputs-root', required=True, help='root that contains the sample input/image paths')
    parser.add_argument('--model', required=True, help='.rknn to run with bridge v2 (9 or 6 outputs)')
    parser.add_argument('--library-v2', required=True, help='librknn_bridge_v2.so')
    parser.add_argument('--reference-model', help='.rknn for the v1 bridge (default: --model)')
    parser.add_argument('--library-v1', help='librknn_bridge.so of the first round (enables the v1 comparison)')
    parser.add_argument('--onnx', help='frozen ONNX for the numeric gates (needs onnxruntime)')
    parser.add_argument('--config', action='append', default=[],
                        help='name:io_mode,input_mode,output_mode,core_mask[,mem_flags]; repeatable')
    parser.add_argument('--repeats', type=int, default=3, help='timed repeats per image per configuration')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--nms-iou', type=float, default=0.45)
    parser.add_argument('--box-atol', type=float, default=2e-3)
    parser.add_argument('--limit', type=int, default=0, help='use only the first N samples (debugging)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(args.samples).read_text(encoding='utf-8'))
    samples = manifest['samples'][:args.limit or None]
    root = Path(args.inputs_root)
    configs = [parse_config(c) for c in args.config] or [parse_config('zerocopy_native:1,0,0,7'), parse_config('legacy:0,0,0,7')]
    bridge_v2 = load_module('rknn_bridge_v2', ROOT / 'src/apps/rknn_bridge_v2.py')

    # frozen inputs
    frozen = []
    for s in samples:
        data = (root / s['input']).read_bytes()
        if digest_bytes(data) != s['input_sha256']:
            raise ValueError(f"frozen input hash mismatch for {s['id']}")
        frozen.append({'id': s['id'], 'rgb': np.frombuffer(data, np.uint8).reshape(640, 640, 3), 'transform': s['transform'],
                       'image': root / s['image'] if 'image' in s else None})

    summary = {'schema_version': 1, 'record_type': 'l2_bridge_comparison', 'model': str(args.model),
               'model_sha256': hashlib.sha256(Path(args.model).read_bytes()).hexdigest(), 'samples': len(frozen),
               'repeats': args.repeats, 'warmup': args.warmup, 'platform': platform.platform(),
               'configurations': [], 'reference_v1': None, 'scope': 'development inputs only; not an acceptance result'}
    validation = {'schema_version': 1, 'record_type': 'l2_validation', 'checks': []}
    per_image = (out / 'per_image.jsonl').open('w', encoding='utf-8')

    # ---------------- v1 reference outputs (float32, want_float) ----------------
    v1_outputs = {}
    if args.library_v1:
        v1 = load_module('rknn_bridge_v1', SCRIPTS / 'rknn_bridge.py')
        model_v1 = args.reference_model or args.model
        timings = []
        with v1.RknnSession(model_v1, args.library_v1, detail=False, core_mask=7) as session:
            for _ in range(args.warmup):
                session.run(frozen[0]['rgb'])
            for f in frozen:
                outputs, metrics = session.run(f['rgb'])
                v1_outputs[f['id']] = outputs
                timings.append(metrics)
                dets = yolo_common.postprocess(outputs, f['transform'], args.confidence, args.nms_iou)
                per_image.write(json.dumps({'config': 'v1_reference', 'image_id': f['id'], 'api': metrics,
                                            'detections': int(len(dets))}) + '\n')
        summary['reference_v1'] = {'model': str(model_v1), 'library': str(args.library_v1),
                                   'inputs_set_ms': stats([t['inputs_set_ms'] for t in timings]),
                                   'run_ms': stats([t['run_ms'] for t in timings]),
                                   'outputs_get_ms': stats([t['outputs_get_ms'] for t in timings]),
                                   'wall_ms': stats([t['wall_ms'] for t in timings])}

    # ---------------- ONNX reference (optional numeric gates) ----------------
    reference_outputs = {}
    if args.onnx:
        import onnxruntime as ort
        prepare_model = load_module('prepare_model', SCRIPTS / 'prepare_model.py')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        ref = ort.InferenceSession(args.onnx, options, providers=['CPUExecutionProvider'])
        name = ref.get_inputs()[0].name
        for f in frozen:
            tensor = np.ascontiguousarray(f['rgb'].transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255.)
            reference_outputs[f['id']] = ref.run(None, {name: tensor})

    # ---------------- bridge v2 configurations ----------------
    for cfg in configs:
        rows = []
        checks = {'config': cfg['name'], 'outputs_bit_identical_to_v1': 0, 'outputs_compared_to_v1': 0,
                  'max_abs_vs_v1': 0.0, 'detections_match_reference_post': 0, 'detections_compared': 0,
                  'max_box_abs_px': 0.0, 'numeric_gate_passed': None, 'numeric_gate_failures': [],
                  'nonfinite_outputs': 0, 'letterbox_hash_matches': 0, 'letterbox_checked': 0}
        with bridge_v2.RknnSessionV2(args.model, args.library_v2, detail=False, core_mask=cfg['core_mask'],
                                     io_mode=cfg['io_mode'], input_mode=cfg['input_mode'],
                                     output_mode=cfg['output_mode'], mem_flags=cfg['mem_flags']) as session:
            meta = session.metadata
            for _ in range(args.warmup):
                session.prepare_rgb(frozen[0]['rgb']); session.run_prepared(); session.release()
            for f in frozen:
                # correctness pass (once per image)
                session.prepare_rgb(f['rgb'])
                t_run = session.run_prepared()
                dets_v2, post_stats, t_post = session.postprocess(f['transform'], args.confidence, args.nms_iou)
                outputs = session.outputs_f32()
                session.release()
                canonical = model_adapter.adapt_outputs(outputs) if session.logits6 else outputs
                finite = all(np.isfinite(a).all() for a in outputs)
                checks['nonfinite_outputs'] += int(not finite)
                dets_ref = yolo_common.postprocess(canonical, f['transform'], args.confidence, args.nms_iou)
                same = (dets_ref.shape == dets_v2.shape and (len(dets_ref) == 0 or (
                    np.array_equal(dets_ref[:, 5], dets_v2[:, 5]) and np.allclose(dets_ref[:, 4], dets_v2[:, 4], atol=2e-7, rtol=0)
                    and np.allclose(dets_ref[:, :4], dets_v2[:, :4], atol=args.box_atol, rtol=0))))
                box_abs = float(np.max(np.abs(dets_ref[:, :4] - dets_v2[:, :4]))) if dets_ref.shape == dets_v2.shape and len(dets_ref) else 0.0
                checks['detections_compared'] += 1
                checks['detections_match_reference_post'] += int(same)
                checks['max_box_abs_px'] = max(checks['max_box_abs_px'], box_abs)
                row = {'config': cfg['name'], 'image_id': f['id'], 'detections_v2': int(len(dets_v2)),
                       'detections_reference_post': int(len(dets_ref)), 'post_match': bool(same), 'post_stats': post_stats,
                       'run_once': t_run, 'post_once': t_post}
                if f['id'] in v1_outputs and not session.logits6:
                    diffs = [float(np.max(np.abs(a - b))) for a, b in zip(v1_outputs[f['id']], outputs)]
                    row['max_abs_vs_v1'] = max(diffs)
                    checks['outputs_compared_to_v1'] += 1
                    checks['outputs_bit_identical_to_v1'] += int(all(np.array_equal(a, b) for a, b in zip(v1_outputs[f['id']], outputs)))
                    checks['max_abs_vs_v1'] = max(checks['max_abs_vs_v1'], max(diffs))
                if f['id'] in reference_outputs:
                    gate = prepare_model.compare_canonical(reference_outputs[f['id']], canonical)
                    row['numeric_gate'] = {'passed': gate['passed'], 'strict_2pct_passed': gate['strict_2pct_passed'],
                                           'tensors': [{k: t[k] for k in ('index', 'role', 'relative_l2', 'rmse', 'mae', 'max_abs', 'passed')} for t in gate['tensors']]}
                    checks['numeric_gate_passed'] = (checks['numeric_gate_passed'] is not False) and gate['passed']
                    if not gate['passed']:
                        checks['numeric_gate_failures'].append(f['id'])
                    matching = verification.match_detections(yolo_common.postprocess(reference_outputs[f['id']], f['transform'], args.confidence, args.nms_iou), dets_v2, 0.9)
                    row['detection_matching'] = {k: matching[k] for k in ('reference_count', 'actual_count', 'matched', 'reference_match_fraction', 'actual_match_fraction')}
                if f['image'] is not None and f['image'].is_file():
                    import cv2
                    bgr = cv2.imdecode(np.frombuffer(f['image'].read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                    transform, t_prep = session.prepare_bgr(bgr)
                    rgb = session.letterboxed_rgb()
                    checks['letterbox_checked'] += 1
                    checks['letterbox_hash_matches'] += int(digest_bytes(rgb.tobytes()) == digest_bytes(f['rgb'].tobytes()))
                    row['letterbox'] = {'transform_equal': transform == f['transform'], 'prepare': t_prep}
                # timed repeats: prepare (pack) + run + post, no export
                timed = []
                for _ in range(args.repeats):
                    t0 = time.perf_counter_ns()
                    t_prep = session.prepare_rgb(f['rgb'])
                    t_run = session.run_prepared()
                    _, _, t_post = session.postprocess(f['transform'], args.confidence, args.nms_iou)
                    t_rel = session.release()
                    timed.append({'pack_ms': t_prep['pack_ms'], 'inputs_set_ms': t_run['inputs_set_ms'], 'run_ms': t_run['run_ms'],
                                  'outputs_get_ms': t_run['outputs_get_ms'], 'perf_run_us': t_run['perf_run_us'],
                                  'post_ms': t_post['post_ms'], 'outputs_release_ms': t_rel['outputs_release_ms'],
                                  'wall_ms': (time.perf_counter_ns() - t0) / 1e6})
                row['timed'] = timed
                rows.append(row)
                per_image.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(f"{cfg['name']} image={f['id']} run_ms={timed[-1]['run_ms']:.2f} post_ms={timed[-1]['post_ms']:.3f} wall_ms={timed[-1]['wall_ms']:.2f} post_match={same}", flush=True)
        pooled = [t for r in rows for t in r['timed']]
        summary['configurations'].append({**cfg, 'metadata': meta, 'pooled': {k: stats([t[k] for t in pooled]) for k in pooled[0]}})
        validation['checks'].append(checks)
    per_image.close()
    summary['passed'] = all(c['detections_match_reference_post'] == c['detections_compared'] and c['nonfinite_outputs'] == 0
                            and (c['numeric_gate_passed'] in (None, True)) and c['letterbox_hash_matches'] == c['letterbox_checked']
                            and (c['outputs_compared_to_v1'] == 0 or c['outputs_bit_identical_to_v1'] == c['outputs_compared_to_v1'])
                            for c in validation['checks'])
    validation['passed'] = summary['passed']
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (out / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    sys.exit(0 if summary['passed'] else 1)


if __name__ == '__main__':
    main()
