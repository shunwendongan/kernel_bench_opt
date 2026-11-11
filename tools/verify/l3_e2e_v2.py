#!/usr/bin/env python3
"""L3 on the RK3588 board: end-to-end timing of the bridge v2 fast path on the frozen 40 images.

Same protocol as the first-round normal run (legacy_runtime.py --mode baseline): model
resident, 3 warm-ups, every image repeated --repeats times in a seed-42 shuffled order,
hardware snapshots per round. Stages per call:
  decode_ms      cv2.imdecode of the JPEG bytes (unchanged from the first round)
  letterbox_ms   pre-letterbox-neon (resize + pad + BGR->RGB)
  pack_ms        uint8 -> model input memory (zero-copy) / 0 in legacy mode
  run_ms         rknn_run (zero-copy) or inputs_set + run + outputs_get (legacy)
  post_ms        post-dfl-nms-neon
  e2e_ms         decode .. detections, host perf_counter_ns
Repeat 0 of every image also exports the float32 outputs and checks the NEON detections
against yolo_common.postprocess (correctness tie-in; not timed).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'benchmarks/yolov8n_general/scripts'
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT))
import model_adapter  # noqa: E402
import yolo_common  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hardware_snapshot():
    values = {}
    for pattern in ('/sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq', '/sys/class/devfreq/*/cur_freq',
                    '/sys/class/thermal/thermal_zone*/temp', '/sys/class/thermal/thermal_zone*/type', '/proc/loadavg'):
        for path in glob.glob(pattern):
            try:
                values[path] = Path(path).read_text().strip()
            except OSError:
                values[path] = None
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    return {'wall_clock_utc': datetime.now(timezone.utc).isoformat(), 'sysfs': values, 'cpu_affinity': affinity,
            'frequency_locked': False}


def stats(values):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    return {'count': len(values), 'mean': statistics.fmean(values), 'median': values[len(values) // 2],
            'p90': values[min(len(values) - 1, int(round((len(values) - 1) * 0.9)))],
            'p95': values[min(len(values) - 1, int(round((len(values) - 1) * 0.95)))], 'min': values[0], 'max': values[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--samples', required=True)
    parser.add_argument('--inputs-root', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--library-v2', required=True)
    parser.add_argument('--io-mode', type=int, default=1)
    parser.add_argument('--input-mode', type=int, default=0)
    parser.add_argument('--output-mode', type=int, default=0)
    parser.add_argument('--core-mask', type=int, default=7)
    parser.add_argument('--mem-flags', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--nms-iou', type=float, default=0.45)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    import cv2
    cv2.setNumThreads(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'outputs').mkdir()
    manifest = json.loads(Path(args.samples).read_text(encoding='utf-8'))
    samples = manifest['samples']
    root = Path(args.inputs_root)
    images = {}
    for s in samples:
        data = (root / s['image']).read_bytes()
        if hashlib.sha256(data).hexdigest() != s['image_sha256']:
            raise ValueError(f"image hash mismatch for {s['id']}")
        images[s['id']] = data
    bridge_v2 = load_module('rknn_bridge_v2', ROOT / 'src/apps/rknn_bridge_v2.py')
    config = {'io_mode': args.io_mode, 'input_mode': args.input_mode, 'output_mode': args.output_mode,
              'core_mask': args.core_mask, 'mem_flags': args.mem_flags}
    start_record = {'schema_version': 1, 'record_type': 'l3_e2e_v2_start', 'model': str(args.model),
                    'model_sha256': hashlib.sha256(Path(args.model).read_bytes()).hexdigest(), 'config': config,
                    'repeats': args.repeats, 'warmup': args.warmup, 'seed': args.seed, 'platform': platform.platform(),
                    'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__,
                    'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                    'resources': hardware_snapshot()}
    (out / 'start.json').write_text(json.dumps(start_record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    rng = random.Random(args.seed)
    schedule = []
    for repeat in range(args.repeats):
        order = list(samples)
        rng.shuffle(order)
        schedule.extend((repeat, s) for s in order)
    t0 = time.perf_counter_ns()
    session = bridge_v2.RknnSessionV2(args.model, args.library_v2, detail=False, **config)
    init_ms = (time.perf_counter_ns() - t0) / 1e6
    (out / 'runtime.json').write_text(json.dumps({'init_wall_ms': init_ms, 'metadata': session.metadata}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    failures, warmups = [], []
    try:
        warm = cv2.imdecode(np.frombuffer(images[samples[0]['id']], np.uint8), cv2.IMREAD_COLOR)
        for i in range(args.warmup):
            _, _, metrics = session.infer_bgr(warm, args.confidence, args.nms_iou)
            warmups.append({'index': i, 'metrics': metrics})
        (out / 'warmup.json').write_text(json.dumps(warmups, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        current = None
        with (out / 'measurements.jsonl').open('w', encoding='utf-8') as stream, (out / 'resources.jsonl').open('w', encoding='utf-8') as resources:
            for repeat, sample in schedule:
                if repeat != current:
                    resources.write(json.dumps({'round': repeat, **hardware_snapshot()}) + '\n'); resources.flush(); current = repeat
                record = {'sample_id': sample['sample_id'], 'image_id': sample['id'], 'repeat': repeat, 'clock_domain': 'host_perf_counter_ns'}
                try:
                    started = time.perf_counter_ns()
                    bgr = cv2.imdecode(np.frombuffer(images[sample['id']], np.uint8), cv2.IMREAD_COLOR)
                    decoded = time.perf_counter_ns()
                    dets, transform, metrics = session.infer_bgr(bgr, args.confidence, args.nms_iou)
                    finished = time.perf_counter_ns()
                    record.update({'status': 'success', 'decode_ms': (decoded - started) / 1e6,
                                   'letterbox_ms': metrics['letterbox_ms'], 'pack_ms': metrics['pack_ms'],
                                   'inputs_set_ms': metrics['inputs_set_ms'], 'run_ms': metrics['run_ms'],
                                   'outputs_get_ms': metrics['outputs_get_ms'], 'perf_run_us': metrics['perf_run_us'],
                                   'post_ms': metrics['post_ms'], 'inference_wall_ms': metrics['wall_ms'],
                                   'e2e_ms': (finished - started) / 1e6, 'detection_count': int(len(dets)),
                                   'post_stats': metrics['post_stats']})
                    if transform != sample['transform']:
                        record['transform_mismatch'] = True
                    if repeat == 0:
                        # correctness tie-in (not timed): export float32 outputs and compare with the Python post-processing
                        session.prepare_bgr(bgr); session.run_prepared()
                        outputs = session.outputs_f32(); session.release()
                        canonical = model_adapter.adapt_outputs(outputs) if session.logits6 else outputs
                        ref = yolo_common.postprocess(canonical, transform, args.confidence, args.nms_iou)
                        same = ref.shape == dets.shape and (len(ref) == 0 or (np.array_equal(ref[:, 5], dets[:, 5])
                               and np.allclose(ref[:, 4], dets[:, 4], atol=2e-7, rtol=0) and np.allclose(ref[:, :4], dets[:, :4], atol=2e-3, rtol=0)))
                        np.savez_compressed(out / 'outputs' / f'{sample["id"]:012d}.npz', **{f'output_{i}': a for i, a in enumerate(outputs)})
                        (out / 'outputs' / f'{sample["id"]:012d}.detections.json').write_text(json.dumps(dets.tolist()) + '\n', encoding='utf-8')
                        record['reference_post_match'] = bool(same)
                    print(f"round={repeat} image={sample['id']} e2e_ms={record['e2e_ms']:.2f} run_ms={record['run_ms']:.2f} dets={len(dets)}", flush=True)
                except Exception as exc:
                    record.update({'status': 'failed', 'error': str(exc)})
                    failures.append(record)
                    print(f"FAILED {sample['id']}: {exc}", flush=True)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n'); stream.flush()
    finally:
        session.close()
    rows = [json.loads(l) for l in (out / 'measurements.jsonl').read_text(encoding='utf-8').splitlines()]
    ok = [r for r in rows if r.get('status') == 'success']
    keys = ('decode_ms', 'letterbox_ms', 'pack_ms', 'inputs_set_ms', 'run_ms', 'outputs_get_ms', 'post_ms', 'inference_wall_ms', 'e2e_ms')
    summary = {'schema_version': 1, 'record_type': 'l3_e2e_v2_summary', 'config': config, 'expected': len(schedule),
               'success': len(ok), 'failed': len(failures), 'pooled_repetitions': {k: stats([r[k] for r in ok]) for k in keys},
               'perf_run_us': stats([r['perf_run_us'] for r in ok]),
               'reference_post_match_count': sum(1 for r in ok if r.get('reference_post_match')),
               'reference_post_checked': sum(1 for r in ok if 'reference_post_match' in r),
               'transform_mismatches': sum(1 for r in ok if r.get('transform_mismatch')),
               'resources_end': hardware_snapshot(), 'frequency_locked': False,
               'scope': 'first-round 40 development images; e2e = decode + letterbox + pack + run + post; diagnostic until the quality acceptance passes'}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('success', 'failed', 'pooled_repetitions', 'reference_post_match_count')}, ensure_ascii=False, indent=2))
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
