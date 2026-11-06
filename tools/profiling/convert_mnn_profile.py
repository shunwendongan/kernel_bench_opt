#!/usr/bin/env python3
"""Preserve OmniProfile events in bounded normalized files for hotspot_pipeline.

Input is the task's MNN callback JSONL (optionally gzip), not an official MNN
profile format. No model is loaded and no inference is executed. Module API and
operator callback durations are separate; neither is a device event duration.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def run(args):
    src = args.profile.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    before = (src.stat().st_size, src.stat().st_mtime_ns)
    identity = {'path': str(src), 'sha256': sha(src), 'size_bytes': before[0]}
    opener = gzip.open if src.suffix == '.gz' else open
    chunks = {}
    files = []
    counts = collections.Counter()
    ranks = {}
    runtime = []
    warnings = []

    def close_chunk(scope):
        entry = chunks.pop(scope)
        entry['file'].write(']}\n')
        entry['file'].close()
        files.append({'path': str(entry['path']), 'scope': scope,
                      'records': entry['count']})

    with opener(src, 'rt', encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append({'source_line': line_no, 'error': str(exc), 'raw': line.rstrip()})
                continue
            kind = raw.get('kind', 'unknown')
            counts[kind] += 1
            if kind == 'runtime':
                runtime.append({'source_line': line_no, 'raw_event': raw})
                continue
            if kind not in {'op', 'module_api'}:
                warnings.append({'source_line': line_no, 'reason': 'unrecognized non-timing event', 'raw_event': raw})
                continue
            try:
                duration = float(raw['elapsed_us'])
                if not math.isfinite(duration) or duration < 0:
                    raise ValueError('non-finite or negative duration')
            except (KeyError, ValueError, TypeError) as exc:
                warnings.append({'source_line': line_no, 'error': str(exc), 'raw_event': raw})
                continue
            stage = str(raw.get('stage', raw.get('module', 'unscoped')))
            executor = str(raw.get('executor', 'module_api'))
            backend_codes = raw.get('output_tensor_backends', [])
            backend = 'tensor_owners=' + json.dumps(backend_codes, separators=(',', ':'))
            name = str(raw.get('name', raw.get('module', '[unnamed]'))) or '[unnamed]'
            op_type = str(raw.get('type', 'ModuleAPI')) or '[unknown]'
            scope = kind + '__' + executor + '__' + stage
            slug = re.sub('[^a-zA-Z0-9_-]', '_', scope) + '_' + hashlib.sha256(scope.encode()).hexdigest()[:6]
            if scope not in chunks:
                part = sum(x['scope'] == scope for x in files)
                path = out / (slug + f'__{part:04d}.profile.json')
                file = path.open('x', encoding='utf-8')
                metadata = {
                    'format': 'csv', 'adapter': 'omni_profile_callback_jsonl_v1',
                    'native_source_sha256': identity['sha256'], 'original_source': identity,
                    'aggregation': 'none', 'synthetic_data': False,
                    'scope': scope, 'input_time_units': ['us'],
                    'measurement': 'host synchronized callback; includes dispatch/wait/instrumentation',
                    'backend_evidence': 'output tensor ownership only; not a device execution trace',
                    'timestamps_available': False, 'is_end_to_end': False,
                    'caller_label': args.label,
                }
                file.write(json.dumps({'schema_version': 1, 'metadata': metadata, 'warnings': []}, ensure_ascii=False)[:-1] + ',"rows":[')
                chunks[scope] = {'file': file, 'path': path, 'count': 0}
            row = {
                'record_id': f'mnn_{identity["sha256"][:12]}_l{line_no:09d}',
                'layer_id': f'l{line_no:09d}', 'layer_name': name, 'op_type': op_type,
                'time_us': duration, 'backend': backend, 'dtype': '',
                'input_shape': json.dumps(raw.get('inputs', []), separators=(',', ':')),
                'output_shape': json.dumps(raw.get('outputs', []), separators=(',', ':')),
                'source_line': line_no, 'source_lines': [line_no], 'sample_count': 1,
                'event_kind': 'mnn_op_callback' if kind == 'op' else 'mnn_module_api',
                'duration_semantics': 'synchronized_host_operator_elapsed' if kind == 'op' else 'inclusive_synchronized_host_module_elapsed',
                'clock_domain': 'host_steady_clock_duration_only',
                'source_name': src.name, 'source_row_ordinal': line_no,
                'stage': stage, 'executor': executor, 'raw_event': raw,
            }
            entry = chunks[scope]
            if entry['count']:
                entry['file'].write(',')
            entry['file'].write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
            entry['count'] += 1
            key = (scope, backend, op_type, name)
            rank = ranks.setdefault(key, {'scope': scope, 'backend': backend, 'op_type': op_type,
                                         'name': name, 'count': 0, 'total_us': 0., 'max_us': 0.})
            rank['count'] += 1
            rank['total_us'] += duration
            rank['max_us'] = max(rank['max_us'], duration)
            if entry['count'] >= args.max_rows:
                close_chunk(scope)
    for scope in list(chunks):
        close_chunk(scope)
    if before != (src.stat().st_size, src.stat().st_mtime_ns) or identity['sha256'] != sha(src):
        raise RuntimeError('Input changed during import; retain this directory as failed and retry a stable snapshot')
    summary = {
        'source': identity, 'input_event_counts': dict(counts), 'files': files,
        'runtime_configuration_events': runtime, 'warnings': warnings,
        'rankings': sorted(ranks.values(), key=lambda r: (r['scope'], -r['total_us'])),
        'notes': [
            'Every valid timing event is retained, including repeated names and shapes.',
            'Rankings sum repeated records within identical scope and tensor ownership; no cross-domain total is defined.',
            'Module API and op callback durations overlap and must never be added.',
            'Source has no timestamps, thread IDs, request IDs or device-event duration; none are invented.',
            'Native profile.json is the lossless toolkit input. Do not reimport an exported CSV with event metadata.',
            'This import does not certify the source was measured; evidence kind must be verified by caller.',
        ],
    }
    (out / 'import_manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (out / 'rankings.csv').open('x', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['scope', 'backend', 'op_type', 'name', 'count', 'total_us', 'max_us'])
        w.writeheader(); w.writerows(summary['rankings'])
    print(json.dumps({'input_events': dict(counts), 'normalized_files': len(files),
                      'timing_records': sum(x['records'] for x in files),
                      'warnings': len(warnings), 'manifest': str(out / 'import_manifest.json')}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('profile', type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--label', default='', help='Caller-provided run label; not an inferred sample ID')
    p.add_argument('--max-rows', type=int, default=10000)
    args = p.parse_args()
    if args.max_rows < 1:
        p.error('--max-rows must be positive')
    run(args)
