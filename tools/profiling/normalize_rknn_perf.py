#!/usr/bin/env python3
"""Read RKNN 2.3.2 PERF_DETAIL with optional blank MacUsage/FullName fields.

Only measured per-layer tables are accepted. Ranking tables and cycle-only
compiler tables are not timings. Raw rows and unparsed lines remain available.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ROW = re.compile(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(CPU|GPU|NPU)\s+(.+?)\s+(\d+/\d+/\d+)\s+([0-9.eE+-]+)\s+(.*?)\s*([0-9.]+%/[0-9.]+%/[0-9.]+%)\s+([0-9.]+)\s*(.*?)\s*$')
PREFIX = re.compile(r'^\s*[DVIWE]\s+RKNN\s*:?\s*(?:\[[^\]]+\]\s*)?')


def normalize(src):
    raw_bytes = src.read_bytes()
    source_sha = hashlib.sha256(raw_bytes).hexdigest()
    rows, warnings, summaries = [], [], []
    active = False
    table = 0
    for line_no, raw in enumerate(raw_bytes.decode('utf-8-sig').splitlines(), 1):
        line = PREFIX.sub('', raw)
        if re.search(r'\bID\s+OpType\b', line):
            active = 'Time(us)' in line and 'WorkLoad(0/1/2)' in line
            if active:
                table += 1
            else:
                warnings.append({'source_line': line_no, 'reason': 'unsupported or cycle-only table', 'raw': raw})
            continue
        if line.strip().startswith('Total Operator Elapsed Per Frame Time(us):'):
            summaries.append({'table_index': table, 'raw': raw, 'source_line': line_no,
                              'reported_total_us': float(line.rsplit(':', 1)[1])})
            active = False
            continue
        if 'Operator Time Consuming Ranking Table' in line:
            active = False
        if not active or not re.match(r'^\s*\d+\s', line):
            continue
        match = ROW.match(line)
        if not match:
            warnings.append({'source_line': line_no, 'reason': 'unrecognized primary table row', 'raw': raw})
            continue
        layer_id, op, dtype, backend, shapes, cycles, duration, macusage, workload, rw_kb, name = match.groups()
        split_shapes = re.split(r'\s{2,}', shapes.strip())
        if len(split_shapes) != 2:
            warnings.append({'source_line': line_no, 'reason': 'ambiguous shape columns', 'raw': raw})
            continue
        duration = float(duration)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError(f'Invalid duration on source line {line_no}')
        rows.append({'record_id': f'rknn_{source_sha[:12]}_l{line_no:08d}',
                     'layer_id': layer_id, 'layer_name': name or f'[unnamed compiler unit {layer_id}]',
                     'original_layer_name': name, 'op_type': op, 'dtype': dtype, 'backend': backend,
                     'time_us': duration, 'input_shape': split_shapes[0], 'output_shape': split_shapes[1],
                     'rw_kb': float(rw_kb), 'cycles_ddr_npu_total': [int(x) for x in cycles.split('/')],
                     'mac_usage_reported': macusage.strip() or None, 'workload_reported': workload,
                     'source_line': line_no, 'source_lines': [line_no], 'sample_count': 1,
                     'table_index': table, 'raw_line': raw, 'event_kind': 'rknn_compiled_unit',
                     'duration_semantics': 'runtime_reported_operator_duration',
                     'clock_domain': 'rknn_perf_detail_duration_only'})
    if not rows:
        raise ValueError('No supported measured RKNN per-layer rows; inspect source format')
    for summary in summaries:
        summary['parsed_row_sum_us'] = sum(r['time_us'] for r in rows if r['table_index'] == summary['table_index'])
        summary['parsed_sum_matches_reported'] = abs(summary['parsed_row_sum_us'] - summary['reported_total_us']) < .001
        if not summary['parsed_sum_matches_reported']:
            warnings.append({'source_line': summary['source_line'], 'reason': 'parsed timing sum differs from runtime reported sum', 'raw': summary['raw']})
    return {'schema_version': 1, 'metadata': {'format': 'rknn',
            'adapter': 'task_rknn_2_3_2_optional_blank_fields_v1', 'aggregation': 'none',
            'native_source_sha256': source_sha, 'source_file': str(src.resolve()),
            'source_size_bytes': len(raw_bytes), 'table_count': table,
            'input_time_units': ['us'], 'is_end_to_end': False, 'reported_summaries': summaries},
            'rows': rows, 'warnings': warnings}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('profile', type=Path)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    result = normalize(a.profile)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'rows': len(result['rows']), 'warnings': len(result['warnings']),
                      'summaries': result['metadata']['reported_summaries']}))
