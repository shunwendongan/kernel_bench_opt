#!/usr/bin/env python3
"""Match exact BigVGAN callback names to a newly exported reference ONNX.

Every match remains a structural candidate. This cannot establish the deployed
MNN model's historical conversion lineage or numerical equivalence.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def candidates(args):
    import onnx
    graph = onnx.load(str(args.onnx), load_external_data=False)
    by_name = {}
    for index, node in enumerate(graph.graph.node):
        record = {'index': index, 'node_id': f'n{index:06d}', 'name': node.name,
                  'op_type': node.op_type, 'inputs': list(node.input), 'outputs': list(node.output)}
        for name in [node.name, *node.output]:
            if name:
                by_name.setdefault(name, []).append(record)
    opener = gzip.open if args.profile.suffix == '.gz' else open
    ranked = {}
    invalid = []
    with opener(args.profile, 'rt', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid.append(line_number)
                continue
            if row.get('kind') != 'op' or row.get('stage') != 'bigvgan':
                continue
            key = (row['executor'], row['name'], row['type'], json.dumps(row['output_tensor_backends']))
            result = ranked.setdefault(key, {'executor': row['executor'], 'runtime_name': row['name'],
                'runtime_type': row['type'], 'output_tensor_backends': row['output_tensor_backends'],
                'calls': 0, 'total_instrumented_host_us': 0., 'max_instrumented_host_us': 0.,
                'first_source_line': line_number, 'reference_candidates': by_name.get(row['name'], []),
                'mapping_status': 'reference_structure_candidate' if row['name'] in by_name else 'unmapped',
                'historical_MNN_provenance_confirmed': False, 'numerical_equivalence_confirmed': False})
            result['calls'] += 1
            result['total_instrumented_host_us'] += float(row['elapsed_us'])
            result['max_instrumented_host_us'] = max(result['max_instrumented_host_us'], float(row['elapsed_us']))
    result = {'reference_onnx': {'path': str(args.onnx.resolve()), 'sha256': sha(args.onnx)},
              'input_profile': {'path': str(args.profile.resolve()), 'sha256': sha(args.profile)},
              'match_rule': 'Exact callback name equals reference node name or output tensor name; no fuzzy matching',
              'mapping_scope': 'new_reference_structure_only', 'invalid_json_lines': invalid,
              'records': sorted(ranked.values(), key=lambda r: -r['total_instrumented_host_us']),
              'limitations': [
                  'The reference ONNX was newly exported from original weights. It is not the historical source of the deployed MNN snapshot.',
                  'All matches are review candidates; quantization, activation export rewrites and runtime fusion can change the executed computation.',
                  'Durations are instrumented synchronized host callbacks, not pure OpenCL kernel times or clean end-to-end timings.',
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'distinct_runtime_records': len(ranked),
                      'candidate_records': sum(r['mapping_status'] == 'reference_structure_candidate' for r in ranked.values()),
                      'output': str(args.output.resolve())}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', required=True, type=Path)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    candidates(parser.parse_args())
