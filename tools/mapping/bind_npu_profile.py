#!/usr/bin/env python3
"""Bind reviewed prefix graph boundaries to one actual RKNN PERF_DETAIL file."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--profile', type=Path, required=True)
p.add_argument('--rknn', type=Path, required=True, help='Exact deployed binary used for this profile; caller must verify the collection provenance')
p.add_argument('--out', type=Path, required=True)
p.add_argument('--toolkit', type=Path, required=True)
p.add_argument('--source-map', type=Path, required=True)
args = p.parse_args()
sys.path.insert(0, str(args.toolkit))
from hotspot_pipeline.cli import read_profile
from hotspot_pipeline.graph import index_model, profile_fingerprint

source = json.loads(args.source_map.read_text())
onnx = Path(source['onnx']['path'])
index = index_model(onnx)
if index['model']['sha256'] != source['onnx']['sha256']:
    raise SystemExit('ONNX changed since static review; repeat the source review.')
profile = read_profile(args.profile, format='auto')
if profile.get('metadata', {}).get('format') != 'rknn':
    raise SystemExit('Expected normalized RKNN PERF_DETAIL input.')
if any('row' in w.get('reason', '') or 'differs' in w.get('reason', '') for w in profile.get('warnings', [])):
    raise SystemExit('Source parser skipped rows; inspect warnings and normalize complete PERF_DETAIL before mapping.')
digest = hashlib.sha256(args.rknn.read_bytes()).hexdigest()
if digest != source['deployed_rknn_sha256_in_contract']:
    raise SystemExit('RKNN changed since static review; do not apply historical source mapping.')
rules = {
    ('Conv:/Conv', 'ConvExGelu'): 0,
    ('Mul:/Mul_6', 'Mul'): 1,
    ('Conv:/Conv_1', 'ConvExGelu'): 2,
    ('Slice:/Slice', 'Slice'): 3,
}
mapping = {}
for row in profile['rows']:
    rule = rules.get((row['layer_name'], row['op_type']))
    if rule is None:
        continue
    if row.get('backend') != 'NPU' or row.get('dtype') != 'FLOAT16':
        raise SystemExit('Reviewed source rules require NPU FLOAT16 rows; inspect changed profile first.')
    group = source['source_groups'][rule]
    mapping[row['record_id']] = {
        'node_ids': group['onnx_node_ids'],
        'reason': 'Reviewed actual RKNN-input ONNX boundary; ' + ' '.join(group['evidence']),
    }
if not mapping:
    raise SystemExit('No reviewed runtime units matched; do not infer from shape or ordinal order.')
result = {'model_sha256': index['model']['sha256'],
          'profile_sha256': profile_fingerprint(profile),
          'mappings': mapping, 'review_source': str(args.source_map.resolve()),
          'caller_supplied_rknn': {'path': str(args.rknn.resolve()), 'sha256': digest},
          'collection_provenance': 'Caller must independently verify this profile was collected from this binary.',
          'scope': 'Actual deployed prefix export; not full original HF model ONNX',
          'unmapped_policy': 'CPU IO and compiler-generated mask Conv have no original source node.'}
args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
print(json.dumps({'bound_records': len(mapping), 'total_records': len(profile['rows']),
                  'manual_map': str(args.out.resolve())}))
