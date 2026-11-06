"""核对已保存的真实输出。只产生诊断证据，不修改原有质量门槛。"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'dataset/samples.json').read_text(encoding='utf-8'))
verification = json.loads((ROOT / 'runs/verify_01/verification_samples.json').read_text(encoding='utf-8'))
comparisons = []
ranges = []
for sample in manifest['samples']:
    filename = f'{sample["id"]:012d}.npz'
    with np.load(ROOT / 'runs/verify_01/outputs' / filename) as reference:
        for run in ('baseline_diagnostic_01', 'detail_diagnostic_01'):
            path = ROOT / 'runs' / run / 'outputs' / filename
            with np.load(path) as actual:
                assert actual.files == reference.files
                comparisons.append({'image_id': sample['id'], 'run': run,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'all_outputs_exact_equal': all(np.array_equal(actual[k], reference[k]) for k in reference.files),
                    'maximum_abs_difference': max(float(np.max(np.abs(actual[k] - reference[k]))) for k in reference.files)})
        for index in range(9):
            a = reference[f'output_{index}']
            ranges.append({'image_id': sample['id'], 'index': index, 'minimum': float(a.min()),
                           'maximum': float(a.max()), 'fraction_at_minimum': float(np.mean(a == a.min()))})
per_output = []
for index in range(9):
    rows = [r['tensors']['tensors'][index] for r in verification]
    rs = [r for r in ranges if r['index'] == index]
    per_output.append({'index': index, 'tensor_count': len(rows), 'failed_relative_l2': sum(not t['passed'] for t in rows),
                       'relative_l2_min': min(t['relative_l2'] for t in rows),
                       'relative_l2_max': max(t['relative_l2'] for t in rows),
                       'max_abs_max': max(t['max_abs'] for t in rows),
                       'observed_output_minima': sorted(set(r['minimum'] for r in rs)),
                       'fraction_at_minimum_mean': float(np.mean([r['fraction_at_minimum'] for r in rs]))})
result = {'image_count': len(manifest['samples']), 'comparison_count': len(comparisons),
          'all_saved_outputs_equal_across_modes': all(r['all_outputs_exact_equal'] for r in comparisons),
          'scope': '各模式每图保留的首次正式输出；其余正常轮重复未保存张量',
          'quality_status': 'failed', 'quality_thresholds_changed': False,
          'per_output': per_output, 'comparisons': comparisons,
          'lite_crosscheck': json.loads((ROOT / 'evidence/lite_crosscheck.json').read_text(encoding='utf-8'))}
(ROOT / 'evidence/output_diagnostics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ('comparisons', 'lite_crosscheck')}, ensure_ascii=False, indent=2))
