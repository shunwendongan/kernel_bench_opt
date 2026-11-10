#!/usr/bin/env python3
"""Summarize bench_deconv_v2 JSONL files: per-shape best candidate and the
variant with the lowest summed time per kernel size (-> OMNI_DECONV_V2_K<k>)."""
import collections, json, re, sys
from pathlib import Path

root = Path(sys.argv[1])
rows, failures, per_k = [], [], collections.defaultdict(lambda: collections.defaultdict(float))
counts = collections.defaultdict(lambda: collections.defaultdict(int))
shapes_per_k = collections.defaultdict(set)
for f in sorted(root.glob('*.jsonl')):
    man, summ = None, {}
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r['kind'] == 'manifest': man = r
        elif r['kind'] == 'correctness' and r['bit_mismatches']: failures.append((f.name, r['label'], r['bit_mismatches']))
        elif r['kind'] == 'summary': summ[r['label']] = r
    if not man or not summ: continue
    base = min(s['baseline_host_med_us'] for s in summ.values())
    tile4 = next((s['cand_host_med_us'] for l, s in summ.items() if l.startswith('tile4')), None)
    v2 = {l: s['cand_host_med_us'] for l, s in summ.items() if l.startswith('v2_')}
    best = min(v2, key=v2.get)
    rows.append((f.stem, man['K'], man['T'], base, tile4, best, v2[best]))
    shapes_per_k[man['K']].add(f.stem)
    for l, t in v2.items():
        per_k[man['K']][l] += t; counts[man['K']][l] += 1

print(f'# bench_deconv_v2 summary: {root}\n')
print('| shape | K | T | baseline deconv_2d us | tile4 us | best v2 | v2 us | v2 vs baseline | v2 vs tile4 |')
print('|---|---:|---:|---:|---:|---|---:|---:|---:|')
for name, K, T, base, tile4, best, t in rows:
    t4 = f'{tile4:.1f}' if tile4 else '—'
    vt4 = f'{tile4 / t:.2f}x' if tile4 else '—'
    print(f'| {name} | {K} | {T} | {base:.1f} | {t4} | {best} | {t:.1f} | {base / t:.2f}x | {vt4} |')
print('\n## Bit-pattern gate')
print('All candidates bit-equal to deconv_2d.' if not failures else '\n'.join(f'- FAIL {a} {b}: {c} mismatches' for a, b, c in failures))
print('\n## Recommended per-kernel-size configuration (lowest summed time over all measured shapes)\n')
print('```bash')
for K, tot in sorted(per_k.items(), reverse=True):
    full = {l: t for l, t in tot.items() if counts[K][l] == len(shapes_per_k[K])}
    if not full: continue
    l = min(full, key=full.get)
    m = re.match(r'v2_tt(\d+)_ocb(\d+)_lws(.+)', l)
    tt, ocb, lws = m.groups()
    suffix = '' if lws == '0,0,0' else f':{lws}'
    print(f'export OMNI_DECONV_V2_K{K}={tt}:{ocb}{suffix}   # {l}, sum {full[l]:.1f} us')
print('```')
sys.exit(1 if failures else 0)
