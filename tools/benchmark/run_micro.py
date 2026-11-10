#!/usr/bin/env python3
"""Run explicit local OpenCL operator tests. Does not manage other workloads."""
import argparse
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--captures', type=Path, required=True)
    p.add_argument('--mnn-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['formal', 'ablation'], default='formal')
    p.add_argument('--dry-run', action='store_true', help='Print commands without running inference')
    a = p.parse_args()
    cases = []
    if a.phase == 'formal':
        for t in [52, 120, 136]:
            for profile in [1, 0]:
                cases.append((t, 4, '0,0,0', 15, profile, f'T{t}_profile{profile}'))
    else:
        cases = [(136, tile, '0,0,0', 5, 1, f'ablation_tile{tile}') for tile in [1, 2, 4]]
        cases.append((136, 4, '8,1,4', 5, 1, 'ablation_old_lws'))
    baseline = a.mnn_source.resolve() / 'source/backend/opencl/execution/cl/deconv_2d.cl'
    for t, tile, old_lws, repeats, profile, name in cases:
        output = a.output.resolve() / 'bench' / name
        if output.exists():
            p.error(f'Refusing to overwrite {output}')
        cmd = [str(a.binary.resolve()), '--input', str(a.captures.resolve() / f'input_T{t}.f32'),
               '--weights', str(a.captures.resolve() / 'packed_weight.bin'),
               '--bias', str(a.captures.resolve() / 'packed_bias.bin'),
               '--T', str(t), '--axis', 'h', '--baseline', str(baseline),
               '--candidate', str(REPO / 'kernels/bigvgan-convtranspose-tile4/bigvgan_deconv_s5_tiled.cl'),
               '--out', str(output), '--tile', str(tile), '--old-lws', old_lws,
               '--new-lws', '8,4,1', '--warmup', '3', '--reps', str(repeats),
               '--profile', str(profile), '--reference', str(a.captures.resolve() / f'output_T{t}.f32')]
        command = [sys.executable, str(Path(__file__).with_name('run_recorded.py')),
                   str(a.output.resolve() / 'records' / name)] + cmd
        if a.dry_run:
            import shlex
            print(shlex.join(command))
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
