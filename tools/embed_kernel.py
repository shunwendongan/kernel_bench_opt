#!/usr/bin/env python3
"""Regenerate the MNN embedded kernels from the standalone OpenCL sources."""
import argparse
from pathlib import Path

root = Path(__file__).resolve().parents[1]
KERNELS = [
    ('kernels/bigvgan_deconv_s5_tiled.cl', 'kernels/bigvgan_deconv_source.hpp',
     'kBigvganDeconvSource', 'BIGVGANCL'),
    ('kernels/v2/bigvgan_deconv1d_v2.cl', 'kernels/v2/bigvgan_deconv1d_v2_source.hpp',
     'kBigvganDeconv1dV2Source', 'DECONV1DV2CL'),
]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
failed = False
for cl, hpp, symbol, delim in KERNELS:
    source = (root / cl).read_text()
    header = ('#pragma once\n'
              f'// Generated from {Path(cl).name}; regenerate after kernel edits.\n'
              f'static const char* {symbol} = R"{delim}(\n' + source + f'\n){delim}";\n')
    path = root / hpp
    if args.check:
        if not path.is_file() or path.read_text() != header:
            print(f'{hpp} differs; run python3 tools/embed_kernel.py')
            failed = True
        else:
            print(f'{hpp} matches {cl}')
    else:
        path.write_text(header)
if failed:
    raise SystemExit(1)
