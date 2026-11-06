#!/usr/bin/env python3
"""Regenerate the MNN embedded kernel from the standalone OpenCL source."""
import argparse
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
source = (root / 'kernels/bigvgan_deconv_s5_tiled.cl').read_text()
header = ('#pragma once\n'
          '// Generated from bigvgan_deconv_s5_tiled.cl; regenerate after kernel edits.\n'
          'static const char* kBigvganDeconvSource = R"BIGVGANCL(\n' + source + '\n)BIGVGANCL";\n')
path = root / 'kernels/bigvgan_deconv_source.hpp'
if args.check:
    if path.read_text() != header:
        raise SystemExit('Embedded kernel differs; run python3 tools/embed_kernel.py')
    print('Embedded kernel matches standalone source')
else:
    path.write_text(header)
