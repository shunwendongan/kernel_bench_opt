#!/usr/bin/env python3
"""Exhaustive proof that bigvgan_deconv1d_v2 issues the same per-output mad
sequence as MNN deconv_2d (USE_BUFFER) for 1-D ConvTranspose.

For every output position o, both kernels loop ic blocks in the same outer
order, so it suffices to compare the inner (tap k, input ix, kind) sequence,
kind in {load, zero}. Skipped taps are absent from the sequence. Precision-
and device-independent; complements the GPU bit-pattern tests.
"""
import argparse, itertools, json, sys

def trunc_div(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q

def mnn_seq(o, T, K, S, P):
    trans_pad = K - 1 - P
    align = S - 1 - trans_pad
    start = max(0, trunc_div(o + align, S))
    deal = K - (start * S + trans_pad) + o - 1
    seq, ix, k = [], start, deal
    while k >= 0:
        seq.append((k, ix, 'zero' if ix >= T else 'load'))
        k -= S; ix += 1
    return seq

def v2_seq(o, T, K, S, P, TT):
    q, p = divmod(o, S)
    q0 = (q // TT) * TT; t = q - q0
    d_max = (K - 1 - P - p) // S
    d_min = -((P + p) // S)
    ntaps = d_max - d_min + 1
    ix0 = q0 - d_max
    k0 = P + p + S * d_max
    interior = ix0 >= 0 and ix0 + TT + ntaps - 2 < T
    seq = []
    for j in range(ntaps):
        ix = ix0 + t + j
        if not interior and ix < 0:
            continue
        assert not (interior and (ix < 0 or ix >= T))
        seq.append((k0 - S * j, ix, 'zero' if ix >= T else 'load'))
    return seq

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--tmax', type=int, default=300)
args = ap.parse_args()
configs = [(11, 5, 3), (7, 3, 2), (4, 2, 1)]   # BigVGAN ups.0 / ups.1 / ups.2-5
checked = 0
for (K, S, P), TT, T in itertools.product(configs, [1, 2, 4, 6, 8, 12, 16], range(1, args.tmax + 1)):
    for o in range(S * T):
        a, b = mnn_seq(o, T, K, S, P), v2_seq(o, T, K, S, P, TT)
        if a != b:
            print(json.dumps({'K': K, 'S': S, 'P': P, 'TT': TT, 'T': T, 'o': o, 'mnn': a, 'v2': b}))
            sys.exit(1)
        checked += 1
print(json.dumps({'passed': True, 'outputs_checked': checked, 'configs': configs,
                  'TT': [1, 2, 4, 6, 8, 12, 16], 'T': [1, args.tmax]}))
