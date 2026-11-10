/*
 * Generic 1-D strided ConvTranspose for MNN OpenCL Buffer (deconv_2d, USE_BUFFER).
 * Candidate: bigvgan-deconv1d-v2 (parent: bigvgan-convtranspose-tile4).
 *
 * Covers every BigVGAN upsampler (ups.0..5: K11/S5/P3, K7/S3/P2, K4/S2/P1) and
 * any temporal length T, with output length S*T (requires K - 2*PAD == S).
 *
 * Layouts are unchanged from MNN:
 *   input   [CI4, T, 4]          (batch 1, one spatial axis == 1)
 *   weights [CI4*4, CO4, K, 4]   (deconv CONV2D_FILTER packing, untouched bytes)
 *   output  [CO4, S*T, 4]
 *
 * Phase decomposition: output o = S*q + p. For phase p the contributing taps are
 * k = PAD + p + S*d, d in [d_min(p), d_max(p)], input ix = q - d.
 * Launch: global = {CO4/OCB, ceil(T/TT), S}; each work item owns OCB output
 * channel blocks x TT consecutive q of one phase.
 *
 * Bit-exactness contract vs MNN deconv_2d: every accumulator receives exactly
 * the same mad sequence (bias, then ic block ascending -> k descending ->
 * x,y,z,w). Taps with ix < 0 are skipped (MNN starts at max(0,...)); taps with
 * ix >= T are executed as mad with zero (MNN outBoundry), so -0.0 handling is
 * identical. Only data reuse changes:
 *   - one weight quad per (oc block, tap) is shared by TT time positions;
 *   - one input window of TT+MAXTAPS-1 vectors per ic block is shared by all
 *     taps and all OCB output blocks (tile4 reloaded inputs per tap).
 *
 * Compile-time: DC_K DC_S DC_PAD DC_CI4 DC_CO4 DC_TT DC_OCB, FLOAT/FLOAT4 from MNN.
 *
 * Performance note: the kernel is bound by weight traffic (each weight quad is
 * fetched once per q tile). Choose a local size that spans time (dim1), e.g.
 * {4,8,1}, so all q tiles of an oc block hit the same cached weights; a local
 * size spanning only channels ({32,1,1}) was 3-8x slower in the sweep.
 */
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifndef DC_TT
#define DC_TT 4
#endif
#ifndef DC_OCB
#define DC_OCB 2
#endif
#if !defined(DC_K) || !defined(DC_S) || !defined(DC_PAD) || !defined(DC_CI4) || !defined(DC_CO4)
#error DC_K_DC_S_DC_PAD_DC_CI4_DC_CO4_required
#endif
#if (DC_K - 2 * DC_PAD) != DC_S
#error output_length_must_be_S_times_T
#endif
#if DC_TT < 1 || DC_TT > 16
#error DC_TT_out_of_range
#endif
#if DC_OCB != 1 && DC_OCB != 2 && DC_OCB != 4
#error DC_OCB_must_be_1_2_or_4
#endif
#if (DC_CO4 % DC_OCB) != 0
#error DC_CO4_must_be_divisible_by_DC_OCB
#endif

#define DC_MAXTAPS ((DC_K + DC_S - 1) / DC_S)
#define DC_WIN (DC_TT + DC_MAXTAPS - 1)
#define DC_WSTRIDE (DC_CO4 * DC_K) /* half4/float4 units between ic rows */

#define DC_MAD4(acc, xv, w0, w1, w2, w3) \
    acc = mad((xv).x, w0, acc);          \
    acc = mad((xv).y, w1, acc);          \
    acc = mad((xv).z, w2, acc);          \
    acc = mad((xv).w, w3, acc)

__kernel void bigvgan_deconv1d_v2(
    const int global_size_dim0, const int global_size_dim1,
    const int global_size_dim2,
    __global const FLOAT *input, __global const FLOAT *weights,
    __global const FLOAT *bias, __global FLOAT *output, const int T) {
    const int ocg = get_global_id(0);
    const int qt = get_global_id(1);
    const int p = get_global_id(2);
    if (ocg >= global_size_dim0 || qt >= global_size_dim1 || p >= global_size_dim2) return;

    const int oc0 = ocg * DC_OCB;
    const int q0 = qt * DC_TT;
    const int d_max = (DC_K - 1 - DC_PAD - p) / DC_S;   /* numerator >= 0 */
    const int d_min = -((DC_PAD + p) / DC_S);
    const int ntaps = d_max - d_min + 1;                /* <= DC_MAXTAPS */
    const int ix0 = q0 - d_max;                         /* window start */
    const int k0 = DC_PAD + p + DC_S * d_max;           /* first (largest) tap */
    /* Interior tile: no negative index to skip, no zero tap to synthesize. */
    const int interior = (ix0 >= 0) && (ix0 + DC_TT + ntaps - 2 < T);

    FLOAT4 acc[DC_OCB][DC_TT];
#pragma unroll
    for (int o = 0; o < DC_OCB; ++o) {
        const FLOAT4 b = vload4(oc0 + o, bias);
#pragma unroll
        for (int t = 0; t < DC_TT; ++t) acc[o][t] = b;
    }

    __global const FLOAT *wrow = weights + (size_t)(oc0 * DC_K + k0) * 4;
    __global const FLOAT *xrow = input;

    if (interior) {
        for (int ic = 0; ic < DC_CI4; ++ic) {
            FLOAT4 win[DC_WIN];
#pragma unroll
            for (int i = 0; i < DC_WIN; ++i)
                win[i] = (i < DC_TT + ntaps - 1) ? vload4(ix0 + i, xrow) : (FLOAT4)0;
#pragma unroll
            for (int j = 0; j < DC_MAXTAPS; ++j) {
                if (j < ntaps) {
#pragma unroll
                    for (int o = 0; o < DC_OCB; ++o) {
                        const int wo = o * DC_K - j * DC_S;
                        const FLOAT4 w0 = vload4(wo, wrow);
                        const FLOAT4 w1 = vload4(wo + DC_WSTRIDE, wrow);
                        const FLOAT4 w2 = vload4(wo + 2 * DC_WSTRIDE, wrow);
                        const FLOAT4 w3 = vload4(wo + 3 * DC_WSTRIDE, wrow);
#pragma unroll
                        for (int t = 0; t < DC_TT; ++t) {
                            DC_MAD4(acc[o][t], win[t + j], w0, w1, w2, w3);
                        }
                    }
                }
            }
            wrow += 4 * DC_WSTRIDE * 4;
            xrow += T * 4;
        }
    } else {
        for (int ic = 0; ic < DC_CI4; ++ic) {
            FLOAT4 win[DC_WIN];
#pragma unroll
            for (int i = 0; i < DC_WIN; ++i) {
                const int ix = ix0 + i;
                win[i] = (ix >= 0 && ix < T) ? vload4(ix, xrow) : (FLOAT4)0;
            }
#pragma unroll
            for (int j = 0; j < DC_MAXTAPS; ++j) {
                if (j < ntaps) {
#pragma unroll
                    for (int o = 0; o < DC_OCB; ++o) {
                        const int wo = o * DC_K - j * DC_S;
                        const FLOAT4 w0 = vload4(wo, wrow);
                        const FLOAT4 w1 = vload4(wo + DC_WSTRIDE, wrow);
                        const FLOAT4 w2 = vload4(wo + 2 * DC_WSTRIDE, wrow);
                        const FLOAT4 w3 = vload4(wo + 3 * DC_WSTRIDE, wrow);
#pragma unroll
                        for (int t = 0; t < DC_TT; ++t) {
                            /* MNN never visits ix < 0; ix >= T is a zero mad. */
                            if (ix0 + t + j >= 0) {
                                DC_MAD4(acc[o][t], win[t + j], w0, w1, w2, w3);
                            }
                        }
                    }
                }
            }
            wrow += 4 * DC_WSTRIDE * 4;
            xrow += T * 4;
        }
    }

    const int outT = DC_S * T;
#pragma unroll
    for (int o = 0; o < DC_OCB; ++o) {
        __global FLOAT *orow = output + (size_t)(oc0 + o) * outT * 4;
#pragma unroll
        for (int t = 0; t < DC_TT; ++t) {
            if (q0 + t < T) {
                FLOAT4 v = acc[o][t];
#ifdef RELU
                v = fmax(v, (FLOAT4)0);
#endif
#ifdef RELU6
                v = clamp(v, (FLOAT4)0, (FLOAT4)6);
#endif
                vstore4(v, DC_S * (q0 + t) + p, orow);
            }
        }
    }
}
