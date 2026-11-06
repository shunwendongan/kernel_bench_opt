/*
 * BigVGAN first ConvTranspose, same precision and packed weights as MNN.
 * Contract: N=1, Ci=1536, Co=768, K=11, S=5, P=3, D=1, groups=1,
 *           one spatial axis is 1, output temporal length = 5 * input T.
 *
 * The API deliberately matches deconv_2d(USE_BUFFER, BIAS), but launch is:
 *   global = {192, ceil(T / BIGVGAN_TILE), 5}.
 * Each work item computes BIGVGAN_TILE time positions with the same stride
 * phase. The four C4 weight vectors are shared by those output positions.
 * Per-output order stays ic-block -> descending tap -> x,y,z,w mad.
 * FLOAT and FLOAT4 must be inherited unchanged from the current runtime.
 */
#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#ifndef BIGVGAN_TILE
#define BIGVGAN_TILE 4
#endif
#if BIGVGAN_TILE != 1 && BIGVGAN_TILE != 2 && BIGVGAN_TILE != 4
#error BIGVGAN_TILE_must_be_1_2_or_4
#endif

#define ACCUMULATE_SAME_ORDER(dst, in) \
    dst = mad(in.x, w0, dst); \
    dst = mad(in.y, w1, dst); \
    dst = mad(in.z, w2, dst); \
    dst = mad(in.w, w3, dst)

__kernel void bigvgan_deconv_s5_tiled(
    const int global_size_dim0, const int global_size_dim1,
    const int global_size_dim2,
    __global const FLOAT *input, __global const FLOAT *weights,
    __global const FLOAT *bias, __global FLOAT *output, const int batch,
    const int2 input_shape, const int2 output_shape, const int2 stride_shape,
    const int2 align_shape, const int2 padding_shape, const int2 kernel_shape,
    const int kernel_size, const int in_channel_blocks,
    const int out_channel_blocks) {
    const int oc = get_global_id(0);
    const int qtile = get_global_id(1);
    const int phase = get_global_id(2);
    if (oc >= global_size_dim0 || qtile >= global_size_dim1 ||
        phase >= global_size_dim2) return;

    const int T = input_shape.x * input_shape.y;
    const int outT = 5 * T;
    const int q = qtile * BIGVGAN_TILE;
    const int lead = phase < 3;
    const int first_k = phase + 3 + 5 * lead;
    const FLOAT4 b = vload4(oc, bias);
    FLOAT4 a0 = b;
#if BIGVGAN_TILE >= 2
    FLOAT4 a1 = b;
#endif
#if BIGVGAN_TILE >= 4
    FLOAT4 a2 = b;
    FLOAT4 a3 = b;
#endif

    for (int ic = 0; ic < 384; ++ic) {
        const int wbase = (ic * 4) * (192 * 11) + oc * 11;
        int ix = q - lead;
        for (int k = first_k; k >= 0; k -= 5, ++ix) {
            const FLOAT4 w0 = vload4(wbase + k, weights);
            const FLOAT4 w1 = vload4(wbase + 192 * 11 + k, weights);
            const FLOAT4 w2 = vload4(wbase + 2 * 192 * 11 + k, weights);
            const FLOAT4 w3 = vload4(wbase + 3 * 192 * 11 + k, weights);

            // MNN skips negative input taps; retain that exact behavior.
            // Its right padding is an explicit zero-valued mad, kept here.
            if (ix >= 0) {
                const FLOAT4 x0 = ix < T ? vload4(ic * T + ix, input) : (FLOAT4)0;
                ACCUMULATE_SAME_ORDER(a0, x0);
            }
#if BIGVGAN_TILE >= 2
            if (ix + 1 >= 0) {
                const FLOAT4 x1 = ix + 1 < T ? vload4(ic * T + ix + 1, input) : (FLOAT4)0;
                ACCUMULATE_SAME_ORDER(a1, x1);
            }
#endif
#if BIGVGAN_TILE >= 4
            if (ix + 2 >= 0) {
                const FLOAT4 x2 = ix + 2 < T ? vload4(ic * T + ix + 2, input) : (FLOAT4)0;
                ACCUMULATE_SAME_ORDER(a2, x2);
            }
            if (ix + 3 >= 0) {
                const FLOAT4 x3 = ix + 3 < T ? vload4(ic * T + ix + 3, input) : (FLOAT4)0;
                ACCUMULATE_SAME_ORDER(a3, x3);
            }
#endif
        }
    }

    // No activation fusion is permitted by the dispatch guard.
    const int obase = oc * outT + q * 5 + phase;
    if (q < T) vstore4(a0, obase, output);
#if BIGVGAN_TILE >= 2
    if (q + 1 < T) vstore4(a1, obase + 5, output);
#endif
#if BIGVGAN_TILE >= 4
    if (q + 2 < T) vstore4(a2, obase + 10, output);
    if (q + 3 < T) vstore4(a3, obase + 15, output);
#endif
}
