/* pre-letterbox-neon implementation. See letterbox.h. */
#include "letterbox.h"
#include "../common/fp16.h"

#include <math.h>
#include <string.h>

#if YOLO_HAVE_NEON && defined(__aarch64__)
#define YOLO_LB_NEON 1
#else
#define YOLO_LB_NEON 0
#endif

#define COEF_BITS 11
#define COEF_SCALE (1 << COEF_BITS)        /* 2048, OpenCV INTER_RESIZE_COEF_SCALE */

int32_t yolo_letterbox_has_neon(void) { return YOLO_LB_NEON; }

/* Python round(): round half to even on IEEE doubles. */
static inline int32_t py_round(double v) { return (int32_t)nearbyint(v); }

void yolo_letterbox_geometry(int32_t w, int32_t h, int32_t size, yolo_letterbox_transform* t) {
    double ratio_w = (double)size / (double)w, ratio_h = (double)size / (double)h;
    double ratio = ratio_w < ratio_h ? ratio_w : ratio_h;
    int32_t new_w = py_round((double)w * ratio), new_h = py_round((double)h * ratio);
    double dw = (double)(size - new_w) / 2.0, dh = (double)(size - new_h) / 2.0;
    t->original_width = w; t->original_height = h;
    t->ratio = ratio;
    t->scale_x = (double)new_w / (double)w;
    t->scale_y = (double)new_h / (double)h;
    t->pad_left = py_round(dw - 0.1); t->pad_right = py_round(dw + 0.1);
    t->pad_top = py_round(dh - 0.1);  t->pad_bottom = py_round(dh + 0.1);
    t->resized_width = new_w; t->resized_height = new_h;
    t->size = size; t->fill = 114;
}

size_t yolo_letterbox_workspace_bytes(int32_t size, int32_t source_width) {
    (void)source_width;   /* the horizontal pass reads the source in place */
    size_t dwidth = 3 * (size_t)size + 16;             /* output row in elements (+ slack) */
    /* xofs (int32) + alpha (2 x int16) per element, two int32 rows, yofs/beta per output row, alignment */
    return dwidth * (sizeof(int32_t) + 2 * sizeof(int16_t)) + 2 * dwidth * sizeof(int32_t)
         + ((size_t)size + 16) * (sizeof(int32_t) + 2 * sizeof(int16_t)) + 4 * 64;
}

/* saturate_cast<short>(float) as used by OpenCV: round to nearest. */
static inline int16_t coef_from_float(float v) { return (int16_t)lrintf(v * (float)COEF_SCALE); }

static inline uint8_t sat_u8(int32_t v) { return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }

/* Vertical combine of two int32 rows (already weighted by 2048 horizontally). */
static void vresize_row(const int32_t* S0, const int32_t* S1, int16_t b0, int16_t b1,
                        uint8_t* dst, int32_t width, int32_t rounding) {
    int32_t simd_end = 0;
    if (rounding == YOLO_ROUND_OPENCV) simd_end = width;            /* VResizeLinear<uchar,...> specialisation */
    else if (rounding == YOLO_ROUND_OPENCV_TAIL22) simd_end = width & ~15;
    int32_t x = 0;
#if YOLO_LB_NEON
    {
        int16x8_t vb0 = vdupq_n_s16(b0), vb1 = vdupq_n_s16(b1);
        for (; x + 16 <= simd_end; x += 16) {
            int16x8_t t0a = vcombine_s16(vqmovn_s32(vshrq_n_s32(vld1q_s32(S0 + x), 4)),
                                         vqmovn_s32(vshrq_n_s32(vld1q_s32(S0 + x + 4), 4)));
            int16x8_t t0b = vcombine_s16(vqmovn_s32(vshrq_n_s32(vld1q_s32(S0 + x + 8), 4)),
                                         vqmovn_s32(vshrq_n_s32(vld1q_s32(S0 + x + 12), 4)));
            int16x8_t t1a = vcombine_s16(vqmovn_s32(vshrq_n_s32(vld1q_s32(S1 + x), 4)),
                                         vqmovn_s32(vshrq_n_s32(vld1q_s32(S1 + x + 4), 4)));
            int16x8_t t1b = vcombine_s16(vqmovn_s32(vshrq_n_s32(vld1q_s32(S1 + x + 8), 4)),
                                         vqmovn_s32(vshrq_n_s32(vld1q_s32(S1 + x + 12), 4)));
            /* v_mul_hi: (a*b) >> 16 per int16 lane */
            int16x8_t ma = vaddq_s16(vshrn_high_n_s32(vshrn_n_s32(vmull_s16(vget_low_s16(t0a), vget_low_s16(vb0)), 16),
                                                      vmull_s16(vget_high_s16(t0a), vget_high_s16(vb0)), 16),
                                     vshrn_high_n_s32(vshrn_n_s32(vmull_s16(vget_low_s16(t1a), vget_low_s16(vb1)), 16),
                                                      vmull_s16(vget_high_s16(t1a), vget_high_s16(vb1)), 16));
            int16x8_t mb = vaddq_s16(vshrn_high_n_s32(vshrn_n_s32(vmull_s16(vget_low_s16(t0b), vget_low_s16(vb0)), 16),
                                                      vmull_s16(vget_high_s16(t0b), vget_high_s16(vb0)), 16),
                                     vshrn_high_n_s32(vshrn_n_s32(vmull_s16(vget_low_s16(t1b), vget_low_s16(vb1)), 16),
                                                      vmull_s16(vget_high_s16(t1b), vget_high_s16(vb1)), 16));
            /* v_rshr_pack_u<2>: (v + 2) >> 2 with unsigned saturation */
            uint8x16_t out = vcombine_u8(vqrshrun_n_s16(ma, 2), vqrshrun_n_s16(mb, 2));
            vst1q_u8(dst + x, out);
        }
    }
#endif
    for (; x < simd_end; ++x) {   /* same rule without NEON: ((b0*(S0>>4))>>16 + (b1*(S1>>4))>>16 + 2) >> 2 */
        int32_t t0 = (int16_t)(S0[x] >> 4), t1 = (int16_t)(S1[x] >> 4);
        int32_t m = ((t0 * (int32_t)b0) >> 16) + ((t1 * (int32_t)b1) >> 16);
        dst[x] = sat_u8((m + 2) >> 2);
    }
    for (; x < width; ++x) {      /* generic FixedPtCast<int, uchar, 22> (diagnostic modes only) */
        int32_t v = S0[x] * (int32_t)b0 + S1[x] * (int32_t)b1;
        dst[x] = sat_u8((v + (1 << 21)) >> 22);
    }
}

int32_t yolo_resize_linear_u8c3(const uint8_t* src, int32_t sw, int32_t sh, int64_t pitch,
                                uint8_t* dst, int32_t dw, int32_t dh, int64_t dst_pitch,
                                int32_t rounding, void* workspace, size_t workspace_bytes) {
    if (!src || !dst || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return -1;
    if (workspace_bytes < yolo_letterbox_workspace_bytes(dw > dh ? dw : dh, sw)) return -3;
    if (dw == sw && dh == sh) {     /* cv::resize: same size => plain copy */
        for (int32_t y = 0; y < sh; ++y) memcpy(dst + y * dst_pitch, src + y * pitch, (size_t)sw * 3);
        return 0;
    }
    const int32_t cn = 3;
    const int32_t dwidth = dw * cn;
    uintptr_t p = ((uintptr_t)workspace + 63) & ~(uintptr_t)63;
    int32_t* xofs = (int32_t*)p; p += (size_t)(dwidth + 16) * sizeof(int32_t);
    int16_t* alpha = (int16_t*)p; p += (size_t)(dwidth + 16) * 2 * sizeof(int16_t);
    int32_t* yofs = (int32_t*)p; p += (size_t)(dh + 16) * sizeof(int32_t);
    int16_t* beta = (int16_t*)p; p += (size_t)(dh + 16) * 2 * sizeof(int16_t);
    p = (p + 63) & ~(uintptr_t)63;
    int32_t* row0 = (int32_t*)p; p += (size_t)(dwidth + 16) * sizeof(int32_t);
    p = (p + 63) & ~(uintptr_t)63;
    int32_t* row1 = (int32_t*)p;

    /* OpenCV: inv_scale = dsize/ssize, scale = 1/inv_scale */
    double scale_x = 1.0 / ((double)dw / (double)sw);
    double scale_y = 1.0 / ((double)dh / (double)sh);
    int32_t xmax = dw;
    for (int32_t dx = 0; dx < dw; ++dx) {
        float fx = (float)(((double)dx + 0.5) * scale_x - 0.5);
        int32_t sx = (int32_t)floorf(fx);
        fx -= (float)sx;
        if (sx < 0) { fx = 0.0f; sx = 0; }
        if (sx + 1 >= sw) { if (xmax > dx) xmax = dx; if (sx >= sw - 1) { fx = 0.0f; sx = sw - 1; } }
        for (int32_t k = 0; k < cn; ++k) {
            xofs[dx * cn + k] = sx * cn + k;
            alpha[(dx * cn + k) * 2] = coef_from_float(1.0f - fx);
            alpha[(dx * cn + k) * 2 + 1] = coef_from_float(fx);
        }
    }
    /* OpenCV clamps the *row indices* when fetching rows but keeps the raw fy
     * for the vertical coefficients (only fx is zeroed at the horizontal border). */
    for (int32_t dy = 0; dy < dh; ++dy) {
        float fy = (float)(((double)dy + 0.5) * scale_y - 0.5);
        int32_t sy = (int32_t)floorf(fy);
        fy -= (float)sy;
        yofs[dy] = sy;
        beta[dy * 2] = coef_from_float(1.0f - fy);
        beta[dy * 2 + 1] = coef_from_float(fy);
    }
    const int32_t xmax_elems = xmax * cn;
    int32_t cached0 = -1, cached1 = -1;
    for (int32_t dy = 0; dy < dh; ++dy) {
        int32_t sy0 = yofs[dy] < 0 ? 0 : yofs[dy] >= sh ? sh - 1 : yofs[dy];
        int32_t sy1 = yofs[dy] + 1 < 0 ? 0 : yofs[dy] + 1 >= sh ? sh - 1 : yofs[dy] + 1;
        /* horizontal pass for the two source rows (reuse when rows repeat) */
        for (int pass = 0; pass < 2; ++pass) {
            int32_t sy = pass ? sy1 : sy0;
            int32_t* D = pass ? row1 : row0;
            int32_t* cached = pass ? &cached1 : &cached0;
            if (*cached == sy) continue;
            if (pass == 1 && sy == sy0) { memcpy(row1, row0, (size_t)dwidth * sizeof(int32_t)); cached1 = sy; continue; }
            const uint8_t* S = src + (int64_t)sy * pitch;
            int32_t dx = 0;
            for (; dx < xmax_elems; ++dx) {
                int32_t sx = xofs[dx];
                D[dx] = S[sx] * (int32_t)alpha[dx * 2] + S[sx + cn] * (int32_t)alpha[dx * 2 + 1];
            }
            for (; dx < dwidth; ++dx) D[dx] = S[xofs[dx]] * COEF_SCALE;
            *cached = sy;
        }
        vresize_row(row0, row1, beta[dy * 2], beta[dy * 2 + 1], dst + (int64_t)dy * dst_pitch, dwidth, rounding);
    }
    return 0;
}

int32_t yolo_letterbox_rgb_u8(const uint8_t* bgr, int32_t width, int32_t height, int64_t pitch,
                              uint8_t* rgb_out, int32_t size, int32_t rounding,
                              yolo_letterbox_transform* t, void* workspace, size_t workspace_bytes) {
    if (!bgr || !rgb_out || !t || width <= 0 || height <= 0 || size <= 0) return -1;
    yolo_letterbox_geometry(width, height, size, t);
    const int32_t new_w = t->resized_width, new_h = t->resized_height;
    const int64_t out_pitch = (int64_t)size * 3;
    /* fill top/bottom bands and left/right columns with 114 */
    memset(rgb_out, t->fill, (size_t)(t->pad_top * out_pitch));
    memset(rgb_out + (int64_t)(t->pad_top + new_h) * out_pitch, t->fill, (size_t)(t->pad_bottom * out_pitch));
    for (int32_t y = 0; y < new_h; ++y) {
        uint8_t* row = rgb_out + (int64_t)(t->pad_top + y) * out_pitch;
        memset(row, t->fill, (size_t)t->pad_left * 3);
        memset(row + (int64_t)(t->pad_left + new_w) * 3, t->fill, (size_t)t->pad_right * 3);
    }
    uint8_t* dst = rgb_out + (int64_t)t->pad_top * out_pitch + (int64_t)t->pad_left * 3;
    int32_t rc = yolo_resize_linear_u8c3(bgr, width, height, pitch, dst, new_w, new_h, out_pitch,
                                         rounding, workspace, workspace_bytes);
    if (rc) return rc;
    /* BGR -> RGB in place on the resized region */
    for (int32_t y = 0; y < new_h; ++y) {
        uint8_t* row = dst + (int64_t)y * out_pitch;
        int32_t x = 0;
#if YOLO_LB_NEON
        for (; x + 16 <= new_w; x += 16) {
            uint8x16x3_t px = vld3q_u8(row + x * 3);
            uint8x16_t b = px.val[0]; px.val[0] = px.val[2]; px.val[2] = b;
            vst3q_u8(row + x * 3, px);
        }
#endif
        for (; x < new_w; ++x) { uint8_t b = row[x * 3]; row[x * 3] = row[x * 3 + 2]; row[x * 3 + 2] = b; }
    }
    return 0;
}

int64_t yolo_input_pack(const uint8_t* rgb, int32_t size, int32_t format, int32_t w_stride, int32_t c2,
                        float scale, int32_t divide, void* dst, size_t dst_bytes) {
    if (!rgb || !dst || size <= 0) return -1;
    if (divide) { if (scale == 0.0f) return -1; }
    /* value = divide ? pixel / scale : pixel * scale (the reference divides by 255 in float32) */
    const float mul = divide ? 0.0f : scale;
    const int32_t ws = w_stride > 0 ? w_stride : size;
    if (ws < size) return -1;
    const int64_t hw = (int64_t)size * size;
    switch (format) {
    case YOLO_INPUT_NHWC_U8: {
        int64_t need = (int64_t)size * ws * 3;
        if ((int64_t)dst_bytes < need) return -2;
        for (int32_t y = 0; y < size; ++y)
            memcpy((uint8_t*)dst + (int64_t)y * ws * 3, rgb + (int64_t)y * size * 3, (size_t)size * 3);
        return need;
    }
    case YOLO_INPUT_NHWC_F16: {
        int64_t need = (int64_t)size * ws * 3 * 2;
        if ((int64_t)dst_bytes < need) return -2;
        for (int32_t y = 0; y < size; ++y) {
            const uint8_t* s = rgb + (int64_t)y * size * 3;
            uint16_t* d = (uint16_t*)dst + (int64_t)y * ws * 3;
            int32_t n = size * 3, i = 0;
#if YOLO_LB_NEON
            float32x4_t vscale = vdupq_n_f32(mul);
            for (; !divide && i + 16 <= n; i += 16) {
                uint8x16_t v = vld1q_u8(s + i);
                uint16x8_t lo = vmovl_u8(vget_low_u8(v)), hi = vmovl_u8(vget_high_u8(v));
                float32x4_t f0 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))), vscale);
                float32x4_t f1 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo))), vscale);
                float32x4_t f2 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))), vscale);
                float32x4_t f3 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi))), vscale);
                yolo_store4_fp16(d + i, f0); yolo_store4_fp16(d + i + 4, f1);
                yolo_store4_fp16(d + i + 8, f2); yolo_store4_fp16(d + i + 12, f3);
            }
#endif
            for (; i < n; ++i) d[i] = yolo_f32_to_fp16(divide ? (float)s[i] / scale : (float)s[i] * mul);
        }
        return need;
    }
    case YOLO_INPUT_NCHW_F16:
    case YOLO_INPUT_NCHW_F32: {
        int elem = format == YOLO_INPUT_NCHW_F16 ? 2 : 4;
        int64_t need = (int64_t)3 * size * ws * elem;
        if ((int64_t)dst_bytes < need) return -2;
        for (int32_t c = 0; c < 3; ++c)
            for (int32_t y = 0; y < size; ++y) {
                const uint8_t* s = rgb + (int64_t)y * size * 3 + c;
                int64_t base = ((int64_t)c * size + y) * ws;
                if (elem == 2) { uint16_t* d = (uint16_t*)dst + base; for (int32_t x = 0; x < size; ++x) d[x] = yolo_f32_to_fp16(divide ? (float)s[x * 3] / scale : (float)s[x * 3] * mul); }
                else { float* d = (float*)dst + base; for (int32_t x = 0; x < size; ++x) d[x] = divide ? (float)s[x * 3] / scale : (float)s[x * 3] * mul; }
            }
        return need;
    }
    case YOLO_INPUT_NC1HWC2_F16: {
        if (c2 < 3) return -1;
        int64_t need = (int64_t)size * ws * c2 * 2;   /* C1 == 1 for 3 channels */
        if ((int64_t)dst_bytes < need) return -2;
        uint16_t* d = (uint16_t*)dst;
        for (int32_t y = 0; y < size; ++y) {
            const uint8_t* s = rgb + (int64_t)y * size * 3;
            uint16_t* drow = d + (int64_t)y * ws * c2;
            for (int32_t x = 0; x < size; ++x) {
                uint16_t* px = drow + (int64_t)x * c2;
                for (int32_t k = 0; k < 3; ++k)
                    px[k] = yolo_f32_to_fp16(divide ? (float)s[x * 3 + k] / scale : (float)s[x * 3 + k] * mul);
                for (int32_t k = 3; k < c2; ++k) px[k] = 0;
            }
        }
        return need;
    }
    default:
        return -1;
    }
    (void)hw;
}
