/* Minimal description of an RKNN output buffer as seen by the host kernels.
 *
 * RKNN zero-copy outputs arrive either as NCHW (optionally with a padded
 * width stride) or in the NPU native NC1HWC2 layout (channels grouped in
 * blocks of C2, fp16 => C2 = 8 on RK3588, read from the native attribute).
 * Element types: fp16 (RKNN FLOAT16) or fp32 (legacy want_float path / ORT).
 */
#ifndef YOLO_KERNELS_TENSOR_VIEW_H
#define YOLO_KERNELS_TENSOR_VIEW_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum yolo_layout { YOLO_LAYOUT_NCHW = 0, YOLO_LAYOUT_NC1HWC2 = 1 };
enum yolo_dtype { YOLO_DTYPE_F16 = 0, YOLO_DTYPE_F32 = 1 };

typedef struct yolo_tensor_view {
    const void* data;      /* first element (batch 0) */
    int32_t layout;        /* enum yolo_layout */
    int32_t dtype;         /* enum yolo_dtype */
    int32_t channels;      /* logical C */
    int32_t height;        /* H */
    int32_t width;         /* W */
    int32_t w_stride;      /* padded W (elements); 0 => width */
    int32_t c2;            /* channel block for NC1HWC2; ignored for NCHW */
} yolo_tensor_view;

static inline int32_t yolo_view_wstride(const yolo_tensor_view* v) {
    return v->w_stride > 0 ? v->w_stride : v->width;
}

/* Linear element index of (c, y, x) in the buffer for either layout. */
static inline int64_t yolo_view_index(const yolo_tensor_view* v, int32_t c, int32_t y, int32_t x) {
    int64_t ws = yolo_view_wstride(v);
    if (v->layout == YOLO_LAYOUT_NC1HWC2) {
        int64_t c1 = c / v->c2, c2 = c % v->c2;
        return (((c1 * v->height + y) * ws) + x) * v->c2 + c2;
    }
    return ((int64_t)c * v->height + y) * ws + x;
}

#ifdef __cplusplus
}
#endif
#endif
