/* post-dfl-nms-neon: YOLOv8n detection post-processing on the RK3588 host CPU.
 *
 * Replaces benchmarks/yolov8n_general/scripts/yolo_common.py::postprocess and
 * the six-output branch of model_adapter.py::adapt_outputs with one pass that
 * reads the RKNN output buffers in place (fp16 NCHW / fp16 NC1HWC2 / fp32 NCHW):
 *
 *   1. class scan      : per anchor max/argmax over 80 classes (NEON), optional
 *                        exact prefilter by the clipped score-sum output;
 *   2. DFL decode      : softmax expectation over 4 x 16 bins for the survivors;
 *   3. class-wise NMS  : greedy, identical ordering rules to the numpy reference;
 *   4. restore         : letterbox -> original image coordinates, clipped.
 *
 * Numerics follow the Python reference (float32 everywhere, stable sigmoid for
 * logits). Ordering rules (scale-major, row-major candidates; class ascending;
 * stable sorts) are reproduced so that tie-breaking matches numpy.
 */
#ifndef YOLO_POST_H
#define YOLO_POST_H

#include <stddef.h>
#include <stdint.h>
#include "../common/tensor_view.h"

#ifdef __cplusplus
extern "C" {
#endif

#define YOLO_POST_VERSION 1

typedef struct yolo_post_scale {
    yolo_tensor_view reg;   /* 4*reg_max channels (64) */
    yolo_tensor_view cls;   /* num_classes channels (80): probabilities or logits */
    yolo_tensor_view sum;   /* optional 1-channel clipped probability sum; data NULL => absent */
    int32_t side;           /* 80 / 40 / 20 */
} yolo_post_scale;

typedef struct yolo_post_params {
    int32_t cls_is_logits;  /* 1 => cls holds logits, apply float32 sigmoid */
    int32_t num_classes;    /* 80 */
    int32_t reg_max;        /* 16 */
    int32_t input_size;     /* 640 */
    float confidence;       /* 0.25 */
    float nms_iou;          /* 0.45 */
    int32_t max_det;        /* 300 */
    int32_t restore;        /* 1 => map to original image and clip */
    float pad_left, pad_top, scale_x, scale_y;
    float original_width, original_height;
} yolo_post_params;

/* Same memory layout as one row of the numpy (N, 6) float32 result. */
typedef struct yolo_detection { float x1, y1, x2, y2, score, cls; } yolo_detection;

typedef struct yolo_post_stats {
    int32_t candidates;     /* anchors passing the confidence filter */
    int32_t kept;           /* detections returned */
    int32_t nonfinite;      /* non-finite values seen in cls scan / reg decode */
    int32_t prefiltered;    /* anchors skipped by the score-sum prefilter */
    double scan_ms, decode_ms, nms_ms, total_ms;
} yolo_post_stats;

/* Bytes of scratch memory needed for `total_anchors` (8400 for 640x640). */
size_t yolo_post_workspace_bytes(int32_t total_anchors);

/* Returns number of detections written (<= capacity), or a negative error code. */
int32_t yolo_post_run(const yolo_post_scale* scales, int32_t n_scales,
                      const yolo_post_params* params,
                      yolo_detection* out, int32_t capacity,
                      void* workspace, size_t workspace_bytes,
                      yolo_post_stats* stats);

/* Portable reference (no NEON) with identical semantics; used by the tests. */
int32_t yolo_post_run_reference(const yolo_post_scale* scales, int32_t n_scales,
                                const yolo_post_params* params,
                                yolo_detection* out, int32_t capacity,
                                void* workspace, size_t workspace_bytes,
                                yolo_post_stats* stats);

/* ---- layout / adapter helpers (also used by the bridge for canonical9 export) ---- */

/* Copy a view into dense float32 NCHW (batch 0). Returns number of elements. */
int64_t yolo_view_to_nchw_f32(const yolo_tensor_view* view, float* dst);

/* logits view -> float32 probabilities (NCHW) and clipped per-anchor sum (H*W),
 * i.e. the six-output -> canonical9 adaptation of model_adapter.py.
 * `probs` may be NULL if only the sum is wanted, `sum` may be NULL as well. */
int64_t yolo_sigmoid_sum(const yolo_tensor_view* logits, float* probs, float* sum);

/* Stable float32 sigmoid identical to model_adapter.stable_sigmoid. */
float yolo_sigmoid_f32(float x);

/* 1 if NEON code paths are compiled in. */
int32_t yolo_post_has_neon(void);

#ifdef __cplusplus
}
#endif
#endif
