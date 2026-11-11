/* L1 micro-benchmark and self-check for post-dfl-nms-neon.
 *
 * Synthetic fp16 outputs (NCHW and NC1HWC2), configurable hot-anchor fraction.
 * Reports per-call time of the NEON path and the portable reference path and
 * verifies that both return identical detections. Numbers on a non-RK3588
 * host only indicate direction; the board run is the measurement.
 *
 *   bench_post [repeats=200] [hot_fraction=0.02] [confidence=0.25]
 */
/* glibc hides clock_gettime/CLOCK_MONOTONIC under strict -std=c11 without this. */
#if !defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809L
#undef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "yolo_post.h"
#include "../common/fp16.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}
static uint32_t rng_state = 20261111u;
static float frand(void) { rng_state = rng_state * 1664525u + 1013904223u; return (float)(rng_state >> 8) / 16777216.0f; }
static float gauss(float mean, float sd) {
    float u = frand() + 1e-7f, v = frand();
    return mean + sd * sqrtf(-2.0f * logf(u)) * cosf(6.2831853f * v);
}
static int cmp_double(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }

typedef struct { uint16_t *reg, *cls, *sum; int side; } scale_buffers;

static void fill_scale(scale_buffers* b, int side, float hot_fraction, int logits) {
    size_t hw = (size_t)side * side;
    b->side = side;
    b->reg = malloc(hw * 64 * 2); b->cls = malloc(hw * 80 * 2); b->sum = malloc(hw * 2);
    for (size_t i = 0; i < hw * 64; ++i) b->reg[i] = yolo_f32_to_fp16(gauss(0.0f, 2.0f));
    for (size_t a = 0; a < hw; ++a) {
        int hot = frand() < hot_fraction;
        float boost = hot ? 4.0f + 6.0f * frand() : 0.0f;
        float total = 0.0f;
        for (int c = 0; c < 80; ++c) {
            float raw = gauss(-6.0f, 1.5f) + boost;
            float value = logits ? raw : yolo_sigmoid_f32(raw);
            uint16_t h = yolo_f32_to_fp16(value);
            b->cls[(size_t)c * hw + a] = h;
            total += yolo_fp16_to_f32(h);
        }
        if (total > 1.0f) total = 1.0f;
        b->sum[a] = yolo_f32_to_fp16(total);
    }
}

/* NCHW fp16 -> NC1HWC2 fp16 with channel blocks of 8 and width padded to 8 */
static uint16_t* to_nc1hwc2(const uint16_t* src, int channels, int side, int* w_stride_out) {
    int c1 = (channels + 7) / 8, wp = (side + 7) / 8 * 8;
    uint16_t* dst = calloc((size_t)c1 * side * wp * 8, 2);
    for (int c = 0; c < channels; ++c)
        for (int y = 0; y < side; ++y)
            for (int x = 0; x < side; ++x)
                dst[((((size_t)(c / 8) * side + y) * wp) + x) * 8 + (c % 8)] = src[((size_t)c * side + y) * side + x];
    *w_stride_out = wp;
    return dst;
}

static yolo_tensor_view view_nchw(const uint16_t* data, int channels, int side) {
    yolo_tensor_view v = { data, YOLO_LAYOUT_NCHW, YOLO_DTYPE_F16, channels, side, side, 0, 0 };
    return v;
}
static yolo_tensor_view view_blk(const uint16_t* data, int channels, int side, int wp) {
    yolo_tensor_view v = { data, YOLO_LAYOUT_NC1HWC2, YOLO_DTYPE_F16, channels, side, side, wp, 8 };
    return v;
}

static void report(const char* label, double* samples, int n, const yolo_post_stats* st, int dets) {
    qsort(samples, (size_t)n, sizeof(double), cmp_double);
    double sum = 0; for (int i = 0; i < n; ++i) sum += samples[i];
    printf("%-34s mean %7.3f ms  median %7.3f  p95 %7.3f  min %7.3f | candidates %d prefiltered %d dets %d (scan %.3f decode %.3f nms %.3f)\n",
           label, sum / n, samples[n / 2], samples[(int)((n - 1) * 0.95)], samples[0],
           st->candidates, st->prefiltered, dets, st->scan_ms, st->decode_ms, st->nms_ms);
}

int main(int argc, char** argv) {
    int repeats = argc > 1 ? atoi(argv[1]) : 200;
    float hot = argc > 2 ? (float)atof(argv[2]) : 0.02f;
    float conf = argc > 3 ? (float)atof(argv[3]) : 0.25f;
    printf("post-dfl-nms-neon bench: neon=%d repeats=%d hot_fraction=%.3f confidence=%.3f\n",
           yolo_post_has_neon(), repeats, hot, conf);
    yolo_post_params p = { 0, 80, 16, 640, conf, 0.45f, 300, 1, 0.0f, 80.0f, 1.28f, 1.28f, 500.0f, 375.0f };
    int sides[3] = { 80, 40, 20 };
    size_t ws_bytes = yolo_post_workspace_bytes(8400);
    void* ws = malloc(ws_bytes);
    yolo_detection* out = malloc(sizeof(yolo_detection) * 8400);
    yolo_detection* out_ref = malloc(sizeof(yolo_detection) * 8400);
    double* samples = malloc(sizeof(double) * (size_t)repeats);
    for (int logits = 0; logits < 2; ++logits) {
        rng_state = 20261111u;
        scale_buffers b[3];
        for (int s = 0; s < 3; ++s) fill_scale(&b[s], sides[s], hot, logits);
        p.cls_is_logits = logits;
        for (int layout = 0; layout < 2; ++layout) {
            for (int use_sum = 0; use_sum < (logits ? 1 : 2); ++use_sum) {
                yolo_post_scale scales[3];
                uint16_t* blocks[9] = {0};
                for (int s = 0; s < 3; ++s) {
                    scales[s].side = sides[s];
                    if (layout == 0) {
                        scales[s].reg = view_nchw(b[s].reg, 64, sides[s]);
                        scales[s].cls = view_nchw(b[s].cls, 80, sides[s]);
                        scales[s].sum = use_sum ? view_nchw(b[s].sum, 1, sides[s]) : (yolo_tensor_view){0};
                    } else {
                        int wp;
                        blocks[s * 3] = to_nc1hwc2(b[s].reg, 64, sides[s], &wp); scales[s].reg = view_blk(blocks[s * 3], 64, sides[s], wp);
                        blocks[s * 3 + 1] = to_nc1hwc2(b[s].cls, 80, sides[s], &wp); scales[s].cls = view_blk(blocks[s * 3 + 1], 80, sides[s], wp);
                        blocks[s * 3 + 2] = to_nc1hwc2(b[s].sum, 1, sides[s], &wp);
                        scales[s].sum = use_sum ? view_blk(blocks[s * 3 + 2], 1, sides[s], wp) : (yolo_tensor_view){0};
                    }
                }
                yolo_post_stats st = {0}, st_ref = {0};
                int n_ref = yolo_post_run_reference(scales, 3, &p, out_ref, 8400, ws, ws_bytes, &st_ref);
                int n = 0;
                for (int r = 0; r < repeats; ++r) {
                    double t0 = now_ms();
                    n = yolo_post_run(scales, 3, &p, out, 8400, ws, ws_bytes, &st);
                    samples[r] = now_ms() - t0;
                }
                int identical = n == n_ref && memcmp(out, out_ref, sizeof(yolo_detection) * (size_t)(n > 0 ? n : 0)) == 0;
                char label[128];
                snprintf(label, sizeof label, "%s %s%s neon", logits ? "logits6" : "canonical9",
                         layout ? "NC1HWC2" : "NCHW", use_sum ? "+sum" : "");
                report(label, samples, repeats, &st, n);
                for (int r = 0; r < repeats; ++r) {
                    double t0 = now_ms();
                    yolo_post_run_reference(scales, 3, &p, out_ref, 8400, ws, ws_bytes, &st_ref);
                    samples[r] = now_ms() - t0;
                }
                snprintf(label, sizeof label, "%s %s%s reference", logits ? "logits6" : "canonical9",
                         layout ? "NC1HWC2" : "NCHW", use_sum ? "+sum" : "");
                report(label, samples, repeats, &st_ref, n_ref);
                printf("    neon == reference: %s\n", identical ? "yes" : "NO");
                if (!identical) return 1;
                for (int i = 0; i < 9; ++i) free(blocks[i]);
            }
        }
        for (int s = 0; s < 3; ++s) { free(b[s].reg); free(b[s].cls); free(b[s].sum); }
    }
    free(ws); free(out); free(out_ref); free(samples);
    return 0;
}
