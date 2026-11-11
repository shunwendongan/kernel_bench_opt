/* L1 micro-benchmark for pre-letterbox-neon: letterbox (resize + pad + BGR->RGB)
 * and the input packing into the RKNN input memory formats.
 *   bench_letterbox [repeats=100]
 * Non-RK3588 numbers indicate direction only.
 */
#include "letterbox.h"
#include "../common/fp16.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}
static int cmp_double(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }
static void report(const char* label, double* s, int n) {
    qsort(s, (size_t)n, sizeof(double), cmp_double);
    double sum = 0; for (int i = 0; i < n; ++i) sum += s[i];
    printf("%-44s mean %7.3f ms  median %7.3f  p95 %7.3f  min %7.3f\n", label, sum / n, s[n / 2], s[(int)((n - 1) * 0.95)], s[0]);
}

int main(int argc, char** argv) {
    int repeats = argc > 1 ? atoi(argv[1]) : 100;
    printf("pre-letterbox-neon bench: neon=%d repeats=%d\n", yolo_letterbox_has_neon(), repeats);
    int sizes[][2] = { {640, 480}, {500, 375}, {427, 640}, {1280, 720}, {1920, 1080} };
    double* samples = malloc(sizeof(double) * (size_t)repeats);
    uint8_t* rgb = malloc(640 * 640 * 3);
    size_t ws_bytes = yolo_letterbox_workspace_bytes(640, 1920);
    void* ws = malloc(ws_bytes);
    uint32_t seed = 7u;
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; ++i) {
        int w = sizes[i][0], h = sizes[i][1];
        uint8_t* bgr = malloc((size_t)w * h * 3);
        for (size_t k = 0; k < (size_t)w * h * 3; ++k) { seed = seed * 1664525u + 1013904223u; bgr[k] = (uint8_t)(seed >> 24); }
        yolo_letterbox_transform t;
        for (int r = 0; r < repeats; ++r) {
            double t0 = now_ms();
            yolo_letterbox_rgb_u8(bgr, w, h, (int64_t)w * 3, rgb, 640, YOLO_ROUND_OPENCV, &t, ws, ws_bytes);
            samples[r] = now_ms() - t0;
        }
        char label[96];
        snprintf(label, sizeof label, "letterbox %dx%d -> 640 (resized %dx%d)", w, h, t.resized_width, t.resized_height);
        report(label, samples, repeats);
        free(bgr);
    }
    struct { int fmt; int ws; int c2; const char* name; } packs[] = {
        { YOLO_INPUT_NHWC_F16, 640, 0, "pack NHWC fp16 (RKNN input attr)" },
        { YOLO_INPUT_NHWC_U8, 640, 0, "pack NHWC uint8 (legacy inputs_set)" },
        { YOLO_INPUT_NC1HWC2_F16, 640, 8, "pack NC1HWC2 fp16 c2=8" },
        { YOLO_INPUT_NCHW_F16, 640, 0, "pack NCHW fp16" },
    };
    void* dst = malloc((size_t)640 * 640 * 8 * 2);
    for (size_t i = 0; i < sizeof packs / sizeof packs[0]; ++i) {
        for (int r = 0; r < repeats; ++r) {
            double t0 = now_ms();
            yolo_input_pack(rgb, 640, packs[i].fmt, packs[i].ws, packs[i].c2, 1.0f, 0, dst, (size_t)640 * 640 * 8 * 2);
            samples[r] = now_ms() - t0;
        }
        report(packs[i].name, samples, repeats);
    }
    free(dst); free(ws); free(rgb); free(samples);
    return 0;
}
