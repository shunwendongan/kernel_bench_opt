/* post-dfl-nms-neon implementation. See yolo_post.h for the contract. */
/* glibc hides clock_gettime/CLOCK_MONOTONIC under strict -std=c11 without this. */
#if !defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809L
#undef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "yolo_post.h"
#include "../common/fp16.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if YOLO_HAVE_NEON && defined(__aarch64__)
#define YOLO_POST_NEON 1
#else
#define YOLO_POST_NEON 0
#endif

/* ------------------------------------------------------------------------ */
/* small utilities                                                           */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static inline float read_value(const yolo_tensor_view* v, int32_t c, int32_t y, int32_t x) {
    int64_t index = yolo_view_index(v, c, y, x);
    if (v->dtype == YOLO_DTYPE_F16) return yolo_fp16_to_f32(((const uint16_t*)v->data)[index]);
    return ((const float*)v->data)[index];
}

static inline int is_nonfinite(float v) { return !(v == v) || v == INFINITY || v == -INFINITY; }

float yolo_sigmoid_f32(float x) {
    /* model_adapter.stable_sigmoid: never exponentiate a positive number. */
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    float e = expf(x);
    return e / (1.0f + e);
}

int32_t yolo_post_has_neon(void) { return YOLO_POST_NEON; }

/* numpy pairwise_sum for exactly 16 contiguous float32 values. */
static inline float pairwise_sum16(const float* a) {
    float r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j] + a[j + 8];
    return ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
}

/* ------------------------------------------------------------------------ */
/* candidates                                                                */

typedef struct candidate {
    float x1, y1, x2, y2, score;
    int32_t cls;
    int32_t order;      /* scale-major, row-major rank */
    int32_t scale, y, x;
} candidate;

typedef struct workspace_layout {
    candidate* cands;
    int32_t* perm;
    int32_t* kept;
    int32_t capacity;
} workspace_layout;

size_t yolo_post_workspace_bytes(int32_t total_anchors) {
    size_t n = (size_t)(total_anchors > 0 ? total_anchors : 1);
    return n * sizeof(candidate) + 2 * n * sizeof(int32_t) + 64;
}

static int carve_workspace(void* ws, size_t bytes, int32_t total, workspace_layout* out) {
    if (!ws || bytes < yolo_post_workspace_bytes(total)) return 0;
    uintptr_t p = ((uintptr_t)ws + 15) & ~(uintptr_t)15;
    out->cands = (candidate*)p;
    p += (size_t)total * sizeof(candidate);
    out->perm = (int32_t*)p;
    p += (size_t)total * sizeof(int32_t);
    out->kept = (int32_t*)p;
    out->capacity = total;
    return 1;
}

/* ------------------------------------------------------------------------ */
/* class scan                                                                */

typedef struct scan_ctx {
    const yolo_post_params* p;
    candidate* cands;
    int32_t count, capacity;
    int32_t order;
    int32_t nonfinite, prefiltered;
} scan_ctx;

static inline void push_candidate(scan_ctx* ctx, int32_t scale, int32_t y, int32_t x,
                                  float max_value, int32_t cls) {
    float score = ctx->p->cls_is_logits ? yolo_sigmoid_f32(max_value) : max_value;
    ctx->order++;
    if (!(score >= ctx->p->confidence)) return;
    if (ctx->count >= ctx->capacity) return;
    candidate* c = &ctx->cands[ctx->count++];
    c->score = score; c->cls = cls; c->order = ctx->order - 1;
    c->scale = scale; c->y = y; c->x = x;
    c->x1 = c->y1 = c->x2 = c->y2 = 0.0f;
}

static void scan_scalar(scan_ctx* ctx, const yolo_post_scale* s, int32_t scale) {
    const yolo_post_params* p = ctx->p;
    const int use_sum = s->sum.data != NULL && !p->cls_is_logits;
    for (int32_t y = 0; y < s->side; ++y) {
        for (int32_t x = 0; x < s->side; ++x) {
            if (use_sum) {
                float total = read_value(&s->sum, 0, y, x);
                if (is_nonfinite(total)) {
                    ctx->nonfinite++;
                } else if (total < p->confidence) {
                    /* sum >= max of the probabilities, so max < confidence as well */
                    ctx->prefiltered++;
                    ctx->order++;
                    continue;
                }
            }
            float m = -INFINITY;
            int32_t mi = 0;
            for (int32_t c = 0; c < p->num_classes; ++c) {
                float v = read_value(&s->cls, c, y, x);
                if (is_nonfinite(v)) ctx->nonfinite++;
                if (v > m) { m = v; mi = c; }
            }
            push_candidate(ctx, scale, y, x, m, mi);
        }
    }
}

#if YOLO_POST_NEON
static inline uint32_t hsum_u32(uint32x4_t v) { return vaddvq_u32(v); }

static inline uint32x4_t nonfinite_mask(float32x4_t v) {
    uint32x4_t nan = vmvnq_u32(vceqq_f32(v, v));
    uint32x4_t inf = vceqq_f32(vabsq_f32(v), vdupq_n_f32(INFINITY));
    return vorrq_u32(nan, inf);
}

/* NCHW, fp16 or fp32: 4 anchors (x..x+3) per iteration, channel loop outside. */
static void scan_nchw_neon(scan_ctx* ctx, const yolo_post_scale* s, int32_t scale) {
    const yolo_post_params* p = ctx->p;
    const yolo_tensor_view* cls = &s->cls;
    const int use_sum = s->sum.data != NULL && !p->cls_is_logits;
    const int64_t ws = yolo_view_wstride(cls);
    const int64_t plane = (int64_t)cls->height * ws;
    const int64_t sum_ws = use_sum ? yolo_view_wstride(&s->sum) : 0;
    const float32x4_t conf = vdupq_n_f32(p->confidence);
    const uint32x4_t one = vdupq_n_u32(1);
    uint32x4_t nf_total = vdupq_n_u32(0);
    int32_t prefiltered = 0;
    const int32_t W = s->side;
    const int32_t W4 = W & ~3;
    for (int32_t y = 0; y < s->side; ++y) {
        for (int32_t x = 0; x < W4; x += 4) {
            uint32x4_t keep = vdupq_n_u32(0xFFFFFFFFu);
            if (use_sum) {
                float32x4_t total;
                if (s->sum.layout == YOLO_LAYOUT_NCHW) {
                    int64_t off = (int64_t)y * sum_ws + x;
                    total = s->sum.dtype == YOLO_DTYPE_F16
                        ? yolo_load4_fp16((const uint16_t*)s->sum.data + off)
                        : vld1q_f32((const float*)s->sum.data + off);
                } else {
                    float tmp[4];
                    for (int k = 0; k < 4; ++k) tmp[k] = read_value(&s->sum, 0, y, x + k);
                    total = vld1q_f32(tmp);
                }
                uint32x4_t nan = vmvnq_u32(vceqq_f32(total, total));
                nf_total = vaddq_u32(nf_total, vandq_u32(nan, one));
                keep = vorrq_u32(vcgeq_f32(total, conf), nan);
                uint32_t keep_lanes = hsum_u32(vandq_u32(keep, one));
                prefiltered += 4 - (int32_t)keep_lanes;
                if (keep_lanes == 0) { ctx->order += 4; continue; }
            }
            float32x4_t m = vdupq_n_f32(-INFINITY);
            uint32x4_t mi = vdupq_n_u32(0);
            uint32x4_t nf = vdupq_n_u32(0);
            int64_t base = (int64_t)y * ws + x;
            if (cls->dtype == YOLO_DTYPE_F16) {
                const uint16_t* d = (const uint16_t*)cls->data + base;
                for (int32_t c = 0; c < p->num_classes; ++c, d += plane) {
                    float32x4_t v = yolo_load4_fp16(d);
                    uint32x4_t gt = vcgtq_f32(v, m);
                    m = vbslq_f32(gt, v, m);
                    mi = vbslq_u32(gt, vdupq_n_u32((uint32_t)c), mi);
                    nf = vorrq_u32(nf, nonfinite_mask(v));
                    nf_total = vaddq_u32(nf_total, vandq_u32(nonfinite_mask(v), one));
                }
            } else {
                const float* d = (const float*)cls->data + base;
                for (int32_t c = 0; c < p->num_classes; ++c, d += plane) {
                    float32x4_t v = vld1q_f32(d);
                    uint32x4_t gt = vcgtq_f32(v, m);
                    m = vbslq_f32(gt, v, m);
                    mi = vbslq_u32(gt, vdupq_n_u32((uint32_t)c), mi);
                    nf_total = vaddq_u32(nf_total, vandq_u32(nonfinite_mask(v), one));
                }
            }
            float mv[4]; uint32_t miv[4]; uint32_t keepv[4];
            vst1q_f32(mv, m); vst1q_u32(miv, mi); vst1q_u32(keepv, keep);
            for (int k = 0; k < 4; ++k) {
                if (!keepv[k]) { ctx->order++; continue; }
                push_candidate(ctx, scale, y, x + k, mv[k], (int32_t)miv[k]);
            }
        }
        for (int32_t x = W4; x < W; ++x) {   /* tail (not hit for 80/40/20) */
            float m = -INFINITY; int32_t mi = 0;
            for (int32_t c = 0; c < p->num_classes; ++c) {
                float v = read_value(cls, c, y, x);
                if (is_nonfinite(v)) ctx->nonfinite++;
                if (v > m) { m = v; mi = c; }
            }
            push_candidate(ctx, scale, y, x, m, mi);
        }
    }
    ctx->nonfinite += (int32_t)hsum_u32(nf_total);
    ctx->prefiltered += prefiltered;
}

/* NC1HWC2 fp16 with C2 == 8: one anchor per iteration, 8 channels per load. */
static void scan_nc1hwc2_neon(scan_ctx* ctx, const yolo_post_scale* s, int32_t scale) {
    const yolo_post_params* p = ctx->p;
    const yolo_tensor_view* cls = &s->cls;
    const int use_sum = s->sum.data != NULL && !p->cls_is_logits;
    const int64_t ws = yolo_view_wstride(cls);
    const int32_t c2 = cls->c2;                      /* == 8 */
    const int32_t blocks = (p->num_classes + c2 - 1) / c2;
    const int64_t block_stride = (int64_t)cls->height * ws * c2;
    const uint32x4_t one = vdupq_n_u32(1);
    const uint32x4_t lane_lo = {0u, 1u, 2u, 3u};
    const uint32x4_t lane_hi = {4u, 5u, 6u, 7u};
    uint32x4_t nf_total = vdupq_n_u32(0);
    for (int32_t y = 0; y < s->side; ++y) {
        for (int32_t x = 0; x < s->side; ++x) {
            if (use_sum) {
                float total = read_value(&s->sum, 0, y, x);
                if (is_nonfinite(total)) {
                    ctx->nonfinite++;
                } else if (total < p->confidence) {
                    ctx->prefiltered++; ctx->order++; continue;
                }
            }
            const uint16_t* d = (const uint16_t*)cls->data + ((int64_t)y * ws + x) * c2;
            float32x4_t m_lo = vdupq_n_f32(-INFINITY), m_hi = vdupq_n_f32(-INFINITY);
            uint32x4_t i_lo = lane_lo, i_hi = lane_hi;
            for (int32_t b = 0; b < blocks; ++b, d += block_stride) {
                float32x4_t v_lo = yolo_load4_fp16(d);
                float32x4_t v_hi = yolo_load4_fp16(d + 4);
                uint32x4_t idx_lo = vaddq_u32(vdupq_n_u32((uint32_t)(b * c2)), lane_lo);
                uint32x4_t idx_hi = vaddq_u32(vdupq_n_u32((uint32_t)(b * c2)), lane_hi);
                uint32x4_t gt_lo = vcgtq_f32(v_lo, m_lo), gt_hi = vcgtq_f32(v_hi, m_hi);
                m_lo = vbslq_f32(gt_lo, v_lo, m_lo); i_lo = vbslq_u32(gt_lo, idx_lo, i_lo);
                m_hi = vbslq_f32(gt_hi, v_hi, m_hi); i_hi = vbslq_u32(gt_hi, idx_hi, i_hi);
                nf_total = vaddq_u32(nf_total, vandq_u32(nonfinite_mask(v_lo), one));
                nf_total = vaddq_u32(nf_total, vandq_u32(nonfinite_mask(v_hi), one));
            }
            float mv[8]; uint32_t iv[8];
            vst1q_f32(mv, m_lo); vst1q_f32(mv + 4, m_hi);
            vst1q_u32(iv, i_lo); vst1q_u32(iv + 4, i_hi);
            float m = -INFINITY; int32_t mi = 0;
            for (int k = 0; k < 8; ++k) {
                if ((int32_t)iv[k] >= p->num_classes) continue;   /* padded channels */
                if (mv[k] > m || (mv[k] == m && (int32_t)iv[k] < mi)) { m = mv[k]; mi = (int32_t)iv[k]; }
            }
            push_candidate(ctx, scale, y, x, m, mi);
        }
    }
    ctx->nonfinite += (int32_t)hsum_u32(nf_total);
}
#endif

/* ------------------------------------------------------------------------ */
/* DFL decode                                                                */

static int decode_candidates(candidate* cands, int32_t count, const yolo_post_scale* scales,
                             const yolo_post_params* p, int32_t* nonfinite) {
    const int32_t rm = p->reg_max;
    if (rm != 16) return -2;      /* pairwise_sum16 mirrors numpy for 16 bins */
    for (int32_t i = 0; i < count; ++i) {
        candidate* c = &cands[i];
        const yolo_post_scale* s = &scales[c->scale];
        const float stride = (float)p->input_size / (float)s->side;
        float ltrb[4];
        for (int side = 0; side < 4; ++side) {
            float d[16], e[16], w[16];
            float m = -INFINITY;
            for (int b = 0; b < 16; ++b) {
                d[b] = read_value(&s->reg, side * 16 + b, c->y, c->x);
                if (is_nonfinite(d[b])) (*nonfinite)++;
                if (d[b] > m) m = d[b];
            }
            for (int b = 0; b < 16; ++b) e[b] = expf(d[b] - m);
            float total = pairwise_sum16(e);
            for (int b = 0; b < 16; ++b) w[b] = (e[b] / total) * (float)b;
            ltrb[side] = pairwise_sum16(w);
        }
        /* numpy: center is float64 (int + 0.5), boxes float64, cast to float32 once */
        double cx = (double)c->x + 0.5, cy = (double)c->y + 0.5;
        c->x1 = (float)((cx - (double)ltrb[0]) * (double)stride);
        c->y1 = (float)((cy - (double)ltrb[1]) * (double)stride);
        c->x2 = (float)((cx + (double)ltrb[2]) * (double)stride);
        c->y2 = (float)((cy + (double)ltrb[3]) * (double)stride);
    }
    return 0;
}

/* ------------------------------------------------------------------------ */
/* NMS                                                                       */

static const candidate* g_sort_cands;

static int cmp_class_score(const void* a, const void* b) {
    const candidate* ca = &g_sort_cands[*(const int32_t*)a];
    const candidate* cb = &g_sort_cands[*(const int32_t*)b];
    if (ca->cls != cb->cls) return ca->cls < cb->cls ? -1 : 1;
    if (ca->score != cb->score) return ca->score > cb->score ? -1 : 1;   /* descending */
    return ca->order < cb->order ? -1 : (ca->order > cb->order);
}

typedef struct kept_key { int32_t cand; int32_t position; } kept_key;
static const candidate* g_kept_cands;
static int cmp_kept(const void* a, const void* b) {
    const kept_key* ka = (const kept_key*)a; const kept_key* kb = (const kept_key*)b;
    float sa = g_kept_cands[ka->cand].score, sb = g_kept_cands[kb->cand].score;
    if (sa != sb) return sa > sb ? -1 : 1;
    return ka->position < kb->position ? -1 : (ka->position > kb->position);
}

static inline float box_area(const candidate* c) {
    float w = c->x2 - c->x1, h = c->y2 - c->y1;
    if (w < 0.0f) w = 0.0f;
    if (h < 0.0f) h = 0.0f;
    return w * h;
}

static inline float iou_f32(const candidate* a, const candidate* b, float area_a, float area_b) {
    float x1 = a->x1 > b->x1 ? a->x1 : b->x1;
    float y1 = a->y1 > b->y1 ? a->y1 : b->y1;
    float x2 = a->x2 < b->x2 ? a->x2 : b->x2;
    float y2 = a->y2 < b->y2 ? a->y2 : b->y2;
    float w = x2 - x1, h = y2 - y1;
    if (w < 0.0f) w = 0.0f;
    if (h < 0.0f) h = 0.0f;
    float inter = w * h;
    float denominator = area_a + area_b - inter;
    if (denominator < 1e-12f) denominator = 1e-12f;
    return inter / denominator;
}

static int32_t nms(candidate* cands, int32_t count, int32_t* perm, int32_t* kept, const yolo_post_params* p) {
    for (int32_t i = 0; i < count; ++i) perm[i] = i;
    g_sort_cands = cands;
    qsort(perm, (size_t)count, sizeof(int32_t), cmp_class_score);
    int32_t n_kept = 0;
    int32_t start = 0;
    while (start < count) {
        int32_t cls = cands[perm[start]].cls;
        int32_t end = start;
        while (end < count && cands[perm[end]].cls == cls) ++end;
        /* greedy within [start, end): perm already score-descending, order-stable */
        for (int32_t i = start; i < end; ++i) {
            if (perm[i] < 0) continue;
            const candidate* a = &cands[perm[i]];
            kept[n_kept++] = perm[i];
            float area_a = box_area(a);
            for (int32_t j = i + 1; j < end; ++j) {
                if (perm[j] < 0) continue;
                const candidate* b = &cands[perm[j]];
                if (iou_f32(a, b, area_a, box_area(b)) > p->nms_iou) perm[j] = -1;
            }
        }
        start = end;
    }
    return n_kept;
}

/* ------------------------------------------------------------------------ */
/* driver                                                                    */

static int32_t run_impl(const yolo_post_scale* scales, int32_t n_scales, const yolo_post_params* p,
                        yolo_detection* out, int32_t capacity, void* workspace, size_t workspace_bytes,
                        yolo_post_stats* stats, int use_neon) {
    yolo_post_stats local = {0};
    double t0 = now_ms();
    if (!scales || !p || !out || n_scales <= 0 || p->num_classes <= 0) return -1;
    int32_t total = 0;
    for (int32_t i = 0; i < n_scales; ++i) total += scales[i].side * scales[i].side;
    workspace_layout ws;
    if (!carve_workspace(workspace, workspace_bytes, total, &ws)) return -3;

    scan_ctx ctx = {p, ws.cands, 0, total, 0, 0, 0};
    for (int32_t i = 0; i < n_scales; ++i) {
        const yolo_post_scale* s = &scales[i];
        if (s->cls.channels < p->num_classes || s->reg.channels < 4 * p->reg_max) return -4;
        if (s->cls.height != s->side || s->cls.width != s->side) return -4;
#if YOLO_POST_NEON
        if (use_neon && s->cls.layout == YOLO_LAYOUT_NCHW) { scan_nchw_neon(&ctx, s, i); continue; }
        if (use_neon && s->cls.layout == YOLO_LAYOUT_NC1HWC2 && s->cls.dtype == YOLO_DTYPE_F16 && s->cls.c2 == 8) {
            scan_nc1hwc2_neon(&ctx, s, i); continue;
        }
#else
        (void)use_neon;
#endif
        scan_scalar(&ctx, s, i);
    }
    local.scan_ms = now_ms() - t0;
    local.candidates = ctx.count;
    local.nonfinite = ctx.nonfinite;
    local.prefiltered = ctx.prefiltered;

    double t1 = now_ms();
    int rc = decode_candidates(ws.cands, ctx.count, scales, p, &local.nonfinite);
    if (rc) return rc;
    local.decode_ms = now_ms() - t1;

    double t2 = now_ms();
    int32_t n_kept = nms(ws.cands, ctx.count, ws.perm, ws.kept, p);
    /* final: stable sort by score descending over the kept list, then max_det */
    kept_key* keys = (kept_key*)ws.perm;   /* perm no longer needed; same size class */
    for (int32_t i = 0; i < n_kept; ++i) { keys[i].cand = ws.kept[i]; keys[i].position = i; }
    g_kept_cands = ws.cands;
    qsort(keys, (size_t)n_kept, sizeof(kept_key), cmp_kept);
    int32_t n_out = n_kept < p->max_det ? n_kept : p->max_det;
    if (n_out > capacity) n_out = capacity;
    for (int32_t i = 0; i < n_out; ++i) {
        const candidate* c = &ws.cands[keys[i].cand];
        float x1 = c->x1, y1 = c->y1, x2 = c->x2, y2 = c->y2;
        if (p->restore) {
            x1 = (x1 - p->pad_left) / p->scale_x; x2 = (x2 - p->pad_left) / p->scale_x;
            y1 = (y1 - p->pad_top) / p->scale_y;  y2 = (y2 - p->pad_top) / p->scale_y;
            if (x1 < 0.0f) x1 = 0.0f; if (x1 > p->original_width) x1 = p->original_width;
            if (x2 < 0.0f) x2 = 0.0f; if (x2 > p->original_width) x2 = p->original_width;
            if (y1 < 0.0f) y1 = 0.0f; if (y1 > p->original_height) y1 = p->original_height;
            if (y2 < 0.0f) y2 = 0.0f; if (y2 > p->original_height) y2 = p->original_height;
        }
        out[i].x1 = x1; out[i].y1 = y1; out[i].x2 = x2; out[i].y2 = y2;
        out[i].score = c->score; out[i].cls = (float)c->cls;
    }
    local.nms_ms = now_ms() - t2;
    local.kept = n_out;
    local.total_ms = now_ms() - t0;
    if (stats) *stats = local;
    return n_out;
}

int32_t yolo_post_run(const yolo_post_scale* scales, int32_t n_scales, const yolo_post_params* params,
                      yolo_detection* out, int32_t capacity, void* workspace, size_t workspace_bytes,
                      yolo_post_stats* stats) {
    return run_impl(scales, n_scales, params, out, capacity, workspace, workspace_bytes, stats, 1);
}

int32_t yolo_post_run_reference(const yolo_post_scale* scales, int32_t n_scales, const yolo_post_params* params,
                                yolo_detection* out, int32_t capacity, void* workspace, size_t workspace_bytes,
                                yolo_post_stats* stats) {
    return run_impl(scales, n_scales, params, out, capacity, workspace, workspace_bytes, stats, 0);
}

/* ------------------------------------------------------------------------ */
/* layout / adapter helpers                                                   */

int64_t yolo_view_to_nchw_f32(const yolo_tensor_view* v, float* dst) {
    int64_t n = 0;
#if YOLO_POST_NEON
    if (v->layout == YOLO_LAYOUT_NCHW && v->dtype == YOLO_DTYPE_F16) {
        const int64_t ws = yolo_view_wstride(v);
        for (int32_t c = 0; c < v->channels; ++c) {
            for (int32_t y = 0; y < v->height; ++y) {
                const uint16_t* row = (const uint16_t*)v->data + ((int64_t)c * v->height + y) * ws;
                int32_t x = 0;
                for (; x + 4 <= v->width; x += 4) vst1q_f32(dst + n + x, yolo_load4_fp16(row + x));
                for (; x < v->width; ++x) dst[n + x] = yolo_fp16_to_f32(row[x]);
                n += v->width;
            }
        }
        return n;
    }
#endif
    for (int32_t c = 0; c < v->channels; ++c)
        for (int32_t y = 0; y < v->height; ++y)
            for (int32_t x = 0; x < v->width; ++x) dst[n++] = read_value(v, c, y, x);
    return n;
}

#if YOLO_POST_NEON
/* Cephes-style expf on 4 lanes; |rel err| ~1e-7, clamps to avoid inf/denormal garbage. */
static inline float32x4_t expf_f32x4(float32x4_t x) {
    const float32x4_t hi = vdupq_n_f32(88.3762626647949f), lo = vdupq_n_f32(-88.3762626647949f);
    x = vminq_f32(vmaxq_f32(x, lo), hi);
    float32x4_t fx = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
    float32x4_t r = vmlsq_f32(x, fx, vdupq_n_f32(0.693359375f));
    r = vmlsq_f32(r, fx, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t y = vdupq_n_f32(1.9875691500E-4f);
    y = vmlaq_f32(vdupq_n_f32(1.3981999507E-3f), y, r);
    y = vmlaq_f32(vdupq_n_f32(8.3334519073E-3f), y, r);
    y = vmlaq_f32(vdupq_n_f32(4.1665795894E-2f), y, r);
    y = vmlaq_f32(vdupq_n_f32(1.6666665459E-1f), y, r);
    y = vmlaq_f32(vdupq_n_f32(5.0000001201E-1f), y, r);
    y = vmlaq_f32(vaddq_f32(r, vdupq_n_f32(1.0f)), y, vmulq_f32(r, r));
    int32x4_t e = vcvtq_s32_f32(fx);
    e = vshlq_n_s32(vaddq_s32(e, vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(e));
}

static inline float32x4_t sigmoid_f32x4(float32x4_t x) {
    float32x4_t e = expf_f32x4(vnegq_f32(vabsq_f32(x)));       /* exp(-|x|) in (0, 1] */
    float32x4_t denominator = vaddq_f32(vdupq_n_f32(1.0f), e);
    float32x4_t pos = vdivq_f32(vdupq_n_f32(1.0f), denominator);
    float32x4_t neg = vdivq_f32(e, denominator);
    return vbslq_f32(vcgeq_f32(x, vdupq_n_f32(0.0f)), pos, neg);
}
#endif

int64_t yolo_sigmoid_sum(const yolo_tensor_view* logits, float* probs, float* sum) {
    const int32_t H = logits->height, W = logits->width, C = logits->channels;
    const int64_t hw = (int64_t)H * W;
    if (sum) memset(sum, 0, sizeof(float) * (size_t)hw);
#if YOLO_POST_NEON
    if (logits->layout == YOLO_LAYOUT_NCHW && logits->dtype == YOLO_DTYPE_F16) {
        const int64_t ws = yolo_view_wstride(logits);
        for (int32_t c = 0; c < C; ++c) {
            for (int32_t y = 0; y < H; ++y) {
                const uint16_t* row = (const uint16_t*)logits->data + ((int64_t)c * H + y) * ws;
                float* prow = probs ? probs + ((int64_t)c * H + y) * W : NULL;
                float* srow = sum ? sum + (int64_t)y * W : NULL;
                int32_t x = 0;
                for (; x + 4 <= W; x += 4) {
                    float32x4_t pv = sigmoid_f32x4(yolo_load4_fp16(row + x));
                    if (prow) vst1q_f32(prow + x, pv);
                    if (srow) vst1q_f32(srow + x, vaddq_f32(vld1q_f32(srow + x), pv));
                }
                for (; x < W; ++x) {
                    float pv = yolo_sigmoid_f32(yolo_fp16_to_f32(row[x]));
                    if (prow) prow[x] = pv;
                    if (srow) srow[x] += pv;
                }
            }
        }
    } else
#endif
    {
        for (int32_t c = 0; c < C; ++c)
            for (int32_t y = 0; y < H; ++y)
                for (int32_t x = 0; x < W; ++x) {
                    float pv = yolo_sigmoid_f32(read_value(logits, c, y, x));
                    if (probs) probs[((int64_t)c * H + y) * W + x] = pv;
                    if (sum) sum[(int64_t)y * W + x] += pv;
                }
    }
    if (sum) for (int64_t i = 0; i < hw; ++i) { if (sum[i] < 0.0f) sum[i] = 0.0f; if (sum[i] > 1.0f) sum[i] = 1.0f; }
    return (int64_t)C * hw;
}
