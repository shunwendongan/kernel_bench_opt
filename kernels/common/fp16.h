/* fp16 (IEEE binary16) <-> fp32 helpers shared by the YOLOv8n host kernels.
 *
 * The scalar conversions are bit-exact software implementations (denormals,
 * infinities and NaNs included) so that the portable path and the NEON path
 * produce identical float32 values. On AArch64 the NEON path uses the
 * hardware FCVTL/FCVTN instructions, which are also exact (RNE for fp32->fp16).
 */
#ifndef YOLO_KERNELS_FP16_H
#define YOLO_KERNELS_FP16_H

#include <stdint.h>
#include <string.h>

#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && !defined(YOLO_FORCE_SCALAR)
#include <arm_neon.h>
#define YOLO_HAVE_NEON 1
#else
#define YOLO_HAVE_NEON 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

static inline float yolo_bits_to_f32(uint32_t bits) {
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static inline uint32_t yolo_f32_to_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    return bits;
}

/* Exact binary16 -> binary32 (sign/exponent/mantissa widening). */
static inline float yolo_fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x3FFu;
    if (exponent == 0) {
        if (mantissa == 0) return yolo_bits_to_f32(sign);
        /* subnormal: value = mantissa * 2^-24 (exact in float32) */
        float v = (float)mantissa * 5.9604644775390625e-8f;
        return sign ? -v : v;
    }
    if (exponent == 31) {
        return yolo_bits_to_f32(sign | 0x7F800000u | (mantissa << 13));
    }
    return yolo_bits_to_f32(sign | ((exponent + 112u) << 23) | (mantissa << 13));
}

/* binary32 -> binary16 with round-to-nearest-even; overflow -> inf. */
static inline uint16_t yolo_f32_to_fp16(float value) {
    uint32_t bits = yolo_f32_to_bits(value);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = (int32_t)((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;
    if (((bits >> 23) & 0xFFu) == 0xFFu) {  /* inf or nan */
        return (uint16_t)(sign | 0x7C00u | (mantissa ? 0x200u | (mantissa >> 13) : 0u));
    }
    if (exponent >= 31) return (uint16_t)(sign | 0x7C00u);
    if (exponent <= 0) {
        if (exponent < -10) return (uint16_t)sign;
        mantissa |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exponent);
        uint32_t half = mantissa >> shift;
        uint32_t remainder = mantissa & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half & 1u))) half++;
        return (uint16_t)(sign | half);
    }
    uint32_t half = (uint32_t)(exponent << 10) | (mantissa >> 13);
    uint32_t remainder = mantissa & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) half++;
    return (uint16_t)(sign | half);
}

#if YOLO_HAVE_NEON
/* Load 4 fp16 values and widen to float32x4 (exact). */
static inline float32x4_t yolo_load4_fp16(const uint16_t* p) {
    return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p)));
}
/* Convert 4 float32 to fp16 (RNE) and store. */
static inline void yolo_store4_fp16(uint16_t* p, float32x4_t v) {
    vst1_u16(p, vreinterpret_u16_f16(vcvt_f16_f32(v)));
}
#endif

#ifdef __cplusplus
}
#endif
#endif
