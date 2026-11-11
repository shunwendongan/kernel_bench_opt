/* pre-letterbox-neon: YOLOv8n input preparation on the RK3588 host CPU.
 *
 * Replaces yolo_common.py::letterbox (cv2.resize INTER_LINEAR + copyMakeBorder
 * + BGR->RGB) and the uint8 -> model-input conversion that the RKNN runtime
 * performs inside rknn_inputs_set (13 ms on the first-round baseline).
 *
 * Geometry reproduces the Python arithmetic bit for bit (IEEE double,
 * round-half-even). The resize reproduces OpenCV's fixed-point INTER_LINEAR
 * for 8-bit images: 11-bit coefficients, int32 horizontal pass, raw (unclamped)
 * vertical coefficients with clamped row indices, and the 8-bit vertical
 * combine ((b0*(S0>>4))>>16 + (b1*(S1>>4))>>16 + 2) >> 2 that OpenCV's
 * VResizeLinear<uchar,...> specialisation applies to every element. The
 * `rounding` argument keeps two diagnostic rules so the board can confirm
 * which rule reproduces the frozen input hashes.
 */
#ifndef YOLO_LETTERBOX_H
#define YOLO_LETTERBOX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yolo_letterbox_transform {
    int32_t original_width, original_height;
    double ratio, scale_x, scale_y;
    int32_t pad_left, pad_top, pad_right, pad_bottom;
    int32_t resized_width, resized_height;
    int32_t size, fill;
} yolo_letterbox_transform;

enum yolo_resize_rounding {
    YOLO_ROUND_OPENCV = 0,         /* OpenCV VResizeLinear<uchar,int,short,...> rule on every element (default) */
    YOLO_ROUND_SCALAR22 = 1,       /* generic FixedPtCast<int,uchar,22> rule everywhere (diagnostic) */
    YOLO_ROUND_OPENCV_TAIL22 = 2   /* OpenCV rule on width&~15 elements, generic rule on the tail (diagnostic) */
};

enum yolo_input_format {
    YOLO_INPUT_NHWC_U8 = 0,
    YOLO_INPUT_NHWC_F16 = 1,
    YOLO_INPUT_NCHW_F16 = 2,
    YOLO_INPUT_NCHW_F32 = 3,
    YOLO_INPUT_NC1HWC2_F16 = 4
};

/* Same arithmetic as yolo_common.letterbox for the transform dictionary. */
void yolo_letterbox_geometry(int32_t width, int32_t height, int32_t size, yolo_letterbox_transform* t);

size_t yolo_letterbox_workspace_bytes(int32_t size, int32_t source_width);

/* BGR uint8 HWC source (row pitch in bytes) -> RGB uint8 HWC size x size.
 * Returns 0 on success, negative on invalid arguments. */
int32_t yolo_letterbox_rgb_u8(const uint8_t* bgr, int32_t width, int32_t height, int64_t pitch,
                              uint8_t* rgb_out, int32_t size, int32_t rounding,
                              yolo_letterbox_transform* t, void* workspace, size_t workspace_bytes);

/* RGB uint8 HWC (size x size x 3) -> model input memory.
 * w_stride: padded width in pixels (0 => size). c2: channel block for NC1HWC2.
 * scale/divide: value = divide ? pixel / scale : pixel * scale (1.0/0 for the
 * RKNN fp16 input; 255/1 reproduces the float32 ORT reference `rgb / 255`).
 * Returns bytes written or negative. */
int64_t yolo_input_pack(const uint8_t* rgb, int32_t size, int32_t format, int32_t w_stride, int32_t c2,
                        float scale, int32_t divide, void* dst, size_t dst_bytes);

/* Pure OpenCV fixed-point bilinear for one 3-channel 8-bit image (exposed for tests). */
int32_t yolo_resize_linear_u8c3(const uint8_t* src, int32_t sw, int32_t sh, int64_t pitch,
                                uint8_t* dst, int32_t dw, int32_t dh, int64_t dst_pitch,
                                int32_t rounding, void* workspace, size_t workspace_bytes);

int32_t yolo_letterbox_has_neon(void);

#ifdef __cplusplus
}
#endif
#endif
