// rknn_bridge_v2: zero-copy RKNN session for YOLOv8n with host NEON pre/post kernels.
//
// Differences from benchmarks/yolov8n_general/scripts/rknn_bridge.cpp (v1):
//   * io_mode 1 (zero-copy): input and output tensors live in rknn_create_mem buffers
//     bound once with rknn_set_io_mem. The letterboxed image is packed by
//     pre-letterbox-neon straight into the input buffer in the model's own format
//     (fp16 NHWC or native NC1HWC2), so rknn_inputs_set and its uint8->fp16 copy
//     disappear. Outputs are consumed in place as fp16 (native layout), so the
//     fp16->fp32 conversion of rknn_outputs_get(want_float) disappears as well.
//   * io_mode 0 (legacy) reproduces the v1 calls for A/B comparisons in one binary.
//   * post-dfl-nms-neon runs on the output buffers (6 logits outputs or 9 canonical).
//   * core_mask is a parameter (1/2/4/3/7/0 auto) instead of a hard-coded 7.
// The ABI exposes only pointers, fixed-width integers and doubles (see rknn_bridge_v2.py).
#include "rknn_api.h"
#include "../../kernels/post-dfl-nms-neon/yolo_post.h"
#include "../../kernels/pre-letterbox-neon/letterbox.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct V2Timing {
    double letterbox_ms, pack_ms, inputs_set_ms, run_ms, outputs_get_ms, outputs_release_ms,
           perf_query_ms, post_ms, export_ms;
    int64_t perf_run_us;
};
struct V2OutputView {
    const void* data; uint64_t bytes;
    int32_t layout, dtype, channels, height, width, w_stride, c2;
};
struct V2Detection { float x1, y1, x2, y2, score, cls; };

namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void check(int status, const char* api) {
    if (status != RKNN_SUCC)
        throw std::runtime_error(std::string(api) + " returned " + std::to_string(status));
}
void error_text(char* error, size_t capacity, const std::string& message) {
    if (error && capacity) std::snprintf(error, capacity, "%s", message.c_str());
}
std::string current_exception() {
    try { throw; }
    catch (const std::exception& e) { return e.what(); }
    catch (...) { return "unknown C++ exception"; }
}
std::string quoted(const char* data, size_t capacity) {
    std::ostringstream out;
    out << '"';
    for (size_t i = 0; i < capacity && data[i]; ++i) {
        unsigned char c = static_cast<unsigned char>(data[i]);
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else out << c;
    }
    out << '"';
    return out.str();
}
std::string attr_json(const rknn_tensor_attr& a) {
    std::ostringstream out;
    out << std::setprecision(9) << "{\"index\":" << a.index << ",\"name\":" << quoted(a.name, sizeof(a.name))
        << ",\"n_dims\":" << a.n_dims << ",\"dims\":[";
    for (uint32_t d = 0; d < a.n_dims && d < RKNN_MAX_DIMS; ++d) { if (d) out << ','; out << a.dims[d]; }
    out << "],\"n_elems\":" << a.n_elems << ",\"size\":" << a.size
        << ",\"fmt\":" << quoted(get_format_string(a.fmt), 64) << ",\"fmt_id\":" << a.fmt
        << ",\"type\":" << quoted(get_type_string(a.type), 64) << ",\"type_id\":" << a.type
        << ",\"qnt_type\":" << quoted(get_qnt_type_string(a.qnt_type), 64) << ",\"zp\":" << a.zp
        << ",\"scale\":" << a.scale << ",\"w_stride\":" << a.w_stride << ",\"h_stride\":" << a.h_stride
        << ",\"size_with_stride\":" << a.size_with_stride << ",\"pass_through\":" << unsigned(a.pass_through) << '}';
    return out.str();
}
std::string attrs_json(const std::vector<rknn_tensor_attr>& attrs) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < attrs.size(); ++i) { if (i) out << ','; out << attr_json(attrs[i]); }
    out << ']';
    return out.str();
}
int elem_bytes(rknn_tensor_type type) {
    switch (type) {
    case RKNN_TENSOR_FLOAT16: return 2;
    case RKNN_TENSOR_FLOAT32: return 4;
    case RKNN_TENSOR_UINT8: case RKNN_TENSOR_INT8: return 1;
    default: return 0;
    }
}

struct OutputSlot {
    rknn_tensor_attr plain{};      // RKNN_QUERY_OUTPUT_ATTR
    rknn_tensor_attr native{};     // RKNN_QUERY_NATIVE_OUTPUT_ATTR (may fail -> native_ok=false)
    bool native_ok = false;
    rknn_tensor_attr bound{};      // attr passed to rknn_set_io_mem
    rknn_tensor_mem* mem = nullptr;
    V2OutputView view{};
};

struct Session {
    rknn_context context = 0;
    bool detail = false;
    uint32_t io_mode = 1, input_mode = 0, output_mode = 0, mem_flags = 0, core_mask = 7;
    rknn_tensor_attr input_plain{}, input_native{}, input_nhwc{};
    bool input_native_ok = false, input_nhwc_ok = false;
    rknn_tensor_attr input_bound{};
    rknn_tensor_mem* input_mem = nullptr;
    int32_t pack_format = YOLO_INPUT_NHWC_F16, pack_w_stride = 0, pack_c2 = 0;
    std::vector<OutputSlot> outputs;
    std::string metadata;
    // host buffers
    std::vector<uint8_t> rgb;                 // letterboxed RGB uint8 HWC 640x640
    bool rgb_valid = false;
    std::vector<uint8_t> letterbox_ws;
    std::vector<uint8_t> post_ws;
    std::vector<rknn_output> legacy_outputs;  // io_mode 0
    bool legacy_pending = false, poisoned = false;
    yolo_letterbox_transform transform{};
    int32_t input_size = 640;
};

void query_attr(rknn_context ctx, rknn_query_cmd cmd, uint32_t index, rknn_tensor_attr& attr, const char* label) {
    std::memset(&attr, 0, sizeof attr);
    attr.index = index;
    check(rknn_query(ctx, cmd, &attr, sizeof attr), label);
    if (!attr.n_dims || attr.n_dims > RKNN_MAX_DIMS) throw std::runtime_error(std::string(label) + ": bad n_dims");
}
bool try_query_attr(rknn_context ctx, rknn_query_cmd cmd, uint32_t index, rknn_tensor_attr& attr) {
    std::memset(&attr, 0, sizeof attr);
    attr.index = index;
    return rknn_query(ctx, cmd, &attr, sizeof attr) == RKNN_SUCC && attr.n_dims && attr.n_dims <= RKNN_MAX_DIMS;
}

rknn_tensor_mem* create_mem(Session& s, uint32_t size, bool is_input) {
    rknn_tensor_mem* mem = nullptr;
    if (is_input && (s.mem_flags & 1u)) mem = rknn_create_mem2(s.context, size, RKNN_FLAG_MEMORY_NON_CACHEABLE);
    else mem = rknn_create_mem(s.context, size);
    if (!mem || !mem->virt_addr) throw std::runtime_error("rknn_create_mem failed");
    return mem;
}

// Derive the host view (layout, logical dims, strides) of a bound output buffer.
V2OutputView make_view(const rknn_tensor_attr& plain, const rknn_tensor_attr& bound, const rknn_tensor_mem* mem) {
    V2OutputView v{};
    v.data = mem->virt_addr;
    v.bytes = mem->size;
    int eb = elem_bytes(bound.type);
    if (eb != 2 && eb != 4) throw std::runtime_error("unsupported output element type for the host kernels");
    v.dtype = eb == 2 ? YOLO_DTYPE_F16 : YOLO_DTYPE_F32;
    if (plain.n_dims != 4) throw std::runtime_error("outputs must be rank-4 NCHW tensors");
    // plain attr is NCHW for this model family; accept NHWC plain attrs defensively
    if (plain.fmt == RKNN_TENSOR_NHWC) { v.channels = int32_t(plain.dims[3]); v.height = int32_t(plain.dims[1]); v.width = int32_t(plain.dims[2]); }
    else { v.channels = int32_t(plain.dims[1]); v.height = int32_t(plain.dims[2]); v.width = int32_t(plain.dims[3]); }
    if (bound.fmt == RKNN_TENSOR_NC1HWC2) {
        if (bound.n_dims != 5) throw std::runtime_error("NC1HWC2 attr must have 5 dims");
        v.layout = YOLO_LAYOUT_NC1HWC2;
        v.c2 = int32_t(bound.dims[4]);
        int32_t c1 = int32_t(bound.dims[1]);
        uint64_t row_bytes = uint64_t(c1) * uint64_t(v.height) * uint64_t(v.c2) * uint64_t(eb);
        v.w_stride = bound.w_stride ? int32_t(bound.w_stride)
                     : (row_bytes && bound.size_with_stride % row_bytes == 0 ? int32_t(bound.size_with_stride / row_bytes) : v.width);
    } else if (bound.fmt == RKNN_TENSOR_NCHW) {
        v.layout = YOLO_LAYOUT_NCHW;
        v.c2 = 0;
        uint64_t row_bytes = uint64_t(v.channels) * uint64_t(v.height) * uint64_t(eb);
        v.w_stride = bound.w_stride ? int32_t(bound.w_stride)
                     : (row_bytes && bound.size_with_stride % row_bytes == 0 ? int32_t(bound.size_with_stride / row_bytes) : v.width);
    } else {
        throw std::runtime_error("unsupported bound output format: " + std::string(get_format_string(bound.fmt)));
    }
    if (v.w_stride < v.width) throw std::runtime_error("derived w_stride smaller than width");
    return v;
}

void setup_zero_copy(Session& s) {
    // ---- input ----
    bool want_native = s.input_mode == 2 || (s.input_mode == 0 && s.input_native_ok);
    if (want_native && !s.input_native_ok) throw std::runtime_error("native input attr unavailable");
    if (want_native) {
        s.input_bound = s.input_native;
        s.input_bound.pass_through = 1;
        int eb = elem_bytes(s.input_bound.type);
        if (s.input_bound.fmt == RKNN_TENSOR_NC1HWC2) {
            if (s.input_bound.n_dims != 5 || eb != 2) throw std::runtime_error("unsupported NC1HWC2 input (need fp16, 5 dims)");
            s.pack_format = YOLO_INPUT_NC1HWC2_F16;
            s.pack_c2 = int32_t(s.input_bound.dims[4]);
            s.input_size = int32_t(s.input_bound.dims[2]);
            uint64_t row = uint64_t(s.input_bound.dims[1]) * s.input_bound.dims[2] * s.pack_c2 * eb;
            s.pack_w_stride = s.input_bound.w_stride ? int32_t(s.input_bound.w_stride)
                              : (row && s.input_bound.size_with_stride % row == 0 ? int32_t(s.input_bound.size_with_stride / row) : s.input_size);
        } else if (s.input_bound.fmt == RKNN_TENSOR_NHWC) {
            s.input_size = int32_t(s.input_bound.dims[1]);
            if (s.input_bound.dims[3] != 3) throw std::runtime_error("NHWC input must have 3 channels");
            s.pack_w_stride = s.input_bound.w_stride ? int32_t(s.input_bound.w_stride) : s.input_size;
            if (eb == 2) s.pack_format = YOLO_INPUT_NHWC_F16;
            else if (eb == 1) s.pack_format = YOLO_INPUT_NHWC_U8;
            else throw std::runtime_error("unsupported NHWC input element type");
        } else if (s.input_bound.fmt == RKNN_TENSOR_NCHW) {
            s.input_size = int32_t(s.input_bound.dims[2]);
            s.pack_w_stride = s.input_bound.w_stride ? int32_t(s.input_bound.w_stride) : s.input_size;
            if (eb == 2) s.pack_format = YOLO_INPUT_NCHW_F16;
            else if (eb == 4) s.pack_format = YOLO_INPUT_NCHW_F32;
            else throw std::runtime_error("unsupported NCHW input element type");
        } else {
            throw std::runtime_error("unsupported native input format");
        }
    } else {
        // declare uint8 NHWC, let the runtime convert inside rknn_run (still zero-copy memory)
        s.input_bound = s.input_plain;
        s.input_bound.type = RKNN_TENSOR_UINT8;
        s.input_bound.fmt = RKNN_TENSOR_NHWC;
        s.input_bound.pass_through = 0;
        s.input_size = int32_t(s.input_plain.fmt == RKNN_TENSOR_NHWC ? s.input_plain.dims[1] : s.input_plain.dims[2]);
        s.pack_format = YOLO_INPUT_NHWC_U8;
        s.pack_w_stride = s.input_plain.w_stride ? int32_t(s.input_plain.w_stride) : s.input_size;
        s.input_bound.size = uint32_t(s.input_size) * uint32_t(s.pack_w_stride) * 3u;
        s.input_bound.size_with_stride = s.input_bound.size;
    }
    if (s.input_size != 640) throw std::runtime_error("this bridge expects a 640x640 input");
    uint32_t in_bytes = s.input_bound.size_with_stride ? s.input_bound.size_with_stride : s.input_bound.size;
    s.input_mem = create_mem(s, in_bytes, true);
    std::memset(s.input_mem->virt_addr, 0, in_bytes);
    check(rknn_set_io_mem(s.context, s.input_mem, &s.input_bound), "rknn_set_io_mem(input)");
    // ---- outputs ----
    for (auto& slot : s.outputs) {
        bool use_native = s.output_mode == 0 && slot.native_ok;
        slot.bound = use_native ? slot.native : slot.plain;
        if (!use_native && slot.bound.type != RKNN_TENSOR_FLOAT16 && slot.bound.type != RKNN_TENSOR_FLOAT32)
            slot.bound.type = RKNN_TENSOR_FLOAT16;
        uint32_t bytes = slot.bound.size_with_stride ? slot.bound.size_with_stride : slot.bound.size;
        slot.mem = create_mem(s, bytes, false);
        check(rknn_set_io_mem(s.context, slot.mem, &slot.bound), "rknn_set_io_mem(output)");
        slot.view = make_view(slot.plain, slot.bound, slot.mem);
    }
}

void build_metadata(Session& s, uint32_t flags, double init_ms, Clock::time_point start) {
    rknn_sdk_version v{};
    check(rknn_query(s.context, RKNN_QUERY_SDK_VERSION, &v, sizeof v), "rknn_query(SDK_VERSION)");
    rknn_mem_size m{};
    check(rknn_query(s.context, RKNN_QUERY_MEM_SIZE, &m, sizeof m), "rknn_query(MEM_SIZE)");
    std::ostringstream out;
    out << std::setprecision(17) << "{\"schema_version\":2,\"bridge\":\"rknn_bridge_v2\",\"n_input\":1,\"n_output\":" << s.outputs.size()
        << ",\"detail\":" << (s.detail ? "true" : "false") << ",\"core_mask\":" << s.core_mask
        << ",\"io_mode\":" << s.io_mode << ",\"input_mode\":" << s.input_mode << ",\"output_mode\":" << s.output_mode
        << ",\"mem_flags\":" << s.mem_flags << ",\"init_flags\":" << flags << ",\"init_ms\":" << init_ms
        << ",\"create_ms\":" << elapsed_ms(start)
        << ",\"sdk_version\":{\"api_version\":" << quoted(v.api_version, sizeof v.api_version)
        << ",\"drv_version\":" << quoted(v.drv_version, sizeof v.drv_version) << "}"
        << ",\"mem_size\":{\"total_weight_size\":" << m.total_weight_size << ",\"total_internal_size\":" << m.total_internal_size
        << ",\"total_dma_allocated_size\":" << m.total_dma_allocated_size << ",\"total_sram_size\":" << m.total_sram_size
        << ",\"free_sram_size\":" << m.free_sram_size << "}"
        << ",\"inputs\":[" << attr_json(s.input_plain) << "]"
        << ",\"input_native\":" << (s.input_native_ok ? attr_json(s.input_native) : "null")
        << ",\"input_native_nhwc\":" << (s.input_nhwc_ok ? attr_json(s.input_nhwc) : "null");
    std::vector<rknn_tensor_attr> plain, native, bound;
    for (auto& o : s.outputs) { plain.push_back(o.plain); if (o.native_ok) native.push_back(o.native); bound.push_back(o.bound); }
    out << ",\"outputs\":" << attrs_json(plain) << ",\"outputs_native\":" << attrs_json(native);
    if (s.io_mode == 1) {
        out << ",\"input_bound\":" << attr_json(s.input_bound) << ",\"outputs_bound\":" << attrs_json(bound)
            << ",\"pack_format\":" << s.pack_format << ",\"pack_w_stride\":" << s.pack_w_stride << ",\"pack_c2\":" << s.pack_c2
            << ",\"output_views\":[";
        for (size_t i = 0; i < s.outputs.size(); ++i) {
            const auto& vw = s.outputs[i].view;
            if (i) out << ',';
            out << "{\"layout\":" << vw.layout << ",\"dtype\":" << vw.dtype << ",\"channels\":" << vw.channels
                << ",\"height\":" << vw.height << ",\"width\":" << vw.width << ",\"w_stride\":" << vw.w_stride
                << ",\"c2\":" << vw.c2 << ",\"bytes\":" << vw.bytes << '}';
        }
        out << ']';
    }
    out << ",\"host_kernels\":{\"post_neon\":" << yolo_post_has_neon() << ",\"letterbox_neon\":" << yolo_letterbox_has_neon() << "}}";
    s.metadata = out.str();
}

void release_legacy(Session& s, V2Timing* timing) {
    if (!s.legacy_pending) return;
    s.legacy_pending = false;
    auto start = Clock::now();
    int status = rknn_outputs_release(s.context, uint32_t(s.legacy_outputs.size()), s.legacy_outputs.data());
    if (timing) timing->outputs_release_ms = elapsed_ms(start);
    if (status != RKNN_SUCC) { s.poisoned = true; check(status, "rknn_outputs_release"); }
}

void destroy_session(Session& s, std::string& message) {
    try { release_legacy(s, nullptr); } catch (...) { message += current_exception(); }
    for (auto& o : s.outputs) if (o.mem) { rknn_destroy_mem(s.context, o.mem); o.mem = nullptr; }
    if (s.input_mem) { rknn_destroy_mem(s.context, s.input_mem); s.input_mem = nullptr; }
    if (s.context) {
        int status = rknn_destroy(s.context);
        s.context = 0;
        if (status != RKNN_SUCC) message += "; rknn_destroy returned " + std::to_string(status);
    }
}

yolo_tensor_view as_kernel_view(const V2OutputView& v) {
    yolo_tensor_view k{};
    k.data = v.data; k.layout = v.layout; k.dtype = v.dtype; k.channels = v.channels;
    k.height = v.height; k.width = v.width; k.w_stride = v.w_stride; k.c2 = v.c2;
    return k;
}
}  // namespace

extern "C" int rknn_bridge_v2_create(const char* model_path, uint32_t detail, uint32_t core_mask, uint32_t io_mode,
                                     uint32_t input_mode, uint32_t output_mode, uint32_t mem_flags,
                                     void** result, char* error, size_t capacity) noexcept {
    if (result) *result = nullptr;
    std::unique_ptr<Session> s;
    try {
        auto start = Clock::now();
        if (!model_path || !result) throw std::runtime_error("null create argument");
        if (io_mode > 1 || input_mode > 2 || output_mode > 1) throw std::runtime_error("invalid io/input/output mode");
        if (!(core_mask == 0 || core_mask == 1 || core_mask == 2 || core_mask == 3 || core_mask == 4 || core_mask == 7 || core_mask == 0xffff))
            throw std::runtime_error("core_mask must be one of 0(auto),1,2,3,4,7,0xffff");
        s = std::make_unique<Session>();
        s->detail = detail != 0; s->io_mode = io_mode; s->input_mode = input_mode; s->output_mode = output_mode;
        s->mem_flags = mem_flags; s->core_mask = core_mask;
        uint32_t flags = detail ? RKNN_FLAG_COLLECT_PERF_MASK : 0;
        auto init_start = Clock::now();
        check(rknn_init(&s->context, const_cast<char*>(model_path), 0, flags, nullptr), "rknn_init");
        double init_ms = elapsed_ms(init_start);
        if (core_mask != 0) check(rknn_set_core_mask(s->context, static_cast<rknn_core_mask>(core_mask)), "rknn_set_core_mask");
        rknn_input_output_num num{};
        check(rknn_query(s->context, RKNN_QUERY_IN_OUT_NUM, &num, sizeof num), "rknn_query(IN_OUT_NUM)");
        if (num.n_input != 1 || (num.n_output != 9 && num.n_output != 6))
            throw std::runtime_error("expected 1 input and 6 or 9 outputs");
        query_attr(s->context, RKNN_QUERY_INPUT_ATTR, 0, s->input_plain, "rknn_query(INPUT_ATTR)");
        s->input_native_ok = try_query_attr(s->context, RKNN_QUERY_NATIVE_INPUT_ATTR, 0, s->input_native);
        s->input_nhwc_ok = try_query_attr(s->context, RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR, 0, s->input_nhwc);
        s->outputs.resize(num.n_output);
        for (uint32_t i = 0; i < num.n_output; ++i) {
            query_attr(s->context, RKNN_QUERY_OUTPUT_ATTR, i, s->outputs[i].plain, "rknn_query(OUTPUT_ATTR)");
            s->outputs[i].native_ok = try_query_attr(s->context, RKNN_QUERY_NATIVE_OUTPUT_ATTR, i, s->outputs[i].native);
        }
        if (io_mode == 1) setup_zero_copy(*s);
        else {
            s->input_size = int32_t(s->input_plain.fmt == RKNN_TENSOR_NHWC ? s->input_plain.dims[1] : s->input_plain.dims[2]);
            if (s->input_size != 640) throw std::runtime_error("this bridge expects a 640x640 input");
            for (auto& o : s->outputs) o.bound = o.plain;
        }
        s->rgb.assign(size_t(640) * 640 * 3, 0);
        s->letterbox_ws.assign(yolo_letterbox_workspace_bytes(640, 4096), 0);
        s->post_ws.assign(yolo_post_workspace_bytes(8400), 0);
        build_metadata(*s, flags, init_ms, start);
        *result = s.release();
        return 0;
    } catch (...) {
        std::string message = current_exception();
        if (s) destroy_session(*s, message);
        error_text(error, capacity, message);
        return -1;
    }
}

extern "C" int rknn_bridge_v2_metadata_json(void* handle, const char** result, char* error, size_t capacity) noexcept {
    if (!handle || !result) { error_text(error, capacity, "null metadata argument"); return -1; }
    *result = static_cast<Session*>(handle)->metadata.c_str();
    return 0;
}

extern "C" int rknn_bridge_v2_destroy(void* handle, char* error, size_t capacity) noexcept {
    std::unique_ptr<Session> s(static_cast<Session*>(handle));
    std::string message;
    if (s) destroy_session(*s, message);
    if (message.empty()) return 0;
    error_text(error, capacity, message);
    return -1;
}

namespace {
void pack_input(Session& s, V2Timing* timing) {
    if (s.io_mode != 1) return;   // legacy mode submits the uint8 buffer in run()
    auto start = Clock::now();
    int64_t n = yolo_input_pack(s.rgb.data(), s.input_size, s.pack_format, s.pack_w_stride, s.pack_c2, 1.0f, 0,
                                s.input_mem->virt_addr, s.input_mem->size);
    if (timing) timing->pack_ms = elapsed_ms(start);
    if (n < 0) throw std::runtime_error("yolo_input_pack failed with " + std::to_string(n));
}
}

extern "C" int rknn_bridge_v2_prepare_bgr(void* handle, const uint8_t* bgr, int32_t width, int32_t height, int64_t pitch,
                                          int32_t rounding, yolo_letterbox_transform* transform, V2Timing* timing,
                                          char* error, size_t capacity) noexcept {
    auto* s = static_cast<Session*>(handle);
    if (timing) *timing = {};
    try {
        if (!s || !bgr || width <= 0 || height <= 0) throw std::runtime_error("null/invalid prepare_bgr argument");
        if (s->legacy_pending || s->poisoned) throw std::runtime_error("session has pending/failed output release");
        auto start = Clock::now();
        int rc = yolo_letterbox_rgb_u8(bgr, width, height, pitch ? pitch : int64_t(width) * 3, s->rgb.data(), s->input_size,
                                       rounding, &s->transform, s->letterbox_ws.data(), s->letterbox_ws.size());
        if (timing) timing->letterbox_ms = elapsed_ms(start);
        if (rc) throw std::runtime_error("yolo_letterbox_rgb_u8 failed with " + std::to_string(rc));
        s->rgb_valid = true;
        if (transform) *transform = s->transform;
        pack_input(*s, timing);
        return 0;
    } catch (...) { error_text(error, capacity, current_exception()); return -1; }
}

extern "C" int rknn_bridge_v2_prepare_rgb(void* handle, const uint8_t* rgb, size_t bytes, V2Timing* timing,
                                          char* error, size_t capacity) noexcept {
    auto* s = static_cast<Session*>(handle);
    if (timing) *timing = {};
    try {
        if (!s || !rgb) throw std::runtime_error("null prepare_rgb argument");
        if (s->legacy_pending || s->poisoned) throw std::runtime_error("session has pending/failed output release");
        if (bytes != s->rgb.size()) throw std::runtime_error("rgb bytes must be 640*640*3");
        std::memcpy(s->rgb.data(), rgb, bytes);
        s->rgb_valid = true;
        pack_input(*s, timing);
        return 0;
    } catch (...) { error_text(error, capacity, current_exception()); return -1; }
}

extern "C" int rknn_bridge_v2_letterboxed_rgb(void* handle, const uint8_t** rgb, size_t* bytes, char* error, size_t capacity) noexcept {
    auto* s = static_cast<Session*>(handle);
    if (!s || !rgb || !bytes || !s->rgb_valid) { error_text(error, capacity, "no letterboxed image available"); return -1; }
    *rgb = s->rgb.data(); *bytes = s->rgb.size();
    return 0;
}

extern "C" int rknn_bridge_v2_run(void* handle, const char* perf_path, V2Timing* timing, char* error, size_t capacity) noexcept {
    auto* s = static_cast<Session*>(handle);
    try {
        if (!s || !timing) throw std::runtime_error("null run argument");
        if (!s->rgb_valid) throw std::runtime_error("prepare an input before run");
        if (s->legacy_pending || s->poisoned) throw std::runtime_error("session has pending/failed output release");
        if (s->detail != bool(perf_path && perf_path[0])) throw std::runtime_error("detail mode requires perf_path; baseline forbids it");
        if (s->io_mode == 0) {
            rknn_input input{};
            input.index = 0; input.buf = s->rgb.data(); input.size = uint32_t(s->rgb.size());
            input.pass_through = 0; input.type = RKNN_TENSOR_UINT8; input.fmt = RKNN_TENSOR_NHWC;
            auto start = Clock::now();
            int status = rknn_inputs_set(s->context, 1, &input);
            timing->inputs_set_ms = elapsed_ms(start);
            check(status, "rknn_inputs_set");
        }
        auto start = Clock::now();
        int status = rknn_run(s->context, nullptr);
        timing->run_ms = elapsed_ms(start);
        check(status, "rknn_run");
        if (s->io_mode == 0) {
            s->legacy_outputs.assign(s->outputs.size(), rknn_output{});
            for (uint32_t i = 0; i < s->legacy_outputs.size(); ++i) { s->legacy_outputs[i].index = i; s->legacy_outputs[i].want_float = 1; }
            start = Clock::now();
            status = rknn_outputs_get(s->context, uint32_t(s->legacy_outputs.size()), s->legacy_outputs.data(), nullptr);
            timing->outputs_get_ms = elapsed_ms(start);
            s->legacy_pending = status == RKNN_SUCC;
            for (const auto& o : s->legacy_outputs) if (o.buf) s->legacy_pending = true;
            check(status, "rknn_outputs_get");
            for (size_t i = 0; i < s->outputs.size(); ++i) {
                const auto& raw = s->legacy_outputs[i];
                uint64_t expected = uint64_t(s->outputs[i].plain.n_elems) * sizeof(float);
                if (!raw.buf || raw.index != i || raw.size != expected) throw std::runtime_error("output buffer size/index mismatch at " + std::to_string(i));
                V2OutputView v{};
                v.data = raw.buf; v.bytes = raw.size; v.layout = YOLO_LAYOUT_NCHW; v.dtype = YOLO_DTYPE_F32;
                const auto& p = s->outputs[i].plain;
                v.channels = int32_t(p.dims[1]); v.height = int32_t(p.dims[2]); v.width = int32_t(p.dims[3]); v.w_stride = v.width; v.c2 = 0;
                s->outputs[i].view = v;
            }
        }
        start = Clock::now();
        if (s->detail) {
            rknn_perf_detail report{};
            check(rknn_query(s->context, RKNN_QUERY_PERF_DETAIL, &report, sizeof report), "rknn_query(PERF_DETAIL)");
            if (!report.perf_data || !report.data_len) throw std::runtime_error("empty PERF_DETAIL report");
            std::ofstream file(perf_path, std::ios::binary | std::ios::trunc);
            if (!file) throw std::runtime_error("cannot open PERF_DETAIL path");
            file.write(report.perf_data, std::streamsize(report.data_len));
            file.close();
            if (!file) throw std::runtime_error("cannot save PERF_DETAIL report");
        }
        rknn_perf_run perf{};
        check(rknn_query(s->context, RKNN_QUERY_PERF_RUN, &perf, sizeof perf), "rknn_query(PERF_RUN)");
        timing->perf_query_ms = elapsed_ms(start);
        timing->perf_run_us = perf.run_duration;
        return 0;
    } catch (...) {
        std::string message = current_exception();
        if (s) { try { release_legacy(*s, timing); } catch (...) { message += "; " + current_exception(); } }
        error_text(error, capacity, message);
        return -1;
    }
}

extern "C" int rknn_bridge_v2_output_count(void* handle) noexcept {
    auto* s = static_cast<Session*>(handle);
    return s ? int(s->outputs.size()) : -1;
}

extern "C" int rknn_bridge_v2_output_view(void* handle, uint32_t index, V2OutputView* view, char* error, size_t capacity) noexcept {
    auto* s = static_cast<Session*>(handle);
    if (!s || !view || index >= s->outputs.size()) { error_text(error, capacity, "invalid output index"); return -1; }
    if (s->io_mode == 0 && !s->legacy_pending) { error_text(error, capacity, "legacy outputs not held; call run first"); return -1; }
    *view = s->outputs[index].view;
    return 0;
}

extern "C" int rknn_bridge_v2_output_f32(void* handle, uint32_t index, float* dst, uint64_t capacity_elems,
                                         V2Timing* timing, char* error, size_t capacity) noexcept {
    auto* s = static_cast<Session*>(handle);
    try {
        if (!s || !dst || index >= s->outputs.size()) throw std::runtime_error("invalid output_f32 argument");
        if (s->io_mode == 0 && !s->legacy_pending) throw std::runtime_error("legacy outputs not held; call run first");
        const auto& v = s->outputs[index].view;
        uint64_t need = uint64_t(v.channels) * v.height * v.width;
        if (capacity_elems < need) throw std::runtime_error("output_f32 destination too small");
        auto start = Clock::now();
        yolo_tensor_view k = as_kernel_view(v);
        yolo_view_to_nchw_f32(&k, dst);
        if (timing) timing->export_ms += elapsed_ms(start);
        return 0;
    } catch (...) { error_text(error, capacity, current_exception()); return -1; }
}

extern "C" int rknn_bridge_v2_postprocess(void* handle, const yolo_post_params* params, V2Detection* out, int32_t out_capacity,
                                          int32_t* count, yolo_post_stats* stats, V2Timing* timing,
                                          char* error, size_t capacity) noexcept {
    auto* s = static_cast<Session*>(handle);
    if (count) *count = 0;
    try {
        if (!s || !params || !out || !count) throw std::runtime_error("null postprocess argument");
        if (s->io_mode == 0 && !s->legacy_pending) throw std::runtime_error("legacy outputs not held; call run first");
        auto start = Clock::now();
        const int per_scale = s->outputs.size() == 6 ? 2 : 3;
        yolo_post_scale scales[3];
        for (int i = 0; i < 3; ++i) {
            const auto& reg = s->outputs[size_t(i * per_scale)].view;
            const auto& cls = s->outputs[size_t(i * per_scale + 1)].view;
            if (reg.channels != 64 || cls.channels != 80 || reg.height != cls.height || reg.height != reg.width)
                throw std::runtime_error("output views do not match the YOLOv8n head contract");
            scales[i].reg = as_kernel_view(reg);
            scales[i].cls = as_kernel_view(cls);
            scales[i].sum = yolo_tensor_view{};
            if (per_scale == 3) scales[i].sum = as_kernel_view(s->outputs[size_t(i * 3 + 2)].view);
            scales[i].side = reg.height;
        }
        yolo_post_params p = *params;
        p.cls_is_logits = per_scale == 2 ? 1 : 0;
        p.num_classes = 80; p.reg_max = 16; p.input_size = s->input_size;
        yolo_post_stats local{};
        int32_t n = yolo_post_run(scales, 3, &p, reinterpret_cast<yolo_detection*>(out), out_capacity,
                                  s->post_ws.data(), s->post_ws.size(), &local);
        if (n < 0) throw std::runtime_error("yolo_post_run failed with " + std::to_string(n));
        if (local.nonfinite) throw std::runtime_error("model outputs contain non-finite values: " + std::to_string(local.nonfinite));
        if (timing) timing->post_ms = elapsed_ms(start);
        if (stats) *stats = local;
        *count = n;
        return 0;
    } catch (...) { error_text(error, capacity, current_exception()); return -1; }
}

extern "C" int rknn_bridge_v2_release(void* handle, V2Timing* timing, char* error, size_t capacity) noexcept {
    try {
        if (!handle) throw std::runtime_error("null release handle");
        release_legacy(*static_cast<Session*>(handle), timing);
        return 0;
    } catch (...) { error_text(error, capacity, current_exception()); return -1; }
}
