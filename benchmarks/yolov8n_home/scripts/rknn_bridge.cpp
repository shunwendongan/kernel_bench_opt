// RKNN ABI 只在 C++ 中引用真实头文件，Python 仅使用本文件的窄 C 接口。
#include "rknn_api.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// 本桥接自有 ABI：只含指针、定宽整数与 double，不向 Python 暴露 RKNN 结构。
struct BridgeOutput { const float* data; uint64_t n_elems; uint64_t size_bytes; };
struct BridgeTiming {
    double inputs_set_ms, run_ms, outputs_get_ms, outputs_release_ms, perf_query_ms;
    int64_t perf_run_us;
};

namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
struct Session {
    rknn_context context = 0;
    bool detail = false;
    std::vector<rknn_tensor_attr> inputs, outputs;
    std::string metadata;
    std::vector<rknn_output> raw_outputs;
    std::vector<BridgeOutput> views;
    bool pending = false, poisoned = false;
};
void check(int status, const char* api) {
    if (status != RKNN_SUCC)
        throw std::runtime_error(std::string(api) + " returned " + std::to_string(status));
}
void error_text(char* error, size_t capacity, const std::string& message) {
    if (error && capacity) std::snprintf(error, capacity, "%s", message.c_str());
}
std::string current_exception() {
    try { throw; }
    catch (const std::exception& error) { return error.what(); }
    catch (...) { return "unknown C++ exception"; }
}
void release_outputs(Session& s, BridgeTiming* timing) {
    if (!s.pending) return;
    s.pending = false;  // 失败时也不重试释放未知状态的缓冲区。
    auto start = Clock::now();
    int status = rknn_outputs_release(s.context, uint32_t(s.raw_outputs.size()), s.raw_outputs.data());
    if (timing) timing->outputs_release_ms = elapsed_ms(start);
    if (status != RKNN_SUCC) s.poisoned = true;
    check(status, "rknn_outputs_release");
}
std::string quoted(const char* data, size_t capacity) {
    std::ostringstream out;
    out << '"';
    for (size_t i = 0; i < capacity && data[i]; ++i) {
        unsigned char c = static_cast<unsigned char>(data[i]);
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4)
                            << std::setfill('0') << unsigned(c) << std::dec;
        else out << c;
    }
    out << '"';
    return out.str();
}
void validate_attr(const rknn_tensor_attr& a) {
    if (!a.n_dims || a.n_dims > RKNN_MAX_DIMS)
        throw std::runtime_error("tensor n_dims outside supported range");
    uint64_t n = 1;
    for (uint32_t i = 0; i < a.n_dims; ++i) {
        if (!a.dims[i] || n > std::numeric_limits<uint32_t>::max() / a.dims[i])
            throw std::runtime_error("tensor dims contain zero or overflow");
        n *= a.dims[i];
    }
    if (n != a.n_elems) throw std::runtime_error("tensor dims/n_elems mismatch");
    if (!std::isfinite(a.scale)) throw std::runtime_error("tensor scale is not finite");
}
std::string attrs_json(const std::vector<rknn_tensor_attr>& attrs) {
    std::ostringstream out;
    out << '[' << std::setprecision(9);
    for (size_t i = 0; i < attrs.size(); ++i) {
        const auto& a = attrs[i];
        if (i) out << ',';
        out << "{\"index\":" << a.index << ",\"name\":" << quoted(a.name, sizeof(a.name))
            << ",\"n_dims\":" << a.n_dims << ",\"dims\":[";
        for (uint32_t d = 0; d < a.n_dims; ++d) { if (d) out << ','; out << a.dims[d]; }
        out << "],\"n_elems\":" << a.n_elems << ",\"size\":" << a.size
            << ",\"fmt\":" << quoted(get_format_string(a.fmt), 64) << ",\"fmt_id\":" << a.fmt
            << ",\"type\":" << quoted(get_type_string(a.type), 64) << ",\"type_id\":" << a.type
            << ",\"qnt_type\":" << quoted(get_qnt_type_string(a.qnt_type), 64)
            << ",\"qnt_type_id\":" << a.qnt_type << ",\"fl\":" << int(a.fl)
            << ",\"zp\":" << a.zp << ",\"scale\":" << a.scale
            << ",\"w_stride\":" << a.w_stride << ",\"h_stride\":" << a.h_stride
            << ",\"size_with_stride\":" << a.size_with_stride
            << ",\"pass_through\":" << unsigned(a.pass_through) << '}';
    }
    out << ']';
    return out.str();
}
void query_metadata(Session& s, uint32_t flags, double init_ms, Clock::time_point start) {
    rknn_input_output_num num{};
    check(rknn_query(s.context, RKNN_QUERY_IN_OUT_NUM, &num, sizeof(num)), "rknn_query(IN_OUT_NUM)");
    if (!num.n_input || !num.n_output || num.n_input > 4096 || num.n_output > 4096)
        throw std::runtime_error("invalid input/output count");
    s.inputs.resize(num.n_input); s.outputs.resize(num.n_output);
    for (uint32_t i = 0; i < num.n_input; ++i) {
        auto& a = s.inputs[i]; a.index = i;
        check(rknn_query(s.context, RKNN_QUERY_INPUT_ATTR, &a, sizeof(a)), "rknn_query(INPUT_ATTR)");
        if (a.index != i) throw std::runtime_error("input index mismatch");
        validate_attr(a);
    }
    for (uint32_t i = 0; i < num.n_output; ++i) {
        auto& a = s.outputs[i]; a.index = i;
        check(rknn_query(s.context, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a)), "rknn_query(OUTPUT_ATTR)");
        if (a.index != i) throw std::runtime_error("output index mismatch");
        validate_attr(a);
    }
    rknn_sdk_version v{};
    check(rknn_query(s.context, RKNN_QUERY_SDK_VERSION, &v, sizeof(v)), "rknn_query(SDK_VERSION)");
    rknn_mem_size m{};
    check(rknn_query(s.context, RKNN_QUERY_MEM_SIZE, &m, sizeof(m)), "rknn_query(MEM_SIZE)");
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\"schema_version\":1,\"n_input\":" << num.n_input << ",\"n_output\":" << num.n_output
        << ",\"detail\":" << (s.detail ? "true" : "false")
        << ",\"core_mask\":7,\"init_flags\":" << flags
        << ",\"init_ms\":" << init_ms << ",\"create_ms\":" << elapsed_ms(start)
        << ",\"sdk_version\":{\"api_version\":" << quoted(v.api_version, sizeof(v.api_version))
        << ",\"drv_version\":" << quoted(v.drv_version, sizeof(v.drv_version))
        << "},\"mem_size\":{\"total_weight_size\":" << m.total_weight_size
        << ",\"total_internal_size\":" << m.total_internal_size
        << ",\"total_dma_allocated_size\":" << m.total_dma_allocated_size
        << ",\"total_sram_size\":" << m.total_sram_size << ",\"free_sram_size\":" << m.free_sram_size
        << "},\"inputs\":" << attrs_json(s.inputs) << ",\"outputs\":" << attrs_json(s.outputs) << '}';
    s.metadata = out.str();
}
}

extern "C" int rknn_bridge_create(const char* model_path, uint32_t detail,
                                  uint32_t core_mask, void** result,
                                  char* error, size_t capacity) noexcept {
    if (result) *result = nullptr;
    std::unique_ptr<Session> session;
    try {
        auto start = Clock::now();
        if (!model_path || !result) throw std::runtime_error("null create argument");
        if (core_mask != RKNN_NPU_CORE_0_1_2)
            throw std::runtime_error("core_mask must be 7");
        session = std::make_unique<Session>();
        session->detail = detail != 0;
        uint32_t flags = detail ? RKNN_FLAG_COLLECT_PERF_MASK : 0;
        auto init_start = Clock::now();
        check(rknn_init(&session->context, const_cast<char*>(model_path), 0,
                       flags, nullptr), "rknn_init");
        double init_ms = elapsed_ms(init_start);
        check(rknn_set_core_mask(session->context, RKNN_NPU_CORE_0_1_2), "rknn_set_core_mask");
        query_metadata(*session, flags, init_ms, start);
        *result = session.release();
        return 0;
    } catch (...) {
        std::string message = current_exception();
        if (session && session->context) {
            int status = rknn_destroy(session->context);
            if (status != RKNN_SUCC) message += "; rknn_destroy returned " + std::to_string(status);
        }
        error_text(error, capacity, message);
        return -1;
    }
}

extern "C" int rknn_bridge_metadata_json(void* handle, const char** result,
                                         char* error, size_t capacity) noexcept {
    if (!handle || !result) { error_text(error, capacity, "null metadata argument"); return -1; }
    *result = static_cast<Session*>(handle)->metadata.c_str();
    return 0;
}

extern "C" int rknn_bridge_destroy(void* handle, char* error, size_t capacity) noexcept {
    std::unique_ptr<Session> session(static_cast<Session*>(handle));
    std::string message;
    if (session) {
        try { release_outputs(*session, nullptr); }
        catch (...) { message = current_exception(); }
        int status = rknn_destroy(session->context);
        if (status != RKNN_SUCC) message += "; rknn_destroy returned " + std::to_string(status);
    }
    if (message.empty()) return 0;
    error_text(error, capacity, message);
    return -1;
}

extern "C" int rknn_bridge_run(void* handle, const uint8_t* rgb, size_t bytes,
                               const char* perf_path, const BridgeOutput** outputs,
                               size_t* count, BridgeTiming* timing,
                               char* error, size_t capacity) noexcept {
    auto* s = static_cast<Session*>(handle);
    if (outputs) *outputs = nullptr;
    if (count) *count = 0;
    if (timing) *timing = {};
    try {
        if (!s || !rgb || !outputs || !count || !timing)
            throw std::runtime_error("null run argument");
        if (s->pending || s->poisoned) throw std::runtime_error("session has pending/failed output release");
        if (s->inputs.size() != 1 || bytes != s->inputs[0].n_elems)
            throw std::runtime_error("input bytes/count mismatch");
        if (s->detail != bool(perf_path && perf_path[0]))
            throw std::runtime_error("detail mode requires perf_path; baseline forbids it");
        rknn_input input{};
        input.index = 0; input.buf = const_cast<uint8_t*>(rgb); input.size = uint32_t(bytes);
        input.pass_through = 0; input.type = RKNN_TENSOR_UINT8; input.fmt = RKNN_TENSOR_NHWC;
        auto start = Clock::now();
        int status = rknn_inputs_set(s->context, 1, &input);
        timing->inputs_set_ms = elapsed_ms(start);
        check(status, "rknn_inputs_set");
        start = Clock::now();
        status = rknn_run(s->context, nullptr);
        timing->run_ms = elapsed_ms(start);
        check(status, "rknn_run");
        s->raw_outputs.assign(s->outputs.size(), rknn_output{});
        s->views.resize(s->outputs.size());
        for (uint32_t i = 0; i < s->raw_outputs.size(); ++i) {
            s->raw_outputs[i].index = i; s->raw_outputs[i].want_float = 1;
        }
        start = Clock::now();
        status = rknn_outputs_get(s->context, uint32_t(s->raw_outputs.size()), s->raw_outputs.data(), nullptr);
        timing->outputs_get_ms = elapsed_ms(start);
        s->pending = status == RKNN_SUCC;
        for (const auto& output : s->raw_outputs) if (output.buf) s->pending = true;
        check(status, "rknn_outputs_get");
        start = Clock::now();
        if (s->detail) {
            rknn_perf_detail report{};
            check(rknn_query(s->context, RKNN_QUERY_PERF_DETAIL, &report, sizeof(report)),
                  "rknn_query(PERF_DETAIL)");
            if (!report.perf_data || !report.data_len)
                throw std::runtime_error("empty PERF_DETAIL report");
            if (report.data_len > uint64_t(std::numeric_limits<std::streamsize>::max()))
                throw std::runtime_error("PERF_DETAIL report too large");
            // 报告指针归 RKNN 所有，下一次 RKNN API 前立即按原始字节写入并关闭。
            std::ofstream file(perf_path, std::ios::binary | std::ios::trunc);
            if (!file) throw std::runtime_error("cannot open PERF_DETAIL path");
            file.write(report.perf_data, std::streamsize(report.data_len));
            file.close();
            if (!file) throw std::runtime_error("cannot save PERF_DETAIL report");
        }
        rknn_perf_run perf{};
        check(rknn_query(s->context, RKNN_QUERY_PERF_RUN, &perf, sizeof(perf)), "rknn_query(PERF_RUN)");
        timing->perf_query_ms = elapsed_ms(start);
        if (perf.run_duration < 0) throw std::runtime_error("negative PERF_RUN duration");
        timing->perf_run_us = perf.run_duration;  // 厂商头文件明确单位为 us。
        for (size_t i = 0; i < s->outputs.size(); ++i) {
            const auto& raw = s->raw_outputs[i];
            uint64_t bytes_expected = uint64_t(s->outputs[i].n_elems) * sizeof(float);
            if (!raw.buf || raw.index != i || raw.size != bytes_expected)
                throw std::runtime_error("output buffer size/index mismatch at " + std::to_string(i));
            s->views[i] = {static_cast<const float*>(raw.buf), s->outputs[i].n_elems, raw.size};
        }
        *outputs = s->views.data(); *count = s->views.size();
        return 0;
    } catch (...) {
        std::string message = current_exception();
        if (s) {
            try { release_outputs(*s, timing); }
            catch (...) { message += "; " + current_exception(); }
        }
        error_text(error, capacity, message);
        return -1;
    }
}

extern "C" int rknn_bridge_release(void* handle, BridgeTiming* timing,
                                   char* error, size_t capacity) noexcept {
    try {
        if (!handle) throw std::runtime_error("null release handle");
        release_outputs(*static_cast<Session*>(handle), timing);
        return 0;
    } catch (...) { error_text(error, capacity, current_exception()); return -1; }
}
