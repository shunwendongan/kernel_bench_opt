#pragma once
#include <MNN/expr/Expr.hpp>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace MNN { namespace Transformer { namespace OmniCapture {
inline const char* directory() { const char* p = std::getenv("OMNI_CAPTURE_DIR"); return p && *p ? p : nullptr; }
inline int& nextId() { static int id = 0; return id; }
inline void dump(const Express::VARP& var, const std::string& basename, int id, const char* kind) {
    const auto* info = var->getInfo();
    const auto* ptr = var->readMap<float>();
    if (!info || !ptr || info->type != halide_type_of<float>()) {
        std::fprintf(stderr, "OMNI_CAPTURE_INVALID %s\n", kind); std::abort();
    }
    const std::string path = std::string(directory()) + "/" + basename;
    std::ofstream data(path, std::ios::binary);
    data.write(reinterpret_cast<const char*>(ptr), info->size * sizeof(float));
    data.close();
    if (!data) { std::fprintf(stderr, "OMNI_CAPTURE_WRITE_FAILED %s\n", path.c_str()); std::abort(); }
    std::ofstream meta(std::string(directory()) + "/" + kind + "_sequence.jsonl", std::ios::app);
    meta << "{\"id\":" << id << ",\"file\":\"" << basename << "\",\"shape\":[";
    for (size_t i = 0; i < info->dim.size(); ++i) { if (i) meta << ','; meta << info->dim[i]; }
    meta << "],\"dtype\":\"float32\",\"layout\":\"" << (info->order == Express::NCHW ? "NCHW" : "other")
         << "\",\"elements\":" << info->size << ",\"origin\":\"native_full_inference_invocation\"}\n";
    if (!meta) { std::fprintf(stderr, "OMNI_CAPTURE_MANIFEST_FAILED\n"); std::abort(); }
}
inline int before(const Express::VARP& mel) {
    if (!directory()) return -1;
    const int id = nextId()++;
    char name[80]; std::snprintf(name, sizeof(name), "mel_%06d.f32", id);
    dump(mel, name, id, "mel");
    return id;
}
inline void after(const Express::VARP& waveform, int id) {
    if (id < 0) return;
    char name[80]; std::snprintf(name, sizeof(name), "wave_%06d.f32", id);
    dump(waveform, name, id, "wave");
}
}}}
