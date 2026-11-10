#pragma once
// Candidate rknn-audio-prefix-v2 (parent: the v1 adapter in omni_rknn_prefix.hpp).
// Same graph (Conv1 -> GELU -> Conv2 -> GELU, MNN-decoded weights, RKNN FP16),
// same tiling contract (516-frame halo window -> 256 output frames), same class
// name and API, so omni.cpp is unchanged. Engineering changes only:
//
//   * persistent per-context tile/mask/output buffers; outputs are pre-allocated
//     (is_prealloc) so no runtime malloc/free per tile        [numerics unchanged]
//   * optional NPU core binding           OMNI_RKNN_CORE_MASK=auto|0|1|2|01|012|all
//   * optional concurrent tiles for requests longer than one tile:
//       OMNI_RKNN_PARALLEL_TILES=N (1..3) duplicates the context with
//       rknn_dup_context and binds context i to NPU core i   [numerics unchanged
//       per tile: each tile is still one independent rknn_run of the same graph]
//   * optional FP16 input feed            OMNI_RKNN_INPUT_FP16=1
//       transposes + rounds (RNE) on the CPU in one pass instead of handing FP32
//       to rknn_inputs_set. Rounding may differ from the runtime's own conversion,
//       so it is opt-in and covered by the self-check below.
//   * OMNI_RKNN_SELFCHECK=1: on the first request, every tile is also computed
//     with the v1 settings (FP32 input, context 0) and compared bit-for-bit; on
//     any mismatch the adapter prints it and falls back to v1 settings for the
//     remaining lifetime.
//
// Unset variables reproduce the v1 adapter's runtime behaviour.
#include <rknn_api.h>
#include <MNN/expr/ExprCreator.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

class OmniRknnPrefix {
    static constexpr int kInFrames = 516, kMel = 128, kOutFrames = 256, kChannels = 1280;
    struct Slot {
        rknn_context ctx = 0;
        bool owner = false;
        std::vector<float> tile, mask, out;
        std::vector<uint16_t> tile16;
        double prepareUs = 0, setUs = 0, runUs = 0, getUs = 0;
        int calls = 0;
    };
    std::vector<std::unique_ptr<Slot>> slots_;
    unsigned perf_sequence_ = 0;
    const char* perf_dir_ = nullptr;
    bool input_fp16_ = false;
    bool selfcheck_pending_ = false;
    std::string core_mask_name_ = "unset";

    static double us(std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count();
    }
    static uint16_t toHalf(float f) {
#if defined(__aarch64__)
        __fp16 h = static_cast<__fp16>(f);   // IEEE RNE, hardware conversion
        uint16_t bits; std::memcpy(&bits, &h, 2); return bits;
#else
        uint32_t v; std::memcpy(&v, &f, 4);
        uint32_t sign = (v >> 16) & 0x8000u, e = (v >> 23) & 255u, m = v & 0x7fffffu;
        if (e == 255) return uint16_t(sign | 0x7c00u | (m ? 0x200u : 0));
        int he = int(e) - 127 + 15;
        if (he >= 31) return uint16_t(sign | 0x7c00u);
        if (he <= 0) {
            if (he < -10) return uint16_t(sign);
            m |= 0x800000u; const int sh = 14 - he; uint32_t q = m >> sh, r = m & ((1u << sh) - 1), h = 1u << (sh - 1);
            q += (r > h || (r == h && (q & 1))); return uint16_t(sign | q);
        }
        uint32_t q = m >> 13, r = m & 8191u; q += (r > 4096u || (r == 4096u && (q & 1)));
        if (q == 1024u) { q = 0; ++he; } return uint16_t(sign | (uint32_t(he) << 10) | q);
#endif
    }
    static bool parseMask(const char* s, rknn_core_mask& m, std::string& name) {
        if (!s || !*s) return false;
        name = s;
        if (name == "auto") m = RKNN_NPU_CORE_AUTO;
        else if (name == "0") m = RKNN_NPU_CORE_0;
        else if (name == "1") m = RKNN_NPU_CORE_1;
        else if (name == "2") m = RKNN_NPU_CORE_2;
        else if (name == "01") m = RKNN_NPU_CORE_0_1;
        else if (name == "012") m = RKNN_NPU_CORE_0_1_2;
        else if (name == "all") m = RKNN_NPU_CORE_ALL;
        else { std::printf("OMNI_RKNN_V2 ignoring unknown OMNI_RKNN_CORE_MASK=%s\n", s); name = "unset"; return false; }
        return true;
    }
    static void allocate(Slot& s) {
        s.tile.assign(kInFrames * kMel, 0.f);
        s.mask.assign(kInFrames, 0.f);
        s.tile16.assign(kInFrames * kMel, 0);
        s.out.assign(size_t(kChannels) * kOutFrames, 0.f);
    }
    // One independent tile: identical window/mask construction to v1.
    bool runTile(Slot& s, const float* src, int t, int start, bool fp16) {
        auto st = std::chrono::steady_clock::now();
        std::fill(s.mask.begin(), s.mask.end(), 0.f);
        if (fp16) std::fill(s.tile16.begin(), s.tile16.end(), uint16_t(0));
        else std::fill(s.tile.begin(), s.tile.end(), 0.f);
        for (int w = 0; w < kInFrames; ++w) {
            const int pos = 2 * start - 2 + w;
            if (pos < 0 || pos >= t) continue;
            s.mask[w] = 1.f;
            if (fp16) { for (int c = 0; c < kMel; ++c) s.tile16[w * kMel + c] = toHalf(src[c * t + pos]); }
            else { for (int c = 0; c < kMel; ++c) s.tile[w * kMel + c] = src[c * t + pos]; }
        }
        s.prepareUs += us(st);
        rknn_input in[2]{};
        in[0].index = 0; in[0].fmt = RKNN_TENSOR_NHWC;
        if (fp16) { in[0].buf = s.tile16.data(); in[0].size = s.tile16.size() * 2; in[0].type = RKNN_TENSOR_FLOAT16; }
        else { in[0].buf = s.tile.data(); in[0].size = s.tile.size() * 4; in[0].type = RKNN_TENSOR_FLOAT32; }
        in[1].index = 1; in[1].buf = s.mask.data(); in[1].size = s.mask.size() * 4;
        in[1].type = RKNN_TENSOR_FLOAT32; in[1].fmt = RKNN_TENSOR_NHWC;
        st = std::chrono::steady_clock::now(); int ret = rknn_inputs_set(s.ctx, 2, in); s.setUs += us(st); if (ret) return false;
        st = std::chrono::steady_clock::now(); ret = rknn_run(s.ctx, nullptr); s.runUs += us(st); if (ret) return false;
        rknn_output y{}; y.index = 0; y.want_float = 1; y.is_prealloc = 1;
        y.buf = s.out.data(); y.size = uint32_t(s.out.size() * sizeof(float));
        st = std::chrono::steady_clock::now(); ret = rknn_outputs_get(s.ctx, 1, &y, nullptr); s.getUs += us(st); if (ret) return false;
        ret = rknn_outputs_release(s.ctx, 1, &y);
        ++s.calls;
        return ret == 0;
    }
    void dumpPerf(Slot& s, int t, int start) {
        if (!perf_dir_) return;
        rknn_perf_detail detail{};
        const int query_ret = rknn_query(s.ctx, RKNN_QUERY_PERF_DETAIL, &detail, sizeof(detail));
        const std::string stem = std::string(perf_dir_) + "/tile_" + std::to_string(perf_sequence_++);
        if (query_ret == RKNN_SUCC && detail.perf_data) {
            std::ofstream report(stem + ".txt", std::ios::binary);
            report.write(detail.perf_data, detail.data_len);
        }
        std::ofstream metadata(stem + ".json");
        metadata << "{\"mel_frames\":" << t << ",\"output_tile_start\":" << start << ",\"query_return\":" << query_ret << "}\n";
    }

public:
    ~OmniRknnPrefix() {
        // Duplicated contexts first, then the original.
        for (size_t i = slots_.size(); i-- > 0;) if (slots_[i]->ctx) rknn_destroy(slots_[i]->ctx);
    }
    bool load(const char* path) {
        perf_dir_ = std::getenv("OMNI_RKNN_PERF_DIR");
        auto first = std::make_unique<Slot>();
        if (rknn_init(&first->ctx, const_cast<char*>(path), 0, perf_dir_ ? RKNN_FLAG_COLLECT_PERF_MASK : 0, nullptr) != RKNN_SUCC) return false;
        first->owner = true;
        rknn_input_output_num io{};
        if (rknn_query(first->ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)) != 0 || io.n_input != 2 || io.n_output != 1) return false;
        rknn_tensor_attr a{}; a.index = 0;
        if (rknn_query(first->ctx, RKNN_QUERY_INPUT_ATTR, &a, sizeof(a)) != 0 || a.n_elems != kMel * kInFrames || a.fmt != RKNN_TENSOR_NHWC) return false;
        a = {}; a.index = 0;
        if (rknn_query(first->ctx, RKNN_QUERY_OUTPUT_ATTR, &a, sizeof(a)) != 0 || a.n_elems != kChannels * kOutFrames || a.fmt != RKNN_TENSOR_NCHW) return false;
        allocate(*first);
        slots_.push_back(std::move(first));

        const char* par = std::getenv("OMNI_RKNN_PARALLEL_TILES");
        const int parallel = std::max(1, std::min(3, par ? std::atoi(par) : 1));
        for (int i = 1; i < parallel; ++i) {
            auto s = std::make_unique<Slot>();
            if (rknn_dup_context(&slots_[0]->ctx, &s->ctx) != RKNN_SUCC) {
                std::printf("OMNI_RKNN_V2 rknn_dup_context failed; continuing with %zu context(s)\n", slots_.size());
                break;
            }
            allocate(*s);
            slots_.push_back(std::move(s));
        }
        rknn_core_mask mask{};
        std::string maskName;
        if (slots_.size() > 1) {
            // One core per context so concurrent tiles do not contend.
            const rknn_core_mask perCore[3] = {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2};
            for (size_t i = 0; i < slots_.size(); ++i) rknn_set_core_mask(slots_[i]->ctx, perCore[i]);
            core_mask_name_ = "per_context_0_1_2";
        } else if (parseMask(std::getenv("OMNI_RKNN_CORE_MASK"), mask, maskName)) {
            if (rknn_set_core_mask(slots_[0]->ctx, mask) == RKNN_SUCC) core_mask_name_ = maskName;
            else std::printf("OMNI_RKNN_V2 rknn_set_core_mask(%s) failed; runtime default kept\n", maskName.c_str());
        }
        const char* f16 = std::getenv("OMNI_RKNN_INPUT_FP16");
        input_fp16_ = f16 && std::atoi(f16) == 1;
        const char* sc = std::getenv("OMNI_RKNN_SELFCHECK");
        selfcheck_pending_ = sc && std::atoi(sc) == 1;

        rknn_sdk_version version{}; rknn_query(slots_[0]->ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
        std::printf("OMNI_RKNN_INIT api=%s driver=%s graph=conv1_gelu_conv2_gelu input=NHWC output=NCHW adapter=v2 contexts=%zu core_mask=%s input_fp16=%d selfcheck=%d\n",
                    version.api_version, version.drv_version, slots_.size(), core_mask_name_.c_str(), int(input_fp16_), int(selfcheck_pending_));
        return true;
    }

    MNN::Express::VARP forward(MNN::Express::VARP mel) {
        using namespace MNN::Express;
        if (slots_.empty() || mel.get() == nullptr || !mel->getInfo()) return nullptr;
        auto d = mel->getInfo()->dim;
        if (d.size() != 3 || d[0] != 1 || d[1] != kMel || d[2] < 1) return nullptr;
        const int t = d[2], s = (t + 1) / 2;
        auto begin = std::chrono::steady_clock::now();
        auto x = _Convert(mel, NCHW); const float* src = x->readMap<float>(); if (!src) return nullptr;
        auto out = _Input({1, kChannels, s}, NCHW, halide_type_of<float>()); float* dst = out->writeMap<float>();
        if (!dst) return nullptr;
        for (auto& sl : slots_) { sl->prepareUs = sl->setUs = sl->runUs = sl->getUs = 0; sl->calls = 0; }
        double copyUs = 0;
        const int tiles = (s + kOutFrames - 1) / kOutFrames;
        auto scatterFrom = [&](const float* tileOut, int start) {
            auto st = std::chrono::steady_clock::now();
            const int n = std::min(kOutFrames, s - start);
            for (int c = 0; c < kChannels; ++c) std::memcpy(dst + size_t(c) * s + start, tileOut + size_t(c) * kOutFrames, n * sizeof(float));
            return us(st);
        };
        auto scatter = [&](Slot& sl, int start) { return scatterFrom(sl.out.data(), start); };

        if (selfcheck_pending_) {
            // v1 settings on context 0 vs v2 settings on every slot, tile by tile.
            selfcheck_pending_ = false;
            Slot& ref = *slots_[0];
            std::vector<float> golden(ref.out.size());
            size_t mismatches = 0; double maxAbs = 0;
            for (int tIdx = 0; tIdx < tiles; ++tIdx) {
                const int start = tIdx * kOutFrames;
                if (!runTile(ref, src, t, start, false)) return nullptr;
                golden = ref.out;
                Slot& cand = *slots_[tIdx % slots_.size()];
                if (!runTile(cand, src, t, start, input_fp16_)) return nullptr;
                const int n = std::min(kOutFrames, s - start);
                for (int c = 0; c < kChannels; ++c) for (int i = 0; i < n; ++i) {
                    const float a = golden[size_t(c) * kOutFrames + i], b = cand.out[size_t(c) * kOutFrames + i];
                    if (std::memcmp(&a, &b, 4)) { ++mismatches; maxAbs = std::max(maxAbs, double(std::fabs(a - b))); }
                }
                copyUs += scatterFrom(golden.data(), start);   // emit the v1-settings result
            }
            std::printf("OMNI_RKNN_SELFCHECK tiles=%d bit_mismatches=%zu max_abs=%.9g input_fp16=%d contexts=%zu result=%s\n",
                        tiles, mismatches, maxAbs, int(input_fp16_), slots_.size(), mismatches ? "FALLBACK_V1" : "PASS");
            if (mismatches) {
                input_fp16_ = false;
                while (slots_.size() > 1) { rknn_destroy(slots_.back()->ctx); slots_.pop_back(); }
            }
        } else if (slots_.size() == 1 || tiles == 1) {
            Slot& sl = *slots_[0];
            for (int start = 0; start < s; start += kOutFrames) {
                if (!runTile(sl, src, t, start, input_fp16_)) return nullptr;
                dumpPerf(sl, t, start);
                copyUs += scatter(sl, start);
            }
        } else {
            // Tile i runs on slot i % N; each worker owns its slot and writes a
            // disjoint column range of dst.
            std::atomic<bool> ok{true};
            std::vector<double> copy(slots_.size(), 0.0);
            std::vector<std::thread> workers;
            for (size_t w = 0; w < slots_.size(); ++w) {
                workers.emplace_back([&, w] {
                    Slot& sl = *slots_[w];
                    for (int tIdx = int(w); tIdx < tiles && ok; tIdx += int(slots_.size())) {
                        if (!runTile(sl, src, t, tIdx * kOutFrames, input_fp16_)) { ok = false; return; }
                        copy[w] += scatter(sl, tIdx * kOutFrames);
                    }
                });
            }
            for (auto& th : workers) th.join();
            if (!ok) return nullptr;
            for (double c : copy) copyUs += c;
        }
        double prepareUs = 0, setUs = 0, runUs = 0, getUs = 0; int calls = 0;
        for (auto& sl : slots_) { prepareUs += sl->prepareUs; setUs += sl->setUs; runUs += sl->runUs; getUs += sl->getUs; calls += sl->calls; }
        // Stage sums add across concurrent contexts; total_us is wall time.
        std::printf("OMNI_RKNN_REQUEST frames=%d output_frames=%d calls=%d total_us=%.1f prepare_us=%.1f inputs_set_us=%.1f run_us=%.1f outputs_get_us=%.1f copy_release_us=%.1f adapter=v2 contexts=%zu\n",
                    t, s, calls, us(begin), prepareUs, setUs, runUs, getUs, copyUs, slots_.size());
        return out;
    }
};
