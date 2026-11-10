// Portable correctness + autotune bench for 1-D ConvTranspose OpenCL kernels.
// Baseline: MNN deconv_2d (USE_BUFFER,BIAS). Candidates: bigvgan_deconv1d_v2
// variants (TT x OCB x LWS) and, for the K11/S5/P3/1536->768 contract, tile4.
// Every candidate must match the baseline output bit pattern before it is
// timed. Same resident buffers, alternating A/B order, host enqueue->finish
// and (optional) device event time. Builds on Linux/Android (libOpenCL) and
// macOS (-framework OpenCL, fp32 only there).
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static void check(cl_int x, const char* what) {
    if (x != CL_SUCCESS) throw std::runtime_error(std::string(what) + ": " + std::to_string(x));
}
static std::string readText(const std::string& n) {
    std::ifstream f(n, std::ios::binary); if (!f) throw std::runtime_error("Cannot open " + n);
    std::stringstream s; s << f.rdbuf(); return s.str();
}
static std::vector<char> readBin(const std::string& n) { auto s = readText(n); return {s.begin(), s.end()}; }
static uint16_t f2h(float f) {
    uint32_t v; std::memcpy(&v, &f, 4);
    uint32_t sign = (v >> 16) & 0x8000u, e = (v >> 23) & 255u, m = v & 0x7fffffu;
    if (e == 255) return uint16_t(sign | 0x7c00u | (m ? 0x200u : 0));
    int he = int(e) - 127 + 15;
    if (he >= 31) return uint16_t(sign | 0x7c00u);
    if (he <= 0) {
        if (he < -10) return uint16_t(sign);
        m |= 0x800000u; int sh = 14 - he; uint32_t q = m >> sh, r = m & ((1u << sh) - 1), h = 1u << (sh - 1);
        q += (r > h || (r == h && (q & 1))); return uint16_t(sign | q);
    }
    uint32_t q = m >> 13, r = m & 8191u; q += (r > 4096u || (r == 4096u && (q & 1)));
    if (q == 1024u) { q = 0; ++he; } return uint16_t(sign | (uint32_t(he) << 10) | q);
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); size_t n = v.size(); return n % 2 ? v[n/2] : 0.5*(v[n/2-1]+v[n/2]); }
static std::array<size_t,3> parse3(std::string s) {
    std::replace(s.begin(), s.end(), ',', ' '); std::istringstream f(s); std::array<size_t,3> a{};
    if (!(f >> a[0] >> a[1] >> a[2])) throw std::runtime_error("expected a,b,c: " + s); return a;
}
static std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> o; std::string cur; for (char x : s) { if (x == c) { if (!cur.empty()) o.push_back(cur); cur.clear(); } else cur += x; }
    if (!cur.empty()) o.push_back(cur); return o;
}
static std::string a3(const std::array<size_t,3>& a) { return "[" + std::to_string(a[0]) + "," + std::to_string(a[1]) + "," + std::to_string(a[2]) + "]"; }

struct Kern {
    std::string label; cl_program prog{}; cl_kernel k{}; std::array<size_t,3> logical{}, global{}, local{}; bool hasLocal{};
    double build_ms{};
};

int main(int argc, char** argv) { try {
    std::map<std::string, std::string> p;
    for (int i = 1; i + 1 < argc; i += 2) p[argv[i]] = argv[i + 1];
    auto get = [&](const char* k, const std::string& d) { auto x = p.find(k); return x == p.end() ? d : x->second; };
    if (!p.count("--baseline") || !p.count("--v2")) {
        std::cerr << "Usage: bench_deconv_v2 --baseline deconv_2d.cl --v2 bigvgan_deconv1d_v2.cl [--tile4 bigvgan_deconv_s5_tiled.cl]\n"
                     "  [--K 11 --S 5 --P 3 --ci 1536 --co 768 --T 136 --axis h|w --precision fp16|fp32]\n"
                     "  [--input F [--input-layout packed|nchw-f32] --weights F --bias F  (MNN packed, element type = precision; else seeded random)]\n"
                     "  [--variants 4:1,8:1,8:2,16:1 --lws 0,0,0;8,4,1;16,4,1;32,2,1;64,1,1 --warmup 3 --reps 15 --profile 1 --seed 1]\n"
                     "  [--out results.jsonl]\n";
        return 1;
    }
    const int K = std::stoi(get("--K", "11")), S = std::stoi(get("--S", "5")), P = std::stoi(get("--P", "3"));
    const int CI = std::stoi(get("--ci", "1536")), CO = std::stoi(get("--co", "768")), T = std::stoi(get("--T", "136"));
    const bool axisH = get("--axis", "h") == "h", fp16 = get("--precision", "fp16") == "fp16";
    const int warmup = std::stoi(get("--warmup", "3")), reps = std::stoi(get("--reps", "15")), profiling = std::stoi(get("--profile", "1"));
    if (K - 2 * P != S || CI % 4 || CO % 4 || T < 1) throw std::runtime_error("unsupported shape");
    const int CI4 = CI / 4, CO4 = CO / 4, outT = S * T, transPad = K - 1 - P, align = S - 1 - transPad;
    const size_t es = fp16 ? 2 : 4, nIn = size_t(CI) * T, nW = size_t(CI) * CO * K, nOut = size_t(CO) * outT;

    // ---- data (MNN packed layouts) ----
    std::vector<char> in(nIn * es), w(nW * es), b(size_t(CO) * es);
    auto fill = [&](std::vector<char>& dst, size_t n, float scale, uint32_t seed) {
        std::mt19937 g(seed); std::uniform_real_distribution<float> u(-scale, scale);
        for (size_t i = 0; i < n; ++i) { float v = u(g); if (fp16) { uint16_t h = f2h(v); std::memcpy(&dst[i*2], &h, 2); } else std::memcpy(&dst[i*4], &v, 4); }
    };
    const uint32_t seed = std::stoul(get("--seed", "1"));
    if (p.count("--input") && get("--input-layout", "packed") == "nchw-f32") {
        // Captured MNN target input ([CI,T] float32) -> MNN buffer layout [CI4,T,4] in the run precision.
        auto raw = readBin(p["--input"]); if (raw.size() != nIn * 4) throw std::runtime_error("nchw-f32 input size mismatch");
        const float* f = reinterpret_cast<const float*>(raw.data());
        for (int c = 0; c < CI; ++c) for (int t = 0; t < T; ++t) {
            const size_t dst = (size_t(c / 4) * T + t) * 4 + c % 4; const float v = f[size_t(c) * T + t];
            if (fp16) { uint16_t h = f2h(v); std::memcpy(&in[dst * 2], &h, 2); } else std::memcpy(&in[dst * 4], &v, 4);
        }
    } else if (p.count("--input")) in = readBin(p["--input"]); else fill(in, nIn, 1.0f, seed);
    if (p.count("--weights")) w = readBin(p["--weights"]); else fill(w, nW, 0.05f, seed + 1);
    if (p.count("--bias")) b = readBin(p["--bias"]); else fill(b, CO, 0.1f, seed + 2);
    if (in.size() != nIn*es || w.size() != nW*es || b.size() != size_t(CO)*es) throw std::runtime_error("input/weight/bias byte size mismatch");

    // ---- device ----
    cl_uint np = 0; check(clGetPlatformIDs(0, nullptr, &np), "platforms"); std::vector<cl_platform_id> plats(np); clGetPlatformIDs(np, plats.data(), nullptr);
    cl_device_id dev = nullptr; for (auto pl : plats) { cl_uint nd = 0; if (clGetDeviceIDs(pl, CL_DEVICE_TYPE_GPU, 1, &dev, &nd) == CL_SUCCESS && nd) break; dev = nullptr; }
    if (!dev) throw std::runtime_error("no OpenCL GPU");
    char name[256] = {}, drv[256] = {}; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof name, name, nullptr); clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof drv, drv, nullptr);
    cl_int err; cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); check(err, "context");
    cl_command_queue q = clCreateCommandQueue(ctx, dev, profiling ? CL_QUEUE_PROFILING_ENABLE : 0, &err); check(err, "queue");
    auto mk = [&](size_t n, void* d) { cl_mem m = clCreateBuffer(ctx, d ? (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR) : CL_MEM_READ_WRITE, n, d, &err); check(err, "buffer"); return m; };
    cl_mem bi = mk(in.size(), in.data()), bw = mk(w.size(), w.data()), bb = mk(b.size(), b.data()), bo = mk(nOut*es, nullptr), bc = mk(nOut*es, nullptr);

    const std::string prec = fp16 ? " -DFLOAT=half -DFLOAT4=half4 -DMNN_SUPPORT_FP16" : " -DFLOAT=float -DFLOAT4=float4";
    const std::string common = "-DUSE_BUFFER -DBIAS -cl-mad-enable -w" + prec;
    auto build = [&](const std::string& path, const char* kname, const std::string& opts, const std::string& label,
                     std::array<size_t,3> logical, std::array<size_t,3> local) {
        Kern k; k.label = label; k.logical = logical; k.global = logical; k.local = local; k.hasLocal = local[0] && local[1] && local[2];
        if (k.hasLocal) for (int i = 0; i < 3; ++i) k.global[i] = (logical[i] + local[i] - 1) / local[i] * local[i];
        auto src = readText(path); const char* s = src.c_str(); size_t n = src.size();
        auto t0 = Clock::now(); k.prog = clCreateProgramWithSource(ctx, 1, &s, &n, &err); check(err, "program");
        err = clBuildProgram(k.prog, 1, &dev, opts.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) { size_t ln; clGetProgramBuildInfo(k.prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ln); std::string log(ln, 0);
            clGetProgramBuildInfo(k.prog, dev, CL_PROGRAM_BUILD_LOG, ln, &log[0], nullptr); std::cerr << label << " build log:\n" << log << "\n"; check(err, "build"); }
        k.k = clCreateKernel(k.prog, kname, &err); check(err, "kernel"); k.build_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        size_t maxwg = 0; clGetKernelWorkGroupInfo(k.k, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof maxwg, &maxwg, nullptr);
        if (k.hasLocal && local[0]*local[1]*local[2] > maxwg) { clReleaseKernel(k.k); clReleaseProgram(k.prog); k.k = nullptr; }
        return k;
    };
    auto setArg = [&](cl_kernel k, int& i, size_t n, const void* v) { check(clSetKernelArg(k, i++, n, v), "setArg"); };
    auto pair = [](int x, int y) { cl_int2 v; v.s[0] = x; v.s[1] = y; return v; };
    // MNN deconv_2d-style argument list (used by baseline and tile4).
    auto mnnArgs = [&](Kern& k, cl_mem out) {
        int i = 0; for (size_t g : k.logical) { int v = int(g); setArg(k.k, i, sizeof v, &v); }
        setArg(k.k, i, sizeof bi, &bi); setArg(k.k, i, sizeof bw, &bw); setArg(k.k, i, sizeof bb, &bb); setArg(k.k, i, sizeof out, &out);
        int batch = 1; setArg(k.k, i, sizeof batch, &batch);
        cl_int2 sh[] = {axisH ? pair(T,1) : pair(1,T), axisH ? pair(outT,1) : pair(1,outT), axisH ? pair(S,1) : pair(1,S),
                        axisH ? pair(align,0) : pair(0,align), axisH ? pair(transPad,0) : pair(0,transPad), axisH ? pair(K,1) : pair(1,K)};
        for (auto& s : sh) setArg(k.k, i, sizeof s, &s);
        for (int v : {K, CI4, CO4}) setArg(k.k, i, sizeof v, &v);
    };
    auto v2Args = [&](Kern& k, cl_mem out) {
        int i = 0; for (size_t g : k.logical) { int v = int(g); setArg(k.k, i, sizeof v, &v); }
        setArg(k.k, i, sizeof bi, &bi); setArg(k.k, i, sizeof bw, &bw); setArg(k.k, i, sizeof bb, &bb); setArg(k.k, i, sizeof out, &out);
        setArg(k.k, i, sizeof T, &T);
    };
    auto run = [&](Kern& k, double& host, double& dev_us) {
        check(clFinish(q), "finish"); cl_event e{}; auto t0 = Clock::now();
        check(clEnqueueNDRangeKernel(q, k.k, 3, nullptr, k.global.data(), k.hasLocal ? k.local.data() : nullptr, 0, nullptr, profiling ? &e : nullptr), "enqueue");
        check(clFinish(q), "finish"); host = std::chrono::duration<double, std::micro>(Clock::now() - t0).count(); dev_us = -1;
        if (profiling) { cl_ulong a, z; clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof a, &a, nullptr); clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof z, &z, nullptr); dev_us = (z - a) * 1e-3; clReleaseEvent(e); }
    };
    auto download = [&](cl_mem m) { std::vector<char> v(nOut*es); check(clEnqueueReadBuffer(q, m, CL_TRUE, 0, v.size(), v.data(), 0, nullptr, nullptr), "read"); return v; };

    std::ofstream logf(get("--out", "bench_v2_results.jsonl")); logf << std::setprecision(10);
    std::ostream& log = logf;
    Kern base = build(p["--baseline"], "deconv_2d", common, "baseline_deconv_2d",
                      {size_t(CO4), axisH ? 1u : size_t(outT), axisH ? size_t(outT) : 1u}, {0,0,0});
    mnnArgs(base, bo); double h, d; run(base, h, d); const auto ref = download(bo);
    log << "{\"kind\":\"manifest\",\"device\":\"" << name << "\",\"driver\":\"" << drv << "\",\"K\":" << K << ",\"S\":" << S << ",\"P\":" << P
        << ",\"ci\":" << CI << ",\"co\":" << CO << ",\"T\":" << T << ",\"axis\":\"" << (axisH ? "h" : "w") << "\",\"precision\":\"" << (fp16 ? "fp16" : "fp32")
        << "\",\"data\":\"" << (p.count("--input") ? "files" : "seeded_random") << "\",\"warmup\":" << warmup << ",\"reps\":" << reps << "}\n";

    std::vector<Kern> cands;
    if (p.count("--tile4") && K == 11 && S == 5 && P == 3 && CI == 1536 && CO == 768)
        cands.push_back(build(p["--tile4"], "bigvgan_deconv_s5_tiled", common + " -DBIGVGAN_TILE=4", "tile4_lws8,4,1",
                              {192, size_t((T + 3) / 4), 5}, {8, 4, 1}));
    for (auto& var : split(get("--variants", "4:1,8:1,8:2,12:1,16:1"), ',')) {
        auto f = split(var, ':'); int TT = std::stoi(f[0]), OCB = f.size() > 1 ? std::stoi(f[1]) : 1;
        if (CO4 % OCB) continue;
        std::string opts = common + " -DDC_K=" + std::to_string(K) + " -DDC_S=" + std::to_string(S) + " -DDC_PAD=" + std::to_string(P)
            + " -DDC_CI4=" + std::to_string(CI4) + " -DDC_CO4=" + std::to_string(CO4) + " -DDC_TT=" + std::to_string(TT) + " -DDC_OCB=" + std::to_string(OCB);
        for (auto& l : split(get("--lws", "0,0,0;8,4,1;16,4,1;32,2,1;64,1,1;16,1,1"), ';')) {
            auto L = parse3(l);
            auto k = build(p["--v2"], "bigvgan_deconv1d_v2", opts, "v2_tt" + std::to_string(TT) + "_ocb" + std::to_string(OCB) + "_lws" + l,
                           {size_t(CO4 / OCB), size_t((T + TT - 1) / TT), size_t(S)}, L);
            if (k.k) cands.push_back(k);
        }
    }
    struct Res { std::string label; double host_med, dev_med; bool bit_equal; };
    std::vector<Res> results;
    for (auto& c : cands) {
        if (c.label.rfind("tile4", 0) == 0) mnnArgs(c, bc); else v2Args(c, bc);
        std::vector<char> zero(nOut*es, 0x7f); check(clEnqueueWriteBuffer(q, bc, CL_TRUE, 0, zero.size(), zero.data(), 0, nullptr, nullptr), "poison");
        run(c, h, d); auto got = download(bc);
        size_t mism = 0; for (size_t i = 0; i < nOut*es; i += es) if (std::memcmp(&got[i], &ref[i], es)) ++mism;
        log << "{\"kind\":\"correctness\",\"label\":\"" << c.label << "\",\"bit_mismatches\":" << mism << ",\"elements\":" << nOut << ",\"build_ms\":" << c.build_ms << "}\n";
        if (mism) { std::cerr << c.label << ": " << mism << " bit mismatches; not timed\n"; results.push_back({c.label, -1, -1, false}); continue; }
        for (int i = 0; i < warmup; ++i) { run(base, h, d); run(c, h, d); }
        std::vector<double> bh, bd, ch, cd;
        for (int r = 0; r < reps; ++r) for (int o = 0; o < 2; ++o) {
            bool isNew = ((r + o) & 1) != 0; run(isNew ? c : base, h, d);
            (isNew ? ch : bh).push_back(h); (isNew ? cd : bd).push_back(d);
            log << "{\"kind\":\"sample\",\"label\":\"" << (isNew ? c.label : base.label) << "\",\"pair_with\":\"" << c.label << "\",\"rep\":" << r << ",\"host_us\":" << h << ",\"kernel_us\":" << d << "}\n";
        }
        results.push_back({c.label, median(ch), median(cd), true});
        log << "{\"kind\":\"summary\",\"label\":\"" << c.label << "\",\"global\":" << a3(c.global) << ",\"local\":" << a3(c.local)
            << ",\"baseline_host_med_us\":" << median(bh) << ",\"cand_host_med_us\":" << median(ch)
            << ",\"baseline_kernel_med_us\":" << median(bd) << ",\"cand_kernel_med_us\":" << median(cd)
            << ",\"speedup_host\":" << median(bh) / median(ch) << "}\n";
        std::cout << std::left << std::setw(34) << c.label << " base " << std::setw(10) << median(bh) << " us  cand " << std::setw(10) << median(ch)
                  << " us  x" << median(bh) / median(ch) << "  bit_equal\n";
    }
    std::sort(results.begin(), results.end(), [](const Res& a, const Res& b) { return a.bit_equal != b.bit_equal ? a.bit_equal : a.host_med < b.host_med; });
    if (!results.empty() && results[0].bit_equal)
        log << "{\"kind\":\"best\",\"label\":\"" << results[0].label << "\",\"host_med_us\":" << results[0].host_med << "}\n",
        std::cout << "BEST " << results[0].label << " " << results[0].host_med << " us\n";
    bool anyFail = false; for (auto& r : results) anyFail |= !r.bit_equal;
    return anyFail ? 3 : 0;
} catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << "\n"; return 2; } }
