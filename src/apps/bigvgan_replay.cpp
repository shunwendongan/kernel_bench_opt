#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "jsonhpp/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace MNN;
using namespace MNN::Express;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

static double elapsedMs(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
static void writeJson(const fs::path& p, const Json& j) { std::ofstream(p) << j.dump(2) << '\n'; }
static bool writeRaw(const fs::path& p, const float* data, size_t count) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data), count * sizeof(float));
    return f.good();
}
static bool dumpTensor(Tensor* t, const fs::path& root, const std::string& stem) {
    if (!t || t->getType() != halide_type_of<float>()) return false;
    t->wait(Tensor::MAP_TENSOR_READ, true);
    Tensor host(t, Tensor::CAFFE, true);
    if (!t->copyToHostTensor(&host)) return false;
    if (!writeRaw(root / (stem + ".f32"), host.host<float>(), host.elementSize())) return false;
    writeJson(root / (stem + ".json"), {{"shape", host.shape()}, {"dtype", "float32"},
             {"layout", "NCHW"}, {"elements", host.elementSize()},
             {"representation", "canonical_host_values_from_current_backend"}});
    return true;
}
struct Case {
    int id = -1;
    int T = -1;
    std::vector<int> shape;
    std::vector<float> mel;
    VARP input;
};

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: bigvgan_replay MODEL CAPTURE_DIR OUTPUT MODE[capture|shape|sequence] [WARMUP=1] [REPEATS=3]\n";
        return 2;
    }
    const fs::path model = fs::absolute(argv[1]), source = fs::absolute(argv[2]), out = fs::absolute(argv[3]);
    const std::string mode = argv[4];
    if (mode != "capture" && mode != "shape" && mode != "sequence") return 2;
    const bool capture = mode == "capture";
    const int warmup = capture ? 0 : argc > 5 ? std::stoi(argv[5]) : 1;
    const int repeats = capture ? 1 : argc > 6 ? std::stoi(argv[6]) : 3;
    if (warmup < 0 || repeats < 1) return 2;
    fs::create_directories(out);
    std::vector<Case> cases;
    std::set<int> selected;
    std::ifstream manifest(source / "mel_sequence.jsonl");
    std::string line;
    while (std::getline(manifest, line)) {
        auto j = Json::parse(line, nullptr, false);
        if (j.is_discarded()) return 3;
        Case c;
        c.id = j.at("id").get<int>();
        c.shape = j.at("shape").get<std::vector<int>>();
        if (c.shape.size() != 3 || c.shape[0] != 1 || c.shape[1] != 80 || c.shape[2] < 1 ||
            j.at("dtype") != "float32" || j.at("layout") != "NCHW") return 3;
        c.T = c.shape[2];
        if (mode != "sequence" && !selected.insert(c.T).second) continue;
        c.mel.resize(size_t(80) * c.T);
        std::ifstream f(source / j.at("file").get<std::string>(), std::ios::binary | std::ios::ate);
        if (!f || f.tellg() != std::streampos(c.mel.size() * sizeof(float))) return 3;
        f.seekg(0); f.read(reinterpret_cast<char*>(c.mel.data()), c.mel.size() * sizeof(float));
        if (!f) return 3;
        cases.emplace_back(std::move(c));
    }
    if (cases.empty()) return 3;

    BackendConfig cpu;
    cpu.precision = BackendConfig::Precision_Low;
    cpu.memory = BackendConfig::Memory_Low;
    auto executor = Executor::newExecutor(MNN_FORWARD_CPU, cpu, 4);
    ExecutorScope scope(executor);
    BackendConfig gpu;
    gpu.precision = BackendConfig::Precision_Normal;
    gpu.memory = BackendConfig::Memory_Low;
    ScheduleConfig schedule;
    schedule.type = MNN_FORWARD_OPENCL;
    schedule.numThread = 1 | 64 | 512;
    schedule.backendConfig = &gpu;
    auto runtime = std::shared_ptr<Executor::RuntimeManager>(Executor::RuntimeManager::createRuntimeManager(schedule));
    if (!runtime) return 4;
    runtime->setHint(Interpreter::INIT_THREAD_NUMBER, 4);
    runtime->setHint(Interpreter::MEM_ALLOCATOR_TYPE, 0);
    runtime->setHint(Interpreter::DYNAMIC_QUANT_OPTIONS, 0);
    runtime->setHint(Interpreter::ATTENTION_OPTION, 8);
    runtime->setCache((out / "mnn_cachefile.bin").string());
    runtime->setExternalFile(model.string() + ".weight");

    int activeT = -1, activeId = -1;
    int capturesBefore = 0, capturesAfter = 0;
    bool tensorCaptureOk = true;
    if (capture) {
        runtime->setMode(Interpreter::Session_Debug);
        executor->setCallBack(
            [&](const std::vector<Tensor*>& ts, const OperatorInfo* op) {
                if (op->name() != "/ups.0.0/ConvTranspose_output_0") return true;
                ++capturesBefore;
                tensorCaptureOk = !ts.empty() && dumpTensor(ts[0], out, "input_T" + std::to_string(activeT)) && tensorCaptureOk;
                return true;
            },
            [&](const std::vector<Tensor*>& ts, const OperatorInfo* op) {
                if (op->name() != "/ups.0.0/ConvTranspose_output_0") return true;
                ++capturesAfter;
                tensorCaptureOk = !ts.empty() && dumpTensor(ts[0], out, "output_T" + std::to_string(activeT)) && tensorCaptureOk;
                writeJson(out / ("origin_T" + std::to_string(activeT) + ".json"), {{"native_mel_id", activeId},
                    {"source_capture_dir", source.string()}, {"node_name", op->name()}, {"type", op->type()},
                    {"origin", "replay_of_captured_native_bigvgan_input"}});
                return true;
            });
    }
    Module::Config config;
    config.shapeMutable = false;
    config.rearrange = true;
    const auto loadStart = Clock::now();
    auto module = std::shared_ptr<Module>(Module::load({"generated_mel"}, {"waveform"}, model.string().c_str(), runtime, &config), Module::destroy);
    const double loadMs = elapsedMs(loadStart);
    if (!module) return 4;
    for (auto& c : cases) c.input = _Input(c.shape, NCHW, halide_type_of<float>());

    std::ofstream results(out / "latency.jsonl");
    bool finite = true, filesOk = true;
    auto invoke = [&](Case& c, int iteration, bool measured) {
        std::memcpy(c.input->writeMap<float>(), c.mel.data(), c.mel.size() * sizeof(float));
        activeT = c.T; activeId = c.id;
        const auto start = Clock::now();
        auto outputs = module->onForward({c.input});
        if (outputs.size() != 1 || !outputs[0]->getInfo()) { std::cerr << "Invalid outputs\n"; std::exit(5); }
        const float* wave = outputs[0]->readMap<float>();
        const double ms = elapsedMs(start);
        if (!wave) { std::cerr << "Null output data\n"; std::exit(5); }
        const int size = outputs[0]->getInfo()->size;
        for (int i = 0; i < size; ++i) if (!std::isfinite(wave[i])) finite = false;
        if (measured) {
            results << Json({{"mode", mode}, {"iteration", iteration}, {"native_mel_id", c.id}, {"T", c.T},
                {"latency_ms", ms}, {"wave_samples", size}, {"profile_enabled", capture},
                {"scope", "Module_onForward_plus_synchronized_float32_wave_readMap"}}).dump() << '\n';
            results.flush();
        }
        if (measured && iteration == repeats - 1) {
            const std::string stem = "wave_id" + std::to_string(c.id) + "_T" + std::to_string(c.T);
            filesOk = writeRaw(out / (stem + ".f32"), wave, size) && filesOk;
            writeJson(out / (stem + ".json"), {{"shape", outputs[0]->getInfo()->dim}, {"dtype", "float32"},
                {"native_mel_id", c.id}, {"T", c.T}, {"elements", size}});
        }
        return ms;
    };
    std::vector<double> sequenceMs;
    if (mode == "sequence") {
        for (int it = -warmup; it < repeats; ++it) {
            double total = 0;
            for (auto& c : cases) total += invoke(c, it, it >= 0);
            if (it >= 0) sequenceMs.push_back(total);
            std::cout << "SEQUENCE_DONE iteration=" << it << " synchronized_module_sum_ms=" << total << std::endl;
        }
    } else {
        for (auto& c : cases) {
            for (int it = -warmup; it < repeats; ++it) invoke(c, it, it >= 0);
            std::cout << "SHAPE_DONE T=" << c.T << std::endl;
        }
    }
    runtime->updateCache();
    const bool captureCountOk = !capture || (capturesBefore == cases.size() && capturesAfter == cases.size());
    writeJson(out / "run.json", {{"model", model.string()}, {"source", source.string()}, {"mode", mode},
        {"warmup", warmup}, {"repeats", repeats}, {"case_count", cases.size()}, {"load_ms", loadMs},
        {"sequence_module_sum_ms", sequenceMs}, {"finite", finite}, {"files_ok", filesOk},
        {"callback_before_hits", capturesBefore}, {"callback_after_hits", capturesAfter},
        {"capture_count_ok", captureCountOk}, {"tensor_capture_ok", tensorCaptureOk},
        {"precision", "MNN_Precision_Normal_unchanged"}, {"module_shape_mutable", false},
        {"module_rearrange", true}, {"num_thread_flags", schedule.numThread},
        {"timing_excludes", "file_read_and_host_input_copy_and_validation_and_dump"},
        {"timing_includes", "module_dispatch_device_work_input_transfer_wave_readMap_sync_and_output_transfer"}});
    return finite && filesOk && tensorCaptureOk && captureCountOk ? 0 : 6;
}
