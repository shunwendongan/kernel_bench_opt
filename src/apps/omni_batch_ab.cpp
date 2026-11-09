#include "llm/llm.hpp"
#include "jsonhpp/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <sys/resource.h>

using namespace MNN::Transformer;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

static double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}
static void saveJson(const fs::path& p, const Json& j) {
    std::ofstream f(p.string() + ".tmp");
    f << j.dump(2) << '\n'; f.close();
    fs::rename(p.string() + ".tmp", p);
}
static long memAvailable() {
    std::ifstream f("/proc/meminfo"); std::string key, unit; long value;
    while (f >> key >> value >> unit) if (key == "MemAvailable:") return value;
    return -1;
}
static long rss() {
    std::ifstream f("/proc/self/status"); std::string line;
    while (std::getline(f, line)) if (line.compare(0, 6, "VmRSS:") == 0) return std::stol(line.substr(6));
    return -1;
}
static bool saveWav(const fs::path& p, const std::vector<float>& wave) {
    if (wave.empty() || wave.size() > (UINT32_MAX - 36) / 2) return false;
    std::ofstream f(p, std::ios::binary);
    auto u16 = [&](uint16_t x) { char b[2] = {char(x), char(x >> 8)}; f.write(b, 2); };
    auto u32 = [&](uint32_t x) { char b[4] = {char(x), char(x >> 8), char(x >> 16), char(x >> 24)}; f.write(b, 4); };
    f.write("RIFF", 4); u32(36 + wave.size() * 2); f.write("WAVEfmt ", 8); u32(16); u16(1); u16(1);
    u32(24000); u32(48000); u16(2); u16(16); f.write("data", 4); u32(wave.size() * 2);
    for (float x : wave) { x = std::max(-1.0f, std::min(1.0f, x)); u16(uint16_t(int16_t(std::lround(x * 32767)))); }
    return f.good();
}

struct RunResult {
    Json metrics;
    std::vector<float> wave;
    std::vector<int> textTokens, speechTokens;
    bool success = false;
};

class Instance {
public:
    std::string name;
    int tile;
    std::string prefix;
    double loadSeconds = 0;
    std::mutex mutex;
    std::condition_variable ready;
    std::vector<float> wave;
    bool finalWave = false;
    double firstAudio = -1;
    size_t chunks = 0;
    Clock::time_point requestStart;
    std::unique_ptr<Llm, void(*)(Llm*)> llm{nullptr, Llm::destroy};

    Instance(std::string n, int t, std::string p) : name(std::move(n)), tile(t), prefix(std::move(p)) {}
    ~Instance() { llm.reset(); }
    void select() const {
        setenv("OMNI_BIGVGAN_DECONV_TILE", std::to_string(tile).c_str(), 1);
        if (prefix.empty() || prefix == "-") unsetenv("OMNI_RKNN_AUDIO");
        else setenv("OMNI_RKNN_AUDIO", prefix.c_str(), 1);
    }

    bool load(const fs::path& config, const fs::path& root) {
        select();
        const fs::path tmp = root / ("cache_" + name);
        fs::create_directories(tmp);
        const auto begin = Clock::now();
        llm.reset(Llm::createLLM(config.string()));
        if (!llm || !llm->set_config(Json({{"tmp_path", tmp.string()}, {"is_visual", false}}).dump()) || !llm->load()) return false;
        loadSeconds = seconds(begin, Clock::now());
        std::ofstream(root / ("effective_config_" + name + ".json")) << llm->dump_config() << '\n';
        saveJson(root / ("route_" + name + ".json"), {
            {"variant", name}, {"tile", tile}, {"rknn_prefix", prefix},
            {"rknn_enabled", !prefix.empty() && prefix != "-"},
            {"predit_sync", std::getenv("OMNI_PREDIT_SYNC") != nullptr},
            {"route_note", prefix == "-" ? "original MNN encoder runtime selected by unchanged config" : "RKNN audio prefix plus existing CPU tail"},
        });
        wave.reserve(1024 * 480);
        llm->setWavformCallback([this](const float* data, size_t count, bool last) {
            std::lock_guard<std::mutex> lock(mutex);
            if (data && count) {
                if (firstAudio < 0) firstAudio = seconds(requestStart, Clock::now());
                wave.insert(wave.end(), data, data + count); ++chunks;
            }
            if (last) { finalWave = true; ready.notify_all(); }
            return true;
        });
        return llm->getTalkerContext() != nullptr;
    }

    RunResult run(const ChatMessages& messages, const Json& item, const fs::path& dir, int pair, bool warmup) {
        select();
        const bool textOnly = item.at("task_type").get<std::string>() == "audio2text";
        fs::create_directories(dir);
        if (warmup) setenv("BIGVGAN_DISPATCH_LOG", (dir / "dispatch.jsonl").string().c_str(), 1);
        else unsetenv("BIGVGAN_DISPATCH_LOG");
        llm->reset();
        {
            std::lock_guard<std::mutex> lock(mutex);
            wave.clear(); finalWave = false; firstAudio = -1; chunks = 0;
        }
        const auto previousTalker = llm->getTalkerContext();
        const int64_t previousAudio = previousTalker ? previousTalker->audio_us : 0;
        const int64_t previousDit = previousTalker ? previousTalker->vision_us : 0;
        const long availableBefore = memAvailable(), rssBefore = rss();
        std::ostringstream generated;
        std::cout << "AB_REQUEST_START variant=" << name << " pair=" << pair << " warmup=" << warmup << std::endl;
        requestStart = Clock::now();
        llm->response(messages, &generated, nullptr, item.value("max_new_tokens", 1024));
        const auto thinkerDone = Clock::now();
        const auto thinker = llm->getContext();
        if (!textOnly && thinker->status != LlmStatus::INTERNAL_ERROR) {
            llm->generateWavform();
            std::unique_lock<std::mutex> lock(mutex);
            ready.wait_for(lock, std::chrono::seconds(600), [&] { return finalWave; });
        }
        const auto requestDone = Clock::now();
        RunResult result;
        const auto talker = llm->getTalkerContext();
        std::lock_guard<std::mutex> lock(mutex);
        result.wave = wave;
        result.textTokens = thinker->output_tokens;
        if (!textOnly && talker) result.speechTokens = talker->output_tokens;
        size_t nonfinite = 0;
        double energy = 0, peak = 0;
        for (float x : result.wave) {
            if (!std::isfinite(x)) { ++nonfinite; continue; }
            energy += double(x) * x; peak = std::max(peak, double(std::abs(x)));
        }
        const bool thinkerOk = thinker->status == LlmStatus::NORMAL_FINISHED || thinker->status == LlmStatus::MAX_TOKENS_FINISHED;
        const bool talkerOk = talker && talker->status != LlmStatus::INTERNAL_ERROR && talker->status != LlmStatus::TIMEOUT &&
                              talker->status != LlmStatus::USER_CANCEL && talker->status != LlmStatus::NOT_LOADED;
        const bool validOutput = textOnly ? !thinker->generate_str.empty() :
            talkerOk && finalWave && !result.wave.empty() && nonfinite == 0 && peak > 1e-6;
        result.success = thinkerOk && validOutput;
        struct rusage usage{}; getrusage(RUSAGE_SELF, &usage);
        result.metrics = {
            {"variant", name}, {"tile", tile}, {"rknn_prefix", prefix},
            {"rknn_enabled", !prefix.empty() && prefix != "-"},
            {"task_type", item.at("task_type")}, {"subset", item.at("subset")},
            {"text_only", textOnly}, {"pair", pair}, {"warmup", warmup},
            {"row", item.at("original_row_number")}, {"id", item.at("index")}, {"success", result.success},
            {"request_s", seconds(requestStart, requestDone)}, {"request_wall_s", seconds(requestStart, requestDone)}, {"ttfa_s", firstAudio < 0 ? Json(nullptr) : Json(firstAudio)},
            {"thinker_response_s", seconds(requestStart, thinkerDone)}, {"waveform_phase_s", seconds(thinkerDone, requestDone)},
            {"audio_encoder_including_prepare_s", thinker->audio_us / 1e6},
            {"thinker_prefill_s", thinker->prefill_us / 1e6}, {"thinker_decode_s", thinker->decode_us / 1e6},
            {"thinker_sample_s", thinker->sample_us / 1e6}, {"thinker_status", int(thinker->status)},
            {"thinker_truncated", thinker->status == LlmStatus::MAX_TOKENS_FINISHED},
            {"prompt_tokens", thinker->prompt_len}, {"text_tokens", thinker->gen_seq_len},
            {"text_token_ids", result.textTokens}, {"speech_token_ids", result.speechTokens},
            {"speech_tokens", result.speechTokens.size()}, {"text", thinker->generate_str},
            {"wave_samples", result.wave.size()}, {"wave_duration_s", result.wave.size() / 24000.0},
            {"wave_chunks", chunks}, {"wave_last_received", finalWave}, {"nonfinite_samples", nonfinite},
            {"wave_rms", result.wave.empty() ? 0 : std::sqrt(energy / result.wave.size())}, {"wave_peak", peak},
            {"wave_float_file", "waveform.f32"}, {"wave_float_dtype", "little_endian_float32"},
            {"memavailable_before_kib", availableBefore}, {"memavailable_after_kib", memAvailable()},
            {"process_rss_before_kib", rssBefore}, {"process_rss_after_kib", rss()},
            {"process_max_rss_kib", usage.ru_maxrss}, {"load_s", loadSeconds},
            {"profile_enabled", false}, {"capture_enabled", false}, {"resident_instances", 2},
            {"request_scope", textOnly ? "response_start_through_text_response_return" : "response_start_through_completed_waveform_callback_and_generateWavform_return"},
            {"timing_note", "host monotonic request timing; phase counters can overlap and must not be added"},
        };
        if (!textOnly && talker) {
            result.metrics["talker_status"] = int(talker->status);
            result.metrics["talker_prefill_s"] = talker->prefill_us / 1e6;
            result.metrics["talker_decode_s"] = talker->decode_us / 1e6;
            result.metrics["talker_sample_s"] = talker->sample_us / 1e6;
            result.metrics["token2wav_s"] = (talker->audio_us - previousAudio) / 1e6;
            result.metrics["dit_s"] = nullptr;
            result.metrics["dit_timing_note"] = "not separately instrumented in this Qwen2.5 path";
            result.metrics["talker_vision_counter_delta_s"] = (talker->vision_us - previousDit) / 1e6;
        }
        if (!textOnly && !finalWave) result.metrics["error"] = "waveform_completion_timeout_or_generation_failure";
        std::ofstream raw(dir / "waveform.f32", std::ios::binary);
        raw.write(reinterpret_cast<const char*>(result.wave.data()), result.wave.size() * sizeof(float));
        raw.close();
        const bool filesOk = raw.good() && (nonfinite == 0 && !result.wave.empty() ? saveWav(dir / "answer.wav", result.wave) : true);
        result.success = result.success && filesOk;
        result.metrics["success"] = result.success;
        result.metrics["unsafe_reuse"] = !textOnly && !finalWave;
        result.metrics["files_ok"] = filesOk;
        std::ofstream(dir / "answer.txt") << thinker->generate_str << '\n';
        std::ofstream(dir / "stream.txt") << generated.str();
        saveJson(dir / "metrics.json", result.metrics);
        std::cout << "AB_REQUEST_DONE variant=" << name << " pair=" << pair << " warmup=" << warmup
                  << " request_s=" << result.metrics["request_s"] << " speech_tokens=" << result.speechTokens.size()
                  << " success=" << result.success << std::endl;
        return result;
    }
};

static Json compare(const RunResult& old, const RunResult& next) {
    const bool sizeEqual = old.wave.size() == next.wave.size();
    const bool waveEqual = sizeEqual && (old.wave.empty() || std::memcmp(old.wave.data(), next.wave.data(), old.wave.size() * sizeof(float)) == 0);
    const bool textEqual = old.textTokens == next.textTokens, speechEqual = old.speechTokens == next.speechTokens;
    double sum = 0, maxError = 0, referenceEnergy = 0;
    if (sizeEqual) for (size_t i = 0; i < old.wave.size(); ++i) {
        const double d = double(old.wave[i]) - next.wave[i];
        sum += d * d; maxError = std::max(maxError, std::abs(d)); referenceEnergy += double(old.wave[i]) * old.wave[i];
    }
    return {{"text_tokens_equal", textEqual}, {"speech_tokens_equal", speechEqual},
        {"wave_size_equal", sizeEqual}, {"wave_bitwise_equal", waveEqual},
        {"wave_max_abs", sizeEqual ? Json(maxError) : Json(nullptr)},
        {"wave_rmse", sizeEqual && !old.wave.empty() ? Json(std::sqrt(sum / old.wave.size())) : Json(nullptr)},
        {"wave_relative_rmse", sizeEqual && referenceEnergy ? Json(std::sqrt(sum / referenceEnergy)) : Json(nullptr)},
        {"outputs_valid", old.success && next.success},
        {"matched_generated_workload", old.success && next.success && textEqual && speechEqual && sizeEqual},
        {"strict_correctness_pass", old.success && next.success && textEqual && speechEqual && waveEqual},
        {"passed", old.success && next.success && textEqual && speechEqual && waveEqual},
        {"interpretation", "System-route comparison; different generated workload is not an isolated kernel speedup"}};
}

static ChatMessages messagesFor(const Json& item, const fs::path& bundle) {
    ChatMessages messages;
    messages.emplace_back("system", item.at("system_prompt").get<std::string>());
    for (const auto& message : item.at("chat_messages")) {
        std::string content = message.at("content").get<std::string>();
        const std::string marker = "{BUNDLE}", root = bundle.string();
        size_t pos = 0;
        while ((pos = content.find(marker, pos)) != std::string::npos) {
            content.replace(pos, marker.size(), root); pos += root.size();
        }
        messages.emplace_back(message.at("role").get<std::string>(), content);
    }
    return messages;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: omni_batch_ab CONFIG MANIFEST BUNDLE OUTPUT [PAIRS=1] [OLD_PREFIX=-] [NEW_PREFIX=-]\n";
        return 2;
    }
    const fs::path config = fs::absolute(argv[1]), manifest = fs::absolute(argv[2]);
    const fs::path bundle = fs::absolute(argv[3]), out = fs::absolute(argv[4]);
    const int pairs = argc > 5 ? std::stoi(argv[5]) : 1;
    const std::string oldPrefix = argc > 6 ? argv[6] : "-", newPrefix = argc > 7 ? argv[7] : "-";
    if (pairs < 1) return 2;
    if (fs::exists(out)) {
        std::cerr << "OUTPUT already exists; use a fresh directory to preserve previous results\n";
        return 2;
    }
    std::ifstream input(manifest);
    if (!input) return 3;
    std::vector<Json> items;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        auto item = Json::parse(line, nullptr, false);
        if (item.is_discarded() || !item.is_object() || !item.contains("original_row_number") ||
            !item.contains("task_type") || !item.contains("index") || !item.contains("chat_messages")) return 3;
        items.push_back(std::move(item));
    }
    if (items.empty()) return 3;
    fs::create_directories(out);
    for (const char* variable : {"OMNI_PROFILE_PATH", "OMNI_CAPTURE_DIR", "DECONV_CAPTURE_DIR",
                                 "OMNI_RKNN_PERF_DIR", "BIGVGAN_DISPATCH_LOG"}) unsetenv(variable);
    if (!std::getenv("OMNI_TEST_SEED")) setenv("OMNI_TEST_SEED", "20261109", 1);
    Json state = {{"state", "loading"}, {"config", config.string()}, {"manifest", manifest.string()},
        {"sample_count", items.size()}, {"pairs_per_sample", pairs}, {"seed", std::getenv("OMNI_TEST_SEED")},
        {"old_prefix", oldPrefix}, {"new_prefix", newPrefix}, {"old_tile", 0}, {"new_tile", 4},
        {"predit_sync", std::getenv("OMNI_PREDIT_SYNC") != nullptr},
        {"warmup_per_instance", 1}, {"formal_pairs_completed", 0}, {"formal_samples_completed", 0},
        {"order", "warmup old,new on first audio2audio if present; formal pairs alternate old,new then new,old"},
        {"profile_enabled", false}, {"capture_enabled", false}, {"resident_instances", 2},
        {"comparison_kind", "original_MNN_audio_route_and_old_deconv_vs_RKNN_prefix_and_tile4"},
        {"correctness_note", "NPU prefix may differ numerically; mismatched output is recorded and not treated as strict kernel speedup"}};
    saveJson(out / "run.json", state);
    Instance old("old", 0, oldPrefix), candidate("new", 4, newPrefix);
    if (!old.load(config, out) || !candidate.load(config, out)) {
        state["state"] = "load_failed"; saveJson(out / "run.json", state); return 4;
    }
    state["distinct_executor_objects"] = old.llm->getExecutor().get() != candidate.llm->getExecutor().get();
    state["state"] = "running"; saveJson(out / "run.json", state);
    std::ofstream results(out / "results.jsonl");
    Json comparisons = Json::array();
    bool runtimeOk = true, strictAll = true, workloadAll = true;
    int formalPairs = 0, samplesCompleted = 0;
    auto warm = std::find_if(items.begin(), items.end(), [](const Json& item) {
        return item.at("task_type").get<std::string>() != "audio2text";
    });
    if (warm == items.end()) warm = items.begin();
    state["warmup_row"] = warm->at("original_row_number");
    state["warmup_id"] = warm->at("index");
    saveJson(out / "run.json", state);
    auto execute = [&](Instance& instance, const Json& item, const fs::path& path, int pair, bool warmup) {
        auto value = instance.run(messagesFor(item, bundle), item, path, pair, warmup);
        value.metrics["output_directory"] = fs::relative(path, out).generic_string();
        // The added directory field must be present both in per-request and JSONL records.
        saveJson(path / "metrics.json", value.metrics);
        results << value.metrics.dump() << '\n'; results.flush();
        return value;
    };
    {
        auto a = execute(old, *warm, out / "warmup_old", -1, true);
        if (!a.success) runtimeOk = false;
        if (runtimeOk) {
            auto b = execute(candidate, *warm, out / "warmup_new", -1, true);
            runtimeOk = b.success;
            Json check = {{"warmup", true}, {"row", warm->at("original_row_number")},
                          {"id", warm->at("index")}, {"paired", compare(a, b)}};
            comparisons.push_back(check);
            saveJson(out / "comparisons.json", comparisons);
        }
    }
    for (size_t sample = 0; sample < items.size() && runtimeOk; ++sample) {
        const Json& item = items[sample];
        const fs::path sampleDir = out / ("sample_" + std::to_string(sample + 1));
        fs::create_directories(sampleDir);
        saveJson(sampleDir / "input.json", item);
        state["current_sample_ordinal"] = sample + 1;
        state["current_row"] = item.at("original_row_number");
        state["current_id"] = item.at("index");
        saveJson(out / "run.json", state);
        int pairsForSample = 0;
        for (int pair = 0; pair < pairs && runtimeOk; ++pair) {
            const bool newFirst = ((sample * size_t(pairs) + size_t(pair)) % 2) != 0;
            const std::string prefix = "pair_" + std::to_string(pair) + "_";
            RunResult a, b;
            auto perform = [&](bool isNew) {
                auto& instance = isNew ? candidate : old;
                auto value = execute(instance, item, sampleDir / (prefix + instance.name), pair, false);
                if (isNew) b = std::move(value); else a = std::move(value);
            };
            perform(newFirst);
            if (!(newFirst ? b.success : a.success)) { runtimeOk = false; break; }
            perform(!newFirst);
            if (!(newFirst ? a.success : b.success)) { runtimeOk = false; break; }
            const Json paired = compare(a, b);
            Json check = {{"warmup", false}, {"sample_ordinal", sample + 1},
                {"row", item.at("original_row_number")}, {"id", item.at("index")},
                {"task_type", item.at("task_type")}, {"pair", pair}, {"new_first", newFirst}, {"paired", paired}};
            comparisons.push_back(check);
            ++formalPairs; ++pairsForSample;
            strictAll = strictAll && paired.at("strict_correctness_pass").get<bool>();
            workloadAll = workloadAll && paired.at("matched_generated_workload").get<bool>();
            saveJson(out / "comparisons.json", comparisons);
            state["formal_pairs_completed"] = formalPairs;
            state["strict_correctness_all_formal"] = strictAll;
            state["matched_generated_workload_all_formal"] = workloadAll;
            saveJson(out / "run.json", state);
        }
        if (pairsForSample == pairs) ++samplesCompleted;
        state["formal_samples_completed"] = samplesCompleted;
        saveJson(out / "run.json", state);
    }
    old.select(); old.llm.reset();
    candidate.select(); candidate.llm.reset();
    state["state"] = runtimeOk && samplesCompleted == int(items.size()) ? "completed" : "aborted_runtime_failure";
    state["formal_pairs_completed"] = formalPairs;
    state["formal_samples_completed"] = samplesCompleted;
    state["strict_correctness_all_formal"] = strictAll && formalPairs > 0;
    state["matched_generated_workload_all_formal"] = workloadAll && formalPairs > 0;
    state["performance_interpretation"] = "Report all system-route pairs; only strict equal-output pairs support unchanged-output optimization claims";
    saveJson(out / "run.json", state);
    return runtimeOk && samplesCompleted == int(items.size()) ? 0 : 6;
}
