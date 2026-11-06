#include "llm/llm.hpp"
#include "jsonhpp/json.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>
#include <sys/resource.h>

using namespace MNN::Transformer;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b-a).count();
}
static long memAvailable() {
    std::ifstream f("/proc/meminfo"); std::string key, unit; long value;
    while (f >> key >> value >> unit) if (key == "MemAvailable:") return value;
    return -1;
}
static long rss() {
    std::ifstream f("/proc/self/status"); std::string line;
    while (std::getline(f,line)) if (line.compare(0,6,"VmRSS:")==0) return std::stol(line.substr(6));
    return -1;
}
static void saveJson(const std::filesystem::path& p, const Json& j) {
    std::ofstream f(p.string()+".tmp"); f << j.dump(2) << '\n'; f.close();
    std::filesystem::rename(p.string()+".tmp",p);
}
static bool saveWav(const std::filesystem::path& p, const std::vector<float>& wave) {
    if (wave.empty() || wave.size() > (UINT32_MAX-36)/2) return false;
    std::ofstream f(p,std::ios::binary);
    auto u16=[&](uint16_t x) { char b[2]={char(x),char(x>>8)}; f.write(b,2); };
    auto u32=[&](uint32_t x) { char b[4]={char(x),char(x>>8),char(x>>16),char(x>>24)}; f.write(b,4); };
    f.write("RIFF",4); u32(36+wave.size()*2); f.write("WAVEfmt ",8); u32(16); u16(1); u16(1);
    u32(24000); u32(48000); u16(2); u16(16); f.write("data",4); u32(wave.size()*2);
    for (float x: wave) { x=std::max(-1.0f,std::min(1.0f,x)); u16(uint16_t(int16_t(std::lround(x*32767)))); }
    return f.good();
}
int main(int argc,char** argv) {
    if (argc < 5) { std::cerr << "usage: vcb_eval CONFIG MANIFEST BUNDLE OUTPUT [START_ROW=1] [COUNT=300] [TEXT_ONLY=0]\n"; return 2; }
    const std::filesystem::path config=std::filesystem::absolute(argv[1]), manifest=argv[2], bundle=argv[3], out=argv[4];
    const int start=argc>5?std::stoi(argv[5]):1, count=argc>6?std::stoi(argv[6]):300;
    const bool forceTextOnly=argc>7 && std::stoi(argv[7])!=0;
    std::filesystem::create_directories(out);
    Json startup={{"state","loading"},{"config",config.string()},{"manifest",manifest.string()},{"start_row",start},{"count",count},{"force_text_only",forceTextOnly}};
    saveJson(out/"run.json",startup);
    auto loadStart=Clock::now();
    std::unique_ptr<Llm,void(*)(Llm*)> llm(Llm::createLLM(config.string()), Llm::destroy);
    if (llm) llm->set_config(forceTextOnly ? "{\"is_visual\":false,\"has_talker\":false}" : "{\"is_visual\":false}");
    if (!llm || !llm->load()) { startup["state"]="load_failed"; saveJson(out/"run.json",startup); return 3; }
    const double loadSeconds=seconds(loadStart,Clock::now());
    std::ofstream(out/"effective_config.json") << llm->dump_config() << '\n';
    std::mutex waveMutex; std::condition_variable waveReady;
    std::vector<float> wave; bool finalWave=false; double firstAudio=-1; size_t chunks=0;
    Clock::time_point requestStart;
    if (!forceTextOnly) llm->setWavformCallback([&](const float* p,size_t n,bool last) {
        std::lock_guard<std::mutex> lock(waveMutex);
        if (n && p) { if (firstAudio<0) firstAudio=seconds(requestStart,Clock::now()); wave.insert(wave.end(),p,p+n); ++chunks; }
        if (last) { finalWave=true; waveReady.notify_all(); }
        return true;
    });
    startup["state"]="running"; startup["load_s"]=loadSeconds; saveJson(out/"run.json",startup);
    std::ifstream mf(manifest); if (!mf) return 4;
    std::string line; int attempted=0, succeeded=0; bool abortRun=false;
    std::ofstream summary(out/"results.jsonl",std::ios::app);
    while (std::getline(mf,line)) {
        Json item=Json::parse(line,nullptr,false);
        if (item.is_discarded()) { std::cerr << "invalid manifest JSON\n"; return 5; }
        const int row=item.at("original_row_number").get<int>();
        if (row<start || row>=start+count) continue;
        const std::string id=item.at("index").get<std::string>();
        const auto dir=out/(std::to_string(row)+"_"+id);
        if (std::filesystem::exists(dir/"metrics.json")) continue;
        std::filesystem::create_directories(dir);
        const bool textOnly=forceTextOnly || item.at("task_type").get<std::string>()=="audio2text";
        ChatMessages messages;
        messages.emplace_back("system",item.at("system_prompt").get<std::string>());
        for(const auto& message:item.at("chat_messages")) {
            std::string content=message.at("content").get<std::string>();
            const std::string marker="{BUNDLE}";
            size_t pos=0;
            while((pos=content.find(marker,pos))!=std::string::npos) {
                const std::string root=std::filesystem::absolute(bundle).string();
                content.replace(pos,marker.size(),root); pos+=root.size();
            }
            messages.emplace_back(message.at("role").get<std::string>(),content);
        }
        Json inputOnly={{"task_type",item.at("task_type")},{"messages",item.at("chat_messages")},
            {"prepared_audios",item.at("prepared_audios")}};
        saveJson(dir/"input.json",inputOnly);
        Json result={{"row",row},{"id",id},{"state","started"},{"prepared_audios",item.at("prepared_audios")},{"task_type",item.at("task_type")},{"subset",item.at("subset")},{"input_duration_s",item.at("prepared_duration_s")},{"load_s",loadSeconds},{"resident",true},{"text_only",textOnly},{"profile_enabled",std::getenv("OMNI_PROFILE_PATH")!=nullptr}};
        saveJson(dir/"state.json",result);
        llm->reset();
        { std::lock_guard<std::mutex> lock(waveMutex); wave.clear(); finalWave=false; firstAudio=-1; chunks=0; }
        const auto tc0=llm->getTalkerContext();
        const int64_t talkerAudioBefore=tc0?tc0->audio_us:0, talkerDitBefore=tc0?tc0->vision_us:0;
        std::atomic<bool> sampling{true}; std::atomic<long> minimumAvailable{memAvailable()}, peakRss{rss()};
        std::thread monitor([&] { while(sampling.load()) { long a=memAvailable(),r=rss(); if(a>0 && a<minimumAvailable) minimumAvailable=a; if(r>peakRss) peakRss=r; std::this_thread::sleep_for(std::chrono::milliseconds(500)); } });
        requestStart=Clock::now();
        std::cout << "SAMPLE_START row=" << row << " id=" << id << std::endl;
        std::ostringstream text;
        llm->response(messages,&text,nullptr,item.value("max_new_tokens",1024));
        auto thinkerDone=Clock::now();
        const auto c=llm->getContext();
        std::ofstream(dir/"answer.txt") << c->generate_str << '\n';
        std::cout << "THINKER_DONE row=" << row << " response_s=" << seconds(requestStart,thinkerDone) << " text=" << c->generate_str << std::endl;
        if (!textOnly && c->status!=LlmStatus::INTERNAL_ERROR) {
            llm->generateWavform();
            std::unique_lock<std::mutex> lock(waveMutex);
            waveReady.wait_for(lock,std::chrono::seconds(600),[&]{return finalWave;});
        }
        auto requestDone=Clock::now(); sampling=false; monitor.join();
        std::unique_lock<std::mutex> waveLock(waveMutex);
        if (!textOnly && !finalWave) abortRun=true;
        result["request_s"]=seconds(requestStart,requestDone);
        result["thinker_response_s"]=seconds(requestStart,thinkerDone);
        result["waveform_phase_s"]=seconds(thinkerDone,requestDone);
        result["ttfa_s"]=firstAudio<0?Json(nullptr):Json(firstAudio);
        result["audio_encoder_including_prepare_s"]=c->audio_us/1e6;
        result["thinker_prefill_s"]=c->prefill_us/1e6; result["thinker_decode_s"]=c->decode_us/1e6;
        result["thinker_sample_s"]=c->sample_us/1e6; result["thinker_status"]=static_cast<int>(c->status);
        result["prompt_tokens"]=c->prompt_len; result["text_tokens"]=c->gen_seq_len;
        result["text_token_ids"]=c->output_tokens; result["text"]=c->generate_str;
        result["thinker_truncated"]=c->status==LlmStatus::MAX_TOKENS_FINISHED;
        const auto tc=llm->getTalkerContext();
        if (tc && !textOnly) { result["speech_tokens"]=tc->output_tokens.size(); result["speech_token_ids"]=tc->output_tokens; result["talker_prefill_s"]=tc->prefill_us/1e6;
            result["talker_decode_s"]=tc->decode_us/1e6; result["talker_status"]=static_cast<int>(tc->status);
            result["token2wav_s"]= (tc->audio_us-talkerAudioBefore)/1e6; result["dit_s"]=nullptr;
            result["dit_timing_note"]="not separately instrumented in this Qwen2.5 path";
            result["talker_vision_counter_delta_s"]=(tc->vision_us-talkerDitBefore)/1e6;
            result["talker_limit_reached"]=tc->output_tokens.size()>=1024;
        }
        size_t nonfinite=0; double energy=0,peak=0;
        for (float x:wave) { if(!std::isfinite(x)) {++nonfinite;continue;} energy+=double(x)*x; peak=std::max(peak,double(std::abs(x))); }
        result["wave_samples"]=wave.size(); result["wave_duration_s"]=wave.size()/24000.0;
        result["wave_chunks"]=chunks; result["wave_last_received"]=finalWave; result["nonfinite_samples"]=nonfinite;
        result["wave_rms"]=wave.empty()?0:std::sqrt(energy/wave.size()); result["wave_peak"]=peak;
        result["minimum_memavailable_kib"]=minimumAvailable.load(); result["sampled_peak_rss_kib"]=peakRss.load();
        struct rusage usage{}; getrusage(RUSAGE_SELF,&usage); result["process_max_rss_kib"]=usage.ru_maxrss;
        result["rtf_vs_input"]=result["request_s"].get<double>()/item.at("prepared_duration_s").get<double>();
        bool valid=textOnly? !c->generate_str.empty() : finalWave && !wave.empty() && !nonfinite && peak>1e-6;
        if (!textOnly && valid) valid=saveWav(dir/"answer.wav",wave);
        const bool thinkerOk=c->status==LlmStatus::NORMAL_FINISHED || c->status==LlmStatus::MAX_TOKENS_FINISHED;
        const bool talkerOk=textOnly || (tc && tc->status!=LlmStatus::INTERNAL_ERROR && tc->status!=LlmStatus::TIMEOUT && tc->status!=LlmStatus::USER_CANCEL && tc->status!=LlmStatus::NOT_LOADED);
        const bool success=valid && thinkerOk && talkerOk;
        if (abortRun) result["error"]="waveform_completion_timeout; stopping to prevent cross-request state reuse";
        result["state"]=success?"success":"failed"; result["success"]=success;
        std::ofstream(dir/"answer.txt") << c->generate_str << '\n';
        std::ofstream(dir/"stream.txt") << text.str();
        saveJson(dir/"metrics.json",result); summary<<result.dump()<<'\n'; summary.flush();
        ++attempted; if(success) ++succeeded;
        std::cout << "SAMPLE_DONE row=" << row << " success=" << success << " request_s=" << result["request_s"] << " speech_s=" << result["wave_duration_s"] << " text=" << c->generate_str << std::endl;
        waveLock.unlock();
        if (abortRun) break;
    }
    llm.reset(); // Join waveform workers while captured state is still alive.
    startup["requested_count"]=count; startup["state"]=abortRun?"aborted":"completed"; startup["attempted_this_process"]=attempted; startup["succeeded_this_process"]=succeeded;
    saveJson(out/"run.json",startup);
    return attempted==succeeded && attempted==count?0:6;
}
