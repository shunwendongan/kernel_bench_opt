#pragma once
// Optional diagnostic instrumentation; ordinary inference does not install callbacks or synchronize.
#include <MNN/Interpreter.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>
#include "core/TensorUtils.hpp"
#include "core/Backend.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

namespace MNN { namespace Transformer { namespace OmniProfile {
using Clock = std::chrono::steady_clock;
inline bool enabled() { const char* p=std::getenv("OMNI_PROFILE_PATH"); return p && *p; }
inline std::string quote(const std::string& s) {
    std::string r="\""; for(unsigned char c:s) {
        if(c=='"'||c=='\\') {r+='\\';r+=char(c);} else if(c=='\n') r+="\\n";
        else if(c=='\r') r+="\\r"; else if(c=='\t') r+="\\t";
        else if(c<32) {char b[7];std::snprintf(b,sizeof(b),"\\u%04x",c);r+=b;} else r+=char(c);
    } return r+'"';
}
struct Sink {
    std::mutex mutex; FILE* file=nullptr;
    Sink() { const char* p=std::getenv("OMNI_PROFILE_PATH"); if(p && *p) file=std::fopen(p,"a"); }
    ~Sink() { if(file) std::fclose(file); }
};
inline Sink& sink() { static Sink s;return s; }
inline void emit(const std::string& s) {
    auto& log=sink(); std::lock_guard<std::mutex> lock(log.mutex);
    if(log.file) { std::fwrite(s.data(),1,s.size(),log.file); std::fputc('\n',log.file); }
}
inline const char*& stage() { static thread_local const char* s="unscoped";return s; }
struct Scope {
    const char* previous; bool active;
    explicit Scope(const char* s):previous(stage()),active(enabled()) {if(active) stage()=s;}
    ~Scope(){if(active) stage()=previous;}
};
inline std::string shapes(const std::vector<Tensor*>& ts) {
    std::ostringstream o;o<<'[';bool comma=false;for(auto t:ts){if(comma)o<<',';comma=true;
        if(!t){o<<"null";continue;}o<<'[';auto ds=t->shape();for(size_t i=0;i<ds.size();++i){if(i)o<<',';o<<ds[i];}o<<']';}o<<']';return o.str();
}
inline std::string backends(const std::vector<Tensor*>& ts) {
    std::ostringstream o;o<<'[';bool comma=false;for(auto t:ts){if(comma)o<<',';comma=true;
        auto d=t?TensorUtils::getDescribeOrigin(t):nullptr;auto b=d?d->getBackend():nullptr;
        if(b)o<<int(b->type());else o<<"null";}o<<']';return o.str();
}
inline void attach(const std::shared_ptr<Express::Executor>& executor,const char* label) {
    if(!enabled() || !executor)return;
    struct State {Clock::time_point start;std::string inputs,scope;};
    auto state=std::make_shared<State>();std::string owner(label);
    executor->setCallBack(
      [state](const std::vector<Tensor*>& inputs,const OperatorInfo*) {
        for(auto t:inputs)if(t)t->wait(Tensor::MAP_TENSOR_READ,true);
        state->inputs=shapes(inputs);state->scope=stage();state->start=Clock::now();return true;
      },
      [state,owner](const std::vector<Tensor*>& outputs,const OperatorInfo* op) {
        for(auto t:outputs)if(t)t->wait(Tensor::MAP_TENSOR_READ,true);
        double us=std::chrono::duration<double,std::micro>(Clock::now()-state->start).count();
        std::ostringstream o;o<<"{\"kind\":\"op\",\"executor\":"<<quote(owner)<<",\"stage\":"<<quote(state->scope)
          <<",\"name\":"<<quote(op->name())<<",\"type\":"<<quote(op->type())<<",\"elapsed_us\":"<<us
          <<",\"inputs\":"<<state->inputs<<",\"outputs\":"<<shapes(outputs)
          <<",\"output_tensor_backends\":"<<backends(outputs)<<",\"synchronized\":true}";
        emit(o.str());return true;
      });
}
inline void debug(const std::shared_ptr<Express::Executor::RuntimeManager>& manager,int requestedBackend,const char* label) {
    if(!enabled() || !manager)return;
    manager->setMode(Interpreter::Session_Debug);
    emit("{\"kind\":\"runtime\",\"module\":"+quote(label)+",\"requested_backend\":"+std::to_string(requestedBackend)+"}");
}
inline Express::VARPS forward(Express::Module* module,const Express::VARPS& inputs,const char* label) {
    if(!enabled())return module->onForward(inputs);
    Scope scope(label);auto start=Clock::now();auto result=module->onForward(inputs);
    Express::Variable::compute(result);
    std::vector<Tensor*> tensors;for(auto v:result)if(v.get()){auto t=const_cast<Tensor*>(v->getTensor());if(t){t->wait(Tensor::MAP_TENSOR_READ,true);tensors.push_back(t);}}
    double us=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
    std::ostringstream o;o<<"{\"kind\":\"module_api\",\"module\":"<<quote(label)<<",\"elapsed_us\":"<<us
       <<",\"output_tensor_backends\":"<<backends(tensors)<<",\"synchronized\":true}";emit(o.str());return result;
}
}}}
