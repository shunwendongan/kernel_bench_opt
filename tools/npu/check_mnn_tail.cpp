#include "omni_rknn_prefix.hpp"
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
using namespace MNN;using namespace MNN::Express;
static void write(const std::string&p,VARP y){y=_Convert(y,NCHW);auto info=y->getInfo();if(!info){std::exit(5);}auto data=y->readMap<float>();if(!data)std::exit(6);std::ofstream f(p,std::ios::binary);f.write((char*)data,info->size*4);std::cout<<p<<" shape=";for(int x:info->dim)std::cout<<x<<",";std::cout<<"\n";}
int main(int argc,char**argv){
 if(argc!=7){std::cerr<<"model weight mel.f32 frames output_base mode(prefix|tail)\n";return 2;}
 BackendConfig bc;bc.precision=BackendConfig::Precision_High;bc.memory=BackendConfig::Memory_Low;
 ScheduleConfig sc;sc.type=MNN_FORWARD_CPU;sc.numThread=4;sc.backendConfig=&bc;
 std::shared_ptr<Executor::RuntimeManager>rt(Executor::RuntimeManager::createRuntimeManager(sc));rt->setExternalFile(argv[2]);
 Module::Config cfg;cfg.shapeMutable=true;cfg.rearrange=true;
 int t=std::atoi(argv[4]),s=(t+1)/2;
 auto mel=_Input({1,128,t},NCHW,halide_type_of<float>());
 std::ifstream f(argv[3],std::ios::binary);f.read((char*)mel->writeMap<float>(),128*t*4);if(!f)return 3;
 auto mask=_Input({1,s,s},NCHW,halide_type_of<float>());auto mp=mask->writeMap<float>();for(int i=0;i<s;++i)for(int j=0;j<s;++j)mp[i*s+j]=i/100==j/100?0:-1e9;
 std::shared_ptr<Module>prefix(Module::load({"input_features"},{"/Mul_3_output_0"},argv[1],rt,&cfg));if(!prefix)return 4;
 auto mid=prefix->onForward({mel})[0];write(std::string(argv[5])+"_prefix.f32",mid);
 if(std::string(argv[6])=="prefix")return 0;
 std::shared_ptr<Module>tail(Module::load({"/Mul_3_output_0","attention_mask"},{"audio_embeds"},argv[1],rt,&cfg));if(!tail)return 7;
 auto m=_Input({1,1280,s},NCHW,halide_type_of<float>());auto midCpu=_Convert(mid,NCHW);std::memcpy(m->writeMap<float>(),midCpu->readMap<float>(),1280*s*4);
 auto y=tail->onForward({m,mask})[0];write(std::string(argv[5])+"_tail.f32",y);
 const char* rknnPath=std::getenv("OMNI_RKNN_AUDIO"); if(!rknnPath){std::cerr<<"Set OMNI_RKNN_AUDIO to the prefix model\n";return 9;}
 OmniRknnPrefix npu; if(!npu.load(rknnPath))return 9; auto nm=npu.forward(mel); if(nm.get()==nullptr)return 10;write(std::string(argv[5])+"_npu_prefix.f32",nm); auto ny=tail->onForward({nm,mask})[0];write(std::string(argv[5])+"_npu_tail.f32",ny);
 std::shared_ptr<Module>full(Module::load({"input_features","attention_mask"},{"audio_embeds"},argv[1],rt,&cfg));if(!full)return 8;
 auto z=full->onForward({mel,mask})[0];write(std::string(argv[5])+"_full.f32",z);
 return 0;
}
