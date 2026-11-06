#include "MNN_generated.h"
#include "core/ConvolutionCommon.hpp"
#include <MNN/expr/Module.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>
using namespace MNN;
using namespace MNN::Express;
static void save(const std::string& path,const void* p,size_t size) {
    std::ofstream f(path,std::ios::binary); f.write((const char*)p,size);
    if(!f) {std::cerr<<"write failed "<<path<<"\n"; std::exit(3);}
}
int main(int argc,char** argv) {
    if(argc!=4) {std::cerr<<"audio.mnn audio.mnn.weight output_dir\n";return 2;}
    std::ifstream f(argv[1],std::ios::binary);
    std::vector<char>b((std::istreambuf_iterator<char>(f)),{});
    std::unique_ptr<NetT> net(GetNet(b.data())->UnPack());
    for(int index:{4,8}) {
        auto& op=net->oplists.at(index); op->externalPath=argv[2];
        flatbuffers::FlatBufferBuilder fb;
        fb.Finish(Op::Pack(fb,op.get()));
        auto runtimeOp=flatbuffers::GetRoot<Op>(fb.GetBufferPointer());
        auto decoded=ConvolutionCommon::load(runtimeOp,nullptr,true);
        if(!decoded || !decoded->weightFloat.get()) {std::cerr<<"decode failed\n";return 4;}
        const std::string base=std::string(argv[3])+(index==4?"/mnn_conv1":"/mnn_conv2");
        save(base+"_weight.f32",decoded->weightFloat.get(),decoded->weightFloat.size()*sizeof(float));
        auto c=op->main.AsConvolution2D(); auto ext=c->external;
        std::ifstream wf(argv[2],std::ios::binary);wf.seekg(ext[0]+ext[1]+ext[2]);
        std::vector<char> bias(ext[3]);wf.read(bias.data(),bias.size());
        if(!wf) return 5;
        save(base+"_bias.f32",bias.data(),bias.size());
        std::cout<<"decoded "<<op->name<<" float_count="<<decoded->weightFloat.size()<<" bias_bytes="<<bias.size()<<"\n";
    }
    net->oplists.resize(11);
    net->oplists.erase(net->oplists.begin()+1); // unused original attention_mask Input
    net->outputName={"/Mul_3_output_0"};
    flatbuffers::FlatBufferBuilder fb;
    fb.Finish(Net::Pack(fb,net.get()));
    save(std::string(argv[3])+"/mnn_conv_prefix.mnn",fb.GetBufferPointer(),fb.GetSize());
    std::cout<<"PREFIX_EXTRACTED\n";
}
