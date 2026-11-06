#pragma once
// Original Omni audio Conv1 -> GELU -> Conv2 -> GELU, using the deployed
// MNN 4-bit weights decoded to float, compiled by RKNN as FP16.
// Input/output stay in the MNN process. This adapter owns one RKNN context.
#include <rknn_api.h>
#include <MNN/expr/ExprCreator.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <string>
class OmniRknnPrefix {
    rknn_context ctx_ = 0;
    unsigned perf_sequence_ = 0;
    const char* perf_dir_ = nullptr;
    static double us(std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count();
    }
public:
    ~OmniRknnPrefix() { if(ctx_)rknn_destroy(ctx_); }
    bool load(const char* path) {
        perf_dir_=std::getenv("OMNI_RKNN_PERF_DIR");
        if(rknn_init(&ctx_,const_cast<char*>(path),0,perf_dir_?RKNN_FLAG_COLLECT_PERF_MASK:0,nullptr)!=RKNN_SUCC)return false;
        rknn_input_output_num io{};
        if(rknn_query(ctx_,RKNN_QUERY_IN_OUT_NUM,&io,sizeof(io))!=0 || io.n_input!=2 || io.n_output!=1)return false;
        rknn_tensor_attr a{}; a.index=0;
        if(rknn_query(ctx_,RKNN_QUERY_INPUT_ATTR,&a,sizeof(a))!=0 || a.n_elems!=128*516 || a.fmt!=RKNN_TENSOR_NHWC)return false;
        a={};a.index=0;
        if(rknn_query(ctx_,RKNN_QUERY_OUTPUT_ATTR,&a,sizeof(a))!=0 || a.n_elems!=1280*256 || a.fmt!=RKNN_TENSOR_NCHW)return false;
        rknn_sdk_version version{};rknn_query(ctx_,RKNN_QUERY_SDK_VERSION,&version,sizeof(version));
        std::printf("OMNI_RKNN_INIT api=%s driver=%s graph=conv1_gelu_conv2_gelu input=NHWC output=NCHW\n",version.api_version,version.drv_version);
        return true;
    }
    MNN::Express::VARP forward(MNN::Express::VARP mel) {
        using namespace MNN::Express;
        if(!ctx_ || mel.get() == nullptr || !mel->getInfo())return nullptr;
        auto d=mel->getInfo()->dim;
        if(d.size()!=3 || d[0]!=1 || d[1]!=128 || d[2]<1)return nullptr;
        const int t=d[2], s=(t+1)/2;
        auto begin=std::chrono::steady_clock::now();
        auto x=_Convert(mel,NCHW);const float*src=x->readMap<float>();if(!src)return nullptr;
        auto out=_Input({1,1280,s},NCHW,halide_type_of<float>());float*dst=out->writeMap<float>();
        std::vector<float>tile(516*128),mask(516);int calls=0;double prepareUs=0,setUs=0,runUs=0,getUs=0,copyUs=0;
        for(int start=0;start<s;start+=256) {
            auto st=std::chrono::steady_clock::now();
            std::fill(tile.begin(),tile.end(),0);std::fill(mask.begin(),mask.end(),0);
            for(int w=0;w<516;++w) {int pos=2*start-2+w;if(pos>=0 && pos<t){mask[w]=1;for(int c=0;c<128;++c)tile[w*128+c]=src[c*t+pos];}}
            prepareUs+=us(st);
            rknn_input in[2]{};
            in[0].index=0;in[0].buf=tile.data();in[0].size=tile.size()*sizeof(float);in[0].type=RKNN_TENSOR_FLOAT32;in[0].fmt=RKNN_TENSOR_NHWC;
            in[1].index=1;in[1].buf=mask.data();in[1].size=mask.size()*sizeof(float);in[1].type=RKNN_TENSOR_FLOAT32;in[1].fmt=RKNN_TENSOR_NHWC;
            st=std::chrono::steady_clock::now();int ret=rknn_inputs_set(ctx_,2,in);setUs+=us(st);if(ret)return nullptr;
            st=std::chrono::steady_clock::now();ret=rknn_run(ctx_,nullptr);runUs+=us(st);if(ret)return nullptr;
            rknn_output y{};y.index=0;y.want_float=1;
            st=std::chrono::steady_clock::now();ret=rknn_outputs_get(ctx_,1,&y,nullptr);getUs+=us(st);if(ret)return nullptr;
            if(perf_dir_) {
                rknn_perf_detail detail{};
                const int query_ret=rknn_query(ctx_,RKNN_QUERY_PERF_DETAIL,&detail,sizeof(detail));
                const std::string stem=std::string(perf_dir_)+"/tile_"+std::to_string(perf_sequence_++);
                if(query_ret==RKNN_SUCC && detail.perf_data) {
                    std::ofstream report(stem+".txt",std::ios::binary);
                    report.write(detail.perf_data,detail.data_len);
                }
                std::ofstream metadata(stem+".json");
                metadata<<"{\"mel_frames\":"<<t<<",\"output_tile_start\":"<<start
                  <<",\"query_return\":"<<query_ret<<"}\n";
            }
            if(y.size!=1280*256*sizeof(float)){rknn_outputs_release(ctx_,1,&y);return nullptr;}
            st=std::chrono::steady_clock::now();int n=std::min(256,s-start);
            for(int c=0;c<1280;++c)std::memcpy(dst+c*s+start,static_cast<float*>(y.buf)+c*256,n*sizeof(float));
            ret=rknn_outputs_release(ctx_,1,&y);copyUs+=us(st);if(ret)return nullptr;++calls;
        }
        std::printf("OMNI_RKNN_REQUEST frames=%d output_frames=%d calls=%d total_us=%.1f prepare_us=%.1f inputs_set_us=%.1f run_us=%.1f outputs_get_us=%.1f copy_release_us=%.1f\n",t,s,calls,us(begin),prepareUs,setUs,runUs,getUs,copyUs);
        return out;
    }
};
