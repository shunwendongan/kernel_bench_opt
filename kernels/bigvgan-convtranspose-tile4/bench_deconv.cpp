// Standalone OpenCL replay for the deployed BigVGAN first ConvTranspose.
// Build against MNN/3rd_party/OpenCLHeaders and the board's libOpenCL.
// The baseline and candidate consume the identical resident buffers.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>

using Clock = std::chrono::steady_clock;
static void check(cl_int x, const char *what) {
    if (x != CL_SUCCESS) throw std::runtime_error(std::string(what)+": "+std::to_string(x));
}
static std::string quote(const std::string& x) {
    std::string out="\"";
    for (char c:x) { if (c=='\\' || c=='\"') out+='\\'; if(c=='\n') out+="\\n"; else out+=c; }
    return out+'\"';
}
static std::vector<char> read(const std::string& name) {
    std::ifstream f(name, std::ios::binary|std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open "+name);
    const auto n=f.tellg(); if(n<0) throw std::runtime_error("Cannot size "+name);
    std::vector<char> x(static_cast<size_t>(n)); f.seekg(0); f.read(x.data(),x.size());
    if (!f) throw std::runtime_error("Cannot read "+name); return x;
}
template<class T> static std::vector<T> typed(const std::string& name, size_t expected) {
    auto b=read(name); if(b.size()!=expected*sizeof(T))
        throw std::runtime_error(name+" wrong size: "+std::to_string(b.size())+" expected "+std::to_string(expected*sizeof(T)));
    std::vector<T> x(expected); std::memcpy(x.data(),b.data(),b.size());return x;
}
template<class T> static void save(const std::string& path, const std::vector<T>& v) {
    std::ofstream f(path,std::ios::binary); f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));
    if(!f) throw std::runtime_error("Cannot write "+path);
}
static uint16_t f2h(float f) {
    uint32_t v; std::memcpy(&v,&f,4);
    uint32_t sign=(v>>16)&0x8000u, e=(v>>23)&255u, m=v&0x7fffffu;
    if(e==255) return static_cast<uint16_t>(sign|0x7c00u|(m?0x0200u:0));
    int he=int(e)-127+15;
    if(he>=31) return static_cast<uint16_t>(sign|0x7c00u);
    if(he<=0) {
        if(he < -10) return static_cast<uint16_t>(sign);
        m|=0x800000u; const int sh=14-he; uint32_t q=m>>sh, rem=m&((1u<<sh)-1);
        const uint32_t half=1u<<(sh-1); q+=(rem>half || (rem==half && (q&1)));
        return static_cast<uint16_t>(sign|q);
    }
    uint32_t q=m>>13, rem=m&8191u; q+=(rem>4096u || (rem==4096u && (q&1)));
    if(q==1024u){q=0;he++;}return static_cast<uint16_t>(sign|(uint32_t(he)<<10)|q);
}
static float h2f(uint16_t h) {
    uint32_t sign=uint32_t(h&0x8000u)<<16, e=(h>>10)&31u, m=h&1023u, v;
    if(e==0){if(!m)v=sign;else{int exp=-14;while(!(m&1024u)){m<<=1;--exp;}v=sign|(uint32_t(exp+127)<<23)|((m&1023u)<<13);}}
    else if(e==31)v=sign|0x7f800000u|(m<<13); else v=sign|((e+112)<<23)|(m<<13);
    float f; std::memcpy(&f,&v,4);return f;
}
static std::array<size_t,3> dims(std::string s) {
    std::replace(s.begin(),s.end(),',',' '); std::istringstream f(s);std::array<size_t,3> a{};
    if(!(f>>a[0]>>a[1]>>a[2]))throw std::runtime_error("Expected three LWS integers");
    const bool any=a[0]||a[1]||a[2],all=a[0]&&a[1]&&a[2];
    if(any&&!all)throw std::runtime_error("LWS must be all positive or 0,0,0"); return a;
}
static std::string arr(const std::array<size_t,3>& a){return "["+std::to_string(a[0])+","+std::to_string(a[1])+","+std::to_string(a[2])+"]";}
struct K {
    cl_program p{};cl_kernel k{};std::array<size_t,3> logical{},global{},local{};
    double build_ms{};
};
static K compile(cl_context ctx,cl_device_id dev,const std::string& path,const char* name,const std::string& opts,
                 std::array<size_t,3> g,std::array<size_t,3> l) {
    K k;k.logical=g;k.global=g;k.local=l;
    if(l[0])for(int i=0;i<3;++i)k.global[i]=((g[i]+l[i]-1)/l[i])*l[i];
    auto bytes=read(path);const char* src=bytes.data();size_t n=bytes.size();cl_int err;
    auto t=Clock::now();k.p=clCreateProgramWithSource(ctx,1,&src,&n,&err);check(err,"clCreateProgram");
    err=clBuildProgram(k.p,1,&dev,opts.c_str(),nullptr,nullptr);
    if(err!=CL_SUCCESS){size_t nlog;clGetProgramBuildInfo(k.p,dev,CL_PROGRAM_BUILD_LOG,0,nullptr,&nlog);std::vector<char> log(nlog);clGetProgramBuildInfo(k.p,dev,CL_PROGRAM_BUILD_LOG,nlog,log.data(),nullptr);std::cerr<<log.data()<<std::endl;check(err,"clBuildProgram");}
    k.k=clCreateKernel(k.p,name,&err);check(err,"clCreateKernel");
    k.build_ms=std::chrono::duration<double,std::milli>(Clock::now()-t).count();
    size_t maxWG=0;check(clGetKernelWorkGroupInfo(k.k,dev,CL_KERNEL_WORK_GROUP_SIZE,sizeof(maxWG),&maxWG,nullptr),"clGetKernelWorkGroupInfo");
    if(l[0]&&l[0]*l[1]*l[2]>maxWG)throw std::runtime_error("LWS exceeds kernel maximum");return k;
}
static cl_int2 pair(int x,int y){cl_int2 v;v.s[0]=x;v.s[1]=y;return v;}
static void args(K& k,cl_mem in,cl_mem w,cl_mem b,cl_mem out,int T,bool axisH) {
    int idx=0;auto set=[&](size_t n,const void* p){check(clSetKernelArg(k.k,idx++,n,p),"clSetKernelArg");};
    for(size_t x:k.logical){int v=int(x);set(sizeof(v),&v);}
    set(sizeof(in),&in);set(sizeof(w),&w);set(sizeof(b),&b);set(sizeof(out),&out);
    int batch=1;set(sizeof(batch),&batch);
    const cl_int2 shapes[]={axisH?pair(T,1):pair(1,T),axisH?pair(5*T,1):pair(1,5*T),
                           axisH?pair(5,1):pair(1,5),axisH?pair(-3,0):pair(0,-3),
                           axisH?pair(7,0):pair(0,7),axisH?pair(11,1):pair(1,11)};
    for(const auto& s:shapes)set(sizeof(s),&s);
    for(int v:{11,384,192})set(sizeof(v),&v);
}
struct Time{double host_us,kernel_us;};
static Time run(cl_command_queue q,K& k,bool profiling) {
    check(clFinish(q),"pre-finish");cl_event event{};auto start=Clock::now();
    check(clEnqueueNDRangeKernel(q,k.k,3,nullptr,k.global.data(),k.local[0]?k.local.data():nullptr,0,nullptr,profiling?&event:nullptr),"clEnqueueNDRangeKernel");
    check(clFinish(q),"clFinish");double host=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
    double device=-1;if(profiling){cl_ulong a,b;check(clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_START,sizeof(a),&a,nullptr),"event start");check(clGetEventProfilingInfo(event,CL_PROFILING_COMMAND_END,sizeof(b),&b,nullptr),"event end");device=double(b-a)*1e-3;check(clReleaseEvent(event),"release event");}return {host,device};
}
static std::vector<uint16_t> download(cl_command_queue q,cl_mem b,size_t n){std::vector<uint16_t> x(n);check(clEnqueueReadBuffer(q,b,CL_TRUE,0,n*2,x.data(),0,nullptr,nullptr),"read output");return x;}
static std::vector<float> nchw(const std::vector<uint16_t>& x,int C,int T){std::vector<float> f(C*T);for(int c=0;c<C;++c)for(int t=0;t<T;++t)f[c*T+t]=h2f(x[((c/4)*T+t)*4+c%4]);return f;}
static bool compare(const std::vector<uint16_t>& old,const std::vector<uint16_t>& cand,double atol,double rtol,std::ostream& log) {
    size_t mismatch=0,fail=0,nonfinite=0;double maxabs=0,sq=0;
    for(size_t i=0;i<old.size();++i){if(old[i]!=cand[i])mismatch++;double a=h2f(old[i]),b=h2f(cand[i]);
        if(!std::isfinite(a)||!std::isfinite(b)){nonfinite++;fail++;continue;}double d=std::abs(a-b);maxabs=std::max(maxabs,d);sq+=d*d;if(d>atol+rtol*std::abs(a))fail++;}
    log<<"{\"kind\":\"correctness\",\"elements\":"<<old.size()<<",\"bit_mismatches\":"<<mismatch<<",\"numeric_failures\":"<<fail<<",\"nonfinite\":"<<nonfinite<<",\"max_abs\":"<<maxabs<<",\"rmse\":"<<std::sqrt(sq/old.size())<<",\"atol\":"<<atol<<",\"rtol\":"<<rtol<<",\"passed\":"<<(fail?"false":"true")<<"}\n";log.flush();return fail==0;
}
static void usage(){std::cerr<<"Usage: bench_deconv --input F32_NCHW --weights F16_PACKED --bias F16_PACKED --T N --axis h|w --baseline deconv_2d.cl --candidate bigvgan_deconv_s5_tiled.cl --out DIR [--tile 4 --old-lws 0,0,0 --new-lws 8,4,1 --warmup 3 --reps 15 --profile 1 --atol 0 --rtol 0 --reference F32_NCHW --input-layout nchw-f32|nc4hw4-f16]\n";}
int main(int argc,char**argv){try{
    std::map<std::string,std::string> p;
    for(int i=1;i<argc;i+=2){if(i+1==argc){usage();return 1;}p[argv[i]]=argv[i+1];}
    for(const char* n:{"--input","--weights","--bias","--T","--axis","--baseline","--candidate","--out"})if(!p.count(n)){usage();throw std::runtime_error(std::string("Missing ")+n);}
    auto get=[&](const std::string& k,const std::string& d){auto x=p.find(k);return x==p.end()?d:x->second;};
    const int T=std::stoi(p["--T"]),tile=std::stoi(get("--tile","4")),warmup=std::stoi(get("--warmup","3")),reps=std::stoi(get("--reps","15"));
    const bool axisH=p["--axis"]=="h",profiling=std::stoi(get("--profile","1"));
    const double atol=std::stod(get("--atol","0")),rtol=std::stod(get("--rtol","0"));
    if(T<1||T>100000||warmup<0||reps<1||(tile!=1&&tile!=2&&tile!=4)||(p["--axis"]!="h"&&p["--axis"]!="w")||atol<0||rtol<0)throw std::runtime_error("Invalid configuration");
    const auto oldL=dims(get("--old-lws","0,0,0")),newL=dims(get("--new-lws","8,4,1"));
    const size_t inputN=size_t(1536)*T,outputN=size_t(768)*5*T;
    auto weights=typed<uint16_t>(p["--weights"],size_t(1536)*768*11),bias=typed<uint16_t>(p["--bias"],768);
    std::vector<uint16_t> input(inputN);
    if(get("--input-layout","nchw-f32")=="nc4hw4-f16")input=typed<uint16_t>(p["--input"],inputN);
    else if(get("--input-layout","nchw-f32")=="nchw-f32"){
        auto f=typed<float>(p["--input"],inputN);for(int c=0;c<1536;++c)for(int t=0;t<T;++t){float v=f[c*T+t];if(!std::isfinite(v))throw std::runtime_error("Nonfinite input");input[((c/4)*T+t)*4+c%4]=f2h(v);}
    }else throw std::runtime_error("Unknown input layout");
    for(auto v:input)if(!std::isfinite(h2f(v)))throw std::runtime_error("Nonfinite half input");
    for(auto v:weights)if(!std::isfinite(h2f(v)))throw std::runtime_error("Nonfinite half weight");
    for(auto v:bias)if(!std::isfinite(h2f(v)))throw std::runtime_error("Nonfinite half bias");
    mkdir(p["--out"].c_str(),0755);std::ofstream log(p["--out"]+"/results.jsonl");if(!log)throw std::runtime_error("Cannot open results");log<<std::setprecision(12);
    cl_uint np=0;check(clGetPlatformIDs(0,nullptr,&np),"platform count");std::vector<cl_platform_id> platforms(np);check(clGetPlatformIDs(np,platforms.data(),nullptr),"platforms");
    cl_device_id dev{};for(auto platform:platforms){cl_uint nd=0;if(clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,1,&dev,&nd)==CL_SUCCESS&&nd)break;dev=nullptr;}
    if(!dev)throw std::runtime_error("No OpenCL GPU");char name[1024]={},driver[1024]={};check(clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof(name),name,nullptr),"device name");check(clGetDeviceInfo(dev,CL_DRIVER_VERSION,sizeof(driver),driver,nullptr),"driver");
    cl_int err;cl_context ctx=clCreateContext(nullptr,1,&dev,nullptr,nullptr,&err);check(err,"context");cl_command_queue q=clCreateCommandQueue(ctx,dev,profiling?CL_QUEUE_PROFILING_ENABLE:0,&err);check(err,"queue");
    auto buffer=[&](size_t n,void* data){cl_mem b=clCreateBuffer(ctx,data?(CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR):CL_MEM_READ_WRITE,n,data,&err);check(err,"buffer");return b;};
    cl_mem ib=buffer(input.size()*2,input.data()),wb=buffer(weights.size()*2,weights.data()),bb=buffer(bias.size()*2,bias.data()),ob=buffer(outputN*2,nullptr),nb=buffer(outputN*2,nullptr);
    // Match current MNN deconv FLOAT/FLOAT4 precision and default compiler flags.
    const std::string opts="-DFLOAT=half -DFLOAT4=half4 -DMNN_SUPPORT_FP16 -DUSE_BUFFER -DBIAS -cl-mad-enable -w -DBIGVGAN_TILE="+std::to_string(tile);
    auto old=compile(ctx,dev,p["--baseline"],"deconv_2d",opts,{192,axisH?1u:size_t(5*T),axisH?size_t(5*T):1u},oldL);
    auto cand=compile(ctx,dev,p["--candidate"],"bigvgan_deconv_s5_tiled",opts,{192,size_t((T+tile-1)/tile),5},newL);
    args(old,ib,wb,bb,ob,T,axisH);args(cand,ib,wb,bb,nb,T,axisH);
    log<<"{\"kind\":\"manifest\",\"device\":"<<quote(name)<<",\"driver\":"<<quote(driver)<<",\"T\":"<<T<<",\"axis\":"<<quote(p["--axis"])<<",\"tile\":"<<tile<<",\"warmup\":"<<warmup<<",\"reps\":"<<reps<<",\"profiling_queue\":"<<(profiling?"true":"false")<<",\"old_lws\":"<<arr(oldL)<<",\"new_lws\":"<<arr(newL)<<",\"old_gws\":"<<arr(old.logical)<<",\"new_gws\":"<<arr(cand.logical)<<",\"old_build_ms\":"<<old.build_ms<<",\"candidate_build_ms\":"<<cand.build_ms<<",\"build_options\":"<<quote(opts)<<",\"timing_scope\":\"Resident buffers; device event and synchronized enqueue-to-finish host; no validation/readback/packing\"}\n";log.flush();
    run(q,old,profiling);run(q,cand,profiling);
    const auto oldData=download(q,ob,outputN),newData=download(q,nb,outputN);
    save(p["--out"]+"/old.nc4hw4.f16",oldData);save(p["--out"]+"/candidate.nc4hw4.f16",newData);
    auto oldNCHW=nchw(oldData,768,5*T);save(p["--out"]+"/old.nchw.f32",oldNCHW);save(p["--out"]+"/candidate.nchw.f32",nchw(newData,768,5*T));
    bool valid=compare(oldData,newData,atol,rtol,log);
    if(p.count("--reference")){auto ref=typed<float>(p["--reference"],outputN);size_t mismatch=0;double maxabs=0;for(size_t i=0;i<ref.size();++i){if(!std::isfinite(ref[i])||oldNCHW[i]!=ref[i])mismatch++;maxabs=std::max(maxabs,std::abs(double(oldNCHW[i])-ref[i]));}
        log<<"{\"kind\":\"baseline_vs_mnn_capture\",\"numeric_mismatches\":"<<mismatch<<",\"max_abs\":"<<maxabs<<",\"passed\":"<<(mismatch?"false":"true")<<"}\n";valid=valid&&mismatch==0;}
    log.flush();if(!valid){std::cerr<<"Correctness failed; timings not collected.\n";return 3;}
    for(int i=0;i<warmup;++i){run(q,old,profiling);run(q,cand,profiling);}
    for(int rep=0;rep<reps;++rep)for(int order=0;order<2;++order){bool isNew=((rep+order)&1)!=0;auto t=run(q,isNew?cand:old,profiling);
        log<<"{\"kind\":\"sample\",\"backend\":\""<<(isNew?"candidate":"old")<<"\",\"T\":"<<T<<",\"rep\":"<<rep<<",\"order\":"<<order<<",\"host_us\":"<<t.host_us<<",\"kernel_us\":";if(profiling)log<<t.kernel_us;else log<<"null";log<<"}\n";log.flush();}
    std::cout<<"Completed T="<<T<<" tile="<<tile<<" results="<<p["--out"]<<"/results.jsonl\n";
    for(auto b:{ib,wb,bb,ob,nb})clReleaseMemObject(b);for(auto k:{old,cand}){clReleaseKernel(k.k);clReleaseProgram(k.p);}clReleaseCommandQueue(q);clReleaseContext(ctx);return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 2;}}
