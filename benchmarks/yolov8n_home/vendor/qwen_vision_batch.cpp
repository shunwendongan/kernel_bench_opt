#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include "image_enc.h"

static double now_ms(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static std::vector<std::string> split(const std::string&s){std::vector<std::string>v;std::stringstream ss(s);std::string x;while(std::getline(ss,x,'\t'))v.push_back(x);return v;}
int main(int argc,char**argv){
 if(argc!=4){std::cerr<<"model plan.tsv output_root\n";return 2;}
 std::string root=argv[3];setenv("QDETAIL",(root+"/warmup.perf.txt").c_str(),1);
 rknn_app_context_t ctx{};double init_start=now_ms();
 if(init_imgenc(argv[1],&ctx,3)!=0)return 3;
 if(ctx.io_num.n_output!=4||ctx.model_height!=448||ctx.model_width!=448||ctx.model_image_token!=196||ctx.model_embed_size!=2048){release_imgenc(&ctx);return 4;}
 size_t nelem=size_t(ctx.model_image_token)*ctx.model_embed_size*ctx.io_num.n_output;
 std::vector<float>out(nelem);double init_ms=now_ms()-init_start;
 cv::Mat warm(448,448,CV_8UC3,cv::Scalar(127,127,127));
 for(int i=0;i<3;i++)if(run_imgenc(&ctx,warm.data,out.data())!=0)return 5;
 std::ofstream(root+"/session.json")<<"{\"init_ms\":"<<init_ms<<",\"warmup_runs\":3,\"model_resident\":true}";
 std::ifstream plan(argv[2]);std::string line;int count=0,failed=0;
 while(std::getline(plan,line)){
  auto f=split(line);if(f.size()!=5)return 6;
  std::string dir=root+"/units/"+f[0];
  if(std::ifstream(dir+"/done.json").good())continue;
  double start=now_ms();double actual_msec=-1,actual_frame=-1;cv::Mat bgr;
  try{
   if(f[1]=="image")bgr=cv::imread(f[2],cv::IMREAD_COLOR);
   else{
    cv::VideoCapture cap(f[2]);if(!cap.isOpened())throw std::runtime_error("video_open_failed");
    double frames=cap.get(cv::CAP_PROP_FRAME_COUNT);
    if(!std::isfinite(frames)||frames<1)throw std::runtime_error("invalid_video_frame_count");
    int slot=std::stoi(f[3]),slots=std::stoi(f[4]);
    if(frames<slots)throw std::runtime_error("too_few_video_frames_for_plan");
    double index=slots<=1?0:std::round(slot*(frames-1)/(slots-1));
    if(!cap.set(cv::CAP_PROP_POS_FRAMES,index))throw std::runtime_error("video_seek_failed");
    if(!cap.read(bgr))throw std::runtime_error("video_decode_failed");
    actual_frame=cap.get(cv::CAP_PROP_POS_FRAMES)-1;actual_msec=cap.get(cv::CAP_PROP_POS_MSEC);
   }
   if(bgr.empty())throw std::runtime_error("empty_image");
   cv::Mat rgb,square,input;cv::cvtColor(bgr,rgb,cv::COLOR_BGR2RGB);
   int side=std::max(rgb.cols,rgb.rows);square=cv::Mat(side,side,CV_8UC3,cv::Scalar(127.5,127.5,127.5));
   rgb.copyTo(square(cv::Rect((side-rgb.cols)/2,(side-rgb.rows)/2,rgb.cols,rgb.rows)));
   cv::resize(square,input,cv::Size(448,448),0,0,cv::INTER_LINEAR);
   double prep_ms=now_ms()-start;
   {std::ofstream stream(dir+"/input.u8",std::ios::binary);stream.write((char*)input.data,input.total()*input.elemSize());if(!stream)throw std::runtime_error("input_write_failed");}
   setenv("QDETAIL",(dir+"/perf.txt").c_str(),1);
   double t=now_ms();int ret=run_imgenc(&ctx,input.data,out.data());double inference_ms=now_ms()-t;
   if(ret!=0)throw std::runtime_error("rknn_inference_failed");
   {std::ifstream perf(dir+"/perf.txt",std::ios::binary|std::ios::ate);if(!perf||perf.tellg()<=0)throw std::runtime_error("missing_perf_report");}
   for(float x:out)if(!std::isfinite(x))throw std::runtime_error("nonfinite_output");
   {std::ofstream stream(dir+"/output.f32",std::ios::binary);stream.write((char*)out.data(),out.size()*sizeof(float));if(!stream)throw std::runtime_error("output_write_failed");}
   {std::ofstream j(dir+"/done.tmp");j<<std::setprecision(12)<<"{\"status\":\"success\",\"preprocess_ms\":"<<prep_ms<<",\"vision_call_ms\":"<<inference_ms<<",\"source_width\":"<<bgr.cols<<",\"source_height\":"<<bgr.rows<<",\"actual_frame_index\":"<<actual_frame<<",\"actual_timestamp_ms\":"<<actual_msec<<",\"output_elements\":"<<nelem<<",\"output_finite\":true}";}
   if(std::rename((dir+"/done.tmp").c_str(),(dir+"/done.json").c_str())!=0)throw std::runtime_error("commit_failed");
   std::cout<<"UNIT_OK "<<f[0]<<" "<<inference_ms<<std::endl;
  }catch(const std::exception&e){
   std::ofstream(dir+"/error.txt")<<e.what();std::cerr<<"UNIT_FAILED "<<f[0]<<" "<<e.what()<<std::endl;failed++;
  }
  count++;
 }
 release_imgenc(&ctx);std::cout<<"BATCH_COMPLETE processed="<<count<<" failed="<<failed<<std::endl;return failed?1:0;
}
