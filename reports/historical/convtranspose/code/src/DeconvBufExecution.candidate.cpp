//
//  DeconvBufExecution.cpp
//  MNN
//
//  Created by MNN on 2021/04/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "DeconvBufExecution.task.hpp"
#include "bigvgan_deconv_source.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "backend/opencl/core/OpenCLBackend.hpp"
#include "backend/opencl/core/OpenCLRunningUtils.hpp"
#include "core/ConvolutionCommon.hpp"
#include <fstream>
#include <cstdlib>
#include <sstream>

namespace MNN {
namespace OpenCL {

DeconvBufExecution::DeconvBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : ConvBufCommonExecution(op->main_as_Convolution2D(), backend), CommonExecution(backend, op) {
    if (!mConvComValid) {
        mValid = false;
        return;
    }
    mOpenCLBackend                 = static_cast<OpenCLBackend *>(backend);
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dCommonParams = conv2dCommonParams;
    mResource->mStrides            = {conv2dCommonParams->strideY(), conv2dCommonParams->strideX()};
    mResource->mDilations          = {conv2dCommonParams->dilateY(), conv2dCommonParams->dilateX()};
    int kernelWidth                = conv2dCommonParams->kernelX();
    int kernelHeight               = conv2dCommonParams->kernelY();

    MNN_ASSERT(mResource->mStrides[0] > 0 && mResource->mStrides[1] > 0);
    int outputChannel = conv2dCommonParams->outputCount();

    const float* filterDataPtr = nullptr;
    int weightSize = 0;
    std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon;
    ConvolutionCommon::getConvParameters(&quanCommon, backend, op, &filterDataPtr, &weightSize);

    int inputChannel  = weightSize / (kernelWidth * kernelHeight * outputChannel);
    const char* dumpRoot = std::getenv("DECONV_CAPTURE_DIR");
    if (dumpRoot && inputChannel == 1536 && outputChannel == 768 && kernelWidth * kernelHeight == 11) {
        std::ofstream f(std::string(dumpRoot) + "/deployed_weight_iohw.f32", std::ios::binary);
        f.write(reinterpret_cast<const char*>(filterDataPtr), weightSize * sizeof(float));
        if (conv2dParams->bias()) {
            std::ofstream b(std::string(dumpRoot) + "/deployed_bias.f32", std::ios::binary);
            b.write(reinterpret_cast<const char*>(conv2dParams->bias()->data()), outputChannel * sizeof(float));
        }
        MNN_PRINT("DECONV_CAPTURE decoded weight=%d precision=%d buffer=1\n", weightSize, int(mOpenCLBackend->getPrecision()));
    }

    mResource->mFilter.reset(Tensor::createDevice<float>({1, UP_DIV(outputChannel, 4) * kernelWidth * kernelHeight, 1, 4 * inputChannel}));
    OPENCL_CHECK_ALLOC_CTOR(mOpenCLBackend->onAcquireBuffer(mResource->mFilter.get(), Backend::STATIC));
        
    if (mOpenCLBackend->getRuntime()->hint().useCachedMmap <= 1){
        std::vector<float> filterDataPtrTransformed;
        filterDataPtrTransformed.resize(weightSize);
        IOHW2OIHW<float, int>(filterDataPtr, filterDataPtrTransformed.data(), outputChannel, inputChannel, kernelHeight, kernelWidth);
            
        std::shared_ptr<Tensor> filterBuffer(Tensor::createDevice<float>({outputChannel, inputChannel, kernelHeight, kernelWidth}));
            
        size_t buffer_size = filterBuffer->elementSize() * sizeof(float);
        cl::Buffer filterBufferCL(mOpenCLBackend->getOpenCLRuntime()->context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, buffer_size);
        filterBuffer->buffer().device = (uint64_t)(&filterBufferCL);
        cl_int error;
        auto ptrCL = mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueMapBuffer(filterBufferCL, true, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error);
        if(ptrCL != nullptr && error == CL_SUCCESS){
            ::memcpy(ptrCL, filterDataPtrTransformed.data(), filterBuffer->size());
        }else{
            MNN_ERROR("Map error ptrCL == nullptr \n");
        }
        mOpenCLBackend->getOpenCLRuntime()->commandQueue().enqueueUnmapMemObject(filterBufferCL, ptrCL);
        MNN::OpenCL::BufferConvertor bufferConvertor{mOpenCLBackend->getOpenCLRuntime()};
        bool needTrans = true;
        bufferConvertor.convertToNC4HW4Buffer(filterBuffer.get(), MNN::OpenCL::CONV2D_FILTER, mResource->mFilter.get(), mOpenCLBackend->getPrecision(), needTrans);
    }
    mResource->mBuildOptions.emplace("-DBIAS");
    if (conv2dCommonParams->relu() == true) {
        mResource->mBuildOptions.emplace("-DRELU");
    } else if (conv2dCommonParams->relu6() == true) {
        mResource->mBuildOptions.emplace("-DRELU6");
    }
    mResource->mBuildOptions.emplace("-DUSE_BUFFER");
}

DeconvBufExecution::~DeconvBufExecution() {
    // Do nothing
}

DeconvBufExecution::DeconvBufExecution(std::shared_ptr<ConvBufResource> resource, const MNN::Op* op, Backend *backend)
    : ConvBufCommonExecution(backend), CommonExecution(backend, op) {
    if (!mConvComValid) {
        mValid = false;
        return;
    }
    mResource = resource;
    const auto *conv2dParams       = op->main_as_Convolution2D();
    const auto *conv2dCommonParams = conv2dParams->common();
    mResource->mConv2dParams       = conv2dParams;
    mResource->mConv2dCommonParams  = conv2dCommonParams;
}

bool DeconvBufExecution::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    *dst = new DeconvBufExecution(mResource, op, bn);
    return true;
}

ErrorCode DeconvBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mUnits.resize(1);
    auto &unit = mUnits[0];
    auto output = outputs[0];
    auto input  = inputs[0];

    std::vector<int> inputShape  = tensorShapeFormat(input);
    std::vector<int> outputShape = tensorShapeFormat(output);
    auto runtime = mOpenCLBackend->getOpenCLRuntime();

    const int outputBatch    = outputShape.at(0);
    const int outputHeight   = outputShape.at(1);
    const int outputWidth    = outputShape.at(2);
    const int outputChannels = outputShape.at(3);

    const int inputChannels = inputShape.at(3);

    const int outputChannelBlocks = UP_DIV(outputChannels, 4);
    const int strideHeight        = mResource->mStrides[0];
    const int strideWidth         = mResource->mStrides[1];

    auto pad = ConvolutionCommon::convolutionTransposePad(input, output, mResource->mConv2dCommonParams);
    const int paddingHeight = pad.second;
    const int paddingWidth  = pad.first;

    auto ky               = mResource->mConv2dCommonParams->kernelY();
    auto kx               = mResource->mConv2dCommonParams->kernelX();
    auto kernelSize       = kx * ky;
    const int transPadH   = ky - 1 - pad.second;
    const int transPadW   = kx - 1 - pad.first;
    
    const int alignHeight = mResource->mStrides[0] - 1 - transPadH;
    const int alignWidth  = mResource->mStrides[1] - 1 - transPadW;
    
const auto* c = mResource->mConv2dCommonParams;
bool noOutPadding = true;
if (c->outPads()) {
    for (size_t i = 0; i < c->outPads()->size(); ++i) {
        noOutPadding = noOutPadding && c->outPads()->Get(i) == 0;
    }
}
const bool temporalH = ky == 11 && kx == 1 && strideHeight == 5 && strideWidth == 1
    && inputShape[2] == 1 && outputShape[2] == 1
    && paddingHeight == 3 && paddingWidth == 0
    && outputShape[1] == 5 * inputShape[1];
const bool temporalW = ky == 1 && kx == 11 && strideHeight == 1 && strideWidth == 5
    && inputShape[1] == 1 && outputShape[1] == 1
    && paddingHeight == 0 && paddingWidth == 3
    && outputShape[2] == 5 * inputShape[2];
const int temporalLength = inputShape[1] * inputShape[2];
const bool namedTarget = mOp && mOp->name()
    && mOp->name()->str() == "/ups.0.0/ConvTranspose_output_0";
const char* env = std::getenv("OMNI_BIGVGAN_DECONV_TILE");
const int tile = env ? std::atoi(env) : 0;
const bool specialized = (tile == 4)
    && namedTarget && inputs.size() == 1 && outputs.size() == 1
    && inputShape[0] == 1 && outputBatch == 1
    && inputChannels == 1536 && outputChannels == 768
    && c->group() == 1 && c->dilateX() == 1 && c->dilateY() == 1
    && !c->relu() && !c->relu6() && noOutPadding
    && (temporalLength == 52 || temporalLength == 120 || temporalLength == 136) && (temporalH || temporalW)
    && mOpenCLBackend->getPrecision() == 0;

    if (specialized) {
        if (!mBigvganKernel) {
            auto options = mResource->mBuildOptions;
            options.emplace("-DBIGVGAN_TILE=" + std::to_string(tile));
            mBigvganKernel = runtime->buildKernelFromSource(kBigvganDeconvSource,
                "bigvgan_deconv_s5_tiled", options, mOpenCLBackend->getPrecision());
        }
        if (!mBigvganKernel) return NOT_SUPPORT;
        unit.kernel = mBigvganKernel;
        mGWS = {192u, static_cast<uint32_t>((temporalLength + tile - 1) / tile), 5u};
    } else {
        unit.kernel = runtime->buildKernel("deconv_2d", "deconv_2d", mResource->mBuildOptions, mOpenCLBackend->getPrecision());
        mGWS = {static_cast<uint32_t>(outputChannelBlocks), static_cast<uint32_t>(outputWidth),
                static_cast<uint32_t>(outputHeight * outputBatch)};
    }
    auto maxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));
    if (std::getenv("BIGVGAN_DISPATCH_LOG") && namedTarget) {
        std::ofstream log(std::getenv("BIGVGAN_DISPATCH_LOG"), std::ios::app);
        log << "{\"T\":" << temporalLength << ",\"candidate\":" << (specialized ? "true" : "false")
            << ",\"precision\":" << int(mOpenCLBackend->getPrecision()) << ",\"tile\":" << tile << "}\n";
    }


    int inputImageShape[2]  = {inputShape.at(1), inputShape.at(2)};
    int outputImageShape[2] = {outputHeight, outputWidth};
    int strideShape[2]      = {strideHeight, strideWidth};
    int paddingShape[2]     = {transPadH, transPadW};
    int alignShape[2]       = {alignHeight, alignWidth};
    int kernelShape[2]      = {ky, kx};

    uint32_t idx = 0;
    unit.kernel->get().setArg(idx++, mGWS[0]);
    unit.kernel->get().setArg(idx++, mGWS[1]);
    unit.kernel->get().setArg(idx++, mGWS[2]);
    unit.kernel->get().setArg(idx++, openCLBuffer(input));
    unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mFilter.get()));
    unit.kernel->get().setArg(idx++, openCLBuffer(mResource->mBias.get()));
    unit.kernel->get().setArg(idx++, openCLBuffer(output));
    unit.kernel->get().setArg(idx++, static_cast<int32_t>(outputBatch));
    unit.kernel->get().setArg(idx++, sizeof(inputImageShape), inputImageShape);
    unit.kernel->get().setArg(idx++, sizeof(outputImageShape), outputImageShape);
    unit.kernel->get().setArg(idx++, sizeof(strideShape), strideShape);
    unit.kernel->get().setArg(idx++, sizeof(alignShape), alignShape);
    unit.kernel->get().setArg(idx++, sizeof(paddingShape), paddingShape);
    unit.kernel->get().setArg(idx++, sizeof(kernelShape), kernelShape);
    unit.kernel->get().setArg(idx++, static_cast<int32_t>(kernelSize));
    unit.kernel->get().setArg(idx++, static_cast<int32_t>(UP_DIV(inputChannels, 4)));
    unit.kernel->get().setArg(idx++, static_cast<int32_t>(outputChannelBlocks));
    
    std::string name = "deconv2d_buf";
    if (specialized) {
        mLWS = {8, 4, 1};
    } else {
    mLWS = localWS3DDefault(mGWS, maxWorkGroupSize, mOpenCLBackend->getOpenCLRuntime(), name, unit.kernel, mOpenCLBackend->getCLTuneLevel(), "deconv_2d").first;
    }
    const char* dumpRoot = std::getenv("DECONV_CAPTURE_DIR");
    if (dumpRoot && inputChannels == 1536 && outputChannels == 768 && kernelSize == 11) {
        const int elementBytes = mOpenCLBackend->getPrecision() == 1 ? 4 : 2;
        std::vector<unsigned char> packed(1536 * 768 * 11 * elementBytes);
        auto err = runtime->commandQueue().enqueueReadBuffer(openCLBuffer(mResource->mFilter.get()), CL_TRUE, 0, packed.size(), packed.data());
        MNN_ASSERT(err == CL_SUCCESS);
        std::ofstream w(std::string(dumpRoot) + "/packed_weight.bin", std::ios::binary);
        w.write(reinterpret_cast<const char*>(packed.data()), packed.size());
        std::vector<unsigned char> bias(768 * elementBytes);
        err = runtime->commandQueue().enqueueReadBuffer(openCLBuffer(mResource->mBias.get()), CL_TRUE, 0, bias.size(), bias.data());
        MNN_ASSERT(err == CL_SUCCESS);
        std::ofstream b(std::string(dumpRoot) + "/packed_bias.bin", std::ios::binary);
        b.write(reinterpret_cast<const char*>(bias.data()), bias.size());
        const int T = inputShape[1] * inputShape[2];
        std::ofstream m(std::string(dumpRoot) + "/dispatch_T" + std::to_string(T) + ".json");
        m << "{\"backend\":\"OpenCL_BUFFER\",\"kernel\":\"deconv_2d\",\"precision_enum\":" << int(mOpenCLBackend->getPrecision())
          << ",\"element_bytes\":" << elementBytes << ",\"input_hw\":[" << inputShape[1] << ',' << inputShape[2]
          << "],\"output_hw\":[" << outputHeight << ',' << outputWidth << "],\"kernel_hw\":[" << ky << ',' << kx
          << "],\"stride_hw\":[" << strideHeight << ',' << strideWidth << "],\"original_pad_hw\":[" << paddingHeight << ',' << paddingWidth
          << "],\"gws\":[" << mGWS[0] << ',' << mGWS[1] << ',' << mGWS[2]
          << "],\"lws\":[" << mLWS[0] << ',' << mLWS[1] << ',' << mLWS[2] << "]}\n";
        MNN_PRINT("DECONV_CAPTURE dispatch T=%d H=%d W=%d precision=%d\n", T, inputShape[1], inputShape[2], int(mOpenCLBackend->getPrecision()));
    }
    mOpenCLBackend->recordKernel3d(unit.kernel, mGWS, mLWS);
    unit.globalWorkSize = {mGWS[0], mGWS[1], mGWS[2]};
    unit.localWorkSize = {mLWS[0], mLWS[1], mLWS[2]};
    return NO_ERROR;
}

class DeconvolutionBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~DeconvolutionBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        if(inputs.size() != 1){
            return nullptr;
        }
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        OPENCL_CREATOR_CHECK(new DeconvBufExecution(inputs, op, backend));
    }
};

REGISTER_OPENCL_OP_CREATOR(DeconvolutionBufCreator, OpType_Deconvolution, BUFFER);

} // namespace OpenCL
} // namespace MNN
#endif /* MNN_OPENCL_BUFFER_CLOSED */
