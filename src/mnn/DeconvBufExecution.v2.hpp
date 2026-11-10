//
//  DeconvBufExecution.hpp
//  MNN
//
//  Created by MNN on 2021/04/09.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef DeconvBufExecution_hpp
#define DeconvBufExecution_hpp

#include "backend/opencl/execution/buffer/ConvBufExecution.hpp"
#include <map>
#include <string>
namespace MNN {
namespace OpenCL {

class DeconvBufExecution : public ConvBufCommonExecution, public CommonExecution {
public:
    DeconvBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    DeconvBufExecution(std::shared_ptr<ConvBufResource> resource, const MNN::Op* op, Backend* backend);
    virtual ~DeconvBufExecution();

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    std::shared_ptr<KernelWrap> mBigvganKernel;
    // bigvgan-deconv1d-v2 programs keyed by compile-time configuration.
    std::map<std::string, std::shared_ptr<KernelWrap>> mV2Kernels;
    std::vector<uint32_t> mLWS{0, 0, 0, 0};
    std::vector<uint32_t> mGWS{0, 0, 0, 0};
};

} // namespace OpenCL
} // namespace MNN
#endif /* DeconvBufExecution_hpp */
#endif /* MNN_OPENCL_BUFFER_CLOSED */
