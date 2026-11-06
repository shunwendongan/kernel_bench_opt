# Model artifact provenance

Built with Qwen. The deployed model is the taobao-mnn/Qwen2.5-Omni-3B-MNN 4-bit converted snapshot, revision 00dc2e9131a4bb325b43a47f4210dd6450116687. The release preserves the exact deployed model bytes.

The NPU prefix is a tiled export of MNN-decoded Conv1/Conv2 weights with tanh GELU, FP16 RKNN compilation, halo masking and valid-output cropping. Reference BigVGAN ONNX is a separate FP32 export from the original HF model and is not the deployed MNN historical conversion graph. Captures preserve deployed values and packed FP16 buffers; metadata host paths may be made portable.

Original model research-license conditions are retained here. The upstream MNN model card labels its conversion apache-2.0; that label is not treated as removing conditions on the original Qwen model. The repository Apache license applies to repository code, not as a replacement model license.
