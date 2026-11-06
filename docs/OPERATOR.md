# ConvTranspose 优化合同

目标 MNN 节点为 `/ups.0.0/ConvTranspose_output_0`，执行器是 OpenCL Buffer `DeconvBufExecution`。候选 `bigvgan_deconv_s5_tiled` 对同一 stride 相位的四个时间位置复用权重载入。

```text
y[o,t] = b[o] + Σ_i Σ_k x[i,(t+3-k)/5] * w[i,o,k]
```

只累计 `(t+3-k)/5` 为整数且在输入范围内的位置。合同为 N1/Ci1536/Co768/K11/S5/P3/D1/groups1，无激活融合。输出长度为 `5*T`。

## 数值与内存约束

- 当前部署 precision enum0，在该 kernel 中实际为 half 存储、half4 累加。
- 输入/输出 OpenCL Buffer layout 为 `[C/4,T,4]`；权重为 `[Ci,Co/4,K,4]`，保持原 converter 的 packed bytes。
- 每个输出保留 `ic block → descending valid tap → x/y/z/w mad` 累加次序，不拆分 reduction，不把累加改为 FP32。
- 逻辑 GWS 为 `{192,ceil(T/4),5}`，LWS 为 `{8,4,1}`；边界守卫处理实际工作组补齐。
- 仅当节点名、结构、precision0、tile4 和 T52/120/136 均匹配时使用候选，其他情况回退原 kernel。
- 每个 execution instance 缓存自己的编译后 KernelWrap，避免每次 resize 重新 JIT。

## 三层正确性门

1. 同一份真实输入和 packed 权重：旧 standalone kernel 必须匹配旧 MNN capture，候选必须匹配旧 kernel。
2. 相同真实 mel：旧/新 MNN 的目标输入、目标输出、完整 BigVGAN waveform 都比较 bit pattern。
3. 相同完整请求：文本 token、语音 token、float waveform 以及正式请求对暖机工作量都一致，才解释完整延迟。

FP32 参考是误差诊断，不能用它为改变旧部署精度开绿灯。权重/RNG/shape/DiT 步数/token 上限/语音长度改变时，不能把不同工作量的时间直接比较。

## ONNX 来源边界

当前部署是 MNN 快照，未取得其历史全模型 ONNX 转换输入。后来从原 HF 权重导出的 BigVGAN 参考 ONNX 中，目标为零起算节点22 `/ups.0.0/ConvTranspose`，输入 `/conv_pre/Conv_output_0`，weight `ups.0.0.weight`，bias `ups.0.0.bias`。

参考 ONNX SHA-256 为 `e8596908a2b5817fa6f7aa1a3d082444c0c3160ff781566092cb6c2b862eb333`。此编号只对该文件有效。它是结构溯源候选，不能证明它就是部署 MNN 的原始转换图，也不能把参考 FP32 权重替换为当前部署权重后仍声称权重未变。

`tools/mapping/export_bigvgan_reference_draft.py` 和 `match_bigvgan_reference.py` 保留这一区分。参考子图可用 ONNX 官方 `onnx.utils.extract_model` 按上述输入输出边界提取；必须保留源图哈希、initializer、dtype、shape 和新板验证记录。ONNX权重不进入Git；此前参考ONNX的Release见 [ASSETS.md](ASSETS.md)，后续默认使用本地资产。
