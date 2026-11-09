# Omni 组合路径端到端 A/B

运行状态：`completed`；冻结输入 10 条，取得 20 条正式请求记录；成对数据完整：True。

基线：原 MNN/OpenCL 完整音频编码器 + 旧 ConvTranspose。候选：既有 RKNN/NPU 音频前缀 + CPU 编码器尾部 + BigVGAN tile4。

音频特征与输出均有效的配对请求总时间：**1263.105 → 1158.949 s**；系统时间比 **1.090×**，变化 **8.25%**。此值仅覆盖音频/输出有效的配对，不含编码失败样本，也不代表全部10条通过。

正式请求严格输出一致 1/10 对；生成工作量一致 1/10 对。输出变化的配对只能描述系统路径耗时，不能当作同输出优化收益或答案质量通过。

| 序号 | 样本 | 类型 | 旧 s | 新 s | 旧/新 | 测量有效 | 输出严格一致 | 缺失 |
|---:|---|---|---:|---:|---:|---|---|---|
| 1 | zhuanzhe0124-3 | audio2text | 92.199 | 46.514 | 1.982 | True | False | — |
| 2 | gk_law_02982 | audio2text | 14.699 | 6.055 | 2.428 | True | True | — |
| 3 | en_teshuyinxiao_000060 | audio2audio | 385.068 | 361.363 | 1.066 | True | False | — |
| 4 | topic_shift_neirong_000074 | audio2text | 156.827 | 45.673 | 3.434 | True | False | — |
| 5 | en_xuxie_000273 | audio2text | 56.566 | 173.496 | 0.326 | True | False | — |
| 6 | huisu0141-4 | audio2text | 115.535 | 62.386 | 1.852 | True | False | — |
| 7 | yuyan_000986 | audio2audio | 392.102 | 363.500 | 1.079 | True | False | — |
| 8 | accent_dongbei_cmp_gongqing_000110 | audio2text | 13.053 | 8.816 | 1.481 | True | False | — |
| 9 | zhuanzhe0263-1 | audio2text | 37.058 | 91.147 | 0.407 | True | False | — |
| 10 | dc_induction_conv0081_649_39 | audio2text | 11.387 | 11.745 | 0.970 | False | False | — |

## 按任务类型汇总

| 类型 | 配对数 | 旧总时间 s | 新总时间 s | 旧/新 | 严格同输出对数 |
|---|---:|---:|---:|---:|---:|
| audio2audio | 2 | 777.170 | 724.863 | 1.072 | 0 |
| audio2text | 7 | 485.936 | 434.086 | 1.119 | 1 |

严格同输出子集：1对，耗时 14.699 → 6.055 s，时间比 2.428×；这只覆盖相应子集，不能推广到全部10条。

## 音频编码失败：不得计为加速

外层程序成功返回不代表音频特征提取成功。下列请求在完整日志中出现编码器shape错误/无音频特征，已从有效配对与严格一致性统计中剔除；原始记录和耗时仍完整保留。
- 样本10 / new：Broad cast error, dim1 = 1500, dim2 = 2251；Compute Shape Error for /Add_2_output_0；Media encoder returned no features
- 样本10 / old：Broad cast error, dim1 = 1500, dim2 = 2251；Compute Shape Error for /Add_2_output_0；Media encoder returned no features

## 解释范围

- 每条样本只有一对正式A/B，不报告稳定p95或统计显著性。
- 两个模型实例常驻、请求串行执行；本轮不证明单实例内存节省或并发吞吐。
- 每侧只预热一条语音任务；其他动态shape首次编码/编译成本保留在正式请求时间内。
- 模型源文件相同不等于数值执行相同：RKNN前缀使用FP16，可能改变embedding和生成输出。
- 文本/语音token和波形比较检查输出是否保持，不等于答案质量、语义或听感评测通过。
- audio2text的空波形不作为语音正确性证据。
- 请求计时从response开始至文本返回或语音完整生成；模型加载、预热及输出文件落盘不计入。
- 阶段计数器可能重叠，不相加作为端到端耗时分解。
- 本次是系统路径对比，不证明存在独立优化的Conv2+GELU kernel，也不把收益分别归因于两个改动。

## 产物

- `summary.json`：分组统计、逐对记录、原始文件身份与审查问题。
- `per_sample.csv`：所有预计配对，包括缺失与失败。
- `answers.json`：新旧文本与来源位置，未执行任务质量评分。
