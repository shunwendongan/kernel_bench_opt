# 后续 PreDiT 同步候选

补齐资产时核对到本地另存的后续结果：在PreDiT输出交给FP32 DiT前增加显式等待的独立构建，已完成同一真实样本的3个正式A/B对及2次暖机。原仓库默认异步版本的失败记录仍然保留，两者不能混为同一构建。

同步候选由 `OMNI_PREDIT_SYNC` 控制，唯一语义变更见 [predit-sync.patch](../experiments/predit-sync.patch)。开关按环境变量是否存在判断，禁用应 `unset OMNI_PREDIT_SYNC`，设为`0`仍会启用。

## 已保存的结果

同一同步构建、相同种子和输入、旧tile0/新tile4，8次请求text/speech tokens及float waveform全部一致。每次226 speech tokens、106560音频样本（4.44秒），波形SHA为 `93d680a5e4b2ca5890ee289dfdaf4633fadfecc7dedca298536bbb9f6eb51408`。

| 指标 | 旧中位数s | 新中位数s | 降低 |
|---|---:|---:|---:|
| 完整请求 | 94.060697120 | 91.055044558 | 3.19544% |
| TTFA | 32.236777040 | 31.113332562 | 3.48498% |
| token2wav | 80.535359 | 77.732916 | 3.47977% |

仅一个语音样本、3对，属于描述性结果，不报告稳定p95或统计显著性。T120/T136使用候选，T108回退；不是所有ConvTranspose都被替换。波形重复性不等于答案质量或MOS通过。此同步在当前样本消除了观察到的波形差异，仍不证明所有输入/设备均无问题。

## 选择同步候选进行新板复现

默认源码不自动改动。需要这一候选时，在独立工作副本应用补丁，然后按 [原生构建](BUILD.md) 使用新的独立MNN checkout；容器方式可按 [容器指南](CONTAINERS.md) 显式选择sync变体：

```bash
git apply --check experiments/predit-sync.patch
git apply experiments/predit-sync.patch
# 按docs/BUILD.md运行 scripts/build_rk3588.py，使用全新独立MNN checkout。
export OMNI_PREDIT_SYNC=1
export OMNI_TEST_SEED=20261105
# 再运行docs/REPRODUCE.md的完整omni_ab命令。
```

Release 的 `benchmark-evidence` 包保存 `evidence/e2e/paired_20261106/`（异步失败）与 `evidence/e2e/paired_sync_20261106/`（同步成功）、对应命令/构建记录、源快照及输出。`runtime-rk3588` 中的 `omni_ab_sync` 是历史Linux/aarch64预编译文件，SHA `aa1724e5deb91e962aa19edc433456d4f4bdc5e8d954f8511da079cc0e10f3e7`；新板ABI/驱动仍需匹配，不代替从源码验收。
