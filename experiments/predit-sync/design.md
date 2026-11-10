# predit-sync：PreDiT → FP32 DiT 交接处的显式等待

候选身份：[candidate.json](candidate.json)。变更：[predit-sync.patch](predit-sync.patch)（对仓库 `src/mnn/omni.cpp` 覆盖文件的 8 行补丁）。

优化原理：[principles/05-predit-sync.md](../../principles/05-predit-sync.md)。

## 观察与假设

默认异步构建的完整请求 A/B 中，tile0 与 tile4 的 text/speech token 相同而波形不同（max abs 0.6607，relative RMSE 0.3391），差异集中在波形前段；同一 mel 的 BigVGAN 回放却位级一致。假设是 PreDiT（OpenCL）输出在 FP32 DiT 读取时尚未完成写入。

## 变换与不变量

`OMNI_PREDIT_SYNC` 存在时，对 PreDiT 的每个输出调用 `tensor->wait(MAP_TENSOR_READ, true)` 再交给 DiT。不改任何算子数值；两侧（tile0 / tile4）必须使用同一同步构建。

## 结果与决策

结果、适用范围、补丁应用方式和历史 Release 中的证据位置见 [docs/SYNC_VARIANT.md](../../docs/SYNC_VARIANT.md)。单样本 3 对通过；保留为独立候选，不归入默认异步实现。

## 后续方向

在多样本上复现异步差异并用 event 时间线证明读写重叠；用 MNN 跨后端依赖或同队列替代宿主等待。
