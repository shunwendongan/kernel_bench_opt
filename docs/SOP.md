# 端侧 benchmark → 分析 → 优化 → 知识沉淀 SOP

## 1. 工作对象和职责

仓库按 `<模型名称>_<测试数据集名称>` 分支组织模型/数据协议；分支内以 `candidate_id`、`run_id`、父候选和源码commit区分每次变更。当前分支为 `Qwen2.5-Omni-3B_VCB-Bench`。不同设备、固件和容器镜像属于不同环境记录，不能混成同一基线。

| 位置 | 负责 | 保存 |
|---|---|---|
| GitHub | 版本与知识 | kernel/算子代码、设计、Dockerfile、小型profile、映射、逐次指标、报告和失败经验 |
| 本地主机 | 大资产、分析和agent | 模型/全量数据/张量库，原始完整trace，分析工具包，候选生成 |
| 端侧设备 | 真实执行和采集 | 本次所需资产缓存/共享挂载、设备驱动、容器、临时结果 |

数据服务和自动同步暂不配置。容器的数据路径必须在端侧Docker daemon主机上可见；容器不能直接使用仅存在于台式机的路径。

## 2. 一轮实验的固定步骤

| 步骤 | 操作 | 验收 / 产物 |
|---|---|---|
| ① 冻结基线 | 选定commit、镜像、设备、输入、精度、种子和预算 | run记录、资产哈希、源码/构建身份 |
| ② 部署并检查 | 按容器指南挂载资产、透传明确设备、构建引擎 | 功能与真实后端证据；构建成功不等于推理通过 |
| ③ 代表性采集 | 单独诊断run采集少量真实请求/shape | 原始profile、环境遥测、转换记录、完成状态 |
| ④ 回传分析 | 完成文件回传主机，在工具包runs中分析 | 热点、计时域、覆盖缺口、ONNX候选与实现证据 |
| ⑤ Agent设计实现 | 一次聚焦一个可证伪假设，创建新candidate ID | design、补丁、接口不变量、回退条件 |
| ⑥ 多级评测 | L1算子 → L2子图 → L3完整请求 | 每级正确性先通过，再解释同口径性能 |
| ⑦ 决策与入库 | 比较噪声、收益、适用范围；保留成功和失败 | 小型原始记录、报告、manifest、经验，提交当前分支 |

每个候选绑定父候选；不能默认最后一次修改就是最佳实现。失败后保存原因和日志，不删除失败样本或放宽正确性阈值。相同假设反复没有新证据时重查机制，换一个有依据的方向。自动迭代时必须显式设置样本数、重复数、时间/轮次预算和停止条件，不能继承历史大批量任务。

## 3. 三级 benchmark 的统一含义

| 级别 | 输入和检查 | 性能口径 |
|---|---|---|
| L1 算子 | 真实shape、当前部署权重、相同布局/精度；新旧输出比较 | 驻留buffer微基准或明确包含传输的算子测试；两者分别记录 |
| L2 子图/模块 | 相同真实mel/中间张量；算子边界和整个模块输出 | 子图调用到完成，注明读回/传输/同步；capture时间不充当干净测量 |
| L3 E2E | 同prompt/音频/解码参数/步数/token和语音工作量 | 完整请求、TTFA、RTF等；等待最终输出完成 |

本案例要求部署FP16结果位级一致、DiT保持FP32。开启profiling的GPU event与宿主enqueue→finish属于不同且可能重叠的域，不相加。不同模型可使用自己的预先约定数值阈值，但不得测完后改门槛制造通过。

三种shape的局部收益与一个完整样本的收益分别报告。仅3对E2E是描述性统计，不表示稳定p95或总体显著性；需要泛化结论时再增加独立样本、形状与重复。

## 4. 当前可执行的采集与主机分析

容器编译和配置完成后，在**独立诊断目录**执行一个请求：

```bash
mkdir -p /work/profile-001/rknn
export OMNI_PROFILE_PATH=/work/profile-001/mnn.jsonl
export OMNI_RKNN_PERF_DIR=/work/profile-001/rknn
export OMNI_PREDIT_SYNC=1
export OMNI_RKNN_AUDIO=/assets/models/rknn/audio_conv_prefix_mnn4bit_fp16.rknn
export OMNI_TEST_SEED=20261105
/work/build/bin/vcb_eval /work/config.json \
  /assets/data/vcb_random50/bundle/manifest.jsonl \
  /assets/data/vcb_random50/bundle /work/profile-001/request 14 1 0
```

原bundle第14条是该历史语音样本，更换数据后必须核对。目录须为本次新目录，MNN采集器以追加方式写文件。记录实际命令、退出状态和构建身份；使用 `tools/benchmark/run_recorded.py` 包装命令可附带温度/频率/负载记录。

**当前采集器存在缓冲，分析器要求输入稳定**。可在请求完成后回传文件进行分析；持续低延迟分析需另行实现按请求封闭的分片、完成标志、回传队列和触发器。不能边读正在写的JSONL/SQLite，边把不完整结果当完整实测。主机工具包读取板端生成的数据，不会通过SSH自动变成设备侧profiler。

回传完成后，在本地主机运行项目采集器适配：

```bash
# /path/to/received 是已经回传完成的一次run。
python3 tools/profiling/convert_mnn_profile.py \
  /path/to/received/mnn.jsonl --out /path/to/new-mnn-import
python3 tools/profiling/normalize_rknn_perf.py \
  /path/to/received/rknn/tile_0.txt --out /path/to/new-rknn-profile.json
```

不依赖外部工具包的热点排名可以直接在仓库内完成（输出 `rankings.csv`、`by_type.csv`、`modules.csv`、`top.md`，列与历史 `mnn_operator_rankings.csv` 相同）：

```bash
python3 tools/analysis/rank_hotspots.py /path/to/received/mnn.jsonl --out /path/to/hotspots
# 或复算一份已有排名表：
python3 tools/analysis/rank_hotspots.py --from-rankings-csv reports/historical/evidence/mnn_operator_rankings.csv --out /path/to/hotspots
```

MNN适配按executor/stage分组，输出文件列表见 `import_manifest.json`。需要来源映射、ONNX候选与更深入分析时，在现有Mac工具包中选择其中一个实际 `*.profile.json`：

```bash
TOOLKIT_ROOT="$HOME/Desktop/RK3588_Mac_Analysis_Toolkit"
HOTSPOT_SKILL="$HOME/.codex/skills/rk3588-inference-hotspot-analysis"
"$TOOLKIT_ROOT/.venv/bin/python" "$HOTSPOT_SKILL/scripts/analyze_run.py" \
  --toolkit "$TOOLKIT_ROOT" \
  --profile /path/to/new-mnn-import/SELECTED.profile.json \
  --onnx /path/to/reference.onnx \
  --engine mnn --device rk3588 --evidence-kind measured \
  --name candidate-profile
```

`SELECTED.profile.json` 是占位文件名；ONNX缺失时省略该参数。`measured` 仅用于来源已核对的真实日志。Windows使用附录中的原生入口。报告、映射、子图合同保存在工具包本次 `runs/` 内，入库时整理其体积适中的副本及身份。

MNN callback包含宿主同步开销，RKNN报告的是编译单元，参考ONNX不一定是实际转换源图。保留 `candidate/partial/unmapped` 等状态；找到源码只算实现候选，实际调用还需dispatch或trace证据。闭源NPU kernel不能因为有profile就任意替换注册。

## 5. Git内的沉淀格式

本地任务工作区可用 [sandbox CLI](SANDBOX.md) 初始化 Task、Candidate 和 Run，登记外置资产，冻结快照并归档精选文件；它不启动实验。新Run从planned开始，已有历史目录不自动导入或改写。

每轮从 [run.template.json](../experiments/templates/run.template.json) 和 [design.template.md](../experiments/templates/design.template.md) 创建独立记录，填写已知字段，缺失保留null。模板自身标为 `template_not_measurement`，不可纳入实测样本。

目录布局、候选/运行命名、状态枚举和每轮必须提交的文件以 [CONVENTIONS.md](CONVENTIONS.md) 为准；候选登记在 `experiments/index.json`，运行登记在 `benchmarks/history/index.json`。推荐结构：

```text
experiments/index.json, <candidate-id>/candidate.json, design.md
principles/NN-<candidate-id>.md
kernels/<candidate-id>/, src/<engine>/overlays/<candidate-id>/
benchmarks/history/<run-id>/
  manifest.json, summary.json, validation.json
  bench/               小型逐次微基准、消融
  e2e/                 请求指标、比较门、配置
  correctness/         数值门与外置张量身份
  mapping/             ONNX来源、子图合同与实现证据
  records/             命令、退出状态、环境遥测
reports/<run-id>/       完成后的Markdown/HTML、图和分析摘要
```

本轮历史证据包采用上述结构；未来记录可以补充pipeline版本、所有源码/资产hash、container ID和自动调度ID。完整大trace、波形、模型继续放本地资产库，manifest记录稳定的资产ID/相对路径、大小和SHA256；临时签名URL、SSH凭据、机器专有访问配置不能作知识内容。

Git中只提交明确选中的小型文本/图表，避免整个 `runs/` 或工作目录直接入库。例：

```bash
# 替换candidate/run占位名称；提交前审阅实际文件与差异。
git add -- kernels/ src/mnn/ \
  experiments/CANDIDATE_ID benchmarks/history/RUN_ID reports/RUN_ID
git diff --cached --stat
git diff --cached
git commit -m "Record candidate implementation and benchmark evidence"
git push origin Qwen2.5-Omni-3B_VCB-Bench
```

本次已有上传授权；上例描述未来发布步骤，当前仓库不会运行后台自动push。后续接自动调度器时，发布应以已整理的文件清单为输入，并保留可审阅diff及运行状态。

## 6. 为以后训练模型保留什么

沉淀可检查的“观察 → 假设 → 补丁 → 验证 → 结果 → 决策”，而不是只存最终最快数字。字段包括父候选、失败原因、映射等级、环境、精度、单位、重复次数、噪声、数据来源/许可和适用范围。经验解释可以总结推理依据，不需要收集agent的隐藏思维过程。

同一capture的多shape变体、同一请求的暖机/重复、同一日志的CSV/摘要共享 `split_group`；未来按模型/工作负载/设备或来源组划分训练与评测，避免重复证据泄漏。原数据集的参考答案保留在评分侧，不进入被测prompt；报告与轨迹中也要注明哪些内容来自评分信息。

“正确但慢”“更快但数值失败”“编译失败”“无法映射”“未运行”都是不同标签。静态通过不等于数值通过，微基准收益不等于E2E收益，历史报告不等于新设备验收。

## 7. 当前实现与后续接入

| 环节 | 状态 |
|---|---|
| BigVGAN候选、MNN接入与回退 | 已有源码；原板历史验证 |
| 分级benchmark和离线统计 | 已有CLI；历史小型数据现可从Git复算 |
| 主机热点/ONNX来源分析 | 已有工具包及附录，按稳定文件运行 |
| arm64容器配方、挂载启动器 | 本次新增；参数测试通过，镜像/板端运行未验证 |
| 本地Task/Candidate/Run与归档索引 | 已补齐脚本与配置；流程测试未运行 |
| 台式机大资产管理 | 后续配置，本次不操作 |
| 按请求profile分片与持续回传 | 待实现 |
| 跨机任务队列、设备独占、agent自动循环 | 待实现；不操作其他人的工作负载 |
| 自动选择知识文件并提交Git | 待实现；当前显式整理并提交 |

下一步应先在第二台RK3588验证容器与当前候选，再接“稳定profile回传→分析→生成候选→分级评测”的自动调度。容器只是运行环境，不能代替硬件兼容、正确性和计时验收。
