# 仓库约定：分支、目录、命名与文件管理

这份文档是 `kernel_bench_opt` 所有分支共同遵守的规则。人和 Agent 往仓库提交文件前先读它；`python3 tools/verify/check_repo.py` 会机械地检查其中可检查的部分。SOP 流程本身见 [SOP.md](SOP.md)，本地任务管理见 [SANDBOX.md](SANDBOX.md)。

## 1. 分支模型

| 分支 | 内容 | 不放什么 |
|---|---|---|
| `main` | 共享工具（`tools/`、`scripts/` 中与模型无关的脚本、`tests/`）、SOP 与约定文档、CI、模板、JSON schema | 任何模型/数据集专属的 kernel、引擎覆盖源码、实验记录、性能数据、报告、资产清单 |
| `<Model>_<Dataset>`（工作负载分支） | 某个模型 + 测试数据集协议下的全部内容：候选源码、引擎改动、实验、评测证据、报告、资产身份、原理文档，**外加**从 `main` 合并来的共享工具 | 其他工作负载的内容 |

- 分支名用 `<Model>_<Dataset>`，模型名保留官方大小写与点号，用下划线分隔模型与数据集，例如 `Qwen2.5-Omni-3B_VCB-Bench`。已有的 `yolov8n-COCO` 是早期分支，保持原样，不迁移、不改动。
- 每个工作负载分支根目录有 `workload.json`（`branch_kind: "workload"`），`main` 根目录有 `workload.json`（`branch_kind: "main"`）。脚本据此判断分支类型。
- **共享工具只在 `main` 上修改**，然后在工作负载分支执行 `git merge main`。`main` 与工作负载分支之间已建立合并基准（`-s ours` 空合并），因此 `git merge main` 只带来共享文件的差异，不会删除工作负载文件。在工作负载分支上改了共享工具的，先 cherry-pick 回 `main`。
- 新建工作负载分支：`git checkout -b <Model>_<Dataset> main`，然后创建 `workload.json` 和下列目录。不要从另一个工作负载分支派生。
- `README.md`、`README.html`、`workload.json` 是每个分支各自的文件，`.gitattributes` 把它们标为 `merge=ours`；每个 clone 执行一次 `git config merge.ours.driver true`，之后 `git merge main` 不会用 `main` 的版本覆盖它们。

## 2. 工作负载分支的目录

```text
workload.json                     分支元数据：模型、数据集、引擎、设备、索引位置
experiments/
  index.json                      候选登记表（机器可读；每个候选一项）
  <candidate-id>/candidate.json   候选身份、父候选、假设、状态、源码哈希、运行列表
  <candidate-id>/design.md        观察 → 假设 → 变换 → 评测 → 结果 → 决策
  <candidate-id>/*.patch          该候选独有的补丁（如有）
  templates/                      run / design 模板
principles/
  README.md                       原理文档索引
  NN-<candidate-id>.md            每个候选一份优化原理
kernels/<candidate-id>/           独立 kernel 源码、内嵌头（生成物）、对拍/调参程序
src/<engine>/                     引擎公共覆盖文件（所有候选集共用，文件名与上游一致）
src/<engine>/overlays/<candidate-id>/   候选专属的引擎覆盖文件，文件名与上游一致
src/apps/                         评测程序
scripts/patches/manifest.json     引擎覆盖清单：patches + common_overlays + overlay_sets
benchmarks/history/
  index.json                      运行登记表（每个 run 一项，含 planned）
  <run-id>/run.json               运行记录（planned → completed/failed/cancelled）
  <run-id>/{manifest,summary,validation}.json 与 bench/ e2e/ correctness/ mapping/ records/
reports/<run-id>/                 完成后的 Markdown 报告与图
assets/manifest.json, assets/runs/<run-id>.json   外置大资产（Release）的身份与哈希
docs/                             本工作负载的构建、复现、部署、状态文档
```

`src/<engine>/` 根目录只放公共覆盖文件；凡是只属于某个候选的文件一律放 `overlays/<candidate-id>/`，并用上游文件名命名，这样 `manifest.json` 里的 `path` 一眼能对上。

## 3. 命名

| 对象 | 规则 | 例子 |
|---|---|---|
| 候选 `candidate_id` | 小写 kebab-case：`<模块>-<算子或子图>-<变体>`；迭代用 `-v<N>` 后缀；系统级候选用动作名 | `bigvgan-convtranspose-tile4`、`bigvgan-deconv1d-v2`、`rknn-audio-prefix-v2`、`predit-sync` |
| 候选集 `overlay_sets` 键 | 短名，表示一次可部署的组合 | `tile4`、`v2` |
| 运行 `run_id` | `YYYYMMDD-HHMMSS-<task>-<level>`，level ∈ `l1`/`l2`/`l3`/`profile`；时间为运行（或登记 planned）时刻 | `20261110-180000-bigvgan-deconv1d-v2-l1` |
| 原理文档 | `principles/NN-<candidate-id>.md`，NN 两位递增 | `principles/03-bigvgan-deconv1d-v2.md` |
| kernel 文件 | `kernels/<candidate-id>/<kernel>.cl`、生成头 `<kernel>_source.hpp`、对拍程序 `bench_<name>.cpp` | — |
| 引擎覆盖文件 | 与上游同名 | `DeconvBufExecution.cpp` |
| 报告/证据目录 | 与 `run_id` 同名 | `reports/<run-id>/` |
| 环境变量开关 | `OMNI_<模块>_<开关>`，默认未设置 = 基线行为 | `OMNI_BIGVGAN_DECONV_V2=1` |
| 提交信息 | 首行 `<范围>: <做了什么>`，正文写候选/运行 ID 与证据边界；末尾保留 `Assisted-by:` 等 trailer | — |

历史运行 `20261106-bigvgan-convtranspose`、`20261109-112115-omni-combined-e2e-random10` 早于本规则，保留原名，在 `benchmarks/history/index.json` 中标记 `legacy_id: true`。

## 4. 状态枚举

`candidate.json.status` 只能取下列值，自由文字写在 `status_detail`：

| status | 含义 |
|---|---|
| `proposed` | 只有设计，没有源码 |
| `source_delivered` | 源码已交付，未在目标设备运行 |
| `L1_passed` / `L2_passed` / `L3_passed` | 对应级别正确性与性能记录通过（以最高通过级别为准） |
| `L3_passed_single_sample` | L3 只有单样本描述性结果 |
| `rejected_numeric` | 数值/输出门失败 |
| `rejected_perf` | 数值通过但没有收益或退化 |
| `superseded` | 被子候选取代，保留记录 |
| `abandoned` | 停止，原因写在 `status_detail` |

`run.json.result.status`：`planned` / `completed` / `failed` / `cancelled`。`completed` 表示流程完成，不表示通过；通过与否看 `correctness` 与 `measurement`。

## 5. 文件内容规则

- 新 JSON 文件带 `schema_version` 和 `record_type`，UTF-8、无 BOM、2 空格缩进、末尾换行。schema 见 [`schemas/`](../schemas/)。历史文件 `reports/historical/evidence/mnn_operator_rankings.csv` 带 BOM，读取时用 `encoding="utf-8-sig"`，不要为去 BOM 改写它。
- **冻结的证据目录**：`benchmarks/history/<run-id>/` 和 `reports/historical/` 中被 manifest 记录了哈希的文件一个字节都不改。导航链接失效时，在本文档第 7 节的路径映射表里查，不改原文件。
- 体积：单个文本文件 ≤ 8 MiB，一个 run 目录 ≤ 32 MiB；超过 1 MiB 的 `.jsonl`/`.csv` 压成 `.gz` 再入库。模型、张量、波形、完整 trace、二进制和压缩包走 GitHub Release，在 `assets/manifest.json` 记录 URL、大小、SHA-256。
- 生成物（`*_source.hpp`、`README.html`）要么由脚本重新生成并一起提交，要么由检查脚本报告过期；不要手改。
- 每个数字都写计时口径（host enqueue→finish / 设备 event / 宿主 callback / 完整请求）和设备；非目标设备上的数字标注"仅说明方向"。
- 不写入机器绝对路径、凭据、临时签名 URL；路径用 `${TASK_ROOT}` 之类的符号根。

## 6. 一轮候选要提交的文件

| 时机 | 必须有 | 位置 |
|---|---|---|
| 设计完成 | `candidate.json`（status=proposed）、`design.md`、`principles/NN-<id>.md`、`experiments/index.json` 新增一项 | `experiments/<id>/`、`principles/` |
| 源码交付 | kernel / overlay 源码、`manifest.json` 新候选集或更新、`candidate.json.source_files` 哈希、planned 的 `run.json` 与 `benchmarks/history/index.json` 条目 | `kernels/<id>/`、`src/<engine>/overlays/<id>/`、`benchmarks/history/<run-id>/` |
| 运行完成 | `run.json` 终态、`summary.json`、`validation.json`、`manifest.json`（哈希）、小型原始记录、报告、`assets/runs/<run-id>.json` | `benchmarks/history/<run-id>/`、`reports/<run-id>/` |
| 决策 | `candidate.json.status` 更新、`STATUS.md` 一段、失败原因保留 | — |

提交前运行：

```bash
python3 tools/verify/check_repo.py          # 约定与一致性检查（CI 同样执行）
python3 -m unittest discover -s tests -q
```

## 7. 历史路径映射（2026-11-10 重组）

重组用 `git mv`，文件内容与 SHA-256 不变。冻结证据中的旧路径按此表解析：

| 旧路径 | 新路径 |
|---|---|
| `kernels/bigvgan_deconv_s5_tiled.cl` | `kernels/bigvgan-convtranspose-tile4/bigvgan_deconv_s5_tiled.cl` |
| `kernels/bigvgan_deconv_source.hpp` | `kernels/bigvgan-convtranspose-tile4/bigvgan_deconv_source.hpp` |
| `kernels/bench_deconv.cpp` | `kernels/bigvgan-convtranspose-tile4/bench_deconv.cpp` |
| `kernels/v2/*` | `kernels/bigvgan-deconv1d-v2/*` |
| `src/mnn/DeconvBufExecution.candidate.cpp` | `src/mnn/overlays/bigvgan-convtranspose-tile4/DeconvBufExecution.cpp` |
| `src/mnn/DeconvBufExecution.task.hpp` | `src/mnn/overlays/bigvgan-convtranspose-tile4/DeconvBufExecution.task.hpp` |
| `src/mnn/DeconvBufExecution.v2.cpp` | `src/mnn/overlays/bigvgan-deconv1d-v2/DeconvBufExecution.cpp` |
| `src/mnn/DeconvBufExecution.v2.hpp` | `src/mnn/overlays/bigvgan-deconv1d-v2/DeconvBufExecution.v2.hpp` |
| `src/mnn/omni_rknn_prefix.hpp` | `src/mnn/overlays/rknn-audio-prefix-conv-gelu/omni_rknn_prefix.hpp` |
| `src/mnn/omni_rknn_prefix.v2.hpp` | `src/mnn/overlays/rknn-audio-prefix-v2/omni_rknn_prefix.hpp` |
| `experiments/predit-sync.patch` | `experiments/predit-sync/predit-sync.patch` |

机器可读版本在 [`experiments/path_moves.json`](../experiments/path_moves.json)，`check_repo.py` 用它解析冻结证据里的旧链接。

## 8. 给 Agent 的最短说明

1. 读根目录 `workload.json` 判断分支类型；`main` 上不放工作负载文件，工作负载分支上不改共享工具（改了就 cherry-pick 回 `main`）。
2. 新候选：先 `experiments/<id>/candidate.json` + `design.md` + `principles/NN-<id>.md`，再放源码到 `kernels/<id>/` 与 `src/<engine>/overlays/<id>/`，更新 `experiments/index.json`、`scripts/patches/manifest.json`。
3. 新运行：`benchmarks/history/<run-id>/run.json` 从 planned 开始，终态后补 summary/validation/manifest 与 `reports/<run-id>/`，更新 `benchmarks/history/index.json`。
4. 冻结证据不改；大文件走 Release；提交前跑 `check_repo.py`。
