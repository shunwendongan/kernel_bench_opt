# kernel_bench_opt · main

`main` 只保存**所有工作负载分支共享**的东西：端侧 benchmark → 分析 → 优化 → 沉淀的 SOP、仓库约定、本地任务管理 CLI、检查脚本、记录模板与 JSON schema、单元测试和 CI。模型/数据集专属的 kernel、推理引擎改动、实验记录、性能数据和报告都在各自的 `<Model>_<Dataset>` 分支上。

| 分支 | 内容 |
|---|---|
| [`Qwen2.5-Omni-3B_VCB-Bench`](https://github.com/shunwendongan/kernel_bench_opt/tree/Qwen2.5-Omni-3B_VCB-Bench) | Qwen2.5-Omni-3B 在 VCB-Bench 协议下的候选 kernel、MNN/RKNN 改动、评测证据与报告 |
| [`yolov8n-COCO`](https://github.com/shunwendongan/kernel_bench_opt/tree/yolov8n-COCO) | 早期分支，保持原有组织方式 |

## 这里有什么

```text
AGENTS.md                 给 Agent 与贡献者的最短说明
docs/SOP.md               一轮实验的固定步骤、三级 benchmark 的含义、入库格式
docs/CONVENTIONS.md       分支模型、目录布局、命名规则、状态枚举、文件与体积规则
docs/SANDBOX.md           Task / Candidate / Run 本地管理 CLI
schemas/                  candidate / run / index / workload 等记录文件的 JSON schema
experiments/templates/    run.json 与 design.md 模板
configs/                  sandbox 配置与结果模板
tools/verify/check_repo.py    约定与一致性检查（CI 同样执行）
tools/sandbox/            任务管理 CLI 实现（scripts/sandbox.py 是入口）
tools/profiling/          MNN callback JSONL / RKNN perf 的格式转换
tools/analysis/           rank_hotspots.py 热点排名、compare_raw.py 数值比较
tools/benchmark/run_recorded.py   带温度/频率记录地运行一条命令
scripts/                  prepare_mnn.py（按 manifest 覆盖引擎源码）、资产下载/发布/打包、README 导出
tests/                    以上脚本的单元测试
```

## 新建一个工作负载分支

```bash
git checkout -b <Model>_<Dataset> main
# 写 workload.json（branch_kind: workload），建立 experiments/ kernels/ src/<engine>/ benchmarks/history/ reports/ principles/ assets/ docs/
python3 tools/verify/check_repo.py
```

目录布局、候选 ID / 运行 ID 命名和每轮必须提交的文件见 [docs/CONVENTIONS.md](docs/CONVENTIONS.md)。

## 命名与文件管理规则（Agent 必读摘要）

往本仓库任何分支提交文件前，先读根目录 `workload.json` 判断分支类型，再按下表命名。完整规则、理由和例外见 [docs/CONVENTIONS.md](docs/CONVENTIONS.md)，`tools/verify/check_repo.py` 会在 CI 中机械检查。

**分支归属**

| 分支类型 | 放什么 | 不放什么 |
|---|---|---|
| `main`（`branch_kind: main`） | 共享工具、SOP、约定、schema、模板、测试、CI | 任何模型/数据集专属的 kernel、引擎改动、实验、性能数据、报告 |
| `<Model>_<Dataset>`（`branch_kind: workload`） | 该工作负载的候选源码、引擎改动、实验记录、评测证据、报告、原理文档 | 其他工作负载的内容；对共享工具的独立修改 |

`yolov8n-COCO` 是早期分支，保持原样，不迁移、不改动。

**命名**

| 对象 | 规则 | 例子 |
|---|---|---|
| 工作负载分支 | `<Model>_<Dataset>`，模型名保留官方大小写与点号 | `Qwen2.5-Omni-3B_VCB-Bench` |
| 候选 ID | 小写 kebab-case `<模块>-<算子或子图>-<变体>`，迭代加 `-v<N>` | `bigvgan-deconv1d-v2` |
| 运行 ID | `YYYYMMDD-HHMMSS-<task>-<level>`，level ∈ `l1`/`l2`/`l3`/`profile` | `20261110-150000-bigvgan-deconv1d-v2-l1` |
| 候选集（overlay set） | 短名，表示一次可部署的组合 | `tile4`、`v2` |
| kernel 目录与文件 | `kernels/<候选ID>/<kernel>.cl`、生成头 `<kernel>_source.hpp`、对拍程序 `bench_<name>.cpp` | — |
| 引擎覆盖文件 | `src/<engine>/overlays/<候选ID>/`，文件名与上游一致 | `DeconvBufExecution.cpp` |
| 原理文档 | `principles/NN-<候选ID>.md`，NN 两位递增 | `principles/03-bigvgan-deconv1d-v2.md` |
| 证据与报告目录 | 与运行 ID 同名 | `benchmarks/history/<运行ID>/`、`reports/<运行ID>/` |
| 环境变量开关 | `OMNI_<模块>_<开关>`，未设置即基线行为 | `OMNI_BIGVGAN_DECONV_V2=1` |
| 提交信息 | 首行 `<范围>: <做了什么>`，正文写候选/运行 ID 与证据边界 | — |

**状态**：`candidate.json.status` 只取 `proposed` / `source_delivered` / `L1_passed` / `L2_passed` / `L3_passed` / `L3_passed_single_sample` / `rejected_numeric` / `rejected_perf` / `superseded` / `abandoned`，自由文字写在 `status_detail`。`run.json.result.status` 只取 `planned` / `completed` / `failed` / `cancelled`。

**一轮候选要提交的文件**

1. 设计：`experiments/<候选ID>/candidate.json` + `design.md`、`principles/NN-<候选ID>.md`，并在 `experiments/index.json` 登记。
2. 源码：`kernels/<候选ID>/`、`src/<engine>/overlays/<候选ID>/`，更新 `scripts/patches/manifest.json` 的候选集和 `candidate.json` 的源码哈希，登记 planned 的 `run.json`。
3. 运行完成：`run.json` 终态、`summary.json`、`validation.json`、带哈希的 `manifest.json`、`reports/<运行ID>/`，更新 `benchmarks/history/index.json`。
4. 决策：更新候选状态，在 `docs/STATUS.md` 写一段，失败原因保留。

**文件规则**

- 冻结证据（`benchmarks/history/<运行ID>/` 与 `reports/historical/` 中被 manifest 记录哈希的文件）一个字节都不改；旧路径按 `experiments/path_moves.json` 解析。
- 新 JSON 带 `schema_version` 与 `record_type`，UTF-8 无 BOM，2 空格缩进。
- 单个文本文件不超过 8 MiB，一个运行目录不超过 32 MiB；模型、张量、波形、trace、二进制走 GitHub Release，并在 `assets/manifest.json` 记录 URL、大小和 SHA-256。
- 生成物（`*_source.hpp`、`README.html`）只由脚本生成，不手改。
- 每个数字注明计时口径和设备；不写机器绝对路径、凭据或临时签名 URL。

## 同步共享工具

共享工具只在 `main` 上修改。工作负载分支执行：

```bash
git config merge.ours.driver true   # 每个 clone 一次：README.md / README.html / workload.json 保留分支自己的版本
git merge main
```

`main` 与现有工作负载分支之间已有合并基准，`git merge main` 只带来共享文件的变化。如果在工作负载分支上改了共享工具，先 cherry-pick 回 `main`。

## 提交前

```bash
python3 tools/verify/check_repo.py
python3 -m unittest discover -s tests -q
```
