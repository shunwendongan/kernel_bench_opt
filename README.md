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
