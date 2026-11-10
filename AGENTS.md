# 给 Agent 与贡献者

在这个仓库里新增或修改文件前，先读两处：

1. 根目录 `workload.json`：当前分支是 `main`（共享工具 + SOP + CI）还是某个 `<Model>_<Dataset>` 工作负载分支。
2. [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md)：分支模型、目录布局、命名规则、状态枚举、文件体积与冻结证据规则、一轮候选必须提交的文件。

流程按 [`docs/SOP.md`](docs/SOP.md)；每个候选的优化原理写在 `principles/`。提交前运行：

```bash
python3 tools/verify/check_repo.py
python3 -m unittest discover -s tests -q
```

不要修改 `benchmarks/history/` 与 `reports/historical/` 中被 manifest 记录哈希的文件；不要把模型、张量、波形或二进制提交进 Git；不要在工作负载分支上直接改共享工具，改动应回到 `main` 再合并。
