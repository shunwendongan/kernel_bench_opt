# JSON Schema

仓库记录文件的结构定义（draft-07 子集：type / required / properties / enum / pattern / items）。`tools/verify/check_repo.py` 内置一个只支持这些关键字的校验器，不依赖第三方库。

| schema | 校验对象 |
|---|---|
| `workload.schema.json` | 根目录 `workload.json` |
| `candidate.schema.json` | `experiments/<candidate-id>/candidate.json`（schema_version 2） |
| `candidate_index.schema.json` | `experiments/index.json` |
| `run_index.schema.json` | `benchmarks/history/index.json`（schema_version 2） |
| `run.schema.json` | `benchmarks/history/<run-id>/run.json` |
| `path_moves.schema.json` | `experiments/path_moves.json` |

状态枚举与命名正则的含义见 [docs/CONVENTIONS.md](../docs/CONVENTIONS.md)。历史运行目录里的 `manifest.json` / `summary.json` 属于冻结证据，有两种既有结构（schema_version 1），不在这里强制。
