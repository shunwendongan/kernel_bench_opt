# Task / Candidate / Run 本地管理

本模块把 README 的任务工作区落成文件与命令：初始化任务、登记外置资产、冻结候选和运行身份、归档精选结果、重建和查询索引。使用 Python 3.10+ 标准库，无新增 Python 依赖。

本次交付只准备代码与配置，未执行任务管理流程测试、模型推理、容器构建或板端操作。以下命令是后续使用示例。

## 1. 范围与文件关系

```text
.sandbox/                              默认本地目录，Git忽略
├── workspace.json                     工作区身份
├── index.json                         可重建的查询缓存
├── tasks/<task-id>/
│   ├── task.json                      目标、模型、数据集、分支、标签
│   ├── assets.json                    资产引用清单，不复制模型/数据
│   ├── baseline/
│   ├── candidates/<candidate-id>/
│   │   ├── candidate.json             父候选、假设、代码身份
│   │   ├── design.md
│   │   └── patch/
│   └── runs/<run-id>/
│       ├── run.json                   候选与资产快照、状态、结果
│       ├── raw/
│       ├── analysis/
│       ├── benchmark/
│       └── artifacts/
└── archives/<task-id>/<run-id>/
    ├── manifest.json                  精选文件的大小、哈希、来源
    ├── metadata/                      任务、候选、运行记录副本
    └── files/                         显式选中的结果文件
```

`Task` 定义问题，`Candidate` 定义相对父实现的改动，`Run` 定义一次已计划或已完成的工作。一个候选可以有多个运行；任务之间可以引用同一外置资产，但各自保留资产清单和输出。

**`run-init` 只准备目录和记录。** 本模块不调用 Docker、SSH、推理引擎、profiler 或 Git。设备调度、持续回传与 Agent 自动循环仍未接入。工作区写锁只协调本模块的本地写入，不是硬件独占锁或安全沙箱。

## 2. 初始化任务与候选

从仓库根目录调用；也可用任意位置的绝对脚本路径。`--root` 相对当前 shell 目录解析，所有命令必须指向同一工作区。

```bash
python3 scripts/sandbox.py --root .sandbox init

python3 scripts/sandbox.py --root .sandbox task-init \
  --task bigvgan-tail-shapes \
  --title "BigVGAN 尾块形状优化" \
  --objective "在保持部署权重与精度的前提下评估新的尾块 shape" \
  --model Qwen2.5-Omni-3B --dataset VCB-Bench \
  --branch Qwen2.5-Omni-3B_VCB-Bench --tag rk3588 --tag convtranspose

# 用实际源码commit替换占位值；不会自动检查Git或创建分支。
python3 scripts/sandbox.py --root .sandbox candidate-init \
  --task bigvgan-tail-shapes --candidate baseline \
  --hypothesis "冻结原执行路径" --source-commit YOUR_SOURCE_COMMIT

python3 scripts/sandbox.py --root .sandbox candidate-init \
  --task bigvgan-tail-shapes --candidate tail-v1 --parent baseline \
  --hypothesis "在维持累加顺序的前提下处理真实尾块" \
  --source-commit YOUR_CANDIDATE_COMMIT
```

`--source-commit` 要求实际完整40或64位十六进制SHA；示例中的 `YOUR_SOURCE_COMMIT` 和 `YOUR_CANDIDATE_COMMIT` 必须替换，不能原样执行。

ID 使用最多96个 ASCII 字母、数字、点、下划线或短横线，并以字母/数字开头。相同任务、候选、运行和资产 ID 不会被覆盖。父候选必须在当前任务中存在。候选的设计文档留有待填写项，不会生成假的实验结论。

## 3. 登记资产：文件校验与外部引用分开

登记**已有本地单文件**会流式读取文件，计算 SHA256 和大小；提供 `--sha256` 或 `--size-bytes` 时必须一致。目录不递归扫描，大型文件不会被复制进工作区。

```bash
python3 scripts/sandbox.py --root .sandbox asset-add \
  --task bigvgan-tail-shapes --asset capture-manifest-v1 --kind capture \
  --path /path/to/existing/capture-manifest.json \
  --source "已归档的部署采集清单"
```

只登记尚未挂载的资产，使用 `--uri`：

```bash
python3 scripts/sandbox.py --root .sandbox asset-add \
  --task bigvgan-tail-shapes --asset deployed-model-v1 --kind model \
  --uri asset://models/qwen2.5-omni-3b/deployed-v1 \
  --source "当前部署模型的版本清单" --license "见模型来源许可"
```

`asset://` 是本地约定的身份引用，不是本模块实现的下载协议。URI不会被访问；声明的哈希/大小即便填写，也只标为调用者提供，不能冒充本机核验。不要把登录令牌、临时签名下载地址或凭据写入清单。实际模型、音频、完整trace继续外置。

## 4. 创建运行快照与登记结果

```bash
python3 scripts/sandbox.py --root .sandbox run-init \
  --task bigvgan-tail-shapes --candidate tail-v1 \
  --run board-b-l1-001 --level L1 --device-id rk3588-board-b \
  --asset deployed-model-v1 --asset capture-manifest-v1
```

此时状态为 `planned`，候选记录与显式选择的资产清单被复制为快照；后续资产清单变化不会重写已有 Run。未填写的设备/源码信息保留未知，不能据此声称真实后端已运行。`--level` 可选 `L1`、`L2`、`L3` 或 `profile`。

实际实验由已有 runner 在另行授权的阶段执行。完成后，将已关闭的日志、分析和结果放入该 Run 的相应目录，再登记终态：

```bash
# 以下命令只登记已有结果；结果文件必须来自实际工作，不能直接用空模板。
python3 scripts/sandbox.py --root .sandbox run-finish \
  --task bigvgan-tail-shapes --run board-b-l1-001 \
  --status completed \
  --result .sandbox/tasks/bigvgan-tail-shapes/runs/board-b-l1-001/benchmark/result.json \
  --note "结果来源与计时口径见result.json及原始记录"
```

状态只允许 `planned → completed / failed / cancelled`。`completed` 需要 JSON 对象形式的结果文件；标记 `template_not_measurement` 或 `not_run` 的模板不能作为 completed 结果。失败/取消可用 `--note` 记录原因，并可附上错误结果。

结果以 `caller_declared` 标明来源，**completed 表示调用者登记的流程完成，不自动表示正确性通过或性能更优**。使用 [结果模板](../configs/sandbox-result.template.json) 组织正确性、计时域、单位、原始记录、决策与来源组；真正测量后再填写实际标记。终态记录不重开或覆盖，修正实验应建立新的 Run 并说明关系。

## 5. 精选文件归档

```bash
python3 scripts/sandbox.py --root .sandbox archive-run \
  --task bigvgan-tail-shapes --run board-b-l1-001 \
  --include benchmark/result.json \
  --include benchmark/results.csv \
  --include analysis/hotspots.json
```

只能归档终态 Run，且至少显式选择一个文件。路径相对 Run 目录，不接受绝对路径、`..`、目录、glob 或软链接；不会自动把整个工作目录搬走。任务与运行元数据一同归档，候选记录取自 Run 初始化时的快照。

默认单文件不超过 **8 MiB**、归档总量不超过 **32 MiB**，元数据也计入限额。允许 JSON/CSV/Markdown、静态图片和源码补丁等文本；模型、张量、波形和压缩包不在默认白名单中。默认也不收HTML，完整历史报告可在外部保存并记录引用。配置见 [sandbox.default.json](../configs/sandbox.default.json)。

归档写入临时目录，文件完成复制并记录哈希后才封存到 `archives/<task>/<run>/`。已有归档不覆盖，原 Run 和原始文件不删除。“封存”是脚本的写入约定，不等于操作系统只读、数字签名或不可篡改存储。

## 6. 查询、查看与重建索引

```bash
python3 scripts/sandbox.py --root .sandbox list --kind task
python3 scripts/sandbox.py --root .sandbox list --kind run --task bigvgan-tail-shapes --level L1
python3 scripts/sandbox.py --root .sandbox list --kind archive --status completed
python3 scripts/sandbox.py --root .sandbox list --tag rk3588

python3 scripts/sandbox.py --root .sandbox show \
  --kind run --task bigvgan-tail-shapes --id board-b-l1-001

python3 scripts/sandbox.py --root .sandbox index-rebuild
```

每次通过 CLI 修改主记录后会重建索引；`list` 使用这个缓存，`show` 直接读取选定记录。手动编辑了 JSON 后，需要主动 `index-rebuild`。索引损坏可以重建，主记录才是来源；重建不扫描模型内容、不重新计算性能。

主记录保存成功但索引更新失败时，CLI 会报告 `record_saved: true` 并返回非零状态；先处理索引错误，再重建，避免误以为写入已回滚而重复创建。字段/关系错误不会静默跳过。写锁残留时先确认没有写入进程，再由维护者处理 `.write.lock`，本模块不自动抢锁。进程异常退出留下的 `.pending-*` / `.tmp-*` 临时目录也会使索引明确报错；确认没有写入进程后，检查其中内容再处理，避免直接忽略可能尚未发布的记录。

## 7. 发布到 Git 的边界

`.sandbox/` 默认忽略。它含本地路径、原始结果和待审阅草稿，不整体提交。归档以后选择与任务相关的小型记录，核对来源路径、许可与报告引用，再复制到仓库既定 `benchmarks/history/`、`reports/` 或候选目录并提交。

本模块没有自动 Git push，也没有宣称提供自动敏感信息清理。归档清单保留内容身份，发布副本若做路径归一化，需要同时记录原始与发布哈希。已有历史实验目录保持原组织方式，这次不把历史数据伪装成新 Run。

## 8. 单文件图文 README

GitHub 的 `README.md` 继续引用仓库里的 SVG，打开页面时直接显示。需要一个文件包含完整正文和全部图片时，使用工作负载分支根目录的 `README.html`，下载后直接在浏览器打开。

更新 Markdown 或图后，在有 Pandoc 的文档主机运行：

```bash
python3 scripts/export_readme.py
```

导出脚本嵌入图片字节与样式，不依赖外部字体、图片目录或脚本；其他源码/文档链接指向 GitHub。导出不是模型测试，也不会启动任务。HTML保存 README 与图片来源哈希，便于判断它对应哪一版正文。
