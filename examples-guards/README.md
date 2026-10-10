# examples-guards 目录：example 看护的项目数据

本目录存放 example 看护线的项目配套数据，每个子目录对应一个上游项目。看护流水线在 `.github/workflows/<project>-examples.yml`，公共回路在 `.github/workflows/examples-template.yml`。Quick Start 看护是另一条线，测试在 `tests/<project>/`，文档在 `sources/<project>/`，与本目录无关。

## 分工

- 引擎 `examples-template.yml`：监控、调度、结果发布、状态回写的全回路，不含任何项目名。
- 薄触发器 `<project>-examples.yml`：只声明项目变量，包括 cron、并发组、PR 触发的 `paths`、`project`、`upstream_repo`、dispatch 输入转发和矩阵并行度。
- 项目数据 `examples-guards/<project>/`：`examples_manifest.yaml` 声明 supported 条目即调度参数，`scripts/setup_example.sh` 与 `scripts/run_example.sh` 实现环境准备与运行，`fixtures/` 提供数据，`constraints-npu.txt` 钉住依赖。
- 共享脚本 `scripts/check_supported_entries.py`、`scripts/write_example_result.py` 与结果 schema `.github/workflows/schemas/result.schema.json` 由全部项目共用，修改前先确认对每个接入项目都成立。

加删一条 supported example 只改清单；example 环境变化只改项目脚本；看护回路演进只改引擎。

## 看护回路

schedule 定时轮询两个信号：上游最新 release，以及本项目看护文件的变化。信号触发时，把清单 supported 条目在最新 release tag 上逐条跑矩阵；每条结果写入 result.json 并作为 artifact 上传；矩阵结论回写监控缓存，失败驱动下一轮重试。上游从未发布 release 时，改测默认分支的最新提交。查询 GitHub API 失败时本轮不跑，监控状态不动，下一轮再查。

## 模型与数据集缓存

每个 GitHub 组织的 self-hosted runner 池各挂一块持久缓存盘，同池 runner 共用，跨次运行保留；别的组织投递过的资产在本仓看不见。本仓 runner 池能直连 huggingface.co，项目的 setup 与 run 脚本把 `HF_ENDPOINT` 设为 huggingface.co，不用引擎默认的 hf-mirror：后者把 `t5-base` 这类无命名空间的 id 308 跳转到 huggingface.co，huggingface_hub 不跟随这种跳转。peft 是按这套写法接入的范例。

- 例程硬编码的模型与数据集由 `from_pretrained`、`load_dataset` 自行下载进缓存，下次运行直接命中。
- 以本地路径传给 overlay_args 的模型由项目的 `scripts/hub_cache.py` 解析：缓存里的快照必需文件齐全、且每个 safetensors 文件头完好，就直接用，不联网；否则先从 HuggingFace 下载，失败再从 ModelScope 下载。
- 缓存盘跨 pod 共享，huggingface_hub 自带的 `.locks` 没拦住并发：run 37764199490 里两条 dreambooth 腿同时下 SD v1.5，追加写到同一个文件上，落得比完整文件还大。所以 `hub_cache.py` 不直接下载进缓存，先下到本 job 在同一块盘上的私有暂存目录，校验通过后再逐个文件改名进快照，完好的旧文件保持不动。同一模型另有一个 `mkdir` 锁目录让其余 job 等待，锁只省带宽，不承担正确性。
- 损坏的分片连同它指向的 blob 必须先删掉再重下：`snapshot_download` 会把缓存里已有的 blob 直接复制给下载目标且不重新校验，不删就会把坏字节一路带下去。
- 没有单独的预投递流水线，也不把模型或数据集提交进仓库。

## 结果契约

result.json 必填六个字段：trigger、target_repo、target_ref、path、image、job_status。job_status 只取 success、failure、cancelled。artifact 名为 `<project>-examples-<run_id>-<job-index>`，内容只有 result.json。上传永远在 GitHub 托管 runner 上完成，NPU job 只跑 example 并报告自身状态。

## PR 运行

每个项目的 example 看护 PR 不需要合入即可运行看护。原理：pull_request 事件执行的是 PR merge ref 上的 workflow 文件，PR 里新增的薄触发器和引擎当时即可运行，reviewer 在 checks 页直接看到矩阵结论。

PR 运行语义：

- 只看本项目自己的文件：PR 改动 `examples-guards/<project>/**` 或本项目触发器时才运行。只改共用引擎、共享脚本或 schema 的 PR 不启动任何项目的矩阵，与 Quick Start 触发器一致；这类改动合入后由维护者手动运行受影响的项目。
- 必然执行全量 supported 矩阵。被测 ref 与不填 target_ref 的手动运行相同：取上游最新 release tag，查不到时回落到默认分支。
- 不读不写监控缓存，不影响定时看护的基线状态。
- 同一 PR 新的提交自动取消未完成的旧运行，避免占用 NPU。并发组名必须带项目前缀，即 `<project>-examples-pr-<PR 号>`：组名在整个仓库内共享，不带前缀时，多个项目在同一 PR 上的运行会互相取消。
- result.json 的 trigger 字段记为 pull_request，artifact 命名与定时运行一致。汇总定时看护结论的外部系统按 trigger 字段过滤掉 PR 运行。
- 来自 fork 的 PR 是否需要维护者先批准才能运行，取决于仓库或组织的 fork PR 审批策略，GitHub 默认要求首次贡献者批准一次；同仓分支的 PR 不受此限制。

## 迁移一个新项目

从 cosdt-ci-test/workflows 迁移共用模板套项目：

1. 把源仓 `projects/<project>/` 下的 example 线文件拷到本仓 `examples-guards/<project>/`：清单、脚本、fixtures、constraints。Quick Start 线的 docs 与 tests 不拷贝，它们已在本仓单独迁移。源仓 setup 里依赖 cache-seed 预投递的资产，改成上文「模型与数据集缓存」一节的写法。
2. 新建薄触发器 `.github/workflows/<project>-examples.yml`，参照 `peft-examples.yml`：schedule 保持注释；声明 pull_request 触发，`paths` 只列 `examples-guards/<project>/**` 和触发器自身；PR 并发组名用 `<project>-examples-pr-<PR 号>`；填入 `project` 与 `upstream_repo`。
3. 用 actionlint 检查 workflow。
4. 提 PR 后确认矩阵在 PR 上真实运行且结论符合预期。诚实红条目原样保留，跑红的不能改成 unsupported 换绿灯。
5. PR 合入后手动 dispatch 跑绿几轮。是否打开 cron 由维护者另行决定；打开时槽位错开已有排布，并同时关闭旧仓同项目的 schedule，避免双看护占用 NPU。

## 诚实红约定

清单允许诚实红：在昇腾上真实失败的条目保持 supported 并如实标红。看护噪音例如镜像拉取失败、下载限流、标签排队单独归类，不算被测失败，也不因此放宽清单。
