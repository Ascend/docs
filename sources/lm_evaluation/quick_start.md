# lm-evaluation-harness

[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)（lm-eval）是统一的大模型评测框架。本示例在单卡昇腾 NPU 上用 HuggingFace 后端跑通两个官方任务：`arc_easy` 与 `winogrande`。

## 前置条件

### 硬件

Atlas 900 A2 单卡（Ascend 910B），并按需完成物理机或容器内的设备挂载。

### 基础软件

在运行本文档示例之前，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

本文档示例在 Python 3.12、CANN 9.1.0 环境下验证通过。

本文档配套镜像：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

## 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装 PyTorch 软件栈

安装与 CANN 配套的 PyTorch 和 Torch-NPU，并查看安装版本：

```shell #test id="install-torch"
pip install torch==2.9.0 torch_npu==2.9.0.post2
python -c "import torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__)"
```

输出结果如下：

```shell #test-result id="install-torch" fuzzy='...'
...
torch 2.9.0+cpu
torch_npu 2.9.0.post2
```

## 安装 lm-eval

本示例用 HuggingFace 后端（`hf`）加载模型，也可以换成 vLLM 等其他后端。

```shell #test id="install-lmeval"
pip install "lm_eval[hf]" "transformers<5.0"
python -c "import lm_eval; print('lm_eval', lm_eval.__version__)"
```

输出结果如下：

```shell #test-result id="install-lmeval" fuzzy='...' fuzzy='xxx'
...
lm_eval xxx
```

```{admonition} Note
:class: note
输出中的 `xxx` 表示实际安装的 lm-eval 版本号。
```

## 运行评测

安装示例需要的 ModelScope（用于下载模型与数据集）：

```shell #test-setup id="install-example-deps"
pip install "modelscope==1.37.0"
```

使用 Qwen2.5-0.5B-Instruct 评测 `arc_easy`（科学问答推理）和 `winogrande`（常识指代消歧），并打印两项任务的准确率。使用python执行下面的脚本：

```python #test id="run-eval"
import glob
import json
import os
import shutil
import subprocess
import sys

import lm_eval
from modelscope import snapshot_download

model_dir = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")
arc_repo = snapshot_download("allenai/ai2_arc", repo_type="dataset")
wg_repo = snapshot_download("allenai/winogrande", repo_type="dataset")

tasks_dir = os.path.join(os.path.dirname(lm_eval.__file__), "tasks")

# arc_easy：AI2 推理挑战 Easy 集，考查科学问答推理
arc_data = "arc_easy_data"
shutil.rmtree(arc_data, ignore_errors=True)
os.makedirs(arc_data)
for name in os.listdir(os.path.join(arc_repo, "ARC-Easy")):
    if name.endswith(".parquet"):
        shutil.copy2(os.path.join(arc_repo, "ARC-Easy", name), arc_data)
with open(os.path.join(tasks_dir, "arc", "arc_easy.yaml"), encoding="utf-8") as fh:
    arc_yaml = fh.read().replace("allenai/ai2_arc", os.path.abspath(arc_data))
arc_yaml = "\n".join(
    line for line in arc_yaml.splitlines() if "dataset_name" not in line
)
with open("arc_easy_npu.yaml", "w", encoding="utf-8") as fh:
    fh.write(arc_yaml)

# winogrande：代词消歧任务，考查常识推理
wg_data = "winogrande_xl_data"
shutil.rmtree(wg_data, ignore_errors=True)
os.makedirs(wg_data)
for name in os.listdir(os.path.join(wg_repo, "winogrande_xl")):
    if name.endswith(".parquet"):
        shutil.copy2(os.path.join(wg_repo, "winogrande_xl", name), wg_data)
with open(os.path.join(tasks_dir, "winogrande", "default.yaml"), encoding="utf-8") as fh:
    wg_yaml = fh.read().replace("allenai/winogrande", os.path.abspath(wg_data))
wg_yaml = "\n".join(
    line for line in wg_yaml.splitlines() if "dataset_name" not in line
)
shutil.copy2(
    os.path.join(tasks_dir, "winogrande", "preprocess_winogrande.py"), "."
)
with open("winogrande_npu.yaml", "w", encoding="utf-8") as fh:
    fh.write(wg_yaml)

shutil.rmtree("output/lm_eval_out", ignore_errors=True)
run = [sys.executable, "-m", "lm_eval", "run",
       "--model", "hf", "--model_args", "pretrained=" + model_dir,
       "--device", "npu:0", "--batch_size", "8", "--limit", "10"] #`--limit 10` 表示仅评测 10 条样本，用于验证流程
subprocess.run(run + ["--tasks", "arc_easy_npu.yaml",
                      "--output_path", "output/lm_eval_out/arc"], check=True)
subprocess.run(run + ["--tasks", "winogrande_npu.yaml", "--num_fewshot", "5",
                      "--output_path", "output/lm_eval_out/winogrande"], check=True)

# 从结果 JSON 读取两个任务的准确率
scores = {}
for path in glob.glob("output/lm_eval_out/**/*.json", recursive=True):
    for task, metrics in json.load(open(path)).get("results", {}).items():
        if task not in ("arc_easy", "winogrande"):
            continue
        for key in metrics:
            if key.startswith("acc") and "norm" not in key:
                value = metrics[key]
                scores[task] = value.get("value", value) if isinstance(value, dict) else value
                break

print("评测完成")
print("结果目录：output/lm_eval_out")
for task in ("arc_easy", "winogrande"):
    print(task, "acc=", round(float(scores[task]), 4))
```

输出结果如下：

```shell #test-result id="run-eval" fuzzy='...' fuzzy='xxx'
...
评测完成
结果目录：output/lm_eval_out
arc_easy acc=xxx
winogrande acc=xxx
```

```{admonition} Note
:class: note
输出中的 `xxx` 表示各任务的实际准确率。
```

## 外部链接

- 官方仓库：[EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- 官方快速开始：[Quick Start](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#quick-start)

