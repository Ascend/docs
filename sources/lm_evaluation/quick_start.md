# lm-evaluation-harness

[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)（lm-eval）是统一的大模型评测框架。本示例在单卡昇腾 NPU 上用 HuggingFace 后端跑通两个官方任务：`arc_easy` 与 `winogrande`。

## 前置条件

### 硬件

Atlas 900 A2 单卡（Ascend 910B），并按需完成物理机或容器内的设备挂载。

### 基础软件

在运行本文档示例之前，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与 CANN 匹配的 `torch` + `torch_npu`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)）

本文档示例在 Python 3.12、CANN 9.1.0、torch 2.9.0、torch_npu 2.9.0.post2 环境下验证通过。

## 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装 lm-eval

本示例用 HuggingFace 后端（`hf`）加载模型，也可以换成 vLLM 等其他后端。

```shell #test id="install-lmeval"
pip install "lm_eval[hf]" "transformers<5.0"
python -c "import lm_eval; print('lm_eval', lm_eval.__version__)"
```

输出结果如下，其中 `xxx` 表示实际版本号：

```shell #test-result id="install-lmeval" fuzzy='...' fuzzy='xxx'
...
lm_eval xxx
```

## 运行评测

安装示例需要的 ModelScope（用于下载模型与数据集）：

```shell #test-setup id="install-example-deps"
pip install "modelscope==1.37.0"
```

下面这段脚本一次完成下载、评测与结果检查。模型约 1 GB，两个数据集经 ModelScope 下载 parquet 到本地。运行下面的python脚本：

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
       "--device", "npu:0", "--batch_size", "8", "--limit", "10"]
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

输出结果如下（`xxx` 表示实际准确率）：

```shell #test-result id="run-eval" fuzzy='...' fuzzy='xxx'
...
评测完成
结果目录：output/lm_eval_out
arc_easy acc=xxx
winogrande acc=xxx
```

更多任务、批量评测与更多后端用法见 [lm-eval 官方文档](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)。

