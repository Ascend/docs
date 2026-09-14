# lm-evaluation-harness

[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)（lm-eval）是统一的大模型评测框架，同一个 CLI 能跑数百个 benchmark 任务并接入多种模型后端。本示例在单卡昇腾 NPU 上安装 lm-eval，用 `--model hf`（基于 transformers 的模型加载器，模型与数据均在本地）跑通两个官方任务：`arc_easy`（AI2 推理挑战 Easy 集，0-shot）与 `winogrande`（代词消歧，5-shot）。模型与两个任务的数据集均经 ModelScope 自动下载，最后校验两个任务的准确率都落在 [0,1]。

## 前置条件

### 硬件
Atlas 900 A2 单卡（Ascend NPU），并按需完成物理机或容器内的设备挂载。

### 基础软件
在跑本文档之前，你的机器上需要已经装好并可用：
- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与 CANN 匹配的 `torch` + `torch_npu`，且 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

按上游 README 的方式设置 CANN 环境变量：
```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 本文档示例使用的版本
**配套机器**：Atlas 900 A2（Ascend 910B4，64 GB × 1）。**操作系统**：Ubuntu 22.04。**软件版本**：

| 组件 | 版本 |
| -- | -- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| lm-eval | 0.4.13（PyPI 安装） |
| transformers | <5.0 |
| modelscope | 1.37.0 |
| 模型 | [Qwen/Qwen2.5-0.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct)，约 1 GB |
| 任务 | `arc_easy`、`winogrande` |

## 环境检查
检查 Python 版本：
```shell #test id="check-py"
python --version
```
输出结果如下：
```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```
检查 torch / torch_npu 是否装好且 NPU 设备可用：
```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```
输出结果如下：
```shell #test-result id="check-torch" fuzzy='xxx'
torch=xxx
torch_npu=xxx
is_available: True
count: 1
```
```{admonition} Note
:class: note
如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵
```

## 安装 lm-eval
安装 `lm_eval[hf]`（HuggingFace 模型后端）、`transformers<5.0` 与 `modelscope`，装完打印版本验证：
```shell #test id="install-lmeval"
uv pip install "lm_eval[hf]" "transformers<5.0" "modelscope==1.37.0"
python -c "import lm_eval, transformers, modelscope; print('lm_eval', lm_eval.__version__); print('transformers', transformers.__version__); print('modelscope', modelscope.__version__)"
```
输出结果如下：
```shell #test-result id="install-lmeval" fuzzy='xxx'
lm_eval xxx
transformers xxx
modelscope 1.37.0
```

## 运行评测
下面这段脚本一次完成下载与评测。模型约 1 GB，两个数据集经 ModelScope 下载 parquet 到本地，任务定义复制官方 YAML 并只改 `dataset_path` 为本地目录。`--limit 10` 是跑通口径，不代表真实榜单值。

```shell #test id="run-eval"
python << 'PY'
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

# arc_easy: parquet mirror, pkg name not set since no capture
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

# winogrande: parquet mirror
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
print("LM_EVAL_DONE")
PY
```
输出结果如下（评测日志较长，此处仅校验末尾标记）：
```shell #test-result id="run-eval"
...
LM_EVAL_DONE
```

## 检查评测结果
检查结果 JSON 中两个任务的准确率都在 [0,1] 区间并打印：
```shell #test id="check-acc"
python << 'PY'
import glob
import json

found = {}
for path in glob.glob("output/lm_eval_out/**/*.json", recursive=True):
    for task, metrics in json.load(open(path)).get("results", {}).items():
        if task in ("arc_easy", "winogrande"):
            # v0.4.x stores metrics as "acc,none" / "acc_norm,none"
            for k in metrics:
                if k.startswith("acc") and "norm" not in k:
                    value = metrics[k]
                    found[task] = value.get("value", value) if isinstance(value, dict) else value
                    break
for task in ("arc_easy", "winogrande"):
    assert task in found
    assert 0.0 <= float(found[task]) <= 1.0
    print(task, "acc=", round(float(found[task]), 4))
PY
```
输出结果如下：
```shell #test-result id="check-acc" fuzzy='xxx'
arc_easy acc=xxx
winogrande acc=xxx
```
更多任务、批量评测与更多后端用法见 [lm-eval 官方文档](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md)。

