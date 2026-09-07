# opencompass

[OpenCompass](https://github.com/open-compass/opencompass) 是开源大模型评测框架。本文用原生 Hugging Face 封装和 `torch_npu`，在单卡 910B 上对 [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) 跑 `demo_gsm8k_chat_gen`（64 条样本）。

:::{note}
阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。
:::

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**单卡**。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh` |
| Python | 3.12 |
| PyTorch | `torch==2.9.0` 与 `torch_npu==2.9.0.post2`，见下文安装 |
| OpenCompass | 从 GitHub 克隆 Release（撰写时 `0.5.4`），见下文 |
| 模型 | [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) |
| 数据 | [GSM8K](https://github.com/openai/grade-school-math) 的 `train.jsonl` / `test.jsonl` |

**配套机器**：Atlas 900 A2 PODc（Ascend 910B4）。**配套镜像**：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

本文覆盖原生 Hugging Face + `torch_npu`，不覆盖 MindIE、LMDeploy 或 vLLM-Ascend。

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。常见容器里 `npu-smi` 在 `/usr/local/sbin`，需要把该目录加入 `PATH`。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
```

`PYTHONNOUSERSITE=1` 让 Python 忽略用户目录里的包。本机如果曾经 `pip install --user` 过 CANN 相关包，不设这个变量时，pip 解析器可能被带偏。

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

```shell
npu-smi info
```

命令退出码应为 0，并打印设备列表。表格中的功耗、HBM 占用每次不同，不必与任何样例逐字一致。

:::{note}
若 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。
:::

### 2.2 确认工具可用

```shell #test id="check-tools"
test -n "$ASCEND_HOME_PATH" && command -v npu-smi >/dev/null && echo cann_ready True
```

输出结果如下：

```shell #test-result id="check-tools"
cann_ready True
```

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy="xxx"
Python 3.12.xxx
```

## 3. 安装 PyTorch NPU 栈

昇腾上的 `torch_npu` 要从华为 PyPI 额外索引安装，并钉死与 CANN 匹配的版本。`numpy` 和 `pyyaml` 也要一起装：`torch_npu` 的 wheel 没有声明这两项依赖，缺了会在 `import torch_npu` 之前就失败。

```shell #test id="install-torch"
python -m pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
  torch==2.9.0 torch_npu==2.9.0.post2 'numpy>=1.23.4,<2' pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-torch"
...
torch 2.9.0...
torch_npu 2.9.0.post2
npu_available True
```

`npu_available` 必须是 `True`。`False` 时不要继续，先查 CANN、驱动和可见设备。

## 4. 安装 OpenCompass

源码装到 `$HOME/opencompass-qs`，将 `<ref>` 换成目标 Release tag（撰写时最新为 `0.5.4`）。`--no-deps` 之后按过滤后的 `runtime.txt` 装依赖，并跳过 `torch` 与 GitHub `master` 上的 `rouge_chinese`。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="install-opencompass" load="upstream_ref>>ref"
mkdir -p "$HOME/opencompass-qs"
cd "$HOME/opencompass-qs"
if [ ! -d opencompass/.git ]; then
  GIT_TERMINAL_PROMPT=0
  for _ in 1 2 3; do
    git clone --depth 1 --branch <ref> \
      https://github.com/open-compass/opencompass.git opencompass && break
    rm -rf opencompass
    sleep 5
  done
fi
test -d opencompass/.git
cd opencompass
python -m pip install -e . --no-deps
grep -vE '^[[:space:]]*(#|$)|^torch|rouge_chinese' requirements/runtime.txt \
  > "$HOME/opencompass-qs/runtime-local.txt"
python -m pip install \
  torch==2.9.0 \
  'transformers>=4.37.0,<5' \
  'datasets>=3.0.0,<4.0.0' \
  'numpy>=1.23.4,<2.0.0' \
  'pandas>=2.0,<3' \
  huggingface_hub \
  rouge-chinese \
  -r "$HOME/opencompass-qs/runtime-local.txt"
python -c "import torch, torch_npu, opencompass; print('torch', torch.__version__); print('opencompass', opencompass.__version__); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-opencompass" load="upstream_ref>>ref"
...
torch 2.9.0...
opencompass <ref>
npu_available True
```

`npu_available` 仍必须是 `True`。若 `torch` 不再以 `2.9.0` 开头，说明后面的包把 NPU 栈换掉了，卸掉后按第 3–4 节重装。

## 5. 准备模型和数据

权重从 Hugging Face 拉取。下面打印的本地路径供第 6 节配置使用。

```shell #test-setup store="model_path"
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2-0.5B-Instruct'))" | tail -n 1
```

确认权重目录完整：

```shell #test id="check-model" load="model_path>>model_path"
test -f "<model_path>/config.json" && echo has_config True
```

输出结果如下：

```shell #test-result id="check-model"
has_config True
```

GSM8K 从 [openai/grade-school-math](https://github.com/openai/grade-school-math) 取官方 `jsonl`，放到源码树的 `data/gsm8k/`。OpenCompass 默认读这个相对路径。

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/huggingface/opencompass/gsm8k'
dest="$HOME/opencompass-qs/opencompass/data/gsm8k"
mkdir -p "$dest"
for f in test.jsonl train.jsonl; do
  case "$f" in
    test.jsonl) sum='3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14' ;;
    train.jsonl) sum='17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465' ;;
  esac
  cached="$ci/$f"
  if [ -f "$cached" ]; then
    if echo "$sum  $cached" | sha256sum -c >/dev/null 2>&1; then
      cp -a "$cached" "$dest/$f"
    else
      rm -f "$cached"
    fi
  fi
done
```
-->

```shell #test id="download-gsm8k"
mkdir -p "$HOME/opencompass-qs/opencompass/data/gsm8k"
cd "$HOME/opencompass-qs/opencompass/data/gsm8k"
if [ ! -f test.jsonl ]; then
  curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 \
    -o test.jsonl \
    https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
fi
if [ ! -f train.jsonl ]; then
  curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 \
    -o train.jsonl \
    https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl
fi
test -s test.jsonl && test -s train.jsonl && echo gsm8k_ready True
```

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/huggingface/opencompass/gsm8k'
src="$HOME/opencompass-qs/opencompass/data/gsm8k"
mkdir -p "$ci"
for f in test.jsonl train.jsonl; do
  case "$f" in
    test.jsonl) sum='3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14' ;;
    train.jsonl) sum='17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465' ;;
  esac
  echo "$sum  $src/$f" | sha256sum -c
  if [ ! -f "$ci/$f" ]; then
    cp -a "$src/$f" "$ci/$f.part"
    mv "$ci/$f.part" "$ci/$f"
  fi
done
```
-->

输出结果如下：

```shell #test-result id="download-gsm8k"
gsm8k_ready True
```

## 6. 在 NPU 上评测

下面两份文件放在源码根目录 `$HOME/opencompass-qs/opencompass`。模型子类在生成前打印设备；评测配置把生成长度压到 32。

保存为 `npu_chat.py`：

```python
from opencompass.models.huggingface_above_v4_33 import HuggingFacewithChatTemplate
from opencompass.registry import MODELS


@MODELS.register_module()
class HuggingFaceNPUChat(HuggingFacewithChatTemplate):
    def generate(self, inputs, max_out_len, **kwargs):
        print('\nopencompass_model_device', next(self.model.parameters()).device, flush=True)
        return super().generate(inputs, max_out_len, **kwargs)
```

<!--
```shell #test-setup
cat > "$HOME/opencompass-qs/opencompass/npu_chat.py" <<'PY'
from opencompass.models.huggingface_above_v4_33 import HuggingFacewithChatTemplate
from opencompass.registry import MODELS


@MODELS.register_module()
class HuggingFaceNPUChat(HuggingFacewithChatTemplate):
    def generate(self, inputs, max_out_len, **kwargs):
        print('\nopencompass_model_device', next(self.model.parameters()).device, flush=True)
        return super().generate(inputs, max_out_len, **kwargs)
PY
```
-->

`<model_path>` 换成上一节 `snapshot_download` 打印的路径。保存为 `eval_qwen2_gsm8k.py`：

```python
from mmengine.config import read_base
from npu_chat import HuggingFaceNPUChat

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets

gsm8k_datasets[0]['infer_cfg']['inferencer']['max_out_len'] = 32

datasets = gsm8k_datasets
models = [
    dict(
        type=HuggingFaceNPUChat,
        abbr='qwen2-0.5b-instruct',
        path=r'<model_path>',
        max_out_len=32,
        batch_size=4,
        model_kwargs=dict(torch_dtype='bfloat16'),
        run_cfg=dict(num_gpus=1),
    )
]
```

<!--
```shell #test-setup load="model_path>>model_path"
cat > "$HOME/opencompass-qs/opencompass/eval_qwen2_gsm8k.py" <<'PY'
from mmengine.config import read_base
from npu_chat import HuggingFaceNPUChat

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets

gsm8k_datasets[0]['infer_cfg']['inferencer']['max_out_len'] = 32

datasets = gsm8k_datasets
models = [
    dict(
        type=HuggingFaceNPUChat,
        abbr='qwen2-0.5b-instruct',
        path=r'<model_path>',
        max_out_len=32,
        batch_size=4,
        model_kwargs=dict(torch_dtype='bfloat16'),
        run_cfg=dict(num_gpus=1),
    )
]
PY
```
-->

`--debug` 让调度器在当前进程里跑，日志直接出现在终端。

```shell #test id="eval-gsm8k"
cd "$HOME/opencompass-qs/opencompass"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
opencompass eval_qwen2_gsm8k.py --debug -w "$HOME/opencompass-qs/work" 2>&1
echo '--- summary ---'
cat "$HOME/opencompass-qs/work"/*/summary/summary_*.txt
```

输出结果如下：

```shell #test-result id="eval-gsm8k"
...
opencompass_model_device npu:0
...
tabulate format
...
demo_gsm8k ... accuracy ... gen ...
...
```

`opencompass_model_device` 必须是 `npu:0`。若打印 `cpu`，这次生成没有上 NPU，退出码 0 也不能当成功。

原生 `HuggingFacewithChatTemplate` 在 `mmengine.device.is_npu_available()` 为真时会把 `device_map` 设为 `npu`。