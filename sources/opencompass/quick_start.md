# opencompass

[OpenCompass](https://github.com/open-compass/opencompass) 是开源大模型评测框架。本文在单卡 910B 上对 [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) 跑 `demo_gsm8k_chat_gen`（64 条样本）。

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文覆盖单卡。

### 软件


| 类别          | 要求                                                                                                                                                                                                                 |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CANN        | toolkit 与驱动已安装，并能 `source set_env.sh`。版本按 [昇腾软件配套清单](https://www.hiascend.com/developer/download/compatibility) 选择                                                                                                 |
| Python      | 落在官方配套表给出的范围内，并满足 OpenCompass 下限；上游要求 `>=3.8`                                                                                                                                                                      |
| PyTorch     | 安装官方当前推荐的 `torch` 与 `torch_npu`，见 [CANN 与 PyTorch 配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md) 和 [PyTorch 安装包](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download) |
| OpenCompass | 从 GitHub 克隆 Release tag，用 `--no-build-isolation` 对着已装的 torch 做可编辑安装；不要加 `[vllm]` 或 `[lmdeploy]`                                                                                                                    |
| 模型          | [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)                                                                                                                                        |
| 数据          | [GSM8K](https://github.com/openai/grade-school-math) 的 `train.jsonl` / `test.jsonl`                                                                                                                                |


本文验证环境：Atlas 900 A2 PODc、Ascend 910B4、Python 3.12、镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。常见容器里 `npu-smi` 在 `/usr/local/sbin`，需要把该目录加入 `PATH`。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
```



## 2. 检查环境

```shell
npu-smi info
```

下面确认 CANN 已加载，并且 `npu-smi` 在 `PATH` 里。

```shell #test id="check-tools"
test -n "$ASCEND_HOME_PATH" && command -v npu-smi >/dev/null && echo cann_ready True
```



## 3. 安装 PyTorch NPU

先装官方当前推荐的 `torch_npu`。主索引走 CPU 轮子源，extra 走 PyPI。`numpy` 和 `pyyaml` 也要一起装，缺了会在 `import torch_npu` 时失败。

```shell #test id="install-torch"
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  torch_npu numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```



`npu_available` 必须是 `True`。`False` 时不要继续，先查 CANN、驱动和可见设备。

## 4. 安装 OpenCompass

源码装到 `$HOME/opencompass-qs`，将 `<ref>` 换成目标 Release tag。NPU 栈就绪后，用 `--no-build-isolation` 对着已装的 torch 做可编辑安装。



```shell #test id="install-opencompass" load="upstream_ref>>ref"
mkdir -p "$HOME/opencompass-qs"
cd "$HOME/opencompass-qs"
if [ ! -d opencompass/.git ]; then
  git clone --depth 1 --branch <ref> \
    https://github.com/open-compass/opencompass.git opencompass
fi
cd opencompass
python -m pip install --no-build-isolation -e .
python -c "import torch, torch_npu, opencompass; print('npu_available', torch.npu.is_available()); print('opencompass', opencompass.__version__)"
```



## 5. 准备模型和数据

权重从 Hugging Face 拉取。下面打印的本地路径供第 6 节配置使用。

```shell
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2-0.5B-Instruct'))"
```



```shell #test id="check-model" load="model_path>>model_path"
test -f "<model_path>/config.json" && echo has_config True
```



GSM8K 从 [openai/grade-school-math](https://github.com/openai/grade-school-math) 取官方 `jsonl`，放到源码树的 `data/gsm8k/`。OpenCompass 默认读这个相对路径。



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



## 6. 在 NPU 上评测

下面两份文件写到源码根目录 `$HOME/opencompass-qs/opencompass`。模型子类在生成前打印设备并固定种子；评测配置把生成长度压到 32，并用贪心解码。


| 项                 | 含义             |
| ----------------- | -------------- |
| `set_seed(42)`    | 固定随机种子，便于对照准确率 |
| `do_sample=False` | 贪心解码，不采样       |
| `max_out_len=32`  | 把单次生成长度压到 32   |


保存为 `npu_chat.py`：

```python
from transformers import set_seed
from opencompass.models.huggingface_above_v4_33 import HuggingFacewithChatTemplate
from opencompass.registry import MODELS


@MODELS.register_module()
class HuggingFaceNPUChat(HuggingFacewithChatTemplate):
    def generate(self, inputs, max_out_len, **kwargs):
        set_seed(42)
        print('\nopencompass_model_device', next(self.model.parameters()).device, flush=True)
        return super().generate(inputs, max_out_len, **kwargs)
```



将 `<model_path>` 换成上一节 `snapshot_download` 打印的路径。保存为 `eval_qwen2_gsm8k.py`：

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
        generation_kwargs=dict(do_sample=False),
        run_cfg=dict(num_gpus=1),
    )
]
```



`--debug` 让调度器在当前进程里跑，日志直接出现在终端。

```shell #test id="eval-gsm8k"
cd "$HOME/opencompass-qs/opencompass"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
opencompass eval_qwen2_gsm8k.py --debug -w "$HOME/opencompass-qs/work" 2>&1
echo '--- summary ---'
cat "$HOME/opencompass-qs/work"/*/summary/summary_*.txt
```

完整输出较长，其中应包含：

```text #test-result id="eval-gsm8k"
...
opencompass_model_device npu:0
...
tabulate format
...
| dataset | version | metric | mode | qwen2-0.5b-instruct |
...
| demo_gsm8k | ... | accuracy | gen | 3.12 |
...
demo_gsm8k: {'accuracy': 3.125}
...
```

## 相关链接

- 上游仓库：[open-compass/opencompass](https://github.com/open-compass/opencompass)
- 官方文档：[OpenCompass 文档](https://opencompass.readthedocs.io/zh_CN/latest/)
- 安装说明：[Installation](https://opencompass.readthedocs.io/zh_CN/latest/get_started/installation.html)
- 上游评测入门：[Quick Start](https://opencompass.readthedocs.io/zh_CN/latest/get_started/quick_start.html)
- 本文所用模型：[Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
- GSM8K 官方数据：[openai/grade-school-math](https://github.com/openai/grade-school-math)

