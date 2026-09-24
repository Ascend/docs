# DiffSynth-Studio

[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 是 ModelScope 的扩散模型引擎，用于文生图等生成任务。本文在单卡昇腾上安装它，并用 Stable Diffusion 1.5 生成一张图。


## 前置条件

### 硬件

Atlas 800T / 900 A2 训练系列，Ascend 910B。本文示例为单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动已安装，并能 `source set_env.sh`。版本按 [昇腾软件配套清单](https://www.hiascend.com/developer/download/compatibility) 选择 |
| Python | 落在官方配套表范围内，并满足 DiffSynth-Studio 下限。当前正式版要求 3.10.1 及以上 |
| PyTorch | 安装官方当前推荐的 `torch` 与 `torch_npu`，CPU 轮子的版本号带 `+cpu`。见 [CANN 与 PyTorch 配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md) 和 [PyTorch 安装包](https://www.hiascend.com/developer/software/ai-frameworks/pytorch/download) |
| DiffSynth-Studio | 从 PyPI 安装 `diffsynth` |
| 模型 | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。

### 本文验证环境

看护镜像为 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。该镜像自带 Python 3.12。torch 使用 CPU 轮子，版本号以 `+cpu` 结尾。这组版本不是唯一支持组合。

## 1. 加载 CANN 环境

加载 CANN，并把 `/usr/local/sbin` 加入 PATH。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

```shell
npu-smi info
```

输出类似：

```text
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 5     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 5                                                            |
+===========================+===============+====================================================+
```

如果 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。

### 2.2 确认 CANN 已加载

```shell
test -n "$ASCEND_HOME_PATH"
```

命令成功时没有输出。

## 3. 安装 PyTorch NPU 栈

安装 `torch_npu`、`torchvision`、`numpy` 与 `pyyaml`，并打印 `torch`、`torch_npu` 和 `npu_available`。

```shell #test id="install-torch"
python -m pip install --retries 3 \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  torch_npu torchvision numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

完整输出较长，其中应包含：

```shell #test-result id="install-torch"
...
torch ...+cpu
torch_npu ...
npu_available True
```

`npu_available` 为 True。

## 4. 安装 DiffSynth-Studio

安装 PyPI 上的 `diffsynth`，并打印设备类型和设备名。

```shell #test id="install-diffsynth"
python -m pip install --retries 3 diffsynth
python -c "import torch, torch_npu; from importlib.metadata import version; from diffsynth.core.device.npu_compatible_device import get_device_name, get_device_type; print('diffsynth', version('diffsynth')); print('device_type', get_device_type()); print('device_name', get_device_name())"
```

<!--
```shell #test-result id="install-diffsynth"
...
diffsynth ...
device_type npu
device_name npu:0
```
-->

## 5. 在 NPU 上生成一张图

用 Python 执行以下代码。种子为 42，推理 5 步，高和宽都是 512。权重从 Hugging Face 下载。

```python #test id="generate"
import torch
import torch_npu
from diffsynth.core import ModelConfig
from diffsynth.core.device.npu_compatible_device import get_device_name
from diffsynth.pipelines.stable_diffusion import StableDiffusionPipeline

print("device_name", get_device_name())
pipe = StableDiffusionPipeline.from_pretrained(
    torch_dtype=torch.float32,
    device="npu",
    model_configs=[
        ModelConfig(
            model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
            origin_file_pattern="text_encoder/model.safetensors",
            download_source="huggingface",
        ),
        ModelConfig(
            model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
            origin_file_pattern="unet/diffusion_pytorch_model.safetensors",
            download_source="huggingface",
        ),
        ModelConfig(
            model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
            origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
            download_source="huggingface",
        ),
    ],
    tokenizer_config=ModelConfig(
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        origin_file_pattern="tokenizer/",
        download_source="huggingface",
    ),
)
print("pipe.device", pipe.device)
print("unet.device", next(pipe.unet.parameters()).device)
image = pipe(
    prompt="a photo of an astronaut riding a horse on mars, high quality, detailed",
    negative_prompt="blurry, low quality, deformed",
    cfg_scale=7.5,
    height=512,
    width=512,
    seed=42,
    rand_device="npu",
    num_inference_steps=5,
)
image.save("image.jpg")
print("image_size", image.size)
```

完整输出较长，其中应包含：

```text #test-result id="generate"
device_name npu:0
...
pipe.device npu
unet.device npu:0
image_size (512, 512)
...
```

## 6. 更多用法

本文演示的是单卡上用 Stable Diffusion 1.5 生成一张图。LoRA、视频生成和其他管线与社区文档相同，见 [DiffSynth-Studio 文档中心](https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/)。
