# DiffSynth-Studio

在单卡昇腾上安装 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)，用 Stable Diffusion 1.5 生成一张图。


## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装，并可 `source set_env.sh` |
| Python | 3.12 |
| PyTorch | `torch==2.9.0` 与 `torch_npu==2.9.0.post2`，见下文安装 |
| DiffSynth-Studio | 从 PyPI 安装 `diffsynth`，见下文 |
| 模型 | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) |

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。推荐配套镜像：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

## 1. 加载 CANN 环境

常见容器里 `npu-smi` 在 `/usr/local/sbin`，需要把该目录加入 `PATH`。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

```shell
npu-smi info
```

如果 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。


### 2.2 确认工具可用

下面确认 CANN 已加载，并且 `npu-smi` 与 `python` 都在 `PATH` 里。

```shell #test id="check-tools"
test -n "$ASCEND_HOME_PATH"
command -v npu-smi >/dev/null
python --version
```

输出结果如下：

```shell #test-result id="check-tools"
Python 3.12...
```

## 3. 安装 PyTorch NPU 栈

从昇腾 PyPI 源安装与 CANN 9.1 配对的 `torch` / `torch_npu`，并确认 NPU 运行时可用。

```shell #test id="install-torch"
python -m pip install --retries 3 \
  --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
  torch==2.9.0 torch_npu==2.9.0.post2 torchvision numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-torch"
...
torch 2.9.0...
torch_npu 2.9.0.post2
npu_available True
```

`npu_available` 必须是 `True`。为 `False` 时不要继续，先查 CANN、驱动和可见设备。

## 4. 安装 DiffSynth-Studio

不要运行 `pip install -e ".[npu_aarch64]"`。那个 extra 会把刚装好的 `torch 2.9.0` 降回 `2.7.1`。先装好上一节的 NPU 栈，再装不带 extra 的 `diffsynth`。

```shell #test id="install-diffsynth"
python -m pip install --retries 3 diffsynth
python -c "import torch, torch_npu; from importlib.metadata import version; from diffsynth.core.device.npu_compatible_device import get_device_name, get_device_type; print('diffsynth', version('diffsynth')); print('device_type', get_device_type()); print('device_name', get_device_name())"
```

输出结果如下：

```shell #test-result id="install-diffsynth"
...
diffsynth ...
device_type npu
device_name npu:0
```

`device_name` 为 `npu:0` 只说明设备探测成功。若这里打印 `device_type cpu` 或 `cuda`，先回到第 3 节确认 `torch_npu` 和可见设备。

## 5. 在 NPU 上生成一张图

`num_inference_steps=5` 是第一次跑通的步数。50 步出图更清楚，但第一次验证安装不必等那么久。正式出图时把步数改回 `50`。权重从 Hugging Face 下载，脚本里把 `download_source` 设为 `huggingface`。

保存为 `generate_sd15.py`：

```python
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

<!--
```shell #test-setup
cat > generate_sd15.py <<'PY'
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
PY
```
-->

```shell #test id="generate"
python generate_sd15.py
```

输出结果如下：

```shell #test-result id="generate"
device_name npu:0
...
pipe.device npu
unet.device npu:0
image_size (512, 512)
...
```

`pipe.device` 打印的是传入的字符串 `npu`。`unet.device` 才是参数真正所在的设备。
