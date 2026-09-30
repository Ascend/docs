# Diffusers

在单卡昇腾 NPU 上跑通 [Diffusers](https://github.com/huggingface/diffusers) 文生图全链路：

- **文生图**：SD 1.5「生成 → 更换调度器」整条链路。
- **LoRA 风格加载**：SD 3.5 Large Turbo（8B MMDiT）加载 LoRA 适配器切换生成风格。
- **显存优化**：`enable_model_cpu_offload` 与全量加载的 NPU 显存峰值实测对比。

参考 [Diffusers Quicktour](https://huggingface.co/docs/diffusers/quicktour)：用 `DiffusionPipeline.from_pretrained` 实现文生图，`UniPCMultistepScheduler` 等调度器控制生成速度与质量，`load_lora_weights` 切换生成风格，`enable_model_cpu_offload` 降低显存峰值。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 2）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| transformers | `>=5.0,<6.0`（diffusers 0.40+ 依赖 huggingface-hub>=1.23，而 transformers 4.x 要求 hub<1.0，二者冲突；transformers 5.x 起适配 hub 1.x） |
| accelerate | `>=1.0,<2.0` |
| peft | `>=0.6` |
| modelscope | 1.37.0 |
| diffusers | 最新 release 的源码/二进制 |
| 模型 | [AI-ModelScope/stable-diffusion-v1-5](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5)；显存优化小节用 [stabilityai/stable-diffusion-3.5-large-turbo](https://www.modelscope.cn/models/stabilityai/stable-diffusion-3.5-large-turbo)（8B MMDiT + 三 text encoder，bf16 全量 ~29 GB）+ LoRA [prithivMLmods/SD3.5-Large-Turbo-HyperRealistic-LoRA](https://www.modelscope.cn/models/prithivMLmods/SD3.5-Large-Turbo-HyperRealistic-LoRA) |

## 前置安装

确认能看到 NPU 设备：

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
| 0     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
| 1     910B4               | OK            | 89.9        39                0    / 0             |
| 1                         | 0000:42:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
| No running processes found in NPU 1                                                            |
+===========================+===============+====================================================+
```

```{admonition} Note
:class: note
如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。
```

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

检查 torch / torch_npu 是否装好且 NPU 设备可用(下面的命令用 Python 执行)：

```python #test id="check-torch"
import torch, torch_npu
print('torch=', torch.__version__)
print('torch_npu=', torch_npu.__version__)
print('is_available:', torch.npu.is_available())
print('count:', torch.npu.device_count())
```

输出结果如下：

```shell #test-result id="check-torch"
torch= 2.9.0+cpu
torch_npu= 2.9.0.post2
is_available: True
count: 2
```

```{admonition} Note
:class: note
如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。
```

安装 `transformers` / `accelerate` / `peft` / `modelscope`：

```shell #test-setup
uv pip install 'transformers>=5.0,<6.0' 'accelerate>=1.0,<2.0' 'peft>=0.6' 'modelscope==1.37.0'
```

打印安装版本(下面的命令用 Python 执行)：

```python #test id="install-deps"
import transformers, accelerate, peft, modelscope
print('transformers', transformers.__version__)
print('accelerate', accelerate.__version__)
print('peft', peft.__version__)
print('modelscope', modelscope.__version__)
```

输出结果如下：

```shell #test-result id="install-deps" fuzzy='xxx'
transformers xxx
accelerate xxx
peft xxx
modelscope 1.37.0
```

```{admonition} Note
:class: note
`xxx` 表示实际安装的版本号。
```

## 安装 Diffusers

### 使用 uv 进行安装

通过 PyPI 镜像直接装最新 release 的二进制 wheel：

```shell #test-setup
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple diffusers
```

验证装上的版本(下面的命令用 Python 执行)：

```python #test id="diffusers-install-binary"
import diffusers
print('diffusers', diffusers.__version__)
```

输出结果类似如下：

```shell #test-result id="diffusers-install-binary" fuzzy='xxx'
diffusers xxx
```

```{admonition} Note
:class: note
xxx 表示最新的版本号。
```

<!--
```shell #test-setup
uv pip uninstall diffusers -y
```
-->

### 从源码安装

<!-- 工作流注入的 UPSTREAM_REF（最新 release tag）通过这个隐藏的 #test-setup 捕获并注入到下方 install 命令中-->
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到当前 diffusers 的最新 release tag，安装：

```shell #test-setup load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/huggingface/diffusers.git
cd diffusers
uv pip install -e .
```

```{admonition} Note
:class: note
`<ref>` 替换为 diffusers 当前的最新 release 版本。
```

验证装上的版本(下面的命令用 Python 执行)：

```python #test id="diffusers-install-source"
import diffusers
print('diffusers', diffusers.__version__)
```

输出结果类似如下：

```shell #test-result id="diffusers-install-source" fuzzy='xxx'
diffusers xxx
```

```{admonition} Note
:class: note
xxx 表示最新 release 的版本号。
```

## 使用样例

在单卡昇腾 NPU 上跑通 SD 1.5「文生图 → 更换调度器」整条链路（想调生成效果可修改 `prompt` / `num_inference_steps` / `guidance_scale`，参数含义与官方 Quicktour 完全一致）。SD 3.5 显存优化小节见下文。

### 下载基础模型

默认使用 **ModelScope** 进行模型下载，只拉 pipeline 需要的组件目录：

```shell #test-setup store="model_path"
set -o pipefail
python -c "
from modelscope import snapshot_download
print(snapshot_download(
    'AI-ModelScope/stable-diffusion-v1-5',
    allow_file_pattern=[
        '*.json', '*.txt',
        'scheduler/*', 'tokenizer/*', 'feature_extractor/*',
        'text_encoder/*', 'unet/*', 'vae/*',
    ],
    ignore_file_pattern=['*.bin', '*.fp16*', 'safety_checker/*', 'v1-5-pruned*'],
))" | grep '^/' | tail -n 1
```

输出类似：

```
/root/.cache/modelscope/hub/models/AI-ModelScope/stable-diffusion-v1-5
```

### 文生图（DiffusionPipeline）

`DiffusionPipeline.from_pretrained` 按模型仓库的 `model_index.json` 自动组装 text_encoder / unet / vae / tokenizer / scheduler，加载后传入 prompt 即可出图：

```shell #test id="text-to-image" load="model_path>>model_path"
mkdir -p output
python << 'PY' 2>&1
import torch
import torch_npu
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "<model_path>", dtype=torch.bfloat16, safety_checker=None,
).to("npu:0")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=30).images[0]
image.save("output/astronaut_rides_horse.png")
print("generated:", image.size)
PY
ls -l output/astronaut_rides_horse.png | awk '{print $5, $9}'
```

```{admonition} Note
:class: note
`<model_path>` 在运行时指向「下载基础模型」一节下载的 SD 1.5 模型缓存路径。
```

输出结果如下（`...` 吞掉加载进度等前导日志，`xxx` 为实际文件字节数）：

```shell #test-result id="text-to-image" fuzzy='xxx' fuzzy='...'
...
generated: (512, 512)
xxx output/astronaut_rides_horse.png
```

### 更换调度器（UniPCMultistepScheduler）

调度器决定逐步去噪的算法，直接影响生成速度与质量。用 `UniPCMultistepScheduler.from_config(pipe.scheduler.config)` 复用原调度器配置，即可把 pipeline 热替换为更少步数出图的调度器：

```shell #test id="swap-scheduler" load="model_path>>model_path"
mkdir -p output
python << 'PY' 2>&1
import torch
import torch_npu
from diffusers import DiffusionPipeline, UniPCMultistepScheduler

pipe = DiffusionPipeline.from_pretrained(
    "<model_path>", dtype=torch.bfloat16, safety_checker=None,
).to("npu:0")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=20).images[0]
image.save("output/astronaut_unipc.png")
print("generated:", image.size, "via", type(pipe.scheduler).__name__)
PY
ls -l output/astronaut_unipc.png | awk '{print $5, $9}'
```

```{admonition} Note
:class: note
`<model_path>` 在运行时指向「下载基础模型」一节下载的 SD 1.5 模型缓存路径。
```

输出结果如下（`xxx` 为实际文件字节数）：

```shell #test-result id="swap-scheduler" fuzzy='xxx' fuzzy='...'
...
generated: (512, 512) via UniPCMultistepScheduler
xxx output/astronaut_unipc.png
```

## 大模型显存优化（SD 3.5 Large Turbo）

上面 SD1.5 的所有组件加起来 ~2 GB，单卡显存放得下，不需要任何优化。但现代扩散模型在 32 GB 显存的卡上全量加载已经放不下，`enable_model_cpu_offload()` 让组件**逐个上下场**——text encoder 编码完搬回 CPU 内存，再请 MMDiT 上 NPU 去噪，最后 VAE 解码——显存峰值从「所有组件之和」降到「最大单个组件」。

本节用 SD 3.5 Large Turbo 实测两种加载方式的 NPU 显存峰值对比。

### 下载 SD 3.5 Large Turbo

只下载 diffusers 布局需要的文件。

```shell #test-setup store="sd35_path"
set -o pipefail
python -c "
from modelscope import snapshot_download
print(snapshot_download(
    'stabilityai/stable-diffusion-3.5-large-turbo',
    allow_file_pattern=[
        '*.json', '*.txt', '*.model', '*.tiktoken',
        'transformer/*', 'vae/*',
        'text_encoder/*', 'text_encoder_2/*', 'text_encoder_3/*',
        'tokenizer/*', 'tokenizer_2/*', 'tokenizer_3/*',
    ],
    ignore_file_pattern=['*.fp16*', '*.png', '*.jpg', '*.webp'],
))" | grep '^/' | tail -n 1
```

输出类似：

```
/root/.cache/modelscope/hub/models/stabilityai/stable-diffusion-3.5-large-turbo
```

### 加载 LoRA

LoRA 适配器只往基础模型插入少量可训练参数，推理时把 LoRA 权重加载/合并进 transformer 即可切换生成风格。对应 Quicktour 的 LoRA 一节：使用 `load_lora_weights` 加载适配器，prompt 里带上触发词激活风格。

先下载 LoRA 权重（同样走 ModelScope，落入持久缓存）：

```shell #test-setup store="lora_path"
set -o pipefail
python -c "from modelscope import snapshot_download; print(snapshot_download('prithivMLmods/SD3.5-Large-Turbo-HyperRealistic-LoRA'))" | grep '^/' | tail -n 1
```

输出类似：

```
/root/.cache/modelscope/hub/models/prithivMLmods/SD3.5-Large-Turbo-HyperRealistic-LoRA
```

在 SD 3.5 pipeline 上加载 LoRA 并用触发词生成：

```shell #test id="sd35-lora" load="sd35_path>>sd35_path" load="lora_path>>lora_path"
mkdir -p output
python << 'PY' 2>&1
import torch
import torch_npu
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "<sd35_path>", dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.load_lora_weights(
    "<lora_path>", weight_name="SD3.5-4Step-Large-Turbo-HyperRealistic-LoRA.safetensors",
)

image = pipe(
    "hyper realistic close-up photo of a black falcon twist in the air",
    num_inference_steps=4,
    guidance_scale=0.0,
).images[0]
image.save("output/sd35_lora.png")
pipe.unload_lora_weights()
print("generated:", image.size, "lora loaded & unloaded")
PY
ls -l output/sd35_lora.png | awk '{print $5, $9}'
```

```{admonition} Note
:class: note
`<sd35_path>` / `<lora_path>` 在运行时分别指向「下载 SD 3.5 Large Turbo」「加载 LoRA」两节下载的基础模型与 LoRA 缓存路径。
```

输出结果类似（`xxx` 为实际文件字节数）：

```shell #test-result id="sd35-lora" fuzzy='xxx' fuzzy='...'
...
generated: (1024, 1024) lora loaded & unloaded
xxx output/sd35_lora.png
```

### cpu offload 模式生成

```shell #test id="sd35-offload" load="sd35_path>>sd35_path"
mkdir -p output
python << 'PY' 2>&1
import torch
import torch_npu
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "<sd35_path>", dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

image = pipe(
    "a black falcon twist in the air, close-up photo",
    num_inference_steps=4,
    guidance_scale=0.0,
).images[0]
image.save("output/sd35_offload.png")
torch.npu.synchronize()
peak = torch.npu.max_memory_allocated() / 1024**3
print("generated:", image.size)
print(f"offload peak NPU memory: {peak:.2f} GB")
PY
ls -l output/sd35_offload.png | awk '{print $5, $9}'
```

```{admonition} Note
:class: note
`<sd35_path>` 在运行时指向「下载 SD 3.5 Large Turbo」一节下载的模型缓存路径。
```

输出结果类似：

```shell #test-result id="sd35-offload" fuzzy='xxx' fuzzy='...'
...
generated: (1024, 1024)
offload peak NPU memory: xxx GB
xxx output/sd35_offload.png
```

### 全量加载对比

同一模型不带 offload 直接 `.to("npu:0")` 全量驻留显存：

```shell #test id="sd35-full" load="sd35_path>>sd35_path"
mkdir -p output
python << 'PY' 2>&1
import torch
import torch_npu
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "<sd35_path>", dtype=torch.bfloat16,
).to("npu:0")

image = pipe(
    "a black falcon twist in the air, close-up photo",
    num_inference_steps=4,
    guidance_scale=0.0,
    height=768,
    width=768,
).images[0]
image.save("output/sd35_full.png")
torch.npu.synchronize()
peak = torch.npu.max_memory_allocated() / 1024**3
print("generated:", image.size)
print(f"full-load peak NPU memory: {peak:.2f} GB")
PY
ls -l output/sd35_full.png | awk '{print $5, $9}'
```

```{admonition} Note
:class: note
`<sd35_path>` 在运行时指向「下载 SD 3.5 Large Turbo」一节下载的模型缓存路径。
```

输出结果类似：

```shell #test-result id="sd35-full" fuzzy='xxx' fuzzy='...'
...
generated: (768, 768)
full-load peak NPU memory: xxx GB
xxx output/sd35_full.png
```

## 外部链接

- GitHub：[huggingface/diffusers](https://github.com/huggingface/diffusers)
- 文档中心：[Diffusers Docs](https://huggingface.co/docs/diffusers)
