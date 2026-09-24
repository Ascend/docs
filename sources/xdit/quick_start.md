# xDiT

xDiT（PyPI 包名 `xfuser`）是一套统一的并行推理框架。本示例在单卡昇腾 NPU 上生成第一张图。

## 前置条件

### 硬件

Atlas 900 A2 训练服务器（Ascend 910B），并按需完成物理机或容器内的设备挂载。单卡生成示例需 1 张卡，序列并行示例需 2 张卡。

### 基础软件

在运行本文档示例之前，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

本文档示例在 Python 3.12、CANN 9.1.0 环境下验证通过。

## 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装 PyTorch NPU 栈

参考的版本配套如下（更多组合见 [CANN 与 PyTorch 配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)）：

| CANN | PyTorch | `torch_npu` 安装包 |
| --- | --- | --- |
| 9.1.0 | 2.9.0 | 2.9.0.post6 |
| 9.1.0 | 2.10.0 | 2.10.0.post4 |
| 9.1.0 | 2.11.0 | 2.11.0 |

本示例使用第一行的组合：

```shell #test-setup id="xdit-install-torch"
pip install torch==2.9.0 torch_npu==2.9.0.post6
```

## 安装 xDiT

安装 `xfuser`（PyPI 包名），并打印安装版本：

```shell #test id="xdit-install"
pip install xfuser
python -c "from importlib.metadata import version; print('xDiT version:', version('xfuser'))"
```

输出结果如下：

```shell #test-result id="xdit-install" fuzzy='...' fuzzy='xxx'
...
xDiT version: xxx
```

其中 `xxx` 是安装的 xDiT（`xfuser`）版本号。

## 运行示例：文生图

安装示例使用的 Triton 和模型下载所需的 ModelScope：

```shell #test-setup
pip install triton==3.5.0 "modelscope==1.37.0"
```

用 [SD3 medium](https://modelscope.cn/models/stabilityai/stable-diffusion-3-medium-diffusers) 在单卡上生成一张 256×256 的图。模型约 28 GB。

将下面的 Python 代码保存为 `sd3_npu.py`：

```python
import os
import sys
import time

import torch
import torch_npu
from modelscope import snapshot_download
from transformers import T5EncoderModel
from xfuser import xFuserArgs, xFuserStableDiffusion3Pipeline
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import get_runtime_state, get_world_group

model_path = snapshot_download('stabilityai/stable-diffusion-3-medium-diffusers')

parser = FlexibleArgumentParser(description="xFuser SD3 Arguments")
args = xFuserArgs.add_cli_args(parser).parse_args(['--model', model_path] + sys.argv[1:])
engine_args = xFuserArgs.from_cli_args(args)
engine_config, input_config = engine_args.create_config()
local_rank = get_world_group().rank

text_encoder_3 = T5EncoderModel.from_pretrained(
    model_path, subfolder="text_encoder_3", dtype=torch.float16
)
pipe = xFuserStableDiffusion3Pipeline.from_pretrained(
    pretrained_model_name_or_path=model_path,
    engine_config=engine_config,
    dtype=torch.float16,
    text_encoder_3=text_encoder_3,
).to(f"npu:{local_rank}")
pipe.prepare_run(input_config)

torch.npu.synchronize(device=local_rank)
start = time.perf_counter()
output = pipe(
    height=input_config.height,
    width=input_config.width,
    prompt=input_config.prompt,
    num_inference_steps=input_config.num_inference_steps,
    output_type=input_config.output_type,
    guidance_scale=input_config.guidance_scale,
    generator=torch.Generator(device="npu").manual_seed(input_config.seed),
)
torch.npu.synchronize(device=local_rank)
elapsed = time.perf_counter() - start

os.makedirs("results", exist_ok=True)
if pipe.is_dp_last_group():
    world_size = get_world_group().world_size
    path = f"results/sd3_npu{world_size}_ulysses{engine_args.ulysses_degree}.png"
    output.images[0].save(path)
    print(f"inference time: {elapsed:.2f} sec")
    print(f"image saved to {path}")
get_runtime_state().destroy_distributed_env()
```

用 `torchrun` 在单卡上运行：

```shell #test id="xdit-sd3-smoke"
torchrun --nproc_per_node=1 sd3_npu.py --prompt "a tiny test sketch" --height 256 --width 256 --num_inference_steps 1 --seed 42
```

输出结果如下：

```shell #test-result id="xdit-sd3-smoke" fuzzy='...' fuzzy='xxx'
...
inference time: xxx sec
image saved to results/sd3_npu1_ulysses1.png
```

### 多卡运行示例

同一个脚本、同一个模型，加 `--ulysses_degree 2` 在 2 卡上做序列并行，attention 用 SDPA 后端：

```shell #test id="xdit-sd3-2card"
torchrun --nproc_per_node=2 sd3_npu.py --prompt "a tiny test sketch" --height 256 --width 256 --num_inference_steps 1 --seed 42 --ulysses_degree 2 --attention_backend SDPA
```

输出结果如下：

```shell #test-result id="xdit-sd3-2card" fuzzy='...' fuzzy='xxx'
...
inference time: xxx sec
image saved to results/sd3_npu2_ulysses2.png
```

其中 `xxx` 为实际推理耗时，单位为秒。

## 更多用法

更多模型与多卡并行（PipeFusion / CFG 并行 / Ring 等）见 [xDiT examples](https://github.com/xdit-project/xDiT/tree/main/examples)。
