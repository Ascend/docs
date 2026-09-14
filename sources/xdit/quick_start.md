# xDiT

xDiT（PyPI 包名 `xfuser`）是一套统一的并行推理框架：同一组 `xFuser*Pipeline` API 配合 `xFuserArgs` CLI 参数系统，切换模型或并行策略只改参数。本示例在单卡昇腾 NPU 上生成第一张图，再用同一脚本展示 2 卡序列并行。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

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
| torch | 2.9.0 |
| torch_npu | 2.9.0.post2 |
| triton | 3.5.* |
| xfuser | 最新 release（PyPI） |
| modelscope | 1.37.0 |
| 模型 | [stabilityai/stable-diffusion-3-medium-diffusers](https://www.modelscope.cn/models/stabilityai/stable-diffusion-3-medium-diffusers)，约 28 GB |

```{admonition} Note
:class: note
也可用带 CANN 的昇腾镜像（如 [ascendhub cann 镜像](https://www.hiascend.com/developer/ascendhub)）跳过 CANN 安装，其余步骤相同
```

### 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

输出类似：

```
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
+===========================+====================================================+
```

```{admonition} Note
:class: note
如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动
```

加载 CANN 环境变量：

```shell
source ~/Ascend/ascend-toolkit/set_env.sh
```

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

### 安装 torch + torch_npu + triton

```shell #test-setup id="xdit-install-torch"
pip install uv
uv pip install "torch==2.9.0" "torch_npu==2.9.0.post2" "triton==3.5.*"
```

检查 NPU 运行时可用：

```shell #test id="check-npu-runtime"
python -c "import torch, torch_npu; print(f'torch={torch.__version__}'); print(f'torch_npu={torch_npu.__version__}'); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-npu-runtime" fuzzy='xxx'
torch=xxx
torch_npu=xxx
is_available: True
count: 2
```

```{admonition} Note
:class: note
如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵
```

### 安装 xDiT

安装 `xfuser`（PyPI 包名）：

```shell #test-setup id="xdit-install"
uv pip install xfuser "modelscope==1.37.0"
```

校验 import 链与 NPU 分发（`npu hccl True`）：

```shell #test id="xdit-install-verify"
python -c "
import torch
import torch_npu
from importlib.metadata import version
from xfuser.envs import get_device_name, get_torch_distributed_backend, _is_npu
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('xfuser:', version('xfuser'))
print('npu dispatch:', get_device_name(), get_torch_distributed_backend(), bool(_is_npu()))
"
```

```shell #test-result id="xdit-install-verify" fuzzy='...' fuzzy='xxx'
...
torch: 2.9.xxx
torch_npu: 2.9.xxx
xfuser: xxx
npu dispatch: npu hccl True
```

## 文生图

### 单卡生成

用 [SD3 medium](https://modelscope.cn/models/stabilityai/stable-diffusion-3-medium-diffusers) 在单卡上生成一张 256×256 的图。模型约 28 GB，首次运行时自动下载到 ModelScope 默认缓存：

```shell #test id="xdit-sd3-smoke"
cat > sd3_npu.py <<'PY'
import os
import sys
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

output = pipe(
    height=input_config.height,
    width=input_config.width,
    prompt=input_config.prompt,
    num_inference_steps=input_config.num_inference_steps,
    output_type=input_config.output_type,
    guidance_scale=input_config.guidance_scale,
    generator=torch.Generator(device="npu").manual_seed(input_config.seed),
)
os.makedirs("results", exist_ok=True)
if pipe.is_dp_last_group():
    output.images[0].save("results/sd3_npu.png")
    print("saved: results/sd3_npu.png")
get_runtime_state().destroy_distributed_env()
PY
torchrun --nproc_per_node=1 sd3_npu.py --prompt "a tiny test sketch" --height 256 --width 256 --num_inference_steps 1 --seed 42
```

```shell #test-result id="xdit-sd3-smoke"
...
saved: results/sd3_npu.png
```

### 输出校验

校验生成的图片完整有效（PNG 文件头魔数 + 大小 >50 KB 下限，防止空图 / 坏图）：

```shell #test id="xdit-sd3-output"
python - <<'PY'
import os
p = 'results/sd3_npu.png'
size = os.path.getsize(p)
assert size > 50_000, f'output too small: {size} bytes'
with open(p, 'rb') as fh:
    assert fh.read(8) == bytes([137, 80, 78, 71, 13, 10, 26, 10]), 'not a png'
print('size:', size)
PY
```

```shell #test-result id="xdit-sd3-output" fuzzy='xxx'
size: xxx
```

### 一步到多卡：序列并行

同一个脚本、同一个模型，加 `--ulysses_degree 2` 在 2 卡上做序列并行，attention 用 SDPA 后端：

```shell #test id="xdit-sd3-2card"
torchrun --nproc_per_node=2 sd3_npu.py --prompt "a tiny test sketch" --height 256 --width 256 --num_inference_steps 1 --seed 42 --ulysses_degree 2 --attention_backend SDPA
```

```shell #test-result id="xdit-sd3-2card"
...
saved: results/sd3_npu.png
```

```{admonition} Note
:class: note
如新开终端执行生成，先 `source ~/Ascend/ascend-toolkit/set_env.sh`
```

更多模型与多卡并行（PipeFusion / CFG 并行 / Ring 等）见 [xDiT examples](https://github.com/xdit-project/xDiT/tree/main/examples)。
