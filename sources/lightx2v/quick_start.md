# LightX2V

本示例在单卡昇腾 NPU 上安装 LightX2V，并用官方 Wan2.1-T2V-1.3B 模型完成一次文生视频推理。

## 前置条件

### 硬件

Atlas 900 A2 训练系列（Ascend 910B），单卡，至少 30 GB 可用存储。

### 基础软件

在运行本文档示例之前，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

本文档测试环境使用 Python 3.12、CANN 9.1.0。

## 1. 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 2. 安装 PyTorch NPU 栈

`torch`、`torchvision`、`torch_npu` 与 `triton` 四者版本严格配套，按 [CANN 与 PyTorch 配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md) 选择与 CANN 匹配的组合：

```shell #test-setup id="lightx2v-install-torch"
pip install torch==2.9.0 torchvision==0.24.0 torch_npu==2.9.0.post6 triton==3.5.0
```

确认安装的软件栈版本：

```python #test id="lightx2v-verify-torch"
import torch
import torch_npu
import torchvision
import triton

print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('torch_npu', torch_npu.__version__)
print('triton', triton.__version__)
```

输出结果如下：

```shell #test-result id="lightx2v-verify-torch"
torch 2.9.0+cpu
torchvision 0.24.0
torch_npu 2.9.0.post6
triton 3.5.0
```

## 3. 安装 LightX2V

LightX2V 未发布 PyPI 包，从 GitHub 源码安装。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="lightx2v-install-source" load="upstream_ref>>ref"
git clone --branch "<ref>" https://github.com/ModelTC/LightX2V.git
echo "LightX2V $(git -C LightX2V describe --tags --exact-match HEAD)"
pip install --no-deps ./LightX2V
```

:::{note}
`<ref>` 为上游最新 release 标签（例如 `0.5.0`）。
:::

输出结果如下：

```shell #test-result id="lightx2v-install-source" fuzzy='xxx'
LightX2V xxx
```

:::{note}
输出中的 `xxx` 为实际检出的 release 标签。
:::

## 4. 示例：生成视频

安装示例所需的依赖：

```shell #test-setup id="lightx2v-install-deps"
pip install numpy pillow einops loguru tqdm packaging safetensors regex ftfy gguf imageio imageio-ffmpeg transformers prometheus-client pydantic pyzmq "modelscope==1.37.0"
```

用 [Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) 模型跑文生视频。模型约 17.6 GB，运行下面的 Python 脚本：

```python #test id="lightx2v-wan-t2v"
import os
import time

# LightX2V 通过 PLATFORM 环境变量选择后端，必须在导入 lightx2v 之前设置。
os.environ["PLATFORM"] = "ascend_npu"

from modelscope import snapshot_download

from lightx2v import LightX2VPipeline

model_path = None
for attempt in range(3):
    try:
        model_path = snapshot_download('Wan-AI/Wan2.1-T2V-1.3B')
        break
    except Exception as exc:
        print('download attempt %d/3 failed: %s' % (attempt + 1, exc))
        time.sleep(10)
assert model_path, 'Wan2.1-T2V-1.3B download failed after 3 attempts'

pipe = LightX2VPipeline(
    model_path=model_path,
    model_cls='wan2.1',
    task='t2v',
)
pipe.create_generator(config_json='LightX2V/configs/platforms/ascend_npu/wan_t2v.json')

prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
save_result_path = "save_results/output_lightx2v_wan_t2v.mp4"

pipe.generate(
    seed=42,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
print(f'saved: {save_result_path}')
```

输出结果如下（视频保存路径）：

```shell #test-result id="lightx2v-wan-t2v"
...saved: save_results/output_lightx2v_wan_t2v.mp4
```

## 5. 更多用法

多卡并行、量化、服务化部署等更多用法见 [LightX2V examples](https://github.com/ModelTC/LightX2V/tree/main/examples)。
