# LightX2V

欢迎使用 LightX2V！本指南帮助你在单卡昇腾 NPU 上装好 LightX2V 并生成第一条视频。

## 🚀 系统要求

- **硬件**：Atlas 900 A2 / A3 训练系列产品（Ascend 910B4 / 910B 等），单卡 32 GB HBM 可跑本文全部内容
- **操作系统**：Linux（Ubuntu 22.04）
- **存储**：至少 30 GB 可用空间

| 组件 | 版本 | 来源 |
| --- | --- | --- |
| CANN | ≥ 8.5.1 | 下方环境搭建 |
| Python | ≥ 3.10 | 自备环境 |
| torch / torch_npu | 2.9.0 / 2.9.0.post2 | 下方安装 |
| torchvision | 0.24.* | 下方安装 |
| triton | 3.5.* | 下方安装 |
| lightx2v | 最新 release（[GitHub Releases](https://github.com/ModelTC/LightX2V/releases)） | 下方源码安装 |
| 模型 | [Wan-AI/Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)，约 17.6 GB | 首次运行自动下载 |

**配套机器**：Atlas 900 A2 PODc（Ascend 910B4，32 GB × 1），Ubuntu 22.04。

**配套镜像**：swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

> 也可直接用带 CANN 的昇腾镜像（[ascendhub](https://www.hiascend.com/developer/ascendhub)）跳过 CANN 安装，其余步骤相同。

## 🐧 环境搭建

**安装 CANN。** 版本需与驱动配套，[快速安装脚本](https://ascend.github.io/docs/sources/ascend/quick_install.html)会自动识别卡型。安装完成后加载环境变量：

```shell
source ~/Ascend/ascend-toolkit/set_env.sh
```

**准备 Python 环境。** Python ≥ 3.10。

**安装 torch 栈。** torch、torch_npu、torchvision、triton 四者版本严格配套，跟随 torch 官方线：

```shell #test-setup id="lightx2v-install-torch"
pip install uv
uv pip install "torch==2.9.0" "torchvision==0.24.*" "torch_npu==2.9.0.post2" "triton==3.5.*"
```

## 📦 安装 LightX2V

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

**克隆项目。** 检出最新 release tag，`<ref>` 填上表 lightx2v 行的版本号。后续步骤都在当前目录下执行：

```shell #test-setup id="lightx2v-install-source" load="upstream_ref>>ref"
git clone --branch <ref> https://github.com/ModelTC/LightX2V.git
```

**安装代码与依赖。** LightX2V 未发布 PyPI 包，从克隆目录源码安装。依赖清单里含只在 x86_64 提供预编译包的项，aarch64 上整包解析装不上，先跳过依赖解析装代码，再按 NPU 推理实际用到的清单补齐：

```shell #test-setup id="lightx2v-install-deps"
uv pip install --no-deps ./LightX2V

uv pip install numpy pillow einops loguru tqdm packaging safetensors regex ftfy gguf imageio imageio-ffmpeg transformers prometheus-client pydantic pyzmq "modelscope==1.37.0"
```

**验证安装。** 打印 import 链、版本号与 NPU 平台分发结果。LightX2V 用 `PLATFORM` 环境变量选后端，NPU 上一律加 `PLATFORM=ascend_npu` 前缀：

```shell #test id="lightx2v-install-verify"
PLATFORM=ascend_npu python -c "
import torch, torch_npu, lightx2v
from lightx2v_platform.base.global_var import AI_DEVICE, PLATFORM
print('torch:', torch.__version__)
print('torch_npu:', torch_npu.__version__)
print('lightx2v:', lightx2v.__version__)
print('platform:', PLATFORM, AI_DEVICE)
print('npu available:', torch.npu.is_available(), 'count:', torch.npu.device_count())
"
```

输出结果如下：

```shell #test-result id="lightx2v-install-verify" fuzzy='xxx' fuzzy='...'
...torch: 2.9.xxx
torch_npu: 2.9.xxx
lightx2v: xxx
platform: ascend_npu npu
npu available: True count: xxx
```

## 🎯 生成视频

用官方 [Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) 跑文生视频。模型约 17.6 GB，首次运行由脚本内的 `snapshot_download` 自动下载到默认缓存，无需手动下载。推理参数取仓库自带的 NPU 配置 `configs/platforms/ascend_npu/wan_t2v.json`：

```shell #test id="lightx2v-wan-t2v"
PLATFORM=ascend_npu python - <<'PY'
import time
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

pipe.generate(
    seed=42,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path='save_results/output_lightx2v_wan_t2v.mp4',
)
print('saved: save_results/output_lightx2v_wan_t2v.mp4')
PY
```

输出结果如下：

```shell #test-result id="lightx2v-wan-t2v" fuzzy='...'
...saved: save_results/output_lightx2v_wan_t2v.mp4
```

**校验输出。** 检查 MP4 文件头 `ftyp`、封装索引 `moov` 与文件大小下限，防止空视频和写到一半的坏文件：

```shell #test id="lightx2v-output"
python - <<'PY'
p = 'save_results/output_lightx2v_wan_t2v.mp4'
data = open(p, 'rb').read()
size = len(data)
assert size > 100_000, f'output too small: {size} bytes'
assert data[4:8] == b'ftyp', 'not an mp4'
assert b'moov' in data, 'truncated mp4: no moov box'
print('size:', size)
PY
```

输出结果如下：

```shell #test-result id="lightx2v-output" fuzzy='xxx'
size: xxx
```

> **注意**：如新开终端执行生成，先 `source ~/Ascend/ascend-toolkit/set_env.sh`。多卡并行、量化、服务化部署等更多用法见 [LightX2V examples](https://github.com/ModelTC/LightX2V/tree/main/examples)。
