# stable-diffusion-webui

本示例在单卡昇腾 NPU 上以无头 API 模式运行 stable-diffusion-webui，通过 REST API 完成一次文生图推理，并将生成的图片保存到本地。

## 前置条件

### 硬件

Atlas 900 A2 训练服务器，并按需完成物理机或容器内的设备挂载。

### 基础软件

在运行本文档示例之前，需要准备以下软件：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

本文档示例在 Python 3.10、CANN 9.1.0 环境下验证通过。

## 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装 stable-diffusion-webui

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并切换到最新 release：

```shell #test id="clone-repo" load="upstream_ref>>ref"
set -e
git clone --branch "<ref>" --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
release=$(git -C stable-diffusion-webui describe --tags --exact-match HEAD)
echo "stable-diffusion-webui $release"
```

```{note}
`<ref>` 表示上游最新的 release 标签
```

输出结果如下：

```shell #test-result id="clone-repo" fuzzy='xxx'
stable-diffusion-webui xxx
```

```{note}
`xxx` 表示安装的版本。
```

## 运行示例

### 安装依赖

安装 opencv 运行所需的系统库、上游依赖，以及与 CANN 匹配的 PyTorch 软件栈，并打印关键依赖版本。CLIP 从 GitHub 源码安装，ModelScope 用于下载模型：

```shell #test id="install-deps"
set -e
apt-get update -qq
apt-get install -y -qq --no-install-recommends libgl1 libglib2.0-0
pip install torch==2.9.0 torchvision==0.24.0 torch_npu==2.9.0.post6
pip install -r stable-diffusion-webui/requirements_versions.txt
pip install -r stable-diffusion-webui/requirements.txt
pip install -r stable-diffusion-webui/requirements_npu.txt
pip install modelscope wheel
pip install --no-build-isolation "https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip"
for package in torch torchvision torch-npu transformers gradio modelscope; do
    version=$(pip show "$package" | sed -n 's/^Version: //p')
    echo "$package==${version%%+*}"
done
```

输出结果如下：

```shell #test-result id="install-deps" fuzzy='...' fuzzy='xxx'
...
torch==2.9.0
torchvision==0.24.0
torch-npu==2.9.0.post6
transformers==xxx
gradio==xxx
modelscope==xxx
```

```{note}
`...` 表示省略的安装日志，`xxx` 表示依赖的版本。
```

### 生成图片

下面的代码通过 ModelScope 下载 `sd-turbo` 模型，启动 stable-diffusion-webui API，并完成一次文生图推理。用 Python 运行以下代码：

```python #test-setup
import os
import subprocess
import sys
from pathlib import Path

from modelscope import snapshot_download

repo = Path("stable-diffusion-webui").resolve()

model_dir = Path(snapshot_download("AI-ModelScope/sd-turbo"))
checkpoint = (model_dir / "sd_turbo.safetensors").resolve()
assert checkpoint.is_file()

model_path = repo / "models" / "Stable-diffusion"
model_path.mkdir(parents=True, exist_ok=True)
local_checkpoint = model_path / checkpoint.name
local_checkpoint.symlink_to(checkpoint)
config = (
    repo / "repositories" / "stable-diffusion-stability-ai"
    / "configs" / "stable-diffusion" / "v2-inference.yaml"
)
local_checkpoint.with_suffix(".yaml").symlink_to(config)

env = os.environ.copy()
env["STABLE_DIFFUSION_REPO"] = "https://github.com/w-e-w/stablediffusion.git"
(repo / "db").mkdir(exist_ok=True)

command = [
    sys.executable,
    "launch.py",
    "--nowebui",
    "--skip-torch-cuda-test",
    "--no-half",
    "--ckpt",
    str(local_checkpoint),
    "--port",
    "7861",
]

subprocess.Popen(
    command,
    cwd=repo,
    env=env,
    start_new_session=True,
)
```

服务启动需要几分钟。确认服务启动完成后，发起文生图请求并将返回的图片保存到本地，用 Python 运行以下代码：

```python #test id="txt2img"
import base64
import json
import urllib.request
from pathlib import Path

payload = json.dumps(
    {
        "prompt": "a cute cat",
        "steps": 1,
        "cfg_scale": 1.0,
        "width": 512,
        "height": 512,
    }
).encode("utf-8")

request = urllib.request.Request(
    "http://127.0.0.1:7861/sdapi/v1/txt2img",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(request, timeout=600) as response:
    result = json.load(response)

image = base64.b64decode(result["images"][0])
output = Path("/tmp/sd-turbo-out.png")
output.write_bytes(image)

print("图片生成成功")
print(f"图片保存路径：{output}")
```

输出结果如下：

```shell #test-result id="txt2img"
图片生成成功
图片保存路径：/tmp/sd-turbo-out.png
```

## 外部链接

- 官方仓库：[AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- 官方文档：[stable-diffusion-webui wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki)
