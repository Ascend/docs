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
git clone --branch "<ref>" --depth 1 https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
cd stable-diffusion-webui
echo "Release $(git describe --tags --exact-match HEAD)"
```

`<ref>` 为上游最新 release 标签（例如 `v1.10.1`）。

输出结果如下，其中 `xxx` 表示实际的 release 标签：

```shell #test-result id="clone-repo" fuzzy='xxx'
Release xxx
```

## 运行示例

### 安装依赖

安装 opencv 运行所需的系统库、上游依赖，以及与 CANN 匹配的 PyTorch 软件栈。CLIP 从 GitHub 源码安装，ModelScope 用于下载模型：

```shell #test-setup
apt-get update -qq
apt-get install -y -qq --no-install-recommends libgl1 libglib2.0-0
cd stable-diffusion-webui
pip install torch==2.9.0 torchvision==0.24.0 torch_npu==2.9.0.post6
pip install -r requirements_versions.txt
pip install -r requirements.txt
pip install -r requirements_npu.txt
pip install modelscope wheel
pip install --no-build-isolation "https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip"
```

### 生成图片

下面的代码通过 ModelScope 下载 `sd-turbo` 模型，注入 NPU autocast 补丁，启动 stable-diffusion-webui API，并完成一次文生图推理。

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

# stable-diffusion-webui 的 autocast 判断默认未包含 NPU，需要补充 NPU 分支。
devices = repo / "modules" / "devices.py"
text = devices.read_text(encoding="utf-8")
old = "if has_xpu() or has_mps() or cuda_no_autocast():"
new = (
    "if npu_specific.has_npu or has_xpu() or has_mps() or cuda_no_autocast():"
)
assert old in text
devices.write_text(text.replace(old, new), encoding="utf-8")

env = os.environ.copy()
env["STABLE_DIFFUSION_REPO"] = "https://github.com/w-e-w/stablediffusion.git"
(repo / "db").mkdir(exist_ok=True)

command = [
    sys.executable,
    "launch.py",
    "--nowebui",
    "--skip-torch-cuda-test",
    "--ckpt",
    str(checkpoint),
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

服务启动需要几分钟。确认服务启动完成后，发起一次文生图请求，并将返回的图片保存到本地：

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

更多用法请参考 [stable-diffusion-webui wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki)。
