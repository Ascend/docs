# 快速开始 (Ascend NPU)

在单卡昇腾 NPU 上部署 ComfyUI，并用上游自带的 API 例程 `script_examples/basic_api_example.py` 通过 `/prompt` 接口提交默认文生图工作流，端到端产出一张 PNG。

> ComfyUI 原生支持昇腾 NPU：检测到 `torch_npu` 且 `torch.npu.is_available()` 为真时，设备管理自动选用 `npu` 设备，无需额外启动参数。本文以无头服务（默认端口 8188）+ API 例程的方式运行，全程无需人工交互；交互式 Web UI 用户浏览器访问 `http://<机器IP>:8188` 即可。

## 前置条件

### 硬件

Atlas 900 A2 单卡（Ascend NPU），并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import`、`torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

按上游 README 的方式设置 CANN 环境变量：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 单卡
- **操作系统**：Ubuntu 22.04

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| ComfyUI | master（原生 NPU 支持） |
| huggingface_hub | 最新稳定版 |
| 模型 | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) 的 `v1-5-pruned-emaonly.safetensors`（经 Hugging Face 下载） |

## 环境检查

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

检查 torch / torch_npu 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-torch"
torch= 2.9.0+cpu
torch_npu= 2.9.0.post2
is_available: True
count: 1
```

```{admonition} Note
:class: note
如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。
```

## 获取代码

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到流水线注入的最新 release tag：

```shell #test id="clone-repo" load="upstream_ref>>ref"
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
git checkout <ref>
echo "HEAD $(git log -1 --format=%h)"
```

输出结果如下：

```shell #test-result id="clone-repo" fuzzy='xxx'
HEAD xxx
```

\<ref> 为流水线注入的上游最新 release tag。

## 安装依赖

ComfyUI 的 `requirements.txt` 里 `torchvision` / `torchaudio` 未 pin 版本。为避免 pip 解析器为满足「最新 torchvision/torchaudio」而把已装好的 `torch 2.9.0` 升级、破坏 torch ↔ torch_npu ↔ CANN 配套，先显式安装与 torch 2.9.0 配套的 pair（`torchvision 0.24.x` ↔ `torchaudio 2.9.x`），已装则跳过；再装 `requirements.txt`（已满足的 `torch` 不会被改动）与 `huggingface_hub`：

```shell #test-setup
cd ComfyUI
python -m pip install "torchvision==0.24.0" "torchaudio==2.9.0"
python -m pip install -r requirements.txt
python -m pip install huggingface_hub
```

```{admonition} Note
:class: note
- `requirements.txt` 按上游原样安装；`huggingface_hub` 为 Hugging Face 下载模型所需，需显式安装。
- torch / torch_npu 栈沿用机器上已装好的版本（前置条件），本文档不重装。
```

验证依赖可用：

```shell #test id="install-deps"
python -c "import torchsde, einops, transformers, tokenizers, safetensors, aiohttp, PIL, av, huggingface_hub; print('deps ok')"
```

输出结果如下：

```shell #test-result id="install-deps"
deps ok
```

## 下载模型

默认使用 **Hugging Face Hub** 进行模型下载。

```shell #test-setup store="model_dir"
python -c "
import time, sys
from pathlib import Path
from huggingface_hub import snapshot_download
d = None
for i in range(3):
    try:
        d = snapshot_download('stable-diffusion-v1-5/stable-diffusion-v1-5', allow_patterns=['v1-5-pruned-emaonly.safetensors'])
        if (Path(d) / 'v1-5-pruned-emaonly.safetensors').is_file():
            break
        print('attempt %d/3: file not complete, retrying' % (i+1), file=sys.stderr)
        d = None
    except Exception as e:
        print('attempt %d/3 failed: %s' % (i+1, e), file=sys.stderr)
        time.sleep(15)
assert d, 'snapshot_download failed after 3 attempts'
print(d)
" | tail -n 1
```

```{admonition} Note
:class: note
- `tail -n 1` 过滤下载进度输出，仅保留模型目录路径。
- `allow_patterns` 只下载 `v1-5-pruned-emaonly.safetensors`（约 4.3 GB，SD 1.5 单文件 checkpoint，即 `basic_api_example.py` 默认工作流引用的 `ckpt_name`），跳过仓库里的 diffusers 分片格式，首次运行请耐心等待。
```

## 启动服务（单卡 NPU）

把 checkpoint 链接进 `models/checkpoints/`（`CheckpointLoaderSimple` 的 `ckpt_name` 按文件名索引），后台启动 ComfyUI 服务：

```shell #test-setup store="server_pid" load="model_dir>>model"
cd ComfyUI
mkdir -p models/checkpoints
ln -sf <model>/v1-5-pruned-emaonly.safetensors models/checkpoints/
nohup python main.py --disable-auto-launch --listen 127.0.0.1 --port 8188 > /tmp/comfyui.log 2>&1 &
echo $!
```

- `--disable-auto-launch`：无头容器内不要尝试拉起浏览器。
- 设备选择由 `comfy/model_management.py` 自动完成（检测到 `torch_npu` 即用 `npu` 设备），无需额外参数。

等待服务就绪：

```shell #test id="wait-ready"
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8188/system_stats > /dev/null && break
  sleep 5
done
curl -sf http://127.0.0.1:8188/system_stats > /dev/null || { echo "server not ready"; tail -n 100 /tmp/comfyui.log; exit 1; }
echo "server ready"
```

输出结果如下：

```shell #test-result id="wait-ready"
server ready
```

确认 ComfyUI 的设备管理选中的是 NPU：

```shell #test id="check-device"
cd ComfyUI
python -c "import comfy.model_management as mm; print('comfy device:', mm.get_torch_device())"
```

输出结果如下：

```shell #test-result id="check-device"
comfy device: npu...
```

```{admonition} Note
:class: note
`get_torch_device()` 与服务进程走同一套检测逻辑，单卡上返回 `npu:0`。
```

## 运行 API 例程（文生图）

提交上游例程 `script_examples/basic_api_example.py`（默认文生图工作流：SD 1.5，512x512，20 步 euler），轮询 `/queue` 等待出图完成：

```shell #test id="run-example"
cd ComfyUI
python script_examples/basic_api_example.py && echo "prompt queued"
python - <<'EOF'
import json, time, sys
from urllib import request

deadline = time.time() + 900
while time.time() < deadline:
    try:
        with request.urlopen('http://127.0.0.1:8188/queue') as r:
            d = json.load(r)
    except Exception as e:
        print('queue poll error:', e)
        sys.exit(1)
    run = len(d.get('queue_running') or [])
    pend = len(d.get('queue_pending') or [])
    if (run, pend) == (0, 0):
        print('queue drained')
        sys.exit(0)
    time.sleep(5)
print('queue not drained before deadline')
sys.exit(1)
EOF
ls output/ComfyUI_*.png > /dev/null 2>&1 || { echo "no output image"; tail -n 100 /tmp/comfyui.log; exit 1; }
ls output/ComfyUI_*.png | tail -n 1
```

输出结果如下：

```shell #test-result id="run-example" fuzzy='...' fuzzy='xxx'
...prompt queued
...queue drained
...ComfyUI_000xxx
```

```{admonition} Note
:class: note
- `basic_api_example.py` 只负责把工作流 POST 到 `/prompt`（不等待执行完成），完成后轮询 `/queue` 直到 running / pending 均清空。
- `SaveImage` 节点把结果写到 `output/ComfyUI_00001_.png`（全新 checkout 首跑序号为 00001）。
- 20 步 SD 1.5 在单卡 NPU 上数分钟内完成；若 15 分钟未排空按失败处理。
```

## 检查生成结果

```shell #test id="check-png"
cd ComfyUI
python -c "
import glob
paths = glob.glob('output/ComfyUI_*.png')
assert paths, 'no output png'
data = open(paths[0], 'rb').read()
assert data[:8] == b'\x89PNG\r\n\x1a\n', 'not a png'
assert len(data) > 10000, 'png too small: %d' % len(data)
print('png ok:', paths[0], len(data), 'bytes')
"
```

输出结果如下：

```shell #test-result id="check-png" fuzzy='...' fuzzy='xxx'
png ok: ...ComfyUI_000xxx bytes
```

```{admonition} Note
:class: note
校验 PNG 魔数与体积下限，确保 NPU 上产出的不是空图或损坏文件。
```

## 清理

<!--
```shell #test-setup load="server_pid>>pid"
kill -9 <pid> 2>/dev/null || true
sleep 2
echo "comfyui stopped"
```
-->
