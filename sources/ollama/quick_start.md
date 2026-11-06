# Ollama (Ascend NPU)

Ollama 是一个用于本地运行和管理大语言模型的工具，提供模型导入、命令行对话和 HTTP API 服务。

## 前置条件

### 硬件

Atlas 900 A2 训练系列产品（Ascend 910B4），并按需完成物理机或容器内的设备挂载。

### 基础软件

在运行本文档示例之前，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- Go 1.26.10 或更新版本（参考 [Go 安装文档](https://go.dev/doc/install)）
- CMake 3.24 或更新版本（参考 [CMake 安装文档](https://cmake.org/download/)）
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

本文档示例使用 Python 3.12、Go 1.26.10、CMake 3.31.10、CANN 9.1.0。

本文档配套镜像：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`

## 加载 CANN 环境

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装 Ollama

安装 CMake，并确认构建工具可用：

```shell #test-setup
pip install cmake
```

```shell #test id="build-tools"
go version
cmake --version
```

```shell #test-result id="build-tools" fuzzy='xxx'
go version goxxx
cmake version xxx
```

```{admonition} Note
:class: note
`xxx` 表示实际安装的 Go 和 CMake 版本。
```

获取源码：

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test-setup load="upstream_ref>>ref"
git clone https://github.com/lltiaor/ollama.git
git -C ollama checkout <ref>
```

```{admonition} Note
:class: note
`<ref>` 表示默认分支。
```

配置并构建 CANN 后端：

```shell #test-setup load="upstream_ref>>ref"
cmake -S ollama -B ollama/build \
  -DOLLAMA_LLAMA_BACKENDS=cann \
  -DGGML_CPU_ALL_VARIANTS=OFF \
  -DOLLAMA_VERSION="<ref>"

cmake --build ollama/build --parallel 8
```

```{admonition} Note
:class: note
`<ref>` 与上一步检出的源码版本相同。
```

## 运行示例

启动服务：

```shell #test-setup
export OLLAMA_LLM_LIBRARY=cann
export ASCEND_RT_VISIBLE_DEVICES=0
export OLLAMA_MODELS=./ollama-models
./ollama/ollama serve
```

另开终端，确认服务已启动：

```shell #test id="server-version"
curl -fsS http://127.0.0.1:11434/api/version
```

```shell #test-result id="server-version" fuzzy='xxx'
{"version":"xxx"}
```

```{admonition} Note
:class: note
`xxx` 表示当前检出的源码版本。
```

下载 Qwen2.5-0.5B-Instruct Q4_0 模型：

```python #test-setup
from pathlib import Path
from urllib.request import urlopen
import shutil

model = Path('qwen2.5-0.5b-instruct-q4_0.gguf')
if not model.exists():
    url = 'https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/' + model.name
    partial = model.with_suffix('.gguf.part')
    with urlopen(url, timeout=120) as response, partial.open('wb') as output:
        shutil.copyfileobj(response, output)
    partial.replace(model)
```

创建并导入模型：

```python #test-setup
from pathlib import Path

Path('Modelfile').write_text(
    'FROM ./qwen2.5-0.5b-instruct-q4_0.gguf\n',
    encoding='utf-8',
)
```

```shell #test-setup
./ollama/ollama create qwen2.5-ascend -f Modelfile
```

使用 CLI 对话：

```shell #test id="cli-chat"
./ollama/ollama run qwen2.5-ascend "用一句话介绍你自己。"
```

```shell #test-result id="cli-chat" fuzzy='xxx'
xxx
```

```{admonition} Note
:class: note
`xxx` 表示模型的实际回复。
```

通过 HTTP API 对话：

```python #test id="api-chat"
import json
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

payload = json.dumps({
    'model': 'qwen2.5-ascend',
    'messages': [{'role': 'user', 'content': '用一句话介绍你自己。'}],
    'stream': False,
}).encode('utf-8')
request = Request(
    'http://127.0.0.1:11434/api/chat',
    data=payload,
    headers={'Content-Type': 'application/json'},
    method='POST',
)
with build_opener(ProxyHandler({})).open(request, timeout=120) as raw:
    response = json.loads(raw.read())

Path('chat-response.json').write_text(
    json.dumps(response, ensure_ascii=False, indent=2),
    encoding='utf-8',
)
print('model:', response['model'])
print('reply:', response['message']['content'])
print('tokens:', response['eval_count'])
```

```shell #test-result id="api-chat" fuzzy='xxx'
model: qwen2.5-ascend
reply: xxx
tokens: xxx
```

```{admonition} Note
:class: note
`xxx` 分别表示模型的实际回复和生成的 token 数。
```

## 外部链接

- [Ollama 官方仓库](https://github.com/ollama/ollama)
- [Ollama 官方文档](https://docs.ollama.com/)
