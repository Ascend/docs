# Ollama (Ascend NPU)

Ollama 是一个用于本地运行和管理大语言模型的工具，提供模型导入、命令行对话和 HTTP API 服务。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在运行本文档示例之前，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- Go 1.26.10 或更新版本（参考 [Go 安装文档](https://go.dev/doc/install)）
- GCC 11.4 或更新版本
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

本文档示例在 Atlas 900 A2 上，使用 Python 3.12、Go 1.26.10、GCC 11.4.0 和 CANN 9.1.0，验证通过。

本文档配套镜像：
```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12
```

## 加载 CANN 环境

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装 Ollama

安装 CMake，并确认构建工具可用：

```shell #test id="build-tools"
pip install cmake
go version
cmake --version
```
输出结果如下：
```shell #test-result id="build-tools" fuzzy='...' fuzzy='xxx'
...
go version goxxx
cmake version xxx
```
```{admonition} Note
:class: note
`xxx` 表示安装的 Go 和 CMake 版本。
```

从 [Ascend/ollama](https://github.com/Ascend/ollama) 获取支持 CANN 后端的源码：

```shell #test id="ollama-commit"
git clone https://github.com/Ascend/ollama.git
ref=$(git -C ollama rev-parse origin/HEAD)
git -C ollama checkout --detach "$ref"
git -C ollama rev-parse HEAD
```

输出为当前检出的SHA：

```shell #test-result id="ollama-commit" fuzzy='<ref>'
<ref>
```
```{admonition} Not
:class: note
`<ref>` 为默认分支提交 SHA。
```

配置并构建 CANN 后端：

```shell #test-setup
cmake -S ollama -B ollama/build -DOLLAMA_LLAMA_BACKENDS=cann -DGGML_CPU_ALL_VARIANTS=OFF

cmake --build ollama/build
```

## 运行示例

启动服务：

```shell #test-setup
export OLLAMA_LLM_LIBRARY=cann
export OLLAMA_MODELS=./ollama-models

./ollama/ollama serve
```

保持服务运行，在另一个终端运行以下 Python 代码，下载 Qwen3-4B-Instruct-2507 的 Q4_0 量化模型（约 2.38 GB）：

```python #test-setup
from pathlib import Path
from urllib.request import urlopen
import shutil

model = Path('Qwen3-4B-Instruct-2507-Q4_0.gguf')
if not model.exists():
    url = (
        'https://modelscope.cn/models/unsloth/Qwen3-4B-Instruct-2507-GGUF/'
        f'resolve/master/{model.name}'
    )
    partial = model.with_suffix('.gguf.part')
    with urlopen(url, timeout=120) as response, partial.open('wb') as output:
        shutil.copyfileobj(response, output)
    partial.replace(model)
```

写入 `Modelfile` 并导入模型：

```shell #test-setup
printf '%s\n' 'FROM ./Qwen3-4B-Instruct-2507-Q4_0.gguf' > Modelfile
./ollama/ollama create qwen3-4b-ascend -f Modelfile
```

使用 CLI 对话：

```shell #test id="cli-chat"
./ollama/ollama run qwen3-4b-ascend "1 加 1 等于几？请只输出一个阿拉伯数字。"
```

预期回复：

```shell #test-result id="cli-chat"
2
```

通过 HTTP API 对话， 执行以下python代码：

```python #test id="api-chat"
import json
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

payload = json.dumps({
    'model': 'qwen3-4b-ascend',
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
print('reply:', response['message']['content'])
print('tokens:', response['eval_count'])
```

输出结果如下：
```shell #test-result id="api-chat" fuzzy='xxx'
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
