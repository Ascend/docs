# FastChat

在单张昇腾 NPU 上安装 FastChat，并通过 OpenAI 兼容接口调用 Qwen/Qwen2.5-0.5B-Instruct 完成一次对话。

## 前置条件

### 硬件

Atlas 900 A2 单卡（Ascend NPU），并按需完成物理机或容器内的设备挂载。

### 基础软件

在运行本文档示例之前，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 根据 CANN 版本安装匹配的 `torch_npu`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)）

本文档测试环境使用 Python 3.12、CANN 9.1.0 和 `torch_npu` 2.9.0.post2。

## 加载 CANN 环境

运行示例前加载 CANN 环境变量：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装 FastChat

安装 FastChat：

```shell #test id="install-fastchat"
python -m pip install fschat
python -c "
import fastchat
print('FastChat version:', fastchat.__version__)
"
```

输出结果如下：

```shell #test-result id="install-fastchat" fuzzy='...' fuzzy='xxx'
...
FastChat version: xxx
```

:::{note}
`xxx` 表示实际安装的 FastChat 版本号。
:::

## 运行示例：OpenAI 兼容 API

FastChat 用 controller 管理 model worker，并通过 API server 提供 OpenAI 兼容接口。

安装本示例所需的模型 worker、ModelScope 及 API 服务依赖：

```shell #test-setup id="install-api-deps"
python -m pip install "fschat[model_worker]" "transformers==4.57.6" "modelscope==1.37.0" "fastapi==0.141.1" "uvicorn==0.52.0"
```

controller、model worker 和 API server 都是前台服务，启动后需要保持运行。

**启动 controller。** controller 负责注册和调度 model worker：

```shell #test-setup id="start-controller"
python -m fastchat.serve.controller
```

**启动 model worker。** worker 在 NPU 上加载模型，并以 `Qwen2.5-0.5B-Instruct` 为服务名注册到 controller：

```shell #test-setup id="start-worker"
FASTCHAT_USE_MODELSCOPE=True python -m fastchat.serve.model_worker \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --model-names Qwen2.5-0.5B-Instruct \
  --revision master \
  --device npu
```

**启动 API server。** 服务在 `http://127.0.0.1:8000/v1` 提供 OpenAI 兼容接口：

```shell #test-setup id="start-api"
python -m fastchat.serve.openai_api_server \
  --host 127.0.0.1 \
  --port 8000
```

**检查模型服务。** model worker 加载完成后，通过 `/v1/models` 查看已经注册的模型：

```shell #test id="check-model"
curl -fsS http://127.0.0.1:8000/v1/models -o /tmp/fastchat-models.json
python -m json.tool --no-ensure-ascii /tmp/fastchat-models.json
```

输出结果如下：

```shell #test-result id="check-model" fuzzy='xxx' fuzzy='...'
{
    "object": "list",
    "data": [
        {
            "id": "Qwen2.5-0.5B-Instruct",
            "object": "model",
            "created": xxx,
            "owned_by": "fastchat",
            ...
        }
    ]
}
```

:::{note}
`xxx` 表示动态生成的标识与时间戳，`...` 表示省略的字段。
:::

发送一次对话请求，调用 OpenAI 兼容的 Chat Completions 接口并打印模型回复：

```shell #test id="api-chat"
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"你好"}],"max_tokens":64,"temperature":0}' \
  -o /tmp/fastchat-chat.json
python -m json.tool --no-ensure-ascii /tmp/fastchat-chat.json
```

输出结果如下：

```shell #test-result id="api-chat" fuzzy='xxx' fuzzy='...'
{
    "id": "xxx",
    "object": "chat.completion",
    "created": xxx,
    "model": "Qwen2.5-0.5B-Instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "xxx"
            },
            "finish_reason": xxx
        }
    ],
    "usage": {
        ...
    }
}
```

:::{note}
`xxx` 表示每次请求动态生成的 ID、时间戳、模型回复和 token 统计等内容，`...` 表示省略的字段。
:::

更多 Web UI、多 worker 和评测用法见 [FastChat 官方文档](https://github.com/lm-sys/FastChat)。
