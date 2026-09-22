# AIBrix

AIBrix 是放在 vLLM 前面的推理网关，把 OpenAI 兼容请求转到后面的引擎。本文在一张昇腾卡上启动一个 vLLM 引擎，再用 local mode 把一次 chat completion 转到这个引擎。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列，Ascend **910B**。本文示例为单卡。

### 软件


| 类别          | 要求                                                                                                 |
| ----------- | -------------------------------------------------------------------------------------------------- |
| CANN        | toolkit 与驱动固件已安装，并可 `source set_env.sh`                                                            |
| Python      | 满足当前 CANN 镜像和 [vLLM-Ascend 安装说明](https://docs.vllm.ai/projects/ascend/en/latest/installation.html) |
| Go          | 1.22 及以上。下文安装当前的 linux-arm64 发行版                                                                   |
| Envoy       | 官方 linux-aarch64 发行版。下文安装当前版本                                                                      |
| vLLM-Ascend | 见下文安装，版本说明见 [vLLM-Ascend 安装说明](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)   |
| 模型          | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)                    |




### 本文验证环境


| 项目     | 内容                                                                              |
| ------ | ------------------------------------------------------------------------------- |
| 镜像     | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12` |
| 设备     | 单卡 Ascend 910B                                                                  |
| Python | 3.12，来自上面的镜像                                                                    |


阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。

## 1. 加载 CANN 环境

加载 CANN 与 ATB，并把 `/usr/local/sbin` 加入 `PATH`。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh
export PATH=/usr/local/sbin:/usr/sbin:$PATH
```



## 2. 检查环境是否就绪



### 2.1 确认 NPU 在线

```shell
npu-smi info
```

如果 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。

### 2.2 确认工具可用

确认 CANN 已加载，并且 `npu-smi` 与 `python` 都在 `PATH` 里。

```shell #test id="check-tools"
test -n "$ASCEND_HOME_PATH"
command -v npu-smi >/dev/null
python --version
```

<!--
```shell #test-result id="check-tools"
Python ...
```
-->

## 3. 安装依赖



### 3.1 安装 iproute2

安装 `iproute2`，其中包含 `ss`。



```shell #test id="install-system-prereqs"
set -eu
apt-get update
apt-get install -y iproute2
ss --version
```

<!--
```shell #test-result id="install-system-prereqs"
...ss utility, iproute2-...
```
-->

### 3.2 安装 Go

从 Go 下载页匹配当前的 linux-arm64 包，解压到 `.aibrix-quick-start/toolchain/go`。

```shell #test id="install-go"
set -eu
mkdir -p .aibrix-quick-start/toolchain
page=$(curl -fsSL --retry 3 --retry-delay 5 --connect-timeout 30 https://go.dev/dl/)
go_tarball=$(printf '%s\n' "$page" | grep -oE 'go[0-9]+\.[0-9]+\.[0-9]+\.linux-arm64\.tar\.gz' | head -n 1)
test -n "$go_tarball"
curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 \
  -o .aibrix-quick-start/go.tar.gz \
  "https://dl.google.com/go/${go_tarball}"
rm -rf .aibrix-quick-start/toolchain/go
tar -C .aibrix-quick-start/toolchain -xzf .aibrix-quick-start/go.tar.gz
.aibrix-quick-start/toolchain/go/bin/go version
```

<!--
```shell #test-result id="install-go"
go version go... linux/arm64
```
-->

### 3.3 安装 Envoy

从 Envoy 当前发行版匹配 linux-aarch64 包，放到 `.aibrix-quick-start/bin/envoy`。

```shell #test id="install-envoy"
set -eu
mkdir -p .aibrix-quick-start/bin
envoy_url=$(curl -fsSL -o /dev/null -w '%{url_effective}' https://github.com/envoyproxy/envoy/releases/latest)
envoy_ver=$(printf '%s\n' "$envoy_url" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+$')
test -n "$envoy_ver"
curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 \
  -o .aibrix-quick-start/bin/envoy \
  "https://github.com/envoyproxy/envoy/releases/download/v${envoy_ver}/envoy-${envoy_ver}-linux-aarch_64"
chmod +x .aibrix-quick-start/bin/envoy
.aibrix-quick-start/bin/envoy --version
```

<!--
```shell #test-result id="install-envoy"
...version...
```
-->

### 3.4 安装 vLLM-Ascend

安装 `vllm`、`vllm-ascend` 和 `triton-ascend`。安装说明见 [vLLM-Ascend 安装说明](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)。

```shell #test id="install-vllm"
set -eu
python -m pip install --retries 3 \
  --extra-index-url https://download.pytorch.org/whl/cpu/ \
  vllm
python -m pip install --retries 3 \
  --extra-index-url https://download.pytorch.org/whl/cpu/ \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  vllm-ascend
python -m pip install --retries 3 --force-reinstall --no-deps \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  triton-ascend
```

<!--
```shell #test-result id="install-vllm"
...
```
-->

## 4. 获取 AIBrix 源码并编译网关

克隆 release tag。`<ref>` 在看护里替换成上游版本，本地可写成例如 `v0.7.0`。



<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="clone-aibrix" load="upstream_ref>>ref"
set -eu
rm -rf .aibrix-quick-start/aibrix
git clone --depth 1 --branch <ref> https://github.com/vllm-project/aibrix.git .aibrix-quick-start/aibrix
git -C .aibrix-quick-start/aibrix describe --tags --exact-match
```

<!--
```shell #test-result id="clone-aibrix" load="upstream_ref>>ref"
<ref>
```
-->

编译 `gateway-plugins`。`GOPATH` 与 `GOCACHE` 放在工作目录。

```shell #test id="build-gateway"
set -eu
export GOPATH="$PWD/.aibrix-quick-start/gopath"
export GOCACHE="$PWD/.aibrix-quick-start/gocache"
mkdir -p "$GOPATH" "$GOCACHE"
cd .aibrix-quick-start/aibrix
CGO_ENABLED=0 "$PWD/../toolchain/go/bin/go" build -tags=nozmq -o bin/gateway-plugins cmd/plugins/main.go
"$PWD/../toolchain/go/bin/go" version -m bin/gateway-plugins
```

<!--
```shell #test-result id="build-gateway"
bin/gateway-plugins: go...
...
```
-->

## 5. 启动 vLLM-Ascend 后端

在单卡上启动一个小规模 vLLM 服务，并等待 `/health` 就绪。

```shell #test id="start-backend"
export PYTHONUNBUFFERED=1
mkdir -p .aibrix-quick-start
: > .aibrix-quick-start/vllm.log
setsid bash -c '
  echo $$ > .aibrix-quick-start/vllm.pid
  exec vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --served-model-name Qwen/Qwen2.5-0.5B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 2048 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.2
' </dev/null >> .aibrix-quick-start/vllm.log 2>&1 &
for i in $(seq 1 180); do
  pid=$(cat .aibrix-quick-start/vllm.pid 2>/dev/null || true)
  if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then
    echo 'vLLM exited before /health succeeded. Full log:'
    cat .aibrix-quick-start/vllm.log
    exit 1
  fi
  if [ -n "${pid}" ] \
      && curl -sf --connect-timeout 2 -- 'http://127.0.0.1:8000/health' >/dev/null \
      && ss -ltnp 'sport = :8000' | grep -q "pid=${pid},"; then
    echo 'vLLM /health OK'
    exit 0
  fi
  sleep 2
done
echo 'timed out waiting for http://127.0.0.1:8000/health. Full log:'
cat .aibrix-quick-start/vllm.log
exit 1
```

输出结果如下：

```shell #test-result id="start-backend"
vLLM /health OK
```

在 vLLM 日志中查找 `backend=hccl`。

```shell #test id="backend-on-npu"
grep -F 'backend=hccl' .aibrix-quick-start/vllm.log
```

<!--
```shell #test-result id="backend-on-npu"
...backend=hccl...
```
-->

## 6. 配置网关并启动 local mode

写入 `.aibrix-quick-start/endpoints.yaml`。模型名是 `Qwen/Qwen2.5-0.5B-Instruct`，引擎地址是 `127.0.0.1:8000`。

保存为 `.aibrix-quick-start/endpoints.yaml`：

```yaml
models:
  - name: "Qwen/Qwen2.5-0.5B-Instruct"
    engine: "vllm"
    endpoints:
      - "127.0.0.1:8000"
```

<!--
```shell #test-setup
mkdir -p .aibrix-quick-start
cat > .aibrix-quick-start/endpoints.yaml <<'EOF'
models:
  - name: "Qwen/Qwen2.5-0.5B-Instruct"
    engine: "vllm"
    endpoints:
      - "127.0.0.1:8000"
EOF
```
-->



启动 local mode。

```shell #test id="start-gateway"
export PATH="$PWD/.aibrix-quick-start/bin:$PATH"
bash .aibrix-quick-start/aibrix/deployment/local/run-local.sh \
  -e "$PWD/.aibrix-quick-start/endpoints.yaml"
```

完整输出较长，其中应包含：

```shell #test-result id="start-gateway"
...
AIBrix gateway is running!
...
```



## 7. 发一次推理

经 `127.0.0.1:10080` 发送一次 chat completion。

```python #test id="infer"
import json
import urllib.request

payload = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct',
    'messages': [{'role': 'user', 'content': 'Say hi in one sentence.'}],
    'max_tokens': 32,
    'temperature': 0,
}
request = urllib.request.Request(
    'http://127.0.0.1:10080/v1/chat/completions',
    data=json.dumps(payload).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST',
)
with urllib.request.urlopen(request, timeout=120) as response:
    body = json.load(response)
content = (body['choices'][0]['message']['content'] or '').strip()
print('model', body['model'])
print(content)
```

输出结果如下：

```text #test-result id="infer"
model Qwen/Qwen2.5-0.5B-Instruct
Hello! How can I assist you today?
```



## 8. 更多用法

多引擎路由、Kubernetes 部署与其余模块的用法与社区文档相同。

- 社区文档：[AIBrix 文档](https://aibrix.readthedocs.io/latest/)
- 本机网关：[Local Mode](https://github.com/vllm-project/aibrix/blob/main/deployment/local/README.md)
- Kubernetes 安装：[Quickstart](https://aibrix.readthedocs.io/latest/getting_started/quickstart.html)

