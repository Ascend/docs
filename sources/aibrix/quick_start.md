# aibrix

在单卡昇腾上安装 vLLM-Ascend，拉起 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) 的 OpenAI 兼容服务，再经 AIBrix local mode 转发一次 chat completion。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装，并可 `source set_env.sh` |
| Python | 3.12 |
| Go | 1.22.6，见下文安装 |
| Envoy | 1.39.0 aarch64，见下文安装 |
| vLLM-Ascend | `vllm` / `vllm-ascend` 均为 `0.23.0`，见下文安装 |
| 模型 | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。推荐配套镜像：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

## 1. 加载 CANN 环境

常见容器里 `npu-smi` 在 `/usr/local/sbin`，需要把该目录加入 `PATH`。后面的 vLLM 安装还依赖 ATB。

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

下面确认 CANN 已加载，并且 `npu-smi` 与 `python` 都在 `PATH` 里。

```shell #test id="check-tools"
test -n "$ASCEND_HOME_PATH"
command -v npu-smi >/dev/null
python --version
```

输出结果如下：

```shell #test-result id="check-tools"
Python 3.12...
```

## 3. 安装系统工具

`run-local.sh` 用 `ss` 探测网关端口，没有就装 `iproute2`。

<!--
```shell #test-setup
if getent hosts cache-service.nginx-pypi-cache.svc.cluster.local >/dev/null 2>&1 \
    && [ -f /etc/apt/sources.list ]; then
  sed -Ei 's@(ports|archive).ubuntu.com@cache-service.nginx-pypi-cache.svc.cluster.local:8081@g' /etc/apt/sources.list
fi
```
-->

```shell #test id="install-system-prereqs"
if ! command -v ss >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y iproute2
fi
ss --version
```

输出结果如下：

```shell #test-result id="install-system-prereqs"
...ss utility, iproute2-...
```

## 4. 安装 Go 1.22.6

把官方 linux-arm64 包解压到 `.aibrix-quick-start/toolchain/go`。

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/huggingface/aibrix'
cached="$ci/go1.22.6.linux-arm64.tar.gz"
sum='c15fa895341b8eaf7f219fada25c36a610eb042985dc1a912410c1c90098eaf2'
if [ -f "$cached" ]; then
  if echo "$sum  $cached" | sha256sum -c >/dev/null 2>&1; then
    mkdir -p .aibrix-quick-start/toolchain
    cp -a "$cached" .aibrix-quick-start/go.tar.gz
    tar -C .aibrix-quick-start/toolchain -xzf .aibrix-quick-start/go.tar.gz
  else
    rm -f "$cached"
  fi
fi
```
-->

```shell #test id="install-go"
mkdir -p .aibrix-quick-start/toolchain
if [ ! -x .aibrix-quick-start/toolchain/go/bin/go ]; then
  curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 \
    -o .aibrix-quick-start/go.tar.gz \
    https://dl.google.com/go/go1.22.6.linux-arm64.tar.gz
  tar -C .aibrix-quick-start/toolchain -xzf .aibrix-quick-start/go.tar.gz
fi
.aibrix-quick-start/toolchain/go/bin/go version
```

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/huggingface/aibrix'
src='.aibrix-quick-start/go.tar.gz'
sum='c15fa895341b8eaf7f219fada25c36a610eb042985dc1a912410c1c90098eaf2'
if [ -f "$src" ] && [ ! -f "$ci/go1.22.6.linux-arm64.tar.gz" ]; then
  echo "$sum  $src" | sha256sum -c
  mkdir -p "$ci"
  cp -a "$src" "$ci/go1.22.6.linux-arm64.tar.gz.part"
  mv "$ci/go1.22.6.linux-arm64.tar.gz.part" "$ci/go1.22.6.linux-arm64.tar.gz"
fi
```
-->

输出结果如下：

```shell #test-result id="install-go"
go version go1.22.6 linux/arm64
```

## 5. 安装 Envoy 1.39.0

local mode 经 Envoy 监听 `:10080`。从 GitHub Release 下载官方 aarch64 包。

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/huggingface/aibrix'
cached="$ci/envoy-1.39.0-linux-aarch_64"
sum='ee53a4f5375566f15944dc9cb03afb1fc228df38f61737c677f139213215afcf'
if [ -f "$cached" ]; then
  if echo "$sum  $cached" | sha256sum -c >/dev/null 2>&1; then
    mkdir -p .aibrix-quick-start/bin
    cp -a "$cached" .aibrix-quick-start/bin/envoy
    chmod 0755 .aibrix-quick-start/bin/envoy
  else
    rm -f "$cached"
  fi
fi
```
-->

```shell #test id="install-envoy"
mkdir -p .aibrix-quick-start/bin
if [ ! -x .aibrix-quick-start/bin/envoy ]; then
  curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 \
    -o .aibrix-quick-start/bin/envoy.part \
    https://github.com/envoyproxy/envoy/releases/download/v1.39.0/envoy-1.39.0-linux-aarch_64
  mv .aibrix-quick-start/bin/envoy.part .aibrix-quick-start/bin/envoy
  chmod +x .aibrix-quick-start/bin/envoy
fi
.aibrix-quick-start/bin/envoy --version
```

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/huggingface/aibrix'
src='.aibrix-quick-start/bin/envoy'
sum='ee53a4f5375566f15944dc9cb03afb1fc228df38f61737c677f139213215afcf'
echo "$sum  $src" | sha256sum -c
if [ ! -f "$ci/envoy-1.39.0-linux-aarch_64" ]; then
  mkdir -p "$ci"
  cp -a "$src" "$ci/envoy-1.39.0-linux-aarch_64.part"
  mv "$ci/envoy-1.39.0-linux-aarch_64.part" "$ci/envoy-1.39.0-linux-aarch_64"
fi
```
-->

输出结果如下：

```shell #test-result id="install-envoy"
...1.39.0...
```

## 6. 获取 AIBrix 源码并编译网关

克隆 release tag（`<ref>` 可改为例如 `v0.7.0`）。只用下面的 `go build`，不要跑 `make build-gateway-plugins-nozmq`。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="clone-aibrix" load="upstream_ref>>ref"
if [ ! -d .aibrix-quick-start/aibrix/.git ]; then
  rm -rf .aibrix-quick-start/aibrix
  for _ in 1 2 3; do
    GIT_TERMINAL_PROMPT=0 GIT_HTTP_VERSION=HTTP/1.1 \
      git clone --depth 1 --branch <ref> https://github.com/vllm-project/aibrix.git .aibrix-quick-start/aibrix && break
    rm -rf .aibrix-quick-start/aibrix
    sleep 5
  done
fi
git -C .aibrix-quick-start/aibrix describe --tags --exact-match
```

输出结果如下：

```shell #test-result id="clone-aibrix" load="upstream_ref>>ref"
<ref>
```

`GOPATH` / `GOCACHE` 使用工作目录。

```shell #test id="build-gateway"
export GOPATH="$PWD/.aibrix-quick-start/gopath"
export GOCACHE="$PWD/.aibrix-quick-start/gocache"
mkdir -p "$GOPATH" "$GOCACHE"
cd .aibrix-quick-start/aibrix
CGO_ENABLED=0 "$PWD/../toolchain/go/bin/go" build -tags=nozmq -o bin/gateway-plugins cmd/plugins/main.go
"$PWD/../toolchain/go/bin/go" version -m bin/gateway-plugins
```

输出结果如下：

```shell #test-result id="build-gateway"
bin/gateway-plugins: go1.22.6
...
```

<!--
```shell #test-setup
set -euo pipefail
plugin="$PWD/.aibrix-quick-start/aibrix/bin/gateway-plugins"
if python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 6060))' >/dev/null 2>&1; then
  echo pprof_port_free
  exit 0
fi
test -x "$plugin"
real="$plugin.real"
mv "$plugin" "$real"
cat > "$plugin" <<'WRAP'
#!/usr/bin/env bash
set -euo pipefail
hosts=$(mktemp)
{
  printf '%s\t%s\n' '127.0.0.2' 'localhost'
  printf '%s\t%s\n' '127.0.0.1' "$(hostname)"
} > "$hosts"
exec unshare --user --map-root-user --mount --fork -- \
  bash -c 'mount --bind "$1" /etc/hosts; shift; exec "$@"' \
  bash "$hosts" \
  REAL_PLACEHOLDER "$@"
WRAP
sed -i "s|REAL_PLACEHOLDER|${real}|g" "$plugin"
chmod +x "$plugin"
echo pprof_port_wrapped
```
-->

## 7. 安装 vLLM-Ascend

分三步安装，不要合成一次 `pip install`。最后一步必须 `--force-reinstall --no-deps`，否则会留下社区 CUDA 版 Triton，第一次推理报错。

保存为 `.aibrix-quick-start/print_pkg_versions.py`：

```python
import importlib.metadata as m

for n in ['torch', 'torch-npu', 'vllm', 'vllm-ascend']:
    print(n, m.version(n))
```

<!--
```shell #test-setup
mkdir -p .aibrix-quick-start
cat > .aibrix-quick-start/print_pkg_versions.py <<'PY'
import importlib.metadata as m

for n in ['torch', 'torch-npu', 'vllm', 'vllm-ascend']:
    print(n, m.version(n))
PY
```
-->

```shell #test id="install-vllm"
python -m pip install --retries 3 vllm==0.23.0
python -m pip install \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  vllm-ascend==0.23.0
python -m pip install --force-reinstall --no-deps \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  triton-ascend==3.2.2
python .aibrix-quick-start/print_pkg_versions.py
```

输出结果如下：

```shell #test-result id="install-vllm"
...
torch 2.10.0...
torch-npu 2.10.0.post4
vllm 0.23.0...
vllm-ascend 0.23.0
```

## 8. 启动 vLLM-Ascend 后端

后台启动服务、日志落盘，并轮询 `/health` 直到就绪。`--max-model-len` / `--max-num-seqs` / `--gpu-memory-utilization` 压到单卡演示规模；`--served-model-name` 须与下一节网关配置一致。

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

日志里应有 `backend=hccl`，说明这一次推理走了昇腾集合通信。

```shell #test id="backend-on-npu"
grep -F 'backend=hccl' .aibrix-quick-start/vllm.log
```

输出结果如下：

```shell #test-result id="backend-on-npu"
...backend=hccl...
```

## 9. 配置网关并启动 local mode

模型名与 `--served-model-name` 一致；endpoint 用 `127.0.0.1:8000`。

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
cat > .aibrix-quick-start/endpoints.yaml <<'YAML'
models:
  - name: "Qwen/Qwen2.5-0.5B-Instruct"
    engine: "vllm"
    endpoints:
      - "127.0.0.1:8000"
YAML
```
-->

`run-local.sh` 从 PATH 找 `envoy`；`endpoints.yaml` 须绝对路径。

```shell #test id="start-gateway"
export PATH="$PWD/.aibrix-quick-start/bin:$PATH"
bash .aibrix-quick-start/aibrix/deployment/local/run-local.sh \
  -e "$PWD/.aibrix-quick-start/endpoints.yaml"
```

输出结果如下：

```shell #test-result id="start-gateway"
...
AIBrix gateway is running!
...
```

## 10. 发一次推理

经 `:10080` 发 chat completion。生成内容非确定性，只核对模型名与回复非空。

保存为 `.aibrix-quick-start/chat_completion.py`：

```python
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
print('content_nonempty', 'true' if content else 'false')
print('completion_tokens', body['usage']['completion_tokens'])
```

<!--
```shell #test-setup
cat > .aibrix-quick-start/chat_completion.py <<'PY'
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
print('content_nonempty', 'true' if content else 'false')
print('completion_tokens', body['usage']['completion_tokens'])
PY
```
-->

```shell #test id="infer"
python .aibrix-quick-start/chat_completion.py
```

输出结果如下：

```shell #test-result id="infer"
model Qwen/Qwen2.5-0.5B-Instruct
content_nonempty true
completion_tokens ...
```

## 11. 停掉本机进程

先停网关，再按 PID 停 vLLM。

```shell #test-setup
if [ -x .aibrix-quick-start/aibrix/deployment/local/stop-local.sh ]; then
  bash .aibrix-quick-start/aibrix/deployment/local/stop-local.sh || true
fi
if [ -f .aibrix-quick-start/vllm.pid ]; then
  pid=$(cat .aibrix-quick-start/vllm.pid)
  if [ -n "${pid}" ]; then
    kill "${pid}" 2>/dev/null || true
  fi
fi
```