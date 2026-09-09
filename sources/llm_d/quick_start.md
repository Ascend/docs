# llm-d

在一台机器、一张昇腾 NPU 上，用三个普通进程搭起 [llm-d](https://github.com/llm-d/llm-d) 的最小推理路径。

llm-d 是架在推理引擎之上的路由层。最小部署是三个进程：

- **vLLM worker**：真正跑模型，本文用 [vllm-ascend](https://github.com/vllm-project/vllm-ascend)。
- **EPP**：Endpoint Picker，根据本地 `endpoints.yaml` 选 worker，二进制来自 [llm-d-router](https://github.com/llm-d/llm-d-router)。
- **Envoy**：接收客户端请求，先问 EPP，再转到选中的 worker。

请求路径：客户端 `POST :8081` → Envoy → EPP `:9002` 做路由决策 → vLLM `:8000`。

本文基于上游 [no-kubernetes-deployment](https://github.com/llm-d/llm-d/tree/v0.9.0/guides/no-kubernetes-deployment) 指南，把 model server 换成 vllm-ascend，并把模型换成适合单卡验证的 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)。

工作目录统一用 `/root/llm-d`。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 9.1.0，并且可以 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| NNAL | 并且可以 `source /usr/local/Ascend/nnal/atb/set_env.sh`。vLLM-Ascend 会加载 ATB，只 source CANN 不够 |
| Python | 3.12 |

**配套机器**：Atlas 900 A2（Ascend 910B4）。**配套镜像**：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

**版本要求**

| 组件 | 版本 |
| --- | --- |
| llm-d 配置仓 | 当前最新 Release tag（克隆时换成要用的 tag） |
| EPP（llm-d-router） | v0.10.0 |
| Go | 1.26.6 linux-arm64 |
| vLLM | 0.23.0 |
| vllm-ascend | 0.23.0 |
| triton-ascend | 3.2.2 |
| Envoy | 1.33.2 linux-aarch64 |
| 模型 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |

安装 vLLM 0.23.0 时，pip 可能先拉 `torch` 2.11.0。随后安装 `vllm-ascend==0.23.0` 会按插件约束把 `torch` / `torch-npu` 降到 2.10.0。

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 装好 CANN、NNAL 与驱动。

## 1. 加载 CANN 与 NNAL

`npu-smi` 在常见容器里位于 `/usr/local/sbin` 或 `/usr/local/bin`。下面只加载一次；后面的命令都假定仍在这个 shell 里。

```shell
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## 2. 确认 NPU 在线

```shell #test id="npu-smi"
npu-smi info
```

输出结果如下（设备表、功耗、HBM 每次都不同，不必和任何截图逐字一致）：

```shell #test-result id="npu-smi"
...NPU...
```

若提示找不到 `npu-smi`，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。

启动前确认本机这些端口空闲：`8000`（vLLM）、`8081`（Envoy 入口）、`9002` / `9003` / `9090`（EPP）、`19000`（Envoy 管理口）。

## 3. 安装 vLLM Ascend

`torch` 2.10 会顺带装上社区版 CUDA Triton。这个包没有昇腾后端，所以最后再装一次 `vllm-ascend==0.23.0` 要求的 `triton-ascend==3.2.2`，覆盖那个目录，并且加 `--force-reinstall --no-deps`。

```shell #test id="install"
python -m pip install --retries 3 vllm==0.23.0
python -m pip install \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  vllm-ascend==0.23.0
python -m pip install --force-reinstall --no-deps \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  triton-ascend==3.2.2
python -c "from importlib.metadata import version; print('vllm', version('vllm')); print('vllm-ascend', version('vllm-ascend')); print('triton-ascend', version('triton-ascend'))"
```

输出结果如下：

```shell #test-result id="install"
...vllm 0.23.0
vllm-ascend 0.23.0
triton-ascend 3.2.2
```

## 4. 构建 EPP

EPP 来自 `llm-d-router` 的 `cmd/epp`。`make build-epp` 会起 builder 容器。没有 Docker 时，用 Go 1.26.6 直接 `go build`。`go.mod` 精确要求这个版本。

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/huggingface/llm_d'
go_tar="$ci/go1.26.6.linux-arm64.tar.gz"
sum='d0507e9e9d7fe012aae570108cbd76c15de879e17130ab8cb90d4d7445cb1f2e'
mkdir -p /root/llm-d/bin
if [ -f "$go_tar" ]; then
  if echo "$sum  $go_tar" | sha256sum -c >/dev/null 2>&1; then
    if ! /root/llm-d/go/bin/go version 2>/dev/null | grep -q 'go1.26.6'; then
      rm -rf /root/llm-d/go
      tar -C /root/llm-d -xzf "$go_tar"
    fi
  else
    rm -f "$go_tar"
  fi
fi
if [ -x "$ci/epp-v0.10.0" ] && [ -s "$ci/epp-v0.10.0" ]; then
  cp -a "$ci/epp-v0.10.0" /root/llm-d/bin/epp
  chmod 0755 /root/llm-d/bin/epp
fi
```
-->

```shell #test-setup
mkdir -p /root/llm-d/bin
if ! /root/llm-d/go/bin/go version 2>/dev/null | grep -q 'go1.26.6'; then
  if [ ! -f /root/llm-d/go1.26.6.linux-arm64.tar.gz ]; then
    curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 --max-time 600 \
      -o /root/llm-d/go1.26.6.linux-arm64.tar.gz \
      https://go.dev/dl/go1.26.6.linux-arm64.tar.gz
  fi
  rm -rf /root/llm-d/go
  tar -C /root/llm-d -xzf /root/llm-d/go1.26.6.linux-arm64.tar.gz
fi
export GOROOT=/root/llm-d/go
export GOPATH=/root/llm-d/gopath
export PATH="$GOROOT/bin:$PATH"
mkdir -p "$GOPATH"
if [ ! -x /root/llm-d/bin/epp ]; then
  rm -rf /root/llm-d/llm-d-router
  for _ in 1 2 3; do
    GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch v0.10.0 \
      https://github.com/llm-d/llm-d-router /root/llm-d/llm-d-router && break
    rm -rf /root/llm-d/llm-d-router
    sleep 5
  done
  cd /root/llm-d/llm-d-router
  go build -o /root/llm-d/bin/epp ./cmd/epp
fi
/root/llm-d/go/bin/go version
test -x /root/llm-d/bin/epp
/root/llm-d/bin/epp --help >/dev/null
```

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/huggingface/llm_d'
sum='d0507e9e9d7fe012aae570108cbd76c15de879e17130ab8cb90d4d7445cb1f2e'
mkdir -p "$ci"
if [ -f /root/llm-d/go1.26.6.linux-arm64.tar.gz ]; then
  echo "$sum  /root/llm-d/go1.26.6.linux-arm64.tar.gz" | sha256sum -c
  cp -a /root/llm-d/go1.26.6.linux-arm64.tar.gz "$ci/go1.26.6.linux-arm64.tar.gz.part"
  mv "$ci/go1.26.6.linux-arm64.tar.gz.part" "$ci/go1.26.6.linux-arm64.tar.gz"
elif [ -f "$ci/go1.26.6.linux-arm64.tar.gz" ]; then
  echo "$sum  $ci/go1.26.6.linux-arm64.tar.gz" | sha256sum -c
fi
test -x /root/llm-d/bin/epp
cp -a /root/llm-d/bin/epp "$ci/epp-v0.10.0.part"
mv "$ci/epp-v0.10.0.part" "$ci/epp-v0.10.0"
```
-->

## 5. 获取 Envoy

Envoy v1.33.2 官方 Release 直接提供 Linux ARM64 二进制，不需要 Docker，也不需要从源码编译。

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/huggingface/llm_d'
cached="$ci/envoy-1.33.2-linux-aarch_64"
sum='81ec3689a82122eff0ca680c48176f4d351b6f9881a79cc9d9078a6fb4b0b6a8'
mkdir -p /root/llm-d/bin
if [ -f "$cached" ]; then
  if echo "$sum  $cached" | sha256sum -c >/dev/null 2>&1; then
    cp -a "$cached" /root/llm-d/bin/envoy
    chmod 0755 /root/llm-d/bin/envoy
  else
    rm -f "$cached"
  fi
fi
```
-->

```shell #test-setup
mkdir -p /root/llm-d/bin
if [ ! -x /root/llm-d/bin/envoy ]; then
  curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 --max-time 1800 \
    -o /root/llm-d/bin/envoy.part \
    https://github.com/envoyproxy/envoy/releases/download/v1.33.2/envoy-1.33.2-linux-aarch_64
  chmod 0755 /root/llm-d/bin/envoy.part
  mv /root/llm-d/bin/envoy.part /root/llm-d/bin/envoy
fi
/root/llm-d/bin/envoy --version
```

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/huggingface/llm_d'
sum='81ec3689a82122eff0ca680c48176f4d351b6f9881a79cc9d9078a6fb4b0b6a8'
echo "$sum  /root/llm-d/bin/envoy" | sha256sum -c
mkdir -p "$ci"
cp -a /root/llm-d/bin/envoy "$ci/envoy-1.33.2-linux-aarch_64.part"
mv "$ci/envoy-1.33.2-linux-aarch_64.part" "$ci/envoy-1.33.2-linux-aarch_64"
```
-->

输出里应含 `1.33.2`。

## 6. 准备 llm-d 配置

克隆 llm-d 仓，只取 no-kubernetes 指南里的三份 YAML。默认 EPP 去读 `/etc/epp/endpoints.yaml`，默认模型是 `Qwen/Qwen3-32B`。单机验证改成工作目录里的绝对路径，以及 `Qwen/Qwen3-0.6B`。`address` 必须是字面 IPv4，file-discovery 不会解析主机名。

将下面克隆命令里的 `<ref>` 换成要用的 llm-d Release tag（例如 `v0.9.0`）。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/huggingface/llm_d'
guide="$ci/guides/${UPSTREAM_REF}"
dest='/root/llm-d/src/guides/no-kubernetes-deployment'
if [ -s "$guide/config.yaml" ] && [ -s "$guide/endpoints.yaml" ] && [ -s "$guide/envoy.yaml" ]; then
  if grep -q '/etc/epp/endpoints.yaml' "$guide/config.yaml" \
    && grep -q 'Qwen/Qwen3-32B' "$guide/endpoints.yaml" \
    && grep -q '8081' "$guide/envoy.yaml"; then
    mkdir -p "$dest/router/epp" "$dest/router/envoy"
    cp -a "$guide/config.yaml" "$dest/router/epp/config.yaml"
    cp -a "$guide/endpoints.yaml" "$dest/router/epp/endpoints.yaml"
    cp -a "$guide/envoy.yaml" "$dest/router/envoy/envoy.yaml"
  else
    rm -rf "$guide"
  fi
fi
```
-->

```shell #test id="prepare-config" load="upstream_ref>>ref"
mkdir -p /root/llm-d
if [ ! -f /root/llm-d/src/guides/no-kubernetes-deployment/router/epp/config.yaml ]; then
  rm -rf /root/llm-d/src
  for _ in 1 2 3; do
    GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch <ref> \
      https://github.com/llm-d/llm-d /root/llm-d/src && break
    rm -rf /root/llm-d/src
    sleep 5
  done
fi
cp /root/llm-d/src/guides/no-kubernetes-deployment/router/epp/config.yaml /root/llm-d/config.yaml
cp /root/llm-d/src/guides/no-kubernetes-deployment/router/epp/endpoints.yaml /root/llm-d/endpoints.yaml
cp /root/llm-d/src/guides/no-kubernetes-deployment/router/envoy/envoy.yaml /root/llm-d/envoy.yaml
sed -i 's|/etc/epp/endpoints.yaml|/root/llm-d/endpoints.yaml|' /root/llm-d/config.yaml
sed -i 's|Qwen/Qwen3-32B|Qwen/Qwen3-0.6B|' /root/llm-d/endpoints.yaml
grep '/root/llm-d/endpoints.yaml' /root/llm-d/config.yaml
grep 'Qwen/Qwen3-0.6B' /root/llm-d/endpoints.yaml
```

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/huggingface/llm_d'
src='/root/llm-d/src/guides/no-kubernetes-deployment'
guide="$ci/guides/${UPSTREAM_REF}"
mkdir -p "$guide"
cp -a "$src/router/epp/config.yaml" "$guide/config.yaml.part"
mv "$guide/config.yaml.part" "$guide/config.yaml"
cp -a "$src/router/epp/endpoints.yaml" "$guide/endpoints.yaml.part"
mv "$guide/endpoints.yaml.part" "$guide/endpoints.yaml"
cp -a "$src/router/envoy/envoy.yaml" "$guide/envoy.yaml.part"
mv "$guide/envoy.yaml.part" "$guide/envoy.yaml"
```
-->

输出结果如下：

```shell #test-result id="prepare-config"
.../root/llm-d/endpoints.yaml...Qwen/Qwen3-0.6B...
```

## 7. 启动 vLLM

`--max-model-len 2048` 把上下文压到第一次验证够用的长度，KV cache 更小，启动更快。`--gpu-memory-utilization 0.3` 因为 0.6B 用不了一整张 910B。`--served-model-name Qwen/Qwen3-0.6B` 必须和 `endpoints.yaml` 里的 `model` 标签一致，否则 EPP 选不中这个 worker。`--enforce-eager` 关掉 graph 捕获，第一次启动更快，初始化日志也更完整。权重默认从 Hugging Face Hub 拉取。

```shell #test-setup
mkdir -p /root/llm-d
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
port_busy() {
  (echo >/dev/tcp/127.0.0.1/"$1") >/dev/null 2>&1
}
vllm_alive() {
  [ -f /root/llm-d/vllm.pid ] || return 1
  pid=$(cat /root/llm-d/vllm.pid)
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q vllm
}
if vllm_alive; then
  echo 'vllm already running'
else
  if port_busy 8000; then
    echo 'port 8000 is already in use by another process' >&2
    exit 1
  fi
  setsid -f sh -c 'echo $$ > /root/llm-d/vllm.pid; exec vllm serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8000 --served-model-name Qwen/Qwen3-0.6B --max-model-len 2048 --max-num-seqs 4 --gpu-memory-utilization 0.3 --trust-remote-code --enforce-eager >/root/llm-d/vllm.log 2>&1'
fi
for _ in $(seq 1 50); do
  [ -s /root/llm-d/vllm.pid ] && break
  sleep 0.2
done
pid=$(cat /root/llm-d/vllm.pid)
ok=0
for _ in $(seq 1 480); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo 'vllm process exited' >&2
    tail -c 8000 /root/llm-d/vllm.log >&2 || true
    exit 1
  fi
  if curl -sf --max-time 5 http://127.0.0.1:8000/v1/models | grep -q 'Qwen/Qwen3-0.6B'; then
    curl -sS --max-time 5 http://127.0.0.1:8000/v1/models
    ok=1
    break
  fi
  sleep 5
done
[ "$ok" = 1 ]
```

## 8. 启动 EPP

`--pool-name` 和 `--pool-namespace` 在 file-discovery 模式下不是 Kubernetes 对象，只出现在指标和日志里。`--secure-serving=false` 因为 Envoy 和 EPP 同机，不走 TLS。

```shell #test-setup
mkdir -p /root/llm-d
port_busy() {
  (echo >/dev/tcp/127.0.0.1/"$1") >/dev/null 2>&1
}
epp_alive() {
  [ -f /root/llm-d/epp.pid ] || return 1
  pid=$(cat /root/llm-d/epp.pid)
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q /root/llm-d/bin/epp
}
if epp_alive; then
  echo 'epp already running'
else
  if port_busy 9002 || port_busy 9003 || port_busy 9090; then
    echo 'EPP ports 9002/9003/9090 are already in use' >&2
    exit 1
  fi
  setsid -f sh -c 'echo $$ > /root/llm-d/epp.pid; exec /root/llm-d/bin/epp --config-file=/root/llm-d/config.yaml --pool-name=file-discovery --pool-namespace=default --grpc-port=9002 --grpc-health-port=9003 --metrics-port=9090 --secure-serving=false --v=2 >/root/llm-d/epp.log 2>&1'
fi
for _ in $(seq 1 50); do
  [ -s /root/llm-d/epp.pid ] && break
  sleep 0.2
done
pid=$(cat /root/llm-d/epp.pid)
ok=0
for _ in $(seq 1 60); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo 'epp process exited' >&2
    tail -c 8000 /root/llm-d/epp.log >&2 || true
    exit 1
  fi
  if body=$(curl -sf --max-time 5 http://127.0.0.1:9090/metrics); then
    echo "${body:0:200}"
    ok=1
    break
  fi
  sleep 2
done
[ "$ok" = 1 ]
```

## 9. 启动 Envoy

配置里的入口是 `0.0.0.0:8081`，管理口是 `127.0.0.1:19000`。`--concurrency 2` 对单卡验证够用。

```shell #test-setup
mkdir -p /root/llm-d
port_busy() {
  (echo >/dev/tcp/127.0.0.1/"$1") >/dev/null 2>&1
}
envoy_alive() {
  [ -f /root/llm-d/envoy.pid ] || return 1
  pid=$(cat /root/llm-d/envoy.pid)
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q envoy
}
if envoy_alive; then
  echo 'envoy already running'
else
  if port_busy 8081 || port_busy 19000; then
    echo 'Envoy ports 8081/19000 are already in use' >&2
    exit 1
  fi
  setsid -f sh -c 'echo $$ > /root/llm-d/envoy.pid; exec /root/llm-d/bin/envoy --service-node envoy-proxy --log-level warn --concurrency 2 --drain-strategy immediate --drain-time-s 60 -c /root/llm-d/envoy.yaml >/root/llm-d/envoy.log 2>&1'
fi
for _ in $(seq 1 50); do
  [ -s /root/llm-d/envoy.pid ] && break
  sleep 0.2
done
pid=$(cat /root/llm-d/envoy.pid)
ok=0
for _ in $(seq 1 60); do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo 'envoy process exited' >&2
    tail -c 8000 /root/llm-d/envoy.log >&2 || true
    exit 1
  fi
  if curl -sf --max-time 5 http://127.0.0.1:19000/ready >/dev/null; then
    curl -sS --max-time 5 http://127.0.0.1:19000/ready
    ok=1
    break
  fi
  sleep 2
done
[ "$ok" = 1 ]
```

## 10. 发送请求验证

请求打到 Envoy 的 8081 端口，经过 EPP 选路，再由 vLLM 生成。`max_tokens` 只取 8，用来确认全链通。

```shell #test id="e2e"
curl -sS http://127.0.0.1:8081/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello","max_tokens":8,"temperature":0}'
```

输出结果如下（生成文本每次不同，关键是 JSON 类型和模型名；`curl -sS` 默认打成一行）：

```shell #test-result id="e2e"
...text_completion...Qwen/Qwen3-0.6B...
```

## 11. 确认跑在 NPU 上

`Platform plugin ascend is activated` 只说明插件被发现了，不说明设备初始化成功。初始化日志里的 `backend=hccl` 才表示这次 worker 走了昇腾通信后端。

```shell #test id="npu-anchor"
grep -m1 'backend=hccl' /root/llm-d/vllm.log
```

输出结果如下：

```shell #test-result id="npu-anchor"
...backend=hccl...
```

若这一行不存在，先看 `/root/llm-d/vllm.log` 是否在更早阶段就失败（缺 NNAL、卡不可见、权重没下完）。不要只凭 `/v1/models` 返回 200 下结论。

## 12. 清理

只读取本任务写下的 PID 文件。先看 `/proc/<pid>/cmdline` 是不是自己的进程，再结束对应进程组。对不上就丢掉这个文件。不要 `pkill -f vllm` / `pkill -f epp` / `pkill -f envoy`。

```shell #test-setup
stop_one() {
  local f="$1"
  local needle="$2"
  if [ ! -f "$f" ]; then
    return 0
  fi
  local pid
  pid=$(cat "$f")
  if [ -z "$pid" ] || [ ! -r "/proc/$pid/cmdline" ]; then
    rm -f "$f"
    return 0
  fi
  case "$(tr '\0' ' ' < "/proc/$pid/cmdline")" in
    *"$needle"*) ;;
    *)
      rm -f "$f"
      return 0
      ;;
  esac
  if kill -0 "$pid" 2>/dev/null; then
    kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.2
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$f"
}
stop_one /root/llm-d/envoy.pid envoy
stop_one /root/llm-d/epp.pid /root/llm-d/bin/epp
stop_one /root/llm-d/vllm.pid vllm
```
