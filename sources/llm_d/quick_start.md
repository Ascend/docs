# llm-d

在一台机器、一张昇腾 NPU 上，用三个普通进程搭起 [llm-d](https://github.com/llm-d/llm-d) 的最小推理路径。

llm-d 是架在推理引擎之上的路由层。最小部署是三个进程：

- **vLLM worker**：真正跑模型，本文用 [vllm-ascend](https://github.com/vllm-project/vllm-ascend)。
- **EPP**：Endpoint Picker，根据本地 `endpoints.yaml` 选 worker，二进制来自 [llm-d-router](https://github.com/llm-d/llm-d-router)。
- **Envoy**：接收客户端请求，先问 EPP，再转到选中的 worker。

请求路径：客户端 `POST :8081` → Envoy → EPP `:9002` 做路由决策 → vLLM `:8000`。

本文基于上游 [no-kubernetes-deployment](https://github.com/llm-d/llm-d/tree/v0.9.0/guides/no-kubernetes-deployment) 指南，把 model server 换成 vllm-ascend，并把模型换成适合单卡验证的 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)。工作目录统一用 `/root/llm-d`。

## 前置条件



### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为单卡。

### 软件


| 类别     | 要求                                                                                                      |
| ------ | ------------------------------------------------------------------------------------------------------- |
| CANN   | toolkit 已安装，并且可以 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`                                   |
| NNAL   | 可以 `source /usr/local/Ascend/nnal/atb/set_env.sh`                                                       |
| Python | 落在当前 CANN 镜像和 [vLLM-Ascend 安装说明](https://docs.vllm.ai/projects/ascend/en/latest/installation.html) 的范围内 |




### 本文验证环境

下表是编写本文时跑通的组合，不是唯一支持组合。


| 项目            | 内容                                                                              |
| ------------- | ------------------------------------------------------------------------------- |
| 机器            | Atlas 900 A2，Ascend 910B4，单卡                                                    |
| 镜像            | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12` |
| Python        | 3.12，来自上面的镜像                                                                    |
| CANN          | 9.1.0                                                                           |
| llm-d 配置仓     | Release tag，看护里替换尖括号里的版本                                                        |
| EPP           | llm-d-router v0.10.0                                                            |
| Go            | 1.26.6 linux-arm64                                                              |
| vLLM          | 0.23.0                                                                          |
| vllm-ascend   | 0.23.0                                                                          |
| triton-ascend | 3.2.2                                                                           |
| Envoy         | 1.33.2 linux-aarch64                                                            |
| 模型            | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)                       |


阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 装好 CANN、NNAL 与驱动。

## 1. 加载 CANN 与 NNAL

加载 CANN 与 NNAL，并把 `/usr/local/sbin` 和 `/usr/local/bin` 加入 `PATH`。

```shell
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```



## 2. 确认 NPU 在线

```shell
npu-smi info
```

输出类似：

```text
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B3               | OK            | 91.3        36                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          3465 / 65536         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
+===========================+===============+====================================================+
```

若提示找不到 `npu-smi`，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。

启动前确认本机这些端口空闲：`8000`（vLLM）、`8081`（Envoy 入口）、`9002` / `9003` / `9090`（EPP）、`19000`（Envoy 管理口）。

## 3. 安装 vLLM Ascend

安装 `vllm`、`vllm-ascend` 和 `triton-ascend`。

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

<!--
完整输出较长，其中应包含：

```shell #test-result id="install"
...vllm 0.23.0
vllm-ascend 0.23.0
triton-ascend 3.2.2
```
-->

## 4. 构建 EPP

用 Go 1.26.6 编译 `llm-d-router` 的 `cmd/epp`。

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

```shell #test id="build-epp"
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

输出结果如下：

```shell #test-result id="build-epp"
go version go1.26.6 linux/arm64
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

下载 Envoy 1.33.2 的 Linux ARM64 二进制。

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

```shell #test id="install-envoy"
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

输出里应包含：

```shell #test-result id="install-envoy"
.../root/llm-d/bin/envoy  version: .../1.33.2/Clean/RELEASE/BoringSSL...
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

## 6. 准备 llm-d 配置

克隆 llm-d，把 no-kubernetes 指南里的三份 YAML 复制到工作目录，并把模型改成 `Qwen/Qwen3-0.6B`。复制后的 `endpoints.yaml` 使用字面 IPv4 地址。

将下面克隆命令里的 `<ref>` 换成要用的 llm-d Release tag，例如 `v0.9.0`。

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

```text #test-result id="prepare-config"
      path: /root/llm-d/endpoints.yaml  # EPP must have network access to all worker IPs for metrics scraping
      model: Qwen/Qwen3-0.6B
```



## 7. 启动 vLLM

在 0 号卡上启动 vLLM，模型与服务名都是 `Qwen/Qwen3-0.6B`。

```shell #test id="start-vllm"
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
  echo 'vllm already running' >&2
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

完整输出较长，其中应包含：

```text #test-result id="start-vllm"
...Qwen/Qwen3-0.6B...
```



## 8. 启动 EPP

启动 EPP，配置文件是工作目录里的 `config.yaml`。

```shell #test id="start-epp"
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
  echo 'epp already running' >&2
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
  if curl -sf --max-time 5 http://127.0.0.1:9090/metrics | grep -m1 '^# HELP inference_extension_info '; then
    ok=1
    break
  fi
  sleep 2
done
[ "$ok" = 1 ]
```

输出结果如下：

```text #test-result id="start-epp"
# HELP inference_extension_info [ALPHA] General information of the current build of Inference Extension.
```



## 9. 启动 Envoy

启动 Envoy，入口端口是 8081，管理端口是 19000。

```shell #test id="start-envoy"
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
  echo 'envoy already running' >&2
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

输出结果如下：

```shell #test-result id="start-envoy"
LIVE
```



## 10. 发送请求验证

把补全请求发到 Envoy 的 8081 端口。`temperature` 为 0，`seed` 为 42，`max_tokens` 为 8。

```shell #test id="e2e"
curl -sS http://127.0.0.1:8081/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"Hello","max_tokens":8,"temperature":0,"seed":42}'
```

完整输出较长，其中应包含：

```text #test-result id="e2e"
...text_completion...Qwen/Qwen3-0.6B...
```



## 11. 确认跑在 NPU 上

在 vLLM 日志里查看昇腾通信后端。

```shell #test id="npu-anchor"
grep -m1 'backend=hccl' /root/llm-d/vllm.log
```

完整输出较长，其中应包含：

```shell #test-result id="npu-anchor"
...backend=hccl...
```

Kubernetes 部署、多模型路由和其余模块与上游社区相同。见 [llm-d 文档](https://llm-d.ai)。
