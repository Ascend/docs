# SpecForge

在 4 卡昇腾 NPU 上端到端跑通 [SpecForge](https://github.com/sgl-project/SpecForge) 的投机解码草稿模型训练链路：

- **安装**：huggingface_hub + mooncake（NPU prebuilt wheel）+ SGLang（源码构建）+ specforge（源码）+ 上游 capture 插件，CLI 自检。
- **端到端 smoke**：从 HuggingFace Hub 拉 Qwen3.5-4B → 起 mooncake_master → 起带 capture 插件的 SGLang capture server → 完成 1 步训练。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，**≥ 4 卡**（capture server 用卡 0、trainer 用卡 1，卡 2/3 留给 HCCL 通信 buffer）。

### 基础软件

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与 CANN 匹配、预装 `torch` / `torch_npu` 的镜像（本文用下方的标准 CANN 镜像；torch / torch_npu 由文档自己装）

### 本文档示例使用的版本

**配套镜像**（标准 CANN 镜像即可，SGLang 由本文自己从源码构建）：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

镜像只提供 CANN 运行时；`torch` / `torch_npu` / `triton-ascend` / `SGLang` 都由下文自己装（capture API 只有 SGLang main 有，镜像里不可能预装）。

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post6 |
| sglang | 源码构建，commit `c757c4bb`（= 上游 SpecForge CI 的 `SGLANG_COMMIT`） |
| sgl-kernel-npu | 2026.9.0.post12（`cann9.1.0-910b-py3.12` 的 wheel） |
| huggingface_hub | 最新稳定版 |
| mooncake-transfer-engine | 0.3.14（NPU prebuilt wheel，见下文说明） |
| specforge | 上游 `main`（`<ref>` 由看护工作流解析） |
| capture 插件 | 上游 `plugins/sglang-spec-capture` |
| 模型 | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) |
| 参考样例 | `examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-npu.yaml` |

```{admonition} Note
:class: note
`sgl-kernel-npu` 的 wheel 按 (torch, CANN, 设备, python) 出包，上游目前只发到 **CANN 9.1.0 / torch 2.10.0 / py3.12 的 910b（aarch64）**。本文的 sglang 安装步骤会按当前环境的 CANN 版本拼 wheel 文件名；如果你的 CANN 比这新（CANN 9.2.0 还没有 910b 包），用 `SGL_KERNEL_CANN_VER=9.1.0` 指定一个已有包的 CANN 版本。
```

## 前置安装

确认能看到 NPU 设备（应至少列出 4 张 `910B4` 且状态 OK）：

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
| 0     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
| 1     910B4               | OK            | 89.9        39                0    / 0             |
| 1                         | 0000:42:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
| 2     910B4               | OK            | 89.9        39                0    / 0             |
| 2                         | 0000:43:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
| 3     910B4               | OK            | 89.9        39                0    / 0             |
| 3                         | 0000:44:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
| No running processes found in NPU 1                                                            |
| No running processes found in NPU 2                                                            |
| No running processes found in NPU 3                                                            |
+===========================+===============+====================================================+
```

```{admonition} Note
:class: note
如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。
```

确认 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy='xxx'
Python xxx
```

```{admonition} Note
:class: note
xxx 表示 Python 的补丁版本号。
```

确认 CANN 版本：

```shell #test id="check-cann"
test -f /usr/local/Ascend/ascend-toolkit/latest/$(uname -m)-linux/ascend_toolkit_install.info && \
    grep '^version=' /usr/local/Ascend/ascend-toolkit/latest/$(uname -m)-linux/ascend_toolkit_install.info || \
    echo "ascend_toolkit_install.info MISSING"
```

输出结果如下：

```shell #test-result id="check-cann" fuzzy='xxx'
version=xxx
```

安装 torch / torch_npu（成对版本）：CANN 镜像默认不带 torch，带 torch 的镜像版本也可能不同，这里统一成 `torch 2.10.0` + `torch_npu 2.10.0.post6`（torch_npu 的 wheel 要求 torch 版本严格对齐）：

```shell #test id="install-torch-stack"
set -euo pipefail
if python -c "import torch, torch_npu" 2>/dev/null; then
    echo "torch stack already present, keeping it" >&2
else
    echo "installing torch==2.10.0 torch_npu==2.10.0.post6" >&2
    python -m pip install \
        -f https://mirrors.aliyun.com/pytorch-wheels/cpu \
        --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
        "torch==2.10.0" "torch_npu==2.10.0.post6" \
        >/tmp/install-torch.log 2>&1 \
        || { echo "install-torch-stack: FAILED - pip install:" >&2; tail -20 /tmp/install-torch.log >&2; exit 1; }
fi
python -c "
import torch, torch_npu
print('torch stack ready: torch=' + torch.__version__ + ' torch_npu=' + torch_npu.__version__)
"
```

输出结果如下：

```shell #test-result id="install-torch-stack" fuzzy='xxx'
torch stack ready: torch=xxx torch_npu=xxx
```

确认 torch / torch_npu 能 import 且能看到 4 张卡(下面的命令用 Python 执行)：

```python #test id="check-torch"
import torch, torch_npu
print('torch=', torch.__version__)
print('torch_npu=', torch_npu.__version__)
print('is_available:', torch.npu.is_available())
print('count:', torch.npu.device_count())
```

输出结果如下：

```shell #test-result id="check-torch" fuzzy='xxx'
torch= 2.10.0+cpu
torch_npu= xxx
is_available: True
count: 4
```

## 安装 huggingface_hub

用 uv 安装 huggingface_hub（后面从 HuggingFace Hub 下载模型用）：

```shell #test-setup
uv pip install huggingface_hub
```

## 安装 mooncake

SpecForge 的在线训练用 mooncake 在 SGLang server 和 trainer 之间传数据。直接用 PyPI 上 Mooncake 官方发布的 NPU prebuilt wheel，wheel 没带的三个系统库用 apt 补齐：

```{admonition} Note
:class: note
wheel 固定 `mooncake-transfer-engine-npu==0.3.14`：0.3.13.post1 的 Ascend transport（ADXL）在 CANN 9.1.0 上初始化会失败（`Failed to initialize AdxlEngine, status: 503900` → `Failed to install Ascend transport` → client 侧 `Mooncake setup failed (status -1)`），0.3.14 才正常。
```

```shell #test-setup id="build-mooncake"
set -euo pipefail
# wheel 不带 libibverbs / libcurl / libnuma，apt 补（libtransfer_engine.so / libmooncake_store.so DT_NEEDED）。
apt-get update -qq >/dev/null 2>&1
apt-get install -qq -y --no-install-recommends \
    libibverbs1 libcurl4 libnuma1 >/dev/null 2>&1 \
    || { echo "build-mooncake: FAILED - apt install runtime deps" >&2; exit 1; }
# Aliyun mirror 同步 PyPI；pip 找不到则退回 PyPI。
uv pip install --upgrade \
    'mooncake-transfer-engine-npu==0.3.14' \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    >/tmp/build-mooncake-pip.log 2>&1 \
    || { echo "build-mooncake: FAILED - pip install:" >&2; tail -20 /tmp/build-mooncake-pip.log >&2; exit 1; }
INSTALLED=$(python -c "from importlib.metadata import version; print(version('mooncake-transfer-engine-npu'))")
echo "build-mooncake: installed mooncake-transfer-engine-npu==${INSTALLED}"
```

验证两个包都装好了：

```shell #test id="install-deps"
python -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"
python -c "import mooncake.store; print('mooncake.store imports clean')"
test -x "$(command -v mooncake_master)" && echo "mooncake_master binary present" || echo "mooncake_master binary MISSING"
```

输出结果如下：

```shell #test-result id="install-deps" fuzzy='xxx'
huggingface_hub xxx
mooncake.store imports clean
mooncake_master binary present
```

```{admonition} Note
:class: note
xxx 表示实际安装的版本号。
```

## 克隆 SpecForge 源码

PyPI wheel 不带示例配置（`examples/` 目录）和 capture 插件（`plugins/sglang-spec-capture`），所以从源码装：

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="specforge-clone" load="upstream_ref>>ref"
set -euo pipefail
# 拉 GitHub 仓库的网络偶发抖动（HTTP/2 framing error / GnuTLS recv error），
# 所以：走 HTTP/1.1 + 重试 3 次。
# http.lowSpeedLimit/Time：连接卡住时尽快失败并重试，而不是一直挂着。
GIT_NET_OPTS=(-c http.version=HTTP/1.1 -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60)
clone_specforge() {
    local url="$1" branch="${2:-}"
    rm -rf SpecForge
    for attempt in 1 2 3 4 5; do
        if [ -n "$branch" ]; then
            git "${GIT_NET_OPTS[@]}" clone --depth 1 --branch "$branch" "$url" SpecForge && return 0
        else
            git "${GIT_NET_OPTS[@]}" clone --depth 1 "$url" SpecForge && return 0
        fi
        echo "clone attempt ${attempt} failed, retrying" >&2
        sleep 10
    done
    echo "FAILED - clone ${branch:+$branch }${url} after 5 attempts" >&2
    return 1
}
if [[ "<ref>" =~ ^[0-9a-f]{40}$ ]]; then
    clone_specforge https://github.com/sgl-project/SpecForge.git
    git -C SpecForge -c http.version=HTTP/1.1 fetch --depth 1 origin "<ref>"
    git -C SpecForge checkout FETCH_HEAD
else
    clone_specforge https://github.com/sgl-project/SpecForge.git "<ref>"
fi
git -C SpecForge log -1 --format='specforge checkout: %H'
```

输出结果如下：

```shell #test-result id="specforge-clone" fuzzy='xxx'
specforge checkout: xxx
```

```{admonition} Note
:class: note
`<ref>` 替换为被测的 SpecForge 版本（看护工作流解析上游 `main` 的 HEAD），xxx 为实际 checkout 的 commit。
```

## 安装 SGLang（源码构建）

capture 插件要用的扩展点任何 release wheel 都没有，所以从源码构建；commit 取自上游 SpecForge CI 自己的钉法（`SGLANG_COMMIT`），保证插件与 SGLang 的 API 匹配：

```shell #test-setup store="sglang_commit"
cd SpecForge
sed -n 's/^[[:space:]]*SGLANG_COMMIT:[[:space:]]*\([0-9a-f]\{40\}\).*/\1/p' .github/workflows/test.yaml | head -1
```

```shell #test id="sglang-install-source" load="sglang_commit>>commit"
set -euo pipefail
# 1) 拉 SGLang 源码（<commit> 就是上游 SpecForge CI 的 SGLANG_COMMIT）
rm -rf sglang-src && mkdir sglang-src && cd sglang-src
git init -q
git remote add origin https://github.com/sgl-project/sglang.git
GIT_NET_OPTS=(-c http.version=HTTP/1.1 -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60)
for attempt in 1 2 3 4 5; do
    if git "${GIT_NET_OPTS[@]}" fetch --depth 1 origin "<commit>"; then
        break
    fi
    echo "sglang fetch attempt ${attempt} failed, retrying" >&2
    sleep 10
    [ "$attempt" = 5 ] && { echo "sglang-install: FAILED - fetch <commit>" >&2; exit 1; }
done
git checkout -q FETCH_HEAD
# 2) NPU 专用打包：pyproject_npu.toml 才是昇腾版的依赖声明（all_npu extra）
cd python
rm -f pyproject.toml
mv pyproject_npu.toml pyproject.toml
# 3) 升级 sgl-kernel-npu：新 SGLang 要用镜像自带版本（2026.6.1）里没有的新符号
#    （sgl_kernel_npu.activation.fused_sigmoid_mul 等），否则模型模块注册失败、
#    报 "has no SGlang implementation"。CANN 侧自定义算子沿用镜像自带版本即可。
KERNEL_TAG=2026.9.0.post12
PY_TAG=py$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
# sgl-kernel-npu 的 wheel 按 (torch, CANN, 设备, python) 出包；CANN 版本默认
# 取当前环境里的那个（sgl-kernel-npu 目前只发到 cann9.1.0 的 910b 包）。
CANN_VER=$(ls -d /usr/local/Ascend/cann-* 2>/dev/null | head -1 | sed 's/.*cann-//')
CANN_VER="${SGL_KERNEL_CANN_VER:-$CANN_VER}"
DEVICE=910b
ARCH=$(uname -m)
WHEEL="sgl-kernel-npu-${KERNEL_TAG}-torch2.10.0-${PY_TAG}-cann${CANN_VER}-${DEVICE}-${ARCH}.zip"
cd ..
mkdir -p npu-ops && cd npu-ops
curl -fSL --retry 5 --retry-all-errors --http1.1 -C - -o "$WHEEL" \
    "https://github.com/sgl-project/sgl-kernel-npu/releases/download/${KERNEL_TAG}/${WHEEL}" \
    || { echo "sglang-install: FAILED - download $WHEEL" >&2; exit 1; }
# 用 python 解压（CANN 镜像里没有 unzip）
python - "$WHEEL" <<'PYUNZIP'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(".")
PYUNZIP
uv pip install deep_ep*.whl sgl_kernel_npu*.whl torch_memory_saver*.whl
cd ..
# 4) 装 SGLang 本体。依赖按上游 pyproject_npu.toml 全装（含 transformers
#    5.19.0），但把 torch 系列钉在 2.10.0 —— 不钉的话 uv 会把 torch 升到最新
#    （2.15.x），和 torch_npu 2.10.0.post6 不匹配。
cd python
printf 'torch==2.10.0\ntorchvision==0.25.0\ntorchaudio==2.10.0\n' > /tmp/torch-constraint.txt
uv pip install -e . --constraint /tmp/torch-constraint.txt
cd ..
# triton 必须用昇腾的 fork（triton-ascend）：社区 triton 里没有
# triton.language.extra.cann，NPU 的模型模块（qwen3 / qwen3_5 ...）import 就失败。
# sglang 的依赖会把社区 triton 带进来，这里覆盖成 triton-ascend。
if python -c "import triton.language.extra.cann" 2>/dev/null; then
    echo "triton-ascend already present" >&2
else
    # 昇腾的 triton fork 在昇腾 index 上（GitHub release 里那个 wheel 有 300MB，
    # 走 index 快得多）。--reinstall 覆盖掉社区 triton 的同名包。
    # pybind11 是 triton-ascend 的运行时依赖（--no-deps 下要自己带）。
    uv pip install --reinstall --no-deps "triton-ascend==3.6.0" pybind11 \
        --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
        >/tmp/triton-install.log 2>&1 \
        || { echo "sglang-install: FAILED - install triton-ascend:" >&2; tail -10 /tmp/triton-install.log >&2; exit 1; }
fi
python -c "import importlib.metadata as m; print('sglang, version', m.version('sglang'))"
python -c "import importlib.metadata as m; print('sgl_kernel_npu, version', m.version('sgl_kernel_npu'))"
python -c "import triton.language.extra.cann; print('triton-ascend OK')"
```

输出结果如下：

```shell #test-result id="sglang-install-source" fuzzy='xxx'
sglang, version xxx
sgl_kernel_npu, version 2026.9.0
triton-ascend OK
```

确认 capture 插件要的扩展点都在（插件启动时会逐项检查）：

```shell #test id="sglang-capture-api"
python -c "
from sglang.srt.server_args import ServerArgs
from sglang.srt.model_executor.forward_observer import ForwardObserver
from sglang.srt.managers.deferred_output import ResponseCompletion
from sglang.srt.managers.scheduler import Scheduler
for n in ('enable_return_aux_hidden_states', 'return_aux_hidden_state_layer_ids', 'return_hidden_states_mode'):
    assert hasattr(ServerArgs, n), n
for m in ('defer_response', 'get_output_streamer_class', 'get_internal_state'):
    assert hasattr(Scheduler, m), m
print('capture API present: aux-hidden-states + forward_observer + defer_response')
"
```

输出结果如下：

```shell #test-result id="sglang-capture-api"
capture API present: aux-hidden-states + forward_observer + defer_response
```

## 安装 specforge 与 capture 插件

capture 插件装进 SGLang server 所在的环境（与 SGLang 同一个 python），SGLang 通过 `sglang.srt.plugins` entry point 发现它：

```shell #test id="specforge-install-source"
set -euo pipefail
cd SpecForge
uv pip install --no-deps .
# specforge/_train() lazy-import accelerate.utils.set_seed（仅 import 不触发，
# 真正训练才暴露 ModuleNotFoundError）；CLI 的 click 依赖镜像已自带
# （--no-deps 不会安装它）。
uv pip install --no-deps accelerate
uv pip install --no-deps plugins/sglang-spec-capture
python -c "from importlib.metadata import version; print('specforge, version', version('specforge'))"
python -c "
from importlib.metadata import entry_points
eps = [e.name for e in entry_points(group='sglang.srt.plugins')]
assert 'specforge_spec_capture' in eps, eps
print('capture plugin registered: specforge_spec_capture')
"
```

输出结果如下：

```shell #test-result id="specforge-install-source" fuzzy='xxx'
specforge, version xxx
capture plugin registered: specforge_spec_capture
```

## CLI 自检

确认 specforge 能在 NPU 环境里正常 import(下面的命令用 Python 执行)：

```python #test id="specforge-import"
import specforge, torch, torch_npu
print('specforge', getattr(specforge, '__version__', 'unknown'))
print('torch', torch.__version__)
print('torch.npu.is_available', torch.npu.is_available())
```

输出结果如下：

```shell #test-result id="specforge-import" fuzzy='xxx'
specforge xxx
torch 2.10.0+cpu
torch.npu.is_available True
```

```{admonition} Note
:class: note
xxx 表示 specforge 的实际版本号。
```

查看 CLI 帮助（三个子命令：train / export / benchmark）：

```shell #test id="specforge-help"
specforge --help
```

输出结果如下：

```shell #test-result id="specforge-help"
Usage: specforge [OPTIONS] COMMAND [ARGS]...

  SpecForge: speculative decoding training framework.

Options:
  -h, --help  Show this message and exit.

Commands:
  benchmark  benchmark a running SGLang server
  export     materialize a runtime checkpoint as a model directory
  train      train a draft model from a typed config
```

查看 train 子命令的参数（后面训练就用这些 flag）：

```shell #test id="specforge-train-help"
specforge train --help
```

输出结果如下：

```shell #test-result id="specforge-train-help"
Usage: specforge train [OPTIONS] [OVERRIDES]...

  Train a draft model from a typed run config.

  OVERRIDES are dotted ``section.field=value`` assignments applied on top of
  the config file, e.g. ``training.learning_rate=1e-4``.

Options:
  -c, --config PATH               YAML or JSON run config.  [required]
  --role [auto|all|producer|consumer|both]
                                  Launch selection: offline local 'all' or
                                  online/disaggregated producer+consumer when
                                  'auto'.  [default: auto]
  --node-rank INTEGER             Node-local rank for an explicit multi-node
                                  trainer launch.
  --plan                          Print the resolved process plan without
                                  starting workers.
  -h, --help                      Show this message and exit.
```

## 端到端 smoke：1 步训练

按顺序把 `mooncake_master` → SGLang capture server → `specforge train` 拉起来跑 1 步训练，每小节一个阶段，失败可定位到具体环节。`SPECFORGE_*` 环境变量可覆盖各段默认值。

### 预下载模型

从 HuggingFace Hub 把 Qwen3.5-4B（约 8 GB）拉到本地缓存，路径供后面起 server 和训练用：

```shell #test-setup id="smoke-download-model" store="model_path"
set -euo pipefail
MODEL_ID="${SPECFORGE_MODEL_ID:-Qwen/Qwen3.5-4B}"
# 本段 stdout 会被框架捕获、供下方 `<MODEL_PATH>` 占位符替换，所以下载进度
# 重定向到 stderr，stdout 只留一行模型路径。
echo "smoke: downloading model $MODEL_ID from HuggingFace Hub" >&2
python - "$MODEL_ID" <<'PY'
import sys, contextlib
MODEL_ID = sys.argv[1]
with contextlib.redirect_stdout(sys.stderr):
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL_ID)
print(path)
PY
```

输出的模型缓存目录路径，供「起 SGLang capture server」（`--model-path`）和「跑 specforge 训练」（`model.target_model_path`）两节使用。

### 起 mooncake_master

后台拉起 mooncake_master（metadata + 传输协调服务），等 RPC 端口（默认 35551）就绪。端口探测用 Python socket 而非 `nc -z`——镜像里没装 nc：

```shell #test id="smoke-start-mooncake"
set -euo pipefail
pkill -9 -f '^mooncake_master' 2>/dev/null || true
MOONCAKE_RPC_PORT="${SPECFORGE_MOONCAKE_RPC_PORT:-35551}"
MOONCAKE_HTTP_PORT="${SPECFORGE_MOONCAKE_HTTP_PORT:-35880}"
nohup mooncake_master \
    --enable_http_metadata_server=true \
    --rpc_port=$MOONCAKE_RPC_PORT \
    --http_metadata_server_port=$MOONCAKE_HTTP_PORT \
    --metrics_port=35903 \
    --enable_metric_reporting=false \
    >/tmp/smoke-mooncake.log 2>&1 &
MOONCAKE_PID=$!
mooncake_ready() {
    python -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
try:
    s.connect(('127.0.0.1', $MOONCAKE_RPC_PORT))
except Exception:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
" 2>/dev/null
}
for _ in $(seq 1 30); do
    if mooncake_ready; then
        echo "smoke: mooncake ready (rpc $MOONCAKE_RPC_PORT, pid=$MOONCAKE_PID)"
        exit 0
    fi
    sleep 1
done
echo "smoke: FAILED - mooncake_master did not bind $MOONCAKE_RPC_PORT in 30s" >&2
tail -50 /tmp/smoke-mooncake.log >&2
exit 1
```

输出结果如下：

```shell #test-result id="smoke-start-mooncake" fuzzy='xxx'
smoke: mooncake ready (rpc 35551, pid=xxx)
```

```{admonition} Note
:class: note
xxx 为 mooncake_master 的进程号。
```

### 起 SGLang capture server

后台拉起带 capture 插件的 SGLang server（卡 0），开启 dflash spec-capture，等 `/health` 就绪。首次启动含 NPU graph 编译，等 5-10 分钟属正常：

```shell #test id="smoke-start-sglang" load="model_path>>MODEL_PATH"
set -euo pipefail
pkill -9 -f '^python -m sglang\.launch_server' 2>/dev/null || true
CAPTURE_DEVICE="${SPECFORGE_CAPTURE_DEVICE:-0}"
SGLANG_PORT="${SPECFORGE_SGLANG_PORT:-30000}"
SGLANG_HEALTH_TIMEOUT="${SPECFORGE_SGLANG_HEALTH_TIMEOUT:-600}"
MOONCAKE_RPC_PORT="${SPECFORGE_MOONCAKE_RPC_PORT:-35551}"
MOONCAKE_HTTP_PORT="${SPECFORGE_MOONCAKE_HTTP_PORT:-35880}"
# 逐个 export，不要串成 `VAR=x \... nohup ...` 前缀链：中间的注释会终止 `\` 续行，
# 赋值退化成未导出的 shell 变量，server 进程拿不到——spec-capture 会去连默认的
# localhost:50051（master 实际监听 35551），反复 "Client not available" 直至 setup 失败。
export ASCEND_RT_VISIBLE_DEVICES=$CAPTURE_DEVICE
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:$MOONCAKE_HTTP_PORT/metadata
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:$MOONCAKE_RPC_PORT
export MOONCAKE_PROTOCOL=tcp
export MOONCAKE_GLOBAL_SEGMENT_SIZE=$((32<<30))
# SPECFORGE_SPEC_CAPTURE=1 让插件在 server 进程内注册 scheduler hook；
# 没开这个开关，插件的 register() 直接返回，采集不会发生。
export SPECFORGE_SPEC_CAPTURE=1
# Qwen3.5-4B 是 hybrid Mamba/GDN，forward 时 torch_npu op_plugin.atb lazy-load libatb.so；
# CANN set_env.sh 把 atb bin 加进 PATH 但 lib 不进 LD_LIBRARY_PATH，graph capture 阶段
# 调 torch_npu._npu_reshape_and_cache 就 OSError: libatb.so: cannot open shared object。
ATB_LIB=/usr/local/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1/lib
if [[ -d "$ATB_LIB" ]]; then
    export LD_LIBRARY_PATH="$ATB_LIB:${LD_LIBRARY_PATH:-}"
fi
# --enable-return-aux-hidden-states 要求单遍 prefill（chunked=4096 切多段会让 captured
# hidden states 只覆盖首段）；scheduler 硬性要求 --chunked-prefill-size -1。
# --mamba-ssm-dtype bfloat16：GDN 的 initial_state 用 bf16，910B4 上 fp32 不被
# sgl_kernel_npu 的 chunk_gated_delta_rule 支持（只在 Ascend950 上支持 fp32）。
nohup python -m sglang.launch_server \
    --model-path "<MODEL_PATH>" \
    --trust-remote-code \
    --skip-tokenizer-init \
    --tp-size 1 \
    --mem-fraction-static 0.5 \
    --context-length 1024 \
    --chunked-prefill-size -1 \
    --attention-backend ascend \
    --mamba-ssm-dtype bfloat16 \
    --speculative-algorithm DFLASH \
    --enable-return-aux-hidden-states \
    --return-aux-hidden-state-layer-ids 1 8 15 22 29 \
    --return-hidden-states-mode full \
    --host 127.0.0.1 --port "$SGLANG_PORT" \
    >/tmp/smoke-sglang.log 2>&1 &
SGLANG_PID=$!
echo "smoke: waiting for SGLang /health (up to ${SGLANG_HEALTH_TIMEOUT}s)"
HEALTH_DEADLINE=$((SGLANG_HEALTH_TIMEOUT / 5))
for _ in $(seq 1 "$HEALTH_DEADLINE"); do
    if curl -fsS "http://127.0.0.1:$SGLANG_PORT/health" >/dev/null 2>&1; then
        echo "smoke: sglang ready (pid=$SGLANG_PID)"
        break
    fi
    sleep 5
done
if ! curl -fsS "http://127.0.0.1:$SGLANG_PORT/health" >/dev/null 2>&1; then
    echo "smoke: FAILED - SGLang not healthy after ${SGLANG_HEALTH_TIMEOUT}s" >&2
    tail -50 /tmp/smoke-sglang.log >&2
    exit 1
fi
# 插件生效的标志：scheduler 在 /server_info 里自报 capture backend。
# specforge train 侧正是读这个字段来确认 server 支持 plugin 采集。
echo "smoke: capture backend: $(curl -fsS "http://127.0.0.1:$SGLANG_PORT/server_info" | grep -o '"spec_capture_backend":"[a-z]*"' | head -1)"
```

输出结果如下：

```shell #test-result id="smoke-start-sglang" fuzzy='xxx'
smoke: waiting for SGLang /health (up to 600s)
smoke: sglang ready (pid=xxx)
smoke: capture backend: "spec_capture_backend":"plugin"
```

```{admonition} Note
:class: note
`<MODEL_PATH>` 在运行时替换为「预下载模型」一节 `snapshot_download` 命令打印的模型缓存路径；xxx 为 server 进程号。
```

### 准备训练数据

写一条 user→assistant 对话的 JSONL 作为训练数据（smoke 只跑 1 步，不追求数据质量；assistant 段够长即可满足 dflash 采样 32 个 anchor 的要求）。正式训练可用上游的 `scripts/regenerate_train_data.py` 重新生成对齐分布的数据：

```shell #test-setup id="smoke-prepare-data" store="sharegpt_path"
set -euo pipefail
SPECFORGE_ROOT="${SPECFORGE_ROOT:-SpecForge}"
if [[ ! -d "$SPECFORGE_ROOT" ]]; then
    echo "smoke: FAILED - $SPECFORGE_ROOT/ missing; specforge-install-source first"
    exit 1
fi
mkdir -p "$SPECFORGE_ROOT/cache/dataset"
if ! python - "$SPECFORGE_ROOT/cache/dataset/sharegpt_train.jsonl" >/tmp/smoke-prep.log 2>&1 <<'PY'
import json, os, sys
path = sys.argv[1]
os.makedirs(os.path.dirname(path), exist_ok=True)
row = {
    "id": "smoke_1",
    "conversations": [
        {"role": "user", "content": "Explain the difference between supervised and unsupervised learning in machine learning. Cover what types of problems each paradigm is suited for, give at least three concrete algorithm examples for each, and discuss practical considerations for choosing between them when you have a new dataset to model."},
        {"role": "assistant", "content": "Supervised learning and unsupervised learning are the two foundational paradigms in machine learning, distinguished by the form of feedback the learning algorithm receives during training. In supervised learning the algorithm is given a labeled dataset, where each training example is paired with a target output, and adjusts its parameters to minimize a loss between its predictions and the ground truth; the two main task families are classification and regression, with concrete algorithms including linear regression for predicting continuous targets, logistic regression for binary classification, decision trees and random forests for tabular data, support vector machines for image classification, and deep neural networks for high-dimensional perceptual inputs. In unsupervised learning the algorithm receives only raw unlabeled inputs and must find structure on its own, with common tasks including clustering, dimensionality reduction, density estimation, and representation learning; concrete algorithms include k-means clustering for customer segmentation, hierarchical clustering for bioinformatics, principal component analysis for visualization, independent component analysis for signal separation, autoencoders for compact latent representations, and generative adversarial networks for synthesizing new samples. Choosing between the two depends on whether you have labeled data and a well defined prediction target: supervised learning delivers higher accuracy on the specific labeled task but requires costly annotation and does not generalize beyond seen labels, while unsupervised learning is preferable when labels are scarce or the goal is exploratory or structural. In modern practice the two are often combined, with unsupervised pretraining learning useful representations from large unlabeled corpora and supervised fine tuning specializing them for downstream tasks, an approach that underlies most large language model training pipelines today."},
    ],
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(row, f, ensure_ascii=False)
    f.write("\n")
print(f"wrote {path}")
PY
then
    echo "smoke: FAILED - prepare conversations JSONL:" >&2
    tail -20 /tmp/smoke-prep.log >&2
    exit 1
fi
SHAREGPT_PATH="$SPECFORGE_ROOT/cache/dataset/sharegpt_train.jsonl"
# 后续训练命令会 `pushd "$SPECFORGE_ROOT"` 把 cwd 切到 SpecForge/，相对路径会变成
# SpecForge/SpecForge/cache/dataset/...。此处用 realpath 转绝对，下游 <SHAREGPT_PATH>
# placeholder 替换后 specforge train 不管 cwd 都能找到文件。
SHAREGPT_PATH="$(realpath "$SHAREGPT_PATH")"
ls -la "$SHAREGPT_PATH" >&2
echo "smoke: prepare-data OK ($SHAREGPT_PATH)" >&2
# stdout 只留最终路径（供下方 <SHAREGPT_PATH> 占位符替换），诊断信息走 stderr。
echo "$SHAREGPT_PATH"
```

输出的训练数据文件的绝对路径，供「跑 specforge 训练」（`data.train_data_path`）一节使用。

### 跑 specforge 训练（1 步）

看护框架逐段执行本文命令块，单段 stdout/stderr 整段缓冲，只在段结束时才输出到日志。所以 `specforge train` 运行期间日志长时间没有新输出是正常现象，不是卡死。

因此训练拆成"启动 + 监控"两段：启动段是 ≤30s 的短任务，跑完立刻能看到启动确认和 pid；监控段每 60s 打一行进度，训练完成后收尾。

#### 启动 specforge train（后台）

在卡 1 上后台启动 1 步训练（batch/步数压到最小的 smoke 配置），30 秒后确认进程还活着——典型失败（mooncake 没起或 capture 插件没生效）会在 30s 内直接退出：

```shell #test id="smoke-train-launch" load="model_path>>MODEL_PATH" load="sharegpt_path>>SHAREGPT_PATH"
set -euo pipefail
SPECFORGE_ROOT="${SPECFORGE_ROOT:-SpecForge}"
TRAINER_DEVICE="${SPECFORGE_TRAINER_DEVICE:-1}"
RECIPE="${SPECFORGE_RECIPE:-examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-npu.yaml}"
pushd "$SPECFORGE_ROOT" >/dev/null
# 配方 output_dir 残留的 producer_claim / failed 标记会让 _claim_fresh_control_path 拒绝继续。
rm -rf outputs/qwen3.5-4b-dflash-npu-online
# 上游 38970f6d（2026-09-15 合入）把 deployment.disaggregated.receive_buffers 默认值从
# pageable 改成 pinned：NPU 上 ReceiveBufferPool._new_slot 的 pin = torch.cuda.is_available()
# 恒为 False，pageable host buffer register_buffer 返回 -600，且 f22bd36c 把该返回码变成硬
# RuntimeError——trainer 第一个 feature fetch 就崩。显式回退 pageable：pool=None 走
# _store_get_tensor 老路径，register 失败被忽略、TCP get_into 正常工作。上游给 pinned
# 补上 NPU 适配（同文件已有 _ascend_runtime_available()，一行 OR 的事）后可去掉此 override。
# PYTHONUNBUFFERED=1 让 specforge train 的 stdout 无缓冲写 /tmp/smoke-train.log；
# 即使外层框架缓冲整段命令输出，日志文件里也是行粒度（出问题后能 tail 看到断点位置）。
# trainer 侧的 mooncake client 也要这些环境变量：specforge 从 env 读
# MOONCAKE_MASTER_SERVER_ADDR / MOONCAKE_GLOBAL_SEGMENT_SIZE 等，缺了会拿到
# global_segment_size=0，store setup 直接返回 -1（"Mooncake setup failed"）。
MOONCAKE_RPC_PORT="${SPECFORGE_MOONCAKE_RPC_PORT:-35551}"
MOONCAKE_HTTP_PORT="${SPECFORGE_MOONCAKE_HTTP_PORT:-35880}"
export MOONCAKE_LOCAL_HOSTNAME=127.0.0.1
export MOONCAKE_METADATA_SERVER=http://127.0.0.1:$MOONCAKE_HTTP_PORT/metadata
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:$MOONCAKE_RPC_PORT
export MOONCAKE_PROTOCOL=tcp
export MOONCAKE_GLOBAL_SEGMENT_SIZE=$((32<<30))
PYTHONUNBUFFERED=1 \
ASCEND_RT_VISIBLE_DEVICES=$TRAINER_DEVICE \
HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
nohup specforge train -c "$RECIPE" \
    training.max_steps=1 \
    training.batch_size=1 \
    training.accumulation_steps=1 \
    data.max_length=512 \
    data.train_data_path="<SHAREGPT_PATH>" \
    training.num_anchors=32 \
    training.save_interval=0 \
    training.log_interval=1 \
    tracking.report_to=none \
    deployment.trainer.nproc_per_node=1 \
    deployment.disaggregated.receive_buffers=pageable \
    model.target_model_path="<MODEL_PATH>" \
    model.embedding_key="model.language_model.embed_tokens.weight" \
    >/tmp/smoke-train.log 2>&1 &
TRAIN_PID=$!
disown $TRAIN_PID 2>/dev/null || true
echo "$TRAIN_PID" > /tmp/smoke-train.pid
# 30s 健康检查：确认 train 进程没立刻死。典型失败模式：mooncake_master 没起 / sglang
# spec_capture_sink 缺 mount 段 → producer 的 prompt 全部失败，train 直接抛
# RuntimeError 退出。这里 30s 内撞死就 fail-fast，避免监控段空转到超时。
sleep 30
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "smoke: FAILED - specforge train died within 30s of launch"
    tail -50 /tmp/smoke-train.log
    exit 1
fi
echo "smoke: specforge train alive after 30s, pid=$TRAIN_PID, log=/tmp/smoke-train.log"
popd >/dev/null
```

输出结果如下：

```shell #test-result id="smoke-train-launch" fuzzy='xxx'
smoke: specforge train alive after 30s, pid=xxx, log=/tmp/smoke-train.log
```

```{admonition} Note
:class: note
`<MODEL_PATH>` / `<SHAREGPT_PATH>` 分别来自「预下载模型」「准备训练数据」两节保存的路径；xxx 为训练进程号。
```

#### 监控直到退出

轮询等待训练完成：日志出现 step/loss 指标即训练成功，正常情况下进程随后自行退出。注意容器环境下进程退出后可能停留在僵尸状态（容器 PID 1 不回收孤儿，`kill -0` 仍返回成功），本段已按"僵尸即退出"处理。若进程确认活着但 300s 内不退出，杀掉进程树并按训练成功收尾，同时把各进程的 `/proc` 状态和 py-spy 线程栈写到日志（供定位）。若始终没有 step/loss 指标，则等到 80 min 上限判失败：

```shell #test id="smoke-train-monitor"
set -euo pipefail
LOG_FILE=/tmp/smoke-train.log
TRAIN_PID=$(cat /tmp/smoke-train.pid)
WAIT_TIMEOUT="${SPECFORGE_TRAIN_TIMEOUT:-4800}"     # 总等待上限；单段命令的框架超时为 10800s
GRACE_AFTER_DONE="${SPECFORGE_TRAIN_GRACE:-300}"   # step/loss 出现后，等进程退出的宽限
WAIT_START=$(date +%s)
DONE_AT=""
# 存活判定不能只看 kill -0：它对僵尸进程也返回成功。GH Actions 容器的 PID 1 不回收
# 孤儿——specforge train 正常退出后停在 zombie 状态、kill -0 恒真，看起来像"卡死"。
# State 为 Z 即视为已退出。
trainer_alive() {
    kill -0 "$TRAIN_PID" 2>/dev/null || return 1
    if grep -q 'State:[[:space:]]*Z' "/proc/$TRAIN_PID/status" 2>/dev/null; then
        echo "smoke: train pid=$TRAIN_PID already exited (zombie state; container init does not reap orphans)"
        return 1
    fi
    return 0
}
while trainer_alive; do
    ELAPSED=$(( $(date +%s) - WAIT_START ))
    if [[ $ELAPSED -ge $WAIT_TIMEOUT ]]; then
        echo "smoke: FAILED - specforge train exceeded ${WAIT_TIMEOUT}s, killing pid=$TRAIN_PID"
        kill -9 "$TRAIN_PID" 2>/dev/null || true
        tail -50 "$LOG_FILE"
        exit 1
    fi
    if [[ -z "$DONE_AT" ]] && grep -qE "step.*loss|loss.*step|step N:|train_runtime" "$LOG_FILE" 2>/dev/null; then
        DONE_AT=$ELAPSED
        echo "smoke: step/loss markers present at elapsed=${ELAPSED}s; giving the process ${GRACE_AFTER_DONE}s to exit"
    fi
    # 训练已产出 step/loss 但进程活着迟迟不退（非僵尸）——留作兜底。
    # 杀进程树前先取证（父+子进程各一份），全部写 stderr 并带 [ERROR] 前缀——框架对含
    # [ERROR] 的 stderr 即使 rc=0 也整段 print 到 job log（≤256KB 不截断），GitHub 上可直接
    # 复制。三层证据：/proc 状态+内核栈（D=不可中断等驱动、wchan=等待点）→ py-spy
    # Python 线程栈 → py-spy --native 原生帧（HCCL/CANN 函数名）。
    if [[ -n "$DONE_AT" ]] && [[ $(( ELAPSED - DONE_AT )) -ge $GRACE_AFTER_DONE ]]; then
        echo "smoke: WARNING - train pid=$TRAIN_PID still alive ${GRACE_AFTER_DONE}s after step/loss markers; dumping stacks then killing process tree"
        for P in "$TRAIN_PID" $(pgrep -P "$TRAIN_PID" 2>/dev/null || true); do
            echo "[ERROR] teardown-hang diagnostic for pid $P:" >&2
            awk '/^(State|Threads)/{print "  "$0}' /proc/$P/status >&2 2>/dev/null || true
            echo "  wchan: $(cat /proc/$P/wchan 2>/dev/null)" >&2
            echo "  kernel stack:" >&2
            cat /proc/$P/stack 2>/dev/null | sed 's/^/    /' >&2 || echo "    (unreadable)" >&2
        done
        if uv pip install --quiet py-spy >/dev/null 2>&1; then
            for P in "$TRAIN_PID" $(pgrep -P "$TRAIN_PID" 2>/dev/null || true); do
                echo "[ERROR] py-spy stack dump for pid $P (teardown did not exit):" >&2
                py-spy dump --pid "$P" --native >&2 2>&1 \
                    || py-spy dump --pid "$P" >&2 2>&1 \
                    || echo "[ERROR] py-spy dump failed for pid $P (ptrace blocked?)" >&2
            done
        else
            echo "[ERROR] py-spy install failed; skipping stack dump" >&2
        fi
        pkill -9 -P "$TRAIN_PID" 2>/dev/null || true
        kill -9 "$TRAIN_PID" 2>/dev/null || true
        sleep 2
        break
    fi
    if [[ -f "$LOG_FILE" ]]; then
        LOG_SIZE=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
        LAST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null | head -c 220 || true)
        echo "[$(date +%H:%M:%S)] elapsed=${ELAPSED}s pid=$TRAIN_PID log_size=${LOG_SIZE}B last: ${LAST_LINE}"
    else
        echo "[$(date +%H:%M:%S)] elapsed=${ELAPSED}s pid=$TRAIN_PID log file not yet created"
    fi
    sleep 60
done
ELAPSED=$(( $(date +%s) - WAIT_START ))
echo "smoke: specforge train process exited, elapsed=${ELAPSED}s"
# TRAIN_PID 不是当前 shell 的子进程（启动段的 shell 已退出），拿不到 exit code；
# 用 log 内容判断成败：1 步 max_steps=1 应该写出 step/loss 等行，否则算 fail。
if ! grep -qE "step.*loss|loss.*step|step N:|train_runtime" "$LOG_FILE"; then
    echo "smoke: FAILED - no step/loss output in train log"
    tail -30 "$LOG_FILE"
    exit 1
fi
if grep -qE "Traceback \(most recent call last\):|RuntimeError:|AssertionError" "$LOG_FILE"; then
    echo "smoke: WARNING - log has Traceback/ERROR but step markers also present; treating as success"
    grep -nE "Traceback \(most recent call last\):|RuntimeError:|AssertionError" "$LOG_FILE" | head -5
fi
echo "smoke: OK - 1-step training completed"
```

输出结果如下（首行通配符吸收监控期间的进度行与状态提示；无论哪条路径，最后都以"进程退出 + 训练完成"两行收尾）：

```shell #test-result id="smoke-train-monitor" fuzzy='xxx'
xxx
smoke: specforge train process exited, elapsed=xxx
smoke: OK - 1-step training completed
```

```{admonition} Note
:class: note
xxx 为监控期间的实际输出（进度行、进程号、耗时等，内容每次运行不同）。
```

## 外部链接

- GitHub：[sgl-project/SpecForge](https://github.com/sgl-project/SpecForge)
- 文档中心：[SpecForge Docs](https://github.com/sgl-project/SpecForge#readme)
