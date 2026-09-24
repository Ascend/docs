# LMCache

LMCache 是大模型推理的 KV cache 管理层，把已经算过的 KV cache 卸到 CPU 等存储里再复用。昇腾实现在 [LMCache-Ascend](https://github.com/LMCache/LMCache-Ascend)。主仓 [LMCache](https://github.com/LMCache/LMCache) 没有昇腾实现。这篇在单卡上安装 vLLM-Ascend 和 LMCache-Ascend，再用 `vllm.LLM` 做一次离线 KV 卸载。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列，Ascend **910B**。本文示例为单卡。编译会读 `npu-smi` 得到芯片型号，A2 上常见 `Ascend910B3` 或 `Ascend910B4`。没有 `npu-smi` 时再设 `SOC_VERSION`。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装，并可 `source set_env.sh` |
| ATB | Ascend Transformer Boost。vLLM EngineCore 子进程要加载 `libatb.so` |
| Python | 3.10 到 3.13，见 [vLLM-Ascend 安装文档](https://docs.vllm.ai/projects/ascend/en/latest/installation.html) |
| 编译依赖 | `cmake`、`ninja`、`libnuma-dev`，见下文安装 |
| vLLM-Ascend | 装带轮子的最新 `vllm-ascend`，`vllm` 装与之主版本号相同的版本，见下文安装 |
| LMCache | PyPI `lmcache` 的版本号等于 Release tag 去掉开头的 `v`，再编译 LMCache-Ascend |
| 模型 | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。

本文验证环境：镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`，Python 3.12。这一行记录的是本文跑通时的环境。

## 1. 加载 CANN 环境

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

输出类似：

```text
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:c1:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
```

如果 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。

### 2.2 确认工具可用

下面确认 CANN 已加载，并且 `python` 在 `PATH` 里。

```shell #test id="check-tools"
test -n "$ASCEND_HOME_PATH"
python --version
```

<!--
完整输出较长，其中应包含：

```shell #test-result id="check-tools"
Python 3...
```
-->

## 3. 安装编译依赖

安装 `cmake`、`ninja` 和 NUMA 头文件。

<!--
```shell #test-setup
if getent hosts cache-service.nginx-pypi-cache.svc.cluster.local >/dev/null 2>&1 \
    && [ -f /etc/apt/sources.list ]; then
  sed -Ei 's@(ports|archive).ubuntu.com@cache-service.nginx-pypi-cache.svc.cluster.local:8081@g' /etc/apt/sources.list
fi
```
-->

```shell #test id="install-system-prereqs"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y cmake ninja-build libnuma-dev
cmake --version
```

<!--
完整输出较长，其中应包含：

```shell #test-result id="install-system-prereqs"
...
cmake version 3...
```
-->

## 4. 安装 vLLM-Ascend

先装带轮子的最新 `vllm-ascend`，再以 `--no-deps` 安装主版本号相同的 `vllm`，最后用 `--force-reinstall --no-deps` 重装同一版 `triton-ascend`。安装写法见 [vLLM-Ascend 安装文档](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)。

```shell #test id="install-vllm"
python -m pip install --prefer-binary \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  vllm-ascend
va=$(python -c "import importlib.metadata as m; print(m.version('vllm-ascend').split('.post')[0])")
python -m pip install --retries 3 --no-deps "vllm==$va"
ta=$(python -c "import importlib.metadata as m; print(m.version('triton-ascend'))")
python -m pip install --force-reinstall --no-deps \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  "triton-ascend==$ta"
python -c "import importlib.metadata as m
for n in ['torch', 'torch-npu', 'vllm', 'vllm-ascend']:
    print(n, m.version(n))"
```

<!--
完整输出较长，其中应包含：

```shell #test-result id="install-vllm"
...
torch ...
torch-npu ...
vllm ...
vllm-ascend ...
```
-->

## 5. 安装 lmcache

将 `<ver>` 换成最新 Release 的 tag 去掉开头的 `v`。

<!--
```shell #test-setup store="lmcache_ver"
echo "${UPSTREAM_REF#v}"
```
-->

```shell #test id="install-lmcache" load="lmcache_ver>>ver"
NO_CUDA_EXT=1 python -m pip install --no-build-isolation lmcache==<ver> --no-deps
python -m pip install \
  aiofile aiofiles blake3 aiohttp msgspec numpy psutil pyyaml pyzmq \
  redis safetensors sortedcontainers transformers huggingface_hub \
  prometheus_client py-cpuinfo
python -c "import importlib.metadata as m; print('lmcache', m.version('lmcache'))"
```

完整输出较长，其中应包含：

```shell #test-result id="install-lmcache" load="lmcache_ver>>ver"
...
lmcache <ver>
```

> `<ver>` 是最新 Release 去掉开头的 `v`。撰写时最新 Release 是 `v0.4.4`，对应 `lmcache==0.4.4`。

## 6. 克隆并编译 LMCache-Ascend

将 `<ref>` 换成最新 Release 的 tag。主仓在 GitHub。子模块和主仓分开拉。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="clone-lmcache-ascend" load="upstream_ref>>ref"
if [ ! -f LMCache-Ascend/csrc/hixl/CMakeLists.txt ]; then
  rm -rf LMCache-Ascend
  for _ in 1 2 3; do
    GIT_TERMINAL_PROMPT=0 GIT_HTTP_VERSION=HTTP/1.1 \
      git clone --depth 1 --branch <ref> https://github.com/LMCache/LMCache-Ascend.git LMCache-Ascend && break
    rm -rf LMCache-Ascend
    sleep 5
  done
fi
GIT_TERMINAL_PROMPT=0 GIT_HTTP_VERSION=HTTP/1.1 \
  git -C LMCache-Ascend submodule update --init --recursive
git -C LMCache-Ascend describe --tags --exact-match
```

完整输出较长，其中应包含：

```shell #test-result id="clone-lmcache-ascend" load="upstream_ref>>ref"
...
<ref>
```

> `<ref>` 是最新 Release 的 tag。撰写时为 `v0.4.4`。

运行下面的脚本，再编译。脚本给 HIXL 的编译补上 `pkg_inc` 头文件目录，并把 `LMCacheAscendConnectorV1Dynamic.__init__` 改成透传 `*args, **kwargs`。源码里已经是这两种形态时，脚本不改文件。

```python #test id="patch-lmcache-ascend"
import re
from pathlib import Path

cmake = Path("LMCache-Ascend/csrc/hixl/CMakeLists.txt")
text = cmake.read_text()
runtime = "${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/pkg_inc/runtime"
pkg = "${ASCEND_CANN_PACKAGE_PATH}/${ARCH_SUBDIR}/pkg_inc"
if not re.search(r"/pkg_inc\s*$", text, re.M):
    if runtime not in text:
        raise SystemExit("hixl cmake include line not found")
    cmake.write_text(text.replace(runtime, runtime + "\n    " + pkg, 1))

conn = Path(
    "LMCache-Ascend/lmcache_ascend/integration/vllm/"
    "lmcache_ascend_connector_v1.py"
)
text = conn.read_text()
marker = "class LMCacheAscendConnectorV1Dynamic(LMCacheConnectorV1Dynamic):"
if marker not in text:
    raise SystemExit("connector class not found")
has_forwarder = "def __init__(self, *args, **kwargs)" in text
has_override = "def __init__" in text
if has_override and not has_forwarder:
    init_pat = re.compile(
        r"    def __init__\(self,.*?\) -> None:\n"
        r"        super\(\)\.__init__\(.*?\)\n",
        re.S,
    )
    forwarder = (
        "    def __init__(self, *args, **kwargs) -> None:\n"
        "        super().__init__(*args, **kwargs)\n"
    )
    new_text, n = init_pat.subn(forwarder, text, count=1)
    if n == 0:
        raise SystemExit("connector __init__ not found")
    conn.write_text(new_text)
print("patched_ok", True)
```

输出结果如下：

```shell #test-result id="patch-lmcache-ascend"
patched_ok True
```

删除已有的 `LMCache-Ascend/build`，再在当前目录安装 `./LMCache-Ascend`。

```shell #test id="install-lmcache-ascend"
rm -rf LMCache-Ascend/build
python -m pip install -v --no-build-isolation -e ./LMCache-Ascend
python -c "import lmcache, lmcache_ascend, torch, torch_npu; from lmcache_ascend import _build_info as b; import lmcache_ascend.c_ops; print('lmcache', lmcache.__version__); print('soc', b.__soc_version__); print('c_ops_ok', True); print('npu_available', torch.npu.is_available())"
```

完整输出较长，其中应包含：

```shell #test-result id="install-lmcache-ascend" load="lmcache_ver>>ver"
...
lmcache <ver>
soc Ascend910B...
c_ops_ok True
npu_available True
```

`soc` 应和本机 `npu-smi info -t board` 的 Chip Name 一致。如果编译缺少 `numaif.h`，回到第 3 节执行 `apt-get install`，装上 `cmake`、`ninja-build` 和 `libnuma-dev`，然后从本节重新执行补丁和编译，再继续第 7 节。

## 7. 用离线 LLM 做一次 KV 卸载

用 `vllm.LLM` 做一次离线 KV 卸载。模型是 Hugging Face 上的 Qwen/Qwen2.5-0.5B-Instruct。连接器与上游 `examples/offload.py` 相同。

| 参数 | 含义 |
| --- | --- |
| `temperature` | `0`，固定本次生成 |
| `top_p` | `0.95` |
| `max_tokens` | `8` |
| `max_model_len` | `512` |
| `gpu_memory_utilization` | `0.4` |

运行下面的脚本。

```python #test id="offload"
import os
import sys

os.environ["PYTHONHASHSEED"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
os.dup2(sys.stdout.fileno(), sys.stderr.fileno())

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
import lmcache_ascend
import torch
import torch_npu
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


def main():
    os.environ.setdefault("LMCACHE_CHUNK_SIZE", "256")
    os.environ.setdefault("LMCACHE_LOCAL_CPU", "True")
    os.environ.setdefault("LMCACHE_MAX_LOCAL_CPU_SIZE", "2")

    model = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")
    ktc = KVTransferConfig(
        kv_connector="LMCacheAscendConnectorV1Dynamic",
        kv_role="kv_both",
        kv_connector_module_path=(
            "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
        ),
    )
    llm = LLM(
        model=model,
        enforce_eager=True,
        kv_transfer_config=ktc,
        max_model_len=512,
        gpu_memory_utilization=0.4,
        trust_remote_code=True,
    )
    params = SamplingParams(temperature=0, top_p=0.95, max_tokens=8)
    outputs = llm.generate(
        ["Hello, my name is", "Tell me a short story"],
        params,
    )
    for item in outputs:
        print("reply", item.outputs[0].text)
    print("current_device", f"npu:{torch.npu.current_device()}")
    print("npu_available", torch.npu.is_available())
    print("workload_ok", True)
    LMCacheEngineBuilder.destroy(ENGINE_NAME)


if __name__ == "__main__":
    main()
```

完整输出较长，其中应包含：

```text #test-result id="offload"
...Platform plugin ascend is activated...Using NPU for LMCache engine...
current_device npu:0
npu_available True
workload_ok True
...
```

脚本会打印两条 `reply`。结束时若出现 `Assertion failed: pfd.revents`，退出码为 0 且上面几行都在即可。

## 8. 更多文档

存储后端、框架对接和多机部署与社区相同。见 [LMCache 文档](https://docs.lmcache.ai/)。昇腾侧仓库见 [LMCache-Ascend](https://github.com/LMCache/LMCache-Ascend)。
