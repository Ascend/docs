# LMCache

在单卡昇腾上安装 vLLM-Ascend，编译 [LMCache-Ascend](https://github.com/LMCache/LMCache-Ascend)，再用 `vllm.LLM` 运行 KV 卸载连接器。主仓 [LMCache/LMCache](https://github.com/LMCache/LMCache) 没有昇腾实现，昇腾侧在同组织的 **LMCache-Ascend**。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为单卡。编译 LMCache-Ascend 时把 `SOC_VERSION` 设成 `Ascend910B4`。

### 软件


| 类别          | 要求                                                                              |
| ----------- | ------------------------------------------------------------------------------- |
| CANN        | toolkit + 驱动固件已安装，并可 `source set_env.sh`                                        |
| ATB         | Ascend Transformer Boost。vLLM EngineCore 子进程要加载 `libatb.so`                     |
| Python      | 3.12                                                                            |
| 编译依赖        | `cmake`、`ninja`、`libnuma-dev`，见下文安装                                             |
| vLLM-Ascend | `vllm` / `vllm-ascend` 均为 `0.23.0`，见下文安装                                        |
| LMCache     | PyPI `lmcache` 的版本号等于 Release tag 去掉开头的 `v`，再编译 LMCache-Ascend                  |
| 模型          | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |


阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。推荐配套镜像：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

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



## 3. 安装编译依赖

LMCache-Ascend 的 C++ 插件需要 `cmake`、`ninja` 和 NUMA 头文件。没有就装上。

<!--
```shell #test-setup
if getent hosts cache-service.nginx-pypi-cache.svc.cluster.local >/dev/null 2>&1 \
    && [ -f /etc/apt/sources.list ]; then
  sed -Ei 's@(ports|archive).ubuntu.com@cache-service.nginx-pypi-cache.svc.cluster.local:8081@g' /etc/apt/sources.list
fi
```
-->

```shell #test id="install-system-prereqs"
if ! command -v cmake >/dev/null 2>&1 \
    || ! command -v ninja >/dev/null 2>&1 \
    || [ ! -f /usr/include/numaif.h ]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y cmake ninja-build libnuma-dev
fi
cmake --version
```

输出结果如下：

```shell #test-result id="install-system-prereqs"
...
cmake version 3...
```

## 4. 安装 vLLM-Ascend

分三步安装，不要合成一次 `pip install`。最后一步必须 `--force-reinstall --no-deps`，否则会留下社区 CUDA 版 Triton，第一次推理报错。

```shell #test id="install-vllm"
python -m pip install --retries 3 vllm==0.23.0
python -m pip install \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  vllm-ascend==0.23.0
python -m pip install --force-reinstall --no-deps \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  triton-ascend==3.2.2
python -c "import importlib.metadata as m
for n in ['torch', 'torch-npu', 'vllm', 'vllm-ascend']:
    print(n, m.version(n))"
```

输出结果如下：

```shell #test-result id="install-vllm"
...
torch 2.10.0...
torch-npu 2.10.0.post4
vllm 0.23.0...
vllm-ascend 0.23.0
```



## 5. 安装 lmcache

将 `<ver>` 换成目标 tag 去掉开头的 `v`。撰写时最新 Release 是 `v0.4.4`，对应 `lmcache==0.4.4`。

<!--
```shell #test-setup store="lmcache_ver"
echo "${UPSTREAM_REF#v}"
```
-->

```shell #test id="install-lmcache" load="lmcache_ver>>ver"
NO_CUDA_EXT=1 python -m pip install --no-build-isolation lmcache==<ver> --no-deps
python -m pip install \
  aiofile aiofiles blake3 aiohttp msgspec numpy psutil pyyaml pyzmq \
  redis safetensors sortedcontainers transformers huggingface_hub
python -c "import importlib.metadata as m; print('lmcache', m.version('lmcache'))"
```

输出结果如下：

```shell #test-result id="install-lmcache" load="lmcache_ver>>ver"
...
lmcache <ver>
```



## 6. 克隆并编译 LMCache-Ascend

克隆 release tag（`<ref>` 可改为例如 `v0.4.4`）。主仓在 GitHub；子模块和主仓分开拉。若 `LMCache-Ascend/csrc/hixl/CMakeLists.txt` 已经在，就跳过 clone。

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

输出结果如下：

```shell #test-result id="clone-lmcache-ascend" load="upstream_ref>>ref"
...
<ref>
```

最新 Release `v0.4.4` 在 CANN 9.1 上编 HIXL 会缺 `runtime/rt_external_device.h`，在 vLLM 0.23 上会拒两参数 KV connector；克隆后先跑下面的补丁再编译。源码里已经有对应内容时脚本会跳过。

保存为 `patch_lmcache_ascend.py`：

```python
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
if "kv_cache_config" not in text:
    old_init = (
        "class LMCacheAscendConnectorV1Dynamic(LMCacheConnectorV1Dynamic):\n"
        "    def __init__(self, vllm_config: \"VllmConfig\", role: KVConnectorRole) -> None:\n"
        "        super().__init__(vllm_config=vllm_config, role=role)\n"
    )
    new_init = (
        "class LMCacheAscendConnectorV1Dynamic(LMCacheConnectorV1Dynamic):\n"
        "    def __init__(\n"
        "        self,\n"
        "        vllm_config: \"VllmConfig\",\n"
        "        role: KVConnectorRole,\n"
        "        kv_cache_config=None,\n"
        "    ) -> None:\n"
        "        super().__init__(\n"
        "            vllm_config=vllm_config,\n"
        "            role=role,\n"
        "            kv_cache_config=kv_cache_config,\n"
        "        )\n"
    )
    if old_init not in text:
        raise SystemExit("connector __init__ not found")
    conn.write_text(text.replace(old_init, new_init, 1))
print("patched_ok", True)
```

<!--
```shell #test-setup
cat > patch_lmcache_ascend.py <<'PY'
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
if "kv_cache_config" not in text:
    old_init = (
        "class LMCacheAscendConnectorV1Dynamic(LMCacheConnectorV1Dynamic):\n"
        "    def __init__(self, vllm_config: \"VllmConfig\", role: KVConnectorRole) -> None:\n"
        "        super().__init__(vllm_config=vllm_config, role=role)\n"
    )
    new_init = (
        "class LMCacheAscendConnectorV1Dynamic(LMCacheConnectorV1Dynamic):\n"
        "    def __init__(\n"
        "        self,\n"
        "        vllm_config: \"VllmConfig\",\n"
        "        role: KVConnectorRole,\n"
        "        kv_cache_config=None,\n"
        "    ) -> None:\n"
        "        super().__init__(\n"
        "            vllm_config=vllm_config,\n"
        "            role=role,\n"
        "            kv_cache_config=kv_cache_config,\n"
        "        )\n"
    )
    if old_init not in text:
        raise SystemExit("connector __init__ not found")
    conn.write_text(text.replace(old_init, new_init, 1))
print("patched_ok", True)
PY
```
-->

编译前删掉 `LMCache-Ascend/build`。CANN 把昇腾核目标链接进 `.o` 时会原地改文件，留着上次的产物再编，链接器会报 `unknown file type`。用 `-e ./LMCache-Ascend` 安装，不要 `cd` 进克隆目录。

```shell #test id="install-lmcache-ascend"
python patch_lmcache_ascend.py
rm -rf LMCache-Ascend/build
SOC_VERSION=Ascend910B4 python -m pip install -v --no-build-isolation -e ./LMCache-Ascend
python -c "import lmcache, lmcache_ascend, torch, torch_npu; from lmcache_ascend import _build_info as b; import lmcache_ascend.c_ops; print('lmcache', lmcache.__version__); print('soc', b.__soc_version__); print('c_ops_ok', True); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-lmcache-ascend" load="lmcache_ver>>ver"
...
patched_ok True
...
lmcache <ver>
soc Ascend910B4
c_ops_ok True
npu_available True
```

`soc` 必须是编译时设的 `Ascend910B4`。缺 `numaif.h` 时先回到第 3 节。

## 7. 用离线 LLM 做一次 KV 卸载

第一次跑通不要开 `vllm serve`，服务进程不会自己退出。下面用和上游 `examples/offload.py` 同一套连接器，换成 Hugging Face 上的 Qwen2.5-0.5B-Instruct，并把 `max_model_len` 收到 512、`gpu_memory_utilization` 收到 0.4，方便单卡几分钟内结束。

`VLLM_WORKER_MULTIPROC_METHOD=spawn` 避免父进程初始化过 NPU 之后子进程再 `fork`。`PYTHONHASHSEED=0` 让 LMCache 跨进程的 token hash 稳定。`VLLM_LOGGING_LEVEL=INFO` 让昇腾插件和 LMCache 引擎的设备日志打出来。脚本必须先 `import vllm`，再 `import lmcache_ascend`，后者只在检测到 vLLM 已经加载时才会把设备检测换成 NPU 版。末尾 `2>&1` 把打在 stderr 的设备日志并进标准输出。

保存为 `offload_qs.py`：

```python
import os

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
    llm.generate(["Hello, my name is", "Tell me a short story"], params)
    print("current_device", f"npu:{torch.npu.current_device()}")
    print("npu_available", torch.npu.is_available())
    print("workload_ok", True)
    LMCacheEngineBuilder.destroy(ENGINE_NAME)


if __name__ == "__main__":
    main()
```

<!--
```shell #test-setup
cat > offload_qs.py <<'PY'
import os

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
    llm.generate(["Hello, my name is", "Tell me a short story"], params)
    print("current_device", f"npu:{torch.npu.current_device()}")
    print("npu_available", torch.npu.is_available())
    print("workload_ok", True)
    LMCacheEngineBuilder.destroy(ENGINE_NAME)


if __name__ == "__main__":
    main()
PY
```
-->

```shell #test id="offload"
export PYTHONHASHSEED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=INFO
python offload_qs.py 2>&1
```

输出结果如下：

```shell #test-result id="offload"
...Platform plugin ascend is activated...Using NPU for LMCache engine...
current_device npu:0
npu_available True
workload_ok True
...
```

生成文字每次可能不同。`current_device npu:0` 和 `Using NPU for LMCache engine` 才是这一次上了昇腾、并且 LMCache 引擎也在 NPU 上的证据。结束时 ZMQ 可能打一条 `Assertion failed: pfd.revents`，只要退出码是 0、上面几行都在，可以忽略。