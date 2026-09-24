# onnxruntime

ONNX Runtime 是跨平台的机器学习模型推理引擎，同一份 ONNX 模型可以换硬件后端执行。本文在单卡昇腾上，从 [ONNX Runtime](https://github.com/microsoft/onnxruntime) 当前正式 Release 源码编译带 [CANN Execution Provider](https://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html) 的 `onnxruntime-cann`，生成一个两向量相加的模型，并完成一次推理。安装包名是 `onnxruntime-cann`，Python 导入名是 `onnxruntime`。

## 前置条件

### 硬件

Atlas 800T、900 A2 训练系列，Ascend 910B。本文示例为单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动已安装，并能 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。版本按 [昇腾软件配套清单](https://www.hiascend.com/developer/download/compatibility) 选择 |
| Python | 落在上面这份配套清单给出的范围内 |
| 编译 | gcc-12、g++-12、cmake 3.28 及以上且低于 4、ninja、git |
| 包管理 | `python -m pip` |
| NumPy | `numpy<2` |

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 装好 CANN 与驱动。

### 本文验证环境

本文在配套镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12` 上验证，镜像内为 CANN 9.1.0 与 Python 3.12。`onnxruntime-cann` 由当前 GitHub 正式 Release 源码编译，NumPy 使用 1.26。这不是唯一支持组合。

## 1. 加载 CANN 环境

把 `npu-smi` 所在目录放进 `PATH`，并加载 CANN。

```shell
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
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
| 5     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 5                                                            |
+===========================+===============+====================================================+
```

如果 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。

### 2.2 确认工具可用

确认 CANN 已加载，并查看 Python 版本。

```shell #test id="check-tools"
test -n "$ASCEND_HOME_PATH"
python --version
```

<!--
```shell #test-result id="check-tools"
Python ...
```
-->

## 3. 安装编译工具

安装 gcc-12、g++-12、ninja、git，以及 cmake 与 NumPy。

<!--
```shell #test-setup
if getent hosts cache-service.nginx-pypi-cache.svc.cluster.local >/dev/null 2>&1 \
    && [ -f /etc/apt/sources.list ]; then
  sed -Ei 's@(ports|archive).ubuntu.com@cache-service.nginx-pypi-cache.svc.cluster.local:8081@g' /etc/apt/sources.list
fi
```
-->

```shell #test id="toolchain"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y gcc-12 g++-12 ninja-build git
python -m pip install --retries 3 'cmake>=3.28,<4' 'numpy<2' packaging wheel setuptools
export PATH="$(python -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):$PATH"
hash -r
gcc-12 --version | head -n 1
cmake --version | head -n 1
```

<!--
```shell #test-result id="toolchain"
...gcc-12...
cmake version 3.3...
```
-->

## 4. 获取源码

工作目录为 `/root/onnxruntime-qs`。把 `<UPSTREAM_REF>` 换成 [Releases](https://github.com/microsoft/onnxruntime/releases) 里当前正式 tag。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

<!--
```shell #test-setup load="upstream_ref>>UPSTREAM_REF"
wd=/root/onnxruntime-qs
ci=/root/.cache/huggingface/onnxruntime
ref='<UPSTREAM_REF>'
mkdir -p "$wd/dist"
for w in "$ci/wheels/$ref"/onnxruntime_cann-*.whl; do
  [ -f "$w" ] || continue
  if python -m zipfile -l "$w" >/dev/null 2>&1; then
    cp -a "$w" "$wd/dist/"
  else
    rm -f "$w"
  fi
done
if [ -d "$ci/src/$ref/.git" ] && [ ! -d "$wd/onnxruntime/.git" ]; then
  rm -rf "$wd/onnxruntime"
  git clone --depth 1 "$ci/src/$ref" "$wd/onnxruntime"
fi
```
-->

```shell #test id="clone" load="upstream_ref>>UPSTREAM_REF"
mkdir -p /root/onnxruntime-qs
if [ ! -d /root/onnxruntime-qs/onnxruntime/.git ]; then
  GIT_TERMINAL_PROMPT=0 GIT_HTTP_VERSION=HTTP/1.1 git clone --depth 1 --branch "<UPSTREAM_REF>" \
    https://github.com/microsoft/onnxruntime.git /root/onnxruntime-qs/onnxruntime
fi
ls /root/onnxruntime-qs/onnxruntime/build.sh
```

<!--
```shell #test-result id="clone"
.../root/onnxruntime-qs/onnxruntime/build.sh
```
-->

## 5. 编译 onnxruntime-cann

编译带 CANN Execution Provider 的 Python wheel。并行路数取 CPU 核数，最多 32。工作目录里已经有 wheel 时，跳过编译。

<!--
```shell #test-setup
wd=/root/onnxruntime-qs
ci=/root/.cache/huggingface/onnxruntime
if [ -d "$ci/cmake-mirror" ] && [ -d "$wd/onnxruntime" ]; then
  ln -sfn "$ci/cmake-mirror" "$wd/onnxruntime/.cmake-mirror"
fi
```
-->

```shell #test id="compile"
cd /root/onnxruntime-qs
if ! compgen -G "dist/onnxruntime_cann-*.whl" >/dev/null; then
  cd onnxruntime
  export CC=gcc-12 CXX=g++-12
  njobs=$(nproc)
  if [ "$njobs" -gt 32 ]; then
    njobs=32
  fi
  MIRROR=()
  if [ -d .cmake-mirror ]; then
    MIRROR+=(--cmake_deps_mirror_dir "$PWD/.cmake-mirror")
  fi
  ./build.sh --config Release --build_shared_lib --use_cann --build_wheel \
    --parallel "$njobs" --skip_tests --skip_submodule_sync \
    --compile_no_warning_as_error --allow_running_as_root \
    --cmake_generator Ninja \
    --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
    "${MIRROR[@]}"
  mkdir -p /root/onnxruntime-qs/dist
  cp -a build/Linux/Release/dist/onnxruntime_cann-*.whl /root/onnxruntime-qs/dist/
fi
ls /root/onnxruntime-qs/dist/onnxruntime_cann-*.whl
```

<!--
```shell #test-result id="compile"
...onnxruntime_cann-...whl
```
-->

<!--
```shell #test-setup load="upstream_ref>>UPSTREAM_REF"
wd=/root/onnxruntime-qs
ci=/root/.cache/huggingface/onnxruntime
ref='<UPSTREAM_REF>'
mkdir -p "$ci/wheels/$ref"
for w in "$wd/dist"/onnxruntime_cann-*.whl; do
  [ -f "$w" ] || continue
  base=$(basename "$w")
  dest="$ci/wheels/$ref/$base"
  if [ -f "$dest" ]; then
    continue
  fi
  cp -a "$w" "${dest}.part"
  mv "${dest}.part" "$dest"
done
if [ -d "$wd/onnxruntime/.git" ] && [ ! -d "$ci/src/$ref/.git" ]; then
  mkdir -p "$ci/src"
  rm -rf "$ci/src/${ref}.part"
  git clone --depth 1 "$wd/onnxruntime" "$ci/src/${ref}.part"
  mv "$ci/src/${ref}.part" "$ci/src/$ref"
fi
```
-->

## 6. 安装 wheel 与算子编译依赖

安装编出的 `onnxruntime-cann`，以及 `onnx`、`numpy`、`decorator`、`scipy`、`attrs`、`psutil`、`sympy`。

```shell #test id="install"
python -m pip install --retries 3 \
    /root/onnxruntime-qs/dist/onnxruntime_cann-*.whl \
    onnx \
    'numpy<2' \
    decorator \
    'scipy>=1.11,<1.15' \
    attrs \
    psutil \
    sympy
python -c "from importlib.metadata import version; print('onnxruntime-cann', version('onnxruntime-cann'))"
```

<!--
```shell #test-result id="install"
...onnxruntime-cann ...
```
-->

## 7. 确认昇腾后端

列出当前 Python 里已注册的 Execution Provider。

```shell #test id="providers"
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

<!--
```shell #test-result id="providers"
...CANNExecutionProvider...
```
-->

## 8. 造一个最小 ONNX 模型

用已安装的 `onnx` 写一个两向量相加的图，并保存到 `/root/onnxruntime-qs/add_model.onnx`。

```python #test id="make-model"
import os

import onnx
from onnx import TensorProto, helper

os.makedirs("/root/onnxruntime-qs", exist_ok=True)
x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2])
graph = helper.make_graph(
    [helper.make_node("Add", ["X", "Y"], ["Z"])],
    "add",
    [x, y],
    [z],
)
model = helper.make_model(
    graph,
    ir_version=13,
    opset_imports=[helper.make_opsetid("", 13)],
)
onnx.checker.check_model(model)
path = "/root/onnxruntime-qs/add_model.onnx"
onnx.save(model, path)
node = model.graph.node[0]
print("check ok")
print(
    "model", model.graph.name,
    "node", node.op_type,
    "inputs", node.input[0], node.input[1],
    "output", node.output[0],
)
print("file", path)
```

输出结果如下：

```text #test-result id="make-model"
check ok
model add node Add inputs X Y output Z
file /root/onnxruntime-qs/add_model.onnx
```

## 9. 用昇腾跑第一次推理

用 `CANNExecutionProvider` 读取 `add_model.onnx`，计算 `[1.0, 2.0]` 与 `[3.0, 4.0]` 的和。

```python #test id="infer"
import numpy as np
import onnxruntime as ort

so = ort.SessionOptions()
so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
sess = ort.InferenceSession(
    "/root/onnxruntime-qs/add_model.onnx",
    sess_options=so,
    providers=["CANNExecutionProvider"],
    enable_fallback=False,
)
x = np.array([1.0, 2.0], dtype=np.float32)
y = np.array([3.0, 4.0], dtype=np.float32)
out = sess.run(None, {"X": x, "Y": y})[0]
print("provider", sess.get_providers()[0])
print("add [1.0, 2.0] + [3.0, 4.0] =", [float(v) for v in out])
```

完整输出较长，其中应包含：

```text #test-result id="infer"
...provider CANNExecutionProvider
add [1.0, 2.0] + [3.0, 4.0] = [4.0, 6.0]
...
```

## 10. 更多文档

模型格式、Execution Provider 选项和其余模块与上游社区相同。

- CANN 后端说明：[CANN Execution Provider](https://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html)
- 上游文档：[ONNX Runtime 文档](https://onnxruntime.ai/docs/)
