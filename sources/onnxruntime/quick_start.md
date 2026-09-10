# onnxruntime

本文在单卡昇腾上从 [ONNX Runtime](https://github.com/microsoft/onnxruntime) 当前正式 Release 源码编译 `onnxruntime-cann`，当场生成一个最小加法模型，并用 [CANN Execution Provider](https://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html) 做一次推理。包名是 `onnxruntime-cann`，导入名仍是 `onnxruntime`；不要再装一份 CPU 包 `onnxruntime`，两个包会抢同一个导入名。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动已安装，并能 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| Python | 3.12 |
| 编译 | gcc-12、g++-12、cmake 3.28 到 3.31、ninja、git |
| 包管理 | `python -m pip` |

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 装好 CANN 与驱动。推荐配套镜像：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| onnxruntime-cann | 当前 GitHub 正式 Release 源码编译 |
| numpy | 1.26.x，必须 `<2` |

## 1. 加载 CANN 环境

常见容器里 `npu-smi` 位于 `/usr/local/sbin` 或 `/usr/local/bin`。先把这两个目录放进 `PATH`，再加载 CANN：

```shell
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
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

## 3. 安装编译工具

Ubuntu 22.04 默认 gcc 11 编不了 aarch64 上的 ONNX Runtime，需要 gcc-12。cmake 需要 3.28 及以上且不要装到 4；`python -m pip` 装的 cmake 要放到 `PATH` 前面，否则会继续用系统自带的 3.22。

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
if ! command -v gcc-12 >/dev/null || ! command -v g++-12 >/dev/null || ! command -v ninja >/dev/null; then
  apt-get update
  apt-get install -y gcc-12 g++-12 ninja-build git
fi
python -m pip install --retries 3 'cmake>=3.28,<4' 'numpy<2' packaging wheel setuptools
export PATH="$(python -c 'import sysconfig; print(sysconfig.get_path("scripts"))'):$PATH"
hash -r
gcc-12 --version | head -n 1
cmake --version | head -n 1
```

输出结果如下：

```shell #test-result id="toolchain"
...gcc-12...
cmake version 3.3...
```

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

输出结果如下：

```shell #test-result id="clone"
/root/onnxruntime-qs/onnxruntime/build.sh
```

## 5. 编译 onnxruntime-cann

开启 CANN Execution Provider 并打出 Python wheel。工作目录里如果已有 `dist/onnxruntime_cann-*.whl`，这一步会跳过编译；首次编译可能要几十分钟，并行路数封顶 32。

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

输出结果如下：

```shell #test-result id="compile"
...onnxruntime_cann-...whl
```

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

把刚编出的 `onnxruntime-cann` 装进当前 Python。`onnx` 用来当场生成模型；这个 wheel 按 NumPy 1.x 编译，不钉 `numpy<2` 时 `import onnxruntime` 会失败。CANN 编译算子还要用 `decorator`、`scipy`、`attrs`、`psutil`、`sympy`，不装的话第一次 `sess.run()` 会报 `aclgrphBuildInitialize` 或 `aclopCompileAndExecute`。

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

输出结果如下：

```shell #test-result id="install"
...onnxruntime-cann ...
```

## 7. 确认昇腾后端

下面列出当前 Python 里已注册的 Execution Provider。没有 `CANNExecutionProvider` 时不要继续推理，先看文末常见问题。

```shell #test id="providers"
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

输出结果如下：

```shell #test-result id="providers"
...CANNExecutionProvider...
```

列表里有 CPU 并不等于这次推理会走 CPU。真正决定后端的是下一节创建 `InferenceSession` 时传入的 `providers`。

## 8. 造一个最小 ONNX 模型

用已安装的 `onnx` 写一个两向量相加的图，生成工作目录里的 `add_model.onnx`。

保存为 `/root/onnxruntime-qs/make_add_model.py`：

```python
import onnx
from onnx import TensorProto, helper

x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2])
graph = helper.make_graph(
    [helper.make_node("Add", ["X", "Y"], ["Z"])],
    "add",
    [x, y],
    [z],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.save(model, "add_model.onnx")
print("wrote add_model.onnx")
```

<!--
```shell #test-setup
mkdir -p /root/onnxruntime-qs
cat > /root/onnxruntime-qs/make_add_model.py <<'PY'
import onnx
from onnx import TensorProto, helper

x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2])
graph = helper.make_graph(
    [helper.make_node("Add", ["X", "Y"], ["Z"])],
    "add",
    [x, y],
    [z],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.save(model, "add_model.onnx")
print("wrote add_model.onnx")
PY
```
-->

```shell #test id="make-model"
cd /root/onnxruntime-qs
python make_add_model.py
```

输出结果如下：

```shell #test-result id="make-model"
wrote add_model.onnx
```

## 9. 用昇腾跑第一次推理

ONNX Runtime 默认 `enable_fallback=1`：CANN 会话创建失败时会静默改在 CPU 上重建会话，加法结果仍然正确。这里只注册 `CANNExecutionProvider`，并同时关掉会话级和 Python 级回退。

输入是 `[1.0, 2.0]` 和 `[3.0, 4.0]`，昇腾上的加法结果应是 `[4.0, 6.0]`。

保存为 `/root/onnxruntime-qs/infer_add.py`：

```python
import numpy as np
import onnxruntime as ort

so = ort.SessionOptions()
so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
sess = ort.InferenceSession(
    "add_model.onnx",
    sess_options=so,
    providers=["CANNExecutionProvider"],
    enable_fallback=False,
)
providers = sess.get_providers()
print(providers)
assert providers[0] == "CANNExecutionProvider", providers
x = np.array([1.0, 2.0], dtype=np.float32)
y = np.array([3.0, 4.0], dtype=np.float32)
out = sess.run(None, {"X": x, "Y": y})[0]
print("result", [float(v) for v in out])
```

<!--
```shell #test-setup
mkdir -p /root/onnxruntime-qs
cat > /root/onnxruntime-qs/infer_add.py <<'PY'
import numpy as np
import onnxruntime as ort

so = ort.SessionOptions()
so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
sess = ort.InferenceSession(
    "add_model.onnx",
    sess_options=so,
    providers=["CANNExecutionProvider"],
    enable_fallback=False,
)
providers = sess.get_providers()
print(providers)
assert providers[0] == "CANNExecutionProvider", providers
x = np.array([1.0, 2.0], dtype=np.float32)
y = np.array([3.0, 4.0], dtype=np.float32)
out = sess.run(None, {"X": x, "Y": y})[0]
print("result", [float(v) for v in out])
PY
```
-->

```shell #test id="infer"
cd /root/onnxruntime-qs
python infer_add.py
```

输出结果如下。第一项必须是 `CANNExecutionProvider`。

```shell #test-result id="infer"
...CANNExecutionProvider...
result [4.0, 6.0]
```
