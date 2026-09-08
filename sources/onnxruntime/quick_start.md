# onnxruntime

本文在单卡昇腾上安装 [ONNX Runtime](https://github.com/microsoft/onnxruntime) 的 `onnxruntime-cann`，当场生成一个最小加法模型，并用 [CANN Execution Provider](https://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html) 在昇腾上进行推理。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装，并且可以 `source set_env.sh` |
| Python | 3.12 |

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。

**本文验证过的版本**

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| onnxruntime-cann | 1.24.4 |
| onnx | 1.22.0 |
| numpy | 1.26.x（必须 `<2`） |

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。常见容器里 `npu-smi` 在 `/usr/local/sbin` 或 `/usr/local/bin`，需要把这些目录加入 `PATH`。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
```

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

下面确认驱动能看到设备。表格中的功耗、HBM 占用每次不同，不必与样例逐字一致。

```shell
npu-smi info
```

如果 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。

### 2.2 确认工具可用

下面确认 CANN 已加载，并且 `npu-smi` 与 `python` 都在 `PATH` 里。

```shell #test id="check-tools"
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
python --version
```

输出结果如下：

```shell #test-result id="check-tools"
...
Python 3.12...
```

## 3. 安装 onnxruntime-cann

安装钉死版本的 `onnxruntime-cann`，并带上生成模型用的 `onnx`、必须 `<2` 的 `numpy`，以及 CANN 编译算子要用的 `decorator`、`scipy`、`attrs`、`psutil`。`scipy` 钉在 1.15 以下，避免把 NumPy 升到 2。

```shell #test id="install"
python -m pip install --retries 3 \
    onnxruntime-cann==1.24.4 \
    onnx==1.22.0 \
    'numpy<2' \
    decorator \
    'scipy>=1.11,<1.15' \
    attrs \
    psutil
python -c "from importlib.metadata import version; print('onnxruntime-cann', version('onnxruntime-cann')); print('onnx', version('onnx'))"
```

输出结果如下：

```shell #test-result id="install"
...onnxruntime-cann 1.24.4
onnx 1.22.0
```

## 4. 确认昇腾后端

下面列出当前 Python 里已注册的 Execution Provider。没有 `CANNExecutionProvider` 时不要继续推理，先看文末「常见问题」。

```shell #test id="providers"
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

输出结果如下（列表里必须有 `CANNExecutionProvider`，前后还可能有 `CPUExecutionProvider` 等）：

```shell #test-result id="providers"
...CANNExecutionProvider...
```

列表里有 CPU 并不等于这次推理会走 CPU。真正决定后端的是下一节创建 `InferenceSession` 时传入的 `providers`。

## 5. 造一个最小 ONNX 模型

用已安装的 `onnx` 在当前目录写一个两向量相加的图，保存为相对路径 `add_model.onnx`。不下载任何文件。

保存为 `make_add_model.py`：

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
cat > make_add_model.py <<'PY'
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
python make_add_model.py
```

输出结果如下：

```shell #test-result id="make-model"
wrote add_model.onnx
```

## 6. 用昇腾跑第一次推理

ONNX Runtime 默认 `enable_fallback=1`：CANN 会话创建失败时，它会静默改在 CPU 上重建会话，加法结果仍然正确，进程退出码也是 0。这里只注册 `CANNExecutionProvider`，并同时关掉会话级和 Python 级回退。装错包、CANN 没加载、版本对不上时，进程必须失败。

输入是 `[1.0, 2.0]` 和 `[3.0, 4.0]`，昇腾上的加法结果应是 `[4.0, 6.0]`。模型文件是当前目录下的 `add_model.onnx`。

保存为 `infer_add.py`：

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
cat > infer_add.py <<'PY'
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
python infer_add.py
```

输出结果如下。第一项必须是 `CANNExecutionProvider`。列表里是否还出现 CPU，以本机实际打印为准。

```shell #test-result id="infer"
...CANNExecutionProvider...
result [4.0, 6.0]
```
