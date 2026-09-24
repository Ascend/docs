# Ray

使用 Ray 将 Task 和 Actor 分别调度到两张昇腾 NPU，并在各自的进程中
运行 `torch_npu` 张量计算。通过本示例，可以了解 Ray 如何发现 NPU、
分配设备以及隔离任务。更多调度方式参见 Ray 的
[Accelerator Support](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html) 文档。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品，至少有两张可用的 Ascend NPU，并已完成
物理机或容器中的设备与驱动配置。

### 基础软件

在运行本文档之前，机器上需要已经安装并可用：

- Linux aarch64 操作系统；
- 可用的 Python 环境；
- 可用的 CANN toolkit 和驱动；
- 与 CANN 匹配的 `torch` 和 `torch_npu`，且
  `torch.npu.is_available() == True`；
- `npu-smi` 能正常显示 NPU 设备。

CANN 与 PyTorch-NPU 的安装和版本匹配可参考
[Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)。

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 2）；
- **操作系统**：Ubuntu 22.04。

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| Ray | 当前最新 release，Linux aarch64 wheel |
| NPU | Ascend 910B4 × 2 |

### 检查前置是否满足

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```text #test-result id="check-py" fuzzy="xxx"
Python 3.12.xxx
```

检查 Torch、Torch-NPU 和 NPU 设备：

```python #test id="check-torch"
import torch
import torch_npu

print("torch=", torch.__version__)
print("torch_npu=", torch_npu.__version__)
print("is_available:", torch.npu.is_available())
print("count:", torch.npu.device_count())
```

输出结果如下：

```text #test-result id="check-torch"
torch= 2.9.0+cpu
torch_npu= 2.9.0.post2
is_available: True
count: 2
```

确认 `npu-smi` 可以看到两张 NPU：

```shell
npu-smi info
```

如果 `npu-smi` 不存在或 `import torch_npu` 失败，请先修复驱动、CANN、
Torch 与 Torch-NPU 的版本匹配问题。

## 安装 Ray

安装当前最新 Ray release，并验证安装结果：

```shell #test id="ray-install"
python -m pip install -q -U "ray[default]"
python -c "import ray; print('ray', ray.__version__)"
```

输出结果如下：

```text #test-result id="ray-install" fuzzy="xxx"
ray xxx
```

## 查看 Ray 检测到的 NPU

启动 Ray 后，查看集群中注册的 `NPU` 资源数量。下面的环境应显示两张 NPU。

```python #test id="ray-detects-npus"
import ray

ray.init(include_dashboard=False, log_to_driver=False)
resources = ray.cluster_resources()
count = int(resources.get("NPU", 0))
assert count == 2, resources
print("Ray NPU resources:", count)
ray.shutdown()
```

输出结果如下：

```text #test-result id="ray-detects-npus" fuzzy="xxx"
Ray NPU resources: 2
```

## 在 Task 和 Actor 中使用 NPU

下面分别启动一个 Actor 和一个 Task，各请求一张 NPU。Ray 会将分配的设备
ID 写入 `ASCEND_RT_VISIBLE_DEVICES`；两个进程各自在可见的 NPU 上计算。

```python #test id="ray-isolates-npus"
import os
import ray

ray.init(include_dashboard=False, log_to_driver=False)

def assigned_npu():
    import torch
    import torch_npu

    ids = [str(value) for value in ray.get_runtime_context().get_accelerator_ids()["NPU"]]
    assert len(ids) == 1, ids
    visible = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
    assert visible == ids[0], (visible, ids)
    value = torch.ones(4, device="npu:0").sum().cpu().item()
    return ids[0], value

@ray.remote(resources={"NPU": 1})
class NPUActor:
    def ping(self):
        return assigned_npu()

@ray.remote(resources={"NPU": 1})
def npu_task():
    return assigned_npu()

actor = NPUActor.remote()
actor_result = ray.get(actor.ping.remote())
task_result = ray.get(npu_task.remote())
ids = sorted([actor_result[0], task_result[0]])
values = sorted([actor_result[1], task_result[1]])
assert ids == ["0", "1"], ids
assert values == [4.0, 4.0], values
print("assigned NPU IDs:", ",".join(ids))
print("tensor sums:", ",".join(str(value) for value in values))
ray.kill(actor)
ray.shutdown()
```

输出结果如下：

```text #test-result id="ray-isolates-npus"
assigned NPU IDs: 0,1
tensor sums: 4.0,4.0
```

Ray 的 NPU 是逻辑资源。`resources={"NPU": 0.25}` 这样的请求允许多个
任务共享一张 NPU，但不会提供显存或算力隔离，应用仍需自行保证共享安全。
