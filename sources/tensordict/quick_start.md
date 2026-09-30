# TensorDict

在单卡昇腾 NPU 上跑通 [tensordict](https://github.com/pytorch/tensordict) 的核心特性链：逐节验证 `TensorDict` 在 NPU 上的 13 个核心入口：

- **基础构造**：在 NPU 上直接构造 `TensorDict`，元数据访问（device / dtype / shape / numel）。
- **字典语义与嵌套**：dict 接口（`td[key]` / `set` / `del`），以及 `next` 嵌套结构。
- **类张量运算**：`+` / `*` 逐元素算子、整数下标、`unsqueeze` / `squeeze`。
- **上下文管理器**：`lock_` / `unlock_` 保护共享数据。
- **分布式与状态字典**：`isend` / `irecv` / `send` / `recv` 接口，`TensorDictParams` + `TensorDict.from_module`。
- **函数式前向**：`torch.func.functional_call` + `TensorDictParams`。
- **持久化与预处理**：`.memmap` 落盘、`load_memmap` 重载、`td.apply` 逐叶子预处理。
- **懒分配与 tensorclass**：`make_tensordict` 直接从 dict 构造，`@tensorclass` dataclass 风格。

tensordict 通过 `torch_npu` 间接支持昇腾 NPU：底层 `TensorDict` 的全部操作建立在 `torch.Tensor` 之上，`torch_npu` 把这些算子正确路由到 NPU 上，tensordict 自身不需要额外的 NPU 适配层。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档之前，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，32 GB × 1）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| tensordict | 最新 release 的源码/二进制 |

## 前置安装

确认能看到 NPU 设备：

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
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
+===========================+===============+====================================================+
```

```{admonition} Note
:class: note
如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。
```

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

```{admonition} Note
:class: note
xxx 表示 Python 的补丁版本号。
```

检查 torch / torch_npu 是否装好且 NPU 设备可用(下面的命令用 Python 执行)：

```python #test id="check-torch"
import torch, torch_npu
print('torch=', torch.__version__)
print('torch_npu=', torch_npu.__version__)
print('is_available:', torch.npu.is_available())
print('count:', torch.npu.device_count())
```

输出结果如下：

```shell #test-result id="check-torch"
torch= 2.9.0+cpu
torch_npu= 2.9.0.post2
is_available: True
count: 1
```

```{admonition} Note
:class: note
如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。
```

## 安装 tensordict

tensordict 同时支持 PyPI 二进制安装与 GitHub 源码安装，两条路径都把核心模块（`TensorDict` / `TensorDictBase` / `LazyStackedTensorDict` / `MemoryMappedTensor` / `tensorclass` 等）一起打包。

### 使用 uv 进行安装

```shell #test id="tensordict-install-binary"
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple tensordict
python -c "import tensordict; print('tensordict', tensordict.__version__)"
```

输出结果类似如下：

```shell #test-result id="tensordict-install-binary" fuzzy='xxx'
tensordict xxx
```

```{admonition} Note
:class: note
xxx 表示实际安装的 tensordict 版本号。
```

### 从源码安装

<!--
```shell #test-setup
uv pip uninstall tensordict -y
```
-->

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="tensordict-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/pytorch/tensordict.git
cd tensordict
uv pip install -e . --config-settings editable_mode=compat
python -c "import tensordict; print('tensordict', tensordict.__version__)"
```

```{admonition} Note
:class: note
`<ref>` 替换为 tensordict 当前的最新 release 版本。
```

输出结果类似如下：

```shell #test-result id="tensordict-install-source" fuzzy='xxx'
tensordict xxx
```

```{admonition} Note
:class: note
xxx 表示实际安装的 tensordict 版本号。
```

## 核心特性验证

### 1. Basic usage — 在 NPU 上构造 TensorDict

`TensorDict(source={...}, batch_size=[N], device='npu:0')` 直接构造在 NPU 上，外层 batch 维决定后续 batch 索引(下面的命令用 Python 执行)：

```python #test id="td-basic"
import torch, torch_npu
from tensordict import TensorDict
td = TensorDict(source={'obs': torch.zeros(3, 4), 'action': torch.zeros(3, 2)}, batch_size=[3], device='npu:0')
print('type', type(td).__name__)
print('batch_size', td.batch_size)
print('shape', td.shape)
print('keys', sorted(td.keys()))
```

输出结果如下：

```shell #test-result id="td-basic"
type TensorDict
batch_size torch.Size([3])
shape torch.Size([3])
keys ['action', 'obs']
```

### 2. TensorDict's Metadata — 设备 / dtype / 形状

TensorDict 是 `Tensor` 的"超集"，自带的元数据包括 `device` / `dtype` / `shape` / `ndim` / `numel`，对顶层 batch 维生效（`numel()` 等于 batch_size 的元素积）(下面的命令用 Python 执行)：

```python #test id="td-metadata"
import torch, torch_npu
from tensordict import TensorDict
td = TensorDict(source={'obs': torch.zeros(3, 4), 'action': torch.zeros(3, 2)}, batch_size=[3], device='npu:0')
print('device', td.device)
print('dtype', td['obs'].dtype)
print('shape', td['obs'].shape)
print('td_ndim', td.ndim)
print('td_numel', td.numel())
print('obs_numel', td['obs'].numel())
```

输出结果如下：

```shell #test-result id="td-metadata"
device npu:0
dtype torch.float32
shape torch.Size([3, 4])
td_ndim 1
td_numel 3
obs_numel 12
```

### 3. TensorDict as a specialized dictionary — 字典语义

TensorDict 是 dict 的超集：`td[key]` 取值、`'k' in td` 判断键存在、`td.get('k')` 取值（缺键返 `None`）、`td.set('k', v)` 原地写入、`del td['k']` 删除键——这些操作都和 NPU 上的张量配合，不复制底层数据(下面的命令用 Python 执行)：

```python #test id="td-dict"
import torch, torch_npu
from tensordict import TensorDict
td = TensorDict(source={'obs': torch.zeros(3, 4), 'action': torch.zeros(3, 2)}, batch_size=[3], device='npu:0')
print('has next', 'next' in td)
print('get next', td.get('next'))
print('get next default', td.get('next', 'missing'))
td.set('reward', torch.ones(3, device='npu:0'))
print('reward', td['reward'])
print('keys after set', sorted(td.keys()))
del td['obs']
print('keys after del', sorted(td.keys()))
```

输出结果如下：

```shell #test-result id="td-dict"
has next False
get next None
get next default missing
reward tensor([1., 1., 1.], device='npu:0')
keys after set ['action', 'obs', 'reward']
keys after del ['action', 'reward']
```

### 4. Nesting TensorDicts — 嵌套 TensorDict

把另一个 batch 维兼容的 TensorDict 作为 `next` 字段塞进外层 TensorDict，即可表达时序/层级结构（典型用法：RL transition 里的 `obs / action / next.obs`）。内层 TensorDict 自身也保留 `batch_size` / 索引 / `device`(下面的命令用 Python 执行)：

```python #test id="td-nesting"
import torch, torch_npu
from tensordict import TensorDict
td = TensorDict(source={'obs': torch.zeros(3, 4), 'action': torch.zeros(3, 2)}, batch_size=[3], device='npu:0')
nested = TensorDict(source={'reward': torch.ones(3), 'done': torch.zeros(3)}, batch_size=[3], device='npu:0')
td.set('next', nested)
print('outer keys', sorted(td.keys()))
print('next type', type(td['next']).__name__)
print('next batch_size', td['next'].batch_size)
print('next reward sum', td['next']['reward'].sum().item())
```

输出结果如下：

```shell #test-result id="td-nesting"
outer keys ['action', 'next', 'obs']
next type TensorDict
next batch_size torch.Size([3])
next reward sum 3.0
```

### 5. Tensor-like features — 类张量运算

TensorDict 支持 `+` / `*` 等逐元素算子（作用到每个叶子张量）、`.shape`、整数下标（`td[i]` 取出第 i 个 batch）、`.unsqueeze(dim)` / `.squeeze(dim)` 在 batch 维上插入/压缩长度为 1 的轴(下面的命令用 Python 执行)：

```python #test id="td-tensor-like"
import torch, torch_npu
from tensordict import TensorDict
td = TensorDict(source={'obs': torch.ones(3, 4)}, batch_size=[3], device='npu:0')
print('td.shape', td.shape)
print('td + 1 obs sum', (td + 1)['obs'].sum().item())
print('td[0] obs', td[0]['obs'])
print('unsqueeze shape', td.unsqueeze(0).shape)
print('squeeze shape', td.unsqueeze(0).squeeze(0).shape)
```

输出结果如下：

```shell #test-result id="td-tensor-like"
td.shape torch.Size([3])
td + 1 obs sum 24.0
td[0] obs tensor([1., 1., 1., 1.], device='npu:0')
unsqueeze shape torch.Size([1, 3])
squeeze shape torch.Size([3])
```

### 6. TensorDicts as context managers — lock / unlock

`td.lock_()` 把 TensorDict 锁住，阻止后续意外写入；需要写入时用 `with td.unlock_():` 上下文管理器临时解锁，操作完成后再自动回到 locked 状态（避免误改共享数据）。在 unlock 期间调用 `td.set('k', v)` 即可新增键(下面的命令用 Python 执行)：

```python #test id="td-lock"
import torch, torch_npu
from tensordict import TensorDict
td = TensorDict(source={'obs': torch.zeros(3, 4)}, batch_size=[3], device='npu:0')
td.lock_()
print('locked', td.is_locked)
with td.unlock_():
    td.set('reward', torch.ones(3, device='npu:0'))
print('reward sum', td['reward'].sum().item())
print('locked after', td.is_locked)
```

输出结果如下：

```shell #test-result id="td-lock"
locked True
reward sum 3.0
locked after True
```

### 7. Distributed capabilities — 分布式点对点接口

TensorDict 把 `torch.distributed` 的点对点收发原语直接挂在实例方法上：`td.isend(dst=...)` / `td.irecv(src=...)` 是异步版，`td.send(dst=...)` / `td.recv(src=...)` 是阻塞版。NPU 上对应把后端切到 `hccl`。本节只校验 API 表面可用，不真的启动 `torch.distributed`(下面的命令用 Python 执行)：

```python #test id="td-dist"
import torch, torch_npu
from tensordict import TensorDict
td = TensorDict(source={'obs': torch.zeros(3, 4)}, batch_size=[3], device='npu:0')
print('isend callable', callable(td.isend))
print('irecv callable', callable(td.irecv))
print('send callable', callable(td.send))
print('recv callable', callable(td.recv))
print('numel', td.numel())
```

输出结果如下：

```shell #test-result id="td-dist"
isend callable True
irecv callable True
send callable True
recv callable True
numel 3
```

### 8. TensorDict to represent state-dicts — state_dict 表示

`tensordict.TensorDictParams` 把 `nn.Module.state_dict()` 包成带类型语义的容器（可作为参数容器传递给优化器等）；`TensorDict.from_module(model)` 反向从 `nn.Module` 一次性导出所有参数为 TensorDict(下面的命令用 Python 执行)：

```python #test id="td-state-dict"
import torch, torch_npu
import torch.nn as nn
from tensordict import TensorDict, TensorDictParams
linear = nn.Linear(3, 4).to('npu:0')
sd = linear.state_dict()
print('sd keys', sorted(sd.keys()))
params = TensorDictParams(sd)
print('params type', type(params).__name__)
print('params keys', sorted(params.keys()))
td = TensorDict.from_module(linear)
print('from_module keys', sorted(td.keys()))
print('from_module weight shape', td['weight'].shape)
```

输出结果如下：

```shell #test-result id="td-state-dict"
sd keys ['bias', 'weight']
params type TensorDictParams
params keys ['bias', 'weight']
from_module keys ['bias', 'weight']
from_module weight shape torch.Size([4, 3])
```

### 9. TensorDict for functional programming — 函数式前向

`make_functional` / `make_functional_with_buffers` 在 tensordict `0.14.x` 已移除；当前用 `torch.func.functional_call(module, params_dict, x)` + `TensorDictParams` 组合，把 `TensorDictParams` 通过 `dict(...)` 转成普通 dict 喂给 functional_call(下面的命令用 Python 执行)：

```python #test id="td-functional"
import torch, torch_npu
import torch.nn as nn
from tensordict import TensorDictParams
torch.manual_seed(42)
torch.npu.manual_seed(42)
linear = nn.Linear(3, 4).to('npu:0')
params = TensorDictParams(linear.state_dict())
x = torch.randn(2, 3, device='npu:0')
out = torch.func.functional_call(linear, dict(params), x)
print('out shape', out.shape)
print('out sum', out.sum().item())
```

输出结果如下：

```shell #test-result id="td-functional"
out shape torch.Size([2, 4])
out sum 1.3248364925384521
```

### 10. TensorDict for parameter serialization and building datasets — memmap 落盘

`.memmap(prefix=path)` 把 TensorDict 的每个张量整体落盘成 memory-mapped 文件（外加一份 `meta.json` 记录 shape / dtype / 键顺序）；`load_memmap(prefix=path)` 反向加载，落盘文件路径约定是 `prefix/<key>.memmap`。这套机制可直接当 on-disk dataset 用(下面的命令用 Python 执行)：

```python #test id="td-memmap"
import tempfile, os
import torch, torch_npu
from tensordict import TensorDict, load_memmap
td = TensorDict({'obs': torch.zeros(3, 4), 'action': torch.zeros(3, 2)}, batch_size=[3], device='npu:0').set('reward', torch.arange(3, dtype=torch.float32, device='npu:0'))
with tempfile.TemporaryDirectory() as td_dir:
    prefix = os.path.join(td_dir, 'data')
    td.memmap(prefix=prefix)
    files = sorted(os.listdir(td_dir))
    print('files', files)
    loaded = load_memmap(prefix=prefix)
    print('loaded keys', sorted(loaded.keys()))
    print('loaded batch', loaded.batch_size)
    print('loaded reward', loaded['reward'].tolist())
```

输出结果如下：

```shell #test-result id="td-memmap"
files ['data']
loaded keys ['action', 'obs', 'reward']
loaded batch torch.Size([3])
loaded reward [0.0, 1.0, 2.0]
```

### 11. Preprocessing with TensorDict.map — 逐叶子预处理

`td.map(fn, num_workers=N)` 把 fn 作用到 TensorDict 的每个叶子张量上，支持多进程并行（要求 fn 是可 pickle 的顶层 callable）。本节用 `td.apply(fn)` 做最小验证——`apply` 和 `map` 的逐叶子语义一致，区别仅在 `map` 默认走多进程(下面的命令用 Python 执行)：

```python #test id="td-map"
import torch, torch_npu
from tensordict import TensorDict
td = TensorDict(source={'x': torch.zeros(3, 4), 'y': torch.zeros(3, 4)}, batch_size=[3], device='npu:0')
td['x'].fill_(1)
td['y'].fill_(2)
out = td.apply(lambda x: x + 1)
print('x sum', out['x'].sum().item())
print('y sum', out['y'].sum().item())
```

输出结果如下：

```shell #test-result id="td-map"
x sum 24.0
y sum 36.0
```

### 12. Lazy preallocation — make_tensordict 懒分配

`make_tensordict(dict, batch_size=[N], device='npu:0')` 直接从 `dict` 构造 TensorDict——每个 value 必须已经是 shape 兼容 batch 维的 tensor（不允许只写 shape 占位）。这是构造 TensorDict 的另一种常用入口(下面的命令用 Python 执行)：

```python #test id="td-make-td"
import torch, torch_npu
from tensordict import make_tensordict
td = make_tensordict({'a': torch.zeros(3, 4), 'b': torch.zeros(3, 2)}, batch_size=[3], device='npu:0')
print('batch_size', td.batch_size)
print('keys', sorted(td.keys()))
print('a shape', td['a'].shape)
print('b shape', td['b'].shape)
print('a sum', td['a'].sum().item())
```

输出结果如下：

```shell #test-result id="td-make-td"
batch_size torch.Size([3])
keys ['a', 'b']
a shape torch.Size([3, 4])
b shape torch.Size([3, 2])
a sum 0.0
```

### 13. TensorClass — dataclass 风格的 TensorDict

`@tensorclass` 装饰器把 `dataclass` 风格的类自动转成 TensorDict 兼容容器（用 `batch_size=[N]` 在构造时一次性 autobatch），类属性访问 `obj.x` 等价于 `obj['x']`，也能直接 `.to('npu:0')` / 索引(下面的命令用 Python 执行)：

```python #test id="td-tensorclass"
import torch, torch_npu
from tensordict import tensorclass

@tensorclass
class Data:
    x: torch.Tensor
    y: torch.Tensor

obj = Data(x=torch.ones(3, 4), y=torch.zeros(3, 4), batch_size=[3], device='npu:0')
print('batch_size', obj.batch_size)
print('device', obj.device)
print('x sum', obj.x.sum().item())
print('y sum', obj.y.sum().item())
```

输出结果如下：

```shell #test-result id="td-tensorclass"
batch_size torch.Size([3])
device npu:0
x sum 12.0
y sum 0.0
```

## 外部链接

- GitHub：[pytorch/tensordict](https://github.com/pytorch/tensordict)
- 文档中心：[TensorDict Docs](https://pytorch.org/tensordict/stable/index.html)