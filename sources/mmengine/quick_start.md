# mmengine

在单卡昇腾 NPU 上跑通 [MMEngine](https://github.com/open-mmlab/mmengine) 的核心能力：

- **零代码上 NPU**：自动识别昇腾设备，模型放置、训练、混合精度都无需写设备相关代码；
- **一份配置驱动训练**：纯 Python 配置支持继承与覆写，dataloader、优化器、学习率调度、评估、检查点全部由 `Runner` 自动串起；
- **混合精度与断点续训**：一行配置开启 AMP，检查点保存 / 恢复开箱即用；
- **NPU 性能分析**：训练中一键开启 CANN profiler。

MMEngine 本身**不含模型**，只提供「引擎」。本文用一个 16 维二分类的玩具 MLP + 合成高斯数据集跑通全链路，不需要下载任何模型权重，全部数据本地生成、结果可复现。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品（Ascend 910B4 / 910B 等），并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

MMEngine 的核心包**不声明** `torch` 依赖（它只是一层框架外壳），但推理/训练时跑 forward 与反向都要用到，所以必须提前就绪。

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 1）
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
| mmengine | 最新 release |
| opencv-python-headless | 5.0.0 |

## 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

输出如下类似结果：

```
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
```

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py"
Python 3.12...
```

安装 `torch` / `torch_npu`：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.9.0
uv pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi torch_npu==2.9.0.post2
```

检查 torch / torch_npu 是否装好且 NPU 设备可用（下面代码用 python 运行）：

```python #test id="check-torch"
import torch
import torch_npu

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

## 安装 MMEngine

### 从源码安装（可选）

<!-- 工作流注入的 UPSTREAM_REF（最新 release tag）通过这个隐藏的 #test-setup 捕获并注入到下方 install 命令中-->
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到最新 release tag，安装并且验证。

```shell #test-setup load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/open-mmlab/mmengine.git mmengine-src
cd mmengine-src
printf 'setuptools<81\n' | uv pip install -e . --build-constraints -
```

```{admonition} Note
:class: note
`<ref>` 为安装的最新的 release tag。
```

验证装上的版本：

```shell #test id="mmengine-src-version"
python -c "import mmengine; print('mmengine', mmengine.__version__)"
```

输出结果类似如下：

```shell #test-result id="mmengine-src-version" fuzzy='xxx'
mmengine xxx
```

```{admonition} Note
:class: note
xxx 表示最新的版本号。
```

### 使用 pip 进行安装

从 PyPI 安装并验证版本：

<!-- 若前面执行过「从源码安装」，先卸掉 editable 安装，让 PyPI 的正式 release 干净接管 -->
<!--
```shell #test-setup
uv pip uninstall mmengine
```
-->

```shell #test id="mmengine-version"
uv pip install mmengine
python -c "import mmengine; print('mmengine', mmengine.__version__)"
```

输出结果类似如下：

```shell #test-result id="mmengine-version" fuzzy='xxx'
mmengine xxx
```

```{admonition} Note
:class: note
xxx 表示最新的版本号。
```

### 补齐 opencv 的运行库依赖（无图形界面的环境必做）

MMEngine 的 `mmengine.hooks` 包在**导入时**就硬依赖 `cv2`，而 mmengine 0.10.x 在 PyPI 上声明的依赖是**带 GUI 的** `opencv-python`，需要补上它动态链接的 X11 / GL 运行库：

```shell #test id="mmengine-opencv"
apt-get update -qq >/dev/null
apt-get install -y -qq libxcb1 libgl1 libglib2.0-0 >/dev/null
python -c "import cv2; print('cv2', cv2.__version__)"
```

输出结果如下：

```shell #test-result id="mmengine-opencv"
cv2 ...
```

```{admonition} Note
:class: note
`...` 表示 opencv 的版本号。
```

### CLI 自检：`python -c "import mmengine"`

MMEngine 没有独立 CLI，但可以用一行命令确认安装到位：

```shell #test id="mmengine-selfcheck"
python -c "import mmengine, mmengine.runner, mmengine.hooks, mmengine.visualization; print('mmengine subpackages ok')"
```

输出结果如下：

```shell #test-result id="mmengine-selfcheck"
mmengine subpackages ok
```

## 准备示例工程

下面的例子统一跑在一个「玩具工程」上：模型 / 数据 / 指标 / Hook 各一个文件，配置拆成 base + child 两级，全部放在 `/tmp/mmengine_quickstart` 下。

先创建模型、数据集、指标与自定义 Hook（下面代码用 shell 运行）：

```shell #test-setup
mkdir -p /tmp/mmengine_quickstart
cat > /tmp/mmengine_quickstart/toy_model.py <<'EOF'
import torch
import torch.nn as nn

from mmengine.model import BaseModel
from mmengine.registry import MODELS


@MODELS.register_module()
class TinyMLP(BaseModel):
    """A tiny MLP classifier used as the running example."""

    def __init__(self, in_dim=16, hidden_dim=32, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, data_samples=None, mode='tensor'):
        feats = torch.relu(self.fc1(inputs))
        logits = self.fc2(feats)
        if mode == 'loss':
            labels = torch.tensor(
                [int(s['label']) for s in data_samples], device=logits.device)
            return {'loss': self.criterion(logits, labels)}
        if mode == 'predict':
            preds = logits.argmax(dim=1).cpu().tolist()
            return [{
                'pred': int(p),
                'label': int(s['label'])
            } for p, s in zip(preds, data_samples)]
        return logits
EOF
cat > /tmp/mmengine_quickstart/toy_data.py <<'EOF'
import torch
from torch.utils.data import Dataset

from mmengine.evaluator import BaseMetric
from mmengine.registry import DATASETS, FUNCTIONS, METRICS, TRANSFORMS


@FUNCTIONS.register_module()
def toy_collate(batch):
    """Stack the feature tensors, keep the data samples as a list."""
    return {
        'inputs': torch.stack([item['inputs'] for item in batch]),
        'data_samples': [item['data_samples'] for item in batch]
    }


@TRANSFORMS.register_module()
class ToToyFeature:
    """A trivial transform: turn the annotation into a feature vector."""

    def __init__(self, dim=4):
        self.dim = dim

    def __call__(self, results):
        results['feature'] = torch.full((self.dim, ), float(results['label']))
        return results


@DATASETS.register_module()
class GaussianDataset(Dataset):
    """Two-cluster synthetic dataset, fully deterministic."""

    def __init__(self, num_samples=64, in_dim=16, seed=0):
        self.num_samples = num_samples
        self.in_dim = in_dim
        generator = torch.Generator().manual_seed(seed)
        labels = torch.randint(0, 2, (num_samples, ), generator=generator)
        centers = torch.tensor([[-1.0] * in_dim, [1.0] * in_dim])
        self.features = centers[labels] + 0.1 * torch.randn(
            num_samples, in_dim, generator=generator)
        self.labels = labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'inputs': self.features[idx],
            'data_samples': {
                'label': int(self.labels[idx])
            }
        }


@METRICS.register_module()
class Accuracy(BaseMetric):
    """Top-1 accuracy over the whole validation set."""

    default_prefix = 'accuracy'

    def process(self, data_batch, data_samples):
        correct = sum(int(s['pred'] == s['label']) for s in data_samples)
        self.results.append({
            'correct': correct,
            'total': len(data_samples)
        })

    def compute_metrics(self, results):
        correct = sum(r['correct'] for r in results)
        total = sum(r['total'] for r in results)
        return {'accuracy': correct / total}
EOF
cat > /tmp/mmengine_quickstart/toy_hook.py <<'EOF'
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class IterationCounterHook(Hook):
    """Count how many iterations / epochs the runner went through."""

    priority = 'LOW'

    def __init__(self):
        self.num_iters = 0
        self.num_epochs = 0

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self.num_iters += 1

    def after_train_epoch(self, runner):
        self.num_epochs += 1
EOF
```

再创建两级配置：base 配置放公共部分，child 配置通过 `_base_` 继承并覆写模型与训练轮数（下面代码用 shell 运行）：

```shell #test-setup
cat > /tmp/mmengine_quickstart/base_config.py <<'EOF'
custom_imports = dict(
    imports=['toy_data', 'toy_model', 'toy_hook'],
    allow_failed_imports=False)

work_dir = '/tmp/mmengine_quickstart/work_dir'

lr = 0.01

train_dataloader = dict(
    batch_size=16,
    dataset=dict(type='GaussianDataset', num_samples=64, in_dim=16, seed=0),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='toy_collate'))

val_dataloader = dict(
    batch_size=16,
    dataset=dict(type='GaussianDataset', num_samples=32, in_dim=16, seed=1),
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='toy_collate'))

test_dataloader = dict(
    batch_size=16,
    dataset=dict(type='GaussianDataset', num_samples=32, in_dim=16, seed=2),
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='toy_collate'))

optim_wrapper = dict(optimizer=dict(type='AdamW', lr=lr))

val_evaluator = dict(type='Accuracy')
test_evaluator = dict(type='Accuracy')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=2),
    checkpoint=dict(type='CheckpointHook', interval=1))

randomness = dict(seed=0)
EOF
cat > /tmp/mmengine_quickstart/config.py <<'EOF'
_base_ = ['./base_config.py']

model = dict(type='TinyMLP', in_dim=16, hidden_dim=32, num_classes=2)

# 变量插值：{{_base_.lr}} 会被子配置解析成 base 里的 lr
optim_wrapper = dict(optimizer=dict(type='AdamW', lr={{_base_.lr}}))

train_cfg = dict(by_epoch=True, max_epochs=2, val_interval=1)
val_cfg = dict()
test_cfg = dict()

custom_hooks = [dict(type='IterationCounterHook')]
EOF
```

## NPU 设备抽象

`mmengine.device` 是 MMEngine 感知昇腾的入口。`import torch_npu` 之后，`get_device()` 会返回 `npu`，`is_npu_available()` 返回 `True`（下面代码用 python 运行）：

```python #test id="mmengine-device"
import torch
import torch_npu

from mmengine.device import get_device, is_npu_available
from mmengine.runner import amp

print('get_device:', get_device())
print('is_npu_available:', is_npu_available())
print('npu_count:', torch.npu.device_count())

linear = torch.nn.Linear(8, 2).to('npu')
with amp.autocast(device_type='npu', dtype=torch.float16):
    x = torch.randn(4, 8, device='npu')
    y = linear(x)
print('autocast_dtype:', y.dtype)
print('autocast_device:', y.device)
```

输出结果如下：

```shell #test-result id="mmengine-device"
get_device: npu
is_npu_available: True
npu_count: 1
autocast_dtype: torch.float16
autocast_device: npu:0
```

## 配置系统

MMEngine 的配置就是普通的 Python 文件：公共部分放在 base 配置里，子配置用 `_base_` 继承后只写差异项，还可以用 `{{_base_.lr}}` 直接引用 base 里的变量。下面加载示例工程的两级配置，看继承与覆写合并后的最终结果（下面代码用 python 运行）：

```python #test id="mmengine-config"
import sys

sys.path.insert(0, '/tmp/mmengine_quickstart')

from mmengine.config import Config

cfg = Config.fromfile('/tmp/mmengine_quickstart/config.py')

print('model_type:', cfg.model.type)
print('max_epochs:', cfg.train_cfg.max_epochs)
print('lr:', cfg.optim_wrapper.optimizer.lr)
print('inherited_work_dir:', cfg.work_dir)
print('inherited_batch_size:', cfg.train_dataloader.batch_size)
print('base_key:', cfg.get('_base_', None))
```

输出结果如下：

```shell #test-result id="mmengine-config"
model_type: TinyMLP
max_epochs: 2
lr: 0.01
inherited_work_dir: /tmp/mmengine_quickstart/work_dir
inherited_batch_size: 16
base_key: None
```

## 注册表与构建

MMEngine 把模型、指标、数据集、Hook 等都放进**注册表**，配置里的 `type='XXX'` 就是查表用的键。下面代码用 python 运行：

```python #test id="mmengine-registry"
import sys

sys.path.insert(0, '/tmp/mmengine_quickstart')

from mmengine.registry import (DATASETS, FUNCTIONS, HOOKS, METRICS, MODELS,
                               build_from_cfg)

import toy_data  # noqa: F401  registers GaussianDataset / Accuracy / toy_collate
import toy_hook  # noqa: F401  registers IterationCounterHook
import toy_model  # noqa: F401  registers TinyMLP

model = MODELS.build(dict(type='TinyMLP', in_dim=16, hidden_dim=32))
print('built_model:', type(model).__name__)
print('param_numel:', sum(p.numel() for p in model.parameters()))

metric = METRICS.build(dict(type='Accuracy'))
print('built_metric:', type(metric).__name__)
print('metric_prefix:', metric.prefix)

dataset = DATASETS.build(dict(type='GaussianDataset', num_samples=8, seed=0))
print('built_dataset_len:', len(dataset))

print('collate_fn:', FUNCTIONS.get('toy_collate').__name__)
print('hook_registered:', 'IterationCounterHook' in HOOKS.module_dict)

wrapped = build_from_cfg(
    dict(type='TinyMLP', in_dim=16, hidden_dim=8), registry=MODELS)
print('build_from_cfg:', type(wrapped).__name__)
```

输出结果如下：

```shell #test-result id="mmengine-registry"
built_model: TinyMLP
param_numel: 610
built_metric: Accuracy
metric_prefix: accuracy
built_dataset_len: 8
collate_fn: toy_collate
hook_registered: True
build_from_cfg: TinyMLP
```

## 数据集与变换

`BaseDataset` 读一份 JSON 标注文件，`pipeline` 是一个 `Compose` 变换流水线（下面代码用 python 运行）：

```python #test id="mmengine-dataset"
import json
import sys

sys.path.insert(0, '/tmp/mmengine_quickstart')

from mmengine.dataset import BaseDataset

import toy_data  # noqa: F401  registers ToToyFeature

ann = {
    'metainfo': {
        'dataset_type': 'toy'
    },
    'data_list': [{
        'img_path': 'a',
        'label': 0
    }, {
        'img_path': 'b',
        'label': 1
    }]
}
with open('/tmp/mmengine_quickstart/ann.json', 'w') as fh:
    json.dump(ann, fh)

dataset = BaseDataset(
    ann_file='/tmp/mmengine_quickstart/ann.json',
    pipeline=[dict(type='ToToyFeature', dim=4)],
    test_mode=True)

print('dataset_len:', len(dataset))
print('dataset_meta:', dataset.metainfo['dataset_type'])
print('sample_feature:', dataset[0]['feature'].tolist())
print('sample_label:', dataset[0]['label'])
```

输出结果如下：

```shell #test-result id="mmengine-dataset"
dataset_len: 2
dataset_meta: toy
sample_feature: [0.0, 0.0, 0.0, 0.0]
sample_label: 0
```

## 训练全流程（Runner）

这是 MMEngine 的核心：一个 `Runner` 把 dataloader、优化器包装、评估器、Hook、检查点、日志全部串起来，`runner.train()` 一行跑完训练 + 验证。下面代码用 python 运行：

```python #test id="mmengine-runner"
import contextlib
import io
import os
import sys

sys.path.insert(0, '/tmp/mmengine_quickstart')

from mmengine.config import Config
from mmengine.runner import Runner

from toy_hook import IterationCounterHook

WORK_DIR = '/tmp/mmengine_quickstart/work_dir'
cfg = Config.fromfile('/tmp/mmengine_quickstart/config.py')

# Runner 的日志非常啰嗦，这里先吞掉，只保留我们自己打印的结论
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runner = Runner.from_cfg(cfg)
    runner.train()
log = buf.getvalue()

hook = next(h for h in runner.hooks if isinstance(h, IterationCounterHook))
acc_line = [ln for ln in log.splitlines() if 'accuracy/accuracy' in ln][-1]

print('model_device:', next(runner.model.parameters()).device)
print('hook_iters:', hook.num_iters)
print('hook_epochs:', hook.num_epochs)
print('ckpts:', sorted(f for f in os.listdir(WORK_DIR) if f.endswith('.pth')))
print('val_accuracy:', acc_line.split('accuracy/accuracy:')[1].split()[0])
```

输出结果如下：

```shell #test-result id="mmengine-runner"
model_device: npu:0
hook_iters: 8
hook_epochs: 2
ckpts: ['epoch_1.pth', 'epoch_2.pth']
val_accuracy: 1.0000
```

## 自定义 Hook

Hook 是 MMEngine 的扩展点：覆写生命周期方法即可，优先级用 `priority` 指定。下面代码用 python 运行：

```python #test id="mmengine-hook"
import sys

sys.path.insert(0, '/tmp/mmengine_quickstart')

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class EpochSpanHook(Hook):
    """Record the epoch range the runner went through."""

    priority = 'VERY_LOW'

    def __init__(self):
        self.epochs = []

    def before_train_epoch(self, runner):
        self.epochs.append(('before', runner.epoch))

    def after_train_epoch(self, runner):
        self.epochs.append(('after', runner.epoch))


hook = EpochSpanHook()
print('hook_priority:', hook.priority)
print('is_hook_instance:', isinstance(hook, Hook))
print('stages_count:', len(Hook.stages))
print('before_train_epoch in stages:', 'before_train_epoch' in Hook.stages)
```

输出结果如下：

```shell #test-result id="mmengine-hook"
hook_priority: VERY_LOW
is_hook_instance: True
stages_count: 22
before_train_epoch in stages: True
```

## 混合精度训练（AMP）

`AmpOptimWrapper` 在 NPU 上会自动改用 `torch.npu.amp.GradScaler`，并把 `autocast` 上下文一起打开。下面代码用 python 运行：

```python #test id="mmengine-amp"
import contextlib
import io
import sys

sys.path.insert(0, '/tmp/mmengine_quickstart')

from mmengine.config import Config
from mmengine.runner import Runner

cfg = Config.fromfile('/tmp/mmengine_quickstart/config.py')
cfg.optim_wrapper = dict(
    type='AmpOptimWrapper',
    dtype='float16',
    optimizer=dict(type='AdamW', lr=0.01))

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runner = Runner.from_cfg(cfg)
    runner.train()

print('optim_wrapper_type:', type(runner.optim_wrapper).__name__)
print('loss_scaler_class:', type(runner.optim_wrapper.loss_scaler).__name__)
print('model_device:', next(runner.model.parameters()).device)
```

输出结果如下：

```shell #test-result id="mmengine-amp"
optim_wrapper_type: AmpOptimWrapper
loss_scaler_class: GradScaler
model_device: npu:0
```

## 学习率调度（ParamScheduler）

`param_scheduler` 支持按 epoch 或按 iter 调度，MMEngine 内置 `LinearLR` / `MultiStepLR` / `CosineAnnealingLR` / `PolyLR` 等。下面代码用 python 运行：

```python #test id="mmengine-scheduler"
import contextlib
import io
import sys

sys.path.insert(0, '/tmp/mmengine_quickstart')

from mmengine.config import Config
from mmengine.runner import Runner

cfg = Config.fromfile('/tmp/mmengine_quickstart/config.py')
cfg.param_scheduler = [
    dict(
        type='LinearLR',
        by_epoch=False,
        begin=0,
        end=8,
        start_factor=1.0,
        end_factor=0.1)
]

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runner = Runner.from_cfg(cfg)
    runner.train()
log = buf.getvalue()

lr_lines = [ln for ln in log.splitlines() if 'lr:' in ln]
first_lr = float(lr_lines[0].split('lr:')[1].split()[0])
last_lr = float(lr_lines[-1].split('lr:')[1].split()[0])

print('scheduler_type:', type(runner.param_schedulers[0]).__name__)
print('lr_log_lines:', len(lr_lines))
print('first_lr: %.4f' % first_lr)
print('last_lr: %.4f' % last_lr)
```

输出结果如下：

```shell #test-result id="mmengine-scheduler"
scheduler_type: LinearLR
lr_log_lines: 4
first_lr: 0.0087
last_lr: 0.0010
```

## 评估与测试

`runner.test()` 走一遍 test dataloader 并返回指标字典（下面代码用 python 运行）：

```python #test id="mmengine-test"
import contextlib
import io
import sys

sys.path.insert(0, '/tmp/mmengine_quickstart')

from mmengine.config import Config
from mmengine.runner import Runner

cfg = Config.fromfile('/tmp/mmengine_quickstart/config.py')

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runner = Runner.from_cfg(cfg)
    runner.train()
    metrics = runner.test()

print('test_metrics:', metrics)
print('metric_keys:', sorted(metrics.keys()))
print('model_device:', next(runner.model.parameters()).device)
```

输出结果如下：

```shell #test-result id="mmengine-test"
test_metrics: {'accuracy/accuracy': 1.0}
metric_keys: ['accuracy/accuracy']
model_device: npu:0
```

## 检查点与断点续训

`CheckpointHook` 会把模型权重、优化器状态、`message_hub` 一起存进 `.pth`。下面代码用 python 运行：

```python #test id="mmengine-checkpoint"
import contextlib
import io
import os
import sys

sys.path.insert(0, '/tmp/mmengine_quickstart')

# mmengine 0.10.x 的 checkpoint.py 调用 torch.load() 时没有显式传
# weights_only=False，而 torch>=2.6 默认 weights_only=True，会拒绝反序列化
# checkpoint 里的 MessageHub / HistoryBuffer 等非张量对象。
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import torch

from mmengine.config import Config
from mmengine.runner import Runner

from toy_hook import IterationCounterHook

WORK_DIR = '/tmp/mmengine_quickstart/ckpt_work_dir'
cfg = Config.fromfile('/tmp/mmengine_quickstart/config.py')
cfg.work_dir = WORK_DIR
cfg.train_cfg.max_epochs = 1

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runner = Runner.from_cfg(cfg)
    runner.train()

    ckpt = os.path.join(WORK_DIR, 'epoch_1.pth')
    state = torch.load(ckpt, map_location='cpu')

    # 再起一个 Runner，把检查点加载回来，验证权重往返一致
    cfg2 = Config.fromfile('/tmp/mmengine_quickstart/config.py')
    cfg2.work_dir = '/tmp/mmengine_quickstart/ckpt_work_dir2'
    cfg2.train_cfg.max_epochs = 1
    runner2 = Runner.from_cfg(cfg2)
    runner2.load_checkpoint(ckpt)

    w_saved = runner.model.fc1.weight.detach().cpu()
    w_restored = runner2.model.fc1.weight.detach().cpu()

    # 断点续训：从 epoch 1 的检查点接着跑，训练循环只补跑剩下的 epoch
    cfg3 = Config.fromfile('/tmp/mmengine_quickstart/config.py')
    cfg3.work_dir = '/tmp/mmengine_quickstart/ckpt_work_dir3'
    cfg3.train_cfg.max_epochs = 2
    cfg3.load_from = ckpt
    cfg3.resume = True
    runner3 = Runner.from_cfg(cfg3)
    runner3.train()
    hook3 = next(
        h for h in runner3.hooks if isinstance(h, IterationCounterHook))
    resumed_iters = hook3.num_iters
    resumed_epoch = runner3.epoch

print('ckpt_exists:', os.path.isfile(ckpt))
print('ckpt_keys:', sorted(state.keys()))
print('ckpt_meta_epoch:', state['meta']['epoch'])
print('weights_match:', torch.allclose(w_saved, w_restored))
print('resumed_iters:', resumed_iters)
print('resumed_epoch:', resumed_epoch)
```

输出结果如下：

```shell #test-result id="mmengine-checkpoint"
ckpt_exists: True
ckpt_keys: ['message_hub', 'meta', 'optimizer', 'state_dict']
ckpt_meta_epoch: 1
weights_match: True
resumed_iters: 4
resumed_epoch: 2
```

## 日志与 MessageHub

`MessageHub` 是 Runner 内部共享标量与运行时信息的中心，`MMLogger` 负责输出。下面代码用 python 运行：

```python #test id="mmengine-logging"
import io
import logging
import sys

sys.path.insert(0, '/tmp/mmengine_quickstart')

from mmengine.logging import MessageHub, MMLogger, print_log

logger = MMLogger.get_current_instance()
logger.setLevel(logging.INFO)

# 临时把 MMLogger 的 stdout handler 换成 StringIO，方便断言输出
stream = io.StringIO()
origin_handlers = logger.handlers[:]
logger.handlers = [logging.StreamHandler(stream)]
try:
    print_log('hello from mmengine logger', logger='current', level=logging.INFO)
finally:
    logger.handlers = origin_handlers

hub = MessageHub.get_current_instance()
hub.update_info('epoch', 3)
hub.update_scalar('train/loss', 0.25, 10)

print('log_line:', stream.getvalue().strip().split(' - ')[-1])
print('hub_epoch:', hub.get_info('epoch'))
print('hub_current:', hub.get_scalar('train/loss').current())
print('hub_max:', hub.get_scalar('train/loss').max())
print('hub_mean:', hub.get_scalar('train/loss').mean())
```

输出结果如下：

```shell #test-result id="mmengine-logging"
log_line: hello from mmengine logger
hub_epoch: 3
hub_current: 0.25
hub_max: 0.25
hub_mean: 0.025
```

## 可视化

`Visualizer` 负责画图与记录标量，后端可插拔（本地文件 / TensorBoard / W&B 等）。下面代码用 python 运行：

```python #test id="mmengine-visualizer"
import os
import sys

import numpy as np

sys.path.insert(0, '/tmp/mmengine_quickstart')

from mmengine.visualization import Visualizer

SAVE_DIR = '/tmp/mmengine_quickstart/vis'

visualizer = Visualizer(
    name='toy',
    image=np.zeros((64, 64, 3), dtype=np.uint8),
    vis_backends=[dict(type='LocalVisBackend')],
    save_dir=SAVE_DIR)
visualizer.add_scalars(
    {'train/loss': 0.5},
    step=1,
    file_path=os.path.join(SAVE_DIR, 'scalars.json'))
visualizer.draw_texts('hello npu', np.array([[4, 4]]), colors='g')
visualizer.draw_points(np.array([[16, 16]]), colors='r')
visualizer.add_image('toy_image', visualizer.get_image(), step=1)

files = []
for root, _, names in os.walk(SAVE_DIR):
    for name in names:
        files.append(os.path.relpath(os.path.join(root, name), SAVE_DIR))

print('vis_files:', sorted(files))
print('backend_name:', type(visualizer.get_backend('LocalVisBackend')).__name__)
```

输出结果如下：

```shell #test-result id="mmengine-visualizer"
vis_files: ['scalars.json', 'vis_data/scalars.json', 'vis_data/vis_image/toy_image_1.png']
backend_name: LocalVisBackend
```

## 文件 IO

`mmengine.fileio` 提供统一的读写接口，本地文件走 `LocalBackend`，也可以换成 S3 / HTTP 等后端。下面代码用 python 运行：

```python #test id="mmengine-fileio"
import os
import tempfile

from mmengine.fileio import dump, get, get_local_path, load, put_text

tmp_dir = tempfile.mkdtemp(prefix='mmengine_io_')
text_path = os.path.join(tmp_dir, 'note.txt')
put_text('hello mmengine', text_path)
print('put_get_text:', get(text_path))
print('file_exists:', os.path.isfile(text_path))

pkl_path = os.path.join(tmp_dir, 'data.pkl')
dump({'a': 1}, pkl_path)
print('dump_load_pickle:', load(pkl_path))

json_path = os.path.join(tmp_dir, 'data.json')
dump({'b': 2}, json_path)
print('dump_load_json:', load(json_path))

with get_local_path(text_path) as local_path:
    print('get_local_path:', os.path.basename(str(local_path)))
```

输出结果如下：

```shell #test-result id="mmengine-fileio"
put_get_text: b'hello mmengine'
file_exists: True
dump_load_pickle: {'a': 1}
dump_load_json: {'b': 2}
get_local_path: note.txt
```

## NPU 专属：融合优化器与性能分析

### torch_npu 融合优化器

`mmengine.optim` 在检测到 NPU 时会把 `torch_npu.optim` 下的融合优化器一并注册进 `OPTIMIZERS` 注册表，可以直接在配置里用 `type='NpuFusedAdamW'`。下面代码用 python 运行：

```python #test id="mmengine-npu-optim"
import sys

sys.path.insert(0, '/tmp/mmengine_quickstart')

import torch_npu  # noqa: F401  必须在 mmengine.optim 之前 import

from mmengine.optim import build_optim_wrapper  # noqa: F401  触发 torch_npu.optim 的注册
from mmengine.registry import OPTIMIZERS

npu_optimizers = sorted(n for n in OPTIMIZERS.module_dict if 'Npu' in n)
print('npu_optim_count:', len(npu_optimizers))
print('has_npu_fused_adamw:', 'NpuFusedAdamW' in npu_optimizers)
print('torch_optimizers_registered:', 'AdamW' in OPTIMIZERS.module_dict)
```

输出结果如下：

```shell #test-result id="mmengine-npu-optim"
npu_optim_count: 10
has_npu_fused_adamw: True
torch_optimizers_registered: True
```

### NPU 性能分析

`NPUProfilerHook` 把 `torch_npu` 的 profiler 接进训练循环。先跑一遍带 profiling 的训练（下面代码用 python 运行）：

```python #test-setup
import contextlib
import io
import os
import shutil
import sys

sys.path.insert(0, '/tmp/mmengine_quickstart')

import torch_npu
from torch_npu.profiler import ProfilerActivity, profile

# mmengine 0.10.x 的 NPUProfilerHook 调用的是 torch_npu.npu.profile，
# 该 API 在 torch_npu 2.9 已被移除（改为 torch_npu.profiler.profile）。
if not hasattr(torch_npu.npu, 'profile'):

    def _profile_shim(result_path, use_e2e_profiler=False):
        return profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                result_path))

    torch_npu.npu.profile = _profile_shim

from mmengine.config import Config
from mmengine.runner import Runner

PROFILE_DIR = '/tmp/mmengine_quickstart/cann_profiling'
shutil.rmtree(PROFILE_DIR, ignore_errors=True)

cfg = Config.fromfile('/tmp/mmengine_quickstart/config.py')
cfg.work_dir = '/tmp/mmengine_quickstart/prof_work_dir'
cfg.train_cfg.max_epochs = 1
cfg.custom_hooks = [
    dict(
        type='NPUProfilerHook',
        begin=1,
        end=3,
        result_path=PROFILE_DIR,
        exit_after_profiling=False)
]

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    runner = Runner.from_cfg(cfg)
    runner.train()
```

再检查 profiling 产物目录（下面代码用 python 运行）：

```python #test id="mmengine-npu-profiler"
import os

PROFILE_DIR = '/tmp/mmengine_quickstart/cann_profiling'
entries = sorted(os.listdir(PROFILE_DIR)) if os.path.isdir(PROFILE_DIR) else []
print('profiling_dir_exists:', os.path.isdir(PROFILE_DIR))
print('profiling_entries:', len(entries))
```

输出结果如下：

```shell #test-result id="mmengine-npu-profiler"
profiling_dir_exists: True
profiling_entries: 1
```

## 外部链接

- GitHub：[open-mmlab/mmengine](https://github.com/open-mmlab/mmengine)
- 文档中心：[MMEngine Docs](https://mmengine.readthedocs.io/en/latest/)
