# ROLL

在昇腾 NPU 上安装 ROLL，并用 FrozenLake agentic 强化学习跑通一次完整的训练闭环。

## 前置条件

### 硬件

Atlas 900 A2 或 A3 训练系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在运行本文档示例之前，你的机器上已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

本文档示例面向 Atlas 900 A2 训练系列产品，使用 Python 3.11 和 CANN 9.1.0。

本文档配套镜像：

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.11
```

## 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装依赖

### 安装 vLLM 和 vLLM-Ascend

安装 vLLM 和昇腾 Triton，并从源码构建 vLLM-Ascend。

```shell #test-setup id="install-vllm"
pip install --index-url https://repo.huaweicloud.com/repository/pypi/simple vllm==0.23.0
git clone --depth 1 --branch v0.23.0 --recurse-submodules https://github.com/vllm-project/vllm-ascend.git
pip install --index-url https://repo.huaweicloud.com/repository/pypi/simple \
  --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
  -e ./vllm-ascend triton-ascend==3.2.2
```

核对 vLLM 和 Triton 的版本，运行下面python代码：

```python #test id="verify-vllm-triton"
from importlib.metadata import version

for package in ("vllm", "vllm-ascend", "triton-ascend"):
    print(f"{package}=={version(package)}")
```

输出结果如下：

```shell #test-result id="verify-vllm-triton"
vllm==0.23.0
vllm-ascend==0.23.0
triton-ascend==3.2.2
```

### 安装 torch 和 torch_npu

安装匹配的 PyTorch 软件栈：

```shell #test-setup id="install-torch"
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple torch==2.10.0+cpu torchvision==0.25.0 torchaudio==2.10.0
pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi --no-deps torch-npu==2.10.0.post4
```

核对 PyTorch 软件栈的版本，运行下面python代码：

```python #test id="verify-torch"
from importlib.metadata import version

for package in ("torch", "torch-npu", "torchvision", "torchaudio"):
    print(f"{package}=={version(package)}")
```

输出结果如下：

```shell #test-result id="verify-torch"
torch==2.10.0+cpu
torch-npu==2.10.0.post4
torchvision==0.25.0
torchaudio==2.10.0
```

### 安装 ROLL

**使用源码安装 ROLL** ：

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test-setup id="install-roll" load="upstream_ref>>ref"
git clone --branch <ref> https://github.com/alibaba/ROLL.git
cd ROLL
pip install -r requirements_common.txt "torch==2.10.0+cpu" "tensordict==0.14.3" "pyarrow<26.0.0"
pip install "numpy>=2.0,<2.3"
pip install -e .
cd ..
```

```{admonition} Note
:class: note
`<ref>` 为最新正式 release 的 tag
```

查看已安装的 ROLL 版本，运行下面python代码：

```python #test id="verify-roll"
from importlib.metadata import version

print(f"roll=={version('roll')}")
```

输出结果如下：

```shell #test-result id="verify-roll" fuzzy='xxx'
roll==xxx
```

```{admonition} Note
:class: note
输出中的 `xxx` 为已安装的 ROLL 版本。
```

## 运行示例：FrozenLake agentic 强化学习

FrozenLake 是 ROLL 官方快速入门的示例：Qwen2.5-0.5B-Instruct 作为策略模型，在 4×4 冰面网格中逐轮输出移动方向，绕开冰洞到达终点，环境按结果返回奖励。

本示例使用单机 8 张 NPU 卡。

**设置环境变量。**

```shell #test-setup id="set-example-env"
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

export NPU_MEMORY_FRACTION=0.96
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=1
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1

export CPU_AFFINITY_CONF=2
export OMP_NUM_THREADS=1

export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1

export ACL_OP_COMPILER_CACHE_MODE=enable
export ACL_OP_COMPILER_CACHE_DIR=/tmp/npu_cache
export ASCEND_MAX_OP_CACHE_SIZE=5000
```

**安装示例依赖。**

```shell #test-setup id="install-example-deps"
pip install transformers==4.57.6 tensorboard==2.20.0
```

核对示例依赖的版本，运行下面python代码：

```python #test id="verify-example-deps"
from importlib.metadata import version

for package in ("transformers", "tensorboard"):
    print(f"{package}=={version(package)}")
```

输出结果如下：

```shell #test-result id="verify-example-deps"
transformers==4.57.6
tensorboard==2.20.0
```

**创建 NPU 配置文件。** 在 `ROLL` 目录下创建 `examples/agentic_frozen_lake_npu/quick_start_npu.yaml`：

```shell #test-setup id="write-config"
mkdir -p ROLL/examples/agentic_frozen_lake_npu
cat > ROLL/examples/agentic_frozen_lake_npu/quick_start_npu.yaml <<'EOF'
defaults:
  - ../config/traj_envs@_here_

hydra:
  run:
    dir: .
  output_subdir: null

exp_name: "agentic_frozen_lake_npu"
seed: 42
logging_dir: ./output/logs
output_dir: ./output
render_save_dir: ./output/render
system_envs:
  USE_MODELSCOPE: '1'
  HCCL_CONNECT_TIMEOUT: "3600"
  HCCL_DETERMINISTIC: "false"
  HCCL_OP_EXPANSION_MODE: "AIV"
  HCCL_NPU_SOCKET_PORT_RANGE: "auto"
  VLLM_ASCEND_ENABLE_NZ: "0"
  NPU_MEMORY_FRACTION: "0.96"
  CPU_AFFINITY_CONF: "2"
  OMP_NUM_THREADS: "1"
  VLLM_USE_V1: "1"

checkpoint_config:
  type: file_system
  output_dir: /data/models/${exp_name}

num_gpus_per_node: 8

# ROLL v0.4.0 的 NPU 路径不支持远程批次，单机示例关闭此后端。
transfer_backend:
  backend_name: null

max_steps: 1  # 快速入门只运行一步训练
save_steps: 10000
logging_steps: 1
eval_steps: 10
resume_from_checkpoint: false

rollout_batch_size: 1024
val_batch_size: 1024
sequence_length: 8192

advantage_clip: 0.2
ppo_epochs: 1
adv_estimator: "grpo"
init_kl_coef: 0.0
whiten_advantages: true
entropy_loss_coef: 0
max_grad_norm: 1.0

pretrain: Qwen/Qwen2.5-0.5B-Instruct
reward_pretrain: Qwen/Qwen2.5-0.5B-Instruct

actor_train:
  model_args:
    attn_implementation: fa2          # Use fa2 via transformers, NOT flash_attn
    disable_gradient_checkpointing: false
    dtype: bf16
    model_type: ~
  training_args:
    learning_rate: 1.0e-6
    weight_decay: 0
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 64
    warmup_steps: 10
    lr_scheduler_type: cosine
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 4
      param_dtype: bf16
      reduce_dtype: bf16
      reshard_after_forward: true
      offload_policy: false    # NPU: Must use FSDP2, NOT megatron_train
  device_mapping: list(range(0,4))    # NPU: Training on NPUs 0-3
  infer_batch_size: 2

actor_infer:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
  generating_args:
    max_new_tokens: 128
    top_p: 0.99
    top_k: 100
    num_beams: 1
    temperature: 0.99
    num_return_sequences: 1
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: vllm
    strategy_config:
      gpu_memory_utilization: 0.8
      block_size: 16
      load_format: auto
  device_mapping: list(range(4,8))    # NPU: Inference on NPUs 4-7 (separate from training)
  infer_batch_size: 2

reference:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: fsdp2_infer
    strategy_config:
      fsdp_size: 4
      param_dtype: bf16
      reduce_dtype: bf16
      reshard_after_forward: true
      offload_policy: false
  device_mapping: list(range(4,8))    # NPU: Share inference NPUs with actor_infer
  infer_batch_size: 2

reward_normalization:
  grouping: traj_group_id
  method: mean_std

train_env_manager:
  max_env_num_per_worker: 16
  num_env_groups: 128
  group_size: 8
  tags: [FrozenLake]
  num_groups_partition: [128]

val_env_manager:
  max_env_num_per_worker: 32
  num_env_groups: 1024
  group_size: 1
  tags: [SimpleSokoban, LargerSokoban, SokobanDifferentGridVocab, FrozenLake]
  num_groups_partition: [256, 256, 256, 256]

max_tokens_per_step: 64

custom_envs:
  SimpleSokoban:
    ${custom_env.SimpleSokoban}
  LargerSokoban:
    ${custom_env.LargerSokoban}
  SokobanDifferentGridVocab:
    ${custom_env.SokobanDifferentGridVocab}
  FrozenLake:
    ${custom_env.FrozenLake}
  FrozenLakeThink:
    ${custom_env.FrozenLakeThink}

EOF
```

**启动训练。**

```shell #test-setup id="run-agentic"
cd ROLL
python examples/start_agentic_pipeline.py --config_path agentic_frozen_lake_npu --config_name quick_start_npu
```

**查看训练产物。** 训练完成后，回到 `ROLL` 的上一级目录，再从 `ROLL/output/agentic_frozen_lake_npu` 读取 TensorBoard 指标，运行下面python代码：

```python #test id="verify-output"
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_file = sorted(Path("ROLL/output/agentic_frozen_lake_npu").glob("*/events.out.tfevents.*"))[-1]
metrics = EventAccumulator(str(event_file))
metrics.Reload()

print(f"FrozenLake 平均得分：{metrics.Scalars('critic/score/mean')[-1].value:.3f}")
```

输出结果如下：

```shell #test-result id="verify-output" fuzzy='xxx'
FrozenLake 平均得分：xxx
```

```{admonition} Note
:class: note
输出中的 `xxx` 为本次训练实际得到的平均得分，随训练随机性变化
```

## 外部链接

- GitHub：[alibaba/ROLL](https://github.com/alibaba/ROLL)
- 文档中心：[ROLL Docs](https://alibaba.github.io/ROLL/zh-Hans/)
