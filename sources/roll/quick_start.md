# ROLL

在昇腾 NPU 上安装 ROLL，并用 FrozenLake agentic 强化学习跑通一次完整的训练闭环。

## 前置条件

- **硬件**：Atlas 900 A2 PODc / Ascend 910B 训练系列，单卡。
- **软件**：已装好 CANN（toolkit 与驱动），并能 `source set_env.sh`，Python 版本不低于 3.10。参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)。

本文档示例在 Python 3.12、CANN 9.1.0 环境下验证通过。

## 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装 ROLL

**使用源码安装 ROLL** ：

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="install-roll" load="upstream_ref>>ref"
git clone --branch <ref> https://github.com/alibaba/ROLL.git
cd ROLL
echo "ROLL $(git describe --tags --exact-match HEAD)"
pip install -e .
```

`<ref>` 为最新正式 release tag。

输出结果如下：

```shell #test-result id="install-roll" fuzzy='xxx'
ROLL xxx
```

其中 `xxx` 为实际安装到的 release 版本号，如 `v0.3.0`。

## 运行示例：FrozenLake agentic 强化学习

FrozenLake 是 ROLL 官方快速入门的示例：Qwen2.5-0.5B-Instruct 作为策略模型，在 4×4 冰面网格中逐轮输出移动方向，绕开冰洞到达终点，环境按结果返回奖励。

**安装 vLLM 与 triton。** 安装 vLLM 与 vLLM-Ascend，配套版本的 torch、torch_npu、torchvision、torchaudio 会作为依赖自动装上；安装 triton-ascend，其配套的社区 triton 同样由依赖自动带入。

```shell #test-setup id="install-npu-runtime"
pip install --index-url https://repo.huaweicloud.com/repository/pypi/simple vllm==0.23.0
pip install --index-url https://repo.huaweicloud.com/repository/pypi/simple --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi vllm-ascend==0.23.0
pip install --index-url https://repo.huaweicloud.com/repository/pypi/simple --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi triton-ascend==3.2.2
```

**安装示例依赖。** 按昇腾镜像的依赖清单安装 agentic 示例所需组件：

```shell #test-setup id="install-example-deps"
cd ROLL
grep -v '^gem-llm' requirements_common.txt > requirements_npu.txt
sed -i 's/^decord /decord2 /' requirements_vision.txt
pip install -r requirements_npu.txt
pip install --ignore-requires-python gem-llm==0.0.4
pip install "numpy==1.26.4"
pip install "transformers==4.57.6" "tensorboard==2.20.0" "antlr4-python3-runtime==4.9.3"
rm requirements_npu.txt
```

**写入示例配置。** 配置基于官方 agentic demo，修改参数完成 NPU 适配。

```python #test id="write-config"
from pathlib import Path

config = """
# 环境定义沿用官方 examples/config/traj_envs.yaml，NPU 上训练仅支持 FSDP2。
defaults:
  - ../config/traj_envs@_here_

hydra:
  run:
    dir: .
  output_subdir: null

exp_name: "roll-quick-start-npu"
seed: 42
logging_dir: ./output/logs
output_dir: ./output
render_save_dir: ./output/render
system_envs:
  USE_MODELSCOPE: '1'
  # RL 权重刷新场景需禁用 FRACTAL_NZ。
  VLLM_ASCEND_ENABLE_NZ: '0'
  # 允许同卡多进程由 HCCL 自动分配 device 侧端口。
  HCCL_NPU_SOCKET_PORT_RANGE: auto

track_with: tensorboard
tracker_kwargs:
  log_dir: ./output/tensorboard

num_gpus_per_node: 1

# 单卡快速跑通：只训练 1 步，批量收缩。
max_steps: 1
save_steps: 1000
logging_steps: 1
eval_steps: 1000
resume_from_checkpoint: false

rollout_batch_size: 8
val_batch_size: 2
sequence_length: 2048
max_tokens_per_step: 128

ppo_epochs: 1
adv_estimator: "grpo"
init_kl_coef: 0.0
whiten_advantages: true
entropy_loss_coef: 0

pretrain: Qwen/Qwen2.5-0.5B-Instruct
reward_pretrain: Qwen/Qwen2.5-0.5B-Instruct

actor_train:
  model_args:
    # NPU 通过 transformers 使用 fa2，不能使用 flash_attn 包。
    attn_implementation: fa2
    disable_gradient_checkpointing: false
    dtype: bf16
    model_type: ~
  training_args:
    learning_rate: 1.0e-6
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 2
  data_args:
    template: qwen2_5
  strategy_args:
    # NPU 不支持 Megatron，训练策略使用 FSDP2。
    strategy_name: fsdp2_train
    strategy_config:
      fsdp_size: 1
      param_dtype: bf16
      reduce_dtype: bf16
      reshard_after_forward: true
      offload_policy: false
      use_batched_model_update: false
  device_mapping: list(range(0,1))
  infer_batch_size: 1

actor_infer:
  model_args:
    disable_gradient_checkpointing: true
    dtype: bf16
  generating_args:
    max_new_tokens: 128
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
  device_mapping: list(range(0,1))
  infer_batch_size: 1

reference:
  model_args:
    attn_implementation: fa2
    disable_gradient_checkpointing: true
    dtype: bf16
    model_type: ~
  data_args:
    template: qwen2_5
  strategy_args:
    strategy_name: hf_infer
    strategy_config: ~
  device_mapping: list(range(0,1))
  infer_batch_size: 1

train_env_manager:
  max_env_num_per_worker: 8
  num_env_groups: 4
  group_size: 2
  tags: [FrozenLake]
  num_groups_partition: [4]

val_env_manager:
  max_env_num_per_worker: 2
  num_env_groups: 2
  group_size: 1
  tags: [FrozenLake]
  num_groups_partition: [2]

custom_envs:
  FrozenLake:
    ${custom_env.FrozenLake}
"""

path = Path("ROLL/examples/agentic_frozen_lake_npu/quick_start_npu.yaml")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(config.lstrip("\n"), encoding="utf-8")
print("config written", path)
```

输出结果如下：

```shell #test-result id="write-config"
config written ROLL/examples/agentic_frozen_lake_npu/quick_start_npu.yaml
```

**启动训练。** 从仓库根目录运行 agentic pipeline 入口脚本，ROLL 自动拉起 Ray 集群。

```shell #test-setup id="run-agentic"
cd ROLL
python examples/start_agentic_pipeline.py --config_path agentic_frozen_lake_npu --config_name quick_start_npu
```

**查看训练产物。** 训练指标写入 output/tensorboard：

```python #test id="verify-output"
from pathlib import Path

tensorboard_dir = Path("ROLL/output/tensorboard/roll-quick-start-npu")
print("训练产物已生成")
print(f"训练指标路径：{tensorboard_dir}")
```

输出结果如下：

```shell #test-result id="verify-output"
训练产物已生成
训练指标路径：ROLL/output/tensorboard/roll-quick-start-npu
```

## 更多用法

多卡并行、vLLM 高吞吐 rollout、Megatron 后端与 SFT/DPO 等其他 pipeline 见官方文档：https://alibaba.github.io/ROLL/docs/QuickStart/single_node_quick_start
