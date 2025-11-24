快速开始，单节点部署指引
==================

.. note::

    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 verl 所需的环境。
    
    本篇教程将介绍如何使用 roll 进行快速训练，帮助您快速上手 roll   。

本文档帮助昇腾开发者快速使用 roll × 昇腾 进行 LLM 强化学习训练。可以访问 `这篇官方文档 <https://alibaba.github.io/ROLL/docs/QuickStart/installation#>`_ 获取更多信息。

也可以参考官方的 `昇腾快速开始文档 <https://verl.readthedocs.io/en/latest/ascend_tutorial/ascend_quick_start.html>`_ 。



正式使用前，建议您通过对单节点流水线的训练尝试以检验环境准备和安装的正确性。 由于目前暂不支持 Megatron-LM 训练，请首先将对应文件中 strategy_args 参数修改为 deepspeed 选项。

使用配置文件执行 agentic pipeline
^^^^^^^^^^^^^^^^^^^^^^

.. note:: 
   
   以 qwen2.5-0.5B-agentic 为例（Requires >=4 NPUs）

修改策略文件:

.. code-block:: python

    # vim roll/distributed/strategy/vllm_strategy.py
    enable_prefix_caching: False,


修改配置文件：

.. code-block:: yaml

    # vim examples/qwen2.5-0.5B-agentic/agentic_val_sokoban.yaml
    defaults:
      - ../config/traj_envs@_here_
      - ../config/deepspeed_zero@_here_
      - ../config/deepspeed_zero2@_here_
      - ../config/deepspeed_zero3@_here_
      - ../config/deepspeed_zero3_cpuoffload@_here_
    
    hydra:
      run:
        dir: .
      output_subdir: null
    
    exp_name: "agentic_pipeline"
    seed: 42
    logging_dir: ./output/logs
    output_dir: ./output
    render_save_dir: ./output/render
    system_envs:
      USE_MODELSCOPE: '1'
    
    #track_with: wandb
    #tracker_kwargs:
    #  api_key:
    #  project: roll-agentic
    #  name: ${exp_name}_sokoban
    #  notes: "agentic_pipeline"
    #  tags:
    #    - agentic
    #    - roll
    #    - baseline
    
    track_with: tensorboard
    tracker_kwargs:
      log_dir: ./data/oss_bucket_0/yali/llm/tensorboard/roll_exp/agentic_sokoban
    
    
    checkpoint_config:
      type: file_system
      output_dir: ./data/cpfs_0/rl_examples/models/${exp_name}
    
    num_gpus_per_node: 4
    
    max_steps: 128
    save_steps: 10000
    logging_steps: 1
    eval_steps: 10
    resume_from_checkpoint: false
    
    rollout_batch_size: 16
    val_batch_size: 16
    sequence_length: 1024
    
    advantage_clip: 0.2
    ppo_epochs: 1
    adv_estimator: "grpo"
    #pg_clip: 0.1
    #dual_clip_loss: True
    init_kl_coef: 0.0
    whiten_advantages: true
    entropy_loss_coef: 0
    max_grad_norm: 1.0
    
    pretrain: Qwen/Qwen2.5-0.5B-Instruct
    reward_pretrain: Qwen/Qwen2.5-0.5B-Instruct
    
    actor_train:
      model_args:
        attn_implementation: fa2
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
        strategy_name: deepspeed_train
        strategy_config: ${deepspeed_zero3}
        # strategy_name: megatron_train
        # strategy_config:
        #   tensor_model_parallel_size: 1
        #   pipeline_model_parallel_size: 1
        #   expert_model_parallel_size: 1
        #   use_distributed_optimizer: true
        #   recompute_granularity: full
      device_mapping: list(range(0,2))
      infer_batch_size: 2
    
    actor_infer:
      model_args:
        disable_gradient_checkpointing: true
        dtype: bf16
      generating_args:
        max_new_tokens: 128 # single-turn response length
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
          gpu_memory_utilization: 0.6
          block_size: 16
          load_format: auto
      device_mapping: list(range(2,3))
    
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
      device_mapping: list(range(3,4))
      infer_batch_size: 2
    
    reward_normalization:
      grouping: traj_group_id # 可以tags(env_type)/traj_group_id(group)/batch(rollout_batch)... group_by计算reward/adv
      method: mean_std # asym_clip / identity / mean_std
    
    train_env_manager:
      format_penalty: -0.15 # sokoban env penalty_for_step=-0.1
      max_env_num_per_worker: 4
      num_env_groups: 8
      # under the same group, the env config and env seed are ensured to be equal
      group_size: 1
      tags: [SimpleSokoban]
      num_groups_partition: [8] # If not set, all env names divide nums equally. Under the same group, the env config and env seed (prompt) are equal in each generation
    
    val_env_manager:
      max_env_num_per_worker: 32
      num_env_groups: 64
      group_size: 1 # should be set to 1 because val temperature is set to 0 and same prompt leads to same output
      tags: [SimpleSokoban, LargerSokoban, SokobanDifferentGridVocab, FrozenLake]
      num_groups_partition: [16, 16, 16, 16] # TODO: If not set, all env names divide nums equally. Under the same group, the env config and env seed (prompt) are equal in each generation
    
    
    # Here, you can override variables defined in the imported envs. max_tokens_per_step: 128 in custom_env.SimpleSokoban, here replaced by 64
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


使用配置文件执行：

.. code-block:: bash
    
    python examples/start_agentic_pipeline.py \
            --config_path qwen2.5-0.5B-agentic \
            --config_name agentic_val_sokoban
    
    - ``--config_path`` – 包含您的YAML配置文件的目录。
    - ``--config_name`` – 文件名（不含.yaml后缀）。

若执行遇到报错：

.. code-block:: bash

    ...
    File "/home/coder/miniforge3/envs/roll/lib/python3.10/enum.py",
       line 701, in __new__
    raise ve_exc

   ValueError: <object object at 0xffff839ef4a0> is not a valid Sentinel

可尝试将 Click 版本版本降级到 8.2.1：

.. code-block:: bash
   
    pip install --force-reinstall 'click==8.2.1'

支持现状
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 40 15 15 15

   * - Feature
     - Example
     - Training Backend
     - Inference Backend
     - Hardware
   * - Agentic
     - examples/qwen2.5-0.5B-agentic/run_agentic_pipeline_sokoban.sh
     - DeepSpeed
     - vLLM
     - Atlas 900 A2 PODc
   * - Agentic-Rollout
     - examples/qwen2.5-0.5B-agentic/run_agentic_rollout_sokoban.sh
     - DeepSpeed
     - vLLM
     - Atlas 900 A2 PODc
   * - DPO
     - examples/qwen2.5-3B-dpo_megatron/run_dpo_pipeline.sh
     - DeepSpeed
     - vLLM
     - Atlas 900 A2 PODc
   * - RLVR
     - examples/qwen2.5-7B-rlvr_megatron/run_rlvr_pipeline.sh
     - DeepSpeed
     - vLLM
     - Atlas 900 A2 PODc

声明
^^^^^^^^^^^^^^^^^^^^^^

ROLL 中提供的 Ascend 支持代码皆为参考样例，生产环境使用请通过官方正式途径沟通，谢谢。