快速开始
==================

.. note::

    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 verl 所需的环境。
    
    本篇教程将介绍如何使用 verl 进行快速训练，帮助您快速上手 verl   。

本文档帮助昇腾开发者快速使用 verl × 昇腾 进行 LLM 强化学习训练。可以访问 `这篇官方文档 <https://verl.readthedocs.io/en/latest/start/install.html#>`_ 获取更多信息。

也可以参考官方的 `昇腾快速开始文档 <https://verl.readthedocs.io/en/latest/ascend_tutorial/ascend_quick_start.html>`_

正式使用前，建议通过对 Qwen2.5-0.5B PPO 的训练尝试以检验环境准备和安装的正确性，并熟悉基本的使用流程。

接下来将介绍如何使用单张 NPU 卡使用 verl 进行 PPO 训练：

基于 GSM8K 数据集对 Qwen2.5-0.5B 模型进行 PPO 训练
------------------------

使用 GSM8K 数据集 post-train Qwen2.5-0.5B 模型.

数据集介绍
^^^^^^^^^^^^^^^^^^^^^^

GSM8K 是一个包含初等数学问题的数据集，用于 LLM 的数学推理能力的训练或评估。以下是一组 prompt solution 示例：

Prompt

   James writes a 3-page letter to 2 different friends twice a week. 
   How many pages does he write a year?

Solution

   He writes each friend 3*2=<<3*2=6>>6 pages a week So he writes 
   6*2=<<6*2=12>>12 pages every week That means he writes 
   12*52=<<12*52=624>>624 pages a year #### 624

准备数据集
^^^^^^^^^^^^^^^^^^^^^^

用户可以根据实际需要修改 ``--local_save_dir`` 参数指定数据集的保存路径。

.. code-block:: bash

   python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

准备模型
^^^^^^^^^^^^^^^^^^^^^^

在本实例中，使用 Qwen2.5-0.5B-Instruct 作为基础模型进行 PPO 训练。

用户可以设置 ``VERL_USE_MODELSCOPE=True`` 由 `modelscope <https://www.modelscope.cn>`_ 下载模型。

.. code-block:: bash

   python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"

启动 PPO 训练
^^^^^^^^^^^^^^^^^^^^^^

**Reward Model/Function**

在本实例中，我们使用一个简单的奖励函数来评估生成答案的正确性。我们认为模型产生的位于 “####” 符号后的数值为其给出的答案。
如果该答案与正确答案匹配，则 reward 为 1，否则为 0。

对于其他细节，可以参考 `verl/utils/reward_score/gsm8k.py <https://github.com/volcengine/verl/blob/v0.4.1/verl/utils/reward_score/gsm8k.py>`_.

**Training Script**

根据用户的数据集以及模型的实际位置修改 ``data.train_files`` ,\ ``data.val_files``, ``actor_rollout_ref.model.path`` , ``critic.model.path`` 等参数即可。

.. code-block:: bash

   PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.device=npu 2>&1 | tee verl_demo.log

如果顺利配置环境并运行，将看到如下类似的输出：

.. code-block:: bash

    step:0 - timing/gen:21.470 - timing/ref:4.360 - timing/values:5.800 - actor/reward_kl_penalty:0.000 - actor/reward_kl_penalty_coeff:0.001 - timing/adv:0.109 - timing/update_critic:15.664 
    - critic/vf_loss:14.947 - critic/vf_clipfrac:0.000 - critic/vpred_mean:-2.056 - critic/grad_norm:1023.278 - critic/lr(1e-4):0.100 - timing/update_actor:20.314 - actor/entropy_loss:0.433 
    - actor/pg_loss:-0.005 - actor/pg_clipfrac:0.000 - actor/ppo_kl:0.000 - actor/grad_norm:1.992 - actor/lr(1e-4):0.010 - critic/score/mean:0.004 - critic/score/max:1.000 
    - critic/score/min:0.000 - critic/rewards/mean:0.004 - critic/rewards/max:1.000 - critic/rewards/min:0.000 - critic/advantages/mean:-0.000 - critic/advantages/max:2.360 
    - critic/advantages/min:-2.280 - critic/returns/mean:0.003 - critic/returns/max:0.000 - critic/returns/min:0.000 - critic/values/mean:-2.045 - critic/values/max:9.500 
    - critic/values/min:-14.000 - response_length/mean:239.133 - response_length/max:256.000 - response_length/min:77.000 - prompt_length/mean:104.883 - prompt_length/max:175.000 
    - prompt_length/min:68.000
    step:1 - timing/gen:23.020 - timing/ref:4.322 - timing/values:5.953 - actor/reward_kl_penalty:0.000 - actor/reward_kl_penalty:0.001 - timing/adv:0.118 - timing/update_critic:15.646 
    - critic/vf_loss:18.472 - critic/vf_clipfrac:0.384 - critic/vpred_mean:1.038 - critic/grad_norm:942.924 - critic/lr(1e-4):0.100 - timing/update_actor:20.526 - actor/entropy_loss:0.440 
    - actor/pg_loss:0.000 - actor/pg_clipfrac:0.002 - actor/ppo_kl:0.000 - actor/grad_norm:2.060 - actor/lr(1e-4):0.010 - critic/score/mean:0.000 - critic/score/max:0.000 
    - critic/score/min:0.000 - critic/rewards/mean:0.000 - critic/rewards/max:0.000 - critic/rewards/min:0.000 - critic/advantages/mean:0.000 - critic/advantages/max:2.702 
    - critic/advantages/min:-2.616 - critic/returns/mean:0.000 - critic/returns/max:0.000 - critic/returns/min:0.000 - critic/values/mean:-2.280 - critic/values/max:11.000 
    - critic/values/min:-16.000 - response_length/mean:232.242 - response_length/max:256.000 - response_length/min:91.000 - prompt_length/mean:102.398 - prompt_length/max:185.000 
    - prompt_length/min:70.000

References

.. [1] https://verl.readthedocs.io/en/latest/start/install.html