快速开始
===============

.. note::

    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 TRL (Transformer Reinforcement Learning) ！

本教程以 DPO 方法为例，讲述如何使用 TRL 在昇腾 NPU 上进行模型的后训练。

前置准备
---------------

本篇样例代码为 TRL 官方样例，需提前进行下载：

.. code-block::

    git clone https://github.com/huggingface/trl.git

模型训练
---------------

进入 TRL 项目目录，依次执行如下命令：

.. code-block::

    cd examples/scripts
    python dpo.py

出现如下日志则代表训练成功：

::
    
    Tokenizing train dataset: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 62135/62135 [07:11<00:00, 143.85 examples/s]
    Tokenizing eval dataset: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:06<00:00, 144.73 examples/s]
    Detected kernel version 4.19.90, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
      0%| ... | 0/3883 [00:00<?, ?it/s]/home/sss/github/trl/trl/trainer/dpo_trainer.py:1355: UserWarning: AutoNonVariableTypeMode is deprecated and will be removed in 1.10 release. For kernel implementations please use AutoDispatchBelowADInplaceOrView instead, If you are looking for a user facing API to enable running your inference-only workload, please use c10::InferenceMode. Using AutoDispatchBelowADInplaceOrView in user code is under risk of producing silent wrong result in some edge cases. See Note [AutoDispatchBelowAutograd] for more details. (Triggered internally at build/CMakeFiles/torch_npu.dir/compiler_depend.ts:74.)
      labels[labels == label_pad_token_id] = 0
    Could not estimate the number of tokens of the input, floating-point operations will not be computed
    {'loss': 0.6598, 'grad_norm': 84.0019760131836, 'learning_rate': 4.967808395570435e-07, 'rewards/chosen': -0.15474730730056763, 'rewards/rejected': -0.24127893149852753, 'rewards/accuracies': 0.5799999833106995, 'rewards/margins': 0.0865316167473793, 'logps/rejected': -283.3350830078125, 'logps/chosen': -322.8130187988281, 'logits/rejected': -2.3705289363861084, 'logits/chosen': -2.455843925476074, 'epoch': 0.01}
    {'loss': 0.635, 'grad_norm': 88.99441528320312, 'learning_rate': 4.93561679114087e-07, 'rewards/chosen': -0.4325330853462219, 'rewards/rejected': -0.6220334768295288, 'rewards/accuracies': 0.625, 'rewards/margins': 0.1895003467798233, 'logps/rejected': -298.9117431640625, 'logps/chosen': -323.9031982421875, 'logits/rejected': -2.384589433670044, 'logits/chosen': -2.4548392295837402, 'epoch': 0.01}
    {'eval_loss': 0.6398493647575378, 'eval_runtime': 66.0493, 'eval_samples_per_second': 15.14, 'eval_steps_per_second': 1.893, 'eval_rewards/chosen': -0.5118070840835571, 'eval_rewards/rejected': -0.6984029412269592, 'eval_rewards/accuracies': 0.6269999742507935, 'eval_rewards/margins': 0.18659590184688568, 'eval_logps/rejected': -314.8978271484375, 'eval_logps/chosen': -344.0274658203125, 'eval_logits/rejected': -2.242685556411743, 'eval_logits/chosen': -2.3058021068573, 'epoch': 0.01}
    {'loss': 0.6284, 'grad_norm': 76.736572265625, 'learning_rate': 4.903425186711305e-07, 'rewards/chosen': -0.5593773722648621, 'rewards/rejected': -0.7793089151382446, 'rewards/accuracies': 0.6575000286102295, 'rewards/margins': 0.21993154287338257, 'logps/rejected': -305.3935241699219, 'logps/chosen': -327.55230712890625, 'logits/rejected': -2.3826913833618164, 'logits/chosen': -2.4632484912872314, 'epoch': 0.02}
    {'loss': 0.6126, 'grad_norm': 84.96641540527344, 'learning_rate': 4.87123358228174e-07, 'rewards/chosen': -0.5430492162704468, 'rewards/rejected': -0.8355176448822021, 'rewards/accuracies': 0.6700000166893005, 'rewards/margins': 0.29246845841407776, 'logps/rejected': -327.23089599609375, 'logps/chosen': -353.1753845214844, 'logits/rejected': -2.42580509185791, 'logits/chosen': -2.513734817504883, 'epoch': 0.03}
    {'eval_loss': 0.6248273253440857, 'eval_runtime': 66.049, 'eval_samples_per_second': 15.14, 'eval_steps_per_second': 1.893, 'eval_rewards/chosen': -0.4807929992675781, 'eval_rewards/rejected': -0.7240013480186462, 'eval_rewards/accuracies': 0.6439999938011169, 'eval_rewards/margins': 0.2432083934545517, 'eval_logps/rejected': -315.15380859375, 'eval_logps/chosen': -343.7173767089844, 'eval_logits/rejected': -2.2496635913848877, 'eval_logits/chosen': -2.3131723403930664, 'epoch': 0.03}
    {'loss': 0.6089, 'grad_norm': 78.2168960571289, 'learning_rate': 4.839041977852176e-07, 'rewards/chosen': -0.4145514667034149, 'rewards/rejected': -0.7176669239997864, 'rewards/accuracies': 0.6600000262260437, 'rewards/margins': 0.3031154274940491, 'logps/rejected': -332.7577209472656, 'logps/chosen': -364.76898193359375, 'logits/rejected': -2.436467409133911, 'logits/chosen': -2.530369997024536, 'epoch': 0.03}
    {'loss': 0.5769, 'grad_norm': 75.62930297851562, 'learning_rate': 4.806850373422611e-07, 'rewards/chosen': -0.6155031323432922, 'rewards/rejected': -1.012223720550537, 'rewards/accuracies': 0.7300000190734863, 'rewards/margins': 0.3967204988002777, 'logps/rejected': -291.11419677734375, 'logps/chosen': -331.4106750488281, 'logits/rejected': -2.378931999206543, 'logits/chosen': -2.4724509716033936, 'epoch': 0.04}
    {'eval_loss': 0.6191915273666382, 'eval_runtime': 66.0551, 'eval_samples_per_second': 15.139, 'eval_steps_per_second': 1.892, 'eval_rewards/chosen': -0.7790046334266663, 'eval_rewards/rejected': -1.1102681159973145, 'eval_rewards/accuracies': 0.6460000276565552, 'eval_rewards/margins': 0.33126339316368103, 'eval_logps/rejected': -319.0165100097656, 'eval_logps/chosen': -346.699462890625, 'eval_logits/rejected': -2.2563016414642334, 'eval_logits/chosen': -2.3217742443084717, 'epoch': 0.04}
    {'loss': 0.6159, 'grad_norm': 80.91998291015625, 'learning_rate': 4.774658768993046e-07, 'rewards/chosen': -0.7987264394760132, 'rewards/rejected': -1.1548289060592651, 'rewards/accuracies': 0.6225000023841858, 'rewards/margins': 0.3561025857925415, 'logps/rejected': -307.766357421875, 'logps/chosen': -319.15777587890625, 'logits/rejected': -2.369903326034546, 'logits/chosen': -2.430453300476074, 'epoch': 0.05}
    {'loss': 0.5967, 'grad_norm': 87.26203155517578, 'learning_rate': 4.7424671645634816e-07, 'rewards/chosen': -0.6392844319343567, 'rewards/rejected': -1.015390396118164, 'rewards/accuracies': 0.699999988079071, 'rewards/margins': 0.3761059045791626, 'logps/rejected': -313.5637512207031, 'logps/chosen': -319.4141845703125, 'logits/rejected': -2.3648269176483154, 'logits/chosen': -2.4581611156463623, 'epoch': 0.05}
