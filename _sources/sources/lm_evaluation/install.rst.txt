安装指南
==============

本教程面向使用lm-evaluation-harnes&昇腾的开发者，帮助完成昇腾环境下lm-evaluation-harness的安装。

.. note::

    请确保已经根据 `快速安装昇腾环境 <../ascend/quick_install.html>`_ 指引安装了对应的CANN-toolkit版本以及相应的固件和驱动，并应用了CANN-toolkit环境变量。

.. warning::

    lm-evaluation-harness支持的CANN最低版本为8.0.rc1。安装CANN时，请同事安装Kernel算子包。


lm-evaluation-harness安装
----------------------------------

注意：lm-evaluation-harness从0.4.3开始原生支持昇腾。

- Option 1: Use the latest stable release

.. code-block:: shell
    :linenos:

    pip install --upgrade-strategy=conservative lm-eval


- Option 2: Use the latest main branch under development

.. code-block:: shell
    :linenos:

    pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git


安装校验
----------------

使用以下指令对lm-evaluation-harness的安装进行校验：

.. code-block:: shell
    :linenos:

    lm-eval -h


如下所示，正确显示 `lm-eval` 命令的帮助信息即说明安装成功。

.. code-block:: shell
    :linenos:

    usage: lm-eval [-h] [--model MODEL] [--tasks task1,task2] [--model_args MODEL_ARGS] [--num_fewshot N]
                [--batch_size auto|auto:N|N] [--max_batch_size N] [--device DEVICE]
                [--output_path DIR|DIR/file.json] [--limit N|0<N<1] [--use_cache DIR]
                [--cache_requests {true,refresh,delete}] [--check_integrity] [--write_out] [--log_samples]
                [--system_instruction SYSTEM_INSTRUCTION] [--apply_chat_template] [--fewshot_as_multiturn]
                [--show_config] [--include_path DIR] [--gen_kwargs GEN_KWARGS]
                [--verbosity CRITICAL|ERROR|WARNING|INFO|DEBUG] [--wandb_args WANDB_ARGS]
                [--hf_hub_log_args HF_HUB_LOG_ARGS] [--predict_only] [--seed SEED] [--trust_remote_code]

    options:
    -h, --help            show this help message and exit
    --model MODEL, -m MODEL
                            Name of model e.g. `hf`
    ...


lm-evaluation-harness卸载
----------------------------------

.. code-block:: shell
    :linenos:

    pip uninstall lm-eval
