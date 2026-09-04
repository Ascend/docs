快速开始
==================

.. note::

    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 WeNet ！
    
本文档帮助昇腾开发者快速使用 WeNet × 昇腾 进行自动语音识别（Automatic Speech Recognition, ASR）模型的训练、推理和评估等。

WeNet 提供了多种数据集及模型的实验脚本，该脚本将实验分为几个阶段，包含数据集的下载、模型的训练、推理、评估等，均存放在 `examples <https://github.com/wenet-e2e/wenet/blob/main/examples/>`_ 路径下，
本篇以 aishell-1 数据集的实验为例，基于 WeNet `官方教程 <https://wenet.org.cn/wenet/tutorial_aishell.html#>`_ ，
详述如何使用 `NPU 实验脚本 <https://github.com/wenet-e2e/wenet/blob/main/examples/aishell/s0/run_npu.sh>`_ 进行从零开始的语音模型训练。

首先进入该脚本所在目录下：

.. code-block:: shell
    :linenos:

    cd example/aishell/s0

下载数据
~~~~~~~~~~~~

stage -1 阶段将 aishell-1 数据下载到本地路径 ``$data``：

.. code-block:: shell
    :linenos:

    bash run_npu.sh --stage -1 --stop_stage -1


如果已下载数据，请更改 ``run_npu.sh`` 脚本中的变量 ``$data`` 值为实际数据集存放的绝对路径，并从下一阶段开始。

准备训练数据
~~~~~~~~~~~~

stage 0 阶段为训练数据准备阶段，将使用 ``local/aishell_data_prep.sh`` 脚本将训练数据重新组织为 ``wav.scp`` 和 ``text`` 两部分。

.. note::

    ``wav.scp`` 每行记录两个制表符分隔的列： ``wav_id`` 和 ``wav_path``,
    ``text`` 每行记录两个制表符分隔的列： ``wav_id`` 和 ``text_label``。

.. code-block:: shell
    :linenos:

    bash run_npu.sh --stage 0 --stop_stage 0


提取最佳 cmvn 特征（可选）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

stage 1 阶段从训练数据中提取 cmvn 特征，本阶段为可选阶段，设置 ``cmvn=false`` 可跳过本阶段。

.. code-block:: shell
    :linenos:

    bash run_npu.sh --stage 1 --stop_stage 1

``tools/compute_cmvn_stats.py`` 用于提取全局 cmvn（倒谱均值和方差归一化）统计数据，用来归一化声学特征。

生成 token 字典
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

stage 2 阶段生成训练所需 token 字典，用于 CTC 解码阶段查询，将输出转换为文字。

.. code-block:: shell
    :linenos:

    bash run_npu.sh --stage 2 --stop_stage 2


准备 WeNet 数据格式
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

stage 3 阶段生成 WeNet 所需格式的文件 ``data.list``：

.. code-block:: shell
    :linenos:

    bash run_npu.sh --stage 3 --stop_stage 3

生成的 ``data.list``每一行都是 json 格式，包含 关键词 ``key`` （文件名称），
语音文件地址 ``wav`` 和 对应文本内容 ``txt`` 三个关键数据。如下为一示例：

.. code-block:: shell

    {"key": "BAC009S0002W0122", "wav": "/export/data/asr-data/OpenSLR/33//data_aishell/wav/train/S0002/BAC009S0002W0122.wav", "txt": "而对楼市成交抑制作用最大的限购"}

模型训练
~~~~~~~~~~

stage 4 为模型训练阶段， ``run_npu.sh`` 脚本中实现了 NPU 卡号的自动获取和相关环境变量设置，因此可直接通过以下启动昇腾 NPU 上的模型训练：

.. code-block:: shell
    :linenos:

    bash run_npu.sh --stage 4 --stop_stage 4

如需自行指定 NPU 卡号，请更改 ``run_npu.sh`` 脚本中的变量 ``ASCEND_RT_VISIBLE_DEVICES`` 值为指定卡号。

.. note::

    有关断点重训，参数配置等，请参考 `WeNet 官方文档 <https://wenet.org.cn/wenet/tutorial_aishell.html#stage-4-neural-network-training>`_ 。

测试推理
~~~~~~~~~~~~~~~

stage 5 为模型测试推理阶段，将测试集中语音文件识别为文本：

.. code-block:: shell
    :linenos:

    bash run_npu.sh --stage 5 --stop_stage 5

此外，stage 5 还提供平均模型的功能，平均模型指当 ``${average_checkpoint}``为 ``true`` 时，
将交叉验证集上的最佳的 ``${average_num}`` 个模型平均，生成增强模型。

.. note::

    此阶段还提供解码和 WER 模型评估等功能，详细信息请参考 WeNet `官方文档 <https://wenet.org.cn/wenet/tutorial_aishell.html#stage-5-recognize-wav-using-the-trained-model>`_ 。


导出训练好的模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

stage 6 为模型导出阶段， ``wenet/bin/export_jit.py`` 使用 ``Libtorch`` 导出以上训练好的模型，导出的模型可用于其他编程语言（如 C++）的推理。

.. code-block:: shell
    :linenos:

    bash run_npu.sh --stage 6 --stop_stage 6
