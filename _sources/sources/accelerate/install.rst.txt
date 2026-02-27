安装指南
==============

本教程面向使用 Accelerate & 昇腾的开发者，帮助完成昇腾环境下 Accelerate 的安装。

Accelerate 下载安装
--------------------

.. note::
    
    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境！
    或者直接使用具备昇腾环境的镜像 `ascendai/cann:8.0.rc1-910b-ubuntu22.04 <https://hub.docker.com/layers/ascendai/cann/8.0.rc1-910b-ubuntu22.04/images/sha256-29ef8aacf6b2babd292f06f00b9190c212e7c79a947411e213135e4d41a178a9?context=explore>`_,
    更多的版本可至 `ascendai/cann <https://hub.docker.com/r/ascendai/cann/tags>`_ 获取。

启动镜像
:::::::::::::::::

.. code-block:: shell
  
  docker run -itd --network host -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /etc/ascend_install.info:/etc/ascend_install.info --device /dev/davinci7 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc --shm-size 16G --name accelerate ascendai/cann:8.0.rc1-910b-ubuntu22.04 bash

安装 Accelerate 及依赖包
::::::::::::::::::::::::::

.. code-block:: shell

  pip install torch==2.2.0 torch_npu==2.2.0 accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple

