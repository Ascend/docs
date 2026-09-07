安装指南
==============

本教程面向使用 Whisper.cpp & 昇腾的开发者，帮助完成昇腾环境下 Whisper.cpp 的安装。

昇腾环境安装
------------

请根据已有昇腾产品型号及CPU架构等按照 :doc:`快速安装昇腾环境指引 <../ascend/quick_install>` 进行昇腾环境安装。

.. warning::
  CANN 最低版本为 8.0.rc1，安装 CANN 时，请同时安装 Kernel 算子包。

Whisper.cpp 编译安装
----------------------

1. 下载 Whisper.cpp 项目到本地

.. code-block:: shell
    :linenos:

    git clone https://github.com/ggerganov/whisper.cpp.git

2. 在 Whisper.cpp 项目目录下，创建构建目录并进入该目录

.. code-block:: shell
    :linenos:

    mkdir build
    cd build


3. 编译安装 CANN 版本的 Whisper.cpp

.. code-block:: shell
    :linenos:

    cmake .. -D GGML_CANN=on
    make -j


安装校验
----------------------

编译完毕后，无任何报错信息，并输出以下关键回显即说明安装成功：

.. code-block:: shell

  [ 90%] Built target quantize
  [ 95%] Linking CXX executable ../../bin/main
  [ 95%] Built target main
  [100%] Linking CXX executable ../../bin/server
  [100%] Built target server
