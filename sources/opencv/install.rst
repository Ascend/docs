安装指南
==============

OpenCV 4.9.0 版本开始，增加了图像处理相关高频接口的昇腾原生支持，本教程面向使用 OpenCV & 昇腾开发者，帮助完成昇腾环境下 OpenCV 的安装。

昇腾环境安装
------------

请根据已有昇腾产品型号及CPU架构等按照 :doc:`快速安装昇腾环境指引 <../ascend/quick_install>` 进行昇腾环境安装，或直接获取对应产品的昇腾环境镜像 `ascendai/cann <https://hub.docker.com/r/ascendai/cann/tags>`_ 。

.. warning::
  CANN 最低版本为 8.0.rc1，安装 CANN 时，请同时安装 Kernel 算子包。

OpenCV 安装
----------------------


请遵循以下版本控制：

=======  ==========  ==========
  lib     最低版本    推荐版本
=======  ==========  ==========
OpenCV    4.9.0        latest
Python    3.9          3.10
GCC       9.4.0        9.4.0
=======  ==========  ==========

Python 环境创建
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell
    :linenos:
  
    # 创建名为 opencv 的 python 3.10 的虚拟环境
    conda create -y -n opencv python=3.10
    # 激活虚拟环境
    conda activate opencv


源码编译
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. 下载 OpenCV 和 opencv_contrib

.. code-block:: shell
    :linenos:

    git clone https://github.com/opencv/opencv.git

    cd opencv
    git clone https://github.com/opencv/opencv_contrib.git

2. 编译带有 opencv_contrib 的 OpenCV

.. TODO: check for the simplest cmake config
.. code-block:: shell
    :linenos:

    # 在 opencv 项目目录中创建并进入 build 目录
    mkdir build
    cd build

    # cmake & make
    cmake -D CMAKE_BUILD_TYPE=RELEASE 
        -D CMAKE_INSTALL_PREFIX=pwd/install \
        -D WITH_DEBUG=0 \
        -D OPENCV_EXTRA_MODULES_PATH=/path/to/opencv/opencv_contrib/modules \
        -D DWITH_CUDA=0 \
        -D DWITH_CANN=1 \
        -D DPYTHON3_EXECUTABLE=/path/to/miniconda3/envs/opencv/bin/python \
        -D DPYTHON_LIBRARY=/path/to/miniconda3/envs/opencv \
        -D PYTHON_INCLUDE_DIR=/path/to/miniconda3/envs/opencv/include/python3.10 \
        -D BUILD_opencv_wechat_qrcode=OFF \
        -D BUILD_opencv_xfeatures2d=OFF \
        -D BUILD_opencv_face=OFF \
        -D BUILD_opencv_dnn=OFF \
        -D BUILD_opencv_features2d=OFF \
        -D WITH_CAROTENE=OFF \
        -D WITH_IPP=OFF \
        -D BUILD_DOCS=ON \
        -D BUILD_EXAMPLES=ON ..

    make -j5

当编译出现以下关键回显信息时，说明编译成功。

.. code-block:: shell

  # xxx 为 OpenCV 中某模块名称
  [100%] Built target xxx

安装校验
----------------------

通过以下指令执行昇腾算子单元测试：

.. code-block:: shell
    :linenos:

    cd path/to/opencv/build/bin
    ./opencv_test_cannops

出现以下关键回显说明安装成功：

.. code-block:: shell

  [==========] 72 tests from 4 test cases ran. (40937 ms total)
  [  PASSED  ] 72 tests.

