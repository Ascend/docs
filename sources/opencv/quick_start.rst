快速开始
==================

OpenCV 中昇腾算子入参列表和 cpu 及 cuda 算子保持一致，除了对昇腾必要的初始化、去初始化之外，用户无需学习 CANN API，仅需要将原来的接口添加 cann 包名（C++ 接口为使用 cann 命名空间），整体流程如下图所示：

.. figure:: ./images/opencv_cannop.png
  :align: center
  :scale: 70%



图像处理
-------------
OpenCV 当前支持 20+ 昇腾算子，此处根据图像处理应用场景，选取 ``add``， ``rotate`` 和 ``flip`` 算子的应用作示例代码，
更多算子见 `OpenCV 官方文档 <https://docs.opencv.org/4.x/df/d88/group__cannops__ops.html>`_。

使用 C++
~~~~~~~~~~~~~

.. note::
    
    通过命令行传参 ``input`` 和 ``output`` 来指定输入和输出图像路径

.. code-block:: c++
    :linenos:
    :emphasize-lines: 34,35,38,40,42,48,49

    // This file is part of OpenCV project.
    // It is subject to the license terms in the LICENSE file found in the top-level directory
    // of this distribution and at http://opencv.org/license.html.
    
    #include <iostream>
    #include <opencv2/imgcodecs.hpp>
    #include <opencv2/cann.hpp>
    #include <opencv2/cann_interface.hpp>
    
    int main(int argc, char* argv[])
    {
        cv::CommandLineParser parser(argc, argv,
        "{@input|puppy.png|path to input image}"
        "{@output|output.png|path to output image}"
        "{help||show help}");
        parser.about("This is a sample for image processing with Ascend NPU. \n");
        if (argc != 3 || parser.has("help"))
        {
            parser.printMessage();
            return 0;
        }
    
        std::string imagePath = parser.get<std::string>(0);
        std::string outputPath = parser.get<std::string>(1);
        
        // 读取输入图像
        cv::Mat img = cv::imread(imagePath);
        // 生成高斯噪声
        cv::Mat gaussNoise(img.rows, img.cols, img.type());
        cv::RNG rng;
        rng.fill(gaussNoise, cv::RNG::NORMAL, 0, 25);
        
        // cann 初始化及指定设备
        cv::cann::initAcl();
        cv::cann::setDevice(0);
        
        cv::Mat output;
        // 添加高斯噪声到输入图像
        cv::cann::add(img, gaussNoise, output);
        // 旋转图像 (0, 1, 2, 分别代表旋转 90°, 180°, 270°)
        cv::cann::rotate(output, output, 0);
        // 翻转图像 (0, 正数, 负数, 分别代表沿 x, y, x 和 y 轴进行翻转)
        cv::cann::flip(output, output, 0);
        // 写入输出图像
        cv::imwrite(outputPath, output);

        // cann 去初始化
        cv::cann::resetDevice();
        cv::cann::finalizeAcl();
        return 0;
    }

使用 Python
~~~~~~~~~~~~~

.. note::

    通过命令行传参 ``input`` 和 ``output`` 来指定输入和输出图像路径

.. code-block:: python
    :linenos:
    :emphasize-lines: 20,21,24,26,28,33

    # This file is part of OpenCV project.
    # It is subject to the license terms in the LICENSE file found in the top-level directory
    # of this distribution and at http://opencv.org/license.html.
    
    import numpy as np
    import cv2
    import argparse
    
    parser = argparse.ArgumentParser(description='This is a sample for image processing with Ascend NPU.')
    parser.add_argument('image', help='path to input image')
    parser.add_argument('output', help='path to output image')
    args = parser.parse_args()
    
    # 读取输入图像
    img = cv2.imread(args.image)
    # 生成高斯噪声
    gaussNoise = np.random.normal(0, 25,(img.shape[0], img.shape[1], img.shape[2])).astype(img.dtype)
    
    # cann 初始化及指定设备
    cv2.cann.initAcl()
    cv2.cann.setDevice(0)
    
    # 添加高斯噪声到输入图像
    output = cv2.cann.add(img, gaussNoise)
    # 旋转图像 (0, 1, 2, 分别代表旋转 90°, 180°, 270°)
    output = cv2.cann.rotate(output, 0)
    # 翻转图像 (0, 正数, 负数, 分别代表沿 x, y, x 和 y 轴进行翻转)
    output = cv2.cann.flip(output, 0)
    # 写入输出图像
    cv2.imwrite(args.output, output)

    # cann 去初始化
    cv2.cann.finalizeAcl()


图像处理结果
~~~~~~~~~~~~~~~~~

本示例使用输入图像如图所示：

.. figure:: ./images/input.png
  :align: center
  :scale: 50%

通过上述 Python 或 C++ 示例代码处理，得到的输出图像为：

.. figure:: ./images/result.png
  :align: center
  :scale: 50%

