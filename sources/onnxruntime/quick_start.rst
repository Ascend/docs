快速开始
===========

.. note::
    阅读本篇前，请确保已按照 :doc:`安装指南 <./install>` 准备好昇腾环境及 ONNX Runtime!
    
本教程以一个简单的 resnet50 模型为例，讲述如何在 Ascend NPU上使用 ONNX Runtime 进行模型推理。

环境准备
-----------

安装本教程所依赖的额外必要库。

.. code-block:: shell
  :linenos:

  pip install numpy Pillow onnx

模型准备
-----------

ONNX Runtime 推理需要 ONNX 格式模型作为输入，目前有以下几种主流途径获得 ONNX 模型。

1. 从 `ONNX Model Zoo <https://onnx.ai/models/>`_ 中下载模型。
2. 从 torch、TensorFlow 等框架导出 ONNX 模型。
3. 使用转换工具，完成其他类型到 ONNX 模型的转换。

本教程使用的 resnet50 模型是从 ONNX Model Zoo 中直接下载的，具体的 `下载链接 <https://github.com/onnx/models/blob/main/Computer_Vision/resnet50_Opset16_torch_hub/resnet50_Opset16.onnx>`_

类别标签
-----------

类别标签用于将输出权重转换成人类可读的类别信息，具体的 `下载链接 <https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json>`_

模型推理
-----------

.. code-block:: python
  :linenos:

  import onnxruntime as ort
  import numpy as np
  import onnx
  from PIL import Image

  def preprocess(image_path):
      img = Image.open(image_path)
      img = img.resize((224, 224))
      img = np.array(img).astype(np.float32)

      img = np.transpose(img, (2, 0, 1))
      img = img / 255.0
      mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
      std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
      img = (img - mean) / std
      img = np.expand_dims(img, axis=0)
      return img

  def inference(model_path, img):
      options = ort.SessionOptions()
      providers = [
          (
              "CANNExecutionProvider",
              {
                  "device_id": 0,
                  "arena_extend_strategy": "kNextPowerOfTwo",
                  "npu_mem_limit": 2 * 1024 * 1024 * 1024,
                  "op_select_impl_mode": "high_performance",
                  "optypelist_for_implmode": "Gelu",
                  "enable_cann_graph": True
              },
          ),
          "CPUExecutionProvider",
      ]

      session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
      input_name = session.get_inputs()[0].name
      output_name = session.get_outputs()[0].name

      result = session.run([output_name], {input_name: img})
      return result

  def display(classes_path, result):
      with open(classes_path) as f:
          labels = [line.strip() for line in f.readlines()]
      
      pred_idx = np.argmax(result)
      print(f'Predicted class: {labels[pred_idx]} ({result[0][0][pred_idx]:.4f})')

  if __name__ == '__main__':
      model_path = '~/model/resnet/resnet50.onnx'
      image_path = '~/model/resnet/cat.jpg'
      classes_path = '~/model/resnet/imagenet_classes.txt'

      img = preprocess(image_path)
      result = inference(model_path, img)
      display(classes_path, result)
