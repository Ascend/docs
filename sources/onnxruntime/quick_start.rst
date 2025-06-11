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

1. python推理示例
-----------------------
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

2. C++推理示例
-----------------------
.. code-block:: c++
  :linenos:

    #include <iostream>
    #include <vector>

    #include "onnxruntime_cxx_api.h"

    // path of model, Change to user's own model path
    const char* model_path = "./onnx/resnet50_Opset16.onnx";

    /**
    * @brief Input data preparation provided by user.
    *
    * @param num_input_nodes The number of model input nodes.
    * @return  A collection of input data.
    */
    std::vector<std::vector<float>> input_prepare(size_t num_input_nodes) {
        std::vector<std::vector<float>> input_datas;
        input_datas.reserve(num_input_nodes);

        constexpr size_t input_data_size = 3 * 224 * 224;
        std::vector<float> input_data(input_data_size);
        // initialize input data with values in [0.0, 1.0]
        for (unsigned int i = 0; i < input_data_size; i++)
            input_data[i] = (float)i / (input_data_size + 1);
        input_datas.push_back(input_data);

        return input_datas;
    }

    /**
    * @brief Model output data processing logic(For User updates).
    *
    * @param output_tensors The results of the model output.
    */
    void output_postprocess(std::vector<Ort::Value>& output_tensors) {
        auto floatarr = output_tensors.front().GetTensorMutableData<float>();

        for (int i = 0; i < 5; i++) {
            std::cout << "Score for class [" << i << "] =  " << floatarr[i] << '\n';
        }
        
        std::cout << "Done!" << std::endl;
    }

    /**
    * @brief The main functions for model inference.
    *
    *  The complete model inference process, which generally does not need to be
    * changed here
    */
    void inference() {
        const auto& api = Ort::GetApi();
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING);

        // Enable cann graph in cann provider option.
        OrtCANNProviderOptions* cann_options = nullptr;
        api.CreateCANNProviderOptions(&cann_options);

        // Configurations of EP
        std::vector<const char*> keys{
            "device_id",
            "npu_mem_limit",
            "arena_extend_strategy",
            "enable_cann_graph"};
        std::vector<const char*> values{"0", "4294967296", "kNextPowerOfTwo", "1"};
        api.UpdateCANNProviderOptions(
            cann_options, keys.data(), values.data(), keys.size());

        // Convert to general session options
        Ort::SessionOptions session_options;
        api.SessionOptionsAppendExecutionProvider_CANN(
            static_cast<OrtSessionOptions*>(session_options), cann_options);

        Ort::Session session(env, model_path, session_options);

        Ort::AllocatorWithDefaultOptions allocator;

        // Input Process
        const size_t num_input_nodes = session.GetInputCount();
        std::vector<const char*> input_node_names;
        std::vector<Ort::AllocatedStringPtr> input_names_ptr;
        input_node_names.reserve(num_input_nodes);
        input_names_ptr.reserve(num_input_nodes);
        std::vector<std::vector<int64_t>> input_node_shapes;
        std::cout << num_input_nodes << std::endl;
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session.GetInputNameAllocated(i, allocator);
            input_node_names.push_back(input_name.get());
            input_names_ptr.push_back(std::move(input_name));
            auto type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_node_shapes.push_back(tensor_info.GetShape());
        }

        // Output Process
        const size_t num_output_nodes = session.GetOutputCount();
        std::vector<const char*> output_node_names;
        std::vector<Ort::AllocatedStringPtr> output_names_ptr;
        output_names_ptr.reserve(num_input_nodes);
        output_node_names.reserve(num_output_nodes);
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            output_node_names.push_back(output_name.get());
            output_names_ptr.push_back(std::move(output_name));
        }

        //  User need to generate input date according to real situation.
        std::vector<std::vector<float>> input_datas = input_prepare(num_input_nodes);

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(num_input_nodes);
        for (size_t i = 0; i < input_node_shapes.size(); i++) {
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_datas[i].data(),
                input_datas[i].size(),
                input_node_shapes[i].data(),
                input_node_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(),
            input_tensors.data(),
            num_input_nodes,
            output_node_names.data(),
            output_node_names.size());

        // Processing of out_tensor
        output_postprocess(output_tensors);
    }

    int main(int argc, char* argv[]) {
        inference();
        return 0;
    }