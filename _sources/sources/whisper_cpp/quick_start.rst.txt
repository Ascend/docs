快速开始
==================

.. note::

    阅读本篇前，请确保已按照 :doc:`安装教程 <./install>` 准备好昇腾环境及 Whisper.cpp ！
    
本文档帮助昇腾开发者快速使用 Whisper.cpp × 昇腾 进行自动语音识别（Automatic Speech Recognition, ASR）。

Whisper 模型下载
---------------------

Whisper 模型是 OpenAI 训练并开源的 ASR 神经网络模型，是当前 ASR 领域主流模型之一。
在 Whisper.cpp 中进行语音识别，需要下载 Whisper 模型并加载其 gguf 格式权重文件。
本文提供三种模型的获取方式，请根据需要选择一种即可。

.. note::

    gguf 是一种储存神经网络权重的文件格式，是一种二进制格式，旨在快速加载和保存模型，详见 `ggml 官方文档 <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>`_

1. 使用脚本下载
~~~~~~~~~~~~~~~~~~~~

使用 Whisper.cpp 项目中的 ``download-ggml-model.sh`` 脚本下载预先转换为 gguf 格式的 Whisper 模型：

.. code-block:: shell
    :linenos:

    ./download-ggml-model.sh base.en

其中 ``base.en`` 可替换为所需 Whisper 模型名称，Whisper 模型名称清单：

.. code-block:: python
    :linenos:

    # Whisper models
    models="tiny
            tiny.en
            tiny-q5_1
            tiny.en-q5_1
            base
            base.en
            base-q5_1
            base.en-q5_1
            small
            small.en
            small.en-tdrz
            small-q5_1
            small.en-q5_1
            medium
            medium.en
            medium-q5_0
            medium.en-q5_0
            large-v1
            large-v2
            large-v2-q5_0
            large-v3
            large-v3-q5_0"

2. 手动下载
~~~~~~~~~~~~~~~~~~~~

预先转换为 gguf 格式的 Whisper 模型可由此处下载:

- https://huggingface.co/ggerganov/whisper.cpp/tree/main
- https://ggml.ggerganov.com

3. 自行转换模型
~~~~~~~~~~~~~~~~~~~~

从 `OpenAI 提供的模型 <https://github.com/openai/whisper/blob/main/whisper/__init__.py#L17-L30>`_ 中选择一个下载，使用以下指令完成其到 gguf 模型的转换，并将其移动至 ``./models/`` 目录下：

.. code-block:: shell
    :linenos:

    python models/convert-pt-to-ggml.py ~/.cache/whisper/medium.pt ~/path/to/repo/whisper/ ./models/whisper-medium
    mv ./models/whisper-medium/ggml-model.bin models/ggml-medium.bin


语音文件预处理
---------------------

使用 ffmpeg 转换所需处理的语音文件为 16 bit wav 语音文件，此处以 ``samples/gb0.ogg`` 为例：

.. code-block:: shell
    :linenos:

    ffmpeg -loglevel -0 -y -i samples/gb0.ogg -ar 16000 -ac 1 -c:a pcm_s16le samples/gb0.wav


自动语音识别
---------------------

使用以下指令，即可完成在昇腾 NPU 上的 Whisper.cpp 自动语音识别：

.. code-block:: shell
    :linenos:

    ./build/bin/main -f samples/jfk.wav -m models/ggml-base.en.bin -t 8


输出语音识别结果与对应语音内容一致表明识别正确，以下为 ``samples/jfk.wav`` 语音的正确回显示例：

.. code-block:: shell

    whisper_init_from_file_with_params_no_state: loading model from 'models/ggml-base.en.bin'
    whisper_init_with_params_no_state: use gpu    = 1
    whisper_init_with_params_no_state: flash attn = 0
    whisper_init_with_params_no_state: gpu_device = 0
    whisper_init_with_params_no_state: dtw        = 0
    whisper_model_load: loading model
    whisper_model_load: n_vocab       = 51864
    whisper_model_load: n_audio_ctx   = 1500
    whisper_model_load: n_audio_state = 512
    whisper_model_load: n_audio_head  = 8
    whisper_model_load: n_audio_layer = 6
    whisper_model_load: n_text_ctx    = 448
    whisper_model_load: n_text_state  = 512
    whisper_model_load: n_text_head   = 8
    whisper_model_load: n_text_layer  = 6
    whisper_model_load: n_mels        = 80
    whisper_model_load: ftype         = 1
    whisper_model_load: qntvr         = 0
    whisper_model_load: type          = 2 (base)
    whisper_model_load: adding 1607 extra tokens
    whisper_model_load: n_langs       = 99
    whisper_model_load:      CPU total size =   147.37 MB
    whisper_model_load: model size    =  147.37 MB
    whisper_backend_init_gpu: using CANN backend
    whisper_init_state: kv self size  =   18.87 MB
    whisper_init_state: kv cross size =   18.87 MB
    whisper_init_state: kv pad  size  =    3.15 MB
    whisper_init_state: compute buffer (conv)   =   16.75 MB
    whisper_init_state: compute buffer (encode) =  131.94 MB
    whisper_init_state: compute buffer (cross)  =    5.17 MB
    whisper_init_state: compute buffer (decode) =  153.13 MB

    system_info: n_threads = 8 / 192 | AVX = 0 | AVX2 = 0 | AVX512 = 0 | FMA = 0 | NEON = 1 | ARM_FMA = 1 | METAL = 0 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 | CUDA = 0 | COREML = 0 | OPENVINO = 0 | CANN = 1

    main: processing 'samples/jfk.wav' (176000 samples, 11.0 sec), 8 threads, 1 processors, 5 beams + best of 5, lang = en, task = transcribe, timestamps = 1 ...


    [00:00:00.000 --> 00:00:11.000]   And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.


    whisper_print_timings:     load time =   223.83 ms
    whisper_print_timings:     fallbacks =   0 p /   0 h
    whisper_print_timings:      mel time =    19.95 ms
    whisper_print_timings:   sample time =    94.43 ms /   131 runs (    0.72 ms per run)
    whisper_print_timings:   encode time =   632.05 ms /     1 runs (  632.05 ms per run)
    whisper_print_timings:   decode time =    56.30 ms /     2 runs (   28.15 ms per run)
    whisper_print_timings:   batchd time =   930.68 ms /   125 runs (    7.45 ms per run)
    whisper_print_timings:   prompt time =     0.00 ms /     1 runs (    0.00 ms per run)
    whisper_print_timings:    total time =  2854.32 ms
