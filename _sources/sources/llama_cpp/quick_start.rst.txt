快速开始
============

.. note::
    阅读本篇前，请确保已按照 :doc:`安装指南 <./install>` 准备好昇腾环境及 llama.cpp ！
    
本教程聚焦大语言模型（Large Language Model，LLM）的推理过程，以 Qwen2.5-7B 模型为例，讲述如何使用 llama.cpp 在昇腾 NPU 上进行推理。


模型文件准备及量化
---------------

llama.cpp 的推理需要使用 gguf 格式文件，llama.cpp 提供了两种方式转换 Hugging Face 模型文件：

- 使用 `GGUF-my-repo <https://huggingface.co/spaces/ggml-org/gguf-my-repo>`_ 将模型进行转换。

- 使用项目中的 `convert_hf_to_gguf.py` 文件将 Hugging Face 模型转换为 gguf 格式:

    .. code-block:: shell 
        :linenos:

        python convert_hf_to_gguf.py path/to/model

详情请参考 `Prepare and Quantize <https://github.com/ggerganov/llama.cpp/blob/master/README.md#prepare-and-quantize>`_ 。

注意：目前仅支持 FP16 精度及 Q4_0/Q8_0 量化模型。

推理
------------

有两种设备选择模式:

- 单设备：使用用户指定的一个设备目标。
- 多设备：自动选择具有相同后端的设备。

+---------------+---------------------------------------------+
|   设备选择    |    参数                                     |
+===============+=============================================+
|   单设备      |    --split-mode none --main-gpu DEVICE_ID   |
+---------------+---------------------------------------------+
|   多设备      |    --split-mode layer (default)             |
+---------------+---------------------------------------------+

使用单卡推理
++++++++++++++++

.. code-block:: shell 
    :linenos:

    ./build/bin/llama-cli -m path_to_model -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33 -sm none -mg 0

使用多卡推理
++++++++++++++++

.. code-block:: shell 
    :linenos:

    ./build/bin/llama-cli -m path_to_model -p "Building a website can be done in 10 simple steps:" -n 400 -e -ngl 33 -sm layer

以下为正常推理结果：

.. code-block:: shell
    :linenos:

    Log start
    main: build = 3520 (8e707118)
    main: built with cc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0 for aarch64-linux-gnu
    main: seed  = 1728907816
    llama_model_loader: loaded meta data with 22 key-value pairs and 291 tensors from /home/jiahao/models/llama3-8b-instruct-fp16.gguf (version GGUF V3 (latest))
    llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
    llama_model_loader: - kv   0:                       general.architecture str              = llama
    llama_model_loader: - kv   1:                               general.name str              = Meta-Llama-3-8B-Instruct
    llama_model_loader: - kv   2:                          llama.block_count u32              = 32
    llama_model_loader: - kv   3:                       llama.context_length u32              = 8192
    llama_model_loader: - kv   4:                     llama.embedding_length u32              = 4096
    llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 14336
    llama_model_loader: - kv   6:                 llama.attention.head_count u32              = 32
    llama_model_loader: - kv   7:              llama.attention.head_count_kv u32              = 8
    llama_model_loader: - kv   8:                       llama.rope.freq_base f32              = 500000.000000
    llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
    llama_model_loader: - kv  10:                          general.file_type u32              = 1
    llama_model_loader: - kv  11:                           llama.vocab_size u32              = 128256
    llama_model_loader: - kv  12:                 llama.rope.dimension_count u32              = 128
    llama_model_loader: - kv  13:                       tokenizer.ggml.model str              = gpt2
    llama_model_loader: - kv  14:                         tokenizer.ggml.pre str              = llama-bpe
    llama_model_loader: - kv  15:                      tokenizer.ggml.tokens arr[str,128256]  = ["!", "\"", "#", "$", "%", "&", "'", ...
    llama_model_loader: - kv  16:                  tokenizer.ggml.token_type arr[i32,128256]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
    llama_model_loader: - kv  17:                      tokenizer.ggml.merges arr[str,280147]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
    llama_model_loader: - kv  18:                tokenizer.ggml.bos_token_id u32              = 128000
    llama_model_loader: - kv  19:                tokenizer.ggml.eos_token_id u32              = 128009
    llama_model_loader: - kv  20:                    tokenizer.chat_template str              = {% set loop_messages = messages %}{% ...
    llama_model_loader: - kv  21:               general.quantization_version u32              = 2
    llama_model_loader: - type  f32:   65 tensors
    llama_model_loader: - type  f16:  226 tensors
    llm_load_vocab: special tokens cache size = 256
    llm_load_vocab: token to piece cache size = 0.8000 MB
    llm_load_print_meta: format           = GGUF V3 (latest)
    llm_load_print_meta: arch             = llama
    llm_load_print_meta: vocab type       = BPE
    llm_load_print_meta: n_vocab          = 128256
    llm_load_print_meta: n_merges         = 280147
    llm_load_print_meta: vocab_only       = 0
    llm_load_print_meta: n_ctx_train      = 8192
    llm_load_print_meta: n_embd           = 4096
    llm_load_print_meta: n_layer          = 32
    llm_load_print_meta: n_head           = 32
    llm_load_print_meta: n_head_kv        = 8
    llm_load_print_meta: n_rot            = 128
    llm_load_print_meta: n_swa            = 0
    llm_load_print_meta: n_embd_head_k    = 128
    llm_load_print_meta: n_embd_head_v    = 128
    llm_load_print_meta: n_gqa            = 4
    llm_load_print_meta: n_embd_k_gqa     = 1024
    llm_load_print_meta: n_embd_v_gqa     = 1024
    llm_load_print_meta: f_norm_eps       = 0.0e+00
    llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
    llm_load_print_meta: f_clamp_kqv      = 0.0e+00
    llm_load_print_meta: f_max_alibi_bias = 0.0e+00
    llm_load_print_meta: f_logit_scale    = 0.0e+00
    llm_load_print_meta: n_ff             = 14336
    llm_load_print_meta: n_expert         = 0
    llm_load_print_meta: n_expert_used    = 0
    llm_load_print_meta: causal attn      = 1
    llm_load_print_meta: pooling type     = 0
    llm_load_print_meta: rope type        = 0
    llm_load_print_meta: rope scaling     = linear
    llm_load_print_meta: freq_base_train  = 500000.0
    llm_load_print_meta: freq_scale_train = 1
    llm_load_print_meta: n_ctx_orig_yarn  = 8192
    llm_load_print_meta: rope_finetuned   = unknown
    llm_load_print_meta: ssm_d_conv       = 0
    llm_load_print_meta: ssm_d_inner      = 0
    llm_load_print_meta: ssm_d_state      = 0
    llm_load_print_meta: ssm_dt_rank      = 0
    llm_load_print_meta: model type       = 8B
    llm_load_print_meta: model ftype      = F16
    llm_load_print_meta: model params     = 8.03 B
    llm_load_print_meta: model size       = 14.96 GiB (16.00 BPW) 
    llm_load_print_meta: general.name     = Meta-Llama-3-8B-Instruct
    llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
    llm_load_print_meta: EOS token        = 128009 '<|eot_id|>'
    llm_load_print_meta: LF token         = 128 'Ä'
    llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
    llm_load_print_meta: max token length = 256
    llm_load_tensors: ggml ctx size =    0.27 MiB
    llm_load_tensors:        CPU buffer size = 15317.02 MiB
    llm_load_tensors:       CANN buffer size = 13313.00 MiB
    .........................................................................................
    llama_new_context_with_model: n_ctx      = 8192
    llama_new_context_with_model: n_batch    = 2048
    llama_new_context_with_model: n_ubatch   = 512
    llama_new_context_with_model: flash_attn = 0
    llama_new_context_with_model: freq_base  = 500000.0
    llama_new_context_with_model: freq_scale = 1
    llama_kv_cache_init:       CANN KV buffer size =  1024.00 MiB
    llama_new_context_with_model: KV self size  = 1024.00 MiB, K (f16):  512.00 MiB, V (f16):  512.00 MiB
    llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
    llama_new_context_with_model:       CANN compute buffer size =  1260.50 MiB
    llama_new_context_with_model:        CPU compute buffer size =    24.01 MiB
    llama_new_context_with_model: graph nodes  = 1030
    llama_new_context_with_model: graph splits = 4

    system_info: n_threads = 192 / 192 | AVX = 0 | AVX_VNNI = 0 | AVX2 = 0 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 0 | NEON = 1 | SVE = 0 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 0 | SSSE3 = 0 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
    sampling: 
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
    sampling order: 
    CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature 
    generate: n_ctx = 8192, n_batch = 2048, n_predict = -1, n_keep = 1


    Building a website can be done in 10 simple steps: 1. Define your website's purpose and target audience 2. Choose a domain name and register it with a registrar 3. Select a web hosting service and set up your hosting account 4. Design your website's layout and structure 5. Create content for your website, including text, images, and other media 6. Build a responsive website design that adapts to different devices and screen sizes 7. Choose a Content Management System (CMS) and install it on your website 8. Customize your website's design and layout using a CMS

    llama_print_timings:        load time =    9074.69 ms
    llama_print_timings:      sample time =      31.97 ms /   112 runs   (    0.29 ms per token,  3503.28 tokens per second)
    llama_print_timings: prompt eval time =     238.53 ms /    13 tokens (   18.35 ms per token,    54.50 tokens per second)
    llama_print_timings:        eval time =   13152.29 ms /   111 runs   (  118.49 ms per token,     8.44 tokens per second)
    llama_print_timings:       total time =   13623.53 ms /   124 tokens