================
昇腾开源文档中心
================

.. raw:: html

   <style>
      :root {
          --ascend-blue: #0066cc; /* 与导航栏颜色严格一致 */
          --text-main: #0066cc;
          --text-sub: #6e6e73;
          --border-color: #e1e4e8;
      }

      /* ：颜色对齐 */
      .hero-banner {
          background-color: var(--ascend-blue);
          color: white; padding: 50px 30px; border-radius: 12px;
          text-align: center; margin-bottom: 40px;
      }
      .hero-banner h1 { color: white !important; border: none !important; margin: 0 !important; font-size: 2.6rem !important; font-weight: 700 !important; }
      .hero-btn {
          display: inline-block; background: white; color: var(--ascend-blue);
          padding: 12px 35px; border-radius: 6px; text-decoration: none;
          font-weight: bold; margin-top: 25px; transition: 0.3s ease;
      }
      .hero-btn:hover { transform: translateY(-3px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }

      /* 分类标题样式 */
      .scene-header {
          border-left: 5px solid var(--ascend-blue); padding-left: 15px;
          margin: 40px 0 25px 0; font-size: 1.6rem; font-weight: bold; color: var(--text-main);
      }

      /* 项目网格布局 */
      .grid-container {
          display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
          gap: 20px; margin-bottom: 50px;
      }

      /* 卡片美化 */
      .project-card {
          border: 1px solid var(--border-color); border-radius: 12px; padding: 24px;
          background: #fff; transition: all 0.3s ease; display: flex; flex-direction: column;
          height: 100%;
      }
      .project-card:hover { transform: translateY(-5px); box-shadow: 0 10px 30px rgba(0,102,204,0.12); border-color: var(--ascend-blue); }

      /* 卡片顶部：logo和标题在卡片中居中显示 */
      .card-top {
          display: grid;
          grid-template-columns: 56px minmax(0, 1fr);
          align-items: center;
          justify-content: center;
          margin-bottom: 15px;
          gap: 16px;
          width: min(100%, 360px);
          margin-left: auto;
          margin-right: auto;
      }

      /* Logo容器：固定尺寸，完美居中 */
      .card-icon {
          width: 56px;
          height: 56px;
          min-width: 56px;
          min-height: 56px;
          flex-shrink: 0;
          background-size: contain;
          background-repeat: no-repeat;
          background-position: center;
          background-color: #f8f9fa;
          border-radius: 8px;
          padding: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
      }

      /* 标题：移除所有默认边距，确保垂直居中 */
      .card-title {
          font-size: 1.25rem;
          font-weight: 600;
          color: var(--text-main);
          margin: 0 !important;
          padding: 0 !important;
          line-height: 1.3;
          display: flex;
          align-items: center;
          min-height: 56px;
          justify-content: flex-start;
          text-align: left;
      }
      .card-desc { font-size: 0.92rem; color: var(--text-sub); line-height: 1.6; margin-bottom: 20px; flex-grow: 1; height: 3.2em; overflow: hidden; }

      /* 三要素分散等距居中显示 */
      .card-footer {
          border-top: 1px solid #f2f2f7;
          padding-top: 15px;
          display: flex;
          justify-content: space-evenly;
          align-items: center;
          font-size: 0.88rem;
      }
      .card-footer a {
          text-decoration: none;
          color: var(--ascend-blue);
          font-weight: 500;
      }
      .card-footer a:hover { text-decoration: underline; }
      .split { display: none; }
   </style>

   <div class="hero-banner">
      <h1>昇腾开源生态全景图</h1>
      <p style="opacity: 0.9; margin-top: 12px; font-size: 1.1rem;">赋能全球开发者，构建极致性能的昇腾 AI 算力底座</p>
      <a href="sources/ascend/quick_install.html" class="hero-btn">🚀 快速安装昇腾环境</a>
   </div>

   <h2 class="scene-header">🏗️  底层 AI 框架与基础设施</h2>
   <div class="grid-container">

      <!-- Accelerate -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/huggingface.png')"></div><h3 class="card-title">accelerate</h3></div>
         <p class="card-desc">适用于 Pytorch 的多 GPUs/NPUs 训练工具链。</p>
         <div class="card-footer"><a href="https://github.com/huggingface/accelerate">官方链接</a><span class="split">|</span><a href="sources/accelerate/install.html">安装指南</a><span class="split">|</span><a href="sources/accelerate/quick_start.html">快速上手</a></div>
      </div>

      <!-- DeepSpeed -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/deepspeed.png')"></div><h3 class="card-title">DeepSpeed</h3></div>
         <p class="card-desc">分布式训练优化库，V0.10.1 版本起支持昇腾。</p>
         <div class="card-footer"><a href="https://github.com/microsoft/DeepSpeed">官方链接</a><span class="split">|</span><a href="sources/deepspeed/install.html">安装指南</a><span class="split">|</span>
         <a href="sources/deepspeed/quick_start.html">快速上手</a></div>
      </div>

      <!-- kernels -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/kernels.png')"></div><h3 class="card-title">kernels</h3></div>
         <p class="card-desc">从 Hugging Face Hub 加载高性能计算算子内核。</p>
         <div class="card-footer"><a href="https://github.com/huggingface/kernels">官方链接</a><span class="split">|</span><a href="sources/kernels/install.html">安装指南</a><span class="split">|</span><a href="sources/kernels/quick_start.html">快速上手</a></div>
      </div>

      <!-- PyTorch -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/pytorch.png')"></div><h3 class="card-title">pytorch</h3></div>
         <p class="card-desc">PyTorch AI 框架，2.1 版本起官方原生支持昇腾 NPU。</p>
         <div class="card-footer"><a href="https://pytorch.org">官方链接</a><span class="split">|</span><a href="sources/pytorch/install.html">安装指南</a><span class="split">|</span><a href="sources/pytorch/quick_start.html">快速上手</a></div>
      </div>

      <!-- Transformers -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/huggingface.png')"></div><h3 class="card-title">transformers</h3></div>
         <p class="card-desc">适用于 Pytorch、TensorFlow 和 JAX 先进的机器学习库。</p>
         <div class="card-footer"><a href="https://huggingface.co/docs/transformers/index">官方链接</a><span class="split">|</span><a href="sources/transformers/install.html">安装指南</a><span class="split">|</span><a href="sources/transformers/quick_start.html">快速上手</a></div>
      </div>

   </div>

   <h2 class="scene-header">🧠 大模型/多模态训练与微调</h2>
   <div class="grid-container">

      <!-- LLaMA-Factory -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/LLaMA-Factory.png')"></div><h3 class="card-title">LLaMA-Factory</h3></div>
         <p class="card-desc">Unified Efficient Fine-Tuning of 100+ LLMs & VLMs。</p>
         <div class="card-footer"><a href="https://github.com/hiyouga/LLaMA-Factory">官方链接</a><span class="split">|</span><a href="sources/_generated/sources/LLaMA-Factory/source/advanced/npu_installation.html">安装指南</a><span class="split">|</span><a href="sources/_generated/sources/LLaMA-Factory/source/advanced/npu_training.html">快速上手</a></div>
      </div>

      <!-- ms-swift -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/swift.png')"></div><h3 class="card-title">ms-swift</h3></div>
         <p class="card-desc">高效微调框架，支持 600+ LLM 和 300+ MLLM 适配昇腾。</p>
         <div class="card-footer"><a href="https://github.com/modelscope/ms-swift">官方链接</a><span class="split">|</span><a href="sources/_generated/sources/ms-swift/source/BestPractices/NPU-support.html">安装指南</a><span class="split">|</span><a href="sources/_generated/sources/ms-swift/source/BestPractices/NPU-support.html">快速上手</a></div>
      </div>

      <!-- ROLL -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/roll.png')"></div><h3 class="card-title">ROLL</h3></div>
         <p class="card-desc">大规模强化学习优化，针对昇腾算力平衡进行了调优。</p>
         <div class="card-footer"><a href="https://github.com/alibaba/ROLL">官方链接</a><span class="split">|</span><a href="sources/roll/install.html">安装指南</a><span class="split">|</span><a href="sources/roll/quick_start.html">快速上手</a></div>
      </div>

      <!-- TorchTitan -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/pytorch.png')"></div><h3 class="card-title">torchtitan</h3></div>
         <p class="card-desc">用于语言大模型训练的 PyTorch 原生库。</p>
         <div class="card-footer"><a href="https://github.com/pytorch/torchtitan">官方链接</a><span class="split">|</span><a href="sources/torchtitan/install.html">安装指南</a><span class="split">|</span><a href="sources/torchtitan/quick_start.html">快速上手</a></div>
      </div>

      <!-- trl -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/huggingface.png')"></div><h3 class="card-title">Transformer Reinforcement Learning</h3></div>
         <p class="card-desc">适用于 SFT、PPO、DPO 等方法的模型后训练库。</p>
         <div class="card-footer"><a href="https://github.com/huggingface/trl">官方链接</a><span class="split">|</span><a href="sources/trl/install.html">安装指南</a><span class="split">|</span><a href="sources/trl/quick_start.html">快速上手</a></div>
      </div>

      <!-- VeOmni -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/VeOmni.png')"></div><h3 class="card-title">VeOmni</h3></div>
         <p class="card-desc">Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo。</p>
         <div class="card-footer"><a href="https://github.com/ByteDance-Seed/VeOmni">官方链接</a><span class="split">|</span><a href="sources/_generated/sources/VeOmni/get_started/installation/install_ascend.html">安装指南</a><span class="split">|</span><a href="sources/_generated/sources/VeOmni/hardware_support/get_started_npu.html">快速上手</a></div>
      </div>

      <!-- verl -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/volcano.png')"></div><h3 class="card-title">verl</h3></div>
         <p class="card-desc">用于 LLM 的强化学习训练库，适配昇腾并行计算方案。</p>
         <div class="card-footer"><a href="https://github.com/volcengine/verl">官方链接</a><span class="split">|</span><a href="sources/_generated/sources/verl/quick_start/ascend_quick_start.html">安装指南</a><span class="split">|</span><a href="sources/_generated/sources/verl/quick_start/ascend_quick_start.html">快速上手</a></div>
      </div>

   </div>

   <h2 class="scene-header">🚀 高性能推理与服务</h2>
   <div class="grid-container">

      <!-- llama.cpp -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/llama_cpp.png')"></div><h3 class="card-title">llama.cpp</h3></div>
         <p class="card-desc">C/C++ 实现的 Meta LLaMa 架构，深度适配昇腾后端。</p>
         <div class="card-footer"><a href="https://github.com/ggerganov/llama.cpp">官方链接</a><span class="split">|</span><a href="sources/llama_cpp/install.html">安装指南</a><span class="split">|</span><a href="sources/llama_cpp/quick_start.html">快速上手</a></div>
      </div>

      <!-- LMDeploy -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/lm-deploy.png')"></div><h3 class="card-title">lmdeploy</h3></div>
         <p class="card-desc">用于压缩、部署和服务 LLM 的工具包。</p>
         <div class="card-footer"><a href="https://github.com/InternLM/lmdeploy">官方链接</a><span class="split">|</span><a href="sources/lm_deploy/install.html">安装指南</a><span class="split">|</span><a href="sources/lm_deploy/quick_start.html">快速上手</a></div>
      </div>

      <!-- ONNX Runtime -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/onnxruntime.png')"></div><h3 class="card-title">onnxruntime</h3></div>
         <p class="card-desc">跨平台高性能推理加速器，v1.13.1 起支持昇腾。</p>
         <div class="card-footer"><a href="https://github.com/microsoft/onnxruntime">官方链接</a><span class="split">|</span><a href="sources/onnxruntime/install.html">安装指南</a><span class="split">|</span><a href="sources/_generated/sources/onnxruntime/quick_start.html">快速上手</a></div>
      </div>

      <!-- Sentence Transformers -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/sentence_transformers.png')"></div><h3 class="card-title">sentence-transformers</h3></div>
         <p class="card-desc">高性能文本和图像 Embedding 库。</p>
         <div class="card-footer"><a href="https://github.com/UKPLab/sentence-transformers">官方链接</a><span class="split">|</span><a href="sources/sentence_transformers/install.html">安装指南</a><span class="split">|</span><a href="sources/sentence_transformers/quick_start.html">快速上手</a></div>
      </div>

      <!-- SGLang -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/sglang.png')"></div><h3 class="card-title">sglang</h3></div>
         <p class="card-desc">用于 LLM 和 VLM 的高速服务框架，极致提升昇腾吞吐量。</p>
         <div class="card-footer"><a href="https://github.com/sgl-project/sglang">官方链接</a><span class="split">|</span><a href="sources/sglang/install.html">安装指南</a><span class="split">|</span><a href="sources/sglang/quick_start.html">快速上手</a></div>
      </div>

      <!-- torchchat -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/pytorch.png')"></div><h3 class="card-title">torchchat</h3></div>
         <p class="card-desc">基于 PyTorch 的对话推理交互库。</p>
         <div class="card-footer"><a href="https://github.com/pytorch/torchchat">官方链接</a><span class="split">|</span><a href="sources/torchchat/install.html">安装指南</a><span class="split">|</span><a href="sources/torchchat/quick_start.html">快速上手</a></div>
      </div>

      <!-- vllm-ascend -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/vllm-ascend.png')"></div><h3 class="card-title">vLLM-Ascend</h3></div>
         <p class="card-desc">面向昇腾 NPU 的 vLLM 社区插件，支持主流大模型高性能推理加速。</p>
         <div class="card-footer"><a href="https://github.com/vllm-project/vllm-ascend">官方链接</a><span class="split">|</span><a href="sources/_generated/sources/vllm-ascend/installation.html">安装指南</a><span class="split">|</span><a href="sources/_generated/sources/vllm-ascend/quick_start.html">快速上手</a></div>
      </div>

   </div>

   <h2 class="scene-header">🎨 多模态应用、评测与工具</h2>
   <div class="grid-container">

      <!-- Diffusers -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/diffusers.png')"></div><h3 class="card-title">diffusers</h3></div>
         <p class="card-desc">扩散模型工具链，支持昇腾 NPU 加速图像生成。</p>
         <div class="card-footer"><a href="https://github.com/huggingface/diffusers">官方链接</a><span class="split">|</span><a href="sources/Diffusers/install.html">安装指南</a><span class="split">|</span><a href="sources/Diffusers/quick_start.html">快速上手</a></div>
      </div>

      <!-- LM-Eval -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/lm-evalution.png')"></div><h3 class="card-title">lm-evaluation-harness</h3></div>
         <p class="card-desc">语言模型评测工具，支持昇腾基准。</p>
         <div class="card-footer"><a href="https://github.com/EleutherAI/lm-evaluation-harness">官方链接</a><span class="split">|</span><a href="sources/lm_evaluation/install.html">安装指南</a><span class="split">|</span><a href="sources/lm_evaluation/quick_start.html">快速上手</a></div>
      </div>

      <!-- Open CLIP -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/huggingface.png')"></div><h3 class="card-title">open_clip</h3></div>
         <p class="card-desc">开源 CLIP 模型实现，支持多模态语义对齐。</p>
         <div class="card-footer"><a href="https://github.com/mlfoundations/open_clip">官方链接</a><span class="split">|</span><a href="sources/open_clip/install.html">安装指南</a><span class="split">|</span><a href="sources/open_clip/quick_start.html">快速上手</a></div>
      </div>

      <!-- OpenCompass -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/opencompass.png')"></div><h3 class="card-title">opencompass</h3></div>
         <p class="card-desc">大模型标准测试工具。</p>
         <div class="card-footer"><a href="https://github.com/open-compass/opencompass">官方链接</a><span class="split">|</span><a href="sources/opencompass/install.html">安装指南</a><span class="split">|</span><a href="sources/opencompass/quick_start.html">快速上手</a></div>
      </div>

      <!-- OpenCV -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/opencv.png')"></div><h3 class="card-title">opencv</h3></div>
         <p class="card-desc">开源计算机视觉库，支持昇腾平台处理。</p>
         <div class="card-footer"><a href="https://github.com/opencv/opencv">官方链接</a><span class="split">|</span><a href="sources/opencv/install.html">安装指南</a><span class="split">|</span><a href="sources/opencv/quick_start.html">快速上手</a></div>
      </div>

      <!-- SD WebUI -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/sd-webui.png')"></div><h3 class="card-title">stable-diffusion-webui</h3></div>
         <p class="card-desc">Stable Diffusion 可视化工具链。</p>
         <div class="card-footer"><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">官方链接</a><span class="split">|</span><a href="sources/sd_webui/install.html">安装指南</a><span class="split">|</span><a href="sources/sd_webui/quick_start.html">快速上手</a></div>
      </div>

      <!-- timm -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/huggingface.png')"></div><h3 class="card-title">pytorch-image-models</h3></div>
         <p class="card-desc">PyTorch 图像模型库适配版本。</p>
         <div class="card-footer"><a href="https://github.com/huggingface/pytorch-image-models">官方链接</a><span class="split">|</span><a href="sources/timm/install.html">安装指南</a><span class="split">|</span><a href="sources/timm/quick_start.html">快速上手</a></div>
      </div>

      <!-- WeNet -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/wenet.png')"></div><h3 class="card-title">wenet</h3></div>
         <p class="card-desc">端到端的语音识别工具包。</p>
         <div class="card-footer"><a href="https://github.com/wenet-e2e/wenet">官方链接</a><span class="split">|</span><a href="sources/wenet/install.html">安装指南</a><span class="split">|</span><a href="sources/wenet/quick_start.html">快速上手</a></div>
      </div>

      <!-- Whisper.cpp -->
      <div class="project-card">
         <div class="card-top"><div class="card-icon" style="background-image: url('_static/images/whisper_cpp.png')"></div><h3 class="card-title">whisper.cpp</h3></div>
         <p class="card-desc">Whisper 模型高性能推理语音识别框架。</p>
         <div class="card-footer"><a href="https://github.com/ggerganov/whisper.cpp">官方链接</a><span class="split">|</span><a href="sources/whisper_cpp/install.html">安装指南</a><span class="split">|</span><a href="sources/whisper_cpp/quick_start.html">快速上手</a></div>
      </div>

   </div>

.. -----------------------------------------
.. 权威 TOC 架构：确保侧边栏持久化
.. -----------------------------------------

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🏁 开始使用

   sources/ascend/quick_install.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 🏗️  基础设施与框架

   sources/accelerate/index.rst
   sources/deepspeed/index.rst
   sources/kernels/index.rst
   sources/pytorch/index.rst
   sources/transformers/index.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 🧠 训练与微调框架

   sources/LLaMA-Factory/index.rst
   sources/ms-swift/index.rst
   sources/roll/index.rst
   sources/torchtitan/index.rst
   sources/trl/index.rst
   sources/VeOmni/index.rst
   sources/verl/index.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 🚀 推理与服务

   sources/llama_cpp/index.rst
   sources/lm_deploy/index.rst
   sources/onnxruntime/index.rst
   sources/sentence_transformers/index.rst
   sources/sglang/index.rst
   sources/torchchat/index.rst
   sources/vllm-ascend/index.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 🎨 多模态、应用与评测

   sources/Diffusers/index.rst
   sources/lm_evaluation/index.rst
   sources/open_clip/index.rst
   sources/opencompass/index.rst
   sources/opencv/index.rst
   sources/sd_webui/index.rst
   sources/timm/index.rst
   sources/wenet/index.rst
   sources/whisper_cpp/index.rst

