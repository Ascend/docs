verl - 大模型强化学习训练框架
============================================

.. raw:: html

   <style>
      /* 样式隔离：仅作用于首页入口 */
      #verl-portal {
          --v-primary: #0066cc;
          --v-secondary: #00a86b;
          --v-accent: #ff6b35;
          --v-purple: #9d4edd;
          --v-text-main: #1a1a1a;
          --v-text-sub: #666666;
          --v-border: #e1e4e8;
          --v-bg-light: #f6f8fa;
          font-family: -apple-system, system-ui, Segoe UI, Roboto, Helvetica;
      }

      /* 英雄区 */
      .verl-hero {
          background: linear-gradient(135deg, var(--v-primary) 0%, #004a99 100%);
          color: white;
          padding: 60px 40px;
          border-radius: 12px;
          margin: 20px 0 40px 0;
          text-align: center;
          box-shadow: 0 8px 24px rgba(0, 102, 204, 0.15);
      }
      .verl-hero h1 { color: white !important; border: none !important; margin: 0 0 15px 0 !important; font-size: 2.8rem !important; }
      .verl-hero-subtitle { font-size: 1.2rem; opacity: 0.95; margin-bottom: 30px; }
      .verl-hero-buttons { display: flex; gap: 15px; justify-content: center; flex-wrap: wrap; }
      .verl-btn {
          padding: 12px 30px; border-radius: 6px; text-decoration: none !important;
          font-weight: 600; transition: all 0.3s ease; border: 2px solid white;
      }
      .verl-btn-primary { background: white; color: var(--v-primary) !important; }
      .verl-btn-secondary { background: transparent; color: white !important; }
      .verl-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 16px rgba(0,0,0,0.2); }

      /* 核心特性 - 恢复原版 6 卡片风格 */
      .verl-section-title { text-align: center; color: var(--v-primary); margin: 50px 0 30px 0; font-size: 2rem; border: none !important; }
      .verl-features {
          display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
          gap: 25px; margin-bottom: 50px;
      }
      .verl-feature-card {
          background: white; border: 1px solid var(--v-border);
          border-radius: 10px; padding: 25px; transition: all 0.3s ease;
      }
      .verl-feature-card:hover { border-color: var(--v-primary); transform: translateY(-4px); box-shadow: 0 12px 24px rgba(0, 102, 204, 0.1); }
      .verl-feature-icon { font-size: 2.5rem; margin-bottom: 15px; }
      .verl-feature-card h3 { color: var(--v-primary); margin: 0 0 12px 0 !important; font-size: 1.3rem !important; border: none !important; }
      .verl-feature-card p { color: var(--v-text-sub); line-height: 1.6; margin: 0; font-size: 0.95rem; }

      /* 开发者路径 */
      .verl-path-wrapper {
          background: var(--v-bg-light); border-radius: 12px;
          padding: 40px 25px; margin: 50px 0;
      }
      .verl-path-container {
          display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;
      }
      .verl-path-step {
          background: white; border-top: 4px solid var(--v-primary);
          padding: 20px; border-radius: 8px; flex: 1 1 220px; max-width: 280px;
          box-shadow: 0 4px 6px rgba(0,0,0,0.05); transition: 0.3s;
      }
      .verl-path-step:nth-child(2) { border-top-color: var(--v-secondary); }
      .verl-path-step:nth-child(3) { border-top-color: var(--v-accent); }
      .verl-path-step:nth-child(4) { border-top-color: var(--v-purple); }
      .verl-path-step:hover { transform: translateY(-5px); box-shadow: 0 8px 15px rgba(0,0,0,0.1); }
      .verl-step-num {
          display: inline-block; width: 30px; height: 30px; background: var(--v-primary);
          color: white; border-radius: 50%; text-align: center; line-height: 30px;
          font-weight: bold; margin-bottom: 15px;
      }
      .verl-path-step:nth-child(2) .verl-step-num { background: var(--v-secondary); }
      .verl-path-step:nth-child(3) .verl-step-num { background: var(--v-accent); }
      .verl-path-step:nth-child(4) .verl-step-num { background: var(--v-purple); }
      .verl-path-step h4 { margin: 0 0 10px 0 !important; border: none !important; font-size: 1.1rem !important; }
      .verl-path-step p { color: var(--v-text-sub); font-size: 0.9rem; line-height: 1.5; margin: 0; }

      /* 快速开始卡片跳转 */
      .verl-quick-grid {
          display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 20px; margin-top: 30px;
      }
      .verl-quick-card {
          text-decoration: none !important; border: 1px solid var(--v-border);
          padding: 25px; border-radius: 10px; text-align: center; transition: 0.3s; background: white;
      }
      .verl-quick-card:hover { border-color: var(--v-primary); transform: scale(1.03); box-shadow: 0 10px 20px rgba(0,102,204,0.1); }
      .verl-quick-card h4 { color: var(--v-primary); margin: 15px 0 0 0 !important; border: none !important; }

      @media (max-width: 768px) {
          .verl-hero h1 { font-size: 2rem !important; }
          .verl-quick-grid { grid-template-columns: 1fr 1fr; }
      }
   </style>

   <div id="verl-portal">
      <!-- 英雄区 -->
      <div class="verl-hero">
         <h1>🚀 verl</h1>
         <p class="verl-hero-subtitle">灵活、高效的大语言模型强化学习训练框架</p>
         <div class="verl-hero-buttons">
            <a href="https://github.com/volcengine/verl" class="verl-btn verl-btn-primary">📖 GitHub 仓库</a>
         </div>
      </div>

      <!-- 核心特性 -->
      <h2 class="verl-section-title">✨ 核心特性</h2>
      <div class="verl-features">
         <div class="verl-feature-card">
            <div class="verl-feature-icon">🧬</div>
            <h3>多样化 RL 算法</h3>
            <p>混合编程模型结合单控制器和多控制器范式，实现灵活表示和高效复杂后训练数据流执行</p>
         </div>
         <div class="verl-feature-card">
            <div class="verl-feature-icon">🔗</div>
            <h3>无缝框架集成</h3>
            <p>解耦计算和数据依赖，支持 PyTorch FSDP、Megatron-LM、vLLM 和 SGLang</p>
         </div>
         <div class="verl-feature-card">
            <div class="verl-feature-icon">📊</div>
            <h3>灵活设备映射</h3>
            <p>支持将模型放置在不同的 GPU/NPU 集合上，实现高效的资源利用和并行化</p>
         </div>
         <div class="verl-feature-card">
            <div class="verl-feature-icon">⚡</div>
            <h3>业界领先吞吐量</h3>
            <p>通过集成 SOTA LLM 训练和推理框架，实现高生成和训练吞吐量</p>
         </div>
         <div class="verl-feature-card">
            <div class="verl-feature-icon">🤗</div>
            <h3>HuggingFace 集成</h3>
            <p>与 HuggingFace 生态无缝适配，支持 600+ 开源模型即用</p>
         </div>
         <div class="verl-feature-card">
            <div class="verl-feature-icon">🎯</div>
            <h3>昇腾 NPU 支持</h3>
            <p>深度适配华为昇腾设备，充分发挥国产芯片性能</p>
         </div>
      </div>

      <!-- 开发者路径 -->
      <div class="verl-path-wrapper">
         <h2 style="text-align:center; color:var(--v-primary); margin-bottom:35px; border:none !important;">🎓 RL 后训练开发者学习路径</h2>
         <div class="verl-path-container">
            <div class="verl-path-step">
               <span class="verl-step-num">1</span>
               <h4>基础概念</h4>
               <p>理解 RLHF, PPO/DPO 算法原理及奖励模型设计基础</p>
            </div>
            <div class="verl-path-step">
               <span class="verl-step-num">2</span>
               <h4>快速上手</h4>
               <p>昇腾环境搭建、模型加载与基础微调单卡示例</p>
            </div>
            <div class="verl-path-step">
               <span class="verl-step-num">3</span>
               <h4>高级特性</h4>
               <p>多卡分布式训练调优、性能瓶颈分析与自定义算法</p>
            </div>
            <div class="verl-path-step">
               <span class="verl-step-num">4</span>
               <h4>生产部署</h4>
               <p>推理优化加速、模型评估基准与生产环境发布策略</p>
            </div>
         </div>
      </div>

      <!-- 快速开始 -->
      <h2 class="verl-section-title">🚀 快速开始</h2>
      <div class="verl-quick-grid">
         <a href="../_generated/sources/verl/quick_start/ascend_quick_start.html" class="verl-quick-card">
            <div style="font-size: 2rem;">📚</div>
            <h4>快速入门</h4>
         </a>
         <a href="../_generated/sources/verl/features/ascend_backend_features.html" class="verl-quick-card">
            <div style="font-size: 2rem;">🔧</div>
            <h4>特性指南</h4>
         </a>
         <a href="../_generated/sources/verl/examples/ascend_performance_analysis_guide.html" class="verl-quick-card">
            <div style="font-size: 2rem;">⚙️</div>
            <h4>应用实践</h4>
         </a>
         <a href="../_generated/sources/verl/contribution_guide/ascend_ci_guide_zh.html" class="verl-quick-card">
            <div style="font-size: 2rem;">👨‍💻</div>
            <h4>开源开发</h4>
         </a>
      </div>
   </div>

----

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🚀 快速入门

   ../_generated/sources/verl/quick_start/ascend_quick_start
   ../_generated/sources/verl/quick_start/ascend_sglang_quick_start
   ../_generated/sources/verl/quick_start/dockerfile_build_guidance

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 📚 特性指南

   ../_generated/sources/verl/features/ascend_backend_features
   ../_generated/sources/verl/features/ascend_consistency

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: ⚡ 调优指南

   ../_generated/sources/verl/profiling/ascend_profiling_zh
   ../_generated/sources/verl/profiling/ascend_profiling_en

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 📊 应用与调优践实

   ../_generated/sources/verl/examples/ascend_performance_analysis_guide
   ../_generated/sources/verl/examples/ascend_sglang_best_practices
   ../_generated/sources/verl/examples/dapo_multi_model_optimization_practice
   ../_generated/sources/verl/examples/ascend_retool_best_pratice
   ../_generated/sources/verl/examples/run_qwen3_32B_megatron_1k_256k_npu
   ../_generated/sources/verl/examples/gspo_optimization_practice

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🔆 故障排查(FAQ)

   ../_generated/sources/verl/faq/faq

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 🔧 开源开发

   ../_generated/sources/verl/contribution_guide/ascend_ci_guide_zh