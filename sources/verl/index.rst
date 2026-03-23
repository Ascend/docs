verl
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

      /* 快速开始卡片 */
      .verl-section-title { text-align: center; color: var(--v-primary); margin: 40px 0 25px 0; font-size: 1.8rem; border: none !important; }
      .verl-quick-grid {
          display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 20px; margin-bottom: 50px;
      }
      .verl-quick-card {
          text-decoration: none !important; border: 1px solid var(--v-border);
          padding: 28px 20px; border-radius: 10px; text-align: center; transition: 0.3s; background: white;
      }
      .verl-quick-card:hover { border-color: var(--v-primary); transform: scale(1.03); box-shadow: 0 10px 20px rgba(0,102,204,0.1); }
      .verl-quick-card-icon { font-size: 2.2rem; margin-bottom: 12px; }
      .verl-quick-card h4 { color: var(--v-primary); margin: 0 0 8px 0 !important; border: none !important; font-size: 1.1rem !important; }
      .verl-quick-card p { color: var(--v-text-sub); font-size: 0.85rem; margin: 0; line-height: 1.4; }

      /* 核心特性 - 紧凑标签行 */
      .verl-features-compact {
          background: var(--v-bg-light); border-radius: 12px;
          padding: 30px 25px; margin-bottom: 20px;
      }
      .verl-features-compact-title {
          text-align: center; color: var(--v-primary); margin: 0 0 20px 0;
          font-size: 1.1rem; font-weight: 600; border: none !important;
      }
      .verl-feature-tags {
          display: flex; flex-wrap: wrap; gap: 12px; justify-content: center;
      }
      .verl-feature-tag {
          display: flex; align-items: center; gap: 8px;
          background: white; border: 1px solid var(--v-border);
          border-radius: 20px; padding: 8px 16px;
          font-size: 0.9rem; color: var(--v-text-main);
          white-space: nowrap;
      }
      .verl-feature-tag span { font-size: 1.1rem; }

      @media (max-width: 768px) {
          .verl-hero h1 { font-size: 2rem !important; }
          .verl-quick-grid { grid-template-columns: 1fr 1fr; }
          .verl-feature-tags { gap: 8px; }
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

      <!-- 快速开始 -->
      <h2 class="verl-section-title">🚀 快速开始</h2>
      <div class="verl-quick-grid">
         <a href="../_generated/sources/verl/quick_start/ascend_quick_start.html" class="verl-quick-card">
            <div class="verl-quick-card-icon">📚</div>
            <h4>快速入门</h4>
            <p>昇腾环境搭建与基础训练示例</p>
         </a>
         <a href="../_generated/sources/verl/features/ascend_backend_features.html" class="verl-quick-card">
            <div class="verl-quick-card-icon">🔧</div>
            <h4>特性指南</h4>
            <p>昇腾后端特性与一致性说明</p>
         </a>
         <a href="../_generated/sources/verl/examples/ascend_performance_analysis_guide.html" class="verl-quick-card">
            <div class="verl-quick-card-icon">⚙️</div>
            <h4>应用实践</h4>
            <p>性能分析与最佳实践案例</p>
         </a>
         <a href="../_generated/sources/verl/contribution_guide/ascend_ci_guide_zh.html" class="verl-quick-card">
            <div class="verl-quick-card-icon">👨‍💻</div>
            <h4>开源开发</h4>
            <p>CI 流程与贡献指南</p>
         </a>
      </div>

      <!-- 核心特性（紧凑标签） -->
      <div class="verl-features-compact">
         <p class="verl-features-compact-title">✨ 核心特性</p>
         <div class="verl-feature-tags">
            <div class="verl-feature-tag"><span>🧬</span> 多样化 RL 算法</div>
            <div class="verl-feature-tag"><span>🔗</span> 无缝框架集成</div>
            <div class="verl-feature-tag"><span>📊</span> 灵活设备映射</div>
            <div class="verl-feature-tag"><span>⚡</span> 业界领先吞吐量</div>
            <div class="verl-feature-tag"><span>🤗</span> HuggingFace 集成</div>
            <div class="verl-feature-tag"><span>🎯</span> 昇腾 NPU 支持</div>
         </div>
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
