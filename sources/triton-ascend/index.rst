Triton-Ascend
============================================

.. raw:: html

   <style>
      #triton-ascend-portal {
          --ta-primary: #0066cc;
          --ta-secondary: #00a86b;
          --ta-accent: #ff6b35;
          --ta-text-main: #1a1a1a;
          --ta-text-sub: #666666;
          --ta-border: #e1e4e8;
          --ta-bg-light: #f6f8fa;
          font-family: -apple-system, system-ui, Segoe UI, Roboto, Helvetica;
      }

      .ta-hero {
          background: linear-gradient(135deg, var(--ta-primary) 0%, #004a99 100%);
          color: white;
          padding: 60px 40px;
          border-radius: 12px;
          margin: 20px 0 40px 0;
          text-align: center;
          box-shadow: 0 8px 24px rgba(0, 102, 204, 0.15);
      }
      .ta-hero h1 { color: white !important; border: none !important; margin: 0 0 15px 0 !important; font-size: 2.8rem !important; }
      .ta-hero-subtitle { font-size: 1.2rem; opacity: 0.95; margin-bottom: 30px; }
      .ta-hero-buttons { display: flex; gap: 15px; justify-content: center; flex-wrap: wrap; }
      .ta-btn {
          padding: 12px 30px; border-radius: 6px; text-decoration: none !important;
          font-weight: 600; transition: all 0.3s ease; border: 2px solid white;
      }
      .ta-btn-primary { background: white; color: var(--ta-primary) !important; }
      .ta-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 16px rgba(0,0,0,0.2); }

      .ta-section-title { text-align: center; color: var(--ta-primary); margin: 50px 0 30px 0; font-size: 2rem; border: none !important; }
      .ta-quick-grid {
          display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 20px; margin-top: 30px;
      }
      .ta-quick-card {
          text-decoration: none !important; border: 1px solid var(--ta-border);
          padding: 25px; border-radius: 10px; text-align: center; transition: 0.3s; background: white;
      }
      .ta-quick-card:hover { border-color: var(--ta-primary); transform: scale(1.03); box-shadow: 0 10px 20px rgba(0, 102, 204, 0.1); }
      .ta-quick-card h4 { color: var(--ta-primary); margin: 12px 0 6px 0 !important; border: none !important; font-size: 1.05rem !important; }
      .ta-quick-card p { color: var(--ta-text-sub); font-size: 0.85rem; margin: 0; }

      .ta-features-wrapper {
          background: var(--ta-bg-light); border-radius: 12px;
          padding: 35px 25px; margin: 50px 0 20px 0; text-align: center;
      }
      .ta-feature-tags {
          display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin-top: 20px;
      }
      .ta-feature-tag {
          display: inline-flex; align-items: center; gap: 6px; white-space: nowrap;
          background: white; border: 1px solid var(--ta-border); border-radius: 20px;
          padding: 8px 18px; font-size: 0.9rem; color: var(--ta-text-main);
          transition: all 0.3s ease;
      }
      .ta-feature-tag:hover { border-color: var(--ta-primary); box-shadow: 0 4px 8px rgba(0, 102, 204, 0.1); }

      @media (max-width: 768px) {
          .ta-hero h1 { font-size: 2rem !important; }
          .ta-quick-grid { grid-template-columns: 1fr 1fr; }
      }
   </style>

   <div id="triton-ascend-portal">

      <div class="ta-hero">
         <h1>Triton-Ascend</h1>
         <p class="ta-hero-subtitle">
            适配昇腾 NPU 的 Triton 语言后端 · 高性能算子开发与迁移
         </p>
         <div class="ta-hero-buttons">
            <a href="https://gitcode.com/Ascend/triton-ascend" class="ta-btn ta-btn-primary" target="_blank">📖 GitCode 仓库</a>
         </div>
      </div>

      <h2 class="ta-section-title">🚀 快速开始</h2>
      <div class="ta-quick-grid">
         <a href="../_generated/sources/triton-ascend/installation_guide.html" class="ta-quick-card">
            <div style="font-size: 2rem;">📦</div>
            <h4>安装指南</h4>
            <p>环境准备与安装步骤</p>
         </a>
         <a href="../_generated/sources/triton-ascend/quick_start.html" class="ta-quick-card">
            <div style="font-size: 2rem;">🚀</div>
            <h4>快速上手</h4>
            <p>编写第一个 Triton 算子</p>
         </a>
         <a href="../_generated/sources/triton-ascend/programming_guide.html" class="ta-quick-card">
            <div style="font-size: 2rem;">📖</div>
            <h4>算子开发指南</h4>
            <p>Triton 编程模型与开发实践</p>
         </a>
         <a href="../_generated/sources/triton-ascend/examples/index.html" class="ta-quick-card">
            <div style="font-size: 2rem;">🧪</div>
            <h4>典型算子样例</h4>
            <p>Softmax、Attention、MatMul 等</p>
         </a>
      </div>

      <div class="ta-features-wrapper">
         <h2 style="text-align:center; color:var(--ta-primary); margin: 0 0 5px 0; border:none !important;">✨ 核心特性</h2>
         <div class="ta-feature-tags">
            <span class="ta-feature-tag">🔌 昇腾 NPU 后端适配</span>
            <span class="ta-feature-tag">🧩 丰富算子样例</span>
            <span class="ta-feature-tag">🔧 调试与性能调优工具</span>
            <span class="ta-feature-tag">🤝 完整 API 参考</span>
         </div>
      </div>

   </div>

----

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 快速开始

   ../_generated/sources/triton-ascend/quick_start

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 安装指南

   ../_generated/sources/triton-ascend/installation_guide

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 架构设计与核心特性

   ../_generated/sources/triton-ascend/architecture_design_and_core_features

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Triton 算子开发指南

   ../_generated/sources/triton-ascend/programming_guide

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Triton 算子迁移指南

   ../_generated/sources/triton-ascend/migration_guide/architecture_difference
   ../_generated/sources/triton-ascend/migration_guide/migrate_from_gpu
   ../_generated/sources/triton-ascend/migration_guide/performance_guidelines

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 典型算子样例

   ../_generated/sources/triton-ascend/examples/index
   ../_generated/sources/triton-ascend/examples/01_vector_add_example
   ../_generated/sources/triton-ascend/examples/02_fused_softmax_example
   ../_generated/sources/triton-ascend/examples/03_layer_norm_example
   ../_generated/sources/triton-ascend/examples/04_fused_attention_example
   ../_generated/sources/triton-ascend/examples/05_matrix_multiplication_example
   ../_generated/sources/triton-ascend/examples/06_autotune_example
   ../_generated/sources/triton-ascend/examples/07_accuracy_comparison_example

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 算子调试与调优

   ../_generated/sources/triton-ascend/debug_guide/debugging
   ../_generated/sources/triton-ascend/debug_guide/profiling

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Triton API 接口说明

   ../_generated/sources/triton-ascend/triton_api/index
   ../_generated/sources/triton-ascend/triton_api/triton/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: 昇腾扩展 API 接口说明

   ../_generated/sources/triton-ascend/triton_api_extention/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 环境变量

   ../_generated/sources/triton-ascend/environment_variable_reference

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 常见问题

   ../_generated/sources/triton-ascend/FAQ
   ../_generated/sources/triton-ascend/release_note
