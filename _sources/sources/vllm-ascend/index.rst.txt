vLLM-Ascend
============================================

.. raw:: html

   <style>
      /* 样式隔离：仅作用于 vllm-ascend 首页 */
      #vllm-ascend-portal {
          --va-primary: #0066cc;
          --va-secondary: #00a86b;
          --va-accent: #ff6b35;
          --va-purple: #9d4edd;
          --va-text-main: #1a1a1a;
          --va-text-sub: #666666;
          --va-border: #e1e4e8;
          --va-bg-light: #f6f8fa;
          font-family: -apple-system, system-ui, Segoe UI, Roboto, Helvetica;
      }

      /* 英雄区 */
      .va-hero {
          background: linear-gradient(135deg, var(--va-primary) 0%, #004a99 100%);
          color: white;
          padding: 60px 40px;
          border-radius: 12px;
          margin: 20px 0 40px 0;
          text-align: center;
          box-shadow: 0 8px 24px rgba(0, 102, 204, 0.15);
      }
      .va-hero h1 { color: white !important; border: none !important; margin: 0 0 15px 0 !important; font-size: 2.8rem !important; }
      .va-hero-subtitle { font-size: 1.2rem; opacity: 0.95; margin-bottom: 30px; }
      .va-hero-buttons { display: flex; gap: 15px; justify-content: center; flex-wrap: wrap; }
      .va-btn {
          padding: 12px 30px; border-radius: 6px; text-decoration: none !important;
          font-weight: 600; transition: all 0.3s ease; border: 2px solid white;
      }
      .va-btn-primary { background: white; color: var(--va-primary) !important; }
      .va-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 16px rgba(0,0,0,0.2); }

      /* 快速开始卡片 */
      .va-section-title { text-align: center; color: var(--va-primary); margin: 50px 0 30px 0; font-size: 2rem; border: none !important; }
      .va-quick-grid {
          display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 20px; margin-top: 30px;
      }
      .va-quick-card {
          text-decoration: none !important; border: 1px solid var(--va-border);
          padding: 25px; border-radius: 10px; text-align: center; transition: 0.3s; background: white;
      }
      .va-quick-card:hover { border-color: var(--va-primary); transform: scale(1.03); box-shadow: 0 10px 20px rgba(0,102,204,0.1); }
      .va-quick-card h4 { color: var(--va-primary); margin: 12px 0 6px 0 !important; border: none !important; font-size: 1.05rem !important; }
      .va-quick-card p { color: var(--va-text-sub); font-size: 0.85rem; margin: 0; }

      /* 核心特性标签 */
      .va-features-wrapper {
          background: var(--va-bg-light); border-radius: 12px;
          padding: 35px 25px; margin: 50px 0 20px 0; text-align: center;
      }
      .va-feature-tags {
          display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin-top: 20px;
      }
      .va-feature-tag {
          display: inline-flex; align-items: center; gap: 6px; white-space: nowrap;
          background: white; border: 1px solid var(--va-border); border-radius: 20px;
          padding: 8px 18px; font-size: 0.9rem; color: var(--va-text-main);
          transition: all 0.3s ease;
      }
      .va-feature-tag:hover { border-color: var(--va-primary); box-shadow: 0 4px 8px rgba(0,102,204,0.1); }

      @media (max-width: 768px) {
          .va-hero h1 { font-size: 2rem !important; }
          .va-quick-grid { grid-template-columns: 1fr 1fr; }
      }
   </style>

   <div id="vllm-ascend-portal">

      <!-- 英雄区：仅 GitHub 按钮 -->
      <div class="va-hero">
         <h1>vllm-ascend</h1>
         <p class="va-hero-subtitle">
            面向昇腾 NPU 的 vLLM 社区插件 · 高性能 LLM 推理加速
         </p>
         <div class="va-hero-buttons">
            <a href="https://github.com/vllm-project/vllm-ascend" class="va-btn va-btn-primary" target="_blank">📖 GitHub 仓库</a>
         </div>
      </div>

      <!-- 快速开始：第一视觉，4 张卡片 -->
      <h2 class="va-section-title">🚀 快速开始</h2>
      <div class="va-quick-grid">
         <a href="../_generated/sources/vllm-ascend/installation.html" class="va-quick-card">
            <div style="font-size: 2rem;">📦</div>
            <h4>安装指南</h4>
            <p>环境准备与安装步骤</p>
         </a>
         <a href="../_generated/sources/vllm-ascend/quick_start.html" class="va-quick-card">
            <div style="font-size: 2rem;">🚀</div>
            <h4>快速上手</h4>
            <p>5 分钟跑通推理任务</p>
         </a>
         <a href="../_generated/sources/vllm-ascend/user_guide/feature_guide/index.html" class="va-quick-card">
            <div style="font-size: 2rem;">📖</div>
            <h4>用户指南</h4>
            <p>特性配置与部署方案</p>
         </a>
         <a href="../_generated/sources/vllm-ascend/developer_guide/contribution/index.html" class="va-quick-card">
            <div style="font-size: 2rem;">👨‍💻</div>
            <h4>开发者指南</h4>
            <p>贡献代码与特性开发</p>
         </a>
      </div>

      <!-- 核心特性：标签式一行展示 -->
      <div class="va-features-wrapper">
         <h2 style="text-align:center; color:var(--va-primary); margin: 0 0 5px 0; border:none !important;">✨ 核心特性</h2>
         <div class="va-feature-tags">
            <span class="va-feature-tag">🔌 硬件插件化架构</span>
            <span class="va-feature-tag">⚡ 高性能推理加速</span>
            <span class="va-feature-tag">🧩 丰富模型支持</span>
            <span class="va-feature-tag">🌐 分布式推理</span>
            <span class="va-feature-tag">🔧 完整工具链</span>
            <span class="va-feature-tag">🤝 社区共建</span>
         </div>
      </div>

   </div>

----

.. 以下 toctree 直接引用 upstream submodule 的子目录 index 文件（如 tutorials/models/index），
.. 无需额外的 nav RST 包装文件。路径从 sources/vllm-ascend/ 出发，
.. 通过 ../_generated/sources/vllm-ascend/ 指向 make copy-docs 生成的内容。
.. Makefile 仅删除 upstream 根目录 index（避免与本文件冲突），子目录 index 完整保留。

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   ../_generated/sources/vllm-ascend/quick_start
   ../_generated/sources/vllm-ascend/installation
   ../_generated/sources/vllm-ascend/tutorials/models/index
   ../_generated/sources/vllm-ascend/tutorials/features/index
   ../_generated/sources/vllm-ascend/tutorials/hardwares/index
   ../_generated/sources/vllm-ascend/faqs

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Guide

   ../_generated/sources/vllm-ascend/user_guide/support_matrix/index
   ../_generated/sources/vllm-ascend/user_guide/configuration/index
   ../_generated/sources/vllm-ascend/user_guide/feature_guide/index
   ../_generated/sources/vllm-ascend/user_guide/deployment_guide/index
   ../_generated/sources/vllm-ascend/user_guide/release_notes

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Developer Guide

   ../_generated/sources/vllm-ascend/developer_guide/contribution/index
   ../_generated/sources/vllm-ascend/developer_guide/feature_guide/index
   ../_generated/sources/vllm-ascend/developer_guide/evaluation/index
   ../_generated/sources/vllm-ascend/developer_guide/performance_and_debug/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Community

   ../_generated/sources/vllm-ascend/community/governance
   ../_generated/sources/vllm-ascend/community/contributors
   ../_generated/sources/vllm-ascend/community/versioning_policy
   ../_generated/sources/vllm-ascend/community/user_stories/index
