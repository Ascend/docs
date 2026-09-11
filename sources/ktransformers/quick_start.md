# KTransformers

[KTransformers](https://github.com/kvcache-ai/ktransformers) 是面向大模型异构推理与微调的框架。昇腾上的安装、权重准备和启动命令以上游官方教程为准。

阅读官方教程前，先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备 CANN 与驱动。

项目主页：[KTransformers 文档站](https://kvcache-ai.github.io/ktransformers/)。

## 前置条件

硬件决定走哪一篇教程。上游已验证的昇腾路径目前是 Atlas 300I A2，以及单卡约 64 GB HBM 的 Atlas 800I A2 / Atlas A3。

## 1. 按硬件选择教程


| 硬件                                         | 场景                                                 | 教程                                                                                                                                                        |
| ------------------------------------------ | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Atlas 300I A2                              | DeepSeek-R1 / V3，满血约 400 GB 主机内存                   | [DeepseekR1_V3_tutorial_zh_for_Ascend_NPU.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/zh/DeepseekR1_V3_tutorial_zh_for_Ascend_NPU.md)   |
| Atlas 300I A2                              | Qwen3-MoE，满血约 200 GB 主机内存                          | [Qwen3-MoE_tutorial_zh_for_Ascend_NPU.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/zh/Qwen3-MoE_tutorial_zh_for_Ascend_NPU.md)           |
| Atlas 800I A2，910B3，约 64 GB HBM；或 Atlas A3 | DeepSeek-V4-Flash 单卡，CPU expert offload；磁盘约 570 GB | [DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md) |


