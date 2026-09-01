# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上对 Qwen3-4B-Instruct-2507 进行自我认知微调。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

完整 NPU 适配（镜像选择、DDP/DeepSpeed/MindSpeed、`vllm-ascend` 部署等）请参考 [NPU 最佳实践文档](https://github.com/modelscope/ms-swift/blob/main/docs/source/BestPractices/NPU-support.md)。

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 1）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| transformers | `<5.0` |
| peft | `<0.19` |
| modelscope | 1.37.0 |
| ms-swift | 最新 release 的源码/二进制 |
| 模型 | [Qwen/Qwen3-4B-Instruct-2507](https://www.modelscope.cn/Qwen/Qwen3-4B-Instruct-2507) |
| 数据集 | `AI-ModelScope/alpaca-gpt4-data-zh#500` + `AI-ModelScope/alpaca-gpt4-data-en#500` + `swift/self-cognition#500` |
| 推理后端 | 本文 doc 默认 transformers / torch_npu（`--infer_backend transformers`）。vLLM-Ascend 加速推理不在本文档范围内，请参考 [NPU 最佳实践文档](https://github.com/modelscope/ms-swift/blob/main/docs/source/BestPractices/NPU-support.md#vllm-ascend)。 |

### 前置安装
确认能看到 NPU 设备：

```shell
npu-smi info
```

输出类似：

```
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 5     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 5                                                            |
+===========================+===============+====================================================+
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。

检查 Python 版本：

```shell #test id="check-py"
python --version
```
输出结果如下：
```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

安装 `torch` / `torch_npu`：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.9.0
uv pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi torch_npu==2.9.0.post2
```

检查 torch / torch_npu 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-torch"
torch= 2.9.0+cpu
torch_npu= 2.9.0.post2
is_available: True
count: 1
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

安装 `transformers` / `peft` / `modelscope`：

```shell #test-setup
uv pip install 'transformers<5.0' 'peft<0.19' 'modelscope==1.37.0'
```

打印安装版本：
```shell #test id="install-deps"
python -c "import transformers, peft, modelscope; print('transformers', transformers.__version__); print('peft', peft.__version__); print('modelscope', modelscope.__version__)"
```

输出结果如下：

```shell #test-result id="install-deps" fuzzy='xxx'
transformers xxx
peft xxx
modelscope 1.37.0
```

## 安装 ms-swift

### 使用 uv 进行安装

```shell #test id="swift-install-binary"
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple ms-swift
python -c "import swift; print('ms-swift', swift.__version__)"
```

输出结果类似如下：

```shell #test-result id="swift-install-binary" fuzzy='xxx'
ms-swift xxx
```
- xxx 表示最新的版本号

<!-- 
```shell #test-setup
uv pip uninstall ms-swift -y
```
-->

### 从源码安装
<!-- 
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到工作流注入的最新 release tag，安装并且验证

```shell #test id="swift-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/modelscope/ms-swift.git
cd ms-swift
uv pip install -e .
python -c "import swift; print('ms-swift', swift.__version__)"
```
\<ref> 为安装的最新的release 分支

输出结果类似如下：

```shell #test-result id="swift-install-source" fuzzy='xxx'
ms-swift xxx
```

- xxx 表示最新的版本号

## 使用样例

~2 分钟在单卡昇腾 NPU 上对 Qwen3-4B-Instruct-2507 做 5 步自我认知微调（够快来验证整条链路；想跑完整 1 epoch 预算 ~19 分钟，把下面 5 个 `max_steps/save_steps/logging_steps/eval_strategy/report_to` 参数去掉即可，doc 末尾的 `5/5` 预期输出也会对应变成 `94/94` 左右，需要相应调整 expected）：

```shell #test id="train"
ASCEND_RT_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --tuner_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --max_steps 5 \
    --save_strategy steps \
    --save_steps 5 \
    --logging_steps 1 \
    --eval_strategy no \
    --report_to none
```

输出结果如下：

```shell #test-result id="train" fuzzy='xxx' fuzzy='...'
run sh: xxx
...
{'loss': xxx, 'grad_norm': xxx, 'learning_rate': xxx, 'token_acc': xxx, ... 'global_step/max_steps': '1/5', ...}
...
{'loss': xxx, 'grad_norm': xxx, 'learning_rate': xxx, 'token_acc': xxx, ... 'global_step/max_steps': '5/5', ...}
{'train_runtime': xxx, ... 'train_loss': xxx, ... 'global_step/max_steps': '5/5', ...}
```

捕获 checkpoint 路径供推理阶段复用（将输出结果复制到下文的\<ckpt>中去复用）：

```shell #test-setup store="checkpoint"
ls -dt output/*/checkpoint-* | head -n 1
```

输出类似：
```
output/v0-20260101_120000-1234/checkpoint-5
```
小贴士：

- 如果要使用自定义数据集进行训练，你可以参考[这里](https://github.com/modelscope/ms-swift/blob/main/docs/source/Customization/Custom-dataset.md)组织数据集格式，并指定 `--dataset <dataset_path>`。
- `--model_author` 和 `--model_name` 参数只有当数据集中包含 `swift/self-cognition` 时才生效。
- 如果要使用其他模型进行训练，你只需要修改 `--model <model_id/model_path>` 即可。
- 默认使用 **ModelScope** 进行模型和数据集的下载。如果要使用 HuggingFace，指定 `--use_hf true` 即可。
- 训练启动前会自动应用 NPU 适配（`enable_npu_model_patch` 默认开启）。如需排查，参考 [NPU 最佳实践文档](https://github.com/modelscope/ms-swift/blob/main/docs/source/BestPractices/NPU-support.md) §"NPU 模型 Patch 开关"。

## 训练完成后推理

这里的 `<ckpt>` 占位符跟上面 `ls -dt output/*/checkpoint-* | head -n 1` 输出的是同一个值：训练阶段生成的 last checkpoint 文件夹（本例为 `output/v0-20260101_120000-1234/checkpoint-5`）。

### 交互式命令行推理（transformers / torch_npu 后端）

```shell #test id="infer" load="checkpoint>>ckpt"
ASCEND_RT_VISIBLE_DEVICES=0 \
swift infer \
    --adapters <ckpt> \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048 <<'PROMPT'
你好，请介绍一下自己。
exit
PROMPT
```
\<ckpt>在本例子中替换为：output/v0-20260101_120000-1234/checkpoint-5

输出结果如下：

```shell #test-result id="infer" fuzzy='xxx' fuzzy='...'
run sh: xxx
...
xxx你好xxx
...
```

## 推送 ModelScope

```shell
ASCEND_RT_VISIBLE_DEVICES=0 \
swift export \
    --adapters <ckpt> \
    --push_to_hub true \
    --hub_model_id '<your-model-id>' \
    --hub_token '<your-sdk-token>' \
    --use_hf false
```

这里的 `<ckpt>` 占位符跟上面 `ls -dt output/*/checkpoint-* | head -n 1` 输出的是同一个值：训练阶段生成的 last checkpoint 文件夹（本例为 `output/v0-20260101_120000-1234/checkpoint-5`）。