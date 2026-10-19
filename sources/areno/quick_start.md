# areno

在单卡昇腾 NPU 上安装 AReno，并用 GSPO 算法在 GSM8K 上跑通「安装 → RLVR 训练 → checkpoint 落盘 → OpenAI 兼容推理服务」的最小链路。

本文档是上游 README「Tiny training smoke test」与 [Ascend NPU 后端文档](https://github.com/inclusionAI/AReno/blob/main/docs/concepts/npu.rst)的昇腾落地版：安装走 NPU 路径（检测到 `torch_npu` 后自动切换 NPU 依赖清单并编译 `areno.accel._areno_accel_npu` 加速扩展），训练/推理显式指定 `--backend npu`。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，单卡即可（本文示例 Qwen3-0.6B，bf16 权重约 1.2 GB）。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- Python ≥ 3.10
- CANN 9.0.0，且包含**开发组件**：AReno 安装时会现场编译 Ascend C 内核，需要 toolkit 中的 `tikcpp/ascendc_kernel_cmake`、`libascendcl` 等（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- `cmake`、`nm`（binutils）：NPU 内核构建与符号校验工具

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，32 GB × 1）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-910b-ubuntu22.04-py3.11

**软件版本**（torch ↔ torch_npu ↔ CANN 按 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)三方兼容矩阵选择，与上游 NPU 文档一致）：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11 |
| CANN | 9.0.0 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0.post2 |
| AReno | 最新 release（v0.9.0 起已含 NPU 后端） |
| 模型 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)（HuggingFace Hub） |
| 数据集 | `openai/gsm8k:main`（HuggingFace Hub） |

### 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

输出类似：

```text
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 5     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
```

```{admonition} Note
:class: note
如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。
```

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.11.xxx
```

安装 `torch` / `torch_npu`：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.10.0
uv pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi torch_npu==2.10.0.post2
```

检查 torch / torch_npu 是否装好且 NPU 设备可用（下面的命令用 Python 执行）：

```python #test id="check-torch"
import torch, torch_npu
print('torch=', torch.__version__)
print('torch_npu=', torch_npu.__version__)
print('is_available:', torch.npu.is_available())
print('count:', torch.npu.device_count())
```

输出结果如下：

```shell #test-result id="check-torch"
torch= 2.10.0+cpu
torch_npu= 2.10.0.post2
is_available: True
count: 1
```

```{admonition} Note
:class: note
如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。
```

补齐构建工具与构建后端（AReno 安装时会现场编译 Ascend C 内核与 torch_npu 扩展，`--no-build-isolation` 要求环境里已有 setuptools ≥ 69）：

```shell #test-setup
command -v cmake >/dev/null 2>&1 || { apt-get update -qq >/dev/null; apt-get install -y -qq cmake binutils >/dev/null; }
command -v cmake && command -v nm
uv pip install -U 'setuptools>=69' wheel
```

## 安装 AReno

从源码安装——PyPI 包不含 `examples/` 目录，而本文的训练示例要用到仓库里的 `examples/math/*.py`。

<!-- 工作流注入的 UPSTREAM_REF（最新 release tag）通过这个隐藏的 #test-setup 捕获并注入到下方 install 命令中-->
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到最新 release tag，安装并且验证：

```shell #test id="install-areno" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/inclusionAI/AReno.git
cd AReno
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
uv pip install -e . --no-build-isolation
python -c "from importlib.metadata import version; print('areno', version('areno'))"
```

```{admonition} Note
:class: note
`<ref>` 为安装的最新的 release tag。
```

输出结果类似如下：

```shell #test-result id="install-areno" fuzzy='xxx'
areno xxx
```

```{admonition} Note
:class: note
`areno xxx` 表示实际安装的版本号。
```

## 安装验证

### 验证 NPU 加速扩展已编译

`areno.accel._areno_accel_npu` 是安装时编译出的 NPU 加速扩展，缺失时训练与推理都无法运行（下面的命令用 Python 执行）：

```python #test id="areno-accel-check"
import importlib.util as u
specs = {m: u.find_spec(m) for m in ['areno', 'areno.accel', 'areno.accel._areno_accel_npu']}
for m, s in specs.items():
    print(m, 'ok' if s is not None else 'MISSING')
```

输出结果如下：

```shell #test-result id="areno-accel-check"
areno ok
areno.accel ok
areno.accel._areno_accel_npu ok
```

```{admonition} Note
:class: note
`areno.accel._areno_accel_npu` 显示 MISSING 说明安装时没有走到 NPU 分支——检查 `torch_npu` 是否先于 AReno 安装。
```

### 验证 CLI 入口可用

`areno train` / `areno serve` 两个子命令都能正常加载：

```shell #test id="areno-cli-check"
areno train --help >/dev/null && echo "cli_train_ok"
areno serve --help >/dev/null && echo "cli_serve_ok"
```

输出结果如下：

```shell #test-result id="areno-cli-check"
cli_train_ok
cli_serve_ok
```

## RLVR 训练（GSPO on GSM8K）

对齐上游 README「Tiny training smoke test」：下面把训练压到最小（1 步、生成 256 token），目标是验证 GSPO-on-GSM8K 链路（数据集加载、奖励校验、rollout、训练步）在 NPU 上跑通。想跑完整训练，去掉 `--max-steps` / `--max-new-tokens` 即可。

```shell #test-setup id="train-smoke-run"
cd AReno
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -o pipefail
ASCEND_RT_VISIBLE_DEVICES=0 areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path openai/gsm8k:main \
  --model-hub hf \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --algo gspo \
  --backend npu \
  --attn-backend native \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1 \
  --max-steps 1 \
  --max-new-tokens 256 \
  --save-path outputs/qwen3-gspo \
  --save-interval 1 \
  2>&1 | tee /tmp/areno_train_smoke.log
```

训练日志中的 loss、奖励等数值每次都不一样，无法写死预期值，只检查链路关键阶段的标记（rollout 完成 → 奖励计算 → 训练步完成 → checkpoint 落盘 → 步数上限退出）与 checkpoint 目录落盘：

```shell #test id="train-smoke"
ls -d AReno/outputs/qwen3-gspo/step_000001
grep -E 'stage=rollout_end|metric=reward_mean|train_stats=|stage=save_checkpoint_end|stage=max_steps_reached' /tmp/areno_train_smoke.log \
  | sed -E 's/^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} [A-Z]+ [^ ]+ [^ ]+ - //' \
  | sed -E 's/(metric=reward_mean value=).*/\1.../; s/(train_stats=)\{.*\}/\1{...}/; s|(stage=save_checkpoint_end path=).*|\1...|'
```

输出结果如下：

```shell #test-result id="train-smoke"
AReno/outputs/qwen3-gspo/step_000001
epoch=0 step=0 role=policy stage=rollout_end
epoch=0 step=0 metric=reward_mean value=...
epoch=0 step=0 train_stats={...}
epoch=0 step=0 stage=save_checkpoint_end path=...
epoch=0 step=1 stage=max_steps_reached
```

```{admonition} Note
:class: note
`...` 处为随训练变化的数值与路径；`train_stats` 的完整内容是本步训练指标字典。
```

捕获 checkpoint 路径供推理服务复用：

```shell #test-setup store="areno_ckpt"
realpath AReno/outputs/qwen3-gspo/step_000001
```

## 推理服务

### 启动服务

用训练产物启动 OpenAI 兼容服务：

```shell #test-setup id="serve-start" load="areno_ckpt>>ckpt"
cd AReno
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=0 areno serve \
  --model-path <ckpt> \
  --backend npu \
  --attn-backend native \
  --max-running-prompts 2 \
  --tp-size 1 \
  --world-size 1 \
  --port 8000 > /tmp/areno_serve.log 2>&1 &
echo $! > /tmp/areno_serve.pid
for i in $(seq 1 90); do
  sleep 5
  curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1 && break
done
curl -fsS http://127.0.0.1:8000/health
```

### 验证模型列表

`/v1/models` 列出服务端已注册的模型：

```shell #test id="serve-models"
curl -fsS http://127.0.0.1:8000/v1/models | python -c "
import sys, json
models = json.load(sys.stdin)['data']
assert models, 'no models in listing'
print('models_count:', len(models))
print('owned_by:', models[0]['owned_by'])
"
```

输出结果如下：

```shell #test-result id="serve-models"
models_count: 1
owned_by: areno
```

### 验证端到端生成

用服务端的模型 ID 发一条 chat 请求，验证「服务端 → 生成」全链路可用：

```shell #test id="serve-chat"
MODEL_ID=$(curl -fsS http://127.0.0.1:8000/v1/models | python -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])")
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"${MODEL_ID}\", \"messages\": [{\"role\": \"user\", \"content\": \"用一句话介绍你自己\"}], \"max_tokens\": 64}" \
  | python -c "
import sys, json
resp = json.load(sys.stdin)
choice = resp['choices'][0]
content = choice['message']['content']
assert content, 'empty completion'
print('finish_reason:', choice.get('finish_reason'))
print('content_len:', len(content))
"
```

输出结果类似如下：

```shell #test-result id="serve-chat" fuzzy='xxx'
finish_reason: xxx
content_len: xxx
```

- `xxx` 表示本次实际生成的结束原因与内容长度；生成内容本身具有随机性，只断言非空。

<!--
```shell #test-setup
kill "$(cat /tmp/areno_serve.pid)" 2>/dev/null || true
```
-->

## 外部链接

- GitHub：[inclusionAI/AReno](https://github.com/inclusionAI/AReno)
- 文档中心：[AReno Docs](https://asystem-ai.io/docs/areno/)
- NPU 后端说明：[docs/concepts/npu.rst](https://github.com/inclusionAI/AReno/blob/main/docs/concepts/npu.rst)
