# Speculators

在单卡昇腾 NPU 上跑通 [Speculators](https://github.com/vllm-project/speculators) 投机解码草稿模型的端到端链路：

- **安装**：源码安装 speculators（镜像预装 vllm-ascend 全栈，无需编译）。
- **convert**：把 DFlash draft + verifier 按 DFlash 算法重映射权重。
- **训练**：verifier 抽 hidden states → `torchrun` 单卡训 1 epoch。
- **部署**：`vllm serve` 挂载训好的 draft 做 chat completion smoke。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，至少 1 卡。

### 基础软件

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

### 本文档示例使用的版本

**配套机器**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 1），Ubuntu 22.04

**配套镜像**：

swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/vllm-ascend/vllm-ascend:v0.23.0

镜像预装 vllm 0.23.0 + vllm-ascend 0.23.0 + triton-ascend 3.2.2 + torch 2.10.0+cpu + torch_npu 2.10.0.post4 + torchvision 0.25.0+cpu + torchaudio 2.10.0+cpu + transformers 5.5.4 + modelscope 1.39.1 + CANN 9.1.0 + Python 3.12。

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.10.0+cpu（镜像预装） |
| torch_npu | 2.10.0.post4（镜像预装） |
| torchvision | 0.25.0+cpu（镜像预装） |
| torchaudio | 2.10.0+cpu（镜像预装） |
| vllm | 0.23.0（镜像预装） |
| vllm-ascend | 0.23.0（镜像预装） |
| triton-ascend | 3.2.2（镜像预装） |
| triton | 3.5.0（镜像预装） |
| transformers | 5.5.4（镜像预装；speculators 透传范围 >=4.56.1,<5.15.0） |
| modelscope | 1.39.1（镜像预装；下面步骤会钉到 1.37.0） |
| speculators | 最新 release |
| draft 模型 | z-lab/Qwen3-8B-DFlash-b16 |
| verifier | Qwen/Qwen3-8B |

## 前置安装

确认 NPU 设备可见：

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
| 0     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
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
Python 3.12.xxx
```

```{admonition} Note
:class: note
xxx 表示 Python 的补丁版本号。
```

检查 CANN env 并验证镜像预装的 vllm-ascend 栈（含 torch / torch_npu / torchvision / torchaudio / transformers / vllm / vllm-ascend / triton*）：

```shell #test id="verify-vllm-stack"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -c "
import torch, torch_npu, torchvision, torchaudio, transformers
print(f'torch={torch.__version__}')
print(f'torch_npu={torch_npu.__version__}')
print(f'torchvision={torchvision.__version__}')
print(f'torchaudio={torchaudio.__version__}')
print(f'transformers={transformers.__version__}')
print('is_available:', torch.npu.is_available())
print('npu_count:', torch.npu.device_count())
"

python -c "import importlib.metadata; print(f'vllm={importlib.metadata.version(\"vllm\")}')"
python -c "import importlib.metadata; print(f'vllm_ascend={importlib.metadata.version(\"vllm-ascend\")}')"
python -c "import importlib.metadata; print(f'triton_ascend={importlib.metadata.version(\"triton-ascend\")}')"
python -c "import importlib.metadata; print(f'triton={importlib.metadata.version(\"triton\")}')"
```

输出结果如下：

```shell #test-result id="verify-vllm-stack"
torch=2.10.0+cpu
torch_npu=2.10.0.post4
torchvision=0.25.0+cpu
torchaudio=2.10.0+cpu
transformers=5.5.4
is_available: True
npu_count: 1
vllm=0.23.0+empty
vllm_ascend=0.23.0
triton_ascend=3.2.2
triton=3.5.0
```

装 modelscope（用于从 ModelScope 下载模型）：

```shell #test id="install-deps"
uv pip install 'modelscope==1.37.0'
python -c "import modelscope; print(f'modelscope={modelscope.__version__}')"
```

输出结果如下：

```shell #test-result id="install-deps"
modelscope=1.37.0
```

## 安装 Speculators

### 从源码安装

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆 speculators 上游 release tag 源码到 `/root/speculators/` 并 editable 安装：

```shell #test id="speculators-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/vllm-project/speculators.git /root/speculators
cd /root/speculators
uv pip install -e .
speculators --version
python -c "from importlib.metadata import version; print('speculators', version('speculators'))"
```

输出结果如下：

```shell #test-result id="speculators-install-source" fuzzy='xxx'
speculators version: xxx
speculators xxx
```

```{admonition} Note
:class: note
`<ref>` 替换为 Speculators 当前的最新 release 版本；xxx 表示实际安装的 speculators 版本号。
```

## 端到端：convert → 训练数据生成 → 训练 → 部署

### 前置：下载 draft 与 verifier

从 ModelScope 拉 draft 模型 z-lab/Qwen3-8B-DFlash-b16 与 verifier 模型 Qwen/Qwen3-8B(下面的命令用 Python 执行)：

```python #test-setup
from modelscope import snapshot_download

snapshot_download('z-lab/Qwen3-8B-DFlash-b16')
snapshot_download('Qwen/Qwen3-8B')
```

确认两个模型快照已就位（`config.json` + 权重文件）：

```shell #test id="model-download-check"
ls -1 /root/.cache/modelscope/hub/models/z-lab/Qwen3-8B-DFlash-b16/config.json /root/.cache/modelscope/hub/models/z-lab/Qwen3-8B-DFlash-b16/model.safetensors
ls -1 /root/.cache/modelscope/hub/models/Qwen/Qwen3-8B/config.json /root/.cache/modelscope/hub/models/Qwen/Qwen3-8B/model.safetensors.index.json
```

输出结果如下：

```shell #test-result id="model-download-check"
/root/.cache/modelscope/hub/models/z-lab/Qwen3-8B-DFlash-b16/config.json
/root/.cache/modelscope/hub/models/z-lab/Qwen3-8B-DFlash-b16/model.safetensors
/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B/config.json
/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B/model.safetensors.index.json
```

### convert（DFlash 算法）

`speculators convert` 把本地 draft + verifier 读进来按 DFlash 算法重映射权重、写到 `/root/dflash-qwen3-8b-converted/`。CLI 的 `--algorithm` 不支持 `dflash`，走 Python API(下面的命令用 Python 执行)：

```python #test-setup store="dflash_path"
from speculators.convert import convert_model

OUTPUT = '/root/dflash-qwen3-8b-converted'
convert_model(
    model='/root/.cache/modelscope/hub/models/z-lab/Qwen3-8B-DFlash-b16',
    verifier='/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B',
    algorithm="dflash",
    output_path=OUTPUT,
)
print(OUTPUT)
```

确认 convert 产物存在（`config.json` + `model.safetensors`）：

```shell #test id="pipeline-step1-convert" load="dflash_path>>dflash_path"
ls -1 <dflash_path>/config.json <dflash_path>/model.safetensors
echo <dflash_path>
```

输出结果如下：

```shell #test-result id="pipeline-step1-convert"
/root/dflash-qwen3-8b-converted/config.json
/root/dflash-qwen3-8b-converted/model.safetensors
/root/dflash-qwen3-8b-converted
```

```{admonition} Note
:class: note
`<dflash_path>` 在运行时替换为「convert（DFlash 算法）」一节 `print(OUTPUT)` 命令输出的产物目录路径。
```

### 训练数据预处理

把 10 条 chat 用 verifier tokenizer 跑 chat template 得到 `input_ids`/`loss_mask`，写到 `/tmp/prompts.jsonl`（speculator-format），再交给上游 `prepare-data`。

```shell #test-setup store="data_path"
set -euo pipefail
DATA_DIR=/root/dflash-train-data
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

python << 'PY'
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B")

def _ids(encoded):
    # transformers 4.x: list[int] ; 5.x: BatchEncoding with .input_ids
    return encoded["input_ids"] if hasattr(encoded, "keys") else encoded

# 用 verifier 的 chat template 算 assistant 段分界：
# prefix_ids（add_generation_prompt=True）渲染到 <|im_start|>assistant\n 为止；
# full_ids 在它后面追加 <think>\n\n</think>\n\n + assistant 内容 + <|im_end|>\n
# （Qwen3 thinking 模式 serving 输出形态），所以 [len(prefix_ids):] 就是要算 loss 的 token。
with open("/tmp/prompts.jsonl", "w") as f:
    for i in range(10):
        conv = [
            {"role": "user", "content": f"Briefly describe AI topic #{i}."},
            {"role": "assistant", "content": "AI is a field of computer science."},
        ]
        prefix_ids = _ids(tokenizer.apply_chat_template(
            conv[:-1], tokenize=True, add_generation_prompt=True
        ))
        full_ids = _ids(tokenizer.apply_chat_template(conv, tokenize=True))
        loss_mask = [0] * len(prefix_ids) + [1] * (len(full_ids) - len(prefix_ids))
        f.write(json.dumps({"input_ids": full_ids, "loss_mask": loss_mask}) + "\n")
PY

speculators prepare-data \
  --model "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B" \
  --data /tmp/prompts.jsonl \
  --output "$DATA_DIR" \
  --max-samples 10 \
  --seq-length 8192 \
  --num-preprocessing-workers 4 \
  --overwrite

echo "$DATA_DIR"
```

确认数据集已落盘（行数 + `token_freq.pt` 存在）(下面的命令用 Python 执行)：

```python #test id="pipeline-step2-extract" load="data_path>>data_path"
from datasets import load_from_disk
import torch
from pathlib import Path

data_path = '<data_path>'
print(data_path)
print(len(load_from_disk(data_path)))
p = Path(data_path) / 'token_freq.pt'
print('exists:', p.exists(), 'size:', p.stat().st_size if p.exists() else 0)
freq = torch.load(p, weights_only=True)
print('token_freq keys:', list(freq.keys()) if isinstance(freq, dict) else type(freq).__name__)
print('len:', len(freq))
```

输出结果如下：

```shell #test-result id="pipeline-step2-extract" fuzzy='xxx'
/root/dflash-train-data
10
exists: True size: xxx
token_freq keys: xxx
len: xxx
```

```{admonition} Note
:class: note
xxx 分别为 `token_freq.pt` 的文件大小（字节）、token 表的键列表与条目数；`<data_path>` 在运行时替换为「训练数据预处理」一节 `echo "$DATA_DIR"` 命令输出的数据目录路径。
```

### 训练

先生成 hidden_states 缓存，再离线训。起 vllm 一次性 generate 10 条 hidden_states 写到 `/tmp/hs-train/`。v0.8.0 的 `launch_vllm.py` 有两处行为变化需要留意：按宿主机 CPU 数自动推导 `--api-server-count`，以及不再自动追加 `--no-enable-chunked-prefill`——镜像 vllm 0.23.0 默认开 chunked prefill 而 `ExampleHiddenStatesConnector` 不支持，必须显式关掉。这里 10 条 smoke 数据量很小，固定用 1 个 API server 前端（`--api-server-count 1`）：

```shell #test-setup store="hs_dir" load="data_path>>data_path"
set -euo pipefail

cd /root/speculators

HS_DIR=/tmp/hs-train
rm -rf "$HS_DIR"
mkdir -p "$HS_DIR"

export TORCHDYNAMO_DISABLE=1
rm -rf /root/.cache/vllm/torch_compile_cache 2>/dev/null || true

set +u
export ZSH_VERSION="${ZSH_VERSION:-}"
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

setsid nohup python scripts/launch_vllm.py "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B" \
  --target-layer-ids 2 18 34 \
  --hidden-states-path "$HS_DIR" \
  -- \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --enforce-eager \
  --api-server-count 1 \
  --no-enable-chunked-prefill \
  > /tmp/vllm-gen.log 2>&1 < /dev/null &
VLLM_GEN_PID=$!
VLLM_GEN_PGID=$(ps -o pgid= -p "$VLLM_GEN_PID" | tr -d ' ')
cleanup_vllm_gen() {
  kill -- -"$VLLM_GEN_PGID" 2>/dev/null || true
  for _pid in $(pgrep -P "$VLLM_GEN_PID" 2>/dev/null) "$VLLM_GEN_PID"; do
    kill -9 "$_pid" 2>/dev/null || true
  done
  pkill -9 -x "vllm" 2>/dev/null || true
}
trap cleanup_vllm_gen EXIT

# 等 /health 200（最长 6 min）
VLLM_READY=0
for i in {1..180}; do
  if curl -sf http://127.0.0.1:8000/health > /dev/null; then
    VLLM_READY=1
    break
  fi
  sleep 2
done
if [ "$VLLM_READY" != "1" ]; then
  echo "vllm server failed to come up within 6 min; tail of vllm-gen.log:" >&2
  tail -80 /tmp/vllm-gen.log >&2
  cleanup_vllm_gen
  exit 1
fi

speculators generate-offline-data \
  --model "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B" \
  --preprocessed-data "<data_path>" \
  --output "$HS_DIR" \
  --max-samples 10 \
  --concurrency 4 \
  --validate-outputs >/tmp/hs-gen.log 2>&1 || HS_RC=$?
HS_RC=${HS_RC:-0}
# 日志走 stderr，stdout 只留末行路径（供后续 <hs_dir> 引用）
tail -30 /tmp/hs-gen.log >&2

HS_COUNT=$(ls -1 "$HS_DIR"/hs_*.safetensors 2>/dev/null | wc -l)
if [ "$HS_RC" -ne 0 ] || [ "$HS_COUNT" -ne 10 ]; then
  echo "=== generate-offline-data failed (rc=$HS_RC, hs_count=$HS_COUNT/10); full log ===" >&2
  cat /tmp/hs-gen.log >&2
  cleanup_vllm_gen
  exit 1
fi

cleanup_vllm_gen
sleep 5

echo "$HS_DIR"
```

确认 hidden states 已落盘（10 个 `hs_*.safetensors`）(下面的命令用 Python 执行)：

```python #test id="pipeline-step2b-hs" load="hs_dir>>hs_dir"
from pathlib import Path
print(len(list(Path('<hs_dir>').glob('hs_*.safetensors'))))
```

输出结果如下：

```shell #test-result id="pipeline-step2b-hs"
10
```

```{admonition} Note
:class: note
`<hs_dir>` 在运行时替换为「训练」一节 `echo "$HS_DIR"` 命令输出的 hidden states 目录路径。
```

用 `torchrun -m speculators.train` 单卡训 1 epoch × 10 sample（smoke 验证管线通，不指望 loss 真下降）：

```shell #test-setup store="checkpoint_path" load="hs_dir>>hs_dir" load="data_path>>data_path"
set -euo pipefail
CHECKPOINT_DIR=/root/dflash-trained
rm -rf "$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"

cd /root/speculators

export TORCHDYNAMO_DISABLE=1
export ASCEND_LAUNCH_BLOCKING=1

set +u
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

torchrun --standalone --nproc_per_node=1 -m speculators.train \
  --verifier-name-or-path "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B" \
  --data-path "<data_path>" \
  --hidden-states-path "<hs_dir>" \
  --hidden-states-dtype float32 \
  --save-path "$CHECKPOINT_DIR" \
  --draft-vocab-size 32000 \
  --epochs 1 \
  --lr 3e-4 \
  --speculator-type dflash \
  --block-size 8 \
  --max-anchors 32 \
  --draft-attn-impl sdpa \
  --num-layers 5 \
  --target-layer-ids 2 18 34 \
  --on-missing raise >/tmp/train.log 2>&1 || TRAIN_RC=$?
TRAIN_RC=${TRAIN_RC:-0}

if [ "$TRAIN_RC" -ne 0 ]; then
  echo "=== train failed (rc=$TRAIN_RC); full train.log follows ===" >&2
  cat /tmp/train.log >&2
  exit 1
fi

LATEST_CKPT=$(ls -1d "$CHECKPOINT_DIR"/[0-9]*/ 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_CKPT" ] && [ "$LATEST_CKPT" != "$CHECKPOINT_DIR/" ]; then
  cp -af "$LATEST_CKPT"/. "$CHECKPOINT_DIR"/
fi
if ! test -f "$CHECKPOINT_DIR/config.json" || ! test -f "$CHECKPOINT_DIR/model.safetensors"; then
  echo "=== train rc=0 但 checkpoint 缺失 (looked under $CHECKPOINT_DIR/) ===" >&2
  cat /tmp/train.log >&2
  exit 1
fi
# 清 trainer 元数据：symlinks / 子目录 / optimizer & scheduler state /
# run.yaml / training_state.json / val_metrics.json / config.py
rm -f "$CHECKPOINT_DIR"/checkpoint_best "$CHECKPOINT_DIR"/epoch0_end
rm -rf "$CHECKPOINT_DIR"/[0-9]*/
rm -f "$CHECKPOINT_DIR"/optimizer_state_dict.pt \
      "$CHECKPOINT_DIR"/scheduler_state_dict.pt \
      "$CHECKPOINT_DIR"/run.yaml \
      "$CHECKPOINT_DIR"/training_state.json \
      "$CHECKPOINT_DIR"/val_metrics.json \
      "$CHECKPOINT_DIR"/train_command.txt \
      "$CHECKPOINT_DIR"/config.py

echo "$CHECKPOINT_DIR"
```

确认训练产物与训练真跑完（checkpoint 文件 + 从 train.log 提取 val loss）：

```shell #test id="pipeline-step3-train" load="checkpoint_path>>checkpoint_path"
ls -1 <checkpoint_path>
grep -oE 'val/loss_epoch=[0-9.]+' /tmp/train.log | head -1
```

输出结果如下：

```shell #test-result id="pipeline-step3-train" fuzzy='xxx'
config.json
model.safetensors
val/loss_epoch=xxx
```

```{admonition} Note
:class: note
xxx 为训练日志里的验证 loss 数值；`<checkpoint_path>` 在运行时替换为「训练」一节 `echo "$CHECKPOINT_DIR"` 命令输出的 checkpoint 目录路径。
```

### `vllm serve` 挂 draft 做推理

起 vllm-ascend serve 把训好的 draft 挂上做 chat completion smoke（8 token completion）：

```shell #test id="pipeline-step4-serve" load="checkpoint_path>>draft_model"

export TORCHDYNAMO_DISABLE=1
rm -rf /root/.cache/vllm/torch_compile_cache 2>/dev/null || true

set +u
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

# num_speculative_tokens=5：vllm-ascend 限制 (num_speculative_tokens + 1) ≤ 15
nohup vllm serve "/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B" \
  --host 127.0.0.1 --port 8000 \
  --served-model-name Qwen/Qwen3-8B \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --speculative-config '{"method":"dflash","model":"<draft_model>","num_speculative_tokens":5}' \
  > /tmp/vllm-serve.log 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null" EXIT

# 等 /health 200（最长 6 min）
for i in {1..180}; do
  curl -sf http://127.0.0.1:8000/health > /dev/null && break
  sleep 2
done

echo "input: Hello"
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Hello"}],"max_tokens":8}' \
  | python -c "
import sys, json
r = json.load(sys.stdin)
print('content:', r['choices'][0]['message']['content'])
print('completion_tokens:', r['usage']['completion_tokens'])
print('finish_reason:', r['choices'][0]['finish_reason'])
"

kill "$VLLM_PID" 2>/dev/null || true
```

输出结果如下：

```shell #test-result id="pipeline-step4-serve" fuzzy='xxx' fuzzy='...'
input: Hello
content: xxx
...
completion_tokens: xxx
finish_reason: length
```

```{admonition} Note
:class: note
xxx 分别为模型生成的内容与生成的 token 数（`max_tokens=8` 截断，`finish_reason` 为 length）；`<draft_model>` 在运行时替换为「训练」一节 `echo "$CHECKPOINT_DIR"` 命令输出的 checkpoint 目录路径。
```

### 编程式入口：SpeculatorsConfig / TokenProposalConfig

验证 `SpeculatorsConfig` / `TokenProposalConfig` 可以在 NPU 环境 import + 实例化（不依赖 GPU 计算）(下面的命令用 Python 执行)：

```python #test id="config-import"
from speculators import VerifierConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

verifier = VerifierConfig(
    name_or_path="/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B",
    architectures=["Qwen3ForCausalLM"],
)
proposal = GreedyTokenProposalConfig(
    proposal_type="greedy",
    speculative_tokens=5,
    verifier_accept_k=1,
    accept_tolerance=0.0,
)
print("verifier:", verifier.name_or_path)
print("verifier architectures:", verifier.architectures)
print("proposal type:", proposal.proposal_type)
print("proposal speculative_tokens:", proposal.speculative_tokens)
```

输出结果如下：

```shell #test-result id="config-import"
verifier: /root/.cache/modelscope/hub/models/Qwen/Qwen3-8B
verifier architectures: ['Qwen3ForCausalLM']
proposal type: greedy
proposal speculative_tokens: 5
```

## 外部链接

- GitHub：[vllm-project/speculators](https://github.com/vllm-project/speculators)
- 文档中心：[Speculators Docs](https://vllm-project.github.io/speculators/)
