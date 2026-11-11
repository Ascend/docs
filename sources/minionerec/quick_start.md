# MiniOneRec

[MiniOneRec](https://github.com/AkaliKong/MiniOneRec) 是生成式推荐框架。本文在单张昇腾 NPU 上依次完成 SFT 训练、约束解码评估与推荐导向 RL 三个阶段。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列，芯片为 Ascend **910B**，需支持 bf16。本文示例为单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动固件已安装，并可 `source set_env.sh`，9.1.0 或以上 |
| Python | 3.10 或以上 |
| git | 用于克隆上游仓库 |
| 模型 | [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)（base 版） |

### 本文验证环境

| 项目 | 内容 |
| --- | --- |
| 机器 | Atlas 900 A2，Ascend 910B，单卡 |
| 镜像 | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12` |
| CANN | 9.1.0 |
| Python | 3.12 |
| torch / torch_npu | 2.9.0 / 2.9.0.post2 |

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。

## 1. 加载 CANN 环境

每次新开终端后执行：

```shell
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 2. 检查环境是否就绪

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
npu-smi info
python --version
```

确认 `npu-smi info` 输出设备列表且 `Health` 为 `OK`。若提示找不到 `npu-smi`，按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。

## 3. 获取代码

克隆 MiniOneRec 仓库：

```shell #test-setup id="clone"
git clone --depth 1 --branch main https://github.com/AkaliKong/MiniOneRec.git
cd MiniOneRec
```

```{note}
本文使用 MiniOneRec `main` 分支的最新提交。
```

## 4. 安装依赖

安装以下依赖（与 CANN 9.1.0 配套）：

```shell #test-setup id="install"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install torch==2.9.0 torch-npu==2.9.0.post2
pip install transformers==4.57.1 accelerate==1.10.1 datasets==4.2.0 \
    tokenizers==0.22.1 fire==0.7.1 pandas==2.2.2 tqdm bitsandbytes==0.48.1 \
    trl==0.24.0 scikit-learn==1.5.2 wandb==0.22.2
```

## 5. 确认运行环境

执行以下命令，确认 NPU 与训练设备就绪：

```shell #test id="check-npu"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd MiniOneRec
python -c "
import npu_support
import torch
from transformers import TrainingArguments
print('IS_NPU:', npu_support.IS_NPU)
print('cuda mapped:', torch.cuda.is_available())
print('evaluate.py device:', 'cuda' if torch.cuda.is_available() else 'cpu')
print('trainer device:', TrainingArguments(output_dir='/tmp/check-npu').device)
print('tensor landed on:', torch.zeros(1, device='cuda').device.type)
"
```

输出结果如下：

```shell #test-result id="check-npu"
IS_NPU: True
cuda mapped: True
evaluate.py device: cuda
trainer device: npu:0
tensor landed on: npu
```

检查克隆目录状态：

```shell #test id="no-source-change"
cd MiniOneRec
test -z "$(git status --porcelain)" && echo "source untouched"
```

输出结果如下：

```shell #test-result id="no-source-change"
source untouched
```

---

## 6. 下载基座模型与预处理数据

基座模型取 Qwen2.5 的 **base** 版（Instruct 版会使约束解码失效）。模型与预处理数据从 HuggingFace 下载：

```shell #test-setup id="download"
cd MiniOneRec
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-0.5B', local_dir='./Qwen2.5-0.5B')"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='kkknight/MiniOneRec', allow_patterns=['Amazon/index/Industrial_and_Scientific.index.json','Amazon/index/Industrial_and_Scientific.item.json','Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv','Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv','Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv','Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt'], local_dir='data')"
```

校验资产落位：

```shell #test id="verify-assets"
cd MiniOneRec
test -f Qwen2.5-0.5B/config.json && echo "base model ok"
test -f data/Amazon/index/Industrial_and_Scientific.index.json && echo "sid index ok"
test -f data/Amazon/index/Industrial_and_Scientific.item.json && echo "item meta ok"
test -f data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv && echo "train csv ok"
test -f data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv && echo "valid csv ok"
test -f data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt && echo "info txt ok"
```

输出结果如下：

```shell #test-result id="verify-assets"
base model ok
sid index ok
item meta ok
train csv ok
valid csv ok
info txt ok
```

## 7. SFT 训练

在 0 号卡上执行 1 个 epoch：

```shell #test-setup id="sft-run"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
cd MiniOneRec
python sft.py \
    --base_model ./Qwen2.5-0.5B \
    --train_file ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --eval_file ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --output_dir ./sft_out \
    --batch_size 8 \
    --micro_batch_size 2 \
    --num_epochs 1 \
    --cutoff_len 256 \
    --sample 600 \
    --category Industrial_and_Scientific \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --freeze_LLM False \
    --train_from_scratch False
```

训练完成后校验产物：

```shell #test id="verify-sft"
cd MiniOneRec
test -f sft_out/final_checkpoint/model.safetensors && echo "model saved"
test -f sft_out/final_checkpoint/config.json && echo "config saved"
test -f sft_out/final_checkpoint/tokenizer_config.json && echo "tokenizer saved"
grep -o '"model_type": "[a-z0-9_.-]*"' sft_out/final_checkpoint/config.json
```

输出结果如下：

```shell #test-result id="verify-sft"
model saved
config saved
tokenizer saved
"model_type": "qwen2"
```

## 8. 约束解码评估

用上一节的 `final_checkpoint` 在 test 子集上做约束解码评估并计算 HR/NDCG/CC。**CC 必须为 0。**

切出 test 子集：

```shell #test-setup id="prep-eval"
cd MiniOneRec
head -n 200 data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv > test_small.csv
wc -l test_small.csv
```

运行评估：

```shell #test id="run-eval"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
cd MiniOneRec
python evaluate.py \
    --base_model ./sft_out/final_checkpoint \
    --info_file ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
    --test_data_path test_small.csv \
    --result_json_data ./eval_result.json \
    --category Industrial_and_Scientific \
    --num_beams 20 --batch_size 2
```

输出结果如下：

```shell #test-result id="run-eval"
industrial and scientific items
...
```

校验预测产物：

```shell #test id="verify-eval"
cd MiniOneRec
python -c "import json; d=json.load(open('eval_result.json')); print(len(d)); print('predict' in d[0])"
```

输出结果如下：

```shell #test-result id="verify-eval"
199
True
```

计算 HR/NDCG/CC：

```shell #test id="calc-metrics"
cd MiniOneRec
python calc.py \
    --path ./eval_result.json \
    --item_path ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt
```

输出结果如下：

```shell #test-result id="calc-metrics"
20
[1, 3, 5, 10, 20]
NDCG:...
HR...
0
```

## 9. 推荐导向 RL

在 0 号卡上执行 1 个 epoch。`--sample 40` 用于限制数据规模，请勿省略；`train_batch_size` 必须能被 `num_generations` 整除。

```shell #test id="run-rl"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
cd MiniOneRec
python rl.py \
    --model_path ./sft_out/final_checkpoint \
    --train_file ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --eval_file ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --info_file ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
    --category Industrial_and_Scientific \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --num_generations 2 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --beta 1e-3 \
    --beam_search True \
    --reward_type ranking \
    --sync_ref_model True \
    --test_during_training False \
    --eval_step 1.0 \
    --sample 40 \
    --output_dir ./rl_out \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
```

输出结果如下：

```shell #test-result id="run-rl"
Industrial_and_Scientific
...
```

日志中的 `reward`、`reward_std`、`loss`、`grad_norm`、`kl` 可能为 0，属正常现象。

校验产物：

```shell #test id="verify-rl"
cd MiniOneRec
test -f rl_out/final_checkpoint/model.safetensors && echo "rl model saved"
test -f rl_out/final_checkpoint/config.json && echo "rl config saved"
test -f rl_out/final_checkpoint/tokenizer_config.json && echo "rl tokenizer saved"
```

输出结果如下：

```shell #test-result id="verify-rl"
rl model saved
rl config saved
rl tokenizer saved
```

---

## 相关链接

- 上游仓库：[AkaliKong/MiniOneRec](https://github.com/AkaliKong/MiniOneRec)
- 基座模型：[Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- 预处理数据与 SID 索引：[kkknight/MiniOneRec](https://huggingface.co/kkknight/MiniOneRec)
- 昇腾 PyTorch 安装与兼容性：[Ascend/pytorch](https://gitcode.com/Ascend/pytorch)
