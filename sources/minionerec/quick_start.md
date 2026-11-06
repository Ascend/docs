# MiniOneRec

[MiniOneRec](https://github.com/AkaliKong/MiniOneRec) 是全开源的生成式推荐框架：把每件商品编码成 3 层语义 ID（SID），再用 LLM 按「下一个 SID」的方式生成推荐结果。本文在单张昇腾 NPU 上完成 SID 监督微调（SFT）、约束解码评估与推荐导向 RL 三个阶段。

三个阶段都不需要改动源码：上游入口脚本第一行会导入 `npu_support`，在 `torch_npu` 可用时启用 `transfer_to_npu`，把 `torch.cuda` 这套接口映射到 NPU；`sft.py` 的设备则由 transformers 自己选到 `npu:0`。RL 阶段只需用 `--sample` 把数据规模压下来。

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

新开终端后 CANN 变量不会自动生效。

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

`npu-smi` 打印出设备列表、`Health` 为 `OK` 即可；若提示找不到 `npu-smi`，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。

## 3. 获取代码

将 `<ref>` 替换为上游最新的 `main` 提交。

```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```

```shell #test-setup id="clone" load="upstream_ref>>ref"
git clone --depth 1 https://github.com/AkaliKong/MiniOneRec.git
cd MiniOneRec
git checkout <ref>
```

## 4. 安装依赖

上游 `requirements.txt` 面向 CUDA 环境（`torchrec`、`fbgemm_gpu` 是 `+cu118` 安装包，另带整套 `nvidia-*-cu12` 运行库），NPU 上改用与 CANN 9.1.0 配套的 torch 栈单独安装。

```shell #test-setup id="install"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install torch==2.9.0 torch-npu==2.9.0.post2
pip install transformers==4.57.1 accelerate==1.10.1 datasets==4.2.0 \
    tokenizers==0.22.1 fire==0.7.1 pandas==2.2.2 tqdm bitsandbytes==0.48.1 \
    trl==0.24.0 scikit-learn==1.5.2 wandb==0.22.2
```

`bitsandbytes` 钉在 aarch64 有预编译包的 0.48.1：`sft.py` 与 `evaluate.py` 顶部的 `import bitsandbytes as bnb` 从未被实际使用，但不改源码就绕不开这行 import，必须让它能导入。`trl` 提供 `rl.py` 用的 `GRPOConfig`，`scikit-learn` 提供 `ndcg_score`，`wandb` 则是 `sasrec.py` 无条件导入的。`sft.py` 的 `TrainingArguments` 默认 `report_to="all"`，装好 wandb 后会尝试 `wandb.init()` 并因缺少 API key 失败，所以第 7 节导出 `WANDB_MODE=disabled`。

## 5. 确认 NPU 映射已生效

上游入口脚本的第一行是 `import npu_support`：`torch_npu` 可用时，它会导入 `transfer_to_npu`，把 `torch.cuda.is_available()`、`"cuda"` 设备字符串等整套接口映射到 NPU；在没有 `torch_npu` 的 CUDA / CPU 机器上则是空操作。下面这段命令确认映射生效，也确认 `sft.py` 的设备选择：

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

输出结果如下（代码里写的是 `cuda`，张量实际落在 NPU）：

```shell #test-result id="check-npu"
IS_NPU: True
cuda mapped: True
evaluate.py device: cuda
trainer device: npu:0
tensor landed on: npu
```

确认仓库源码没被动过：

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

基座模型取 Qwen2.5 的 **base** 版，Instruct 版会让约束解码失效。模型与预处理数据都在 HuggingFace 上：`Qwen/Qwen2.5-0.5B` 是权重，`kkknight/MiniOneRec` 是官方预处理好的 Amazon Industrial_and_Scientific 数据。数据侧只取 SID 索引、训练 / 验证 / 测试 CSV 与 `info/` 下的 SID-条目清单；`.npy` 嵌入文件只有 RQ/SID 构建阶段需要，跳过。

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

单卡跑 1 epoch：`sample=600` 限制每个子数据集（SID 序列 / SID-条目对齐 / 融合序列推荐）各取 600 条，`cutoff_len=256` 截断序列。完整训练（8 卡 A100、全量数据）见上游 `sft.sh`。

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

训练完成后校验产物（`model.safetensors` + `config.json` + tokenizer 配置）：

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

用上一节得到的 `final_checkpoint` 在 test 切片上做 beam search 评估，再算 HR/NDCG/CC。**CC（无效 SID 生成数）必须为 0**——`ConstrainedLogitsProcessor` 用 trie 约束每步只允许合法 SID 前缀 token，这是约束解码在 NPU 上生效的关键信号。本节用 `num_beams=20`、`batch_size=2`。

切出 test 子集：

```shell #test-setup id="prep-eval"
cd MiniOneRec
head -n 200 data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv > test_small.csv
wc -l test_small.csv
```

运行评估（进度条与 transformers 警告走 stderr，stdout 只有 category 名）：

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

输出结果如下（进度条走 stderr）：

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

计算 HR/NDCG/CC。HR/NDCG 随训练状态浮动，判断约束解码是否生效看 CC 是否为 0：

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

RL 在 SFT 产物上再做一轮 GRPO：每个 prompt 采样多条候选、组内归一化奖励、并用 KL 约束贴近参考模型；动作空间是有限的 SID 列表，候选由约束 beam search 生成。

上游 `rl.py` 已按 `npu_support.IS_NPU` 选择 `adamw_torch` 优化器（bitsandbytes 的 paged 优化器没有 NPU 实现），无需改动源码。数据规模用 `--sample 40` 控制：`sample > 0` 时三个训练数据集与评估数据集都按这个数取样；不传则会展开整个商品目录，单卡跑不完。

在 0 号卡上跑 1 epoch。`train_batch_size` 必须能被 `num_generations` 整除；`num_generations` 同时是 beam 宽度，本节取 2。

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

输出结果如下（数据集对象与进度条走 stderr）：

```shell #test-result id="run-rl"
Industrial_and_Scientific
...
```

日志里的 `reward`、`reward_std`、`loss`、`grad_norm`、`kl` 会一直是 0，这不是故障。`rule_reward` 要求生成的 SID 与目标完全相等；SFT 阶段只训 1 epoch，精确命中率接近 0，组内两条候选都命不中，`ndcg_rule_reward` 还会把整组补 0，GRPO 的组内优势因此无从计算，梯度自然为 0。到这里三个阶段都已跑通；要看到非零的训练信号，需要更充分的 SFT。

校验 `rl.py` 结尾的两步落盘（`trainer.save_model` 与 `model.save_pretrained`）：

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
