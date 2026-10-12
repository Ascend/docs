# llama.cpp example 看护

本目录是 ggml-org/llama.cpp 的 example 看护数据，给共用引擎 `.github/workflows/examples-template.yml` 用。薄触发器是 `.github/workflows/llama.cpp-examples.yml`。上游默认分支是 `master`。触发器把 `default_branch` 设成 `master`，发布查询失败时回落到这条分支。

被测 ref 优先取上游最新 release。清单路径按 release `v0.6.0` 的 `examples/` 核对过，列出的目录都还在。

## 清单里跑什么

`supported` 保持源仓那一份。这次不把源仓标成 `unsupported` 的条目放回来，也不新加还没在昇腾上跑过的目录。

不跑的三条：

- `examples/idle`：Metal 空闲微基准
- `examples/sycl`：要 Intel oneAPI
- `examples/training`：全量 FP32 微调，CANN 没有对应反向算子

`v0.6.0` 里另外有 `examples/deprecation-warning`、`examples/llama-eval`、`examples/test-cmake`。源仓清单没有它们，这次不调度。

## profile

`scripts/setup_example.sh` 用函数 `setup_<profile>` 解释清单里的 profile。未知 profile 在编译和下载之前退出。

- `cann`：打开 `GGML_CANN`，只编 `exec` 指向的那一个 target。Qwen2.5 0.5B Instruct 的 Q4_0 GGUF 先从 huggingface.co 下，失败再从 ModelScope 下。量化选 Q4_0，因为上游 CANN 文档里 910B 的数据类型表写了 FP16、Q8_0、Q4_0、BF16。
- `cpu`：cmake 不开 `GGML_CANN`。绿灯只表示这个目录在 aarch64 上能编完、能跑完。外部汇总不要把这些条目读成昇腾推理成功。
- `cann-diffusion`：同样打开 CANN，编 `llama-diffusion-cli`。权重是 Dream-v0-Instruct-7B 的官方 safetensors，先 Hugging Face，失败再 ModelScope，再用这次 checkout 的 `convert_hf_to_gguf.py` 转成 Q8_0 GGUF。共用引擎没有前置 job，下载和转换都在这条矩阵腿里做。转换用的 torch 是上游 `v0.6.0` 在 Linux 上钉死的 CPU 轮子 `2.11.0+cpu`，不装 `torch_npu`。llama.cpp 走镜像里的 CANN 和 `GGML_CANN`，不走 PyTorch 昇腾插件。
- `cmake-pkg`：安装之后再用 `find_package` 编 `examples/simple-cmake-pkg`，确认安装包还能链上 CANN 后端。`LD_LIBRARY_PATH` 指到 `inst/lib`，因为消费方二进制的 RUNPATH 盖不住 `libggml` 再加载的 `libggml-cann`。

镜像是 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。编译器用镜像自带的 cmake、g++、make，不改 apt 源。

## 缓存

下载落到 runner 的 `~/.cache`，香港池会把这棵目录留到下一次运行。workflow 不写容器 volume，也不依赖源仓的预投递。

多条 `cann` 腿共用同一个 Qwen GGUF。脚本先下到本 job 的暂存文件，核对 GGUF 文件头之后再改名进缓存，并用 mkdir 锁避开并发追加写。

Dream 的 safetensors 同样先暂存、核对分片头，再转 GGUF。转好的文件留在缓存里，下次命中就不再下载。

## 设备锚点

`cann`、`cann-diffusion`、`cmake-pkg` 的运行日志里必须出现 `CANN` 后接数字的那一行。ggml 在卡没挂上时会静默退回 CPU，进程仍然可以 0 退出。

`examples/simple-chat` 是例外。上游源码把低于 ERROR 的日志丢掉，加载阶段的 CANN 行不会出现。这次不改那份源码，运行脚本也不再要求这行。这条只看进程是不是 0 退出。0 退出不能当成这次用了昇腾卡的证据。

`examples/retrieval` 的上游循环在空输入上不退出，CI 喂一行就会挂到超时。`scripts/patch_retrieval.py` 只改本次工作副本，让空行或输入结束时跳出循环。不提交回上游。

## 触发

`schedule` 保持注释。源仓的定时看护仍然是生产看护。打开本仓 cron 时要错开已有槽位，并同时关掉源仓同项目的 schedule。
