# 在一张昇腾卡上跑一次 kt-kernel 的 llamafile 专家前向

这个脚本用随机小权重调用上游编出来的 `kt_kernel_ext.moe.MOE`。专家矩阵乘在 llamafile 的 CPU 后端里完成。昇腾构建会把这次前向通过 `submit_with_cuda_stream` 挂到当前 NPU 流的 ACL 回调上。回调线程没有跑起来时，输出会停在全 0，脚本会失败。

绿灯只说明这件事：当前检出的 kt-kernel 能以昇腾构建完成这一次小前向。它不加载 DeepSeek-V4-Flash，也不代表官方单卡服务教程在这台机器上跑得通。那条教程需要约 64 GB 显存和约 570 GB 权重。

在 Ascend/docs 仓库根目录、已经装好驱动、CANN、匹配的 torch / torch_npu，并且已经从本次要测的 ktransformers 检出编好带昇腾回调的 kt-kernel 之后：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python examples-guards/ktransformers/example/llamafile_npu_forward.py
```

成功时标准输出有一行 `kt-kernel ascend callback forward device=npu:`。
