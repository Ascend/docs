"""NPU 兼容补丁：bool 张量的 ``cumprod(dim, dtype=...)`` 在昇腾上不可用。

``Tensor.cumprod`` 的 ``dtype`` 参数的文档语义是「先把输入转成 ``dtype`` 再算」
（``torch.cumprod`` 原文：*"If specified, the input tensor is casted to dtype
before the operation is performed."*）。ATen 并不自己做这次转换——它把 bool
输入和 ``dtype`` 关键字一起派发给后端，由后端负责转换。用 TorchDispatchMode
抓一次派发就能看到：

    aten op   : aten.cumprod.default
       input  : torch.bool
       dtype kw: torch.float32

CPU / CUDA / MPS 后端都照做，但 torch_npu 的 NPU 后端把原始 bool 张量直接丢给
``aclnnCumprod``，而它没有 DT_BOOL 的实现：

    RuntimeError: cumprod_out:../third_party/op-plugin/op_plugin/ops/opapi/
      StructKernelNpuOpApi.cpp:1068 NPU function error: call aclnnCumprod
      failed, error code is 161002
    AclNN_Parameter_Error(EZ1001): Tensor input not implemented for DT_BOOL,
      should be in dtype support list [DT_FLOAT, DT_FLOAT16, DT_BFLOAT16, ...]

Speculators 的 DFlash 训练指标正好命中这个形状
（``src/speculators/models/metrics.py`` 的 ``compute_greedy_block_metrics``）::

    correct_and_valid = correct_slots & valid_slots          # bool
    accepted_length = (
        correct_and_valid.cumprod(dim=-1, dtype=torch.float32).sum(dim=-1) + 1.0
    )

于是 ``torchrun`` 拉起的训练 worker 在第一个训练步就挂掉，整条训练链路
（hidden states -> 训练 -> 部署）走不下去。

本模块补上后端漏掉的那次转换：只在「输入是 bool 且显式给了 dtype」时把输入
转成目标 dtype 再交给原来的实现，其他调用原样透传，语义与 CPU / CUDA 完全
一致。上游 speculators 的源码不打任何补丁——它本来就符合 PyTorch 的文档语义，
缺的是 NPU 后端的那次转换。

用法：由测试侧把它装成 ``sitecustomize.py`` 放到 ``PYTHONPATH`` 上
（解释器启动时 ``site`` 会自动 import 它，``torchrun`` 拉起的 worker 同样会
加载）。本模块自身也可以直接 import（import 即完成打补丁）。
"""

from __future__ import annotations

import torch

__all__ = ['patch_cumprod_bool_dtype']

_orig_cumprod = torch.Tensor.cumprod


def _cumprod(self, dim, dtype=None, *args, **kwargs):
    """``Tensor.cumprod`` + ATen 文档里 ``dtype`` 的转换语义。

    只在「输入是 bool、目标 dtype 不是 bool」时补一次转换——这正是 torch_npu
    的 NPU 后端漏掉的那一步。其余情形（bool 输入不指定 dtype、非 bool 输入）
    完全透传给原实现，避免改变已经在 NPU 上跑通的行为。
    """
    if (
        dtype is not None
        and self.dtype == torch.bool
        and dtype != torch.bool
    ):
        self = self.to(dtype)
        dtype = None
    return _orig_cumprod(self, dim, *args, dtype=dtype, **kwargs)


def patch_cumprod_bool_dtype() -> bool:
    """给 ``torch.Tensor.cumprod`` 打上 bool + dtype 的兼容补丁。

    返回 True 表示补丁已生效（重复调用不会重复打）；已经打过时返回 False。
    """
    if getattr(torch.Tensor.cumprod, '_npu_bool_dtype_patched', False):
        return False
    _cumprod._npu_bool_dtype_patched = True
    torch.Tensor.cumprod = _cumprod
    return True


patch_cumprod_bool_dtype()
