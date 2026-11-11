"""Confirm the CANN 9.1 torch pair and an Ascend build of kt-kernel."""

from __future__ import annotations

import torch
import torch_npu
from kt_kernel import kt_kernel_ext

print(f"torch={torch.__version__} torch_npu={torch_npu.__version__}")
if not torch.__version__.startswith("2.14.0") or not torch_npu.__version__.startswith("2.14.0"):
    raise SystemExit("torch and torch_npu must be the 2.14.0 pair used on the CANN 9.1 image")
if not torch.npu.is_available() or torch.npu.device_count() < 1:
    raise SystemExit("NPU is not available after installing torch_npu")
if not hasattr(kt_kernel_ext, "init_ascend_callback_worker"):
    raise SystemExit("kt-kernel was built without the Ascend callback worker")
if not hasattr(kt_kernel_ext, "subscribe_ascend_stream"):
    raise SystemExit("kt-kernel was built without subscribe_ascend_stream")
if not hasattr(kt_kernel_ext.CPUInfer, "submit_with_cuda_stream"):
    raise SystemExit("kt-kernel CPUInfer has no NPU stream submit")
if not hasattr(kt_kernel_ext.moe, "MOE"):
    raise SystemExit("kt-kernel llamafile MOE binding is missing")
print("npu stack ready", torch.npu.get_device_name(0))
