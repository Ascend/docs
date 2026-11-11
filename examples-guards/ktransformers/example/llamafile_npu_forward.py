"""Run one tiny llamafile MoE forward through kt-kernel's Ascend stream callback.

The expert math stays in kt-kernel's llamafile CPU backend. On an Ascend
build, submit_with_cuda_stream places that forward on the current NPU
stream with aclrtLaunchCallback. Without the callback worker the callback
never runs and the output buffer stays zero. This script does not load
DeepSeek-V4-Flash and does not claim that serving path is healthy.
"""

from __future__ import annotations

import torch
import torch_npu
from kt_kernel import kt_kernel_ext
from kt_kernel_ext.kvcache import ggml_type

EXPERT_NUM = 2
TOP_K = 2
HIDDEN = 256
INTERMEDIATE = 256
QLEN = 1
ANCHOR = "kt-kernel ascend callback forward"


def from_float(tensor: torch.Tensor, ggml: ggml_type) -> torch.Tensor:
    count = int(tensor.numel())
    return kt_kernel_ext.utils.from_float(tensor.data_ptr(), count, ggml)


def to_float(tensor: torch.Tensor, count: int, ggml: ggml_type) -> torch.Tensor:
    return kt_kernel_ext.utils.to_float(tensor.data_ptr(), count, ggml)


def require_ascend_build() -> None:
    if not torch.npu.is_available() or torch.npu.device_count() < 1:
        raise SystemExit("NPU is not available")
    missing = [
        name
        for name in (
            "init_ascend_callback_worker",
            "subscribe_ascend_stream",
            "is_ascend_callback_worker_running",
        )
        if not hasattr(kt_kernel_ext, name)
    ]
    if missing:
        raise SystemExit("kt-kernel extension is missing " + ", ".join(missing))
    if not hasattr(kt_kernel_ext.CPUInfer, "submit_with_cuda_stream"):
        raise SystemExit("this kt-kernel build cannot submit work onto an NPU stream")
    if not hasattr(kt_kernel_ext.moe, "MOE"):
        raise SystemExit("llamafile MOE is not bound in this kt-kernel build")


def build_moe(cpu: kt_kernel_ext.CPUInfer) -> tuple[object, list[torch.Tensor]]:
    torch.manual_seed(0)
    gate = torch.randn((EXPERT_NUM, INTERMEDIATE, HIDDEN), dtype=torch.float32)
    up = torch.randn((EXPERT_NUM, INTERMEDIATE, HIDDEN), dtype=torch.float32)
    down = torch.randn((EXPERT_NUM, HIDDEN, INTERMEDIATE), dtype=torch.float32)
    gate_q = from_float(gate, ggml_type.Q4_K)
    up_q = from_float(up, ggml_type.Q4_K)
    down_q = from_float(down, ggml_type.Q6_K)
    config = kt_kernel_ext.moe.MOEConfig(EXPERT_NUM, TOP_K, HIDDEN, INTERMEDIATE)
    config.pool = cpu.backend_
    config.layer_idx = 0
    config.m_block = 256
    config.group_min_len = 10
    config.group_max_len = 256
    config.max_len = 16
    config.gate_proj = gate_q.data_ptr()
    config.up_proj = up_q.data_ptr()
    config.down_proj = down_q.data_ptr()
    config.gate_type = ggml_type.Q4_K
    config.up_type = ggml_type.Q4_K
    config.down_type = ggml_type.Q6_K
    config.hidden_type = ggml_type.BF16
    moe = kt_kernel_ext.moe.MOE(config)
    kept = [gate, up, down, gate_q, up_q, down_q]
    return moe, kept


def main() -> None:
    require_ascend_build()
    # Creating a device tensor opens the ACL context the callback worker needs.
    probe = torch.zeros(1, device="npu")
    torch.npu.synchronize()
    del probe

    kt_kernel_ext.init_ascend_callback_worker()
    if not kt_kernel_ext.is_ascend_callback_worker_running():
        raise SystemExit("Ascend callback worker did not start")

    stream = int(torch.npu.current_stream().npu_stream)
    kt_kernel_ext.subscribe_ascend_stream(stream)

    cpu = kt_kernel_ext.CPUInfer(2)
    moe, weights = build_moe(cpu)
    cpu.submit(moe.load_weights_task())
    cpu.sync()
    cpu.submit(moe.warm_up_task())
    cpu.sync()

    hidden_npu = torch.randn((QLEN, HIDDEN), device="npu", dtype=torch.float32)
    torch.npu.synchronize()
    hidden_cpu = hidden_npu.detach().to(device="cpu", dtype=torch.float32).contiguous()
    expert_ids = torch.tensor([[0, 1]], dtype=torch.int64).contiguous()
    expert_weights = torch.tensor([[0.5, 0.5]], dtype=torch.float32).contiguous()
    qlen = torch.tensor([QLEN], dtype=torch.int32)
    hidden_q = from_float(hidden_cpu, ggml_type.BF16)
    output_q = from_float(torch.zeros((QLEN, HIDDEN), dtype=torch.float32), ggml_type.BF16)
    output_q.zero_()

    task = moe.forward_task(
        qlen.data_ptr(),
        TOP_K,
        expert_ids.data_ptr(),
        expert_weights.data_ptr(),
        hidden_q.data_ptr(),
        output_q.data_ptr(),
    )
    cpu.submit_with_cuda_stream(stream, task)
    cpu.sync_with_cuda_stream(stream)
    torch.npu.synchronize()

    output = to_float(output_q, QLEN * HIDDEN, ggml_type.BF16)
    if output.shape != (QLEN, HIDDEN):
        raise SystemExit(f"unexpected output shape {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise SystemExit("llamafile forward returned a non-finite value")
    absmax = float(output.abs().max())
    if absmax == 0.0:
        raise SystemExit("llamafile forward left the output at zero; the NPU stream callback did not run")

    on_npu = output.to(device="npu")
    if on_npu.device.type != "npu":
        raise SystemExit("forward result could not be placed on the NPU")
    # Keep the buffers alive until the stream callbacks have finished.
    del weights, hidden_cpu, hidden_q, task, on_npu
    device_name = torch.npu.get_device_name(0)
    print(
        f"{ANCHOR} device=npu:0 name={device_name} worker=running output_absmax={absmax:.6e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
