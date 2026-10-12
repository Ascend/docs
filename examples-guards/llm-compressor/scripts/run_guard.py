#!/usr/bin/env python3
"""Run one upstream llm-compressor example and require NPU tensors.

The example file is executed unchanged via runpy. This process registers
torch_npu, remembers the model passed to oneshot so the device check can
see it, and checks the resulting parameters landed on npu. It does not
resize the example's dataset or rewrite the example file.
"""

from __future__ import annotations

import json
import os
import runpy
import shlex
import sys

import llmcompressor
import torch
import torch_npu


def overlay_args() -> list[str]:
    raw = os.environ.get("OVERLAY_ARGS", "").strip()
    if not raw or raw in ("null", '""', "[]"):
        return []
    items = json.loads(raw)
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise SystemExit("OVERLAY_ARGS must be a JSON array of strings")
    args: list[str] = []
    for item in items:
        args.extend(shlex.split(os.path.expandvars(item), posix=True))
    return args


def patch_accelerator_memory() -> None:
    """Some NPU builds omit accelerator memory queries used by logging."""
    accel = getattr(torch, "accelerator", None)
    if accel is None:
        return
    try:
        accel.get_memory_info()
    except Exception:
        accel.get_memory_info = lambda device=None: (32 * 1024**3, 32 * 1024**3)
    try:
        accel.max_memory_allocated()
    except Exception:
        accel.max_memory_allocated = lambda device=None: 0


# Kept alive after the example returns, including when `model` is local to a function.
SEEN_MODEL: dict[str, torch.nn.Module] = {}


def remember_oneshot_model() -> None:
    """Keep the model argument so the device check can see it later."""
    original = llmcompressor.oneshot

    def oneshot(*args, **kwargs):
        model = kwargs.get("model")
        if not isinstance(model, torch.nn.Module) and args and isinstance(args[0], torch.nn.Module):
            model = args[0]
        if isinstance(model, torch.nn.Module):
            SEEN_MODEL["model"] = model
        return original(*args, **kwargs)

    llmcompressor.oneshot = oneshot


def is_rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


def parameter_devices(model: torch.nn.Module) -> set[str]:
    return {str(param.device) for param in model.parameters()}


def assert_on_npu(model: torch.nn.Module) -> None:
    devices = parameter_devices(model)
    if not any(device.startswith("npu") for device in devices):
        raise SystemExit(f"workload not on npu after the example: {sorted(devices)}")
    print("oneshot_param_devices", sorted(devices))


def assert_inference(namespace: dict) -> None:
    model = namespace["compressed_model"]
    inputs = namespace["inputs"]
    param_devices = parameter_devices(model)
    input_devices: set[str] = set()
    values = inputs.values() if hasattr(inputs, "values") else [inputs]
    for value in values:
        if hasattr(value, "device"):
            input_devices.add(str(value.device))
    if param_devices != {"npu:0"} or input_devices != {"npu:0"}:
        raise SystemExit(
            "workload not on npu:0: "
            f"model={sorted(param_devices)} inputs={sorted(input_devices)}"
        )


def find_model(namespace: dict) -> torch.nn.Module | None:
    model = namespace.get("model")
    if isinstance(model, torch.nn.Module):
        return model
    return SEEN_MODEL.get("model")


def main() -> None:
    if not torch.npu.is_available():
        raise SystemExit("torch.npu.is_available() is false before the example")
    patch_accelerator_memory()
    remember_oneshot_model()
    path = os.environ["EXAMPLE_PATH"]
    kind = os.environ["GUARD_KIND"]
    sys.argv = [path, *overlay_args()]
    namespace = runpy.run_path(path)
    if not is_rank0():
        return
    if kind == "npu_inference":
        assert_inference(namespace)
    else:
        model = find_model(namespace)
        if model is None:
            raise SystemExit("oneshot example did not expose model")
        assert_on_npu(model)
    print("LLM_COMPRESSOR_WORKLOAD_DEVICE=npu:0")


if __name__ == "__main__":
    main()
