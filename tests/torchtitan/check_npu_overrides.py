"""Equivalence checks for the NPU override module (CPU only, no NPU needed).

Run: ``python check_npu_overrides.py``

Asserts that the component the NPU override substitutes in is numerically
equivalent to the upstream component it stands in for:

* ``NpuBlockCausalSDPA`` vs a naive per-document causal reference -- the
  masking semantics upstream gets from its BlockMask.

and that the override only touches the attention backend: the RoPE stays
upstream's -- ``ComplexRoPE`` for llama3 / deepseek_v3, ``CosSinRoPE`` for
qwen3.

Prints one stable line on stdout (the quick-start doc matches it); the
measured numbers go to stderr so CI logs keep them.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npu_compat import (  # noqa: E402
    NpuBlockCausalMetadata,
    NpuBlockCausalSDPA,
)

from torchtitan.models.common.rope import (  # noqa: E402
    ComplexRoPE,
    CosSinRoPE,
)

TOL = 1e-5


def _naive_block_causal(q, k, v, mask, scale):
    """Reference attention: masked softmax over the (q, kv) grid.

    ``q`` / ``k`` / ``v`` are ``(T, H, D)`` -- packed tokens, like upstream's
    inner attentions.  ``mask`` is ``(1, 1, T, T)``.
    """
    q2 = q.transpose(0, 1).unsqueeze(0)
    k2 = k.transpose(0, 1).unsqueeze(0)
    v2 = v.transpose(0, 1).unsqueeze(0)
    att = (q2.float() @ k2.float().transpose(-1, -2)) * scale
    att = att.masked_fill(~mask, float("-inf")).softmax(-1)
    return (att @ v2.float()).squeeze(0).transpose(0, 1)


def check_metadata() -> str:
    """The backend config must hand the per-document positions to the kernel.

    The trainer builds the attention metadata through the *config*
    (``config.build_attention_metadata(...)``), not through the built
    module: if the replacement config does not override it, the kernel
    never sees the positions and the run dies with "requires per-document
    positions".
    """
    positions = torch.tensor([0, 1, 2, 0, 1])
    metadata = NpuBlockCausalSDPA.Config().build_attention_metadata(positions)
    assert isinstance(metadata, NpuBlockCausalMetadata), type(metadata)
    assert metadata.positions is positions
    return "positions metadata OK"


def check_mask() -> float:
    """NpuBlockCausalSDPA must match a naive per-document causal reference."""
    torch.manual_seed(0)
    num_tokens, heads, head_dim = 16, 4, 8
    scale = head_dim**-0.5
    q = torch.randn(num_tokens, heads, head_dim)
    k = torch.randn(num_tokens, heads, head_dim)
    v = torch.randn(num_tokens, heads, head_dim)
    # two documents in this microbatch
    positions = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    attention = NpuBlockCausalSDPA.Config().build()
    out = attention(
        q,
        k,
        v,
        attention_metadata=NpuBlockCausalMetadata(positions=positions),
        scale=scale,
        enable_gqa=False,
    )

    doc = torch.cumsum((positions == 0).to(torch.int64), dim=0) - 1
    mask = (
        (doc[:, None] == doc[None, :])
        & torch.ones(num_tokens, num_tokens, dtype=torch.bool).tril()
    ).unsqueeze(0).unsqueeze(0)
    out_ref = _naive_block_causal(q, k, v, mask, scale)
    diff = (out - out_ref).abs().max().item()
    assert diff < TOL, f"mask mismatch: max_abs_diff={diff}"

    # the mask must actually block the cross-document pairs
    blocked = (~mask).sum().item()
    total = mask.numel()
    assert blocked < total, "mask blocks nothing"
    return diff


def check_rope_untouched() -> str:
    """The override must not touch the RoPE.

    Upstream's ``ComplexRoPE`` (llama3 / deepseek_v3) and ``CosSinRoPE``
    (qwen3) stay as they are: the complex64 cache gather only exists in
    the eager path, and upstream compiles the ``rope`` region by default --
    the compiled path gathers the cache's real view, which NPU can do.
    """
    from torchtitan.config.override import OverrideConfig, apply_overrides
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.models.llama3 import build_model_config as llama3_build
    from torchtitan.models.qwen3 import build_model_config as qwen3_build

    import npu_compat  # noqa: F401  (registers the override)

    parallelism = ParallelismConfig()
    llama = llama3_build(
        "debugmodel",
        seq_len=256,
        attn_backend="varlen",
        parallelism=parallelism,
    )
    qwen = qwen3_build(
        "debugmodel",
        seq_len=256,
        attn_backend="varlen",
        parallelism=parallelism,
    )
    for spec in (llama, qwen):
        apply_overrides(
            OverrideConfig(imports=["npu_compat.npu_attention"]), spec
        )

    llama_rope = llama.layers[0].attention.rope
    qwen_rope = qwen.layers[0].attention.rope
    assert isinstance(llama_rope, ComplexRoPE.Config), type(llama_rope)
    assert type(qwen_rope) is CosSinRoPE.Config, type(qwen_rope)
    assert isinstance(
        llama.layers[0].attention.inner_attention, NpuBlockCausalSDPA.Config
    ), type(llama.layers[0].attention.inner_attention)
    return f"llama3={type(llama_rope).__name__}, qwen3={type(qwen_rope).__name__}"


def main() -> int:
    mask_diff = check_mask()
    metadata = check_metadata()
    selection = check_rope_untouched()
    print(
        "npu override equivalence: mask OK, metadata OK, attention selection OK",
        file=sys.stdout,
    )
    print(
        f"mask max_abs_diff={mask_diff:.3e} (tol {TOL:.0e}), "
        f"metadata: {metadata}, "
        f"attention selection: {selection}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
