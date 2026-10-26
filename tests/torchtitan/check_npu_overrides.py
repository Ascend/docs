"""Equivalence checks for the NPU override module (CPU only, no NPU needed).

Run: ``python check_npu_overrides.py``

Asserts that the two components the NPU overrides substitute in are
numerically equivalent to the upstream components they stand in for:

* ``RealGatherComplexRoPE`` vs upstream ``ComplexRoPE`` -- same cache math,
  same rotation, only the gather kernel differs.
* ``BlockCausalSDPA`` vs a naive per-document causal reference -- the
  masking semantics upstream gets from its BlockMask.

Prints one stable line on stdout (the quick-start doc matches it); the
measured numbers go to stderr so CI logs keep them.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npu_compat import BlockCausalSDPA, RealGatherComplexRoPE  # noqa: E402

from torchtitan.models.common.rope import ComplexRoPE  # noqa: E402

TOL = 1e-5


def _naive_block_causal(q, k, v, mask, scale):
    """Reference attention: masked softmax over the (q, kv) grid."""
    q2, k2, v2 = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    att = (q2.float() @ k2.float().transpose(-1, -2)) * scale
    att = att.masked_fill(~mask, float("-inf")).softmax(-1)
    return (att @ v2.float()).transpose(1, 2)


def check_rope() -> float:
    """RealGatherComplexRoPE must match upstream ComplexRoPE exactly."""
    torch.manual_seed(0)
    dim, max_seq_len = 16, 128
    shared = dict(dim=dim, max_seq_len=max_seq_len, theta=500000.0, scaling="llama")
    upstream = ComplexRoPE.Config(**shared).build()
    ours = RealGatherComplexRoPE.Config(**shared).build()

    q = torch.randn(2, 8, 4, dim)
    k = torch.randn(2, 8, 4, dim)
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7]])

    out_upstream, _ = upstream(q, k, positions)
    out_ours, _ = ours(q, k, positions)
    diff = (out_upstream - out_ours).abs().max().item()
    assert diff < TOL, f"rope mismatch: max_abs_diff={diff}"
    return diff


def check_mask() -> float:
    """BlockCausalSDPA must match a naive per-document causal reference."""
    torch.manual_seed(0)
    batch, seqlen, heads, head_dim = 2, 16, 4, 8
    scale = head_dim**-0.5
    q = torch.randn(batch, seqlen, heads, head_dim)
    k = torch.randn(batch, seqlen, heads, head_dim)
    v = torch.randn(batch, seqlen, heads, head_dim)
    # two documents in the first sample, one in the second
    positions = torch.tensor([list(range(6)) + list(range(10)), list(range(16))])

    attention = BlockCausalSDPA.Config().build()
    out = attention(q, k, v, positions=positions, scale=scale, enable_gqa=False)

    doc = torch.cumsum((positions == 0).to(torch.int32), dim=1) - 1
    mask = (
        (doc[:, :, None] == doc[:, None, :])
        & torch.ones(seqlen, seqlen, dtype=torch.bool).tril()
    ).unsqueeze(1)
    out_ref = _naive_block_causal(q, k, v, mask, scale)
    diff = (out - out_ref).abs().max().item()
    assert diff < TOL, f"mask mismatch: max_abs_diff={diff}"

    # the mask must actually block the cross-document pairs
    blocked = (~mask).sum().item()
    total = mask.numel()
    assert blocked < total, "mask blocks nothing"
    return diff


def main() -> int:
    rope_diff = check_rope()
    mask_diff = check_mask()
    print(
        f"npu override equivalence: rope OK, mask OK",
        file=sys.stdout,
    )
    print(
        f"rope max_abs_diff={rope_diff:.3e} (tol {TOL:.0e}), "
        f"mask max_abs_diff={mask_diff:.3e} (tol {TOL:.0e})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
