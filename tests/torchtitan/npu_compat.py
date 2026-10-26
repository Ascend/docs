"""NPU compatibility overrides for torchtitan v0.3.0 on Ascend.

Activated through torchtitan's official override mechanism
(``--override.imports``); no upstream source file is modified -- the
torchtitan checkout stays git clean.

One override replaces one config node (``GQAttention.Config``) because the
override framework rejects *nested* overrides (an override claiming an
ancestor of another override's node raises).  That single replacement
carries both NPU fixes:

1. **RoPE** -- upstream ``ComplexRoPE`` keeps a complex64 cache and indexes
   it with the position ids; ``aclnnIndex`` has no ``DT_COMPLEX64``
   implementation.  ``RealGatherComplexRoPE`` performs the same gather in
   real space (the real / imaginary halves as float32) and hands the result
   back to upstream's own rotation code, so the values -- including the
   llama frequency scaling and the adjacent-dimension pairing convention --
   are upstream's.  Upstream's ``CosSinRoPE`` is *not* a substitute: it
   uses the rotate-half pairing convention and rejects
   ``scaling="llama"`` outright.

2. **Attention** -- upstream builds the per-document block-causal mask
   only for the Flex / Varlen backends, and neither compiles on NPU.
   Upstream's SDPA does compile, but upstream forbids it as a
   language-model backend (``get_attention_config("sdpa")`` raises) because
   it cannot consume per-document positions.  ``BlockCausalSDPA`` feeds
   SDPA an explicit dense block-causal mask instead, which keeps upstream's
   masking semantics: a token attends to the tokens of its own document
   that precede it.

Both replacements are numerically equivalent to the upstream components
they stand in for.  ``check_npu_overrides.py`` (same directory) asserts
that on CPU; the quick-start doc runs it before the training steps.

Upstream private helpers used below (``_maybe_wrap_positions``,
``_maybe_check_max_pos``, ``_reshape_for_broadcast``): if upstream renames
them this module fails loudly at import time.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.config.override import derive, override
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.rope import (
    ComplexRoPE,
    _maybe_check_max_pos,
    _maybe_wrap_positions,
    _reshape_for_broadcast,
)
from torchtitan.protocols.module import Module


class RealGatherComplexRoPE(ComplexRoPE):
    """``ComplexRoPE`` whose position gather happens in real space.

    Upstream's ``ComplexRoPE._reshape_cache`` indexes the complex64 cache
    directly (``rope_cache[positions]``), which NPU cannot do.  Here the
    cache is viewed as float32 (real / imaginary interleaved), gathered as
    a plain float32 tensor, and converted back to complex64 before
    upstream's ``apply_rotary_emb`` runs.  The gathered values are
    bit-identical to upstream's; only the kernel that produces them
    differs.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ComplexRoPE.Config):
        pass

    def _reshape_cache(self, query, positions=None):
        positions = _maybe_wrap_positions(positions, query)
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=self.cache.shape[0] - 1)
        # complex64 (max_seq_len, dim/2) -> float32 (max_seq_len, dim) with
        # real/imag interleaved -> two contiguous (max_seq_len, dim/2) caches.
        real = self.cache.view(torch.float32)
        cos_cache = real[:, 0::2].contiguous()
        sin_cache = real[:, 1::2].contiguous()
        # Same broadcast rules as upstream: the pair shape is
        # (batch, seq, heads, head_dim / 2).
        pair_shape = (*query.shape[:-1], query.shape[-1] // 2)
        cos = _reshape_for_broadcast(cos_cache, pair_shape, positions)
        sin = _reshape_for_broadcast(sin_cache, pair_shape, positions)
        # Back to the complex64 layout upstream's rotation expects.
        return torch.stack([cos, sin], dim=-1).flatten(-2).view(torch.complex64)


def _block_causal_mask(
    positions: torch.Tensor, seqlen: int, device: torch.device
) -> torch.Tensor:
    """Dense ``(B, 1, L, L)`` bool mask: same document, not in the future.

    Mirrors upstream's per-document (block causal) masking: positions reset
    to 0 at every document boundary, so tokens of one document must not
    attend to another document's tokens.
    """
    pos = positions if positions.dim() > 1 else positions.unsqueeze(0)
    doc = torch.cumsum((pos == 0).to(torch.int32), dim=1) - 1  # (B, L)
    same_doc = doc[:, :, None] == doc[:, None, :]  # (B, L, L)
    causal = torch.ones(seqlen, seqlen, dtype=torch.bool, device=device).tril()
    return (same_doc & causal).unsqueeze(1)


class BlockCausalSDPA(Module):
    """``scaled_dot_product_attention`` with an explicit block-causal mask.

    Upstream's ``ScaledDotProductAttention`` refuses an ``attention_masks``
    argument, and the trainer only builds a mask for the Flex / Varlen
    backends.  This backend keeps the per-document causal masking by
    handing SDPA a dense mask (aclnn SDPA accepts ``attn_mask``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        *,
        attention_masks=None,
        positions: torch.Tensor | None = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if positions is None:
            raise ValueError("BlockCausalSDPA requires positions")
        q_BNLH = q_BLNH.transpose(1, 2)
        k_BNLH = k_BLNH.transpose(1, 2)
        v_BNLH = v_BLNH.transpose(1, 2)
        mask = _block_causal_mask(positions, q_BLNH.shape[1], q_BLNH.device)
        out_BNLH = F.scaled_dot_product_attention(
            q_BNLH,
            k_BNLH,
            v_BNLH,
            attn_mask=mask,
            scale=scale,
            is_causal=False,
            enable_gqa=enable_gqa,
        )
        return out_BNLH.transpose(1, 2)


class NpuGQAttention(GQAttention):
    """``GQAttention`` that forwards ``positions`` to the inner attention.

    Upstream's ``GQAttention.forward`` consumes ``positions`` for the RoPE
    but does not pass it on, so an inner attention that has to rebuild the
    per-document mask (everything but Flex / Varlen) cannot see it.  The
    body below is upstream v0.3.0's ``GQAttention.forward`` with exactly
    one addition: ``positions=positions`` in the inner-attention call.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        pass

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xq_BLNH, xk_BLNH, xv_BLNH = self.qkv_linear(x_BLD)

        # Optional QK normalization (before RoPE, per Qwen3)
        if self.q_norm is not None or self.k_norm is not None:
            assert self.q_norm is not None and self.k_norm is not None
            xq_BLNH = self.q_norm(xq_BLNH)
            xk_BLNH = self.k_norm(xk_BLNH)

        # Apply rotary embeddings
        xq_BLNH, xk_BLNH = self.rope(xq_BLNH, xk_BLNH, positions)

        # inner_attention returns (B, L, N, H)
        out_BLNH = self.inner_attention(
            xq_BLNH,
            xk_BLNH,
            xv_BLNH,
            attention_masks=attention_masks,
            positions=positions,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()
        out_BLD = out_BLNH.flatten(2)
        return self.wo(out_BLD)


@override(
    target=GQAttention.Config,
    description=(
        "NPU attention stack: real-space RoPE gather (aclnnIndex has no "
        "DT_COMPLEX64) + SDPA with an explicit block-causal mask"
    ),
)
def npu_attention(cfg: GQAttention.Config) -> NpuGQAttention.Config:
    new = derive(cfg, NpuGQAttention.Config)
    # complex64 RoPE cache -> same values, gathered in real space
    new.rope = derive(new.rope, RealGatherComplexRoPE.Config)
    # flex -> SDPA with a dense block-causal mask.  The config class must
    # not be a Flex / Varlen config, otherwise the trainer would try to
    # build a BlockMask (that path does not compile on NPU).
    new.inner_attention = BlockCausalSDPA.Config()
    return new
