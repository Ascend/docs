"""NPU compatibility override for torchtitan v0.4.0 on Ascend.

Activated through torchtitan's official override mechanism
(``--override npu_compat.npu_attention``); no upstream source file is
modified -- the torchtitan checkout stays git clean.

One override replaces one config node (``GQAttention.Config``) and swaps one
of its children -- the **attention backend**:

* upstream ships two backends for language models:
  ``FlexInnerAttention`` (``torch.nn.attention.flex_attention``) and
  ``VarlenInnerAttention`` (``torch.nn.attention.varlen``).  Neither works
  on NPU: the varlen one calls ``aten::_flash_attention_forward``, which
  torch_npu does not implement, and the flex one fails while building its
  block mask (``torch._inductor.exc.InductorError: SympifyError``).
  Upstream's SDPA does run on NPU, but upstream forbids it as a
  language-model backend (``get_attention_config("sdpa")`` raises) because
  it cannot consume per-document positions.
  ``NpuBlockCausalSDPA`` is an ``InnerAttention`` that rebuilds the dense
  block-causal mask from the per-document positions and hands it to SDPA,
  which keeps upstream's masking semantics: a token attends to the tokens
  of its own document that precede it.

Everything else stays upstream's:

* **RoPE** -- upstream's ``ComplexRoPE`` keeps a complex64 cache and indexes
  it with the position ids; ``aclnnIndex`` has no ``DT_COMPLEX64``
  implementation, so the *eager* gather cannot run on NPU.  It does not have
  to: upstream compiles the ``rope`` region by default
  (``local_compile_regions``), and the compiled path gathers the cache's
  real view (float32) instead of the complex one -- which NPU can do.  So
  the RoPE stays upstream's, and no RoPE replacement is needed.
* **Loss / compiled regions** -- upstream's ``ChunkedLossWrapper`` and its
  compiled ``loss`` / ``fused_binary_activation`` regions run on the
  torch_npu nightly stack as they are.

The replacement is numerically equivalent to the upstream component it
stands in for.  ``check_npu_overrides.py`` (same directory) asserts that on
CPU; the quick-start doc runs it before the training steps.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F

from torchtitan.config.override import derive, override
from torchtitan.models.common.attention import GQAttention, InnerAttention


def _block_causal_mask(
    positions: torch.Tensor,
    num_tokens: int,
    sliding_window_size: int | None = None,
) -> torch.Tensor:
    """Dense ``(1, 1, T, T)`` bool mask: same document, not in the future.

    Mirrors upstream's per-document (block causal) masking: positions reset
    to 0 at every document boundary, so tokens of one document must not
    attend to another document's tokens.  Upstream derives the document ids
    the same way (``get_document_mask_mod``).

    ``sliding_window_size`` narrows the causal window to that many tokens
    (the query included), which is what upstream's sliding-window models
    (muse_glimmer) ask for -- upstream expresses the same window as
    ``window_size=(window-1, 0)`` on its varlen backend.
    """
    if positions.dim() != 1:
        raise ValueError(
            f"block-causal masking expects 1-D positions, got {positions.shape}"
        )
    doc = torch.cumsum((positions == 0).to(torch.int64), dim=0) - 1  # (T,)
    same_doc = doc[:, None] == doc[None, :]  # (T, T)
    causal = torch.ones(
        num_tokens, num_tokens, dtype=torch.bool, device=positions.device
    ).tril()
    if sliding_window_size is not None and sliding_window_size > 0:
        idx = torch.arange(num_tokens, device=positions.device)
        causal = causal & (idx[None, :] >= idx[:, None] - (sliding_window_size - 1))
    return (same_doc & causal).unsqueeze(0).unsqueeze(0)


class NpuBlockCausalMetadata(NamedTuple):
    """Per-microbatch metadata handed to ``NpuBlockCausalSDPA``.

    Upstream's inner attentions receive a backend specific metadata object
    (a ``BlockMask`` for flex, cumulative sequence lengths for varlen).  The
    NPU backend needs the per-document positions themselves, because it
    rebuilds the dense block-causal mask.
    """

    positions: torch.Tensor


class NpuBlockCausalSDPA(InnerAttention):
    """``scaled_dot_product_attention`` with an explicit block-causal mask.

    Upstream's ``ScaledDotProductAttention`` refuses an ``attention_masks``
    argument, and the trainer only builds a mask for the Flex / Varlen
    backends.  This backend keeps the per-document causal masking by
    handing SDPA a dense mask (aclnn SDPA accepts ``attn_mask``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        sliding_window_size: int | None = None
        """Sliding-window size in tokens (the query included).

        ``None`` means "attend to the whole (causal) document", which is what
        every model but the sliding-window ones asks for.  The field is named
        apart from upstream's ``window_size`` on purpose: the varlen backend
        carries a *tuple* ``(left, right)`` under that name, and ``derive()``
        copies same-named fields -- the tuple would land here.
        """

        @property
        def is_full_attention(self) -> bool:
            """Whether the backend attends to the whole (causal) document.

            A sliding-window layer is not full attention -- upstream's varlen
            backend says the same (``window_size[0] == -1``).
            """
            return self.sliding_window_size is None

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> NpuBlockCausalMetadata:
            del padding_mask, max_num_documents, max_context_length
            return NpuBlockCausalMetadata(positions=positions)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.sliding_window_size = config.sliding_window_size

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: NpuBlockCausalMetadata | None = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if attention_metadata is None:
            raise ValueError("NpuBlockCausalSDPA requires per-document positions")
        positions = attention_metadata.positions
        mask = _block_causal_mask(
            positions, q_THK.shape[0], self.sliding_window_size
        )
        out_1HTV = F.scaled_dot_product_attention(
            q_THK.transpose(0, 1).unsqueeze(0),
            k_THK.transpose(0, 1).unsqueeze(0),
            v_THV.transpose(0, 1).unsqueeze(0),
            attn_mask=mask,
            scale=scale,
            is_causal=False,
            enable_gqa=enable_gqa,
        )
        # Return a fresh tensor, not a view: upstream's flex / varlen
        # backends allocate, and a view output makes FSDP2's pre-backward
        # hook unreliable.
        return out_1HTV.squeeze(0).transpose(0, 1).contiguous()


class NpuGQAttention(GQAttention):
    """``GQAttention`` wired for NPU.

    Upstream's ``GQAttention.forward`` already hands the per-document
    metadata to the inner attention, and the NPU inner attention rebuilds
    the block-causal mask from it -- so the forward is upstream's, unchanged.
    This subclass exists so the override has a distinct config class (and a
    distinct name in the override log).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        pass


@override(
    target=GQAttention.Config,
    description=(
        "NPU attention backend: SDPA with an explicit block-causal mask "
        "(upstream's flex / varlen backends do not run on NPU)"
    ),
)
def npu_attention(cfg: GQAttention.Config) -> NpuGQAttention.Config:
    new = derive(cfg, NpuGQAttention.Config)
    # flex / varlen -> SDPA with a dense block-causal mask.  The config class
    # must not be a Flex / Varlen config, otherwise the trainer would try to
    # build a BlockMask (that path does not compile on NPU).
    #
    # derive() keeps the fields the two configs share -- notably
    # ``sharding_config`` (llama3 puts the local-SPMD sharding of q/k/v on
    # the inner attention config), which a fresh Config() would silently drop.
    new.inner_attention = derive(cfg.inner_attention, NpuBlockCausalSDPA.Config)
    return new


__all__ = [
    "NpuBlockCausalMetadata",
    "NpuBlockCausalSDPA",
    "NpuGQAttention",
    "npu_attention",
]
