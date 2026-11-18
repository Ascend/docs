"""NPU override for the muse_glimmer family.

muse_glimmer's attention is its own class (``models/muse_glimmer/model.py``:
``class Attention(GQAttention)``) which adds three config fields on top of
``GQAttention.Config``:

* ``scale_query_by`` -- a tuned scaling applied to the queries after
  q-norm;
* ``o_gate`` -- a sigmoid output gate;
* ``window_size`` -- a per-layer sliding window (the debugmodel flavor uses
  the pattern ``[128, 128, 128, 0]``).

The generic override in ``npu_compat.py`` targets ``GQAttention.Config``, and
the override mechanism matches subclasses -- so it *would* claim muse's
attention node.  But its replacement (``NpuGQAttention``) has none of those
three fields, and ``derive()`` drops the fields the source has and the
target does not: the model would silently train without the query scaling
and without the output gate.

muse therefore gets its own override, whose replacement subclasses muse's
own attention: every field survives, and the NPU fix (the SDPA with an
explicit dense block-causal mask) is applied on top.  The per-layer sliding window travels to the NPU inner
attention, which masks it inside the dense mask -- upstream expresses the
same window as ``window_size=(window-1, 0)`` on its varlen backend.

Because the two overrides would claim the same node, the muse entry
activates **only** this one::

    --override npu_compat_muse.muse_attention
"""

from dataclasses import dataclass

from torchtitan.config.override import derive, override
from torchtitan.models.muse_glimmer.model import Attention as MuseAttention

from npu_compat import NpuBlockCausalSDPA


class NpuMuseAttention(MuseAttention):
    """upstream ``muse_glimmer`` attention, wired for NPU.

    The forward is upstream's: it keeps the query scaling and the output
    gate, forwards the per-document metadata to the inner attention, and the
    NPU inner attention rebuilds the (windowed) block-causal mask from it.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MuseAttention.Config):
        pass


@override(
    target=MuseAttention.Config,
    description=(
        "muse_glimmer attention: SDPA with an explicit block-causal mask, "
        "keeping the query scaling, the output gate and the per-layer "
        "sliding window"
    ),
)
def muse_attention(cfg: MuseAttention.Config) -> NpuMuseAttention.Config:
    new = derive(cfg, NpuMuseAttention.Config)
    # derive() keeps the fields the two configs share -- notably
    # ``sharding_config``, which the varlen backend carries.
    new.inner_attention = derive(cfg.inner_attention, NpuBlockCausalSDPA.Config)
    # The per-layer sliding window (muse's ``window_size``) travels to the
    # inner attention, which masks it inside the dense block-causal mask.
    new.inner_attention.sliding_window_size = cfg.window_size
    return new


__all__ = ["NpuMuseAttention", "muse_attention"]
