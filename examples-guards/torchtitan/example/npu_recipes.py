"""NPU smoke recipes for the torchtitan examples track (latest release).

torchtitan v0.4.0 replaced the old "module + a long list of ``--training.*``
flags" configuration with a **recipe** system: the CLI takes

    --module <module> --config <function>

and calls ``<function>()`` with **no arguments**; the returned
``Trainer.Config`` *is* the configuration.  ``--training.steps`` /
``--comm.mode`` / ``--dump-folder`` are gone (the equivalents are
``--comm-backend`` and ``--output-dir``).

The examples track therefore cannot drive upstream's recipe functions
directly -- they need three adjustments, all of them CI-specific:

1. **Two steps.**  Upstream's recipes run 10 to 1000 steps; a CI smoke needs
   two to prove the whole path (data -> forward -> backward -> optimizer ->
   metrics) works.
2. **The compiled regions torch_npu cannot lower stay eager** (see
   ``UNLOWERABLE_COMPILE_REGIONS``), and **activation checkpointing is off**
   (see ``NO_ACTIVATION_REMAT``).
3. **The bundled dataset.**  The size-tier recipes read HF's
   ``allenai/c4``, which the cluster cannot reach; the upstream checkout
   ships ``tests/assets/c4_test`` for exactly this purpose.

Everything else is upstream's -- the loss (upstream's
``ChunkedLossWrapper``) and the compiled regions run on the torch_npu
nightly stack as they are.

Every function below calls the matching **upstream** recipe and applies
those adjustments on top, so the upstream checkout stays untouched and the
recipes keep tracking upstream's own configuration.  The NPU-incompatible
attention backend (upstream's flex / varlen backends) is swapped separately,
through torchtitan's official override mechanism
(``--override npu_compat.npu_attention``).

The recipes are selected like this::

    torchrun --nproc_per_node=1 -m torchtitan.train \\
        --module npu_recipes --config llama3_debugmodel \\
        --override npu_compat.npu_attention --output-dir /tmp/out
"""

from __future__ import annotations

import dataclasses
from typing import Any

from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models import (
    deepseek_v3,
    llama3,
    muse_glimmer,
    qwen3,
)

SEQ_LEN = 256
"""Training context length (also the RoPE cache length)."""

STEPS = 2
"""How many train steps to run."""

UNLOWERABLE_COMPILE_REGIONS = ("loss", "moe_aux_loss")
"""Compiled regions upstream enables that torch_npu cannot lower.

``loss`` -- the region's graph ends in ``aten.nll_loss_forward``, which
inductor cannot lower for NPU: it registers a *fallback* instead, and
torch 2.15 turns "a decomposition exists, we should switch to it" from a
warning into a hard assertion **whenever the ``CI`` env var is set**
(``torch/_inductor/lowering.py`` guards it with ``os.getenv("CI")``) -- so
on a CI runner the compiled loss region dies with::

    torch._inductor.exc.InductorError: AssertionError: make_fallback
    (aten.nll_loss_forward.default): a decomposition exists, ...

Keeping the region eager avoids the whole question: the eager NPU kernel
handles the loss (and upstream's ``ChunkedLossWrapper`` runs fine that way).

``moe_aux_loss`` -- deepseek_v3's MoE router compiles a reduction that dies
in inductor's SIMD scheduling on NPU (``torch._inductor.exc.InductorError:
SympifyError``).

The remaining regions (``fused_binary_activation`` / ``fp32_to_bf16_split``
/ ...) lower fine on the torch_npu nightly stack and stay compiled.

The ``rope`` region is per-family (see ``_rope_region_is_eager``): the
ComplexRoPE models need it compiled, the CosSinRoPE models need it eager.
"""

NO_ACTIVATION_REMAT = """
# No activation checkpointing: torch_remat's recompute hands FSDP2 a *view* of
# the block output, and FSDP2 then cannot register its pre-backward hook
# reliably ("returned a view tensor" warning) -- on torch_npu that shows up
# as an intermittent SIGSEGV in the backward.  The smokes are small enough
# not to need the memory saving.
"""


def _rope_region_is_eager(config: Trainer.Config) -> bool:
    """Whether the model's ``rope`` region has to stay eager.

    * ``ComplexRoPE`` models (llama3 / deepseek_v3 / muse_glimmer): the
      region **must** stay compiled.  Their eager gather of the complex64
      cache cannot run on NPU (aclnnIndex has no DT_COMPLEX64), while the
      compiled path gathers the cache's real view -- which NPU can do.
    * ``CosSinRoPE`` models (qwen3): the cache is real, so eager works.
      Their *compiled* rope backward hits the ``slice_backward`` fallback,
      which torch 2.15 turns into a hard assertion when CI is set -- so the
      region stays eager for them.
    """
    from torchtitan.models.common.rope import CosSinRoPE

    layers = config.model.layers
    first = layers[0] if isinstance(layers, dict) else layers[0]
    return isinstance(first.attention.rope, CosSinRoPE.Config)


def _npu_smoke(
    config: Trainer.Config,
    *,
    dataset: Any = None,
    bf16: bool = False,
) -> Trainer.Config:
    """Apply the NPU smoke adjustments to an upstream recipe config.

    Args:
        config: the config an upstream recipe returned.
        dataset: the dataset config to use; defaults to upstream's (the
            bundled ``c4_test`` for the debugmodel recipes, HF's ``c4`` for
            the size tiers -- the caller overrides the latter).
        bf16: run the whole training in bfloat16 (upstream's own
            convention for a bf16 run: ``dtype="bfloat16"`` plus
            ``mixed_precision_param="bfloat16"``); the default is pure
            float32, which is what the single-card smokes used before
            v0.4.0.
    """
    # 1) Two steps.
    config.training.steps = STEPS

    # 2) The compiled regions torch_npu cannot lower stay eager (see
    #    UNLOWERABLE_COMPILE_REGIONS and _rope_region_is_eager).
    assert config.model is not None
    unlowerable = set(UNLOWERABLE_COMPILE_REGIONS)
    if _rope_region_is_eager(config):
        unlowerable.add("rope")
    config.model.local_compile_regions = [
        region
        for region in config.model.local_compile_regions
        if region not in unlowerable
    ]

    # 3) No activation checkpointing (see NO_ACTIVATION_REMAT).
    config.activation_remat = None

    # 4) The dataset: the size-tier recipes point at HF's allenai/c4, which
    #    the cluster cannot reach; the bundled c4_test is the local copy.
    if dataset is not None:
        config.dataloader.dataset = dataclasses.replace(
            config.dataloader.dataset, dataset=dataset
        )

    # 5) The CI log is compared as plain text.
    config.metrics.disable_color_printing = True

    # 6) The dtype.  ``dtype`` is the global default dtype -- upstream's
    #    docstring: setting it to "bfloat16" puts parameters, gradients and
    #    optimizer states in bfloat16, without an extra copy of fp32
    #    weights.  ``mixed_precision_param`` is FSDP's
    #    MixedPrecisionPolicy param_dtype (upstream's default is
    #    "bfloat16", i.e. fp32 weights computed in bf16).
    #    The small smokes run pure float32 (what the pre-v0.4.0 examples
    #    ran, and what was verified green on NPU); the two-card entry and
    #    the 16B size tier run the full bfloat16 recipe upstream uses for a
    #    bf16 run, which also exercises the FSDP2 mixed-precision policy.
    #    The 16B tier *needs* it: 16B parameters x 16 bytes (fp32 weight +
    #    fp32 gradient + two fp32 AdamW states) is 256 GB, i.e. exactly the
    #    8 x 32 GB of the eight-card runner, before any activation.
    if bf16:
        config.training.dtype = "bfloat16"
        config.training.mixed_precision_param = "bfloat16"
    else:
        config.training.dtype = "float32"
        config.training.mixed_precision_param = "float32"

    return config


def llama3_debugmodel() -> Trainer.Config:
    """llama3 debugmodel, one card, float32."""
    config = llama3.llama3_debugmodel(seq_len=SEQ_LEN)
    return _npu_smoke(config)


def llama3_debugmodel_bf16() -> Trainer.Config:
    """llama3 debugmodel, two cards, bfloat16."""
    config = llama3.llama3_debugmodel(seq_len=SEQ_LEN)
    return _npu_smoke(config, bf16=True)


def llama3_debugmodel_varlen_attn() -> Trainer.Config:
    """llama3 debugmodel with upstream's varlen attention backend."""
    config = llama3.llama3_debugmodel_varlen_attn(seq_len=SEQ_LEN)
    return _npu_smoke(config)


def llama3_debugmodel_ce_loss() -> Trainer.Config:
    """llama3 debugmodel with upstream's plain (non-chunked) loss."""
    config = llama3.llama3_debugmodel_ce_loss(seq_len=SEQ_LEN)
    return _npu_smoke(config)


def llama3_debugmodel_dist_gemm() -> Trainer.Config:
    """llama3 debugmodel with upstream's async-TP transform.

    Without tensor parallelism the fused modules fall back to the stock
    projections, so the recipe stays runnable on a single rank.
    """
    config = llama3.llama3_debugmodel_dist_gemm(seq_len=SEQ_LEN)
    return _npu_smoke(config)


def sft_debugmodel() -> Trainer.Config:
    """llama3 debugmodel with upstream's SFT (chat) dataset."""
    config = llama3.sft_debugmodel(seq_len=SEQ_LEN)
    return _npu_smoke(config)


def qwen3_debugmodel() -> Trainer.Config:
    """qwen3 debugmodel, one card."""
    config = qwen3.qwen3_debugmodel(seq_len=SEQ_LEN)
    return _npu_smoke(config)


def qwen3_debugmodel_moe_param_groups() -> Trainer.Config:
    """qwen3 MoE debugmodel with upstream's per-parameter optimizer groups."""
    config = qwen3.qwen3_debugmodel_moe_param_groups(seq_len=SEQ_LEN)
    return _npu_smoke(config)


def qwen3_moe_debug() -> Trainer.Config:
    """qwen3 MoE debugmodel."""
    config = qwen3.qwen3_moe_debug(seq_len=SEQ_LEN)
    return _npu_smoke(config)


def qwen3_0_6b() -> Trainer.Config:
    """qwen3 0.6B -- a size tier, with the real tokenizer and the bundled data."""
    config = qwen3.qwen3_0_6b(seq_len=SEQ_LEN)
    return _npu_smoke(config, dataset=DATASETS["c4_test"])


def qwen3_1_7b() -> Trainer.Config:
    """qwen3 1.7B -- a size tier, with the real tokenizer and the bundled data."""
    config = qwen3.qwen3_1_7b(seq_len=SEQ_LEN)
    return _npu_smoke(config, dataset=DATASETS["c4_test"])


def deepseek_v3_debugmodel() -> Trainer.Config:
    """deepseek_v3 debugmodel, one card."""
    config = deepseek_v3.deepseek_v3_debugmodel(seq_len=SEQ_LEN)
    return _npu_smoke(config)


def deepseek_v3_debugmodel_mtp() -> Trainer.Config:
    """deepseek_v3 debugmodel with upstream's multi-token-prediction loss."""
    config = deepseek_v3.deepseek_v3_debugmodel_mtp(seq_len=SEQ_LEN)
    return _npu_smoke(config)


def deepseek_v3_16b() -> Trainer.Config:
    """deepseek_v3 16B -- a size tier with expert parallelism over 8 cards.

    Runs in bfloat16: in float32 the parameters, gradients and optimizer
    states alone need 16B x 16 bytes = 256 GB, which is exactly the eight
    32 GB cards of the runner.
    """
    config = deepseek_v3.deepseek_v3_16b(seq_len=SEQ_LEN)
    return _npu_smoke(config, dataset=DATASETS["c4_test"], bf16=True)


def muse_glimmer_debugmodel() -> Trainer.Config:
    """muse_glimmer debugmodel (text path)."""
    config = muse_glimmer.muse_glimmer_debugmodel(seq_len=SEQ_LEN)
    return _npu_smoke(config)


__all__ = [
    "SEQ_LEN",
    "STEPS",
    "deepseek_v3_16b",
    "deepseek_v3_debugmodel",
    "deepseek_v3_debugmodel_mtp",
    "llama3_debugmodel",
    "llama3_debugmodel_bf16",
    "llama3_debugmodel_ce_loss",
    "llama3_debugmodel_dist_gemm",
    "llama3_debugmodel_varlen_attn",
    "muse_glimmer_debugmodel",
    "qwen3_0_6b",
    "qwen3_1_7b",
    "qwen3_debugmodel",
    "qwen3_debugmodel_moe_param_groups",
    "qwen3_moe_debug",
    "sft_debugmodel",
]
