"""Quick-start recipe for torchtitan (latest release) on Ascend NPU.

torchtitan's config system is recipe-based: a recipe is a plain Python
function that returns a ``Trainer.Config``, and the CLI selects it with
``--module <module> --config <function>``.  The CLI deliberately exposes a
small surface -- there is no ``--training.steps`` / ``--comm.mode`` /
``--dump-folder`` knob any more -- so a quick-start needs a recipe of its
own instead of patching upstream's config tree.

This module provides the tiny recipe the quick-start trains with:

* llama3 ``debugmodel`` (dim 256 / 6 layers / 16 heads / vocab 2048, ~6 M
  parameters);
* 2 training steps;
* one 2048-token microbatch per step;
* the bundled ``tests/assets/c4_test`` dataset -- no download, no network.

Everything else is upstream's: the loss is upstream's ``ChunkedLossWrapper``
around a plain ``CrossEntropyLoss``, and the model's compiled regions
(``loss`` / ``fused_binary_activation`` / ``rope``) are the ones upstream
compiles by default -- they run on the torch_npu nightly stack as they are.

The recipe lives outside the torchtitan checkout (next to the NPU override
modules) and is activated with ``PYTHONPATH``::

    PYTHONPATH="$HOME/tt-npu-overrides" torchrun ... -m torchtitan.train \\
        --module npu_recipe --config llama3_npu_debugmodel \\
        --override npu_compat.npu_attention --output-dir /tmp/torchtitan-quickstart
"""

from __future__ import annotations

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, Optim, OptimizersContainer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.llama3 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

SEQ_LEN = 256
"""Training context length (also the RoPE cache length)."""

NUM_TOKENS_PER_MICROBATCH = 8 * SEQ_LEN
"""Token slots per data-parallel rank per microbatch."""

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
in inductor's SIMD scheduling on NPU (``SympifyError``); it is not part of
llama3, but the filter is shared with the examples track.

The remaining regions (``fused_binary_activation`` / ``rope`` / ...) lower
fine on the torch_npu nightly stack and stay compiled -- the compiled
``rope`` region is also what makes upstream's own ``ComplexRoPE`` work on
NPU (it gathers the cache's real view instead of the complex one).
"""

MAX_NUM_DOCUMENTS = max(16, (NUM_TOKENS_PER_MICROBATCH + 511) // 512)
"""Upper bound on document segments packed into one microbatch.

Same estimate upstream's own recipes use (``c4_document_capacity``): ~16
document slots per 512 tokens, at least 16 for short batches.  It has to be
an upper bound because the packed rows are padded to it.
"""


def _debugmodel_config(*, bf16: bool = False) -> Trainer.Config:
    """Build the quick-start config.

    ``bf16`` switches the FSDP parameter dtype to bfloat16 (mixed
    precision); the fp32 variant keeps the parameters in float32.
    """
    parallelism = ParallelismConfig()
    model_config = build_model_config(
        "debugmodel",
        seq_len=SEQ_LEN,
        attn_backend="varlen",
        parallelism=parallelism,
    )
    # The compiled regions torch_npu cannot lower stay eager (see
    # UNLOWERABLE_COMPILE_REGIONS).
    model_config.local_compile_regions = [
        region
        for region in model_config.local_compile_regions
        if region not in UNLOWERABLE_COMPILE_REGIONS
    ]
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model=model_config,
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
            )
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=NUM_TOKENS_PER_MICROBATCH,
            max_context_length=model_config.max_context_length,
            steps=STEPS,
            mixed_precision_param="bfloat16" if bf16 else "float32",
        ),
        dataloader=GrainDataLoader.Config(
            max_num_documents=MAX_NUM_DOCUMENTS,
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            shuffle=False,
        ),
        metrics=MetricsProcessor.Config(log_freq=1, disable_color_printing=True),
        parallelism=parallelism,
        checkpointer=None,
        # No activation checkpointing: torch_remat's recompute hands FSDP2 a
        # *view* of the block output, and FSDP2 then cannot register its
        # pre-backward hook reliably ("returned a view tensor" warning) --
        # on torch_npu that shows up as an intermittent SIGSEGV in the
        # backward.  A ~6 M debug model does not need the memory saving.
        activation_remat=None,
        validator=None,
    )


def llama3_npu_debugmodel() -> Trainer.Config:
    """Single-card quick-start: 2 steps of the llama3 debugmodel, fp32."""
    return _debugmodel_config()


def llama3_npu_debugmodel_bf16() -> Trainer.Config:
    """Dual-card quick-start: same recipe with bfloat16 parameters."""
    return _debugmodel_config(bf16=True)


__all__ = [
    "SEQ_LEN",
    "STEPS",
    "llama3_npu_debugmodel",
    "llama3_npu_debugmodel_bf16",
]
