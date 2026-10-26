"""Quick-start test: doc under test is ``sources/torchtitan/quick_start.md``.

Run: ``python -m unittest tests.torchtitan.test_quick_start_ascend -v 2>&1``

Env (injected by the engine ``quick-start-template.yml``, triggered by
``torchtitan-quick-start.yml``): ``MONITORED_DOC_URL``, ``UPSTREAM_REF``,
``NPU_READY=true`` (otherwise the class is skipped).

Scope note: the doc body covers the smoke path for **stock torchtitan**
running under ``torch_npu`` PrivateUse1 dispatch. ``torch`` / ``torch_npu`` /
``modelscope`` / ``triton-ascend`` / ``torchtitan`` are all installed by the
doc body via ``uv pip install`` (versions per the
[Ascend PyTorch Compatibility Matrix](https://gitcode.com/Ascend/pytorch/blob/main/COMPATIBILITY.en.md):
torch==2.14.0 / torch_npu==2.14.0 / CANN==9.2.0) — ``torch``'s CPU build
only exists in the Aliyun ``pytorch-wheels/cpu`` find-links source (PyPI's
linux wheels are all CUDA builds), ``torch_npu`` comes from the default
index, which in CI is the in-cluster PyPI cache gateway. ``triton-ascend``
comes from Huawei Cloud's ascend ``test`` index (the community ``triton``
wheel is CUDA-only and must stay out of the environment). ``torchtitan`` is
installed **twice**: first from PyPI (pure-python wheel) to prove the
binary route, then from source (``git clone`` pytorch/torchtitan at the
latest release tag + ``uv pip install -e .``) because the training commands
need the upstream tree (config registry + the ``--override.imports``
mechanism).

The doc verifies:

* package + sub-module imports (``torchtitan``, ``torchtitan.config.override``,
  ``torchtitan.models.common.attention``, ``torchtitan.models.common.rope``)
* the NPU compatibility overrides register correctly (ComplexRoPE ->
  CosSinRoPE, FlexAttention -> SDPA)
* single-card training (``torchrun --nproc_per_node=1``, 2 real steps over
  the Llama 3 debug model) with NPU dispatch, which fails fast if the
  attention / RoPE kernels did not register
* dual-card training (``torchrun --nproc_per_node=2``, FSDP shard=2 +
  bf16), covering HCCL collectives and DTensor parameter sharding

Model download / ModelScope cache is only used to demonstrate the
``snapshot_download`` API — training itself uses torchtitan's bundled
``tests/assets/tokenizer``, so no checkpoint is fetched.
"""

from __future__ import annotations

import os
import re
import subprocess
import unittest

os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0,1')

from doc_test.base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else (including unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


def _doc_ref() -> str:
    """The ref the monitored doc was fetched from.

    The doc pulls the NPU override module from this repo with the same ref,
    so the two never drift.  ``raw.githubusercontent.com/<owner>/<repo>/<ref>/...``
    carries it in the path; the contents API carries it as ``?ref=``.
    """
    url = os.environ.get('MONITORED_DOC_URL', '')
    m = re.search(r'raw\.githubusercontent\.com/[^/]+/[^/]+/([^/]+)/', url)
    if m:
        return m.group(1)
    m = re.search(r'[?&]ref=([^&]+)', url)
    if m:
        return m.group(1)
    return 'main'


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """End-to-end test: fetch doc -> validate contract -> run ``#test-setup``
    / ``#test`` in order -> compare against ``#test-result``.

    ``prepare_environment`` only does runner-side scaffolding (CANN env +
    CUDA exclusion list + ``uv`` bootstrap); ``torch`` / ``torch_npu`` /
    ``modelscope`` / ``triton-ascend`` / ``torchtitan`` are all installed by
    the doc's ``#test-setup`` / ``#test`` blocks, so the test exercises the
    exact install path users get.
    """

    # 60 min per command. torchtitan's smallest meaningful command —
    # `python -m torchtitan.train --module llama3 --config
    # llama3_debugmodel --override.imports npu_compat.npu_attention
    # --training.steps 2 --comm.mode default` — needs to: clone torchtitan
    # (~5 MB sparse), `uv pip install -e .`, pull the NPU override module
    # ($HOME/tt-npu-overrides/npu_compat.py + check_npu_overrides.py, from
    # this repo at the doc's own ref, swapped in via torchtitan's official
    # @override mechanism — no upstream source patching), assert the
    # override is numerically equivalent to the upstream components it
    # stands in for, then run a 2-step HCCL training that builds the
    # Llama 3 debug model (~256 dim / 6 layers) on NPU with SDPA attention
    # (aclnn flash attention + an explicit block-causal mask; the flex path
    # hits torch_npu 2.14.0 lowering gaps, see the doc's compat section).
    # 60 minutes: first cold-cache run still compiles a handful of
    # dataloader Triton kernels through the bisheng compiler (serial,
    # TORCHINDUCTOR_COMPILE_THREADS=1, to stay under the runner cgroup
    # memory limit) plus the model build/init chain. The loss is
    # upstream's default (`llama3_debugmodel` -> ChunkedLossWrapper); the
    # 2026-09 meta-tensor leak no longer reproduces on torch 2.14.
    DEFAULT_COMMAND_TIMEOUT = 3600

    USER_AGENT = 'ascend-docs/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a CANN
    # failure surfaces a full stderr dump (head/tail by default would hide
    # the line that names the failure).
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # CANN toolkit emits this typo (sic)
        'ERR99999',
    )

    # Process-level CUDA exclusion list. Same rationale as the accelerate
    # test: write to /tmp and export, so subprocesses (subprocess.run
    # inherits parent env by default) see it. torchtitan's transitive deps
    # (datasets) occasionally pull CUDA-marked wheels if the resolver sees
    # the cluster cache's CUDA index first.
    # Also blocks community `triton` (CUDA-only backend): on NPU it must be
    # replaced by triton-ascend, and installing both clobbers the same
    # triton/ tree into a broken mix.
    _CUDA_CONSTRAINTS = (
        'triton<0',
        'cuda-toolkit<0',
        'cuda-python<0',
        'cuda-bindings<0',
        'cuda-core<0',
        'cuda-pathfinder<0',
        'flashinfer-python<0',
        'nvidia-cublas<0',
        'nvidia-cuda-runtime<0',
        'nvidia-cuda-nvrtc<0',
        'nvidia-cuda-cupti<0',
        'nvidia-cudnn<0',
        'nvidia-cudnn-frontend<0',
        'nvidia-cufft<0',
        'nvidia-curand<0',
        'nvidia-cusolver<0',
        'nvidia-cusparse<0',
        'nvidia-cusparselt<0',
        'nvidia-cutlass-dsl<0',
        'nvidia-cutlass-dsl-libs-base<0',
        'nvidia-cutlass-dsl-libs-core<0',
        'nvidia-cutlass-dsl-libs-cu12<0',
        'nvidia-ml-py<0',
        'nvidia-nccl<0',
        'nvidia-nvjitlink<0',
        'nvidia-nvtx<0',
        'nvidia-cublas-cu12<0',
        'nvidia-cuda-nvdisasm<0',
        'nvidia-cuda-runtime-cu12<0',
        'nvidia-cuda-nvrtc-cu12<0',
        'nvidia-cuda-cupti-cu12<0',
        'nvidia-cudnn-cu12<0',
        'nvidia-cufft-cu12<0',
        'nvidia-curand-cu12<0',
        'nvidia-cusolver-cu12<0',
        'nvidia-cusparse-cu12<0',
    )
    _CONSTRAINTS_FILE = '/tmp/torchtitan_npu_constraints.txt'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the GitHub workflow container image
    # (``image:`` input of ``torchtitan-quick-start.yml``).
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write the CUDA exclusion list + install uv.

        ``torch`` / ``torch_npu`` / ``modelscope`` / ``triton-ascend`` /
        ``torchtitan`` are intentionally NOT pre-installed here: the doc's
        ``#test-setup`` blocks install them, so a broken install block
        fails loudly instead of being masked by a pre-installed copy.
        """
        # 0) CANN env: source set_env.sh and merge the env stream into
        # os.environ
        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                # Don't overwrite envs explicitly injected by the
                # workflow (jobs.env / steps.env); only fill in CANN
                # keys that are missing, to avoid conflicts.
                os.environ.setdefault(key, value)
            print('setup: sourced CANN env from set_env.sh')
        else:
            print(
                f'setup: skipping CANN env source ({cls._CANN_SET_ENV} not present)'
            )

        # 1) CUDA exclusion list + process-level env
        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['UV_CONSTRAINT'] = cls._CONSTRAINTS_FILE

        # 2) uv: the doc body's install steps are all ``uv pip install ...``,
        # which gives clean stdout (no pip's "Obtaining / Installing
        # collected packages" preamble) so the #test-result fuzzy match
        # isn't polluted.
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) DOC_REF: the ref the monitored doc was fetched from. The doc
        #    pulls the NPU override module (npu_compat.py +
        #    check_npu_overrides.py) from this repo with the same ref, so
        #    the doc and the module never drift apart.
        os.environ['DOC_REF'] = _doc_ref()
        print(f'setup: DOC_REF={os.environ["DOC_REF"]}')

        # 4) uv request-level logging. uv prints NOTHING until the resolve
        # finishes, so a stalled download shows up as a silent hang in the
        # CI log. With RUST_LOG=uv=debug each HTTP request / candidate
        # wheel is logged to stderr, and the runner dumps the captured
        # stderr on timeout — the last logged URL is then the hop that
        # stalled.
        os.environ['RUST_LOG'] = 'uv=debug'

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per class.

        ``@unittest.skipIf`` only skips the test *method* — ``setUpClass``
        itself always runs. The ``if _e2e_enabled()`` body guard below is
        what actually keeps heavy setup from firing when ``NPU_READY`` is
        unset.
        """
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        """Template-method entry point. The base class
        ``run_template()`` runs the full ``pre_process`` -> ``parse`` ->
        ``execute`` -> ``post_process`` flow. ``prepare_environment`` is
        triggered by ``setUpClass`` once, not from ``run_template``."""

        self.run_template()


if __name__ == '__main__':
    unittest.main()
