"""Quick-start test: doc under test is ``sources/specforge/quick_start.md``.

End-to-end case built on top of the ``MarkdownDocTestBase`` contract:
every ``shell`` / ``python`` code block carries one of the ``#test`` /
``#test-setup`` / ``#test-result`` labels plus ``id=`` / ``store=`` /
``load='x>>y'`` / ``fuzzy='xxx'`` parameters.

Run: ``python -m unittest tests.specforge.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the quick-start engine workflow
``quick-start-template.yml``, triggered by ``specforge-quick-start.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``UPSTREAM_REF``              Required; bash reads ``$UPSTREAM_REF`` to get
                                  the latest release ref. The value is
                                  captured into ``captures`` via the
                                  ``#test-setup store="upstream_ref"`` block's
                                  stdout, then substituted into the doc
                                  command body where ``<ref>`` appears.
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU runner:
                                  local dev machines / normal ubuntu runners
                                  have no ``/dev/davinci*`` device, and the
                                  hard run would fail on ``import torch_npu``.
"""

from __future__ import annotations

import os
import subprocess
import unittest

from doc_test.base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else (including unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``quick_start.md`` end-to-end test: fetch doc -> validate
    contract -> run ``#test-setup`` / ``#test`` in order -> compare against
    ``#test-result``.

    Scope: image pre-flight (python / CANN / torch / torch_npu / sglang)
    -> install huggingface_hub + mooncake + specforge (source, upstream
    ref injected) -> CLI self-checks -> the 5-stage smoke: Qwen3.5-4B
    (~8 GB) snapshot_download from HuggingFace Hub, SGLang spec-capture
    patches, mooncake_master + patched SGLang capture server (card 0),
    1-step specforge train (card 1) with launch + monitor split.
    """

    # 3h per command: the cold-cache Qwen3.5-4B snapshot_download
    # (~8 GB at observed ~1.2 MB/s cluster egress) is the long pole
    # (~110 min worst case); SGLang first-time graph compile adds
    # 5-10 min. The workflow job budget (240 min) caps the total.
    DEFAULT_COMMAND_TIMEOUT = 10800

    USER_AGENT = 'cosdt-ci-test/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a CANN
    # failure surfaces a full stderr dump (head/tail by default would hide
    # the line that names the failure).
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # CANN toolkit emits this typo (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list: write to /tmp and export, so
    # subprocesses (subprocess.run inherits parent env by default) see
    # it. specforge's dependency tree pulls ``accelerate`` /
    # ``transformers`` / ``huggingface-hub`` which can transitively drag
    # in CUDA wheels without this constraint.
    _CUDA_CONSTRAINTS = (
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
        'nvidia-cusparselt-cu12<0',
        'nvidia-nccl-cu12<0',
        'nvidia-nvjitlink-cu12<0',
        'nvidia-nvtx-cu12<0',
    )
    _CONSTRAINTS_FILE = '/tmp/specforge_npu_constraints.txt'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the container image pinned by the
    # ``image:`` input of ``specforge-quick-start.yml``.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + uv
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env, write the cuda/nvidia exclusion list, install uv.

        What this hook does NOT pre-install (the doc's own labeled
        blocks pull them, so a broken install block fails loudly
        instead of being masked):
            * ``torch`` / ``torch_npu`` / ``sglang``: pre-installed in
              the pinned image; the doc's ``check-torch`` block validates
              them against the version matrix.
            * ``huggingface_hub`` / ``mooncake``: the doc's install
              blocks pull them.
            * ``specforge``: the doc's ``specforge-install-source``
              block pulls it from the upstream ref injected by the
              workflow.
        """
        # 0) CANN env: source set_env.sh and merge the env stream into
        # os.environ. Don't overwrite envs explicitly injected by the
        # workflow; only fill in CANN keys that are missing.
        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                os.environ.setdefault(key, value)
            print('setup: sourced CANN env from set_env.sh')
        else:
            print(
                f'setup: skipping CANN env source ({cls._CANN_SET_ENV} not present)'
            )

        # 1) CUDA exclusion list: block cuda-toolkit / nvidia-* <0 so a
        # transitive dep can't drag a CUDA runtime onto the NPU runner.
        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['UV_CONSTRAINT'] = cls._CONSTRAINTS_FILE

        # 2) uv: the doc body's install commands all use ``uv pip
        # install``, which handles PEP 517 build deps more reliably
        # than pip. Inherits ``PIP_INDEX_URL`` + ``PIP_TRUSTED_HOST``
        # from the yml job-level env (cluster cache path +
        # trusted-host).
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + CUDA constraints + uv.

        ``torch`` / ``torch_npu`` / ``sglang`` / ``huggingface_hub`` /
        ``mooncake`` / ``specforge`` are NOT installed here — see
        ``prepare_environment`` for why.

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
