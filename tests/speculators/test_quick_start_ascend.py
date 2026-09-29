"""Quick-start test: doc under test is ``sources/speculators/quick_start.md``.

End-to-end case built on top of the ``MarkdownDocTestBase`` contract:
every ``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters.

Run: ``python -m unittest tests.speculators.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the quick-start engine workflow
``quick-start-template.yml``, triggered by ``speculators-quick-start.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``UPSTREAM_REF``              Required; bash reads ``$UPSTREAM_REF`` to get
                                  the latest release tag. The value is
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
from doc_test.model_cache import (
    ensure_safetensors,
    purge_modelscope_corrupt,
    resolve_modelscope_cache,
)


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

    Scope: image pre-flight (vllm-ascend stack version probe) ->
    modelscope install -> speculators source install -> the 4-step
    pipeline: DFlash convert, hidden-states extraction (vllm offline
    generate) + torchrun 1-epoch draft training, vllm-ascend serve with
    the trained draft + chat completion smoke, config import check.
    """

    # 2h per command: the cold-cache verifier download (Qwen/Qwen3-8B,
    # 5 shards ~16.4 GB at observed ~3.1 MB/s per shard ≈ 90 min) is the
    # long pole; the draft (~1 GB) pulls in ~84 s. The runner's
    # NFS-backed modelscope cache makes both one-time (hot runs skip
    # straight to the pipeline). Short enough that the workflow budget
    # (240 min) still caps a full cold run end-to-end.
    DEFAULT_COMMAND_TIMEOUT = 7200

    USER_AGENT = 'cosdt-ci-test/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a CANN
    # failure surfaces a full stderr dump (head/tail by default would hide
    # the line that names the failure).
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # CANN toolkit emits this typo (sic) in its Python driver
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the container image pinned by the
    # ``image:`` input of ``speculators-quick-start.yml``.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + uv + safetensors + modelscope cache purge
    #  (transformers / speculators are installed by the doc's
    #  `### 前置安装` / `## 安装 Speculators` blocks)
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + install uv + safetensors
        + purge stale modelscope cache shards.

        The doc's ``### 前置安装`` and ``## 安装 Speculators`` sections are
        the single source of truth for which packages + versions get
        installed — the vllm-ascend stack ships in the pinned image,
        ``modelscope`` installs via ``install-deps`` and ``speculators``
        via ``speculators-install-source``, all in document order via the
        ``#test`` machinery.

        ModelScope cache (``$MODELSCOPE_CACHE`` or its default
        ``~/.cache/modelscope``) is left at its default — the runner's
        NFS-backed ``/root/.cache/modelscope`` persists across runs.
        Same pattern as peft / diffusers.
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

        # 1) uv: the doc's ``speculators-install-source`` block calls
        # ``uv pip install -e .`` which handles PEP 517 build deps more
        # reliably than pip. Inherits ``PIP_INDEX_URL`` + ``PIP_TRUSTED_HOST``
        # from the yml job-level env (cluster cache path + trusted-host).
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 2) safetensors: speculators reads model weights via
        # safetensors. It's a base dep of speculators (its pyproject
        # lists ``safetensors`` as required), so it ships in once
        # ``speculators-install-source`` runs in the doc; this defensive
        # install catches the case where the base image lacks it before
        # any speculators code touches weight files.
        ensure_safetensors()

        # 3) Cache validation: the persistent NFS-backed cache can hold
        # truncated safetensors from interrupted runs. Walk every shard
        # under each model dir and purge it on failure; modelscope will
        # re-download cleanly on next access. Implementation lives in
        # doc_test.model_cache.
        purge_modelscope_corrupt(resolve_modelscope_cache())

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class: CANN env + uv + safetensors
        + modelscope cache purge.

        ``modelscope`` / ``speculators`` are NOT installed here: the doc's
        own labeled blocks install them in document order, so a broken
        install block fails loudly instead of being masked by a
        pre-installed copy.

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
