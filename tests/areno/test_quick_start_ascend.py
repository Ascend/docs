"""Quick-start-Ascend doc test: end-to-end case built on the ``MarkdownDocTestBase`` contract.

Document under test: ``sources/areno/quick_start.md``
(follows the label contract implemented by ``tests/doc_test/base.py``:
every ``shell`` / ``python`` code block carries one of the ``#test`` /
``#test-setup`` / ``#test-result`` labels, plus ``id=`` / ``store=`` /
``load='x>>y'`` / ``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.areno.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the quick-start engine workflow
``quick-start-template.yml``, triggered by ``areno-quick-start.yml``):
    ``MONITORED_DOC_URL``         required; URL of the document under test.
    ``UPSTREAM_REF``              required; latest release tag. Its value is
                                  captured into ``captures`` from the stdout
                                  of the ``#test-setup store="upstream_ref"``
                                  block, then substituted into ``<ref>`` in
                                  the doc.
    ``NPU_READY=true``            required, otherwise the whole class is
                                  skipped. End-to-end runs only on NPU
                                  runners: local/plain ubuntu runners have
                                  no ``/dev/davinci*`` and would fail hard
                                  at ``import torch_npu``.
"""

from __future__ import annotations

import os
import subprocess
import unittest

from doc_test.base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    """``'true'`` (case-insensitive) -> True; anything else (unset included) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true``, lifting the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``sources/areno/quick_start.md`` end-to-end test: fetch the doc ->
    validate the contract -> run the ``#test-setup`` / ``#test`` blocks in
    order -> compare against ``#test-result``.

    Scope: NPU stack check (torch / torch_npu / device count) + source
    install of the engine-resolved latest release (the installer detects
    ``torch_npu``, takes the ``requirements/npu.txt`` branch and compiles
    ``areno.accel._areno_accel_npu`` with no fallback, so the build must
    succeed on the CANN 9.0.0 image) + CLI entry smoke + a 1-step
    GSPO-on-GSM8K RLVR smoke (Qwen3-0.6B via HuggingFace Hub: rollout +
    reward + train + checkpoint save) + OpenAI-compatible serving of that
    checkpoint (background start, /health polling, /v1/models and
    /v1/chat/completions checks, teardown).
    """

    # 60 minutes: the AscendC kernel + torch_npu extension build inside
    # ``uv pip install -e . --no-build-isolation`` dominates; the 1-step GSPO
    # smoke afterwards (0.6B, 8 x 256-token rollouts) adds a few more minutes
    # after the cold model/dataset download.
    DEFAULT_COMMAND_TIMEOUT = 3600
    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source is a fork under cosdt-ci-test
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # CANN driver spelling (sic)
        'ERR99999',  # CANN unrecoverable-error sentinel
    )

    # CANN toolkit: sourcing this once yields ASCEND_HOME / LD_LIBRARY_PATH
    # etc. Hard-coded path, matching the image bound to the ``image:`` input
    # of ``areno-quick-start.yml``.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + uv (torch / torch_npu / AReno installs
    # are the doc's `#test-setup` / `#test` blocks' job)
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source the CANN environment and install uv.

        ``torch / torch_npu`` and ``areno`` are handled by the doc's
        ``#test-setup`` / ``#test`` blocks; this only does the runner-side
        scaffolding that must happen before those blocks.

        Class-level preparation, triggered once by ``setUpClass`` (not the
        per-test-method ``unittest.TestCase.setUp``).
        """
        # 0) CANN env: source set_env.sh and merge it into os.environ.
        # Workflow-level jobs.env / steps.env take precedence over the CANN
        # defaults. Every doc-block subprocess inherits this environment, so
        # the AscendC build inside ``uv pip install -e .`` sees
        # ASCEND_HOME_PATH even before the doc's own source line.
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

        # 1) uv: the doc's install blocks use ``uv pip install -e .``, which
        # handles PEP 517 build deps (and flash-linear-attention's git
        # direct reference) more reliably than pip. Inherits
        # ``PIP_INDEX_URL`` + ``PIP_TRUSTED_HOST`` from the yml job-level env
        # (cluster cache + trusted host).
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

    # ----------------------------------------------------------
    # Test entry points
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run the environment preparation once per class: CANN env + uv.

        ``areno`` is NOT installed here - see ``prepare_environment`` for why.

        ``@unittest.skipIf`` only skips test *methods*; ``setUpClass`` itself
        always runs. The ``if _e2e_enabled()`` below is what actually keeps
        the heavy preparation from running when ``NPU_READY`` is unset.
        """
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        """Template-method entry point. The base class ``run_template()`` runs
        the full ``pre_process`` -> ``parse`` -> ``execute`` ->
        ``post_process``; ``prepare_environment`` is triggered once by
        ``setUpClass``, not inside ``run_template``."""

        self.run_template()


if __name__ == '__main__':
    unittest.main()
