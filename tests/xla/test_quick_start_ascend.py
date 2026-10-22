"""Quick-start-Ascend documentation test: end-to-end case built on top
of the ``MarkdownDocTestBase`` contract.

Document under test: ``sources/xla/quick_start.md``
(follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters).

Run: ``python -m unittest tests.xla.test_quick_start_ascend -v 2>&1``

Environment variables (injected by GitHub workflow
``xla-quick-start.yml`` via the shared engine
``quick-start-template.yml``):
    ``MONITORED_DOC_URL``         Required; GitHub Contents API URL of the
                                  document under test (the engine's monitor
                                  hashes the same URL).
    ``UPSTREAM_REF``              Required; the doc's hidden
                                  ``#test-setup store="upstream_ref"`` block
                                  captures ``$UPSTREAM_REF`` (the upstream
                                  default-branch HEAD sha) into ``captures``,
                                  then substitutes it into the clone command
                                  where ``<ref>`` appears.
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU
                                  runner: local dev machines / normal ubuntu
                                  runners have no ``/dev/davinci*`` device.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import unittest
from typing import ClassVar

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

    Scope (single card): env check + source build of xla-npu (JDK /
    Bazel download + OpenXLA clone-freeze + bazel compile of the PJRT
    plugin + JAX install) + a minimal JAX-NPU backend smoke + the
    ``examples/ops`` operator CPU-vs-NPU comparison. Pure operator
    workload — no model weights, hence no ModelScope cache handling.

    The doc body is cwd-relative ("wherever you run it is the project
    root"): clones to ``./xla-npu``, builds there, examples run from the
    clone's ``examples/ops``. CI pins the execution cwd in
    ``prepare_environment`` via ``os.chdir('/root/xla-npu-test')``.
    """

    # The bazel build of the PJRT plugin is the dominant cost (OpenXLA
    # checkout + JDK/Bazel download + compile). 4h per command leaves
    # headroom for a cold build on the largest runner; the workflow
    # timeout_minutes (240) matches. The smoke blocks ride on this same
    # timeout (the framework applies one timeout to every block).
    DEFAULT_COMMAND_TIMEOUT = 14400

    # Monitored source is the Ascend/docs repo: the doc lives at
    # sources/xla/quick_start.md and the engine sets MONITORED_DOC_URL
    # to the raw URL for the same path.
    USER_AGENT = 'cosdt-ci-test/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a
    # CANN / XLA-NPU failure surfaces a full stderr dump (head/tail by
    # default would hide the line that names the failure).
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # CANN toolkit emits this typo (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # CANN toolkit set_env.sh candidate paths. The ascendhub CANN image
    # uses the ascend-toolkit layout; the upstream xla-npu README
    # documents the older /usr/local/Ascend/cann layout. Probe both so
    # the runtime (PJRT plugin + afir compiler) finds libascendcl etc.
    _CANN_SET_ENV_CANDIDATES = (
        '/usr/local/Ascend/ascend-toolkit/set_env.sh',
        '/usr/local/Ascend/cann/set_env.sh',
    )

    # Build tools the doc's ``build.sh`` needs at runtime. The engine's
    # "Bootstrap curl, jq, git" step only guarantees curl/jq/git, so the
    # compile-time tools are installed here if the image lacks them.
    _BUILD_TOOLS: ClassVar[dict[str, str]] = {
        'wget': 'wget',
        'xz': 'xz-utils',
        'cmake': 'cmake',
        'g++': 'g++',
        'git': 'git',
    }

    # Working directory pinned across the doc run: clone + build all
    # happen under a clean dir so the checkout tree isn't polluted.
    _WORK_DIR = '/root/xla-npu-test'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + build tools + working dir
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + install missing build tools + pin cwd.

        Everything else (JDK / Bazel / OpenXLA / JAX / the plugin .so)
        is produced by the doc's ``build.sh install`` step, so this hook
        only sets up the environment that build and runtime need.
        """
        # 0) CANN env: source set_env.sh and merge the env stream into
        # os.environ, so subprocesses (subprocess.run inherits parent
        # env) see LD_LIBRARY_PATH etc. across every doc block.
        sourced = False
        for set_env in cls._CANN_SET_ENV_CANDIDATES:
            if os.path.isfile(set_env):
                merged = subprocess.run(
                    ['bash', '-c', f'source {set_env} >/dev/null 2>&1; env'],
                    capture_output=True, text=True, check=True,
                )
                for line in merged.stdout.splitlines():
                    if '=' not in line:
                        continue
                    key, _, value = line.partition('=')
                    # Don't overwrite envs explicitly injected by the
                    # workflow; only fill in CANN keys that are missing.
                    os.environ.setdefault(key, value)
                print(f'setup: sourced CANN env from {set_env}')
                sourced = True
                break
        if not sourced:
            print(
                'setup: no CANN set_env.sh found (expected '
                f'{cls._CANN_SET_ENV_CANDIDATES}) - runtime may fail'
            )

        # 1) Build tools: install anything the image lacks (git is
        # bootstrapped by the engine, but check here too for local runs).
        missing_pkgs = [
            pkg for tool, pkg in cls._BUILD_TOOLS.items()
            if shutil.which(tool) is None
        ]
        if missing_pkgs:
            print(f'setup: installing build tools: {missing_pkgs}')
            subprocess.run(['apt-get', 'update', '-qq'], check=True)
            subprocess.run(
                ['apt-get', 'install', '-y', '-qq', *missing_pkgs],
                check=True,
            )

        # 2) Working dir: the doc's clone/build are cwd-relative, so pin
        # a clean directory before run_template() executes.
        os.makedirs(cls._WORK_DIR, exist_ok=True)
        os.chdir(cls._WORK_DIR)
        print(f'setup: working dir pinned to {cls._WORK_DIR}')

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class.

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
