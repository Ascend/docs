"""Quick-start test: doc under test is ``sources/minionerec/quick_start.md``.

End-to-end case built on top of the ``MarkdownDocTestBase`` contract:
clone ``AkaliKong/MiniOneRec`` -> install the NPU torch stack and the
SFT / evaluation / RL dependencies -> verify the upstream ``npu_support``
shim maps ``torch.cuda`` onto NPU -> download Qwen2.5-0.5B and the
preprocessed Amazon Industrial_and_Scientific subset from HuggingFace ->
one SFT epoch -> constrained-decoding evaluation (CC must be 0) -> one
GRPO epoch.

Upstream PR #85 adds ``npu_support.py`` and imports it first in every
entry script, so no env-level injection or source edit is needed;
``rl.py`` picks ``adamw_torch`` on NPU and ``--sample 40`` caps every
dataset. The doc's ``git status --porcelain`` check asserts the checkout
stays clean.

At this scale every reward stays 0: ``rule_reward`` is exact match, one
SFT epoch never hits, and ``ndcg_rule_reward`` zeroes the whole group
when nothing hits - so the group advantage and the gradient are 0 too.
The RL stage therefore guards flow-through only: ``run-rl`` must exit 0
and ``rl.py``'s closing ``save_model`` / ``model.save_pretrained`` must
leave weights, config and tokenizer under ``rl_out/final_checkpoint``.

Run: ``python -m unittest tests.minionerec.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the engine
``quick-start-template.yml``, triggered by ``minionerec-quick-start.yml``):
    ``MONITORED_DOC_URL``         doc to fetch and execute
    ``UPSTREAM_REF``              ref substituted into the doc's ``<ref>``
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

from doc_test.base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else (unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``sources/minionerec/quick_start.md`` end-to-end test: fetch doc ->
    validate contract -> run ``#test-setup`` / ``#test`` in order ->
    compare against ``#test-result``.

    Working directory is pinned to ``/root/minionerec-test``: the doc is
    cwd-relative (clones to ``./MiniOneRec``, downloads into
    ``./MiniOneRec/data``), so the clone tree must not collide with the
    ``workflows`` checkout the engine runs the test from.
    """

    # SFT (~11 min) and RL (~14 min) dominate; 1h per command leaves
    # headroom for a cold HuggingFace download.
    DEFAULT_COMMAND_TIMEOUT = 3600

    USER_AGENT = 'ascend-docs/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a
    # CANN runtime failure surfaces a full stderr dump instead of a
    # head/tail window that hides the failing line.
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # CANN toolkit emits this typo (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    _CANN_SET_ENV_CANDIDATES = (
        '/usr/local/Ascend/ascend-toolkit/set_env.sh',
        '/usr/local/Ascend/cann/set_env.sh',
    )

    _WORK_DIR = '/root/minionerec-test'

    # ----------------------------------------------------------
    # prepare_environment: PATH + CANN env + cwd
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """CANN env, ``git`` and the working directory.

        Everything business-facing is in the document: the dependency
        install, the shim, the model and data download, SFT, evaluation
        and RL are the steps a reader follows.
        """
        # 0) PATH: /usr/local/sbin for npu-smi.
        current_path = os.environ.get('PATH', '')
        prefix_dirs = '/usr/local/sbin:/usr/local/bin'
        if not current_path.startswith(prefix_dirs):
            os.environ['PATH'] = f'{prefix_dirs}:{current_path}'

        # 1) CANN env: source set_env.sh and merge the env stream into
        # os.environ so every doc block's subprocess inherits
        # LD_LIBRARY_PATH / ASCEND_* without re-sourcing.
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
                    # Don't overwrite what the workflow injected.
                    os.environ.setdefault(key, value)
                print(f'setup: sourced CANN env from {set_env}')
                sourced = True
                break
        if not sourced:
            print(
                'setup: no CANN set_env.sh found (expected '
                f'{cls._CANN_SET_ENV_CANDIDATES}) - every step will fail'
            )

        # 2) git: the engine bootstraps it, but a missing git turns the
        # clone into an opaque `command not found`.
        if shutil.which('git') is None:
            print('setup: installing git')
            subprocess.run(['apt-get', 'update', '-qq'], check=True)
            subprocess.run(['apt-get', 'install', '-y', '-qq', 'git'], check=True)

        # 3) Working directory: pinned before run_template() so the doc's
        # cwd-relative clone never lands in the guard checkout.
        os.makedirs(cls._WORK_DIR, exist_ok=True)
        os.chdir(cls._WORK_DIR)
        print(f'setup: working dir {cls._WORK_DIR}')

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
