"""Quick-start test: doc under test is ``sources/llama_cpp/quick_start.md``.

Run: ``python -m unittest tests.llama_cpp.test_quick_start_ascend -v 2>&1``

Env (injected by the engine ``quick-start-template.yml``, triggered by
``llama_cpp-quick-start.yml``): ``MONITORED_DOC_URL``, ``UPSTREAM_REF``,
``NPU_READY=true`` (otherwise the class is skipped).
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
    """End-to-end test: fetch doc -> validate contract -> run ``#test-setup``
    / ``#test`` in order -> compare against ``#test-result``."""

    # Full cmake + CANN compile can take over an hour; the base class
    # uses one timeout for every subprocess.
    DEFAULT_COMMAND_TIMEOUT = 7200
    USER_AGENT = 'cosdt-ci-test/quick-start'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # typo in CANN's Python driver (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env once so later ``bash -c`` blocks inherit it.

        Class-level setup: run once per test class, triggered by
        ``setUpClass``. Each labeled fence is a new subprocess, so a
        ``source set_env.sh`` block in the document does not persist.

        Merge is overwrite, not ``setdefault``: the container image may
        already ship ``LD_LIBRARY_PATH``, which would otherwise hide the
        CANN increment from ``set_env.sh``.
        """
        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                os.environ[key] = value
            print('setup: sourced CANN env from set_env.sh')
        else:
            print(
                f'setup: skipping CANN env source ({cls._CANN_SET_ENV} not present)'
            )

        path_dirs = '/usr/local/sbin:/usr/local/bin'
        current_path = os.environ.get('PATH', '')
        if path_dirs not in current_path:
            os.environ['PATH'] = f'{path_dirs}:{current_path}'

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per class. ``@unittest.skipIf`` only skips
        the test *method* — ``setUpClass`` itself always runs, so the
        ``if _e2e_enabled()`` guard keeps heavy setup from firing when
        ``NPU_READY`` is unset.
        """
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        """Run the full pre_process -> parse -> execute -> post_process flow."""

        self.run_template()


if __name__ == '__main__':
    unittest.main()
