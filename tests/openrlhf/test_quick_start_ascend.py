"""Quick-start-Ascend test: doc under test is ``sources/openrlhf/quick_start.md``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

from doc_test.base import MarkdownDocTestBase
from doc_test.model_cache import (
    diagnose_mount_environment,
    ensure_safetensors,
    purge_huggingface_corrupt,
    report_huggingface_state,
    resolve_huggingface_cache,
)


def _is_truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 3600
    USER_AGENT = 'ascend-docs/quick-start'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # typo in CANN's Python driver (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    _MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    @classmethod
    def prepare_environment(cls) -> None:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')

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
        # Do not realpath(): venv `python` is a symlink to the system
        # interpreter, and resolving it would put system pip first.
        venv_bin = os.path.dirname(sys.executable)
        os.environ['PATH'] = f'{venv_bin}:{path_dirs}:{os.environ.get("PATH", "")}'

        ensure_safetensors()
        diagnose_mount_environment(model_id=cls._MODEL_ID)
        report_huggingface_state(cls._MODEL_ID)
        purge_huggingface_corrupt(resolve_huggingface_cache())

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per class. ``@unittest.skipIf`` only skips
        the test *method* — ``setUpClass`` itself always runs, so the
        ``if _e2e_enabled()`` guard keeps heavy setup from firing on
        non-NPU runners.
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
