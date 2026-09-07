"""Probe: official pip install of vllm + vllm-ascend on the Hong Kong A2 runner.

Doc under test is ``sources/vllm_install_probe/quick_start.md``.
"""

from __future__ import annotations

import os
import subprocess
import unittest

from doc_test.base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 3600
    USER_AGENT = 'ascend-docs/vllm-install-probe'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',
        'ERR99999',
    )

    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'
    _NNAL_SET_ENV = '/usr/local/Ascend/nnal/atb/set_env.sh'

    @classmethod
    def _merge_sourced_env(cls, script: str) -> None:
        merged = subprocess.run(
            ['bash', '-c', f'source {script} >/dev/null 2>&1; env'],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in merged.stdout.splitlines():
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            os.environ[key] = value

    @classmethod
    def prepare_environment(cls) -> None:
        if os.path.isfile(cls._CANN_SET_ENV):
            cls._merge_sourced_env(cls._CANN_SET_ENV)
            print(f'setup: sourced {cls._CANN_SET_ENV}')
        else:
            print(f'setup: missing {cls._CANN_SET_ENV}')
        if os.path.isfile(cls._NNAL_SET_ENV):
            cls._merge_sourced_env(cls._NNAL_SET_ENV)
            print(f'setup: sourced {cls._NNAL_SET_ENV}')

        path_dirs = '/usr/local/sbin:/usr/local/bin'
        current_path = os.environ.get('PATH', '')
        if path_dirs not in current_path:
            os.environ['PATH'] = f'{path_dirs}:{current_path}'

    @classmethod
    def setUpClass(cls) -> None:
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        self.run_template()


if __name__ == '__main__':
    unittest.main()
