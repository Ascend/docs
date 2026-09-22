"""Quick-start-Ascend test: doc under test is ``sources/roll/quick_start.md``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from doc_test.base import MarkdownDocTestBase, TestCommand
from doc_test.model_cache import (
    ensure_safetensors,
    purge_modelscope_corrupt,
    resolve_modelscope_cache,
)


def _is_truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 1800
    USER_AGENT = 'cosdt-ci-test/quick-start'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',
        'ERR99999',
    )

    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'
    _TENSORBOARD_DIR = Path(
        'ROLL/output/tensorboard/roll-quick-start-npu'
    )

    def _verify_tensorboard_events(self) -> None:
        """CI-side guard: the doc only reports success and the metrics path.

        A missing or empty event file means the run produced no metrics; the
        doc used to assert that inline, which reads like a test script rather
        than usage. Keeping the check here preserves the coverage.
        """
        events = list(self._TENSORBOARD_DIR.glob('*/events.out.tfevents.*'))
        if not events:
            raise AssertionError(
                'no tensorboard event file found under '
                f'{self._TENSORBOARD_DIR}'
            )
        empty = [p for p in events if p.stat().st_size == 0]
        if empty:
            raise AssertionError(
                'tensorboard event file is empty: '
                + ', '.join(str(p) for p in empty)
            )
        self.log(
            '[Step] verified tensorboard events '
            f'({len(events)} file(s), first={events[0].stat().st_size}B): '
            f'{self._TENSORBOARD_DIR}'
        )

    def _run_one(self, cmd, results, env, cwd, timeout, idx):
        if (
            isinstance(cmd, TestCommand)
            and getattr(cmd, 'id', None) == 'verify-output'
        ):
            super()._run_one(cmd, results, env, cwd, timeout, idx)
            self._verify_tensorboard_events()
            return
        return super()._run_one(cmd, results, env, cwd, timeout, idx)

    def pre_process(self) -> str:
        doc = (
            Path(__file__).resolve().parents[2]
            / 'sources' / 'roll' / 'quick_start.md'
        )
        return doc.read_text(encoding='utf-8')

    @classmethod
    def prepare_environment(cls) -> None:
        os.environ['PYTHONNOUSERSITE'] = '1'
        os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')

        if not os.path.isfile(cls._CANN_SET_ENV):
            raise RuntimeError(
                f'CANN environment script is missing: {cls._CANN_SET_ENV}'
            )

        merged = subprocess.run(
            ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in merged.stdout.splitlines():
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            os.environ[key] = value
        path_dirs = '/usr/local/sbin:/usr/local/bin'
        venv_bin = os.path.dirname(sys.executable)
        os.environ['PATH'] = f'{venv_bin}:{path_dirs}:{os.environ.get("PATH", "")}'
        print('setup: sourced CANN environment')

        project_root = Path(__file__).resolve().parents[2] / 'sources' / 'roll'
        os.chdir(str(project_root))
        print(f'setup: cwd -> {project_root}')

        ensure_safetensors()
        purge_modelscope_corrupt(resolve_modelscope_cache())

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
