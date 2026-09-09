"""DeepSpeed quick-start documentation test.
Doc under test: sources/deepspeed/quick_start.md.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from doc_test.base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    return _is_truthy(os.environ.get('NPU_READY'))


def _ensure_torch_npu():
    """Reuse the image's torch stack when torch/torch_npu match 2.9.0;
    reinstall from cluster + Ascend index only on probe failure."""
    _PROBE_SCRIPT = (
        'import torch, torch_npu\n'
        "raise SystemExit(0 if "
        "torch.__version__.startswith('2.9.0') "
        "and torch_npu.__version__.startswith('2.9.0') "
        "else 1)"
    )
    probe = subprocess.run(
        [sys.executable, '-c', _PROBE_SCRIPT],
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0:
        versions = subprocess.run(
            [sys.executable, '-c',
             'import torch, torch_npu; print(torch.__version__, torch_npu.__version__)'],
            capture_output=True, text=True, check=True,
        )
        print(f'setup: reusing image torch stack ({versions.stdout.strip()})')
        return
    print('setup: installing torch==2.9.0 torch_npu==2.9.0.post2')
    subprocess.run(
        [
            sys.executable, '-m', 'pip', 'install',
            '--index-url', 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple',
            '--extra-index-url', 'https://repo.huaweicloud.com/ascend/repos/pypi',
            'torch==2.9.0', 'torch_npu==2.9.0.post2',
        ],
        check=True,
    )
    subprocess.run([sys.executable, '-c', _PROBE_SCRIPT], check=True)
    print('setup: installed torch==2.9.0 torch_npu==2.9.0.post2')


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 1800
    USER_AGENT = 'cosdt-ci-test/quick-start'
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    def pre_process(self) -> str:
        """Use the repo-bundled doc instead of MONITORED_DOC_URL so the
        test always runs the version that ships with the code."""
        doc = Path(__file__).resolve().parent.parent.parent / 'sources' / 'deepspeed' / 'quick_start.md'
        return doc.read_text(encoding='utf-8')

    @classmethod
    def prepare_environment(cls) -> None:
        """One-time CI prep: CANN env (fenced blocks are fresh subprocesses,
        so it must be injected here), 2-card visibility, cwd, MPI."""
        path_dirs = '/usr/local/sbin:/usr/local/bin'
        current_path = os.environ.get('PATH', '')
        if path_dirs not in current_path:
            os.environ['PATH'] = f'{path_dirs}:{current_path}'

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

        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0,1'
        print('setup: pinned ASCEND_RT_VISIBLE_DEVICES=0,1')

        project_root = Path(__file__).resolve().parent.parent.parent / 'sources' / 'deepspeed'
        os.chdir(str(project_root))
        print(f'setup: cwd -> {project_root}')

        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 'libopenmpi-dev'], check=True)
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'mpi4py'],
            check=True,
        )
        print('setup: installed MPI (libopenmpi-dev + mpi4py)')

    @classmethod
    def setUpClass(cls) -> None:
        if _e2e_enabled():
            cls.prepare_environment()
            _ensure_torch_npu()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end tests require NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        self.run_template()


if __name__ == '__main__':
    unittest.main()
