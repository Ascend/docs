"""Quick-start-Ascend documentation test for DeepSpeed on Ascend NPU.

Built on top of ``MarkdownDocTestBase`` (shared engine). The document
under test is ``projects/deepspeed/docs/Quick-start-Ascend.md``, which
follows the ``docs/markdown_doc_test_label.md`` contract: every
``shell`` code block carries one of the ``#test`` / ``#test-setup`` /
``#test-result`` labels plus ``id=`` / ``store=`` / ``load='x>>y'`` /
``fuzzy='xxx'`` parameters.

Environment variables (injected by the shared engine
``quick-start-template.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``UPSTREAM_REF``              Required; captured by the hidden
                                  ``#test-setup store="upstream_ref"`` block
                                  in the doc, then loaded into test commands
                                  where ``<UPSTREAM_REF>`` appears.
    ``NPU_READY=true``            Required; gates the E2E test class.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from doc_test.base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set."""
    return _is_truthy(os.environ.get('NPU_READY'))


def _ensure_torch_npu():
    """Ensure torch + torch_npu are importable and match 2.9.0.

    Probe first: the CANN 9.1.0 image ships ``torch==2.9.0+cpu`` +
    ``torch_npu==2.9.0.post2``; when present, reuse them. Only when the
    probe fails (missing or version mismatch) reinstall from the
    cluster cache + Huawei Ascend dual source so torch and torch_npu
    come from the same compatible build.
    """
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
    # Verify ABI after install.
    subprocess.run([sys.executable, '-c', _PROBE_SCRIPT], check=True)
    print('setup: installed torch==2.9.0 torch_npu==2.9.0.post2')


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``Quick-start-Ascend.md`` end-to-end test: fetch doc -> validate
    contract -> run ``#test-setup`` / ``#test`` in order -> compare against
    ``#test-result``."""

    DEFAULT_COMMAND_TIMEOUT = 1800
    USER_AGENT = 'cosdt-ci-test/quick-start'
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    def pre_process(self) -> str:
        """Read the local doc instead of fetching from MONITORED_DOC_URL.
        The doc is bundled in the repo, so the test always uses the version
        that ships with the code.
        """
        doc = Path(__file__).resolve().parent.parent / 'docs' / 'Quick-start-Ascend.md'
        return doc.read_text(encoding='utf-8')

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env once so later ``bash -c`` blocks inherit it.

        Class-level setup: run once per test class, triggered by
        ``setUpClass``. Each labeled fence is a new subprocess, so a
        ``source set_env.sh`` block in the document does not persist.

        Also pins NPU cards 0-1 (2-card runner; the doc's 2-card
        distributed run needs the launcher to see both devices), chdirs
        to the project root (``projects/deepspeed/``) so doc relative
        paths resolve correctly, and installs CI dependencies (MPI;
        torchvision is pinned in ``setUpClass`` after the torch stack
        is confirmed) that the user would otherwise have to handle.
        """
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

        # ASCEND_RT_VISIBLE_DEVICES=0,1: expose both cards for the
        # single-card + 2-card distributed smoke (--num_gpus 2 needs
        # the launcher to see 2 devices).
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0,1'
        print('setup: pinned ASCEND_RT_VISIBLE_DEVICES=0,1')

        # Chdir to project root so doc relative paths resolve from
        # projects/deepspeed/ (the parent of tests/).
        project_root = Path(__file__).resolve().parent.parent
        os.chdir(project_root)
        print(f'setup: cwd -> {project_root}')

        # Install MPI (libopenmpi-dev + mpi4py) for deepspeed command.
        subprocess.run(['apt-get', 'update'], check=True)
        subprocess.run(['apt-get', 'install', '-y', 'libopenmpi-dev'], check=True)
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'mpi4py'],
            check=True,
        )
        print('setup: installed MPI (libopenmpi-dev + mpi4py)')

        # CIFAR10 download is handled inside the doc's train_cifar10.py
        # (CN mirror fast path + torchvision official fallback).

        # torchvision (and its runtime deps pillow/numpy) is installed by
        # the doc's install-torchvision block, pinned to match torch 2.9.0.

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