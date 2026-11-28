"""Run the Liger Kernel quick start against one Ascend NPU."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from doc_test.base import MarkdownDocTestBase, TestCommand
from doc_test.model_cache import (
    ensure_safetensors,
    purge_modelscope_corrupt,
    resolve_modelscope_cache,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'
_CUDA_CONSTRAINTS = (
    'cuda-toolkit<0', 'cuda-python<0', 'cuda-bindings<0',
    'nvidia-cublas<0', 'nvidia-cuda-runtime<0', 'nvidia-cudnn<0',
    'nvidia-nccl<0', 'nvidia-cublas-cu12<0',
    'nvidia-cuda-runtime-cu12<0', 'nvidia-cudnn-cu12<0',
    'nvidia-nccl-cu12<0',
)


def _check_release_version(installed_version: str, upstream_ref: str) -> None:
    if not upstream_ref.startswith('v'):
        raise AssertionError(f'expected an upstream release tag, got {upstream_ref!r}')
    if installed_version != upstream_ref[1:]:
        raise AssertionError(
            f'installed liger-kernel {installed_version} does not match '
            f'engine release {upstream_ref}; PyPI may be behind the release'
        )


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    USER_AGENT = 'Ascend/docs/liger-kernel-quick-start'
    DEFAULT_COMMAND_TIMEOUT = 3600

    def _run_one(self, cmd, results, env, cwd, timeout, idx):
        super()._run_one(cmd, results, env, cwd, timeout, idx)
        if isinstance(cmd, TestCommand) and cmd.id == 'install-ascend-deps':
            # Importing the backend alone does not establish that an NPU is usable.
            probe_code = (
                'import torch, torch_npu; '
                'from triton.backends import backends; '
                "assert 'ascend' in backends, 'Ascend Triton backend not registered'; "
                "assert torch.npu.is_available(), 'NPU unavailable'; "
                "assert torch.npu.device_count() > 0, 'No visible NPU'"
            )
            probe = subprocess.run(
                [sys.executable, '-c', probe_code],
                capture_output=True, text=True, env=env, cwd=cwd, timeout=120,
            )
            if probe.returncode:
                raise AssertionError(
                    f'Ascend runtime check failed: {probe_code}\n'
                    f'stdout:\n{probe.stdout}\nstderr:\n{probe.stderr}'
                )
            self.log('Ascend Triton backend registered and NPU available')
        if isinstance(cmd, TestCommand) and cmd.id == 'install-liger':
            upstream_ref = env.get('UPSTREAM_REF')
            if not upstream_ref:
                raise RuntimeError('UPSTREAM_REF is required for version alignment')
            installed_version = subprocess.run(
                [sys.executable, '-c',
                 'from importlib.metadata import version; '
                 "print(version('liger-kernel'))"],
                capture_output=True, text=True, check=True, env=env, cwd=cwd,
            ).stdout.strip()
            _check_release_version(installed_version, upstream_ref)
            self.log(f'liger-kernel version matches engine release {upstream_ref}')

    @classmethod
    def prepare_environment(cls) -> None:
        if not Path(_CANN_SET_ENV).is_file():
            raise RuntimeError(f'CANN environment script missing: {_CANN_SET_ENV}')
        sourced = subprocess.run(
            ['bash', '-c', f'source {_CANN_SET_ENV} >/dev/null 2>&1 && env -0'],
            capture_output=True, check=True,
        )
        for entry in sourced.stdout.split(b'\0'):
            if b'=' in entry:
                key, value = entry.split(b'=', 1)
                os.environ[os.fsdecode(key)] = os.fsdecode(value)

        work_dir = Path(tempfile.mkdtemp(prefix='liger-kernel-quick-start-'))
        cls._work_dir = work_dir
        constraints = work_dir / 'pip-constraints.txt'
        constraints.write_text('\n'.join(_CUDA_CONSTRAINTS) + '\n', encoding='utf-8')
        os.environ['PIP_CONSTRAINT'] = str(constraints)
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'

        probe = subprocess.run(
            [sys.executable, '-c',
             'import torch, torch_npu; '
             "assert torch.__version__.startswith('2.9.0'); "
             "assert torch_npu.__version__.startswith('2.9.0')"],
            capture_output=True, text=True,
        )
        if probe.returncode:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install',
                 '--extra-index-url', 'https://repo.huaweicloud.com/ascend/repos/pypi',
                 'torch==2.9.0', 'torch_npu==2.9.0.post2'],
                check=True,
            )

        # Triton-Ascend 3.2.2 requires triton 3.5.0 on this aarch64 runner.
        # Keep that dependency; the document imports the Ascend backend explicitly.
        ensure_safetensors()
        purge_modelscope_corrupt(resolve_modelscope_cache())
        os.chdir(work_dir)

    @classmethod
    def setUpClass(cls) -> None:
        if os.environ.get('NPU_READY', '').lower() == 'true':
            cls.prepare_environment()

    def post_process(self) -> None:
        os.chdir(_REPO_ROOT)
        shutil.rmtree(self._work_dir, ignore_errors=True)

    @unittest.skipUnless(
        os.environ.get('NPU_READY', '').lower() == 'true',
        'end-to-end requires an Ascend NPU runner',
    )
    def test_runs_doc(self) -> None:
        self.run_template()


class TestVersionAlignment(unittest.TestCase):
    def test_matching_release(self) -> None:
        _check_release_version('0.8.3', 'v0.8.3')

    def test_mismatched_release(self) -> None:
        with self.assertRaises(AssertionError):
            _check_release_version('0.8.3', 'v0.8.4')


if __name__ == '__main__':
    unittest.main()
