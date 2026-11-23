"""Guard sources/roll/quick_start.md on an NPU runner."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import ClassVar

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
    # Stop at the first failed command in a document shell block.
    _LANG_RUNNER: ClassVar[dict[str, tuple[str, ...]]] = {
        **MarkdownDocTestBase._LANG_RUNNER,
        'shell': ('bash', '-ec'),
    }
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',
        'ERR99999',
    )

    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'
    _TENSORBOARD_DIR = Path(
        'ROLL/output/agentic_frozen_lake_npu'
    )

    def run_command(self, cmd, env, cwd, timeout, language='shell'):
        if language == 'shell' and '-e ./vllm-ascend' in cmd:
            # Cloning submodules and compiling Ascend operators can exceed 30 minutes.
            timeout = max(timeout, 3600)
        if language != 'shell' or 'examples/start_agentic_pipeline.py' not in cmd:
            return super().run_command(cmd, env, cwd, timeout, language)

        # Filesystem timestamps can lag the wall clock slightly.
        log_cutoff = time.time() - 1
        try:
            rc, out, err = super().run_command(cmd, env, cwd, timeout, language)
        except subprocess.TimeoutExpired as exc:
            self._log_training_failure(
                (exc.stdout or b'').decode('utf-8', errors='replace'),
                (exc.stderr or b'').decode('utf-8', errors='replace'),
                env, log_cutoff,
            )
            raise
        if rc != 0:
            self._log_training_failure(out, err, env, log_cutoff)
        return rc, out, err

    def _log_training_failure(self, out, err, env, log_cutoff):
        for stream, content in (('stdout', out), ('stderr', err)):
            if content:
                self.log(f'ROLL training {stream} (full):\n{content.rstrip()}')

        tmp = Path(env.get('RAY_TMPDIR') or env.get('TMPDIR') or '/tmp')
        log_dir = tmp / 'ray/session_latest/logs'
        found = False
        for path in sorted(log_dir.glob('worker-*.err')):
            try:
                stat = path.stat()
                if stat.st_mtime < log_cutoff or not stat.st_size:
                    continue
                with path.open('rb') as log:
                    head = log.read(4096)
                    log.seek(max(0, stat.st_size - 16384))
                    tail = log.read(16384)
            except OSError as exc:
                self.log(f'Could not read Ray worker log {path.name}: {exc}')
                continue
            if any(marker in head or marker in tail for marker in (
                b'InferWorker', b'EngineCore', b'WorkerProc',
            )):
                found = True
                content = tail.decode('utf-8', errors='replace').rstrip()
                self.log(f'Ray worker stderr ({path.name}, tail):\n{content}')
        if not found:
            self.log(f'No matching Ray worker stderr found under {log_dir}')

    def _run_one(self, cmd, results, env, cwd, timeout, idx):
        super()._run_one(cmd, results, env, cwd, timeout, idx)
        if not isinstance(cmd, TestCommand) or cmd.id != 'verify-example-deps':
            return

        # Read versions before importing the extension so ABI failures also report them.
        probe = """
from importlib.metadata import version

for package in ('torch', 'torch-npu', 'torchvision', 'torchaudio', 'tensordict', 'numpy', 'pyarrow'):
    print(f'{package}: {version(package)}', flush=True)

import pyarrow
import torch
import torch_npu

device_count = torch.npu.device_count() if torch.npu.is_available() else 0
print(f'Visible NPUs: {device_count}', flush=True)
if device_count < 8:
    raise RuntimeError(f'ROLL FrozenLake requires eight visible NPUs, found {device_count}')
"""
        rc, out, err = self.run_command(probe, env, cwd, 60, language='python')
        if rc != 0:
            raise AssertionError(f'ROLL runtime check failed (rc={rc}):\n{out}{err}')
        self.log_block('ROLL runtime', out.splitlines())

    def _verify_tensorboard_events(self) -> None:
        """Require a nonempty TensorBoard event file after training."""
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

    def pre_process(self) -> str:
        doc_path = (
            Path(__file__).resolve().parents[2] / 'sources' / 'roll' / 'quick_start.md'
        )
        if not doc_path.is_file():
            raise RuntimeError(
                f'doc not found in local checkout: {doc_path}'
            )
        return doc_path.read_text(encoding='utf-8')

    @classmethod
    def prepare_environment(cls) -> None:
        """CANN env merge, PYTHONNOUSERSITE, PATH prefix, and
        model-cache sanity. Dependency installs live in the doc itself
        (that is the user path); only CI-side hygiene happens here."""
        os.environ['PYTHONNOUSERSITE'] = '1'

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
        os.chdir(project_root)
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
        self._verify_tensorboard_events()


if __name__ == '__main__':
    unittest.main()
