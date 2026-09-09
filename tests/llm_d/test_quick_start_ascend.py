"""Quick-start test: doc under test is ``sources/llm_d/quick_start.md``.

Run: ``python -m unittest tests.llm_d.test_quick_start_ascend -v 2>&1``

Env (injected by the engine ``quick-start-template.yml``, triggered by
``llm_d-quick-start.yml``): ``MONITORED_DOC_URL``, ``UPSTREAM_REF``,
``NPU_READY=true`` (otherwise the class is skipped).
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import unittest
from pathlib import Path

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


# Stop only process groups named in these files, and only if /proc/pid
# still looks like that process. Never pkill.
# The runner may host other inference jobs.
_OWNED_PROCESSES = (
    (Path('/root/llm-d/envoy.pid'), 'envoy'),
    (Path('/root/llm-d/epp.pid'), 'epp'),
    (Path('/root/llm-d/vllm.pid'), 'vllm'),
)


def _cmdline_of(pid: int) -> str:
    try:
        raw = Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:
        return ''
    return raw.replace(b'\x00', b' ').decode('utf-8', errors='replace')


def _stop_pid_file(pid_file: Path, needle: str, wait_s: float = 15.0) -> None:
    if not pid_file.is_file():
        return
    raw = pid_file.read_text(encoding='utf-8').strip()
    if not raw.isdigit():
        pid_file.unlink(missing_ok=True)
        return
    pid = int(raw)
    if needle not in _cmdline_of(pid):
        pid_file.unlink(missing_ok=True)
        return

    def _signal_group(sig: int) -> None:
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return
        except OSError:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                return

    _signal_group(signal.SIGTERM)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pid_file.unlink(missing_ok=True)
            return
        time.sleep(0.2)
    _signal_group(signal.SIGKILL)
    pid_file.unlink(missing_ok=True)


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 7200
    USER_AGENT = 'ascend-docs/quick-start'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # typo in CANN's Python driver (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    _MODEL_ID = 'Qwen/Qwen3-0.6B'
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'
    _NNAL_SET_ENV = '/usr/local/Ascend/nnal/atb/set_env.sh'

    @classmethod
    def prepare_environment(cls) -> None:
        path_dirs = '/usr/local/sbin:/usr/local/bin'
        current_path = os.environ.get('PATH', '')
        if path_dirs not in current_path:
            os.environ['PATH'] = f'{path_dirs}:{current_path}'

        missing = [
            path for path in (cls._CANN_SET_ENV, cls._NNAL_SET_ENV)
            if not os.path.isfile(path)
        ]
        if missing:
            raise RuntimeError(
                'required env scripts missing: ' + ', '.join(missing)
            )

        # Overwrite, do not setdefault. The container already has
        # LD_LIBRARY_PATH, and setdefault would keep the pre-NNAL value.
        merged = subprocess.run(
            [
                'bash', '-c',
                f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; '
                f'source {cls._NNAL_SET_ENV} >/dev/null 2>&1; env',
            ],
            capture_output=True, text=True, check=True,
        )
        for line in merged.stdout.splitlines():
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            os.environ[key] = value
        print('setup: sourced CANN and NNAL env')

        # huggingface_hub 1.x defaults to Xet. That data plane talks to
        # cas-server.xethub.hf.co and is not covered by HF_ENDPOINT.
        os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

        prev_autoload = os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')
        os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
        try:
            ensure_safetensors()
            diagnose_mount_environment(
                cache_root=resolve_huggingface_cache(),
                model_id=cls._MODEL_ID,
            )
            report_huggingface_state(cls._MODEL_ID)
            try:
                purge_huggingface_corrupt(resolve_huggingface_cache())
            except ModuleNotFoundError as exc:
                print(f'setup: skip model cache purge ({exc})')
        finally:
            if prev_autoload is None:
                os.environ.pop('TORCH_DEVICE_BACKEND_AUTOLOAD', None)
            else:
                os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = prev_autoload

    def post_process(self) -> None:
        for pid_file, needle in _OWNED_PROCESSES:
            _stop_pid_file(pid_file, needle)

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
