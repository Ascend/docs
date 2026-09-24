"""End-to-end test for the FastChat Ascend quick-start document."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from doc_test.base import (
    MarkdownDocTestBase,
    SetupCommand,
    TestCommand,
)
from doc_test.model_cache import (
    ensure_safetensors,
    purge_modelscope_corrupt,
    resolve_modelscope_cache,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_DIR = _REPO_ROOT / 'sources' / 'fastchat'
_WORK_DIR = Path('/tmp/fastchat-quick-start')
_SERVICE_DIR = _WORK_DIR / '.fastchat'
_CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'
_MODEL_ID = 'Qwen2.5-0.5B-Instruct'
_MODELS_URL = 'http://127.0.0.1:8000/v1/models'
_CHAT_RESPONSE_PATH = Path('/tmp/fastchat-chat.json')
_READINESS_TIMEOUT = 900
_SERVICE_MODULES = (
    ('controller', 'fastchat.serve.controller'),
    ('worker', 'fastchat.serve.model_worker'),
    ('api', 'fastchat.serve.openai_api_server'),
)
_RELEASE_VERSION_RE = re.compile(r'[0-9]+(?:\.[0-9]+)+(?:[0-9A-Za-z.!+_-]*)?')


def _is_truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() == 'true')


def _e2e_enabled() -> bool:
    return _is_truthy(os.environ.get('NPU_READY'))


def _release_version(upstream_ref: str) -> str:
    ref = upstream_ref.strip()
    version = ref[1:] if ref.startswith('v') else ref
    if not _RELEASE_VERSION_RE.fullmatch(version):
        raise RuntimeError(
            f'UPSTREAM_REF {upstream_ref!r} is not a release tag that can be '
            'mapped to a PyPI version'
        )
    return version


def _installed_fschat_version() -> str:
    """Return the fschat version actually importable in this interpreter."""

    result = subprocess.run(
        [sys.executable, '-c', 'import fastchat; print(fastchat.__version__)'],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _assert_version_alignment(installed: str, upstream_ref: str) -> None:
    """Fail when the installed fschat differs from the monitored release."""

    installed_version = installed.strip()
    monitored = _release_version(upstream_ref)
    if installed_version != monitored:
        raise RuntimeError(
            'FastChat version mismatch after install: '
            f'UPSTREAM_REF={upstream_ref!r} resolves to {monitored!r}, '
            f'but the installed fschat is {installed_version!r}'
        )


def _validate_chat_response(path: Path) -> None:
    """Validate the documented Chat Completions response."""

    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssertionError(f'invalid chat response file {path}: {exc}') from exc

    try:
        model = data['model']
        message = data['choices'][0]['message']
        role = message['role']
        content = message['content']
    except (KeyError, IndexError, TypeError) as exc:
        raise AssertionError(
            f'chat response is missing required fields: {data!r}'
        ) from exc

    if model != _MODEL_ID:
        raise AssertionError(
            f'chat response model mismatch: expected {_MODEL_ID!r}, got {model!r}'
        )
    if role != 'assistant':
        raise AssertionError(
            f'chat response role mismatch: expected "assistant", got {role!r}'
        )
    if not isinstance(content, str) or not content.strip():
        raise AssertionError('chat response assistant content is empty')


def _merge_sourced_env(script: str) -> None:
    """Source an environment script and merge its exported variables."""

    merged = subprocess.run(
        [
            'bash',
            '-c',
            f'source {shlex.quote(script)} >/dev/null 2>&1; env -0',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    for entry in merged.stdout.split('\0'):
        if not entry or '=' not in entry:
            continue
        key, _, value = entry.partition('=')
        os.environ[key] = value


def _service_log_tail() -> str:
    sections: list[str] = []
    for name in ('controller', 'worker', 'api'):
        path = _SERVICE_DIR / f'{name}.log'
        try:
            lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        except OSError:
            continue
        sections.append(f'--- {name}.log ---\n' + '\n'.join(lines[-50:]))
    return '\n'.join(sections) or '(service logs unavailable)'


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """Run the documented CLI and OpenAI-compatible API flows on one NPU."""

    DEFAULT_COMMAND_TIMEOUT = 1800
    USER_AGENT = 'cosdt-ci-test/quick-start'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',
        'ERR99999',
        'Address already in use',
        'No available worker',
    )

    _CUDA_CONSTRAINTS = (
        'cuda-toolkit<0',
        'cuda-python<0',
        'cuda-bindings<0',
        'cuda-core<0',
        'cuda-pathfinder<0',
        'flashinfer-python<0',
        'nvidia-cublas<0',
        'nvidia-cuda-runtime<0',
        'nvidia-cuda-nvrtc<0',
        'nvidia-cuda-cupti<0',
        'nvidia-cudnn<0',
        'nvidia-cudnn-frontend<0',
        'nvidia-cufft<0',
        'nvidia-curand<0',
        'nvidia-cusolver<0',
        'nvidia-cusparse<0',
        'nvidia-cutlass-dsl<0',
        'nvidia-cutlass-dsl-libs-base<0',
        'nvidia-cutlass-dsl-libs-core<0',
        'nvidia-cutlass-dsl-libs-cu12<0',
        'nvidia-ml-py<0',
        'nvidia-nccl<0',
        'nvidia-nvjitlink<0',
        'nvidia-nvtx<0',
        'nvidia-cublas-cu12<0',
        'nvidia-cuda-nvdisasm<0',
        'nvidia-cuda-runtime-cu12<0',
        'nvidia-cuda-nvrtc-cu12<0',
        'nvidia-cuda-cupti-cu12<0',
        'nvidia-cudnn-cu12<0',
        'nvidia-cufft-cu12<0',
        'nvidia-curand-cu12<0',
        'nvidia-cusolver-cu12<0',
        'nvidia-cusparse-cu12<0',
        'nvidia-cusparselt-cu12<0',
        'nvidia-nccl-cu12<0',
        'nvidia-nvjitlink-cu12<0',
        'nvidia-nvtx-cu12<0',
    )
    _CONSTRAINTS_FILE = '/tmp/fastchat_npu_constraints.txt'
    _CLUSTER_INDEX = (
        'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    )
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    def pre_process(self) -> str:
        doc_path = _PROJECT_DIR / 'quick_start.md'
        if not doc_path.is_file():
            raise RuntimeError(f'doc not found in local checkout: {doc_path}')
        upstream_ref = os.environ.get('UPSTREAM_REF', '')
        if not upstream_ref:
            raise RuntimeError('UPSTREAM_REF is required for FastChat version alignment')
        return doc_path.read_text(encoding='utf-8')

    def _start_documented_service(self, name: str, command: str, env, cwd) -> None:
        """Run a documented foreground service command concurrently for the test."""

        _SERVICE_DIR.mkdir(parents=True, exist_ok=True)
        log_handle = (_SERVICE_DIR / f'{name}.log').open('wb')
        try:
            process = subprocess.Popen(
                ['bash', '-c', command],
                cwd=cwd,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            log_handle.close()
            raise
        services = getattr(self, '_service_processes', {})
        services[name] = (process, log_handle)
        self._service_processes = services
        self.log(f'service start: {name} pid={process.pid}')

    def _stop_documented_services(self) -> None:
        services = getattr(self, '_service_processes', {})
        for process, _log_handle in reversed(tuple(services.values())):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        for name, (process, log_handle) in services.items():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            finally:
                log_handle.close()
            self.log(f'service stop: {name} rc={process.returncode}')
        self._service_processes = {}

    def _wait_for_model_service(self) -> None:
        deadline = time.monotonic() + _READINESS_TIMEOUT
        last_error = 'service has not responded yet'

        while time.monotonic() < deadline:
            services = getattr(self, '_service_processes', {})
            stopped = [
                f'{name} (rc={process.returncode})'
                for name, (process, _log_handle) in services.items()
                if process.poll() is not None
            ]
            missing = {name for name, _module in _SERVICE_MODULES} - services.keys()
            stopped.extend(f'{name} (not started)' for name in sorted(missing))
            if stopped:
                raise AssertionError(
                    'FastChat service exited before model registration: '
                    f'{stopped}\n{_service_log_tail()}'
                )

            try:
                with urllib.request.urlopen(_MODELS_URL, timeout=5) as response:
                    payload = json.load(response)
                items = payload.get('data', []) if isinstance(payload, dict) else []
                model_ids = {
                    item['id']
                    for item in items
                    if isinstance(item, dict) and isinstance(item.get('id'), str)
                }
                if _MODEL_ID in model_ids:
                    self.log(f'readiness: {_MODEL_ID} registered')
                    return
                last_error = f'registered models: {sorted(model_ids)}'
            except (
                json.JSONDecodeError,
                OSError,
                TimeoutError,
                TypeError,
                urllib.error.URLError,
            ) as exc:
                last_error = repr(exc)
            time.sleep(5)

        raise AssertionError(
            f'FastChat model was not ready after {_READINESS_TIMEOUT}s; '
            f'last response: {last_error}\n{_service_log_tail()}'
        )

    def _run_one(self, cmd, results, env, cwd, timeout, idx):
        if isinstance(cmd, SetupCommand):
            for name, module in _SERVICE_MODULES:
                if module in cmd.cmd:
                    self._start_documented_service(name, cmd.cmd, env, cwd)
                    return
            if 'fschat[model_worker]' in cmd.cmd:
                super()._run_one(cmd, results, env, cwd, timeout, idx)
                _assert_version_alignment(
                    _installed_fschat_version(),
                    os.environ.get('UPSTREAM_REF', ''),
                )
                return
        if isinstance(cmd, TestCommand) and cmd.id == 'install-fastchat':
            super()._run_one(cmd, results, env, cwd, timeout, idx)
            _assert_version_alignment(
                _installed_fschat_version(),
                os.environ.get('UPSTREAM_REF', ''),
            )
            return
        if isinstance(cmd, TestCommand) and cmd.id == 'check-model':
            self._wait_for_model_service()
        if isinstance(cmd, TestCommand) and cmd.id == 'api-chat':
            super()._run_one(cmd, results, env, cwd, timeout, idx)
            _validate_chat_response(_CHAT_RESPONSE_PATH)
            return
        return super()._run_one(cmd, results, env, cwd, timeout, idx)

    @classmethod
    def prepare_environment(cls) -> None:
        if not os.path.isfile(_CANN_SET_ENV):
            raise RuntimeError(f'required CANN environment script missing: {_CANN_SET_ENV}')
        _merge_sourced_env(_CANN_SET_ENV)

        Path(cls._CONSTRAINTS_FILE).write_text(
            '\n'.join(cls._CUDA_CONSTRAINTS) + '\n', encoding='utf-8'
        )
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'

        probe = subprocess.run(
            [
                sys.executable,
                '-c',
                'import torch, torch_npu; '
                "raise SystemExit(0 if torch.__version__.startswith('2.9.0') "
                "and torch_npu.__version__.startswith('2.9.0') else 1)",
            ],
            capture_output=True,
            check=False,
        )
        if probe.returncode != 0:
            subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'pip',
                    'install',
                    '--index-url',
                    cls._CLUSTER_INDEX,
                    '--extra-index-url',
                    cls._ASCEND_EXTRA,
                    'torch==2.9.0',
                    'torch_npu==2.9.0.post2',
                ],
                check=True,
            )

        ensure_safetensors()
        purge_modelscope_corrupt(resolve_modelscope_cache())

        if _WORK_DIR.exists():
            shutil.rmtree(_WORK_DIR)
        _WORK_DIR.mkdir(parents=True)
        os.chdir(_WORK_DIR)

    @classmethod
    def setUpClass(cls) -> None:
        if _e2e_enabled():
            cls.prepare_environment()

    def post_process(self) -> None:
        try:
            self._stop_documented_services()
        finally:
            os.chdir(_PROJECT_DIR)
            if _WORK_DIR.exists():
                shutil.rmtree(_WORK_DIR, ignore_errors=True)

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        self.run_template()


class TestVersionAlignment(unittest.TestCase):
    def test_matching_release_and_installed_version(self) -> None:
        _assert_version_alignment('0.2.36', 'v0.2.36')

    def test_mismatch_fails_after_install(self) -> None:
        with self.assertRaisesRegex(RuntimeError, 'version mismatch'):
            _assert_version_alignment('0.2.36', 'v0.2.37')

    def test_non_release_ref_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, 'not a release tag'):
            _assert_version_alignment('0.2.36', 'main')


if __name__ == '__main__':
    unittest.main()
