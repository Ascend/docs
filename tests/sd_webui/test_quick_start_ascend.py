"""Quick-start test for ``sources/sd_webui/quick_start.md``."""

from __future__ import annotations

import os
import subprocess
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from doc_test.base import MarkdownDocTestBase, SetupCommand, TestCommand
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
    _CONSTRAINTS_FILE = '/tmp/sd-webui_npu_constraints.txt'
    _CLUSTER_INDEX = (
        'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    )
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'
    _API_READY_ENDPOINT = 'http://127.0.0.1:7861/docs'
    _API_READY_ATTEMPTS = 120
    _API_READY_INTERVAL_S = 5
    _WEBUI_LOG = Path('/tmp/sdwebui.log')
    _GENERATED_PNG = Path('/tmp/sd-turbo-out.png')

    def _verify_generated_png(self) -> None:
        """CI-side guard: the doc only reports success and the output path.

        Empty or truncated images are the failure mode the doc used to assert
        inline; keeping the check here preserves the coverage without exposing
        size / magic-byte assertions to readers of the quick start.
        """
        if not self._GENERATED_PNG.is_file():
            raise AssertionError(
                f'generated image not found: {self._GENERATED_PNG}'
            )
        image = self._GENERATED_PNG.read_bytes()
        if len(image) <= 10000:
            raise AssertionError(
                'generated image is suspiciously small '
                f'({len(image)} bytes): {self._GENERATED_PNG}'
            )
        if image[:8] != b'\x89PNG\r\n\x1a\n':
            raise AssertionError(
                'generated image is not a PNG '
                f'(magic={image[:8]!r}): {self._GENERATED_PNG}'
            )
        self.log(
            f'[Step] verified generated PNG ({len(image)}B): '
            f'{self._GENERATED_PNG}'
        )

    def _webui_log_tail(self) -> str:
        if not self._WEBUI_LOG.is_file():
            return ''
        return '\n'.join(
            self._WEBUI_LOG.read_text(
                encoding='utf-8', errors='replace'
            ).splitlines()[-100:]
        )

    def _log_webui_tail(self) -> None:
        self.log('--- tail /tmp/sdwebui.log ---')
        self.log(self._webui_log_tail() or '(log file missing or empty)')

    def _wait_for_api(self) -> None:
        for attempt in range(1, self._API_READY_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(
                    self._API_READY_ENDPOINT, timeout=10
                ) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            self.log(
                'waiting for api: '
                f'attempt {attempt}/{self._API_READY_ATTEMPTS}'
            )
            time.sleep(self._API_READY_INTERVAL_S)

        self._log_webui_tail()
        raise RuntimeError(
            f'api not ready after {self._API_READY_ATTEMPTS} attempts'
        )

    def _run_one(self, cmd, results, env, cwd, timeout, idx):
        if (
            isinstance(cmd, SetupCommand)
            and cmd.language == 'python'
            and '"launch.py"' in cmd.cmd
        ):
            actual_cmd = self.substitute_placeholders(
                cmd.cmd, cmd.load, self._captures
            )
            self.log(
                'running WebUI setup with CI output redirected to '
                '/tmp/sdwebui.log'
            )
            try:
                with self._WEBUI_LOG.open(
                    'w', encoding='utf-8'
                ) as log_file:
                    process = subprocess.run(
                        [*self._LANG_RUNNER[cmd.language], actual_cmd],
                        env=env,
                        cwd=cwd,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=timeout,
                    )
            except subprocess.TimeoutExpired:
                self._log_webui_tail()
                raise
            if process.returncode != 0:
                self._log_webui_tail()
                raise AssertionError(
                    'WebUI setup command failed '
                    f'(rc={process.returncode})'
                )
            return

        if (
            isinstance(cmd, TestCommand)
            and getattr(cmd, 'id', None) == 'txt2img'
        ):
            self._wait_for_api()
            super()._run_one(cmd, results, env, cwd, timeout, idx)
            self._verify_generated_png()
            return
        return super()._run_one(cmd, results, env, cwd, timeout, idx)

    def pre_process(self) -> str:
        doc = (
            Path(__file__).resolve().parents[2]
            / 'sources'
            / 'sd_webui'
            / 'quick_start.md'
        )
        return doc.read_text(encoding='utf-8')

    @classmethod
    def prepare_environment(cls) -> None:
        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                [
                    'bash',
                    '-c',
                    f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env',
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                os.environ.setdefault(key, value)
            print('setup: sourced CANN env from set_env.sh')
        else:
            print('setup: skipping CANN env source')

        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as file:
            file.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['UV_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')

        project_root = (
            Path(__file__).resolve().parents[2] / 'sources' / 'sd_webui'
        )
        os.chdir(project_root)
        print(f'setup: cwd -> {project_root}')

        probe_script = (
            'import torch, torch_npu, torchvision\n'
            'torch_version = torch.__version__.split("+", 1)[0]\n'
            'torchvision_version = torchvision.__version__.split("+", 1)[0]\n'
            'raise SystemExit(0 if '
            'torch_version == "2.9.0" and '
            'torchvision_version == "0.24.0" and '
            'torch_npu.__version__ == "2.9.0.post6" else 1)'
        )
        probe = subprocess.run(
            ['python', '-c', probe_script],
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            versions = subprocess.run(
                [
                    'python',
                    '-c',
                    'import torch, torch_npu, torchvision; '
                    'print(torch.__version__, torchvision.__version__, '
                    'torch_npu.__version__)',
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            print(
                'setup: reusing image torch stack '
                f'({versions.stdout.strip()})'
            )
        else:
            print(
                'setup: installing torch==2.9.0 torchvision==0.24.0 '
                'torch_npu==2.9.0.post6'
            )
            subprocess.run(
                [
                    'python',
                    '-m',
                    'pip',
                    'install',
                    '--index-url',
                    cls._CLUSTER_INDEX,
                    '--extra-index-url',
                    cls._ASCEND_EXTRA,
                    'torch==2.9.0',
                    'torchvision==0.24.0',
                    'torch_npu==2.9.0.post6',
                ],
                check=True,
            )

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
        self._kill_webui()
        try:
            self.run_template()
        finally:
            self._kill_webui()

    @staticmethod
    def _kill_webui() -> None:
        subprocess.run(
            ['pkill', '-f', 'launch.py --nowebui'],
            capture_output=True,
            timeout=10,
        )


if __name__ == '__main__':
    unittest.main()
