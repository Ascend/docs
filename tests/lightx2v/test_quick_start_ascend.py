"""Quick-start-Ascend test: doc under test is sources/lightx2v/quick_start.md."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
import zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

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


_MODEL_ID = 'Wan-AI/Wan2.1-T2V-1.3B'
_VAE_FILE = 'Wan2.1_VAE.pth'
_SNAPSHOT_PATH_PREFIX = 'LIGHTX2V_SNAPSHOT_DIR='


def _download_model_snapshot(model_id: str) -> str:
    """Use the installed ModelScope package, not tests/modelscope on sys.path."""
    script = (
        'import sys\n'
        'from modelscope import snapshot_download\n'
        f'print({_SNAPSHOT_PATH_PREFIX!r} + snapshot_download(sys.argv[1]))\n'
    )
    result = subprocess.run(
        [sys.executable, '-I', '-c', script, model_id],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            'ModelScope snapshot download failed:\n'
            f'{result.stderr[-4000:]}\n{result.stdout[-4000:]}'
        )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_SNAPSHOT_PATH_PREFIX):
            return line[len(_SNAPSHOT_PATH_PREFIX):]
    raise RuntimeError(
        'ModelScope snapshot download returned no model directory:\n'
        f'{result.stdout[-4000:]}'
    )


@contextmanager
def _model_cache_lock(cache_root: Path):
    """Serialize downloads and inference sharing the host's ModelScope mount."""
    import fcntl

    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / '.lightx2v-quick-start.lock'
    with lock_path.open('a+b') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _vae_error(path: Path) -> str | None:
    if not path.is_file():
        return 'missing file'
    try:
        with zipfile.ZipFile(path) as archive:
            if not any(name.endswith('/data.pkl') for name in archive.namelist()):
                return 'missing PyTorch data.pkl entry'
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                return f'corrupt ZIP member: {corrupt_member}'
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        return str(exc)
    return None


def _remove_model_cache(model_dir: Path, log) -> None:
    if model_dir.is_symlink():
        target = model_dir.resolve()
        cache_parent = model_dir.parent.resolve()
        if target.parent != cache_parent or target == model_dir:
            raise RuntimeError(
                f'refusing to remove model cache outside {cache_parent}: {target}'
            )
        if target.exists() and not target.is_dir():
            raise RuntimeError(f'model cache target is not a directory: {target}')
        log(f'cache: removing model link {model_dir} and target {target}')
        model_dir.unlink()
        if target.is_dir():
            shutil.rmtree(target)
    elif model_dir.is_dir():
        log(f'cache: removing model directory {model_dir}')
        shutil.rmtree(model_dir)
    elif model_dir.exists():
        raise RuntimeError(f'model cache path is not a directory: {model_dir}')


def _prepare_model_cache(cache_root: Path, download, log) -> None:
    """Repair only this model's cache, then fail if its VAE is still bad."""
    model_dir = cache_root / 'hub' / 'models' / 'Wan-AI' / 'Wan2.1-T2V-1.3B'
    for attempt in range(2):
        downloaded_dir = Path(download(_MODEL_ID))
        if downloaded_dir.resolve() != model_dir.resolve():
            raise RuntimeError(
                f'unexpected ModelScope model directory: {downloaded_dir}'
            )
        vae_path = model_dir / _VAE_FILE
        error = _vae_error(vae_path)
        if error is None:
            log(f'cache: verified {_MODEL_ID}/{_VAE_FILE}')
            return
        log(f'cache: invalid {vae_path}: {error}')
        if attempt == 0:
            _remove_model_cache(model_dir, log)
    raise RuntimeError(
        f'{_MODEL_ID}/{_VAE_FILE} is invalid after one re-download: {error}'
    )


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 3600
    USER_AGENT = 'cosdt-ci-test/quick-start'
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',
        'ERR99999',
        'RuntimeError: Failed to load the backend extension: torch_npu',
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
    _CONSTRAINTS_FILE = '/tmp/lightx2v_npu_constraints.txt'

    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    _PROJECT_ROOT = '/root/lightx2v-test'

    _OUTPUT_VIDEO = Path('save_results/output_lightx2v_wan_t2v.mp4')

    def _verify_output_video(self) -> None:
        """CI-side guard: the doc only reports that the video was saved.

        Empty or truncated videos are the failure mode the doc used to assert
        inline; keeping the check here preserves the coverage without exposing
        container / moov-box assertions to readers of the quick start.
        """
        if not self._OUTPUT_VIDEO.is_file():
            raise AssertionError(
                f'generated video not found: {self._OUTPUT_VIDEO}'
            )
        data = self._OUTPUT_VIDEO.read_bytes()
        if len(data) <= 100_000:
            raise AssertionError(
                'generated video is suspiciously small '
                f'({len(data)} bytes): {self._OUTPUT_VIDEO}'
            )
        if data[4:8] != b'ftyp':
            raise AssertionError(
                'generated video is not an MP4 '
                f'(magic={data[4:8]!r}): {self._OUTPUT_VIDEO}'
            )
        if b'moov' not in data:
            raise AssertionError(
                f'truncated MP4, no moov box: {self._OUTPUT_VIDEO}'
            )
        self.log(
            f'[Step] verified output video ({len(data)}B): '
            f'{self._OUTPUT_VIDEO}'
        )

    def _run_one(self, cmd, results, env, cwd, timeout, idx):
        if (
            isinstance(cmd, TestCommand)
            and getattr(cmd, 'id', None) == 'lightx2v-wan-t2v'
        ):
            cache_root = resolve_modelscope_cache()
            self.log(f'cache: waiting for LightX2V lock in {cache_root}')
            with _model_cache_lock(cache_root):
                self.log('cache: LightX2V lock acquired')
                purge_modelscope_corrupt(cache_root)
                _prepare_model_cache(cache_root, _download_model_snapshot, self.log)
                super()._run_one(cmd, results, env, cwd, timeout, idx)
                self._verify_output_video()
            return
        return super()._run_one(cmd, results, env, cwd, timeout, idx)

    def pre_process(self) -> str:
        doc = Path(__file__).resolve().parent.parent.parent / 'sources' / 'lightx2v' / 'quick_start.md'
        return doc.read_text(encoding='utf-8')

    @classmethod
    def prepare_environment(cls) -> None:
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

        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE

        _PROBE_SCRIPT = (
            'import torch, torch_npu\n'
            'raise SystemExit(0 if torch.npu.is_available() else 1)\n'
        )
        probe = subprocess.run(
            ['python', '-c', _PROBE_SCRIPT],
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            _VERSIONS_SCRIPT = (
                'import torch, torch_npu; '
                'print(torch.__version__, torch_npu.__version__)'
            )
            versions = subprocess.run(
                ['python', '-c', _VERSIONS_SCRIPT], capture_output=True,
                text=True, check=True,
            )
            print(f'setup: reusing image torch stack ({versions.stdout.strip()})')
        else:
            print(
                'setup: torch stack probe failed, the doc install-torch '
                'block will install the pinned stack'
            )

        try:
            os.makedirs(cls._PROJECT_ROOT, exist_ok=True)
        except OSError as exc:
            print(f'setup: doc cwd mkdir failed: {exc}')
        os.chdir(cls._PROJECT_ROOT)
        print(f'setup: cwd -> {os.getcwd()}')

        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'

        ensure_safetensors()

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


class TestModelCacheGuard(unittest.TestCase):
    def test_snapshot_download_uses_isolated_installed_package(self) -> None:
        result = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout='download log\nLIGHTX2V_SNAPSHOT_DIR=/cache/model\n',
            stderr='',
        )
        with mock.patch('subprocess.run', return_value=result) as run:
            self.assertEqual(_download_model_snapshot(_MODEL_ID), '/cache/model')
        command = run.call_args.args[0]
        self.assertEqual(command[0:3], [sys.executable, '-I', '-c'])
        self.assertEqual(command[4], _MODEL_ID)

    def test_snapshot_download_reports_subprocess_failure(self) -> None:
        result = subprocess.CompletedProcess(
            args=[], returncode=1, stdout='', stderr='model download failed',
        )
        with mock.patch('subprocess.run', return_value=result):
            with self.assertRaisesRegex(RuntimeError, 'model download failed'):
                _download_model_snapshot(_MODEL_ID)

    def test_healthy_vae_is_kept(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / 'hub/models/Wan-AI/Wan2.1-T2V-1.3B'
            model.mkdir(parents=True)
            with zipfile.ZipFile(model / _VAE_FILE, 'w') as archive:
                archive.writestr('archive/data.pkl', b'valid')
            download = mock.Mock(return_value=str(model))
            _prepare_model_cache(root, download, lambda _: None)
            download.assert_called_once_with(_MODEL_ID)
            self.assertTrue((model / _VAE_FILE).is_file())

    def test_corrupt_vae_is_redownloaded_once(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / 'hub/models/Wan-AI/Wan2.1-T2V-1.3B'
            sibling = root / 'hub/models/Other/model'
            sibling.mkdir(parents=True)
            (sibling / 'keep').write_text('keep')
            model.mkdir(parents=True)
            (model / _VAE_FILE).write_bytes(b'broken')
            calls = 0

            def download(_model_id):
                nonlocal calls
                calls += 1
                if calls == 2:
                    model.mkdir(parents=True)
                    with zipfile.ZipFile(model / _VAE_FILE, 'w') as archive:
                        archive.writestr('archive/data.pkl', b'valid')
                return str(model)

            _prepare_model_cache(root, download, lambda _: None)
            self.assertEqual(calls, 2)
            self.assertTrue((sibling / 'keep').is_file())

    def test_corrupt_vae_behind_model_link_is_redownloaded(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / 'hub/models/Wan-AI/Wan2.1-T2V-1.3B'
            target = model.with_name('Wan2___1-T2V-1___3B')
            target.mkdir(parents=True)
            (target / _VAE_FILE).write_bytes(b'broken')
            try:
                model.symlink_to(target, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f'directory symlinks unavailable: {exc}')
            calls = 0

            def download(_model_id):
                nonlocal calls
                calls += 1
                if calls == 2:
                    target.mkdir(parents=True)
                    with zipfile.ZipFile(target / _VAE_FILE, 'w') as archive:
                        archive.writestr('archive/data.pkl', b'valid')
                    model.symlink_to(target, target_is_directory=True)
                return str(model)

            _prepare_model_cache(root, download, lambda _: None)
            self.assertEqual(calls, 2)
            self.assertTrue(model.is_symlink())
            self.assertIsNone(_vae_error(model / _VAE_FILE))

    def test_model_link_outside_its_cache_directory_is_not_removed(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / 'hub/models/Wan-AI/Wan2.1-T2V-1.3B'
            model.parent.mkdir(parents=True)
            unrelated = root / 'unrelated'
            unrelated.mkdir()
            (unrelated / 'keep').write_text('keep')
            try:
                model.symlink_to(unrelated, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f'directory symlinks unavailable: {exc}')
            with self.assertRaisesRegex(RuntimeError, 'refusing to remove'):
                _remove_model_cache(model, lambda _: None)
            self.assertTrue(model.is_symlink())
            self.assertTrue((unrelated / 'keep').is_file())

    def test_missing_vae_still_fails_after_one_retry(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / 'hub/models/Wan-AI/Wan2.1-T2V-1.3B'
            model.mkdir(parents=True)
            calls = 0

            def download(_model_id):
                nonlocal calls
                calls += 1
                model.mkdir(parents=True, exist_ok=True)
                return str(model)

            with self.assertRaisesRegex(RuntimeError, 'after one re-download'):
                _prepare_model_cache(root, download, lambda _: None)
            self.assertEqual(calls, 2)

    def test_lock_is_released_on_success_and_failure(self) -> None:
        from tempfile import TemporaryDirectory

        fake_fcntl = mock.Mock(LOCK_EX=2, LOCK_UN=8)
        with TemporaryDirectory() as temp_dir:
            with mock.patch.dict('sys.modules', {'fcntl': fake_fcntl}):
                with _model_cache_lock(Path(temp_dir)):
                    pass
                with self.assertRaisesRegex(RuntimeError, 'test error'):
                    with _model_cache_lock(Path(temp_dir)):
                        raise RuntimeError('test error')
        self.assertEqual(
            [call.args[1] for call in fake_fcntl.flock.call_args_list],
            [2, 8, 2, 8],
        )


if __name__ == '__main__':
    unittest.main()
