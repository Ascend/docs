"""Guard sources/dgl/quick_start.md at the engine-selected DGL-Ascend commit.

MONITORED_DOC_URL selects the document through the GitHub Contents API.
UPSTREAM_REF is the full default-branch SHA resolved for this tagless fork.
NPU_READY=true enables the end-to-end test. User paths are relative to cwd;
the fixture supplies CANN, CUDA exclusions, a temporary cwd, and card 0.
"""

from __future__ import annotations

import importlib
import math
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from doc_test.base import MarkdownDocTestBase


def _checked_run(argv, *, env, cwd, timeout=120):
    result = subprocess.run(
        argv, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout,
    )
    if result.returncode:
        raise AssertionError(
            f'command failed (rc={result.returncode}): {argv!r}\n'
            f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
        )
    return result.stdout.strip()


def _pin_source(repo: Path, upstream_ref: str, env: dict) -> str:
    if not re.fullmatch(r'[0-9a-f]{40}', upstream_ref):
        raise AssertionError('UPSTREAM_REF must be the full engine-resolved SHA')
    # The branch may advance between engine resolution and the document clone.
    exists = subprocess.run(
        ['git', 'cat-file', '-e', upstream_ref + '^{commit}'],
        capture_output=True, env=env, cwd=repo, timeout=120,
    )
    if exists.returncode:
        _checked_run(['git', 'fetch', 'origin', upstream_ref], env=env, cwd=repo)
    _checked_run(['git', 'checkout', '--detach', upstream_ref], env=env, cwd=repo)
    actual = _checked_run(['git', 'rev-parse', 'HEAD'], env=env, cwd=repo)
    if actual != upstream_ref:
        raise AssertionError(f'checkout {actual} differs from engine SHA {upstream_ref}')
    return actual


def _validate_training_output(output: str) -> float:
    match = re.search(r'Epoch 1/1 finished\.[^\n]*Average BPR Loss:\s*(\S+)', output)
    if not match:
        raise AssertionError('LightGCN did not finish its training epoch')
    try:
        loss = float(match.group(1))
    except ValueError as exc:
        raise AssertionError('LightGCN reported an invalid BPR loss') from exc
    if not math.isfinite(loss):
        raise AssertionError(f'LightGCN reported a non-finite BPR loss: {loss}')
    return loss


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 1800
    BUILD_TIMEOUT = 7200
    USER_AGENT = 'ascend-docs/dgl-quick-start'
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

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

    @classmethod
    def prepare_environment(cls) -> None:
        cls._original_cwd = Path.cwd()
        cls._original_env = os.environ.copy()
        cls._work_dir = None
        cls.addClassCleanup(cls._cleanup_environment)
        if not Path(cls._CANN_SET_ENV).is_file():
            raise RuntimeError(f'CANN environment script missing: {cls._CANN_SET_ENV}')
        merged = subprocess.run(
            ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1 && env -0'],
            capture_output=True, check=True, timeout=120,
        )
        for entry in merged.stdout.split(b'\0'):
            if b'=' in entry:
                key, value = entry.split(b'=', 1)
                os.environ[os.fsdecode(key)] = os.fsdecode(value)
        cls._work_dir = Path(tempfile.mkdtemp(prefix='dgl-quick-start-'))
        constraints = cls._work_dir / 'pip-constraints.txt'
        constraints.write_text('\n'.join(cls._CUDA_CONSTRAINTS) + '\n', encoding='utf-8')
        os.environ['PIP_CONSTRAINT'] = str(constraints)
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'
        os.environ['DGLBACKEND'] = 'pytorch'
        os.chdir(cls._work_dir)

    def _run_one(self, cmd, results, env, cwd, timeout, idx):
        repo = Path(cwd) / 'dgl-ascend'
        if (repo / '.git').is_dir() and not getattr(self, '_source_sha', None):
            self._source_sha = _pin_source(repo, env.get('UPSTREAM_REF', ''), env)
            self.log(f'DGL source matches engine SHA {self._source_sha}')
        if 'script/build_dgl_ascend.sh' in cmd.cmd:
            timeout = self.BUILD_TIMEOUT
        if getattr(cmd, 'id', None) == 'lightgcn-train':
            self._verify_training_graph()
        self._active_id = getattr(cmd, 'id', None)
        try:
            super()._run_one(cmd, results, env, cwd, timeout, idx)
        finally:
            self._active_id = None
        if getattr(cmd, 'id', None) == 'install-ascend-deps':
            _checked_run(
                [sys.executable, '-c',
                 'import torch, torch_npu; '
                 "assert torch.npu.is_available(), 'NPU unavailable'; "
                 "assert torch.npu.device_count() > 0, 'No visible NPU'"],
                env=env, cwd=cwd,
            )
        if getattr(cmd, 'id', None) == 'install-dgl':
            self._verify_installed_backend(env, cwd)

    def _verify_installed_backend(self, env, cwd):
        # Validate installation provenance outside the source package directory,
        # then exercise graph message passing and its backward path on the NPU.
        # `tests` is on sys.path for doc_test, so a top-level `import dgl`
        # would resolve to tests/dgl instead of the installed package. Remove
        # that shadowing entry while importing and restore it afterwards.
        tests_root = str(Path(__file__).resolve().parents[1])
        shadow_paths = [
            (index, entry) for index, entry in enumerate(sys.path)
            if entry and Path(entry).resolve() == Path(tests_root)
        ]
        for index, _ in reversed(shadow_paths):
            del sys.path[index]
        try:
            # setup.py install may add an easy-install.pth entry after this test
            # process started; process site-packages again so the egg is visible.
            for site_dir in (*site.getsitepackages(), site.getusersitepackages()):
                if site_dir and Path(site_dir).is_dir():
                    site.addsitedir(site_dir)
            importlib.invalidate_caches()
            import torch
            import torch_npu
            import dgl
            import dgl._ffi._cy3.core

            source = (Path.cwd() / 'dgl-ascend/python').resolve()
            assert not Path(dgl.__file__).resolve().is_relative_to(source), 'Source tree shadows installed DGL'
            assert not Path(dgl._ffi._cy3.core.__file__).resolve().is_relative_to(source), 'Source tree shadows installed Cython extension'
            installed_lib = Path(dgl._ffi.base._LIB._name).resolve()
            assert not installed_lib.is_relative_to(source.parent), 'DGL loads the source-tree library'

            def digest_file(path):
                digest = sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(chunk)
                return digest.digest()

            assert digest_file(source.parent / 'build/libdgl.so') == digest_file(installed_lib), \
                'Installed DGL library differs from this checkout build'
            graph = dgl.graph(([0, 1, 2], [1, 2, 0])).to('npu')
            x = torch.ones(3, 2, device='npu', requires_grad=True)
            graph.ndata['x'] = x
            graph.update_all(dgl.function.copy_u('x', 'm'), dgl.function.sum('m', 'h'))
            assert graph.device.type == 'npu' and graph.ndata['h'].device.type == 'npu'
            graph.ndata['h'].sum().backward()
            assert x.grad is not None and torch.isfinite(x.grad).all().item()
            actual = _checked_run(['git', 'rev-parse', 'HEAD'], env=env, cwd=Path(cwd) / 'dgl-ascend')
            if actual != self._source_sha:
                raise AssertionError('DGL source changed during build or installation')
            self.log('Installed DGL graph operations and backward pass run on NPU')
        finally:
            for index, entry in shadow_paths:
                sys.path.insert(index, entry)

    def _verify_training_graph(self):
        # Check the complete training graph against an independent CPU edge count.
        # Use the same inherited environment as the training command, without
        # forcing launch blocking; retain GraphConv's zero-degree guard.
        import torch
        import torch_npu
        import dgl

        train_file = Path('dgl-ascend/examples/pytorch/lightgcn/gowalla/train.txt')
        users, items = [], []
        for line in train_file.read_text().splitlines():
            values = list(map(int, line.split()))
            users.extend([values[0]] * (len(values) - 1))
            items.extend(values[1:])
        src = torch.tensor(users, dtype=torch.int64)
        n_users = int(src.max()) + 1
        dst = torch.tensor(items, dtype=torch.int64) + n_users
        n_nodes = int(dst.max()) + 1
        src_bidir = torch.cat([src, dst])
        dst_bidir = torch.cat([dst, src])
        expected_in = torch.bincount(dst_bidir, minlength=n_nodes)
        expected_out = torch.bincount(src_bidir, minlength=n_nodes)
        assert (expected_in > 0).all().item(), 'Training data contains isolated nodes'
        graph = dgl.graph((src_bidir, dst_bidir), num_nodes=n_nodes).to('npu')
        for _ in range(3):
            # Interleave PyTorch work with DGL degree queries, as training does.
            _pending = torch.randn(4096, 64, device='npu').square()
            for name, expected in (('in', expected_in), ('out', expected_out)):
                actual = getattr(graph, name + '_degrees')().cpu()
                mismatch = int((actual != expected).sum())
                assert mismatch == 0, f'NPU {name}-degrees differ from CPU at {mismatch} nodes'
        print(f'Gowalla NPU degrees match CPU for all {n_nodes} nodes')
        self.log('Full Gowalla graph degrees match CPU in the training environment')

    def run_command(self, cmd, env, cwd, timeout, language='shell'):
        rc, output, error = super().run_command(cmd, env, cwd, timeout, language)
        if rc == 0 and getattr(self, '_active_id', None) == 'lightgcn-train':
            _validate_training_output(output)
            # Put summary fields on separate lines for the document comparison.
            output = re.sub(r'(?m)^[ \t]+device:', 'device:', output)
            output = re.sub(r'(?m)^(.+?)(Average BPR Loss:)', r'\1\n\2', output)
        return rc, output, error

    @classmethod
    def _cleanup_environment(cls):
        if getattr(cls, '_original_env', None) is None:
            return
        os.chdir(cls._original_cwd)
        os.environ.clear()
        os.environ.update(cls._original_env)
        if cls._work_dir is not None:
            shutil.rmtree(cls._work_dir)
        cls._original_env = None

    def post_process(self):
        self._cleanup_environment()

    @classmethod
    def setUpClass(cls):
        if os.environ.get('NPU_READY', '').strip().lower() == 'true':
            cls.prepare_environment()

    @unittest.skipUnless(
        os.environ.get('NPU_READY', '').strip().lower() == 'true',
        'end-to-end requires an Ascend NPU runner',
    )
    def test_runs_doc(self):
        self.run_template()


class TestGuardValidation(unittest.TestCase):
    def test_checkout_matches_engine_even_if_branch_advanced(self):
        with tempfile.TemporaryDirectory(prefix='dgl-source-test-') as directory:
            repo = Path(directory)
            env = os.environ.copy()
            _checked_run(['git', 'init', '-q'], env=env, cwd=repo)
            sha = None
            for value in ('engine-selected', 'newer-branch-head'):
                (repo / 'revision.txt').write_text(value, encoding='utf-8')
                _checked_run(['git', 'add', 'revision.txt'], env=env, cwd=repo)
                _checked_run(
                    ['git', '-c', 'user.name=Guard Test',
                     '-c', 'user.email=guard-test@example.invalid',
                     'commit', '-q', '-m', value],
                    env=env, cwd=repo,
                )
                if sha is None:
                    sha = _checked_run(['git', 'rev-parse', 'HEAD'], env=env, cwd=repo)
            self.assertEqual(_pin_source(repo, sha, env), sha)
            self.assertEqual((repo / 'revision.txt').read_text(), 'engine-selected')

    def test_missing_or_unresolved_ref_is_rejected(self):
        for ref in ('', 'master', 'v0.0.2'):
            with self.subTest(ref=ref), self.assertRaises(AssertionError):
                _pin_source(Path.cwd(), ref, os.environ.copy())

    def test_completed_training_has_finite_loss(self):
        output = 'Epoch 1/1 finished. Epoch time: 64.64s, Average BPR Loss: 0.6739\n'
        self.assertEqual(_validate_training_output(output), 0.6739)

    def test_invalid_or_incomplete_training_is_rejected(self):
        for output in (
            'Average BPR Loss: 0.6739',
            'Epoch 1/1 finished. Average BPR Loss: nan',
            'Epoch 1/1 finished. Average BPR Loss: inf',
            'Epoch 1/1 finished. Average BPR Loss: invalid',
        ):
            with self.subTest(output=output), self.assertRaises(AssertionError):
                _validate_training_output(output)


if __name__ == '__main__':
    unittest.main()
