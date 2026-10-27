"""Quick-start test: doc under test is ``sources/mmengine/quick_start.md``.

Run: ``python -m unittest tests.mmengine.test_quick_start_ascend -v 2>&1``

Env (injected by the engine ``quick-start-template.yml``, triggered by
``mmengine-quick-start.yml``): ``MONITORED_DOC_URL``, ``UPSTREAM_REF``,
``NPU_READY=true`` (otherwise the class is skipped).

Scope: torch / torch_npu install + MMEngine install (source checkout
at the latest upstream release tag, then the PyPI wheel) + the opencv
runtime-library fix + 17 NPU smokes.

MMEngine is a *framework* (no bundled models), so the doc drives a
self-contained toy project: a 16-dim two-class MLP over a synthetic
Gaussian dataset, plus a custom metric / dataset / hook, all wired
through a two-level ``Config``. Nothing is downloaded from a model
hub, so the run is fast and deterministic.

The NPU-specific assertions are:
  * ``mmengine.device.get_device() == 'npu'``;
  * ``Runner`` places the model on ``npu:0`` (no CPU fallback);
  * ``amp.autocast(device_type='npu')`` yields float16 on device;
  * ``AmpOptimWrapper`` picks the NPU ``GradScaler``;
  * ``torch_npu.optim`` fused optimizers land in ``OPTIMIZERS``;
  * ``NPUProfilerHook`` writes an ``ascend_pt`` profiling tree.

MMEngine's own device handling covers the Ascend path end to end
(``mmengine.device`` -> ``npu``, ``Runner.wrap_model`` ->
``model.to(get_device())``, ``AmpOptimWrapper`` ->
``torch.npu.amp.GradScaler``), so no SDK shim is needed on the
MMEngine side.
"""

from __future__ import annotations

import os
import subprocess
import unittest

os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')

from doc_test.base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else (including unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """End-to-end test: fetch doc -> validate contract -> run ``#test-setup``
    / ``#test`` in order -> compare against ``#test-result``.

    ``prepare_environment`` only does runner-side scaffolding (CANN env +
    CUDA exclusion list + torch stack probe); MMEngine is installed by the
    doc's ``#test-setup`` / ``#test`` blocks, so the test exercises the
    exact install path users get.
    """

    # Cold run: git clone of mmengine + `uv pip install -e .` + the
    # PyPI-wheel reinstall + the opencv runtime-library fix + 17 NPU smokes.
    # Each #test block is individually bounded by DEFAULT_COMMAND_TIMEOUT;
    # 20 min per block leaves room for the heaviest single block (the
    # Runner training smoke) on a slow egress link.
    DEFAULT_COMMAND_TIMEOUT = 1200

    USER_AGENT = 'ascend-docs/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a CANN
    # failure surfaces a full stderr dump (head/tail by default would hide
    # the line that names the failure).
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # CANN toolkit emits this typo (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list. MMEngine itself pulls no CUDA
    # wheel, but its transitive deps (matplotlib / opencv / numpy) can
    # drag CUDA-typed wheels onto an aarch64 cluster node where they
    # neither install nor run. Pre-exclude the whole CUDA toolchain so
    # the resolver falls back to pure-Python / CPU alternatives.
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
        'nvidia-cusparselt<0',
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
    )
    _CONSTRAINTS_FILE = '/tmp/mmengine_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source.
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the GitHub workflow container image
    # (``image:`` input of ``mmengine-quick-start.yml``).
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    @classmethod
    def prepare_environment(cls) -> None:
        """Install CANN env + CUDA constraints + torch stack.

        MMEngine itself is intentionally NOT pre-installed here: the doc
        body runs ``git clone`` + ``uv pip install -e .`` against the
        upstream release tag injected by the workflow, then reinstalls
        the PyPI wheel, so the test exercises the exact install path
        users get.
        """
        # 0) CANN env: source set_env.sh and merge the env stream into
        # os.environ
        if os.path.isfile(cls._CANN_SET_ENV):
            merged = subprocess.run(
                ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
                capture_output=True, text=True, check=True,
            )
            for line in merged.stdout.splitlines():
                if '=' not in line:
                    continue
                key, _, value = line.partition('=')
                # Don't overwrite envs explicitly injected by the
                # workflow (jobs.env / steps.env); only fill in CANN
                # keys that are missing, to avoid conflicts.
                os.environ.setdefault(key, value)
            print('setup: sourced CANN env from set_env.sh')
        else:
            print(
                f'setup: skipping CANN env source ({cls._CANN_SET_ENV} not present)'
            )

        # 1) CUDA exclusion list + process-level env
        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['UV_CONSTRAINT'] = cls._CONSTRAINTS_FILE

        # 2) uv: the doc body's install steps are ``uv pip install ...``,
        # which gives clean stdout (no pip's "Obtaining/Installing
        # collected packages" preamble) so the #test-result fuzzy match
        # isn't polluted.
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) torch stack probe: when the version matches the image's
        # pre-installed wheels, reuse them (avoid the cluster cache
        # triggering ``+cpu`` resolution).
        _PROBE_SCRIPT = (
            'import torch, torch_npu\n'
            "raise SystemExit(0 if "
            "torch.__version__.startswith('2.9.0') "
            "and torch_npu.__version__.startswith('2.9.0') "
            "else 1)"
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
                ['python', '-c', _VERSIONS_SCRIPT],
                capture_output=True, text=True, check=True,
            )
            print(f'setup: reusing image torch stack ({versions.stdout.strip()})')
        else:
            print('setup: installing torch==2.9.0 torch_npu==2.9.0.post2')
            subprocess.run(
                [
                    'python', '-m', 'pip', 'install',
                    '--index-url', cls._CLUSTER_INDEX,
                    '--extra-index-url', cls._ASCEND_EXTRA,
                    'torch==2.9.0', 'torch_npu==2.9.0.post2',
                ],
                check=True,
            )

        # 4) uv request-level logging. uv prints NOTHING until the resolve
        # finishes, so a stalled download shows up as a silent hang in the
        # CI log. With RUST_LOG=uv=debug each HTTP request / candidate
        # wheel is logged to stderr, and the runner dumps the captured
        # stderr on timeout — the last logged URL is then the hop that
        # stalled.
        os.environ['RUST_LOG'] = 'uv=debug'

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per test class.

        ``@unittest.skipIf`` only skips the test *method* — ``setUpClass``
        itself always runs. The ``if _e2e_enabled()`` body guard below is
        what actually keeps heavy setup from firing when ``NPU_READY`` is
        unset.
        """
        if _e2e_enabled():
            cls.prepare_environment()

    @unittest.skipIf(
        not _e2e_enabled(),
        'end-to-end requires NPU runner; set NPU_READY=true',
    )
    def test_runs_doc(self) -> None:
        """Template-method entry point. The base class
        ``run_template()`` runs the full ``pre_process`` -> ``parse`` ->
        ``execute`` -> ``post_process`` flow. ``prepare_environment`` is
        triggered by ``setUpClass`` once, not from ``run_template``."""

        self.run_template()


if __name__ == '__main__':
    unittest.main()
