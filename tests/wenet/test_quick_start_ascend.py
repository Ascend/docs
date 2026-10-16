"""Quick-start test: doc under test is ``sources/wenet/quick_start.md``.

End-to-end case built on top of the ``MarkdownDocTestBase`` contract:
env check + WeNet ``main`` clone (+ ``[torch-npu]`` install) + aishell-1
download (~15.6 GB, Hugging Face) + data prep + conformer training
(stage 4, ``max_epoch=1``) + decode (stage 5) + jit export (stage 6).

Run: ``python -m unittest tests.wenet.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the engine
``quick-start-template.yml``, triggered by ``wenet-quick-start.yml``):
    ``MONITORED_DOC_URL``         doc to fetch and execute
    ``UPSTREAM_REF``              ref substituted into the doc's ``<ref>``;
                                  the trigger pins ``fixed_ref: main`` (the
                                  latest release ``v3.1.0`` has no
                                  ``examples/aishell/s0/run_npu.sh``)
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU runner:
                                  local dev machines / normal ubuntu runners
                                  have no ``/dev/davinci*`` device, and the
                                  hard run would fail on ``import torch_npu``.
"""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

from doc_test.base import MarkdownDocTestBase
from doc_test.model_cache import (
    diagnose_mount_environment,
    ensure_safetensors,
    purge_huggingface_corrupt,
    resolve_huggingface_cache,
)


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else (unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``sources/wenet/quick_start.md`` end-to-end test: fetch doc ->
    validate contract -> run ``#test-setup`` / ``#test`` in order ->
    compare against ``#test-result``.

    Scope: env check + WeNet clone/install + aishell-1 download and prep +
    one conformer epoch on 910B4 + stage 5 decode of 7176 test utts +
    stage 6 jit export. The doc's ``--data`` path keeps the downloaded
    dataset on the shared cache disk, so repeat runs skip stage -1.
    """

    # 6 h per command: the doc runs the full aishell-1 flow per labeled
    # block -- the stage -1 download alone pulls ~15.6 GB, and the stage 4
    # train block runs a conformer epoch over 120k utts, plus stage 5
    # decodes 7176 test utts in 4 modes. 1 h was enough for the
    # mock-data era only.
    DEFAULT_COMMAND_TIMEOUT = 21600

    USER_AGENT = 'cosdt-ci-test/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a CANN
    # failure surfaces a full stderr dump (head/tail by default would hide
    # the line that names the failure).
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # typo in CANN's Python driver (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list. The WeNet dep tree
    # (torch / torchaudio / deepspeed / ...) resolves bare ``torch`` wheels
    # from the default index, which drag ``nvidia-*`` in; the constraints
    # keep an accidental re-resolution of the torch stack from breaking the
    # torch_npu pairing.
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
    _CONSTRAINTS_FILE = '/tmp/wenet_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source.
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is tied to the container image pinned by the ``image:`` input of
    # ``wenet-quick-start.yml`` (cann:8.0.0-910b-ubuntu22.04-py3.10).
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # Doc-driven dataset landing zone: the doc's stage -1 fence pins
    # ``local_dir=/root/.cache/huggingface/datasets/aishell1-slr33``, and the
    # NPU runner starts the container as root, so ``Path.home()`` resolves to
    # ``/root`` here. The byte sizes are the ones openslr.org publishes for
    # the two SLR33 archives; the doc's Hugging Face source mirrors them
    # byte-for-byte.
    _DATASET_DIR = (
        Path.home() / '.cache' / 'huggingface' / 'datasets' / 'aishell1-slr33'
    )
    _DATASET_ARCHIVES = {
        'data_aishell.tgz': 15_582_913_665,
        'resource_aishell.tgz': 1_246_920,
    }

    @classmethod
    def prepare_environment(cls) -> None:
        """Source CANN env + write CUDA exclusion list + pin NPU card 0 +
        torch stack probe + safetensors + Hugging Face cache validation.

        Each labeled fence runs in its own subprocess, so this hook is
        what makes the CANN env visible to all of them -- the doc's own
        ``source set_env.sh`` only reaches the fence it sits in.

        The doc's ``## 4. 安装 WeNet 与依赖`` block is the single source of
        truth for the WeNet / torch stack (``pip install -e .[torch-npu]``
        pins torch / torch-npu / torchaudio at 2.2.0); this hook only
        handles the infrastructure bootstrap, reusing the image's
        pre-installed 2.2 wheels when present.
        """
        path_dirs = '/usr/local/sbin:/usr/local/bin'
        current_path = os.environ.get('PATH', '')
        if path_dirs not in current_path:
            os.environ['PATH'] = f'{path_dirs}:{current_path}'

        # 0) CANN env: source set_env.sh and merge the env stream into
        # os.environ, so every later fence inherits it. Each labeled fence
        # is a new subprocess (the framework runs ``bash -c`` / ``python
        # -c``), so the doc's ``source set_env.sh`` -- which only runs
        # inside the equivalent shell fence -- does not persist.
        #
        # Merge is OVERWRITE, not ``setdefault``: this image already ships
        # ``LD_LIBRARY_PATH`` in its own ENV, so ``setdefault`` kept the
        # stale value and silently dropped the increment ``set_env.sh``
        # adds -- leaving the CANN lib dirs out of every fence. The doc's
        # first consumer of them is the ``check-npu`` fence, and it is a
        # ``python`` block, which cannot source a shell script the way
        # upstream's shell-block version did:
        #   ImportError: libhccl.so: cannot open shared object file
        # Overwriting reproduces exactly the env a
        # ``bash -c "source set_env.sh; python -c ..."`` fence sees --
        # same shell, same stream -- which is the shape upstream's guarded
        # doc used, so the two paths agree by construction.
        if not os.path.isfile(cls._CANN_SET_ENV):
            raise RuntimeError(
                f'required Ascend env script missing: {cls._CANN_SET_ENV}'
            )
        merged = subprocess.run(
            ['bash', '-c', f'source {cls._CANN_SET_ENV} >/dev/null 2>&1; env'],
            capture_output=True, text=True, check=True,
        )
        for line in merged.stdout.splitlines():
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            os.environ[key] = value
        print('setup: sourced CANN env from set_env.sh')

        # 1) CUDA exclusion list + process-level env (subprocess.run
        # inherits it).
        with open(cls._CONSTRAINTS_FILE, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(cls._CUDA_CONSTRAINTS) + '\n')
        os.environ['PIP_CONSTRAINT'] = cls._CONSTRAINTS_FILE
        os.environ['UV_CONSTRAINT'] = cls._CONSTRAINTS_FILE

        # Pin the visible NPU card to 0; the runner may expose several
        # davinci devices.
        os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')

        # 2) torch stack probe: reuse the image's pre-installed wheels when
        # the version matches, to avoid a redundant 2.2 download from the
        # cluster cache.
        _PROBE_SCRIPT = (
            'import torch, torch_npu\n'
            "raise SystemExit(0 if "
            "torch.__version__.startswith('2.2') "
            "and torch_npu.__version__.startswith('2.2') "
            "else 1)"
        )
        probe = subprocess.run(
            ['python', '-c', _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            check=False,  # probe's exit code is the branch signal
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
            # Say why before the install fallback. An ``ImportError`` here
            # means the env is broken (a CANN lib the loader cannot find),
            # not that the version mismatched -- installing over it only
            # defers the same failure to the doc's ``check-npu`` fence,
            # which is what made the first CI run read like a torch
            # problem instead of an env one.
            reason = (probe.stderr or probe.stdout or '').strip()
            print('setup: torch probe failed; last lines:')
            for line in reason.splitlines()[-4:]:
                print(f'  {line}')
            print(
                'setup: LD_LIBRARY_PATH='
                f'{os.environ.get("LD_LIBRARY_PATH", "(unset)")}'
            )
            print('setup: installing torch==2.2.0 torch-npu==2.2.0.post2')
            subprocess.run(
                [
                    'python', '-m', 'pip', 'install',
                    '--index-url', cls._CLUSTER_INDEX,
                    '--extra-index-url', cls._ASCEND_EXTRA,
                    'torch==2.2.0', 'torch-npu==2.2.0.post2',
                ],
                check=True,
            )

        # 3) safetensors: native loader used by the cache validation below.
        ensure_safetensors()

        # 4) Hugging Face cache: the doc's ``download-data`` fence pulls the
        # aishell-1 archives (~15.6 GB) into the dataset ``local_dir`` under
        # ``/root/.cache`` (the runner's shared cache disk), which is what
        # lets repeat runs skip stage -1. The purge walks the hub cache's
        # ``models--*`` model dirs and drops the ones whose safetensors
        # header is CONFIRMED broken, so a persistent host-side cache cannot
        # keep feeding truncated shards to the next run.
        hf_cache = resolve_huggingface_cache()
        diagnose_mount_environment(hf_cache)
        purge_huggingface_corrupt(hf_cache)

        # The dataset itself lands in a ``local_dir`` rather than the hub
        # cache, so the purge above cannot see it. A run killed mid-transfer
        # leaves a short ``data_aishell.tgz`` behind, and ``snapshot_download``
        # would resume from bytes that no longer match the hub. Drop any
        # archive whose size disagrees with the published one.
        for name, expected_bytes in cls._DATASET_ARCHIVES.items():
            archive = cls._DATASET_DIR / name
            if not archive.is_file():
                continue
            actual_bytes = archive.stat().st_size
            if actual_bytes == expected_bytes:
                continue
            print(
                f'setup: dropping truncated {name} '
                f'({actual_bytes} != {expected_bytes} bytes); '
                f'huggingface_hub will re-download'
            )
            archive.unlink()

    @classmethod
    def setUpClass(cls) -> None:
        """Run env setup once per class. ``@unittest.skipIf`` only skips the
        test *method* — ``setUpClass`` itself always runs, so the
        ``if _e2e_enabled()`` guard inside keeps heavy setup from firing on
        non-NPU runners.
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
