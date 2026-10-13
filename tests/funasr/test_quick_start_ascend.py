"""Quick-start documentation test: doc under test is
``sources/funasr/quick_start.md``. End-to-end case built on top of the
``MarkdownDocTestBase`` contract.

Run: ``python -m unittest tests.funasr.test_quick_start_ascend -v 2>&1``

Environment variables (injected by the quick-start engine workflow
``quick-start-template.yml``, triggered by
``funasr-quick-start.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``UPSTREAM_REF``              Required; bash reads ``$UPSTREAM_REF`` to get
                                  the latest release tag. The value is
                                  captured into ``captures`` via the
                                  ``#test-setup store="upstream_ref"`` block's
                                  stdout, then substituted into the doc
                                  command body where ``<ref>`` appears.
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

from doc_test.base import MarkdownDocTestBase


def _is_truthy(value: str | None) -> bool:
    """``'true'`` -> True (case-insensitive); anything else (including unset) -> False."""
    if not value:
        return False
    return value.strip().lower() == 'true'


def _e2e_enabled() -> bool:
    """Return True when ``NPU_READY=true`` is set, releasing the skip."""
    return _is_truthy(os.environ.get('NPU_READY'))


# Single-NPU test runner (``a2-1``): expose card 0 so the doc's
# ``count: 1`` assertion matches. ``setdefault`` lets an outer test
# harness override (e.g. for debugging on a multi-card host).
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    """``Quick-start-Ascend.md`` end-to-end test: fetch doc -> validate
    contract -> run ``#test-setup`` / ``#test`` in order -> compare
    against ``#test-result``.

    Scope: torch / torch_npu / torchaudio install (image wheels reused
    when the version matrix matches) + FunASR source install at the
    latest upstream release tag (``git clone`` + ``uv pip install -e .``
    -- FunASR's ``setup.py`` does NOT declare torch, so the pre-installed
    torch stack is left untouched) + five NPU smokes all executed via
    ``AutoModel(..., device="npu:0")``:

      * paraformer-zh + fsmn-vad + ct-punc + cam++ (production pipeline)
      * iic/SenseVoiceSmall (multilingual ASR + emotion + event tags)
      * paraformer-zh-streaming (chunked streaming)
      * iic/emotion2vec_plus_large (standalone emotion classification)
      * ``funasr`` CLI (`-f json`, parsed downstream to assert fields)

    FunASR natively supports ``device="npu:0"``: ``AutoModel`` resolves
    the device via ``is_npu_available()`` (which ``import torch_npu``s
    and checks ``torch.npu.is_available()``), then places the model with
    ``model.to("npu:0")``. No device-placement workaround is needed --
    unlike modelscope, whose ``verify_device`` rejects ``npu``.

    All models download from ModelScope (default ``hub="ms"``) into
    ``~/.cache/modelscope`` (bind-mounted persistent), so the first run
    pays the download cost and subsequent runs hit cache.
    """

    # Cold run: git clone + ~3.3 GB of model downloads (7 checkpoints)
    # + 5 inferences. Each #test block is individually bounded by
    # DEFAULT_COMMAND_TIMEOUT; 20 min per block leaves room for the
    # heaviest single block (paraformer pipeline, 4 models) on a slow
    # egress link.
    DEFAULT_COMMAND_TIMEOUT = 1200

    # Monitored source is the Ascend/docs repo: the doc lives at
    # sources/funasr/quick_start.md and the engine sets MONITORED_DOC_URL
    # to the raw.githubusercontent.com URL for the same path.
    USER_AGENT = 'cosdt-ci-test/quick-start'

    # Extend the base ERROR_MARKERS with CANN's typo + sentinel so a
    # CANN failure surfaces a full stderr dump.
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        'applicaiton exception',  # CANN toolkit emits this typo (sic)
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
    )

    # Process-level CUDA exclusion list. FunASR's ``uv pip install -e .``
    # pulls transformers / modelscope / librosa etc.; on aarch64 the
    # cluster pip/uv resolver can still pick CUDA-typed transitive wheels
    # (e.g. nvidia-* via a loosely-pinned dep) that fail to run on the
    # NPU. Pre-exclude the whole CUDA toolchain so the resolver falls
    # back to pure-Python / CPU / CANN-typed alternatives.
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
    _CONSTRAINTS_FILE = '/tmp/funasr_npu_constraints.txt'

    # Cluster-internal nginx PyPI cache + Huawei Cloud ascend dual-source.
    _CLUSTER_INDEX = 'http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple'
    _ASCEND_EXTRA = 'https://repo.huaweicloud.com/ascend/repos/pypi'

    # CANN toolkit: source once to get ASCEND_HOME / LD_LIBRARY_PATH etc.
    # Path is hard-coded, tied to the GitHub workflow container image.
    _CANN_SET_ENV = '/usr/local/Ascend/ascend-toolkit/set_env.sh'

    # ----------------------------------------------------------
    # prepare_environment: CANN env + CUDA constraints + torch stack probe
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Install CANN env + CUDA constraints + torch stack.

        FunASR itself is intentionally NOT pre-installed here: the doc
        body runs ``git clone`` + ``uv pip install -e .`` against the
        upstream release tag injected by the workflow, so the test
        exercises the exact install path users get.

        What this hook does pre-install:
            * CANN env (so torch_npu is importable);
            * CUDA exclusion list (so an accidental ``nvidia-cudnn`` pull
              doesn't shadow the NPU build);
            * torch / torch_npu / torchaudio, only if the image's
              pre-installed wheels don't already match the version matrix.
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
                # keys that are missing.
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

        # 2) uv: the doc body's install step is ``uv pip install -e .``,
        # which gives clean stdout (no pip's "Obtaining/Installing" preamble)
        # so the #test-result fuzzy match against ``funasr xxx`` isn't
        # polluted. Inherit PIP_INDEX_URL / PIP_TRUSTED_HOST from the yml.
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'uv'],
            check=True,
        )

        # 3) torch stack probe: when the version matches the image's
        # pre-installed wheels, reuse them (avoid the cluster cache
        # triggering ``+cpu`` resolution). torchaudio must be present
        # too -- FunASR needs it for the ffmpeg-fallback audio loader.
        _PROBE_SCRIPT = (
            'import torch, torch_npu, torchaudio\n'
            "raise SystemExit(0 if "
            "torch.__version__.startswith('2.9.0') "
            "and torch_npu.__version__.startswith('2.9.0') "
            "and torchaudio.__version__.startswith('2.9.0') "
            "else 1)"
        )
        probe = subprocess.run(
            ['python', '-c', _PROBE_SCRIPT],
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            _VERSIONS_SCRIPT = (
                'import torch, torch_npu, torchaudio; '
                'print(torch.__version__, torch_npu.__version__, torchaudio.__version__)'
            )
            versions = subprocess.run(
                ['python', '-c', _VERSIONS_SCRIPT],
                capture_output=True, text=True, check=True,
            )
            print(f'setup: reusing image torch stack ({versions.stdout.strip()})')
        else:
            print('setup: installing torch==2.9.0 torch_npu==2.9.0.post2 torchaudio==2.9.0')
            subprocess.run(
                [
                    'python', '-m', 'pip', 'install',
                    '--index-url', cls._CLUSTER_INDEX,
                    '--extra-index-url', cls._ASCEND_EXTRA,
                    'torch==2.9.0', 'torch_npu==2.9.0.post2', 'torchaudio==2.9.0',
                ],
                check=True,
            )

    # ----------------------------------------------------------
    # test entry
    # ----------------------------------------------------------

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