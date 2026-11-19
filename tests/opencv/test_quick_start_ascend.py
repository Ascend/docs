"""OpenCV zero-patch quick start: end-to-end doc test.

Document under test: ``sources/opencv/quick_start_4x.md`` — the
OpenCV **4.14.0** route that builds ``cv2`` with the CANN backend
(``WITH_CANN=ON``) and **without a single source patch**.

Every ``shell`` / ``python`` code block in the doc carries one of the
``#test`` / ``#test-setup`` / ``#test-result`` labels plus ``id=`` /
``store=`` / ``load='x>>y'`` / ``fuzzy='xxx'`` parameters; this class
fetches the doc, validates that contract, runs the blocks in document
order and compares each ``#test`` output against its ``#test-result``.

What makes this guard different from a plain ``pip install opencv``
smoke test:

* the doc pins OpenCV **4.14.0** and builds it with **zero source
  patches** (``git status --porcelain`` must stay empty), so this guard
  is the regression net for "the CANN backend still compiles untouched
  upstream code" — if a future CANN SDK or OpenCV release breaks that,
  the ``check-source-pristine`` / ``cmake-configure`` blocks fail here
  first;
* the install prefix is **``/usr/local/opencv-cann-414``**: a 4.14.0
  install must never overwrite a patched 5.x install that may live on
  the same self-hosted runner;
* the doc has **no ``<ref>`` placeholder** — the version is pinned in
  the document, so ``UPSTREAM_REF`` is not consumed by the command
  bodies (the engine still injects it; the thin workflow pins
  ``fixed_ref: 4.14.0`` so the monitor watches that tag instead of the
  latest release);
* the heaviest blocks are the source build (Release -O3 with CANN's
  ``all_ops.h`` in the dnn TUs: ~3 GB RSS each) and the CANN/NPU
  MobileNetV2 forward (first forward includes the GE graph compile).

Environment variables (injected by the quick-start engine workflow
``quick-start-template.yml``, triggered by
``opencv-nopatch-quick-start.yml``):
    ``MONITORED_DOC_URL``         Required; raw URL of the document under test.
    ``UPSTREAM_REF``              Injected but unused (doc pins 4.14.0).
    ``NPU_READY=true``            Required, otherwise the class is skipped.
                                  End-to-end tests only run on the NPU runner:
                                  local dev machines / normal ubuntu runners
                                  have no ``/dev/davinci*`` device.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

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
    """``quick_start.md`` end-to-end test: fetch doc -> validate contract ->
    run ``#test-setup`` / ``#test`` in order -> compare against
    ``#test-result``.

    The doc's heaviest steps are:
        1. ``apt-get install`` build deps (cmake / gcc / libjpeg / libpng / libtiff);
        2. ``git clone --depth 1 --branch 4.14.0`` opencv (upstream, untouched);
        3. CANN library layout bridge (3 symlinks into ``<CANN>/lib64``)
           plus the ``nm`` probe that rejects CANN's trimmed
           ``libgraph.so`` (no ``ge::AscendString::operator<``);
        4. ``cmake -DWITH_CANN=ON -DCMAKE_BUILD_TYPE=Release`` configure;
        5. ``cmake --build . --target install`` (Release -O3 — the whole
           point of this variant: no narrow-op-header patch needed, but
           every ``all_ops.h`` TU is heavy: ~3 GB RSS each);
        6. the ``.text`` size assertions (the Release-vs-Debug evidence);
        7. Python binding smoke (cv2 import + the CANN backend registry);
        8. MobileNetV2 ONNX NPU forward (classic DNN engine — 4.14's
           default, no ``OPENCV_FORCE_DNN_ENGINE`` needed) with top-1 = 372;
           the block also raises the log level to INFO and flushes
           ``std::cout`` so the doc can assert three CANN-backend log
           lines (``DNN/CANN: converting ge::Graph to OM buffer`` /
           ``Compile success, model size = ...`` /
           ``DNN/CANN: OM buffer size = ...``) — that is what proves the
           forward really ran on the NPU rather than on the CPU engine
           (both produce top-1 = 372);
           the block builds **two** net instances (CANN/NPU + a plain
           classic/CPU reference) and asserts they agree — same top-1,
           top-1 score within 0.01, same output shape/dtype.  So the doc
           proves both halves: the NPU path was taken (log markers) and it
           computed the right answer (numeric agreement).  The per-forward
           timings are reported but never asserted — the runner's CPU
           differs from any developer's.
        9. videoio smoke (MJPG write 10 frames -> read back 10) — the
           doc's ``BUILD_LIST`` carries ``videoio``, and MJPG rides
           OpenCV's *built-in* MJPEG codec (``modules/videoio/src/mjpeg/``
           + ``cap_mjpeg.cpp``), so it needs no ffmpeg on the runner.
    """

    # The build is the long pole: Release -O3 with CANN's ``all_ops.h``
    # (~1500 ops) in the dnn layer TUs costs ~3 GB RSS and several
    # minutes per TU. 4.14.0 + 6 modules is a small build, and the a2-4
    # runner has far more RAM than the 32 GB cgroup the local
    # verification ran under — hence ``OPENCV_BUILD_JOBS`` below.
    # 9000s per command keeps generous headroom under the engine's job
    # budget for the configure (IPPICV / ADE downloads), the compile
    # itself and the first ACL graph build.
    DEFAULT_COMMAND_TIMEOUT = 9000

    USER_AGENT = 'cosdt-ci-test/quick-start'  # monitored source is the fork under cosdt-ci-test org
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,  # generic [ERROR] + Traceback
        'applicaiton exception',  # CANN toolkit emits this typo (sic) in its Python driver
        'ERR99999',  # CANN sentinel for unrecoverable runtime failure
        # OpenCV's DNN/CANN backend silently falls back to the CPU engine
        # when a layer doesn't support CANN: net_cann.cpp's initBackend()
        # logs this CV_LOG_ONCE_WARNING and then does
        # ``if (!newWasSupported) return;``.  A top-1 of 372 is produced by
        # the CPU engine too (0.9148 vs the NPU's 0.9143), so without this
        # marker the guard cannot tell "ran on the NPU" from "fell back to
        # the CPU".  It is a WARNING-level message, so it is emitted at the
        # default log level and lands on stderr, which the engine scans.
        'Going back to default backend on CPU target',
    )

    # Install prefix of the doc's source build. Distinct from any
    # patched 5.x install (/usr/local/opencv-cann) so a 4.14.0 install
    # can never clobber (or be clobbered by) it when both guards land on
    # the same self-hosted runner.
    _OPENCV_INSTALL = '/usr/local/opencv-cann-414'
    _OPENCV_PYTHONPATH = f'{_OPENCV_INSTALL}/lib/python3.12/site-packages'

    # Build parallelism for the doc's ``cmake --build`` block. The doc
    # defaults to 2 (safe on a 32 GB cgroup); the CI runner has more
    # RAM, so the guard raises it. The doc reads it as
    # ``${OPENCV_BUILD_JOBS:-2}`` — an explicit override, not a silent
    # one.
    _BUILD_JOBS = '6'

    # The ONNX model is 13.9 MB and lives in this repo at
    # tests/opencv/fixtures/. The doc's inference block looks it
    # up under that path (relative to the working dir = repo root), so
    # the CI leg never touches the network for the model.
    _MODEL_NAME = 'mobilenetv2-12.onnx'
    _FIXTURE = Path(__file__).resolve().parent / 'fixtures' / _MODEL_NAME

    # ----------------------------------------------------------
    # prepare_environment
    # ----------------------------------------------------------

    @classmethod
    def prepare_environment(cls) -> None:
        """Set PYTHONPATH to the source-built cv2 and raise the build
        parallelism.

        Each ``#test`` / ``#test-setup`` block is its own
        ``subprocess.run(env=env)`` call and the env dict is captured
        once at the top of ``MarkdownDocTestBase.execute``, so a
        doc-level ``export PYTHONPATH=...`` would only live for that one
        subprocess. We set it once on ``os.environ`` here and the base
        class copies it into every block.
        """
        # 1) PYTHONPATH: PREPEND the source-built OpenCV site-packages
        # so the doc's Python blocks import the cv2 with the CANN
        # backend. PREPEND, not setdefault: the CANN image itself
        # exports PYTHONPATH=/usr/local/Ascend/cann-9.x/python/
        # site-packages:... (its TBE / ACL python bits), and setdefault
        # would silently keep the image value — the GE graph compile in
        # the DNN block then dies with "There is no valid so about
        # OpsKernelInfoStore or GraphOptimizer".
        _existing = os.environ.get('PYTHONPATH', '')
        if cls._OPENCV_PYTHONPATH not in _existing:
            os.environ['PYTHONPATH'] = (
                f'{cls._OPENCV_PYTHONPATH}:{_existing}' if _existing
                else cls._OPENCV_PYTHONPATH
            )
        print(
            f'setup: PYTHONPATH -> {os.environ["PYTHONPATH"]} '
            '(prepended source-built cv2 4.14.0; kept image entries)'
        )

        # 2) Build parallelism (see _BUILD_JOBS).
        os.environ['OPENCV_BUILD_JOBS'] = cls._BUILD_JOBS
        print(f'setup: OPENCV_BUILD_JOBS -> {cls._BUILD_JOBS}')

        # 3) The ONNX fixture must be in the checkout: the doc's
        # inference block copies it from the repo instead of reaching
        # out to github raw, which is flaky from the CI network.
        if not cls._FIXTURE.is_file():
            raise unittest.SkipTest(
                f'fixture {cls._FIXTURE} missing — cannot run the NPU leg offline'
            )
        print(f'setup: fixture -> {cls._FIXTURE} ({cls._FIXTURE.stat().st_size} bytes)')

        # 4) mistune is installed by the engine's test step; nothing to
        # do here. OpenCV itself is intentionally NOT pre-installed:
        # the doc's own blocks clone + configure + build + install it, so
        # a broken source build surfaces as a mismatch in the doc's own
        # blocks rather than being masked by a pre-built copy.

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
