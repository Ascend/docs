"""Guard sources/ollama/quick_start.md against the CANN-enabled Ollama fork.

MONITORED_DOC_URL fetches the document at this run's commit. The engine
resolves the tagless fork's default-branch SHA into UPSTREAM_REF; the fixture
checks that checkout and proves the CANN source path, build products, NPU
device discovery, model import and one completed inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import tarfile
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from doc_test.base import MarkdownDocTestBase, SetupCommand, TestCommand

EXPECTED_ORIGIN = "https://github.com/lltiaor/ollama.git"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_0.gguf"
MODEL_SHA256 = "7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed"
MODEL_SIZE = 428730208
MODEL_NAME = "qwen2.5-ascend"
BASE_URL = "http://127.0.0.1:11434"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def valid_model(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != MODEL_SIZE:
        return False
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            return False
    return sha256(path) == MODEL_SHA256


def _checked_run(argv, *, env, cwd, timeout=120):
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=cwd,
        timeout=timeout,
    )
    if result.returncode:
        raise AssertionError(
            f"command failed (rc={result.returncode}): {argv!r}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _normalized_repo(url: str) -> str:
    return url.strip().rstrip("/").removesuffix(".git").lower()


def _validate_source(repo: Path, upstream_ref: str, env: dict) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", upstream_ref):
        raise AssertionError("UPSTREAM_REF must be the engine-resolved full SHA")
    head = _checked_run(["git", "rev-parse", "HEAD"], env=env, cwd=repo)
    if head != upstream_ref:
        raise AssertionError(f"checkout {head} differs from engine SHA {upstream_ref}")
    origin = _checked_run(
        ["git", "config", "--get", "remote.origin.url"], env=env, cwd=repo
    )
    if _normalized_repo(origin) != _normalized_repo(EXPECTED_ORIGIN):
        raise AssertionError(f"Unexpected Ollama origin: {origin!r}")
    status = _checked_run(["git", "status", "--porcelain"], env=env, cwd=repo)
    if status:
        raise AssertionError(f"Tested Ollama checkout is modified:\n{status}")

    required = {
        "cmake/local.cmake": (
            'elseif(_backend STREQUAL "cann")',
            "TARGETS ggml-cann",
            "-DGGML_CANN=ON",
            "OLLAMA_GPU_BACKEND=cann",
        ),
        "discover/llama_server.go": (
            'strings.HasPrefix(strings.ToUpper(name), "CANN")',
            'return "CANN"',
        ),
        "discover/runner.go": ('os.Getenv("ASCEND_RT_VISIBLE_DEVICES")',),
        "ml/device.go": ('case "CANN":', 'envVar = "ASCEND_RT_VISIBLE_DEVICES"'),
    }
    for relative, markers in required.items():
        text = (repo / relative).read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                raise AssertionError(
                    f"{relative} is missing CANN integration: {marker!r}"
                )

    llama_cpp_ref = (repo / "LLAMA_CPP_VERSION").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"b[0-9]+", llama_cpp_ref):
        raise AssertionError(f"Unexpected LLAMA_CPP_VERSION: {llama_cpp_ref!r}")
    return {"source_commit": head, "llama_cpp_ref": llama_cpp_ref, "origin": origin}


def inference_evidence(response: dict, loaded: dict, log: str) -> dict:
    """Reject empty generation, CPU fallback and partial model offload."""
    allowed = {MODEL_NAME, f"{MODEL_NAME}:latest"}
    if response.get("model") not in allowed or response.get("done") is not True:
        raise AssertionError(f"Unexpected chat response: {response!r}")
    message = response.get("message", {})
    if message.get("role") != "assistant" or not message.get("content", "").strip():
        raise AssertionError("Chat returned no assistant content")
    if response.get("eval_count", 0) <= 0 or response.get("eval_duration", 0) <= 0:
        raise AssertionError("Chat returned no measured token generation")
    models = [item for item in loaded.get("models", []) if item.get("name") in allowed]
    if not models or models[0].get("size_vram", 0) <= 0:
        raise AssertionError("The target model has no device memory in /api/ps")

    starts = list(re.finditer(r"load_tensors:\s+loading model tensors", log))
    latest = log[starts[-1].start() :] if starts else log
    offloads = list(re.finditer(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", latest))
    if not offloads:
        raise AssertionError("No model offload was reported for the latest load")
    offload = offloads[-1]
    used, total = map(int, offload.groups())
    buffers = re.findall(
        r"CANN\d+\s+model buffer size\s*=\s*([\d.]+)\s*MiB",
        latest[offload.end() :],
    )
    if not any(float(value) > 0 for value in buffers):
        raise AssertionError("No positive CANN model buffer was reported")
    if used != total or total <= 0:
        raise AssertionError("No complete model offload was reported")
    return {
        "model": response["model"],
        "eval_count": response["eval_count"],
        "eval_duration": response["eval_duration"],
        "size_vram": models[0]["size_vram"],
        "cann_model_buffer_mib": buffers,
        "offloaded_layers": [used, total],
    }


def ensure_go(cache: Path) -> None:
    """Install the documented Go floor when the runner image has no toolchain."""
    executable = shutil.which("go")
    if executable:
        result = subprocess.run(
            [executable, "version"],
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"go(\d+)\.(\d+)", result.stdout)
        if match and tuple(map(int, match.groups())) >= (1, 26):
            return
    arch = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "amd64"}.get(
        platform.machine()
    )
    if platform.system() != "Linux" or not arch:
        raise RuntimeError("The CANN quick start requires Linux arm64 or amd64")
    version = "go1.26.10"
    target = cache / "toolchains" / f"{version}-{arch}"
    if not (target / "go" / "bin" / "go").is_file():
        with urllib.request.urlopen(
            "https://go.dev/dl/?mode=json&include=all",
            timeout=60,
        ) as response:
            releases = json.load(response)
        package = next(
            item
            for release in releases
            if release["version"] == version
            for item in release["files"]
            if item["os"] == "linux"
            and item["arch"] == arch
            and item["kind"] == "archive"
        )
        archive = cache / "downloads" / package["filename"]
        if not archive.is_file() or sha256(archive) != package["sha256"]:
            partial = archive.with_suffix(".part")
            with (
                urllib.request.urlopen(
                    "https://go.dev/dl/" + package["filename"],
                    timeout=120,
                ) as response,
                partial.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            if sha256(partial) != package["sha256"]:
                partial.unlink()
                raise AssertionError("Go archive checksum mismatch")
            partial.replace(archive)
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as handle:
            handle.extractall(target, filter="data")
    os.environ["PATH"] = str(target / "go" / "bin") + os.pathsep + os.environ["PATH"]


class TestQuickStartAscend(MarkdownDocTestBase, unittest.TestCase):
    DEFAULT_COMMAND_TIMEOUT = 1800
    BUILD_TIMEOUT = 7200
    USER_AGENT = "ascend-docs/ollama-quick-start"
    ERROR_MARKERS = (
        *MarkdownDocTestBase.ERROR_MARKERS,
        "applicaiton exception",  # typo in CANN's Python driver (sic)
        "ERR99999",
    )
    _CANN_SET_ENV = "/usr/local/Ascend/ascend-toolkit/set_env.sh"

    @classmethod
    def prepare_environment(cls) -> None:
        cls._original_cwd = Path.cwd()
        cls._original_env = os.environ.copy()
        cls._work_dir = Path(tempfile.mkdtemp(prefix="ollama-quick-start-"))
        cls.server = None
        cls.server_log = None
        cls._cleaned = False
        cls.ref = os.environ.get("UPSTREAM_REF", "")
        if not re.fullmatch(r"[0-9a-f]{40}", cls.ref):
            raise RuntimeError("UPSTREAM_REF must be the engine-resolved fork SHA")
        cls.addClassCleanup(cls._cleanup_environment)

        cls.cache = Path(os.environ.get("OLLAMA_TEST_CACHE", "/ollama-cache")).resolve()
        for name in (
            "downloads",
            "models",
            "toolchains",
            "go-build",
            "go-mod",
            "ccache",
        ):
            (cls.cache / name).mkdir(parents=True, exist_ok=True)

        if not Path(cls._CANN_SET_ENV).is_file():
            raise RuntimeError(f"CANN environment script missing: {cls._CANN_SET_ENV}")
        merged = subprocess.run(
            ["bash", "-c", f"source {cls._CANN_SET_ENV} >/dev/null 2>&1 && env -0"],
            capture_output=True,
            check=True,
            timeout=120,
        )
        for entry in merged.stdout.split(b"\0"):
            if b"=" in entry:
                key, value = entry.split(b"=", 1)
                os.environ[os.fsdecode(key)] = os.fsdecode(value)
        if not os.environ.get("ASCEND_HOME_PATH"):
            raise AssertionError("CANN did not set ASCEND_HOME_PATH")

        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
        os.environ.pop("OLLAMA_LLAMA_CPP_SOURCE", None)
        os.environ["OLLAMA_HOST"] = BASE_URL
        os.environ["OLLAMA_DEBUG"] = "1"
        os.environ["OLLAMA_KEEP_ALIVE"] = "5m"
        os.environ["OLLAMA_LLM_LIBRARY"] = "cann"
        os.environ["OLLAMA_MODELS"] = str(cls._work_dir / "ollama-models")
        for name in ("NO_PROXY", "no_proxy"):
            os.environ[name] = ",".join(
                filter(
                    None,
                    (
                        os.environ.get(name),
                        "127.0.0.1",
                        "localhost",
                    ),
                )
            )
        os.environ["GOCACHE"] = str(cls.cache / "go-build")
        os.environ["GOMODCACHE"] = str(cls.cache / "go-mod")
        os.environ["CCACHE_DIR"] = str(cls.cache / "ccache")
        os.environ.setdefault("GOPROXY", "https://goproxy.cn,direct")
        os.environ.setdefault("GOSUMDB", "sum.golang.org https://sum.golang.google.cn")
        ensure_go(cls.cache)
        for tool in ("git", "curl", "gcc", "g++", "make"):
            if not shutil.which(tool):
                raise AssertionError(f"Missing build prerequisite: {tool}")

        cached = cls.cache / "downloads" / MODEL_FILE
        if cached.exists() and not valid_model(cached):
            cached.unlink()
            print("setup: removed corrupt cached GGUF")
        if cached.exists():
            (cls._work_dir / MODEL_FILE).symlink_to(cached)
        for blob in (cls.cache / "models" / "blobs").glob("sha256-*"):
            if blob.is_file() and sha256(blob) != blob.name.removeprefix("sha256-"):
                blob.unlink()
                print(f"setup: removed corrupt Ollama blob {blob.name}")
        (cls._work_dir / "ollama-models").symlink_to(
            cls.cache / "models",
            target_is_directory=True,
        )
        (cls._work_dir / "reports").mkdir()
        cls.report = {
            "upstream_ref": cls.ref,
            "source_integration_verified": False,
            "native_build_verified": False,
            "npu_inference_verified": False,
        }
        os.chdir(cls._work_dir)

    @classmethod
    def write_report(cls) -> None:
        body = json.dumps(cls.report, indent=2, ensure_ascii=False)
        (cls._work_dir / "reports" / "ollama-compatibility.json").write_text(
            body + "\n",
            encoding="utf-8",
        )
        print("Ollama verification evidence:\n" + body, flush=True)

    @classmethod
    def _verify_source(cls, env: dict) -> None:
        evidence = _validate_source(cls._work_dir / "ollama", cls.ref, env)
        cls.report.update(evidence)
        cls.report["source_integration_verified"] = True
        cls.write_report()

    @classmethod
    def server_tail(cls) -> str:
        path = cls._work_dir / "reports" / "server.log"
        if not path.exists():
            return "(no server log)"
        return path.read_text(encoding="utf-8", errors="replace")[-16000:]

    @classmethod
    def request(cls, path: str) -> dict:
        if cls.server is None or cls.server.poll() is not None:
            raise AssertionError("Ollama server exited\n" + cls.server_tail())
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(BASE_URL + path, timeout=5) as response:
            return json.load(response)

    @classmethod
    def start_service(cls, command: str, env: dict, cwd: Path) -> None:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", 11434))
        cls.server_log = (cls._work_dir / "reports" / "server.log").open("wb")
        cls.server = subprocess.Popen(
            ["bash", "-c", command],
            env=env,
            cwd=cwd,
            stdout=cls.server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 120
        last_error = ""
        while time.monotonic() < deadline:
            try:
                actual = cls.request("/api/version")
                if actual.get("version") != cls.ref:
                    raise AssertionError(
                        f"Built/server version mismatch: {actual} != {cls.ref}"
                    )
                cls.report["server_version"] = actual["version"]
                cls.write_report()
                return
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
                last_error = str(exc)
            time.sleep(1)
        raise AssertionError(
            f"Ollama not ready after 120s: {last_error}\n{cls.server_tail()}"
        )

    @classmethod
    def _verify_build(cls, env: dict, cwd: Path) -> None:
        source = cls._work_dir / "ollama"
        payload = source / "build" / "lib" / "ollama"
        products = (
            source / "ollama",
            payload / "cann" / "libggml-cann.so",
            payload / "llama-server",
        )
        for path in products:
            if not path.is_file():
                raise AssertionError(f"Missing build product: {path}")

        probe_env = env.copy()
        probe_env["GGML_BACKEND_PATH"] = str(payload / "cann" / "libggml-cann.so")
        probe_env["LD_LIBRARY_PATH"] = ":".join(
            filter(
                None,
                (
                    str(payload),
                    str(payload / "cann"),
                    env.get("LD_LIBRARY_PATH"),
                ),
            )
        )
        probe = subprocess.run(
            [str(payload / "llama-server"), "--list-devices", "--offline"],
            env=probe_env,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        output = probe.stdout + probe.stderr
        print("CANN device probe:\n" + output, flush=True)
        if not re.search(r"CANN\d+:", output):
            raise AssertionError("Built runner did not enumerate a CANN device")
        cls.report["build_products"] = [
            str(path.relative_to(source)) for path in products
        ]
        cls.report["cann_device_probe"] = output.strip().splitlines()
        cls.report["native_build_verified"] = True
        cls.write_report()

    def run_command(self, cmd, env, cwd, timeout, language="shell"):
        rc, output, error = super().run_command(cmd, env, cwd, timeout, language)
        if getattr(self, "_active_id", None) == "cli-chat" and rc == 0:
            if not re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", output).strip():
                raise AssertionError("CLI returned no generated content")
        return rc, output, error

    def _run_one(self, cmd, results, env, cwd, timeout, idx):
        if isinstance(cmd, SetupCommand) and "./ollama/ollama serve" in cmd.cmd:
            self.start_service(cmd.cmd, env, cwd)
            return
        if "cmake --build ollama/build " in cmd.cmd:
            timeout = self.BUILD_TIMEOUT
        self._active_id = getattr(cmd, "id", None)
        try:
            super()._run_one(cmd, results, env, cwd, timeout, idx)
            if "git -C ollama checkout " in cmd.cmd:
                self._verify_source(env)
            elif "cmake --build ollama/build " in cmd.cmd:
                self._verify_build(env, cwd)
            elif (
                cmd.language == "python" and f"model = Path('{MODEL_FILE}')" in cmd.cmd
            ):
                model = self._work_dir / MODEL_FILE
                if not valid_model(model):
                    model.unlink(missing_ok=True)
                    raise AssertionError("Downloaded GGUF checksum or size mismatch")
                cached = self.cache / "downloads" / MODEL_FILE
                if not model.is_symlink():
                    shutil.move(str(model), cached)
                    model.symlink_to(cached)
                self.report["model_sha256"] = MODEL_SHA256
                self.write_report()
            elif isinstance(cmd, TestCommand) and cmd.id == "api-chat":
                response = json.loads(
                    (self._work_dir / "chat-response.json").read_text(encoding="utf-8")
                )
                shutil.copyfile(
                    self._work_dir / "chat-response.json",
                    self._work_dir / "reports" / "chat-response.json",
                )
                loaded = self.request("/api/ps")
                (self._work_dir / "reports" / "loaded-models.json").write_text(
                    json.dumps(loaded),
                    encoding="utf-8",
                )
                log = (self._work_dir / "reports" / "server.log").read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                self.report["inference"] = inference_evidence(response, loaded, log)
                self.report["npu_inference_verified"] = True
                self.write_report()
        except Exception:
            print("Ollama service log tail:\n" + self.server_tail(), flush=True)
            raise
        finally:
            self._active_id = None

    @classmethod
    def _cleanup_environment(cls) -> None:
        if getattr(cls, "_cleaned", False):
            return
        cls._cleaned = True
        process = getattr(cls, "server", None)
        try:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        os.killpg(process.pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.2)
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            handle = getattr(cls, "server_log", None)
            if handle is not None:
                handle.close()
            if hasattr(cls, "cache") and (cls._work_dir / "reports").exists():
                saved = cls.cache / "reports" / cls._work_dir.name
                shutil.copytree(cls._work_dir / "reports", saved, dirs_exist_ok=True)
                print(f"Verification reports saved under {saved}", flush=True)
        finally:
            if getattr(cls, "_original_env", None) is not None:
                os.chdir(cls._original_cwd)
                os.environ.clear()
                os.environ.update(cls._original_env)
                shutil.rmtree(cls._work_dir, ignore_errors=True)
                cls._original_env = None

    def post_process(self) -> None:
        self._cleanup_environment()

    @classmethod
    def setUpClass(cls):
        if os.environ.get("NPU_READY", "").strip().lower() == "true":
            cls.prepare_environment()

    @unittest.skipUnless(
        os.environ.get("NPU_READY", "").strip().lower() == "true",
        "end-to-end requires an Ascend NPU runner",
    )
    def test_runs_doc(self):
        self.run_template()


class TestGuardValidation(unittest.TestCase):
    def test_source_ref_must_be_full_sha(self):
        for ref in ("", "main", "d28e26ba"):
            with self.subTest(ref=ref), self.assertRaises(AssertionError):
                _validate_source(Path.cwd(), ref, os.environ.copy())

    def test_inference_evidence_rejects_cpu_and_partial_offload(self):
        response = {
            "model": MODEL_NAME,
            "done": True,
            "message": {"role": "assistant", "content": "hello"},
            "eval_count": 2,
            "eval_duration": 100,
        }
        log = (
            "load_tensors: loading model tensors\n"
            "offloaded 24/24 layers to GPU\n"
            "CANN0 model buffer size = 421.3 MiB\n"
        )
        self.assertEqual(
            inference_evidence(
                response,
                {"models": [{"name": MODEL_NAME, "size_vram": 1024}]},
                log,
            )["offloaded_layers"],
            [24, 24],
        )
        for loaded, changed_log in (
            ({"models": [{"name": MODEL_NAME, "size_vram": 0}]}, log),
            (
                {"models": [{"name": MODEL_NAME, "size_vram": 1024}]},
                log.replace("24/24", "12/24"),
            ),
        ):
            with self.subTest(loaded=loaded), self.assertRaises(AssertionError):
                inference_evidence(response, loaded, changed_log)


if __name__ == "__main__":
    unittest.main()
