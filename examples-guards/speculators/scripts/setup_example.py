#!/usr/bin/env python3
"""Install the NPU stack and speculators for one manifest profile.

Unknown profiles are rejected by setup_example.sh before this runs.
Image vllm-ascend:v0.23.0 already has the torch / vllm stack.
ensure_vllm_npu_stack installs that pair only when it is missing.
"""
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ASCEND_MIRROR_PIP_INDEX = 'https://mirrors.huaweicloud.com/ascend/repos/pypi'
ASCEND_VARIANT_PIP_INDEX = 'https://mirrors.huaweicloud.com/ascend/repos/pypi/variant'
FALLBACK_PIP_INDEX = 'https://mirrors.aliyun.com/pypi/simple'
CLUSTER_PIP_HOST = 'cache-service.nginx-pypi-cache.svc.cluster.local'
CLUSTER_PIP_INDEX = f'http://{CLUSTER_PIP_HOST}/pypi/simple'
GH_PROXY = 'https://gh-proxy.test.osinfra.cn'
VLLM_REF = 'v0.23.0'
VLLM_ORIGIN = 'https://github.com/vllm-project/vllm.git'
STACK_PREFIXES = {
    'torch': '2.10.0',
    'torch-npu': '2.10.0',
    'vllm': '0.23.0',
    'vllm-ascend': '0.23.0',
}
GUIDELLM_DEPS = (
    'click>=8.1',
    'culsans~=0.10.0',
    'eval_type_backport',
    'faker',
    'ftfy>=6.0.0',
    'httpx[http2]<1.0.0',
    'sanic',
    'tabulate',
    'uvloop>=0.18',
    'more-itertools>=10.8.0',
    'websockets>=13.0',
)
SPECULATORS_DEPS = (
    'aiohttp',
    'click',
    'datasets>=4.0.0,<=5.0.1',
    'httpx',
    'huggingface-hub',
    'loguru>=0.7.2,<=0.7.3',
    'openai',
    'protobuf',
    'psutil',
    'pydantic>=2.0.0',
    'pydantic-settings>=2.0.0',
    'rich',
    'safetensors',
    'tqdm>=4.66.3,<=4.70.0',
    'typer>=0.12.0',
)


def install_cpu_wheel(dist: str, version: str) -> None:
    abi = f'cp{sys.version_info.major}{sys.version_info.minor}'
    url = (
        'https://download.pytorch.org/whl/cpu/'
        f'{dist}-{version}%2Bcpu-{abi}-{abi}-manylinux_2_28_aarch64.whl'
    )
    pip_install('--no-deps', url)


def pip_install(*args: str, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', *args],
        check=True,
        env=env,
    )


def select_pip_index() -> None:
    try:
        urllib.request.urlopen(CLUSTER_PIP_INDEX, timeout=3)
    except urllib.error.HTTPError:
        pass
    except OSError:
        os.environ['PIP_INDEX_URL'] = FALLBACK_PIP_INDEX
        os.environ.pop('PIP_TRUSTED_HOST', None)
        print(f'pip index: {os.environ["PIP_INDEX_URL"]}')
        return
    os.environ['PIP_INDEX_URL'] = CLUSTER_PIP_INDEX
    os.environ['PIP_TRUSTED_HOST'] = CLUSTER_PIP_HOST
    print(f'pip index: {os.environ["PIP_INDEX_URL"]}')


def clone_vllm(dest: Path) -> None:
    if (dest / '.git').is_dir():
        print(f'reusing vllm checkout {dest}')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['GIT_TERMINAL_PROMPT'] = '0'
    env['GIT_HTTP_VERSION'] = 'HTTP/1.1'
    for src in (VLLM_ORIGIN, f'{GH_PROXY}/{VLLM_ORIGIN}'):
        for _ in range(3):
            if dest.exists():
                shutil.rmtree(dest)
            result = subprocess.run(
                ['git', 'clone', '--depth', '1', '--branch', VLLM_REF, src, str(dest)],
                env=env,
                check=False,
            )
            if result.returncode == 0:
                return
            shutil.rmtree(dest, ignore_errors=True)
            time.sleep(5)
    raise SystemExit(f'failed to clone vllm {VLLM_REF}')


def stack_ready() -> bool:
    try:
        for name, prefix in STACK_PREFIXES.items():
            ver = metadata.version(name)
            print('found', name, ver)
            if not ver.startswith(prefix):
                return False
        return True
    except metadata.PackageNotFoundError:
        return False


def ensure_vllm_npu_stack(target_root: Path) -> None:
    if stack_ready():
        print('reusing existing torch 2.10 / vllm 0.23 / vllm-ascend stack')
        return
    print(
        'installing torch==2.10.0+cpu torch-npu==2.10.0.post4 '
        'torchvision==0.25.0+cpu torchaudio==2.10.0+cpu triton-ascend==3.2.2'
    )
    # PyPI hosts the CUDA torch wheel. The CPU build is the +cpu local
    # version published only on download.pytorch.org.
    install_cpu_wheel('torch', '2.10.0')
    install_cpu_wheel('torchvision', '0.25.0')
    install_cpu_wheel('torchaudio', '2.10.0')
    pip_install(
        'filelock', 'typing-extensions>=4.10.0', 'sympy>=1.13.3',
        'networkx>=2.5.1', 'jinja2', 'fsspec>=0.8.5', 'pillow',
        'numpy==1.26.4',
    )
    pip_install(
        '--extra-index-url', ASCEND_VARIANT_PIP_INDEX,
        '--extra-index-url', ASCEND_MIRROR_PIP_INDEX,
        '--find-links', 'https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/',
        'torch-npu==2.10.0.post4',
        'triton-ascend==3.2.2',
    )
    pip_install(
        'cmake>=3.26', 'pyyaml', 'nanobind', 'ninja', 'setuptools-rust', 'wheel',
        'setuptools-scm>=8', 'setuptools>=77,<81',
    )
    workspace = Path(os.environ.get('GITHUB_WORKSPACE') or target_root.parent)
    dest = workspace / 'deps' / 'vllm'
    clone_vllm(dest)
    print(f'installing vllm {VLLM_REF} with VLLM_TARGET_DEVICE=empty')
    pip_install('--no-build-isolation', '-e', str(dest), extra_env={'VLLM_TARGET_DEVICE': 'empty'})
    print('installing vllm-ascend==0.23.0')
    pip_install(
        '--no-build-isolation',
        '--extra-index-url', ASCEND_VARIANT_PIP_INDEX,
        '--extra-index-url', ASCEND_MIRROR_PIP_INDEX,
        'vllm-ascend==0.23.0',
    )


def ensure_pip() -> None:
    try:
        importlib.import_module('pip')
    except ImportError:
        subprocess.run([sys.executable, '-m', 'ensurepip', '--upgrade'], check=True)


def py_fresh(code: str) -> None:
    # Editable installs write .pth files that this process will not see.
    subprocess.run([sys.executable, '-c', code], check=True)


def install_speculators(target_root: Path, constraints: Path) -> None:
    ensure_pip()
    pip_install('-U', 'pip', 'wheel')
    ensure_vllm_npu_stack(target_root)
    os.environ['PIP_CONSTRAINT'] = str(constraints)
    print(f'installing speculators from {target_root} (constraint {constraints})')
    pip_install('modelscope==1.39.1', 'huggingface_hub')
    # --no-deps keeps numpy at 1.26.4. speculators metadata allows numpy
    # 2.4, which breaks triton-ascend 3.2.2.
    pip_install('-e', str(target_root), '--no-deps')
    # speculators.train imports FileTransfer from this in-tree package.
    pip_install('-e', str(target_root / 'hs_connectors'))
    py_fresh(
        'from hs_connectors import FileTransfer, HiddenStatesTransfer; '
        'print("hs_connectors ok")'
    )
    pip_install(*SPECULATORS_DEPS)
    pip_install('numpy==1.26.4', 'setuptools>=77,<81')
    py_fresh(
        'import importlib.metadata as m\n'
        'import torch, torch_npu\n'
        'print("speculators", m.version("speculators"))\n'
        'print("torch", torch.__version__, "torch_npu", torch_npu.__version__)\n'
        'print("vllm", m.version("vllm"), "vllm-ascend", m.version("vllm-ascend"))\n'
        'print("numpy", m.version("numpy"), "setuptools", m.version("setuptools"))\n'
        'assert torch.__version__.startswith("2.10.0"), torch.__version__\n'
        'assert torch_npu.__version__.startswith("2.10.0"), torch_npu.__version__\n'
        'print("npu available", torch.npu.is_available(), "count", torch.npu.device_count())\n'
    )


def setup_evaluate() -> None:
    print('installing guidellm for the evaluate example')
    # guidellm metadata wants numpy>=2.0. triton-ascend 3.2.2 requires
    # numpy==1.26.4. Same pin as speculators: --no-deps then restore numpy.
    os.environ.pop('PIP_CONSTRAINT', None)
    pip_install('guidellm~=0.7.1', '--no-deps')
    pip_install(*GUIDELLM_DEPS)
    pip_install('numpy==1.26.4')
    py_fresh(
        'import guidellm, importlib.metadata as m\n'
        'print("guidellm", m.version("guidellm"), "numpy", m.version("numpy"))\n'
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: setup_example.py <profile>')
    profile = sys.argv[1]
    if profile not in ('train', 'evaluate'):
        raise SystemExit(f'unknown profile: {profile} (supported: train evaluate)')
    target_root = Path(os.environ['TARGET_ROOT'])
    project_root = Path(os.environ['PROJECT_ROOT'])
    constraints = project_root / 'constraints-npu.txt'
    select_pip_index()
    install_speculators(target_root, constraints)
    if profile == 'evaluate':
        setup_evaluate()


if __name__ == '__main__':
    main()
