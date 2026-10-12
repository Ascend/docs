#!/usr/bin/env python3
"""Resolve an example's model, and prefetch the datasets it names.

vLLM 0.23's modelscope file listing expects file['Type'] and crashes on
current responses. A filesystem path skips that listing. Speculator
checkpoints also name a verifier hub id in
speculators_config.verifier.name_or_path; that verifier is snapshotted
too and the config is rewritten to the local path.

Datasets stay addressed by the hub id the example already passes to
load_dataset or `hf download`. They are materialized into the shared
cache under that id before the run goes offline.

Never git add/commit/push the target tree.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from hub_cache import materialize

# Preset names from speculators.data_generation.configs.DATASET_CONFIGS.
# Only sharegpt is used by a supported example; the hf: and org/name
# forms below cover the rest of the manifest.
DATASET_PRESETS = {
    'sharegpt': 'Aeala/ShareGPT_Vicuna_unfiltered',
}

MODEL_RE = re.compile(
    r'^MODEL="\$\{MODEL:-([^}]+)\}"$|^MODEL="([^"]+)"$',
    re.M,
)
DATASET_RE = re.compile(
    r'^DATASET="\$\{DATASET:-([^}]+)\}"\s*(?:#.*)?$'
    r'|^DATASET="([^"]+)"\s*(?:#.*)?$',
    re.M,
)
HF_DOWNLOAD_RE = re.compile(r"""hf download ["']([^"'$]+)["']""")


def default_model_id(script: Path) -> str:
    match = MODEL_RE.search(script.read_text())
    if match is None:
        raise SystemExit(f'no MODEL= assignment in {script}')
    return match.group(1) or match.group(2)


def dataset_repo(raw: str) -> str | None:
    text = raw.strip()
    if not text or text.startswith(('/', './')):
        return None
    if text.startswith('hf:'):
        repo = text.removeprefix('hf:').split(':', 1)[0]
        return repo or None
    if text in DATASET_PRESETS:
        return DATASET_PRESETS[text]
    if '/' in text and ' ' not in text:
        return text
    return None


def dataset_repos(script: Path) -> list[str]:
    text = script.read_text()
    repos: list[str] = []
    match = DATASET_RE.search(text)
    if match:
        repo = dataset_repo(match.group(1) or match.group(2))
        if repo:
            repos.append(repo)
    for found in HF_DOWNLOAD_RE.findall(text):
        repo = dataset_repo(found)
        if repo and repo not in repos:
            repos.append(repo)
    return repos


def rewrite_nested_verifier(speculator_dir: Path) -> None:
    cfg_path = speculator_dir / 'config.json'
    if not cfg_path.is_file():
        return
    cfg = json.loads(cfg_path.read_text())
    spec = cfg.get('speculators_config')
    if not isinstance(spec, dict):
        return
    verifier = spec.get('verifier')
    if not isinstance(verifier, dict):
        return
    name = verifier.get('name_or_path')
    if not isinstance(name, str) or not name or name.startswith('/'):
        return
    local = str(materialize(name, 'model'))
    verifier['name_or_path'] = local
    spec['verifier'] = verifier
    cfg['speculators_config'] = spec
    tmp = cfg_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(cfg, indent=2) + '\n')
    os.replace(tmp, cfg_path)
    print(f'rewrote verifier {name} -> {local}', file=sys.stderr)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit('usage: resolve_model.py <patched-example>')
    script = Path(sys.argv[1])
    hub = os.environ.get('MODEL') or default_model_id(script)
    if hub.startswith('/'):
        model_dir = Path(hub)
    else:
        model_dir = materialize(hub, 'model')
        rewrite_nested_verifier(model_dir)
    for repo in dataset_repos(script):
        materialize(repo, 'dataset')
    resolved = str(model_dir)
    if not resolved.startswith('/'):
        raise SystemExit(f'resolved path is not absolute: {resolved}')
    print(resolved)


if __name__ == '__main__':
    main()
