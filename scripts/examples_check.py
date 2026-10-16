#!/usr/bin/env python3
"""Daily reconciliation of upstream example trees against examples_manifest.yaml.

Design: docs/examples-check-design.md (single spec source, §2.6).

Per project (source of truth = the thin workflow that actually runs, never
projects.yaml): resolve the reconciliation ref exactly like the examples
engine does (latest release, falling back to the workflow's default_branch;
split-mode examples repos always follow their default branch), fetch the
repo tree via the GitHub Trees API (retry once, then error), materialize a
path skeleton, replay the manifest's scan section, and reconcile:

    covering = supported paths (minus source: project) + unsupported
               strings + scan.exclude     (one literal-path rule for all)
    missing  = scanned units covered by nothing   -> failed (red)
    stale    = covering entries matching nothing  -> info only

Statuses: success / failed / error (config | fetch-failed |
scan-root-missing) / skipped (no scan root declared). Exit code: 0 when
every project is success/skipped, 1 when any failed/error, 2 on fatal
usage/registry problems.

Requires PyYAML; curl and git for the fetch layer.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

API_BASE = 'https://api.github.com'
TEMPLATE_USES = 'examples-template.yml'
DEFAULT_DIR_MARKER = 'CMakeLists.txt'
DEFAULT_MAX_DEPTH = 1
SCAN_UNITS = ('files', 'directories', 'mixed')
FETCH_RETRY_WAIT_SECONDS = 10
INTER_PROJECT_PAUSE_SECONDS = 0.5
HTTP_TIMEOUT_SECONDS = 60
CLONE_TIMEOUT_SECONDS = 300
MISSING_ANNOTATION_LIMIT = 20


class FatalError(Exception):
    """Usage / registry level failure -> exit 2."""


class ConfigError(Exception):
    """Manifest or workflow data problem for one project -> error(config)."""


class FetchError(Exception):
    """Upstream fetch failed after retry -> error(fetch-failed)."""


class ScanRootMissing(Exception):
    """Declared scan root absent upstream -> error(scan-root-missing)."""


# ---------------------------------------------------------------------------
# thin workflow read model (§3.2)
# ---------------------------------------------------------------------------

def load_thin_workflows(workflows_dir: Path) -> dict[str, dict]:
    """Map with.project -> engine inputs for every workflow calling the template."""
    projects: dict[str, dict] = {}
    for wf in sorted(workflows_dir.glob('*.yml')):
        if wf.name == 'examples-template.yml':
            continue
        try:
            data = yaml.safe_load(wf.read_text(encoding='utf-8')) or {}
        except yaml.YAMLError as exc:
            raise FatalError(f'{wf.name}: unparseable YAML: {exc}') from exc
        for job in (data.get('jobs') or {}).values():
            uses = (job or {}).get('uses') or ''
            if TEMPLATE_USES not in str(uses):
                continue
            with_inputs = job.get('with') or {}
            name = with_inputs.get('project')
            if not name or not isinstance(name, str):
                raise FatalError(
                    f'{wf.name}: calls {TEMPLATE_USES} without with.project')
            if name in projects:
                raise FatalError(f'{wf.name}: duplicate with.project {name!r}')
            projects[name] = {
                'workflow': wf.name,
                'upstream_repo': with_inputs.get('upstream_repo'),
                'examples_repo': with_inputs.get('examples_repo') or '',
                'default_branch': with_inputs.get('default_branch') or 'main',
            }
    return projects


# ---------------------------------------------------------------------------
# manifest loading and scan config (§2.6 spec)
# ---------------------------------------------------------------------------

def _normalize_extensions(value) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError('include_extensions must be a non-empty list')
    exts = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError('include_extensions items must be non-empty strings')
        ext = item.strip()
        exts.append(ext if ext.startswith('.') else f'.{ext}')
    return tuple(exts)


def build_scan_config(scan: dict) -> dict | None:
    """Translate the scan section; None means no tree scan declared (skipped)."""
    if not isinstance(scan, dict) or 'root' not in scan:
        return None
    root = scan['root']
    if not isinstance(root, str):
        raise ConfigError('scan.root must be a string')
    unit = scan.get('unit') or 'files'
    if unit not in SCAN_UNITS:
        raise ConfigError(f'scan.unit must be one of {SCAN_UNITS}, got {unit!r}')
    max_depth = scan.get('max_depth', DEFAULT_MAX_DEPTH)
    if (not isinstance(max_depth, int) or isinstance(max_depth, bool)
            or max_depth < 1):
        raise ConfigError(f'scan.max_depth must be an integer >= 1, got {max_depth!r}')
    marker = scan.get('marker')
    if unit == 'directories':
        if marker is None:
            marker = DEFAULT_DIR_MARKER
        if not isinstance(marker, str) or not marker.strip():
            raise ConfigError('scan.marker must be a non-empty string')
    elif unit == 'mixed':
        if marker is None:
            marker = ''
        if not isinstance(marker, str):
            raise ConfigError('scan.marker must be a string or empty')
    else:
        marker = None
    extensions: tuple[str, ...] | None = None
    if unit in ('files', 'mixed'):
        extensions = _normalize_extensions(scan.get('include_extensions'))
    exclude = scan.get('exclude') or []
    if not isinstance(exclude, list) or not all(
            isinstance(item, str) for item in exclude):
        raise ConfigError('scan.exclude must be a list of strings')
    return {
        'root': root,
        'unit': unit,
        'marker': marker,
        'max_depth': max_depth,
        'include_extensions': extensions or (),
        'exclude': exclude,
    }


def load_manifest(path: Path) -> dict:
    """Load one manifest; ConfigError on any data problem (§3.3 row 1)."""
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f'unparseable YAML: {exc}') from exc
    if not isinstance(data, dict):
        raise ConfigError('manifest must be a mapping')
    supported = data.get('supported') or []
    if not isinstance(supported, list):
        raise ConfigError('supported must be a list')
    recorded: list[str] = []
    for entry in supported:
        if not isinstance(entry, dict):
            raise ConfigError('supported entries must be mappings')
        if entry.get('source', 'upstream') == 'project':
            continue
        path_value = entry.get('path')
        if not isinstance(path_value, str) or not path_value.strip():
            raise ConfigError('supported entries must carry a path')
        recorded.append(path_value)
    unsupported = data.get('unsupported') or []
    if not isinstance(unsupported, list) or not all(
            isinstance(item, str) for item in unsupported):
        raise ConfigError('unsupported must be a list of strings')
    recorded.extend(unsupported)
    scan = data.get('scan')
    paths_only = isinstance(scan, dict) and 'paths' in scan and 'root' not in scan
    return {
        'scan_config': build_scan_config(scan if isinstance(scan, dict) else None),
        'paths_only': paths_only,
        'recorded': recorded,
    }


# ---------------------------------------------------------------------------
# scan replay (§2.3: skeleton + spec; files unit ignores max_depth per the
# historical bootstrap semantics, §2.6 row 12)
# ---------------------------------------------------------------------------

def _dir_units(examples_root: Path, repo_root: Path, marker: str,
               max_depth: int) -> list[str]:
    found: list[str] = []

    def visit(directory: Path, depth: int) -> None:
        if depth >= max_depth:
            return
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if not child.is_dir():
                continue
            if not marker or (child / marker).is_file():
                found.append(child.relative_to(repo_root).as_posix())
            visit(child, depth + 1)

    visit(examples_root, 0)
    return found


def _file_units(examples_root: Path, repo_root: Path,
                extensions: tuple[str, ...],
                max_depth: int | None) -> list[str]:
    found: list[str] = []
    for path in sorted(examples_root.rglob('*')):
        if not path.is_file() or path.suffix not in extensions:
            continue
        if (max_depth is not None
                and len(path.relative_to(examples_root).parts) > max_depth):
            continue
        found.append(path.relative_to(repo_root).as_posix())
    return found


def scan_units(repo_root: Path, cfg: dict) -> list[str]:
    examples_root = repo_root / cfg['root']
    if not examples_root.is_dir():
        raise ScanRootMissing(f"{cfg['root'] or '.'}/ not found under {repo_root}")
    unit = cfg['unit']
    if unit == 'directories':
        found = _dir_units(examples_root, repo_root, cfg['marker'],
                           cfg['max_depth'])
    elif unit == 'mixed':
        found = _dir_units(examples_root, repo_root, cfg['marker'],
                           cfg['max_depth'])
        found.extend(_file_units(examples_root, repo_root,
                                 cfg['include_extensions'], cfg['max_depth']))
    else:
        found = _file_units(examples_root, repo_root, cfg['include_extensions'],
                            None)
    return sorted(set(found))


def materialize_skeleton(paths: list[str], root: Path) -> None:
    for relative in paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text('', encoding='utf-8')


# ---------------------------------------------------------------------------
# reconciliation (§2.4)
# ---------------------------------------------------------------------------

def covered(path: str, covering: list[str]) -> bool:
    for entry in covering:
        norm = entry.rstrip('/')
        if path == norm or path.startswith(norm + '/'):
            return True
    return False


def reconcile(units: list[str], recorded: list[str],
              exclude: list[str]) -> dict:
    covering = list(recorded) + list(exclude)
    missing = [p for p in units if not covered(p, covering)]
    stale = [r for r in covering if not _entry_matches(r, units)]
    return {'missing': missing, 'stale': stale}


def _entry_matches(entry: str, units: list[str]) -> bool:
    norm = entry.rstrip('/')
    return any(p == norm or p.startswith(norm + '/') for p in units)


# ---------------------------------------------------------------------------
# fetch layer (curl; retry once per §2.3; clone fallback on truncation)
# ---------------------------------------------------------------------------

def _http_json(url: str, token: str | None):
    command = ['curl', '-sSL', '--max-time', str(HTTP_TIMEOUT_SECONDS)]
    if token:
        command += ['-H', f'Authorization: Bearer {token}']
    command += ['-H', 'User-Agent: examples-check',
                '-o', '-', '-w', '\n%{http_code}', url]
    try:
        proc = subprocess.run(command, capture_output=True, text=True,
                              timeout=HTTP_TIMEOUT_SECONDS + 10)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise FetchError(f'http request failed: {exc}') from exc
    if proc.returncode != 0:
        raise FetchError(f'curl exited {proc.returncode}')
    body, _, status = proc.stdout.rpartition('\n')
    try:
        payload = json.loads(body) if body.strip() else None
    except json.JSONDecodeError as exc:
        raise FetchError(f'invalid JSON from {url}: {exc}') from exc
    return int(status or 0), payload


def _walk_clone(repo: str, ref: str) -> tuple[list[str], str]:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / 'repo'
        command = ['git', 'clone', '--quiet', '--depth', '1',
                   '--filter=blob:none', '--branch', ref,
                   f'https://github.com/{repo}.git', str(target)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True,
                           timeout=CLONE_TIMEOUT_SECONDS)
            sha_proc = subprocess.run(
                ['git', '-C', str(target), 'rev-parse', 'HEAD^{tree}'],
                check=True, capture_output=True, text=True, timeout=30)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                OSError) as exc:
            raise FetchError(f'clone fallback failed: {exc}') from exc
        paths = [p.relative_to(target).as_posix()
                 for p in sorted(target.rglob('*')) if p.is_file()]
        return paths, sha_proc.stdout.strip()


def fetch_tree(repo: str, default_branch: str,
               token: str | None, prefer_release: bool = True) -> dict:
    """Resolve the ref like the engine does and return the blob path list.

    Split-mode examples repos have no releases and always follow their own
    default branch (examples-template.yml checks them out without a ref),
    so prefer_release=False resolves the actual default branch from the
    repo metadata instead of trusting the workflow's default_branch input.
    """
    ref = default_branch
    ref_source = 'default-branch'
    if prefer_release:
        status, payload = _http_json(f'{API_BASE}/repos/{repo}/releases/latest',
                                     token)
        if status == 200 and isinstance(payload, dict) and payload.get('tag_name'):
            ref = str(payload['tag_name'])
            ref_source = 'release'
        elif status != 404:
            raise FetchError(f'releases/latest returned HTTP {status}')
    else:
        status, payload = _http_json(f'{API_BASE}/repos/{repo}', token)
        if status != 200 or not isinstance(payload, dict) \
                or not payload.get('default_branch'):
            raise FetchError(f'repos/{repo} returned HTTP {status}')
        ref = str(payload['default_branch'])
    status, payload = _http_json(
        f'{API_BASE}/repos/{repo}/git/trees/{ref}?recursive=1', token)
    if status != 200 or not isinstance(payload, dict):
        raise FetchError(f'trees/{ref} returned HTTP {status}')
    if payload.get('truncated'):
        paths, tree_sha = _walk_clone(repo, ref)
        return {'paths': paths, 'ref': ref, 'ref_source': ref_source,
                'tree_sha': tree_sha}
    paths = [entry['path'] for entry in payload.get('tree') or []
             if entry.get('type') == 'blob']
    return {'paths': paths, 'ref': ref, 'ref_source': ref_source,
            'tree_sha': str(payload.get('sha') or '')}


def fetch_tree_with_retry(repo: str, default_branch: str, token: str | None,
                          prefer_release: bool = True) -> dict:
    try:
        return fetch_tree(repo, default_branch, token, prefer_release)
    except FetchError as first:
        time.sleep(FETCH_RETRY_WAIT_SECONDS)
        try:
            return fetch_tree(repo, default_branch, token, prefer_release)
        except FetchError as second:
            raise FetchError(
                f'fetch failed after retry: {first}; {second}') from second


# ---------------------------------------------------------------------------
# per-project check
# ---------------------------------------------------------------------------

def check_project(project: str, info: dict, manifest_path: Path,
                  fetcher) -> dict:
    """Return one projects[] entry; never raises (failures become statuses)."""
    result = {
        'project': project,
        'status': 'error',
        'scan_repo': info.get('examples_repo') or info.get('upstream_repo') or '',
        'target_ref': '',
        'scanned_count': 0,
        'recorded_count': 0,
        'missing_paths': [],
        'stale_paths': [],
        'detail': '',
    }
    try:
        if not info.get('upstream_repo'):
            raise ConfigError('thin workflow lacks with.upstream_repo')
        manifest = load_manifest(manifest_path)
    except ConfigError as exc:
        result['detail'] = f'config: {exc}'
        return result
    recorded = manifest['recorded']
    result['recorded_count'] = len(recorded)
    cfg = manifest['scan_config']
    if cfg is None:
        result['status'] = 'skipped'
        result['detail'] = ('scan.paths allowlist, no tree scan declared'
                            if manifest['paths_only']
                            else 'no scan section, no tree scan declared')
        return result
    try:
        fetched = fetcher(result['scan_repo'], info['default_branch'])
    except FetchError as exc:
        result['detail'] = f'fetch-failed: {exc}'
        return result
    result['target_ref'] = fetched['ref']
    result['ref_source'] = fetched['ref_source']
    result['tree_sha'] = fetched['tree_sha']
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        materialize_skeleton(fetched['paths'], root)
        try:
            units = scan_units(root, cfg)
        except ScanRootMissing as exc:
            result['detail'] = f'scan-root-missing: {exc}'
            return result
    result['scanned_count'] = len(units)
    outcome = reconcile(units, recorded, cfg['exclude'])
    result['missing_paths'] = outcome['missing']
    result['stale_paths'] = outcome['stale']
    if outcome['missing']:
        result['status'] = 'failed'
    else:
        result['status'] = 'success'
    return result


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def render_summary(result: dict) -> str:
    lines = [
        '## examples-check 对账结果',
        '',
        (f"total {len(result['projects'])}: "
         f"success {result['summary']['success']}, "
         f"failed {result['summary']['failed']}, "
         f"error {result['summary']['error']}, "
         f"skipped {result['summary']['skipped']}"),
        '',
        '| project | status | scan_repo@ref | scanned | recorded | missing | stale | detail |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for entry in result['projects']:
        ref = entry.get('target_ref') or '–'
        lines.append(
            f"| {entry['project']} | {entry['status']} "
            f"| {entry['scan_repo']}@{ref} "
            f"| {entry.get('scanned_count', 0)} "
            f"| {entry.get('recorded_count', 0)} "
            f"| {len(entry['missing_paths'])} "
            f"| {len(entry['stale_paths'])} "
            f"| {entry['detail']} |")
    if result.get('not_in_scope'):
        lines.append('')
        lines.append('不在范围（有清单无 thin workflow，看护未上线）：'
                     + ', '.join(result['not_in_scope']))
    return '\n'.join(lines) + '\n'


def emit_annotations(result: dict) -> None:
    for entry in result['projects']:
        if entry['status'] == 'failed':
            preview = ', '.join(entry['missing_paths'][:MISSING_ANNOTATION_LIMIT])
            extra = (len(entry['missing_paths']) - MISSING_ANNOTATION_LIMIT)
            suffix = f' (+{extra} more)' if extra > 0 else ''
            print(f"::error::{entry['project']} missing {len(entry['missing_paths'])}"
                  f' upstream unit(s): {preview}{suffix}')
        elif entry['status'] == 'error':
            print(f"::error::{entry['project']} {entry['detail']}")
        elif entry['status'] == 'skipped':
            print(f"::notice::{entry['project']} skipped: {entry['detail']}")


def write_step_summary(text: str) -> None:
    path = os.environ.get('GITHUB_STEP_SUMMARY')
    if path:
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(text)


def git_ref() -> str:
    try:
        proc = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, timeout=15)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return 'unknown'


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_result(trigger: str, projects: list[dict],
                 not_in_scope: list[str]) -> dict:
    summary = {status: sum(1 for e in projects if e['status'] == status)
               for status in ('success', 'failed', 'error', 'skipped')}
    return {
        'schema_version': 1,
        'trigger': trigger,
        'run_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'workflows_ref': git_ref(),
        'summary': summary,
        'not_in_scope': not_in_scope,
        'projects': projects,
    }


def run(args: argparse.Namespace) -> int:
    workflows_dir = Path(args.workflows_dir)
    projects_root = Path(args.projects_root)
    thin = load_thin_workflows(workflows_dir)
    if not thin:
        raise FatalError(f'no workflow under {workflows_dir} calls the engine')
    token = args.token or os.environ.get('GH_TOKEN') or None

    manifest_projects = {p.parent.name for p in
                         projects_root.glob('*/examples_manifest.yaml')}
    not_in_scope = sorted(manifest_projects - set(thin))

    names = sorted(thin)
    projects: list[dict] = []
    for index, name in enumerate(names):
        if index and token:
            time.sleep(INTER_PROJECT_PAUSE_SECONDS)
        info = thin[name]
        prefer_release = not info.get('examples_repo')
        fetcher = (lambda repo, branch, _t=token, _p=prefer_release:
                   fetch_tree_with_retry(repo, branch, _t, _p))
        manifest_path = projects_root / name / 'examples_manifest.yaml'
        if not manifest_path.is_file():
            projects.append({
                'project': name, 'status': 'error',
                'scan_repo': info.get('examples_repo')
                or info.get('upstream_repo') or '',
                'target_ref': '', 'scanned_count': 0, 'recorded_count': 0,
                'missing_paths': [], 'stale_paths': [],
                'detail': 'config: examples_manifest.yaml not found',
            })
            continue
        projects.append(check_project(name, info, manifest_path, fetcher))

    result = build_result(os.environ.get('GITHUB_EVENT_NAME') or 'manual',
                          projects, not_in_scope)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                      encoding='utf-8')
    summary_text = render_summary(result)
    sys.stdout.write(summary_text)
    emit_annotations(result)
    write_step_summary(summary_text)

    if result['summary']['failed'] or result['summary']['error']:
        return 1
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflows-dir', default='.github/workflows',
                        help='Directory holding the thin example workflows')
    parser.add_argument('--projects-root', default='projects',
                        help='Directory holding projects/<name>/examples_manifest.yaml')
    parser.add_argument('--output', default='result.json',
                        help='Path of the aggregate result.json')
    parser.add_argument('--token', default='',
                        help='GitHub token (defaults to $GH_TOKEN)')
    args = parser.parse_args(argv)
    try:
        raise SystemExit(run(args))
    except FatalError as exc:
        print(f'fatal: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == '__main__':
    main()
