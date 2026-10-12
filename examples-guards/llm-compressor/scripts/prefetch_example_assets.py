#!/usr/bin/env python3
"""Download models and datasets named by one upstream example.

Does not modify the example. Public repos are fetched from huggingface.co
first. A failure, or a repo that requires a login, falls back to
ModelScope and is published under the Hub id the example already uses.
Gated Llama and Gemma ids are planted from ModelScope first: the Hub
copy 401s without a token, and that 401 is a download failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download, try_to_load_from_cache
from modelscope.hub.snapshot_download import (
    dataset_snapshot_download,
    snapshot_download as ms_snapshot_download,
)
from safetensors import SafetensorError, safe_open

# Hugging Face id -> ModelScope id. Used when the Hub repo is gated.
GATED_MODELSCOPE = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "LLM-Research/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B": "LLM-Research/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B": "LLM-Research/Meta-Llama-3.1-8B",
    "google/gemma-2-2b-it": "LLM-Research/gemma-2-2b-it",
    "google/gemma-2-9b-it": "LLM-Research/gemma-2-9b-it",
    "google/gemma-3-4b-it": "LLM-Research/gemma-3-4b-it",
}

DATASET_ALIASES = {
    "perfectblend": "mlabonne/open-perfectblend",
    "flickr30k": "lmms-lab/flickr30k",
    "flickr": "lmms-lab/flickr30k",
}

PILE_10K = "NeelNanda/pile-10k"

HUB_ROOT = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
STAGING_DIR = "guard-staging"
STAGING_STALE_SECONDS = 24 * 3600
SKIP_DIRS = {".cache", "original"}
SKIP_NAMES = {".mdl", ".msc", ".mv"}


def configure_hub() -> None:
    os.environ.setdefault("MODELSCOPE_CACHE", str(HUB_ROOT.parent / "modelscope"))


def repo_dir(repo_id: str, dataset: bool = False) -> Path:
    kind = "datasets" if dataset else "models"
    return HUB_ROOT / f"{kind}--{repo_id.replace('/', '--')}"


def snapshot_path(repo_id: str, dataset: bool = False) -> Path | None:
    refs = repo_dir(repo_id, dataset) / "refs" / "main"
    if not refs.is_file():
        return None
    # huggingface_hub does not strip this string. A trailing newline misses.
    snap = repo_dir(repo_id, dataset) / "snapshots" / refs.read_text()
    if not snap.is_dir():
        return None
    return snap


def safetensors_header_ok(path: Path) -> bool | None:
    """True when the header parses, False when the file is malformed,
    None when it cannot be read. A read error is not proof of corruption.
    """
    try:
        with safe_open(str(path), framework="numpy") as handle:
            list(handle.keys())
    except SafetensorError:
        return False
    except OSError:
        return None
    return True


def corrupt_shards(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*.safetensors"))
        if safetensors_header_ok(path) is False
    ]


def weight_map_names(snap: Path) -> set[str] | None:
    index = snap / "model.safetensors.index.json"
    if not index.is_file():
        return None
    payload = json.loads(index.read_text(encoding="utf-8"))
    return set(payload.get("weight_map", {}).values())


def has_weights(snap: Path) -> bool:
    """True when the snapshot holds a full weight set, not only config.json.

    A gated Hub response can leave config.json and refs/main behind with
    no shards. Treating that as a hit skips the ModelScope copy, and the
    later offline from_pretrained then fails looking for the weights.
    """
    names = weight_map_names(snap)
    if names:
        return all(
            (snap / name).is_file() and safetensors_header_ok(snap / name) is not False
            for name in names
        )
    single = snap / "model.safetensors"
    if single.is_file():
        return safetensors_header_ok(single) is not False
    if (snap / "pytorch_model.bin").is_file():
        return True
    shards = [path for path in snap.glob("*.safetensors") if path.is_file()]
    if not shards:
        return False
    return all(safetensors_header_ok(path) is not False for path in shards)


def model_ready(model_id: str) -> bool:
    snap = snapshot_path(model_id)
    if snap is None or not (snap / "config.json").is_file():
        return False
    if corrupt_shards(snap):
        return False
    return has_weights(snap)


def drop_weightless(model_id: str) -> None:
    """Remove snapshots that have no weights, and a refs/main that points at one."""
    root = repo_dir(model_id)
    snapshots = root / "snapshots"
    if snapshots.is_dir():
        for snap in list(snapshots.iterdir()):
            if snap.is_dir() and not has_weights(snap):
                print(f"drop weightless snapshot {model_id}: {snap.name}", flush=True)
                shutil.rmtree(snap, ignore_errors=True)
    refs = root / "refs" / "main"
    if not refs.is_file():
        return
    target = snapshots / refs.read_text()
    if not target.is_dir():
        refs.unlink(missing_ok=True)


def purge_corrupt(model_id: str) -> None:
    """Drop a bad shard and the blob it points at.

    snapshot_download copies a blob the cache already holds without
    checking it again, so a damaged blob is reused until it is removed.
    """
    snapshots = repo_dir(model_id) / "snapshots"
    if not snapshots.is_dir():
        return
    for snap in sorted(snapshots.iterdir()):
        if not snap.is_dir():
            continue
        for shard in corrupt_shards(snap):
            blob = Path(os.path.realpath(shard)) if shard.is_symlink() else None
            print(f"purge {model_id}: {shard.relative_to(snap).as_posix()}", flush=True)
            shard.unlink(missing_ok=True)
            if blob is not None and blob != shard:
                blob.unlink(missing_ok=True)


def snap_id(repo_id: str, origin: str) -> str:
    return hashlib.sha1(f"{repo_id}|{origin}".encode()).hexdigest()


def staging_root() -> Path:
    root = HUB_ROOT.parent / STAGING_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def sweep_staging() -> None:
    root = HUB_ROOT.parent / STAGING_DIR
    if not root.is_dir():
        return
    cutoff = time.time() - STAGING_STALE_SECONDS
    for entry in root.iterdir():
        try:
            if entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            continue


def intact(dest: Path, src: Path) -> bool:
    try:
        if dest.stat().st_size != src.stat().st_size:
            return False
    except OSError:
        return False
    if dest.suffix != ".safetensors":
        return True
    return safetensors_header_ok(dest) is not False


def publish(src: Path, repo_id: str, origin: str, dataset: bool = False) -> Path:
    """Hardlink src's files into the Hub snapshot, then point refs/main at it.

    The link target stays when the staging directory is removed. A second
    job reading an intact file is left alone.
    """
    folder = snap_id(repo_id, origin)
    snap = repo_dir(repo_id, dataset) / "snapshots" / folder
    for item in sorted(src.rglob("*")):
        if not item.is_file():
            continue
        rel = item.relative_to(src)
        if rel.parts[0] in SKIP_DIRS or item.name in SKIP_NAMES:
            continue
        dest = snap / rel
        real = item.resolve() if item.is_symlink() else item
        if dest.exists() or dest.is_symlink():
            if intact(dest, real):
                continue
            dest.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(real, dest)
        except OSError:
            shutil.copy2(real, dest)
    refs = repo_dir(repo_id, dataset) / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=refs, prefix=".main-", delete=False) as handle:
        handle.write(folder)
        temp_name = handle.name
    os.replace(temp_name, refs / "main")
    return snap


def require_model_tree(src: Path) -> None:
    if not (src / "config.json").is_file() and not any(src.rglob("config.json")):
        raise RuntimeError(f"download has no config.json: {src}")
    bad = corrupt_shards(src)
    if bad:
        raise RuntimeError(
            "corrupt safetensors downloaded: "
            + ", ".join(path.relative_to(src).as_posix() for path in bad)
        )
    if not has_weights(src):
        raise RuntimeError(f"download has no weight files: {src}")


def flatten_config(src: Path) -> Path:
    """Use the directory that holds config.json when the download nested it."""
    if (src / "config.json").is_file():
        return src
    found = next(src.rglob("config.json"), None)
    if found is None:
        return src
    return found.parent


def download_hf_model(model_id: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=str(dest))
    tree = flatten_config(dest)
    require_model_tree(tree)
    return tree


def download_ms_model(ms_id: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    landed = Path(
        ms_snapshot_download(
            ms_id,
            local_dir=str(dest),
            ignore_file_pattern=["original/*"],
        )
    )
    tree = flatten_config(landed if landed.is_dir() else dest)
    require_model_tree(tree)
    return tree


def install_model(model_id: str, src: Path, origin: str) -> None:
    publish(src, model_id, origin)
    cached = try_to_load_from_cache(model_id, "config.json", cache_dir=str(HUB_ROOT))
    if not cached:
        raise RuntimeError(f"huggingface_hub cannot see planted {model_id}")
    if not model_ready(model_id):
        raise RuntimeError(f"planted {model_id} is missing weights")
    print(f"cached {model_id} via {origin} -> {cached}", flush=True)


def prefetch_model(model_id: str) -> None:
    if model_ready(model_id):
        print(f"cache hit {model_id}", flush=True)
        return
    purge_corrupt(model_id)
    if model_ready(model_id):
        print(f"cache hit {model_id} after purge", flush=True)
        return
    drop_weightless(model_id)
    gated = GATED_MODELSCOPE.get(model_id)
    if gated is not None:
        fetch_gated(model_id, gated)
        return
    fetch_public(model_id)


def fetch_gated(model_id: str, ms_id: str) -> None:
    try:
        fetch_from("modelscope", model_id, lambda dest: download_ms_model(ms_id, dest))
        return
    except Exception as exc:
        print(
            f"modelscope plant failed for {model_id} ({exc}); trying huggingface.co",
            flush=True,
        )
    fetch_from("huggingface", model_id, lambda dest: download_hf_model(model_id, dest))


def fetch_public(model_id: str) -> None:
    try:
        fetch_from("huggingface", model_id, lambda dest: download_hf_model(model_id, dest))
        return
    except Exception as exc:
        print(
            f"huggingface download failed for {model_id} ({exc}); trying modelscope",
            flush=True,
        )
    fetch_from("modelscope", model_id, lambda dest: download_ms_model(model_id, dest))


def fetch_from(origin: str, model_id: str, download) -> None:
    staging = Path(tempfile.mkdtemp(prefix=f"{repo_dir(model_id).name}-", dir=staging_root()))
    try:
        src = download(staging)
        install_model(model_id, src, origin)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def call_load(dataset_id: str, name: str | None, split: str | None, offline: bool) -> None:
    saved = os.environ.get("HF_HUB_OFFLINE")
    if offline:
        # Hub offline only. HF_DATASETS_OFFLINE would skip the planted
        # snapshot and look for an arrow cache that does not exist yet.
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        args = [dataset_id]
        if name:
            args.append(name)
        kwargs = {"split": split} if split else {}
        load_dataset(*args, **kwargs)
    finally:
        if saved is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = saved


def prefetch_dataset(dataset_id: str, name: str | None, split: str | None) -> None:
    try:
        call_load(dataset_id, name, split, offline=False)
    except Exception as exc:
        print(
            f"huggingface dataset failed for {dataset_id} ({exc}); trying modelscope",
            flush=True,
        )
        plant_dataset(dataset_id, name, split)
        return
    print(f"prefetched dataset {dataset_id} {name or '-'} {split or '<all>'}", flush=True)


def plant_dataset(dataset_id: str, name: str | None, split: str | None) -> None:
    staging = Path(
        tempfile.mkdtemp(prefix=f"{repo_dir(dataset_id, dataset=True).name}-", dir=staging_root())
    )
    try:
        landed = Path(dataset_snapshot_download(dataset_id, local_dir=str(staging)))
        src = landed if landed.is_dir() else staging
        files = [path for path in src.rglob("*") if path.is_file() and path.name not in SKIP_NAMES]
        if not files:
            raise RuntimeError(f"modelscope dataset {dataset_id} downloaded no files")
        snap = publish(src, dataset_id, "modelscope", dataset=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    call_load(dataset_id, name, split, offline=True)
    print(f"planted dataset {dataset_id} -> {snap}", flush=True)


def call_bodies(text: str, name: str) -> list[str]:
    key = name + "("
    bodies: list[str] = []
    start = 0
    while True:
        found = text.find(key, start)
        if found < 0:
            break
        index = found + len(key)
        depth = 1
        while index < len(text) and depth:
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
            index += 1
        bodies.append(text[found + len(key) : index - 1])
        start = index
    return bodies


def constants(text: str) -> tuple[dict[str, str], dict[str, str]]:
    strings = dict(re.findall(r"(?m)^([A-Z][A-Z0-9_]*)\s*=\s*[\"']([^\"']+)[\"']\s*$", text))
    ints = dict(re.findall(r"(?m)^([A-Z][A-Z0-9_]*)\s*=\s*(\d+)\s*$", text))
    return strings, ints


def resolve(token: str, strings: dict[str, str], ints: dict[str, str]) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] in "\"'" and token[-1] == token[0]:
        return token[1:-1]
    if token.startswith('f"') or token.startswith("f'"):
        body = token[2:-1]

        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            if name in strings:
                return strings[name]
            if name in ints:
                return ints[name]
            raise KeyError(name)

        return re.sub(r"\{([A-Z][A-Z0-9_]*)\}", repl, body)
    if token in strings:
        return strings[token]
    raise KeyError(token)


def model_ids(text: str) -> list[str]:
    found = re.findall(
        r"(?m)^[ \t]*(?:MODEL_ID|model_id|MODEL_STUB|model_stub|MODEL)\s*=\s*[\"']([^\"']+)[\"']",
        text,
    )
    found.extend(re.findall(r"default\s*=\s*[\"']([^\"']+/[^\"']+)[\"']", text))
    ordered: list[str] = []
    for model_id in found:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", model_id):
            continue
        if model_id not in ordered:
            ordered.append(model_id)
    return ordered


def dataset_requests(text: str) -> list[tuple[str, str | None, str | None]]:
    strings, ints = constants(text)
    requests: list[tuple[str, str | None, str | None]] = []

    def add(dataset_id: str, name: str | None, split: str | None) -> None:
        dataset_id = DATASET_ALIASES.get(dataset_id, dataset_id)
        item = (dataset_id, name, split)
        if item not in requests:
            requests.append(item)

    for body in call_bodies(text, "load_dataset"):
        parts = [part.strip() for part in body.split(",") if part.strip()]
        positional: list[str] = []
        split = None
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                if key.strip() == "split":
                    try:
                        split = resolve(value.strip(), strings, ints)
                    except KeyError:
                        split = None
                continue
            try:
                positional.append(resolve(part, strings, ints))
            except KeyError:
                continue
        if not positional:
            continue
        name = positional[1] if len(positional) > 1 else None
        if split is None and "DATASET_SPLIT" in strings and ints.get("NUM_CALIBRATION_SAMPLES"):
            split = f"{strings['DATASET_SPLIT']}[:{ints['NUM_CALIBRATION_SAMPLES']}]"
        if split is None:
            continue
        add(positional[0], name, split)

    counts = {int(item) for item in re.findall(r"num_calibration_samples\s*=\s*(\d+)", text)}
    if "NUM_CALIBRATION_SAMPLES" in ints:
        counts.add(int(ints["NUM_CALIBRATION_SAMPLES"]))
    if not counts:
        counts.add(512)
    for body in call_bodies(text, "oneshot"):
        for match in re.finditer(r"dataset\s*=\s*([\"'][^\"']+[\"']|[A-Z][A-Z0-9_]*)", body):
            try:
                dataset_id = resolve(match.group(1), strings, ints)
            except KeyError:
                continue
            if dataset_id not in DATASET_ALIASES and "/" not in dataset_id:
                continue
            split_match = re.search(r"splits\s*=\s*([^,\n]+)", body)
            resolved_split = None
            if split_match is not None:
                try:
                    resolved_split = resolve(split_match.group(1).strip(), strings, ints)
                except KeyError:
                    resolved_split = None
            hub_id = DATASET_ALIASES.get(dataset_id, dataset_id)
            if resolved_split:
                add(dataset_id, None, resolved_split)
            elif "flickr30k" in hub_id:
                for count in sorted(counts):
                    add(dataset_id, None, f"test[:{count}]")
            else:
                for count in sorted(counts):
                    add(dataset_id, None, f"train[:{count}]")

    if "get_dataset(" in text:
        add(PILE_10K, None, "train")
    return requests


def prefetch_example(path: Path, dry_run: bool) -> None:
    text = path.read_text(encoding="utf-8")
    models = model_ids(text)
    datasets = dataset_requests(text)
    print(f"example {path}")
    print(f"models {models}")
    print(f"datasets {datasets}")
    if dry_run:
        return
    configure_hub()
    sweep_staging()
    for model_id in models:
        prefetch_model(model_id)
    for dataset_id, name, split in datasets:
        prefetch_dataset(dataset_id, name, split)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: prefetch_example_assets.py <example.py> [--dry-run]\n"
            "       prefetch_example_assets.py --model HF_ID",
            file=sys.stderr,
        )
        return 2
    if argv[0] == "--model":
        if len(argv) != 2:
            print("usage: prefetch_example_assets.py --model HF_ID", file=sys.stderr)
            return 2
        configure_hub()
        sweep_staging()
        prefetch_model(argv[1])
        return 0
    path = Path(argv[0])
    if not path.is_file():
        print(f"example not found: {path}", file=sys.stderr)
        return 1
    prefetch_example(path, "--dry-run" in argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
