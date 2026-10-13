"""Launch the project-owned text example with the same CLI a user runs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def build_command(project_root: Path, example_path: str, output_dir: Path, overlays: str) -> list[str]:
    if example_path != "example/train_text.py":
        raise ValueError(f"Unsupported FLA example: {example_path}")
    script = (project_root / example_path).resolve()
    if not script.is_relative_to(project_root.resolve()) or not script.is_file():
        raise ValueError("Example must exist inside the project directory")
    items = json.loads(overlays or "[]")
    if items is None:
        items = []
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise ValueError("OVERLAY_ARGS must be a JSON array of CLI strings")
    args = [token for item in items for token in shlex.split(os.path.expandvars(item))]
    return [sys.executable, str(script), *args, "--output-dir", str(output_dir)]


def main() -> None:
    from validate_example import validate_result

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example_path")
    args = parser.parse_args()
    if os.environ.get("EXAMPLE_SOURCE") != "project":
        raise ValueError("This FLA example is project-owned; EXAMPLE_SOURCE must be project")
    target_root = Path(os.environ["TARGET_ROOT"])
    output_dir = Path(os.environ["CI_OUTPUT_DIR"]).resolve() / "text-generation"
    command = build_command(
        Path(os.environ["PROJECT_ROOT"]), args.example_path, output_dir,
        os.environ.get("OVERLAY_ARGS", "[]"),
    )
    print(f"Running FLA example: {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=target_root, check=True)
    report = validate_result(output_dir, target_root)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
