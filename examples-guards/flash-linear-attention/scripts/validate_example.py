"""Check that the text example trained, saved, reloaded, and generated on NPU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def validate_result(output_dir: Path, target_root: Path | None = None) -> dict:
    report = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    if report.get("task") != "character-language-model":
        raise ValueError("Unexpected example task")
    if report.get("device") != "npu" or report.get("backend") != "npu":
        raise ValueError("Training and generation must use the NPU backend")
    if target_root is not None and not Path(report["fla_path"]).resolve().is_relative_to(target_root.resolve()):
        raise ValueError("FLA did not come from the target checkout")
    steps = report.get("steps")
    if type(steps) is not int or steps < 1:
        raise ValueError("No training steps completed")
    for field in ("losses", "grad_norms"):
        values = report.get(field)
        if not isinstance(values, list) or len(values) != steps or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Expected one finite {field} value per training step")
    initial, final = report["initial_loss"], report["final_loss"]
    if not math.isfinite(initial) or not math.isfinite(final) or final >= initial:
        raise ValueError("Training loss must be finite and decrease on the same probe batch")
    if report.get("weights_changed") is not True:
        raise ValueError("Training did not update model weights")
    if report.get("checkpoint_roundtrip_ok") is not True:
        raise ValueError("Reloaded checkpoint did not preserve model predictions")
    for name in ("config.json", "model.safetensors", "vocabulary.json"):
        path = output_dir / "checkpoint" / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty checkpoint file: {name}")
    count = report.get("generated_tokens")
    if type(count) is not int or not 0 < count <= report["max_new_tokens"] or not report.get("continuation"):
        raise ValueError("No nonempty text continuation was generated")
    if (output_dir / "generated.txt").read_text(encoding="utf-8") != report["prompt"] + report["continuation"]:
        raise ValueError("Saved generated text does not match the reported continuation")
    print(f"Text example passed: {steps} training steps, loss {initial:.4f} -> {final:.4f}, {count} generated tokens")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--target-root", type=Path)
    args = parser.parse_args()
    validate_result(args.output_dir, args.target_root)


if __name__ == "__main__":
    main()
