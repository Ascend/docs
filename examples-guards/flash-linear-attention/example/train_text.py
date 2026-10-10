"""Train a small FLA language model on local text, save it, and generate text."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def build_vocabulary(text: str) -> list[str]:
    if not text:
        raise ValueError("Training text must not be empty")
    return ["<pad>", "<bos>", "<eos>", *sorted(set(text))]


def encode(text: str, vocabulary: list[str]) -> list[int]:
    lookup = {character: index for index, character in enumerate(vocabulary)}
    try:
        return [lookup[character] for character in text]
    except KeyError as error:
        raise ValueError(f"Prompt character {error.args[0]!r} is absent from the training text") from error


def decode(token_ids: list[int], vocabulary: list[str]) -> str:
    return "".join(vocabulary[index] for index in token_ids if index >= 3)


def make_windows(token_ids: list[int], sequence_length: int) -> list[list[int]]:
    if sequence_length < 2 or len(token_ids) < sequence_length:
        raise ValueError("Training text must contain at least one sequence of length >= 2")
    return [
        token_ids[start:start + sequence_length]
        for start in range(0, len(token_ids) - sequence_length + 1, sequence_length)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-file", type=Path, default=Path(__file__).with_name("corpus.txt"))
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for the model and generated text")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt", default=None, help="Defaults to the beginning of the training text")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.max_new_tokens) < 1 or args.sequence_length < 2:
        parser.error("steps, batch size, and new tokens must be positive; sequence length must be >= 2")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new directory to preserve previous runs")

    text = args.text_file.read_text(encoding="utf-8")
    vocabulary = build_vocabulary(text)
    windows = make_windows(encode(text, vocabulary), args.sequence_length)
    prompt = args.prompt if args.prompt is not None else text[:32]
    if not prompt:
        parser.error("prompt must not be empty")
    prompt_ids = encode(prompt, vocabulary)

    import torch
    import torch_npu
    import transformers
    import triton

    import fla
    from fla.models import GatedDeltaNetConfig
    from fla.utils import IS_NPU, device_platform
    from transformers import AutoModelForCausalLM

    if not torch.npu.is_available() or not IS_NPU or device_platform != "npu":
        raise RuntimeError("This example requires a working Ascend NPU and FLA NPU backend")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    args.output_dir.mkdir(parents=True)

    # SwiGLU still uses FLA's activation kernel; its output projection is separate.
    # The loss uses torch.nn.CrossEntropyLoss instead of FLA's fused losses.
    config = GatedDeltaNetConfig(
        vocab_size=len(vocabulary),
        hidden_size=256,
        num_hidden_layers=2,
        num_heads=4,
        head_dim=64,
        expand_v=1,
        intermediate_size=512,
        attn_mode="chunk",
        max_position_embeddings=max(args.sequence_length, len(prompt_ids) + args.max_new_tokens),
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        fuse_linear_cross_entropy=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = AutoModelForCausalLM.from_config(config).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0, foreach=False)
    initial_weights = model.get_output_embeddings().weight.detach().cpu().clone()
    probe = torch.tensor([windows[i % len(windows)] for i in range(args.batch_size)], device=device)
    model.eval()
    with torch.no_grad():
        initial_loss = model(input_ids=probe, labels=probe, use_cache=False).loss.float().item()

    losses, grad_norms = [], []
    model.train()
    for step in range(args.steps):
        inputs = torch.tensor(rng.choices(windows, k=args.batch_size), dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        # FLA shifts labels internally for next-token prediction.
        loss = model(input_ids=inputs, labels=inputs, use_cache=False).loss
        if loss.device.type != "npu" or not torch.isfinite(loss).item():
            raise RuntimeError("Expected a finite training loss on the NPU")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
        optimizer.step()
        losses.append(loss.detach().float().item())
        grad_norms.append(norm.detach().float().item())
        print(f"step {step + 1}/{args.steps} loss={losses[-1]:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        output = model(input_ids=probe, labels=probe, use_cache=False)
        final_loss = output.loss.float().item()
        reference_logits = output.logits.detach().cpu()
    weights_changed = not torch.equal(initial_weights, model.get_output_embeddings().weight.detach().cpu())

    checkpoint = args.output_dir / "checkpoint"
    # Move weights to CPU only for checkpoint I/O; all model execution uses NPU.
    model.to("cpu").save_pretrained(checkpoint, safe_serialization=True)
    (checkpoint / "vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")
    del optimizer, model, output
    torch.npu.empty_cache()

    reloaded = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True).to(device=device, dtype=dtype).eval()
    vocabulary = json.loads((checkpoint / "vocabulary.json").read_text(encoding="utf-8"))
    with torch.no_grad():
        reloaded_logits = reloaded(input_ids=probe, use_cache=False).logits.detach().cpu()
        checkpoint_roundtrip_ok = torch.allclose(reference_logits.float(), reloaded_logits.float(), rtol=1e-2, atol=1e-2)
        input_ids = torch.tensor([encode(prompt, vocabulary)], dtype=torch.long, device=device)
        generated = reloaded.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            suppress_tokens=[0, 1],
            pad_token_id=0,
            eos_token_id=2,
        )
    torch.npu.synchronize()
    new_ids = generated[0, input_ids.shape[1]:].tolist()
    continuation = decode(new_ids, vocabulary)
    (args.output_dir / "generated.txt").write_text(prompt + continuation, encoding="utf-8")
    metrics = {
        "task": "character-language-model",
        "device": generated.device.type,
        "backend": device_platform,
        "steps": len(losses),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "vocab_size": len(vocabulary),
        "training_windows": len(windows),
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "losses": losses,
        "grad_norms": grad_norms,
        "weights_changed": weights_changed,
        "checkpoint_roundtrip_ok": checkpoint_roundtrip_ok,
        "prompt": prompt,
        "continuation": continuation,
        "generated_tokens": len(new_ids),
        "max_new_tokens": args.max_new_tokens,
        "fla_path": fla.__file__,
        "versions": {
            "fla": fla.__version__, "torch": torch.__version__,
            "torch_npu": torch_npu.__version__, "triton": triton.__version__,
            "transformers": transformers.__version__,
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Training loss on the same probe batch: {initial_loss:.4f} -> {final_loss:.4f}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Prompt: {prompt!r}\nContinuation: {continuation!r}")


if __name__ == "__main__":
    main()
