"""CI shim for the peft examples; run_example.sh installs it as sitecustomize.py.

One sitecustomize.py, four patches, applied at interpreter startup:
1. CUDA->NPU (transfer_to_npu): most peft examples hardcode
   device="cuda"; torch_npu's transfer_to_npu maps torch.cuda onto npu
   (also covers `torch.device("cuda" if torch.cuda.is_available() else
   "cpu")` device-branch examples such as lora_ga). NOT applied to the
   bitsandbytes examples: run_example.sh sets SKIP_TRANSFER_TO_NPU for
   them.
2. Dataset: the sft example calls load_dataset(path) with a local
   .jsonl fixture and reads BOTH "train" and "test" splits
   (splits="train,test"); datasets 3.x cannot infer a builder from a
   single file path - rewrite local data files to
   load_dataset("<builder>", data_files={"train": path, "test": path}).
3. fp16 neutralizer: several finetuning scripts hardcode
   TrainingArguments(fp16=True, ...). accelerate 1.15's fp16 chain
   (optimizer.step(grad_scaler=scaler)) is incompatible with torch_npu
   amp.GradScaler -> TypeError "Adam.step got unexpected keyword
   argument grad_scaler". No supported entry relies on fp16=True, so
   forcing fp16=False globally is a no-op for every other entry.
4. wandb neutralizer: boft_dreambooth hardcodes wandb_init
   mode="online"; force wandb.init mode="disabled" so it becomes
   wandb's own no-op run without an API key.
"""
import os

# Skipped for the bitsandbytes examples: bnb 0.50.2's 4-bit/8-bit work
# via the default CPU backend, but transfer_to_npu makes
# torch.cuda.is_available()=True which routes bnb to its CUDA backend
# and crashes on torch._C._cuda_getCurrentRawStream (torch is +cpu-built).
if not os.environ.get("SKIP_TRANSFER_TO_NPU"):
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

import datasets as _datasets

_original_load_dataset = _datasets.load_dataset

_BUILDERS = {
    ".json": "json",
    ".jsonl": "json",
    ".json.gz": "json",
    ".csv": "csv",
    ".tsv": "csv",
    ".parquet": "parquet",
    ".txt": "text",
}


def _patched_load_dataset(path, *args, **kwargs):
    if isinstance(path, str) and os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in _BUILDERS and "data_files" not in kwargs:
            name = kwargs.pop("name", None)
            if args and name is None:
                name = args[0]
                args = args[1:]
            if name is not None:
                kwargs["name"] = name
            # Map the single file onto both splits: the sft example
            # loads split="train" and split="test" from the same
            # dataset_name.
            kwargs["data_files"] = {"train": path, "test": path}
            return _original_load_dataset(_BUILDERS[ext], *args, **kwargs)
    return _original_load_dataset(path, *args, **kwargs)


_datasets.load_dataset = _patched_load_dataset

import transformers as _tf

_orig_training_args_init = _tf.TrainingArguments.__init__


def _fp16_neutral_init(self, *args, **kwargs):
    if kwargs.get("fp16"):
        kwargs["fp16"] = False
    _orig_training_args_init(self, *args, **kwargs)


_tf.TrainingArguments.__init__ = _fp16_neutral_init

# wandb neutralizer: boft_dreambooth hardcodes wandb_init
# mode="online" and --report_to wandb; force mode="disabled" so
# wandb.init becomes wandb's own no-op run without an API key.
try:
    import wandb as _wandb

    _real_wandb_init = _wandb.init

    def _init_disabled(*args, **kwargs):
        kwargs["mode"] = "disabled"
        return _real_wandb_init(*args, **kwargs)

    _wandb.init = _init_disabled
except Exception:
    pass
