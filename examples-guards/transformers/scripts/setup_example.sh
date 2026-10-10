#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
# A profile selects the smallest dependency set needed by one example family;
# it is not a transformers feature flag. Keep unknown profiles failing before
# any package is installed so a manifest typo cannot run a partial setup.
case "$PROFILE" in
  generation)
    DEPS=(accelerate)
    ;;
  glue)
    DEPS=(accelerate datasets evaluate scikit-learn)
    ;;
  small-training)
    DEPS=(accelerate datasets evaluate seqeval sentencepiece tiktoken)
    ;;
  lm)
    DEPS=(accelerate datasets evaluate)
    ;;
  seq2seq)
    DEPS=(accelerate datasets evaluate sacrebleu rouge-score nltk sentencepiece tiktoken)
    ;;
  vision)
    # librosa: datasets 3.6.0 Audio decode_example imports librosa before
    # falling back to anything else; without it every audio batch fetch dies.
    DEPS=(accelerate datasets evaluate librosa pillow scikit-learn soundfile)
    ;;
  speech)
    # jiwer backs the WER/CER metrics of the speech-recognition examples;
    # librosa backs datasets 3.6.0 Audio decode (same reason as vision).
    DEPS=(accelerate datasets evaluate jiwer librosa soundfile)
    ;;
  *)
    echo "unknown profile: $PROFILE (supported: generation glue small-training lm seq2seq vision speech)" >&2
    exit 1
    ;;
esac

: "${TARGET_ROOT:?TARGET_ROOT is required}"
: "${GITHUB_ENV:?GITHUB_ENV is required}"
# Redirect the pip cache onto the shared persistent runner cache the pool
# mounts at ~/.cache/huggingface (docs/examples-guard-engine.md). Model
# snapshots land in the same shared HF cache (HF_HOME/hub) by default.
SHARED_CACHE_ROOT="${HF_HOME:-${HOME}/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SHARED_CACHE_ROOT}/pip}"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_TRUSTED_HOST="repo.huaweicloud.com"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -m pip install -U pip
# The Ascend image normally contains a compatible torch/torch_npu pair. Reuse
# it when possible because these wheels are large; install only on a missing
# or unusable image stack.
if python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.device_count())"; then
  echo "reusing image torch stack"
else
  python -m pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
    torch==2.9.0 torch_npu==2.9.0.post2
fi
# Install the current target checkout. Its dependencies (regex, tokenizers,
# huggingface-hub, safetensors, ...) are resolved by pip from the index; this is
# safe for the NPU torch stack because transformers does not depend on torch_npu
# and its torch requirement is already satisfied by the image build.
python -m pip install -e "$TARGET_ROOT"
# Pin datasets to a single release. The rebuilt 9.1.0 image no longer
# preinstalls datasets, so an unpinned resolve pulls 5.x, and the whole 4.x
# line requires torchcodec for Audio-feature encode AND decode (verified in
# the 4.2/4.3/4.4 wheels), which needs system FFmpeg we don't ship. 3.6.0 is
# the last release whose Audio encode/decode runs on soundfile; it keeps the
# `**` glob data-files semantics the manifest paths were written against and
# accepts the hub 1.x / pyarrow 25 stack transformers main installs. Pinning
# to a single version also removes the pathological pip backtracking surface
# that once walked an unpinned resolve down to the datasets 0.0.9 sdist.
python -m pip install "${DEPS[@]}" "datasets==3.6.0"
# torchvision is only needed by the vision profile. Its wheel pins an exact
# torch== requirement that would upgrade (and break) the image's NPU
# torch/torch_npu stack, so install it without deps; the examples only use
# the transforms / read_image surface, which links nothing NPU-specific.
if [[ "$PROFILE" == "vision" ]]; then
  python -m pip install --no-deps "torchvision==0.24.1"
fi

# Pre-download example model weights from the HuggingFace hub so the examples
# load them from a local snapshot instead of the blocked HF CDN. All repos
# below exist on HF under the same ids, so the whole download path goes
# through huggingface_hub (no ModelScope dependency): runners are
# China-located, so resolve the endpoint to the hf-mirror.com proxy unless
# the engine already set HF_ENDPOINT. hf-mirror cannot proxy the Xet transfer
# hosts, so keep the legacy transfer path forced (mirrors the engine env).
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
python - <<'PY'
import os
from huggingface_hub import snapshot_download

mapping = {
    "DISTILBERT_PATH": "distilbert/distilbert-base-uncased",
    "TINYGPT2_PATH": "sshleifer/tiny-gpt2",
    "TINYMBART_PATH": "sshleifer/tiny-mbart",
}
for env_name, model_id in mapping.items():
    local = snapshot_download(model_id)
    if model_id == "sshleifer/tiny-mbart":
        import json

        cfg_path = os.path.join(local, "config.json")
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        # The snapshot may lack decoder_start_token_id; transformers main no
        # longer falls back to model-class defaults, so the mbart
        # summarize/seq2seq_qa examples raise "Make sure that
        # `config.decoder_start_token_id` is correctly defined". Inject the
        # mbart standard value (eos </s> id) idempotently.
        if cfg.get("decoder_start_token_id") is None:
            cfg["decoder_start_token_id"] = 2
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
    with open(os.environ["GITHUB_ENV"], "a") as fh:
        fh.write(f"{env_name}={local}\n")
PY

# tiny xlnet backs the beam-search QA examples; kept non-fatal so an endpoint
# outage cannot fail the unrelated example jobs.
if TINYXLNET=$(python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download("sshleifer/tiny-xlnet-base-cased"))
PY
); then
  echo "TINYXLNET_PATH=$TINYXLNET" >> "$GITHUB_ENV"
else
  echo "warning: tiny xlnet download via $HF_ENDPOINT failed; beam-search examples will fail" >&2
fi

# Vision/audio-family tiny random models (hf-internal-testing org) carry their
# config + weights + processor/tokenizer files, so the examples can load
# everything from the local snapshot dir. Same non-fatal policy as above.
for pair in \
  "TINYVIT_PATH:hf-internal-testing/tiny-random-ViTModel" \
  "TINYMAE_PATH:hf-internal-testing/tiny-random-ViTMAEModel" \
  "TINYCLIP_PATH:hf-internal-testing/tiny-random-CLIPModel" \
  "TINYWAV2VEC2_PATH:hf-internal-testing/tiny-random-Wav2Vec2Model" \
  "TINYWHISPER_PATH:hf-internal-testing/tiny-random-WhisperForConditionalGeneration"
do
  env_name="${pair%%:*}"
  repo_id="${pair#*:}"
  if local_dir=$(python - "$repo_id" <<'PY'
import sys
from huggingface_hub import snapshot_download

print(snapshot_download(sys.argv[1]))
PY
  ); then
    echo "$env_name=$local_dir" >> "$GITHUB_ENV"
    # run_speech_recognition_ctc*.py treat any local --model_name_or_path as
    # a Trainer checkpoint to resume from ("if os.path.isdir(...):
    # checkpoint = model_args.model_name_or_path"), and transformers main
    # then demands trainer_state.json inside that directory. The tiny
    # snapshots are plain model dirs, so plant a minimal state
    # (global_step 0 = train from scratch) to make the resume path a
    # no-op; extra files in the snapshot are harmless to other consumers.
    if [[ "$repo_id" == "hf-internal-testing/tiny-random-Wav2Vec2Model" ]]; then
      python - "$local_dir" <<'PY'
import json
import os
import sys

# train_batch_size is divided unconditionally by n_gpu during resume
# validation, and logging/save steps are compared against the run args;
# fill them in so the comparison is a silent no-op. Overwrite any state
# planted by an earlier setup run (persistent HF cache).
state_path = os.path.join(sys.argv[1], "trainer_state.json")
with open(state_path, "w", encoding="utf-8") as fh:
    json.dump(
        {
            "global_step": 0,
            "train_batch_size": 1,
            "logging_steps": 500,
            "save_steps": 500,
            "eval_steps": 500,
        },
        fh,
    )
PY
    fi
  else
    echo "warning: $repo_id download via $HF_ENDPOINT failed; related vision examples will fail" >&2
  fi
done

# LM-family examples infer the dataset loader from the train_file suffix and
# reject the extensionless wiki_text/wiki_00 fixture, so expose it as train.txt
# in the job output dir for the ${CI_OUTPUT_DIR}/train.txt overlay args.
# sentencepiece note: transformers main loads sentencepiece-only vocab files
# (xlnet spiece.model, mbart sentencepiece.bpe.model) via SentencePieceExtractor
# and only falls back to a tiktoken parse when that import/extraction fails, so
# the sentencepiece package must be installed or loading crashes with
# "ValueError: Error parsing line ... in spiece.model".
: "${CI_OUTPUT_DIR:=$GITHUB_WORKSPACE/output}"
mkdir -p "$CI_OUTPUT_DIR"
cp "$TARGET_ROOT/tests/fixtures/tests_samples/wiki_text/wiki_00" "$CI_OUTPUT_DIR/train.txt"

# Trainer-based classification examples map string labels through label_to_id
# inside datasets.map(), but datasets keeps the original string column type and
# casts the ints back to str, so torch_default_data_collator crashes with
# "too many dimensions 'str'". Emit int-label copies of the MRPC fixture for
# the run_glue.py / run_classification.py overlay args.
python - "$TARGET_ROOT" "$CI_OUTPUT_DIR" <<'PY'
import csv
import os
import sys

target, out = sys.argv[1], sys.argv[2]
label2id = {"equivalent": "0", "not_equivalent": "1"}
for split in ("train", "dev"):
    src = os.path.join(target, "tests", "fixtures", "tests_samples", "MRPC", f"{split}.csv")
    with open(src, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)
    with open(os.path.join(out, f"mrpc_{split}.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row["label"] = label2id[row["label"]]
            writer.writerow(row)
PY

# Batch-2 (vision/audio family) data preparation:
# - ImageFolder tree for image classification / MAE / MIM: datasets' imagefolder
#   builder scans <root>/<split>/<label>/*, so mirror the 3 COCO fixture images
#   into a 3-class train/val layout.
# - CLIP: json rows {image_path, captions} referencing the COCO fixture images.
# - Audio classification: silent 1-second wavs laid out as an audiofolder tree
#   (see below); the example unconditionally reads a "validation" split, so
#   emit both train and validation split dirs.
echo "[setup] vision data prep: CI_OUTPUT_DIR=${CI_OUTPUT_DIR:-<unset>} TARGET_ROOT=$TARGET_ROOT"
mkdir -p "$CI_OUTPUT_DIR/imgcls/train" "$CI_OUTPUT_DIR/imgcls/val"
coco_dir="$TARGET_ROOT/tests/fixtures/tests_samples/COCO"
for split in train val; do
  mkdir -p "$CI_OUTPUT_DIR/imgcls/$split/class_a" "$CI_OUTPUT_DIR/imgcls/$split/class_b" "$CI_OUTPUT_DIR/imgcls/$split/class_c"
  cp "$coco_dir/apple.jpg" "$CI_OUTPUT_DIR/imgcls/$split/class_a/apple.jpg"
  cp "$coco_dir/000000039769.png" "$CI_OUTPUT_DIR/imgcls/$split/class_b/000000039769.png"
  cp "$coco_dir/000000004016.png" "$CI_OUTPUT_DIR/imgcls/$split/class_c/000000004016.png"
done
echo "[setup] imagefolder tree:"; ls -R "$CI_OUTPUT_DIR/imgcls" | head -30

python - "$TARGET_ROOT" "$CI_OUTPUT_DIR" <<'PY'
import json
import os
import sys
import wave

target, out = sys.argv[1], sys.argv[2]
coco = os.path.join(target, "tests", "fixtures", "tests_samples", "COCO")

# CLIP: plain string captions; the example auto-detects the first two json
# columns as image_column / caption_column.
clip_rows = [
    {"image_path": os.path.join(coco, "apple.jpg"), "captions": "a red apple"},
    {"image_path": os.path.join(coco, "000000039769.png"), "captions": "cats sitting on a couch"},
    {"image_path": os.path.join(coco, "000000004016.png"), "captions": "a city street"},
]
with open(os.path.join(out, "clip_train.json"), "w", encoding="utf-8") as fh:
    json.dump(clip_rows, fh)

# Audio classification: silent 16 kHz mono wavs, 1 s of 16-bit zeros; the
# feature extractor does its own padding/normalization. The script only
# accepts --dataset_name (its --train_file field is defined but never
# consumed upstream), so the audio files must live in a dataset directory.
# Audio classification: the example reads ClassLabel.names off the label
# column, which the packaged json builder cannot provide, so lay the wavs out
# as an audiofolder tree instead — the builder derives the ClassLabel ("label"
# column) from the class subdirectories and train/validation from split dirs.
audio_root = os.path.join(out, "audio_data")
os.makedirs(audio_root, exist_ok=True)
# Drop json fixtures from earlier iterations: the audiofolder builder must not
# see stale split files left over on the persistent job output volume.
for stale in ("train.json", "validation.json"):
    stale_path = os.path.join(audio_root, stale)
    if os.path.exists(stale_path):
        os.remove(stale_path)
for split, names in (("train", ("a.wav", "b.wav")), ("validation", ("c.wav",))):
    for name in names:
        cls_dir = os.path.join(audio_root, split, "a" if name.startswith("a") else "b")
        os.makedirs(cls_dir, exist_ok=True)
        path = os.path.join(cls_dir, name)
        with wave.open(path, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(16000)
            fh.writeframes(b"\x00\x00" * 16000)
PY

# Batch-3 (speech recognition) data preparation: an audiofolder tree with a
# metadata.jsonl per split, so the builder derives the "audio" column from the
# wav files plus a "text" column from the metadata — the layout the CTC and
# seq2seq scripts consume through --audio_column_name/--text_column_name.
# jsonl over csv: pandas>=3 + pyarrow 25 make pa.Table.from_pandas emit
# large_string, but datasets 3.6.0's folder builder only accepts a metadata
# column equal to Value("string"), so every csv metadata file is rejected with
# "`file_name` or `*_file_name` must be present..."; the jsonl path reads via
# pyarrow's native json reader, which yields plain string.
# Single-char transcripts keep the generated CTC vocab tiny; the scripts
# always load an eval split, so both train and validation dirs are emitted.
python - "$CI_OUTPUT_DIR" <<'PY'
import json
import os
import sys
import wave

out = sys.argv[1]
asr_root = os.path.join(out, "asr_data")
for split, names in (("train", ("b1.wav", "b2.wav")), ("validation", ("b3.wav",))):
    split_dir = os.path.join(asr_root, split)
    os.makedirs(split_dir, exist_ok=True)
    # A leftover metadata.csv from an earlier run would clash with the jsonl
    # ("metadata files with different extensions" is fatal).
    stale = os.path.join(split_dir, "metadata.csv")
    if os.path.exists(stale):
        os.remove(stale)
    rows = []
    for i, name in enumerate(names):
        path = os.path.join(split_dir, name)
        with wave.open(path, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(16000)
            fh.writeframes(b"\x00\x00" * 16000)
        rows.append({"file_name": name, "text": "a" if i == 0 else "b"})
    with open(os.path.join(split_dir, "metadata.jsonl"), "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
PY
