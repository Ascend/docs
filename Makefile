# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
# Default -j 1: parallel sphinx workers multiply RSS; cgroup limits (e.g. 2Gi) often OOM-kill (exit 137) without this.
SPHINXOPTS    ?= -j 1
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Configure all subprojects to process
# Format: "source_directory:target_directory"
PROJECT_CONFIGS = \
    _repos/verl/docs/ascend_tutorial:sources/verl \
    _repos/VeOmni/docs:sources/VeOmni \
    _repos/LLaMA-Factory/docs:sources/LLaMA-Factory \
    _repos/ms-swift/docs:sources/ms-swift \
    _repos/vllm-ascend/docs/source:sources/vllm-ascend \
    _repos/triton-ascend/docs/zh:sources/triton-ascend \
	_repos/deepspeed/docs/_tutorials/accelerator-setup-guide.md:sources/deepspeed/quick_start.md \

# Configure all subprojects generated path
GENERATED_DOCS := sources/_generated

# Ascend config file path
ASCEND_CONFIG := _static/ascend_config.json

# Fetch script
FETCH_SCRIPT := scripts/fetch_ascend_data.py

# Official ONNX Runtime CANN EP quick start source
ONNXRUNTIME_CANN_MD_URL := https://raw.githubusercontent.com/microsoft/onnxruntime/gh-pages/docs/execution-providers/community-maintained/CANN-ExecutionProvider.md
ONNXRUNTIME_CANN_MD_LOCAL := sources/_generated/sources/onnxruntime/quick_start.md

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile copy-docs clean-submodules fetch-config sync-onnxruntime-doc init-submodules

# Fetch ascend config (always run to ensure freshness)
.PHONY: $(ASCEND_CONFIG)
fetch-config:
	@echo "Fetching ascend configuration data..."
	@python3 $(FETCH_SCRIPT)

# Sync latest ONNX Runtime CANN EP doc from official gh-pages branch
sync-onnxruntime-doc:
	@echo "Syncing ONNX Runtime CANN quick start from upstream..."
	@mkdir -p $(dir $(ONNXRUNTIME_CANN_MD_LOCAL))
	@curl -fsSL "$(ONNXRUNTIME_CANN_MD_URL)" -o "$(ONNXRUNTIME_CANN_MD_LOCAL).tmp"
	@awk 'BEGIN{in_fm=0} \
		{sub(/\r$$/, "", $$0)} \
		NR==1 && $$0=="---" {in_fm=1; next} \
		in_fm && $$0=="---" {in_fm=0; next} \
		in_fm {next} \
		$$0 ~ /^##[[:space:]]+Contents[[:space:]]*$$/ {next} \
		$$0 ~ /^\{:[[:space:]]*\.no_toc[[:space:]]*\}$$/ {next} \
		$$0 ~ /^\{:[[:space:]]*toc[[:space:]]*\}$$/ {next} \
		$$0 ~ /^\*[[:space:]]*TOC[[:space:]]+placeholder[[:space:]]*$$/ {next} \
		{print}' "$(ONNXRUNTIME_CANN_MD_LOCAL).tmp" > "$(ONNXRUNTIME_CANN_MD_LOCAL)"
	@rm -f "$(ONNXRUNTIME_CANN_MD_LOCAL).tmp"
	@echo "Synced to $(ONNXRUNTIME_CANN_MD_LOCAL)"

# Initialize submodules (always run to handle empty dirs left by git clone)
init-submodules:
	@git submodule sync --recursive
	@n=0; \
	while [ $$n -lt 3 ]; do \
		git submodule update --init --remote && exit 0; \
		n=$$((n+1)); \
		echo "git submodule update failed (attempt $$n/3), retrying in 8s..."; \
		sleep 8; \
	done; \
	echo "git submodule update failed after 3 attempts"; \
	exit 1

# Copy documentation from submodules
copy-docs: init-submodules
	@echo "Preparing generated docs directory..."
	@mkdir -p $(GENERATED_DOCS)

	# vllm-ascend / triton-ascend: 仅删除根 index（避免与 sources/<comm>/index.rst 冲突），
	#   保留子目录 index（如 tutorials/models/index.md）供 toctree 直接引用。
	# 其他社区: 递归删除所有 index，导航由各社区 sources/<comm>/index.rst 独立定义。
	@echo "Copying project documentation..."
	@for config in $(PROJECT_CONFIGS); do \
		src=$$(echo $$config | cut -d: -f1); \
		rel_dst=$$(echo $$config | cut -d: -f2); \
		dst="$(GENERATED_DOCS)/$$rel_dst"; \
		echo "Copying $$src -> $$dst"; \
		rm -rf "$$dst"; \
		if [ -f "$$src" ]; then \
			mkdir -p "$$(dirname "$$dst")"; \
			echo "Copying $$src to $$dst"; \
			cp "$$src" "$$dst" || echo "  [WARN] Source file missing or copy failed: $$src"; \
		elif [ -d "$$src" ]; then \
			mkdir -p "$$dst"; \
			echo "Copying $$src to $$dst"; \
			cp -r "$$src"/* "$$dst"/ 2>/dev/null || echo "  [WARN] Source directory does not exist or is empty: $$src"; \
		else \
			echo "  [WARN] Source does not exist: $$src"; \
		fi; \
		if [ "$$rel_dst" = "sources/vllm-ascend" ] || [ "$$rel_dst" = "sources/triton-ascend" ]; then \
			rm -f "$$dst/index.md" "$$dst/index.rst" "$$dst/index.html" 2>/dev/null || true; \
		else \
			if [ -d "$$dst" ]; then \
				find "$$dst" -name 'index.*' -delete 2>/dev/null || true; \
			fi; \
		fi; \
	done


# Clean up submodules
clean-submodules:
	@echo "Cleaning up submodules..."
	@git submodule deinit -f _repos/*

# Explicit build targets with prerequisites
html dirhtml singlehtml latex pdf: fetch-config copy-docs sync-onnxruntime-doc
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target for other Sphinx targets (clean, help, etc.)
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
