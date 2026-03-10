# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Configure all subprojects to process
# Format: "source_directory:target_directory"
PROJECT_CONFIGS = \
    _repos/verl/docs/ascend_tutorial:sources/verl \
    _repos/VeOmni/docs:sources/VeOmni \
    _repos/LLaMA-Factory/docs:sources/LLaMA-Factory \
    _repos/ms-swift/docs:sources/ms-swift

# Configure all subprojects generated path
GENERATED_DOCS := sources/_generated

# Ascend config file path
ASCEND_CONFIG := _static/ascend_config.json

# Fetch script
FETCH_SCRIPT := scripts/fetch_ascend_data.py

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile copy-docs clean-submodules fetch-config

# Fetch ascend config (always run to ensure freshness)
.PHONY: $(ASCEND_CONFIG)
fetch-config:
	@echo "Fetching ascend configuration data..."
	@python3 $(FETCH_SCRIPT)

# Initialize submodules if not exists (use pinned commits for reproducibility)
_repos/verl _repos/VeOmni _repos/LLaMA-Factory _repos/ms-swift:
	@echo "Initializing submodules..."
	@git submodule update --init

# Copy documentation from submodules
copy-docs: _repos/verl _repos/VeOmni _repos/LLaMA-Factory _repos/ms-swift
	@echo "Preparing generated docs directory..."
	@mkdir -p $(GENERATED_DOCS)

	@echo "Copying project documentation..."
	@for config in $(PROJECT_CONFIGS); do \
		src=$$(echo $$config | cut -d: -f1); \
		rel_dst=$$(echo $$config | cut -d: -f2); \
		dst="$(GENERATED_DOCS)/$$rel_dst"; \
		echo "Copying $$src -> $$dst"; \
		rm -rf $$dst; \
		mkdir -p $$dst; \
		find $$src -name 'index.*' -delete 2>/dev/null || true; \
		echo "Copying $$src to $$dst"; \
		cp -r "$$src"/* "$$dst"/ 2>/dev/null || echo "  [WARN] Source directory does not exist or is empty: $$src"; \
	done

# Clean up submodules
clean-submodules:
	@echo "Cleaning up submodules..."
	@git submodule deinit -f _repos/*

# Explicit build targets with prerequisites
html dirhtml singlehtml latex pdf: fetch-config copy-docs
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target for other Sphinx targets (clean, help, etc.)
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)