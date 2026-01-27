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

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@echo "Initializing submodules..."
	@git submodule update --init
	
	@echo "Copying project documentation..."
	@for config in $(PROJECT_CONFIGS); do \
		src=$$(echo $$config | cut -d: -f1); \
		dst=$$(echo $$config | cut -d: -f2); \
		find $$src -name 'index.*' -delete 2>/dev/null || true; \
		echo "Copying $$src to $$dst"; \
		cp -r $$src/* $$dst/ 2>/dev/null || echo "  Source directory does not exist or is empty: $$src"; \
	done
	
	@echo "Cleaning up submodules..."
	@git submodule deinit -f _repos/*
	
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
