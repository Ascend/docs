# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
# Default -j 1: parallel sphinx workers multiply RSS; cgroup limits (e.g. 2Gi) often OOM-kill (exit 137) without this.
SPHINXOPTS    ?= -j 1
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Ascend config file path
ASCEND_CONFIG := _static/ascend_config.json

# Fetch script
FETCH_SCRIPT := scripts/fetch_ascend_data.py

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile fetch-config

# Fetch ascend config (always run to ensure freshness)
.PHONY: $(ASCEND_CONFIG)
fetch-config:
	@echo "Fetching ascend configuration data..."
	@python3 $(FETCH_SCRIPT)

# Explicit build targets with prerequisites
html dirhtml singlehtml latex pdf: fetch-config
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target for other Sphinx targets (clean, help, etc.)
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
