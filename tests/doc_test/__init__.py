"""Shared workflow utilities and base classes.

This is the namespace package for code reused across projects (e.g. the
markdown documentation test framework). Submodules are imported as
``workflows.<module>`` after ``sys.path`` is set up to include the repo
root's ``src/`` (typically by each project's ``tests/__init__.py``).

Framework dependencies (mistune) are installed by the common
quick-start workflow template, not at import time here.
"""