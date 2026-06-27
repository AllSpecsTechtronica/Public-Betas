"""Proxy to the repo-root `basic_cnn_for_editing.py`.

Keeping the editable template at repo root makes it easy to open/edit, but many
scenarios prefer referencing `mlops/algos/...` paths. This file just re-exports
the `run()` entrypoint.
"""

from __future__ import annotations

from basic_cnn_for_editing import run

__all__ = ["run"]

