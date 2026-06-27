"""Lightweight, dependency-free cvops filesystem paths.

Importing this must stay cheap. window.py and the lazy panels read these path
constants without importing service.py, which drags in fastapi/pydantic and the
mlops/torch stack — the dominant cold-start cost. service.py re-exports these so
existing ``from .service import CVOPS_STATE_DIR`` call sites keep working.
"""
from __future__ import annotations

from ..config import ROOT_DIR

MLOPS_ROOT = ROOT_DIR / "mlops"
CVOPS_STATE_DIR = ROOT_DIR / "state" / "insight_local" / "cvops"
CVOPS_DB_PATH = CVOPS_STATE_DIR / "jobs.db"
CVOPS_CATALOG_DB_PATH = CVOPS_STATE_DIR / "catalog.db"
