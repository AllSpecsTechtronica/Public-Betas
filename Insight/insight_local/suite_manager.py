"""Suite Manager — named profiles that bind a data integration layer to grid coordinates.

Architecture:
    Suite
      |_ data integration layer  (grid_cells values: type + config per source)
          |_ grid coordinates    (cell numbers 1-8 mapped to source configs)
              |_ final display   (rendered by GridAddOverlay + VideoPane)

Examples: Science Suite, Coding Suite, Engineering Suite
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_DEFAULT_SAVE_PATH = Path.home() / ".insight" / "suites.json"


@dataclass
class Suite:
    """A named grid profile.

    grid_cells maps cell number (1-8) to a source config dict:
        {"type": "web", "url": "..."}
        {"type": "terminal", "cwd": "..."}
        {"type": "media", "path": "..."}
        {"type": "widget", "widget_name": "..."}

    [FUTURE] data_integrations: list — webhooks, REST APIs, Google, program view
    """

    name: str
    grid_cells: dict[int, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "grid_cells": {str(k): v for k, v in self.grid_cells.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Suite":
        return cls(
            name=d["name"],
            grid_cells={int(k): v for k, v in d.get("grid_cells", {}).items()},
        )


class SuiteManager:
    """Manages the full collection of suites with JSON persistence."""

    def __init__(self, save_path: Path = _DEFAULT_SAVE_PATH) -> None:
        self._path = save_path
        self._suites: list[Suite] = []
        self._active_idx: int = 0
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                self._suites = [Suite.from_dict(s) for s in data.get("suites", [])]
                self._active_idx = int(data.get("active_idx", 0))
                if self._suites:
                    self._active_idx = max(0, min(self._active_idx, len(self._suites) - 1))
                else:
                    self._active_idx = 0
        except Exception:
            self._suites = []
            self._active_idx = 0

    def save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "suites": [s.to_dict() for s in self._suites],
                "active_idx": self._active_idx,
            }
            self._path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def suites(self) -> list[Suite]:
        return list(self._suites)

    @property
    def active_idx(self) -> int:
        return self._active_idx

    @property
    def active_suite(self) -> Optional[Suite]:
        if not self._suites:
            return None
        return self._suites[self._active_idx]

    def suite_names(self) -> list[str]:
        return [s.name for s in self._suites]

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_suite(self, name: str) -> int:
        """Add a new empty suite and return its index."""
        self._suites.append(Suite(name=name))
        idx = len(self._suites) - 1
        self.save()
        return idx

    def delete_suite(self, idx: int) -> None:
        if 0 <= idx < len(self._suites):
            self._suites.pop(idx)
            self._active_idx = max(0, min(self._active_idx, len(self._suites) - 1))
            self.save()

    def rename_suite(self, idx: int, name: str) -> None:
        if 0 <= idx < len(self._suites):
            self._suites[idx].name = name.strip()
            self.save()

    def set_active(self, idx: int) -> None:
        if 0 <= idx < len(self._suites):
            self._active_idx = idx
            self.save()

    def update_grid_cells(self, idx: int, grid_cells: dict[int, dict]) -> None:
        """Snapshot the current grid cell layout into the suite at idx."""
        if 0 <= idx < len(self._suites):
            self._suites[idx].grid_cells = dict(grid_cells)
            self.save()
