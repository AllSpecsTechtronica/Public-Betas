"""CI/CD lifecycle bar: champion-challenger alias pointers + promote/revert.

A compact, self-contained control that surfaces the model registry's
``candidate`` / ``staging`` / ``prod`` aliases for a scenario and exposes the two
human actions that make "reversible, promotable" literally true:

* **Promote staging -> prod** — copies the staged challenger's weights live and
  points ``prod`` at it (``POST /scenarios/{s}/runs/{run}/promote`` with
  ``target_alias=prod``).
* **Revert prod** — rolls ``prod`` back to the version it previously pointed at
  (``POST /scenarios/{s}/aliases/prod/revert``).

Data comes from ``GET /scenarios/{s}/aliases``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from .cvops_theme import cvops_color, cvops_rgba, repolish

_ALIAS_TONES = {
    "candidate": "accent_active",
    "staging": "accent_warn",
    "prod": "accent_select",
}


def _short_version(version_id: str) -> str:
    """``scenario:v3`` -> ``v3`` for compact chips."""
    vid = str(version_id or "")
    return vid.rsplit(":", 1)[-1] if ":" in vid else vid


class CiCdLifecycleBar(QFrame):
    """Promote/revert controls + alias chips for one scenario."""

    changed = pyqtSignal()          # emitted after a successful promote/revert
    errorRaised = pyqtSignal(str)

    def __init__(
        self,
        *,
        http_get: Callable[[str], dict[str, Any]],
        http_post: Callable[[str, Optional[dict[str, Any]]], dict[str, Any]],
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ciCdLifecycleBar")
        self._http_get = http_get
        self._http_post = http_post
        self._scenario = ""
        self._aliases: dict[str, dict[str, Any]] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(6)

        title = QLabel("Model lifecycle")
        title.setProperty("isTitle", True)
        outer.addWidget(title)

        chips = QHBoxLayout()
        chips.setSpacing(6)
        self._chips: dict[str, QLabel] = {}
        for alias in ("candidate", "staging", "prod"):
            chip = QLabel(f"{alias}: —")
            chip.setObjectName(f"ciCdChip_{alias}")
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setStyleSheet(self._chip_style(alias, active=False))
            chips.addWidget(chip)
            self._chips[alias] = chip
        chips.addStretch(1)
        outer.addLayout(chips)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self._promote_btn = QPushButton("Promote staging → prod")
        self._promote_btn.setToolTip("Copy the staged challenger live and point prod at it.")
        self._promote_btn.clicked.connect(self._on_promote)
        self._revert_btn = QPushButton("Revert prod")
        self._revert_btn.setToolTip("Roll prod back to the version it previously pointed at.")
        self._revert_btn.clicked.connect(self._on_revert)
        actions.addWidget(self._promote_btn)
        actions.addWidget(self._revert_btn)
        actions.addStretch(1)
        outer.addLayout(actions)

        self._set_buttons_enabled(False, False)

    # ------------------------------------------------------------------ data
    def set_scenario(self, scenario: str) -> None:
        self._scenario = str(scenario or "").strip()
        self.refresh()

    def refresh(self) -> None:
        if not self._scenario:
            self._aliases = {}
            self._render()
            return
        try:
            payload = self._http_get(f"/scenarios/{self._scenario}/aliases") or {}
            aliases = payload.get("aliases") if isinstance(payload.get("aliases"), dict) else {}
            self._aliases = {k: dict(v) for k, v in aliases.items() if isinstance(v, dict)}
        except Exception as exc:
            self._aliases = {}
            self.errorRaised.emit(f"CI/CD lifecycle: {exc}")
        self._render()

    # --------------------------------------------------------------- actions
    def _on_promote(self) -> None:
        staging = self._aliases.get("staging") or {}
        version_id = str(staging.get("version_id") or "")
        if not self._scenario or not version_id:
            return
        run_version = _short_version(version_id)
        try:
            self._http_post(
                f"/scenarios/{self._scenario}/runs/{run_version}/promote",
                {"target_alias": "prod", "actor": "cvops:ui", "reason": "promote staging to prod"},
            )
        except Exception as exc:
            self.errorRaised.emit(f"Promote failed: {exc}")
            return
        self.refresh()
        self.changed.emit()

    def _on_revert(self) -> None:
        if not self._scenario:
            return
        try:
            result = self._http_post(
                f"/scenarios/{self._scenario}/aliases/prod/revert",
                {"actor": "cvops:ui", "reason": "revert prod"},
            ) or {}
        except Exception as exc:
            self.errorRaised.emit(f"Revert failed: {exc}")
            return
        if not bool(result.get("reverted")):
            self.errorRaised.emit("Revert: prod has no prior version to roll back to.")
        self.refresh()
        self.changed.emit()

    # ----------------------------------------------------------------- view
    def _render(self) -> None:
        for alias, chip in self._chips.items():
            entry = self._aliases.get(alias) or {}
            version_id = str(entry.get("version_id") or "")
            label = _short_version(version_id) if version_id else "—"
            chip.setText(f"{alias}: {label}")
            chip.setStyleSheet(self._chip_style(alias, active=bool(version_id)))

        staging_set = bool((self._aliases.get("staging") or {}).get("version_id"))
        prod_history = (self._aliases.get("prod") or {}).get("history") or []
        prod_revertable = isinstance(prod_history, list) and len(prod_history) >= 2
        self._set_buttons_enabled(staging_set, prod_revertable)

    def _set_buttons_enabled(self, promote: bool, revert: bool) -> None:
        self._promote_btn.setEnabled(bool(promote))
        self._revert_btn.setEnabled(bool(revert))

    @staticmethod
    def _chip_style(alias: str, *, active: bool) -> str:
        tone = cvops_color(_ALIAS_TONES.get(alias, "text_iron"))
        if active:
            return (
                f"QLabel {{ border: 1px solid {tone}; border-radius: 8px; padding: 2px 8px; "
                f"color: {cvops_color('text_bright')}; background: {cvops_rgba('bg_panel', 0.35)}; "
                f"font-size: 10px; }}"
            )
        return (
            f"QLabel {{ border: 1px solid {cvops_rgba('line_light', 0.16)}; border-radius: 8px; "
            f"padding: 2px 8px; color: {cvops_color('text_iron')}; font-size: 10px; }}"
        )
