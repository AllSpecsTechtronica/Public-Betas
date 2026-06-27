from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _metric_from_payload(payload: dict[str, Any], key: str) -> float | None:
    direct = _to_float(payload.get(key))
    if direct is not None:
        return direct
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    mapped = _to_float(metrics.get(key))
    if mapped is not None:
        return mapped
    aliases = {
        "map50": ("metrics/mAP50(B)", "mAP50"),
        "precision": ("metrics/precision(B)",),
        "recall": ("metrics/recall(B)",),
    }
    for alt in aliases.get(key, ()):
        v = _to_float(metrics.get(alt))
        if v is not None:
            return v
    return None


class RunComparePanel(QWidget):
    """Side-by-side metrics and hyperparameter diff for two scenario runs."""

    def __init__(
        self,
        *,
        http_get: Callable[[str], dict[str, Any]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._http_get = http_get
        self._scenario = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        row = QHBoxLayout()
        row.addWidget(QLabel("Run A"))
        self._run_a = QComboBox()
        self._run_a.setMinimumWidth(120)
        row.addWidget(self._run_a, stretch=1)
        row.addWidget(QLabel("Run B"))
        self._run_b = QComboBox()
        self._run_b.setMinimumWidth(120)
        row.addWidget(self._run_b, stretch=1)
        self._compare_btn = QPushButton("Compare")
        self._compare_btn.clicked.connect(self._compare)
        row.addWidget(self._compare_btn)
        layout.addLayout(row)

        self._status = QLabel("Select two runs, then Compare.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size: 10px;")
        layout.addWidget(self._status)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._body = QWidget()
        self._grid = QGridLayout(self._body)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(4)
        scroll.setWidget(self._body)
        layout.addWidget(scroll, stretch=1)

    def set_scenario(self, scenario: str) -> None:
        scenario = str(scenario or "").strip()
        if scenario == self._scenario:
            return
        self._scenario = scenario
        self._run_a.clear()
        self._run_b.clear()
        self._clear_grid()
        if not scenario:
            self._status.setText("Select a scenario in the catalog list.")
            return
        try:
            payload = self._http_get(f"/scenarios/{scenario}/history")
        except Exception as exc:
            self._status.setText(f"Unable to load history: {exc}")
            return
        runs = list(payload.get("runs") or [])
        versions: list[str] = []
        for item in runs:
            if isinstance(item, dict):
                v = str(item.get("version") or "").strip()
                if v:
                    versions.append(v)
        for v in versions:
            self._run_a.addItem(v, v)
            self._run_b.addItem(v, v)
        if len(versions) >= 2:
            self._run_b.setCurrentIndex(1)
        self._status.setText(f"{len(versions)} runs loaded for {scenario}.")

    def _clear_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _fetch_metrics(self, scenario: str, version: str) -> dict[str, Any]:
        data = self._http_get(f"/scenarios/{scenario}/runs/{version}/metrics")
        return data if isinstance(data, dict) else {}

    def _compare(self) -> None:
        if not self._scenario:
            self._status.setText("No scenario selected.")
            return
        va = str(self._run_a.currentData() or self._run_a.currentText() or "").strip()
        vb = str(self._run_b.currentData() or self._run_b.currentText() or "").strip()
        if not va or not vb:
            self._status.setText("Choose two run versions.")
            return
        if va == vb:
            self._status.setText("Pick two different runs.")
            return
        try:
            left = self._fetch_metrics(self._scenario, va)
            right = self._fetch_metrics(self._scenario, vb)
        except Exception as exc:
            self._status.setText(f"Metrics fetch failed: {exc}")
            return
        self._clear_grid()
        r = 0
        title = QLabel("Metric / field")
        title.setStyleSheet("font-weight: 600; font-size: 10px;")
        la = QLabel(f"{va}")
        la.setStyleSheet("font-weight: 600; font-size: 10px;")
        lb = QLabel(f"{vb}")
        lb.setStyleSheet("font-weight: 600; font-size: 10px;")
        ld = QLabel("Delta (B - A)")
        ld.setStyleSheet("font-weight: 600; font-size: 10px;")
        self._grid.addWidget(title, r, 0)
        self._grid.addWidget(la, r, 1)
        self._grid.addWidget(lb, r, 2)
        self._grid.addWidget(ld, r, 3)
        r += 1

        metric_keys = [
            ("map50", "mAP50"),
            ("precision", "Precision"),
            ("recall", "Recall"),
        ]
        for key, label in metric_keys:
            a = _metric_from_payload(left, key)
            b = _metric_from_payload(right, key)
            da = "" if a is None or b is None else f"{(b - a):+.6f}"
            self._add_row(r, label, self._fmt(a), self._fmt(b), da)
            r += 1

        self._grid.addWidget(QLabel("Hyperparams (train kwargs)"), r, 0, 1, 4)
        r += 1
        hk = self._hyperparam_keys(left, right)
        for key in sorted(hk):
            lv = self._hyper_get(left, key)
            rv = self._hyper_get(right, key)
            same = lv == rv
            delta = "" if same else "[diff]"
            self._add_row(r, key, lv, rv, delta, muted=same)
            r += 1

        self._status.setText(f"Compared {va} vs {vb}.")

    @staticmethod
    def _fmt(value: float | None) -> str:
        if value is None:
            return ""
        return f"{value:.6f}"

    @staticmethod
    def _hyper_get(payload: dict[str, Any], key: str) -> str:
        hp = payload.get("hyperparams") if isinstance(payload.get("hyperparams"), dict) else {}
        if key in hp:
            return str(hp.get(key))
        eff = payload.get("effective_hyperparams") if isinstance(payload.get("effective_hyperparams"), dict) else {}
        if key in eff:
            return str(eff.get(key))
        guard = payload.get("training_guard") if isinstance(payload.get("training_guard"), dict) else {}
        eff2 = guard.get("effective_hyperparams") if isinstance(guard.get("effective_hyperparams"), dict) else {}
        if key in eff2:
            return str(eff2.get(key))
        return ""

    @staticmethod
    def _hyperparam_keys(a: dict[str, Any], b: dict[str, Any]) -> set[str]:
        keys: set[str] = set()

        def collect(src: dict[str, Any]) -> None:
            hp = src.get("hyperparams") if isinstance(src.get("hyperparams"), dict) else {}
            keys.update(str(k) for k in hp.keys())
            eff = src.get("effective_hyperparams") if isinstance(src.get("effective_hyperparams"), dict) else {}
            keys.update(str(k) for k in eff.keys())
            guard = src.get("training_guard") if isinstance(src.get("training_guard"), dict) else {}
            eff2 = guard.get("effective_hyperparams") if isinstance(guard.get("effective_hyperparams"), dict) else {}
            keys.update(str(k) for k in eff2.keys())

        collect(a)
        collect(b)
        noise = {"data", "project", "name", "exist_ok"}
        return {k for k in keys if k not in noise}

    def _add_row(self, row: int, k: str, a: str, b: str, d: str, *, muted: bool = False) -> None:
        for col, text in enumerate((k, a, b, d)):
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("font-size: 10px;")
            lbl.setProperty("muted", bool(muted))
            self._grid.addWidget(lbl, row, col)
