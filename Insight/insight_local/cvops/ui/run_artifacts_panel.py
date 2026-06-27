from __future__ import annotations

import csv
import io
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...ui.media_utils import pixmap_from_b64_jpeg
from .path_actions import reveal_in_file_manager
from .viewport_geometry import reference_layout_size
from .training_graph import TrainingGraphWidget


_CHART_ORDER = [
    "results.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "BoxPR_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "labels.jpg",
    "labels_correlogram.jpg",
]


def _is_sample(name: str) -> bool:
    low = name.lower()
    return low.startswith("train_batch") or low.startswith("val_batch")


def _thumb_label(pix: QPixmap, caption: str, max_side: int = 180) -> QWidget:
    wrap = QWidget()
    v = QVBoxLayout(wrap)
    v.setContentsMargins(2, 2, 2, 2)
    v.setSpacing(2)
    img_lbl = QLabel()
    img_lbl.setObjectName("artifactThumb")
    img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    if not pix.isNull():
        scaled = pix.scaled(
            max_side,
            max_side,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        img_lbl.setPixmap(scaled)
    else:
        img_lbl.setText("[MISSING]")
    v.addWidget(img_lbl)
    cap = QLabel(caption)
    cap.setObjectName("artifactCaption")
    cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
    cap.setWordWrap(True)
    v.addWidget(cap)
    return wrap


class _Lightbox(QDialog):
    def __init__(self, pix: QPixmap, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(min(pix.width() or 800, 1200), min(pix.height() or 600, 900))
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setPixmap(pix)
        scroll.setWidget(lbl)
        v.addWidget(scroll, stretch=1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        v.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)


class _CollapsibleTextPreview(QWidget):
    def __init__(
        self,
        *,
        title: str,
        detail: str,
        loader: Callable[[], str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._loader = loader
        self._loaded = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle.clicked.connect(self._on_toggled)
        row.addWidget(self._toggle, stretch=1)
        detail_lbl = QLabel(detail)
        detail_lbl.setObjectName("artifactDetail")
        row.addWidget(detail_lbl)
        outer.addLayout(row)

        self._viewer = QPlainTextEdit()
        self._viewer.setObjectName("artifactPreview")
        self._viewer.setReadOnly(True)
        self._viewer.setVisible(False)
        self._viewer.setMaximumBlockCount(0)
        self._viewer.setMinimumHeight(160)
        self._viewer.setMaximumHeight(260)
        self._viewer.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        outer.addWidget(self._viewer)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        if checked and not self._loaded:
            try:
                self._viewer.setPlainText(self._loader())
            except Exception as exc:
                self._viewer.setPlainText(f"Unable to load preview: {exc}")
            self._loaded = True
        self._viewer.setVisible(checked)


class RunArtifactsPanel(QFrame):
    """Surfaces final training artifacts for a selected scenario run."""

    # Emitted when the user clicks [FLAG] next to Reveal Weights. Carries
    # Console run/model context for model-level feedback.
    flagRequested = pyqtSignal(object)

    _ROW_MIN_HEIGHT = 150
    _ROW_MAX_HEIGHT = 380

    def __init__(
        self,
        *,
        base_url: str = "",
        http_get: Callable[[str], dict[str, Any]],
        http_get_text: Callable[[str], str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._base_url = str(base_url or "").rstrip("/")
        self._http_get = http_get
        self._http_get_text = http_get_text
        self._artifacts_path: str = ""
        self._context_label: str = ""
        self._run_dir: str = ""
        self._weights_path: str = ""
        self._is_direct_artifact_run: bool = False
        self._current_items: list[dict[str, Any]] = []
        self._thumb_side = 180

        self.setFrameShape(QFrame.Shape.StyledPanel)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(5)

        head = QHBoxLayout()
        title = QLabel("Final Results")
        title.setObjectName("artifactPanelTitle")
        title.setFixedHeight(18)
        title.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        head.addWidget(title)
        head.addStretch(1)
        self._reveal_run_btn = QPushButton("Reveal Run")
        self._reveal_run_btn.clicked.connect(self._reveal_run_dir)
        head.addWidget(self._reveal_run_btn)
        self._reveal_weights_btn = QPushButton("Reveal Weights")
        self._reveal_weights_btn.clicked.connect(self._reveal_weights)
        head.addWidget(self._reveal_weights_btn)
        self._flag_btn = QPushButton("[FLAG]")
        self._flag_btn.setToolTip(
            "Save feedback against this run's weights."
        )
        self._flag_btn.setEnabled(False)
        self._flag_btn.clicked.connect(self._on_flag_clicked)
        head.addWidget(self._flag_btn)
        self._refresh_btn = QPushButton("Reload")
        self._refresh_btn.clicked.connect(self._reload)
        head.addWidget(self._refresh_btn)
        outer.addLayout(head)

        export_row = QHBoxLayout()
        export_row.setContentsMargins(0, 0, 0, 0)
        export_row.setSpacing(6)
        export_row.addWidget(QLabel("Export weights"))
        self._export_format = QComboBox()
        for label, fmt in (
            ("ONNX", "onnx"),
            ("TorchScript", "torchscript"),
            ("TensorRT engine", "engine"),
        ):
            self._export_format.addItem(label, fmt)
        export_row.addWidget(self._export_format)
        self._export_btn = QPushButton("Download export…")
        self._export_btn.clicked.connect(self._export_weights)
        export_row.addWidget(self._export_btn)
        export_row.addStretch(1)
        self._export_host = QWidget()
        self._export_host.setLayout(export_row)
        outer.addWidget(self._export_host)

        self._eval_insights_host = QWidget()
        self._eval_insights_layout = QVBoxLayout(self._eval_insights_host)
        self._eval_insights_layout.setContentsMargins(0, 0, 0, 0)
        self._eval_insights_layout.setSpacing(4)
        outer.addWidget(self._eval_insights_host)

        self._status = QLabel("No run loaded.")
        self._status.setStyleSheet("border: none; font-size: 10px;")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        # Charts row
        outer.addWidget(self._section_label("Charts"))
        self._charts_scroll = QScrollArea()
        self._charts_scroll.setWidgetResizable(True)
        self._charts_scroll.setMinimumHeight(72)
        self._charts_host = QWidget()
        self._charts_row = QHBoxLayout(self._charts_host)
        self._charts_row.setContentsMargins(2, 2, 2, 2)
        self._charts_row.setSpacing(5)
        self._charts_row.addStretch(1)
        self._charts_scroll.setWidget(self._charts_host)
        outer.addWidget(self._charts_scroll)

        # Samples row
        outer.addWidget(self._section_label("Sample Batches"))
        self._samples_scroll = QScrollArea()
        self._samples_scroll.setWidgetResizable(True)
        self._samples_scroll.setMinimumHeight(72)
        self._samples_host = QWidget()
        self._samples_row = QHBoxLayout(self._samples_host)
        self._samples_row.setContentsMargins(2, 2, 2, 2)
        self._samples_row.setSpacing(5)
        self._samples_row.addStretch(1)
        self._samples_scroll.setWidget(self._samples_host)
        outer.addWidget(self._samples_scroll)

        # Metrics grid
        outer.addWidget(self._section_label("Metrics"))
        self._metrics_host = QWidget()
        self._metrics_grid = QGridLayout(self._metrics_host)
        self._metrics_grid.setContentsMargins(2, 2, 2, 2)
        self._metrics_grid.setHorizontalSpacing(18)
        self._metrics_grid.setVerticalSpacing(2)
        outer.addWidget(self._metrics_host)

        # Files list
        outer.addWidget(self._section_label("Files"))
        self._files_host = QWidget()
        self._files_layout = QVBoxLayout(self._files_host)
        self._files_layout.setContentsMargins(2, 2, 2, 2)
        self._files_layout.setSpacing(3)
        outer.addWidget(self._files_host)

        self.clear()
        self._sync_responsive_geometry()

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("artifactSectionTitle")
        lbl.setFixedHeight(16)
        lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        return lbl

    def clear(self) -> None:
        self._artifacts_path = ""
        self._context_label = ""
        self._run_dir = ""
        self._weights_path = ""
        self._current_items = []
        self._status.setText("No run loaded.")
        self._reveal_run_btn.setEnabled(False)
        self._reveal_weights_btn.setEnabled(False)
        self._flag_btn.setEnabled(False)
        self._export_host.setVisible(False)
        self._clear_eval_insights()
        self._clear_rows()

    def load_job(self, job_id: str) -> None:
        job_id = str(job_id or "").strip()
        if not job_id:
            self.clear()
            return
        self._context_label = f"Job {job_id}"
        self._artifacts_path = f"/jobs/{job_id}/artifacts"
        self._reload()

    def load_run(self, scenario: str, version: str) -> None:
        scenario = str(scenario or "").strip()
        version = str(version or "").strip()
        if not scenario or not version:
            self.clear()
            return
        self._context_label = f"{scenario} {version}"
        self._artifacts_path = f"/scenarios/{scenario}/runs/{version}/artifacts"
        self._reload()

    def _reload(self) -> None:
        if not self._artifacts_path:
            self._status.setText("No run loaded.")
            return
        try:
            payload = self._http_get(self._artifacts_path)
        except Exception as exc:
            self._status.setText(f"Unable to load artifacts: {exc}")
            self._run_dir = ""
            self._weights_path = ""
            self._reveal_run_btn.setEnabled(False)
            self._reveal_weights_btn.setEnabled(False)
            self._export_host.setVisible(False)
            self._clear_eval_insights()
            self._clear_rows()
            return
        if not isinstance(payload, dict):
            self._status.setText("Invalid artifacts response.")
            self._run_dir = ""
            self._weights_path = ""
            self._reveal_run_btn.setEnabled(False)
            self._reveal_weights_btn.setEnabled(False)
            self._export_host.setVisible(False)
            self._clear_eval_insights()
            self._clear_rows()
            return
        self._run_dir = str(payload.get("run_dir") or "")
        backbone_type = str(payload.get("backbone_type") or "")
        is_direct_artifact_run = backbone_type != "yolo_detection"
        self._is_direct_artifact_run = bool(is_direct_artifact_run)
        items = payload.get("items") or []
        if not isinstance(items, list) or not items:
            context = f"{self._context_label}  |  " if self._context_label else ""
            self._status.setText(f"{context}No artifacts found in {self._run_dir or '[unknown]'}")
            self._weights_path = ""
            self._reveal_run_btn.setEnabled(bool(self._run_dir))
            self._reveal_weights_btn.setEnabled(False)
            self._export_host.setVisible(False)
            self._clear_eval_insights()
            self._clear_rows()
            return
        context = f"{self._context_label}  |  " if self._context_label else ""
        self._status.setText(f"{context}{self._run_dir}  |  {len(items)} files")
        typed_items = [it for it in items if isinstance(it, dict)]
        self._weights_path = self._resolve_weights_path(typed_items)
        self._reveal_run_btn.setEnabled(bool(self._run_dir))
        self._reveal_weights_btn.setEnabled(bool(self._weights_path))
        self._flag_btn.setEnabled(bool(self._weights_path))
        self._current_items = typed_items
        self._configure_export_ui(is_direct_artifact_run=is_direct_artifact_run, items=typed_items)
        self._render_items(typed_items)

    def _configure_export_ui(self, *, is_direct_artifact_run: bool, items: list[dict[str, Any]]) -> None:
        """Adjust export UI based on backbone/run contents."""
        if is_direct_artifact_run:
            # Non-YOLO runs: only support direct download of primary artifact.
            self._export_format.blockSignals(True)
            self._export_format.clear()
            self._export_format.addItem("Weights / Model (raw)", "raw")
            self._export_format.addItem("Adapter (.safetensors)", "safetensors")
            self._export_format.addItem("Weights (.pth)", "pth")
            self._export_format.addItem("Weights (.pt)", "pt")
            self._export_format.addItem("Model (.pkl)", "pkl")
            self._export_format.blockSignals(False)
            # Hide unsupported options when files don't exist.
            names = {str(item.get("name") or "") for item in items}
            want = {
                "safetensors": ("adapter/adapter_model.safetensors", "adapter_model.safetensors"),
                "pth": ("weights.pth",),
                "pt": ("weights.pt",),
                "pkl": ("model.pkl",),
            }
            for fmt, fnames in want.items():
                if not any(fname in names for fname in fnames):
                    idx = self._export_format.findData(fmt)
                    if idx >= 0:
                        self._export_format.removeItem(idx)
        else:
            # CV runs: standard exporter formats.
            self._export_format.blockSignals(True)
            if self._export_format.count() == 0 or self._export_format.findData("onnx") < 0:
                self._export_format.clear()
                for label, fmt in (
                    ("ONNX", "onnx"),
                    ("TorchScript", "torchscript"),
                    ("TensorRT engine", "engine"),
                ):
                    self._export_format.addItem(label, fmt)
            self._export_format.blockSignals(False)

    def _clear_eval_insights(self) -> None:
        while self._eval_insights_layout.count():
            item = self._eval_insights_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    @staticmethod
    def _scenario_run_from_artifacts_path(path: str) -> tuple[str, str]:
        parts = [p for p in str(path or "").strip("/").split("/") if p]
        if len(parts) >= 5 and parts[0] == "scenarios" and parts[2] == "runs" and parts[4] == "artifacts":
            return parts[1], parts[3]
        return "", ""

    def _export_weights(self) -> None:
        scenario, version = self._scenario_run_from_artifacts_path(self._artifacts_path)
        if not scenario or not version or not self._base_url:
            self._status.setText("Export requires a scenario run and service URL.")
            return
        fmt = str(self._export_format.currentData() or "onnx")
        q = urllib.parse.urlencode({"format": fmt})
        url = f"{self._base_url}/scenarios/{scenario}/runs/{version}/export?{q}"
        if fmt == "raw":
            ext = Path(self._weights_path).suffix if self._weights_path else ".bin"
        elif fmt == "safetensors":
            ext = ".safetensors"
        else:
            ext = ".onnx" if fmt == "onnx" else ".engine" if fmt == "engine" else ".torchscript"
        final_name = ""
        try:
            text = self._http_get_text(f"{self._artifacts_path}/metrics.json")
            data = json.loads(text)
            if isinstance(data, dict):
                final_name = str(data.get("final_model_name") or "").strip()
        except Exception:
            final_name = ""
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in final_name).strip("._-")
        default = f"{safe_name}{ext}" if safe_name else f"{scenario}_{version}{ext}"
        dest, _flt = QFileDialog.getSaveFileName(self, "Save export", default, "Model files (*.*)")
        if not dest:
            return
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = resp.read()
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            self._status.setText(f"Export failed ({exc.code}): {body or exc.reason}")
            return
        except Exception as exc:
            self._status.setText(f"Export failed: {exc}")
            return
        try:
            Path(dest).write_bytes(data)
        except Exception as exc:
            self._status.setText(f"Unable to write file: {exc}")
            return
        self._status.setText(f"Saved export to {dest}")

    def _render_eval_insights(self, by_name: dict[str, dict[str, Any]]) -> None:
        self._clear_eval_insights()
        scenario, version = self._scenario_run_from_artifacts_path(self._artifacts_path)
        self._export_host.setVisible(bool(scenario and version and self._base_url))
        if not self._artifacts_path:
            self._export_host.setVisible(False)
            return
        # If we don't have weights/model to download, hide export.
        if not self._weights_path:
            self._export_host.setVisible(False)
        if "ci_cd_report.json" in by_name:
            title = QLabel("CI/CD gate report")
            title.setObjectName("artifactSectionTitle")
            title.setFixedHeight(16)
            title.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self._eval_insights_layout.addWidget(title)
            preview = QPlainTextEdit()
            preview.setObjectName("artifactPreview")
            preview.setReadOnly(True)
            preview.setMinimumHeight(100)
            preview.setMaximumHeight(220)
            try:
                raw = self._http_get_text(f"{self._artifacts_path}/ci_cd_report.json")
                data = json.loads(raw)
                if isinstance(data, dict):
                    status = str(data.get("gate_status") or "unknown").upper()
                    failures = data.get("failures") if isinstance(data.get("failures"), list) else []
                    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
                    lines = [
                        f"Gate: {status}",
                        f"Metric: {metrics.get('metric', '')} = {metrics.get('value', '')}",
                        f"Threshold: {metrics.get('threshold', '')}",
                        f"Baseline: {metrics.get('baseline_version_id', '') or 'none'}",
                    ]
                    if failures:
                        lines.append("Failures:")
                        lines.extend(f"- {item}" for item in failures)
                    preview.setPlainText("\n".join(lines))
                else:
                    preview.setPlainText(raw[:80000])
            except Exception as exc:
                preview.setPlainText(f"Unable to load ci_cd_report.json: {exc}")
            self._eval_insights_layout.addWidget(preview)
        if "confusion_matrix.json" in by_name:
            title = QLabel("Confusion matrix (JSON)")
            title.setObjectName("artifactSectionTitle")
            title.setFixedHeight(16)
            title.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self._eval_insights_layout.addWidget(title)
            preview = QPlainTextEdit()
            preview.setObjectName("artifactPreview")
            preview.setReadOnly(True)
            preview.setMaximumBlockCount(0)
            preview.setMinimumHeight(120)
            preview.setMaximumHeight(220)
            try:
                raw = self._http_get_text(f"{self._artifacts_path}/confusion_matrix.json")
                parsed = json.loads(raw)
                preview.setPlainText(json.dumps(parsed, indent=2, ensure_ascii=True)[:120000])
            except Exception as exc:
                preview.setPlainText(f"Unable to load confusion_matrix.json: {exc}")
            self._eval_insights_layout.addWidget(preview)
        if "error_samples.json" in by_name:
            title = QLabel("Error gallery (top samples)")
            title.setObjectName("artifactSectionTitle")
            title.setFixedHeight(16)
            title.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self._eval_insights_layout.addWidget(title)
            preview = QPlainTextEdit()
            preview.setObjectName("artifactPreview")
            preview.setReadOnly(True)
            preview.setMinimumHeight(120)
            preview.setMaximumHeight(260)
            try:
                raw = self._http_get_text(f"{self._artifacts_path}/error_samples.json")
                data = json.loads(raw)
                lines: list[str] = []
                if isinstance(data, dict):
                    for row in list(data.get("samples") or [])[:24]:
                        if not isinstance(row, dict):
                            continue
                        img = str(row.get("image") or "")
                        lines.append(
                            f"score={row.get('score')} fp={row.get('fp')} fn={row.get('fn')} iou={row.get('mean_match_iou')}  {img}"
                        )
                preview.setPlainText("\n".join(lines) if lines else raw[:80000])
            except Exception as exc:
                preview.setPlainText(f"Unable to load error_samples.json: {exc}")
            self._eval_insights_layout.addWidget(preview)

    def _clear_rows(self) -> None:
        for layout in (self._charts_row, self._samples_row):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            layout.addStretch(1)
        while self._metrics_grid.count():
            item = self._metrics_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        while self._files_layout.count():
            item = self._files_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _fetch_thumb_pixmap(self, name: str, full: bool = False) -> QPixmap:
        if not self._artifacts_path:
            return QPixmap()
        path = f"{self._artifacts_path}/{name}"
        if full:
            path += "?full=1"
        try:
            b64 = self._http_get_text(path)
        except Exception:
            return QPixmap()
        return pixmap_from_b64_jpeg(b64.strip())

    def _resolve_weights_path(self, items: list[dict[str, Any]]) -> str:
        if not self._run_dir:
            return ""
        names = {str(item.get("name") or "") for item in items}
        for rel_name in (
            "weights.pth",
            "weights.pt",
            "model.pkl",
            "adapter/adapter_model.safetensors",
            "adapter_model.safetensors",
            "weights/best.pt",
            "weights/last.pt",
        ):
            if rel_name in names:
                return str(Path(self._run_dir) / rel_name)
        return ""

    def _add_thumb(self, row: QHBoxLayout, name: str) -> None:
        pix = self._fetch_thumb_pixmap(name, full=False)
        widget = _thumb_label(pix, name, max_side=self._thumb_side)
        widget.mousePressEvent = lambda _ev, n=name: self._open_lightbox(n)  # type: ignore[assignment]
        widget.setCursor(Qt.CursorShape.PointingHandCursor)
        widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        row.insertWidget(row.count() - 1, widget)

    def _open_lightbox(self, name: str) -> None:
        pix = self._fetch_thumb_pixmap(name, full=True)
        if pix.isNull():
            return
        dlg = _Lightbox(pix, name, parent=self)
        dlg.exec()

    def _reveal_run_dir(self) -> None:
        if not self._run_dir:
            return
        try:
            reveal_in_file_manager(self._run_dir)
        except Exception as exc:
            self._status.setText(f"Reveal failed: {exc}")

    def _reveal_weights(self) -> None:
        if not self._weights_path:
            return
        try:
            reveal_in_file_manager(self._weights_path)
        except Exception as exc:
            self._status.setText(f"Reveal failed: {exc}")

    def _on_flag_clicked(self) -> None:
        if not self._weights_path:
            return
        scenario, version = self._scenario_run_from_artifacts_path(self._artifacts_path)
        self.flagRequested.emit(
            {
                "source": "run_artifacts",
                "weights_path": str(self._weights_path),
                "run_dir": str(self._run_dir),
                "context": str(self._context_label),
                "artifacts_path": str(self._artifacts_path),
                "scenario": scenario,
                "version": version,
                "is_direct_artifact_run": bool(self._is_direct_artifact_run),
                "artifact_count": len(self._current_items),
            }
        )

    def _render_items(self, items: list[dict[str, Any]]) -> None:
        self._sync_responsive_geometry()
        self._current_items = list(items)
        self._clear_rows()
        by_name = {str(it.get("name") or ""): it for it in items if it.get("name")}
        names = list(by_name.keys())

        # Charts
        chart_names = [n for n in _CHART_ORDER if n in by_name]
        # Include any additional images that are not samples and not already listed
        extras = [
            n for n in names
            if by_name[n].get("kind") == "image"
            and n not in chart_names
            and not _is_sample(n)
        ]
        for name in chart_names + extras:
            self._add_thumb(self._charts_row, name)
        # For tabular runs, always render a built-in history graph when available.
        if self._is_direct_artifact_run:
            points = self._load_tabular_history_points(by_name)
            graph = TrainingGraphWidget()
            graph.setMinimumHeight(140)
            if points:
                graph.set_points(points)
            else:
                graph.clear()
            self._charts_row.insertWidget(0, graph)
        elif not (chart_names or extras):
            placeholder = QLabel("No charts available.")
            placeholder.setStyleSheet("border: none; font-size: 10px;")
            self._charts_row.insertWidget(self._charts_row.count() - 1, placeholder)

        # Samples
        samples = sorted(n for n in names if by_name[n].get("kind") == "image" and _is_sample(n))
        for name in samples:
            self._add_thumb(self._samples_row, name)
        if not samples:
            placeholder = QLabel("No sample batches recorded.")
            placeholder.setStyleSheet("border: none; font-size: 10px;")
            self._samples_row.insertWidget(self._samples_row.count() - 1, placeholder)

        # Metrics (from metrics.json + last row of results.csv)
        metrics = self._load_metrics(by_name)
        if metrics:
            row = 0
            for key, value in metrics.items():
                k_lbl = QLabel(str(key))
                k_lbl.setStyleSheet(
                    "border: none; font-size: 10px; font-weight: 600;"
                )
                v_lbl = QLabel(str(value))
                v_lbl.setStyleSheet(
                    "border: none; font-size: 10px;"
                )
                self._metrics_grid.addWidget(k_lbl, row, 0)
                self._metrics_grid.addWidget(v_lbl, row, 1)
                row += 1
            self._metrics_grid.setColumnStretch(2, 1)
        else:
            placeholder = QLabel("No metrics available.")
            placeholder.setStyleSheet("border: none; font-size: 10px;")
            self._metrics_grid.addWidget(placeholder, 0, 0, 1, 2)

        # Files row: non-image files
        other_files = [
            (n, by_name[n])
            for n in sorted(names)
            if by_name[n].get("kind") != "image"
        ]
        if not other_files:
            placeholder = QLabel("No additional files.")
            placeholder.setStyleSheet("border: none; font-size: 10px;")
            self._files_layout.addWidget(placeholder)
        else:
            for name, info in other_files:
                size = int(info.get("size_bytes") or 0)
                if size >= 1024 * 1024:
                    size_s = f"{size / (1024 * 1024):.2f} MB"
                elif size >= 1024:
                    size_s = f"{size / 1024:.1f} KB"
                else:
                    size_s = f"{size} B"
                kind = str(info.get("kind") or "-")
                if kind in {"json", "jsonl", "csv", "modelfile", "yaml"}:
                    preview = _CollapsibleTextPreview(
                        title=f"{name}  ({kind})",
                        detail=size_s,
                        loader=lambda file_name=name, file_kind=kind: self._load_file_preview(file_name, file_kind),
                    )
                    self._files_layout.addWidget(preview)
                    continue
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                lbl = QLabel(f"{name}  ({kind})")
                lbl.setStyleSheet(
                    "border: none; font-size: 10px;"
                )
                size_lbl = QLabel(size_s)
                size_lbl.setStyleSheet("border: none; font-size: 10px;")
                row.addWidget(lbl, stretch=1)
                row.addWidget(size_lbl)
                holder = QWidget()
                holder.setLayout(row)
                self._files_layout.addWidget(holder)

        self._render_eval_insights(by_name)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        old_thumb_side = self._thumb_side
        self._sync_responsive_geometry()
        if self._current_items and old_thumb_side != self._thumb_side:
            self._render_items(self._current_items)

    def refresh_responsive_layout(self) -> None:
        old_thumb_side = self._thumb_side
        self._sync_responsive_geometry()
        if self._current_items and old_thumb_side != self._thumb_side:
            self._render_items(self._current_items)

    def _sync_responsive_geometry(self) -> None:
        panel_w, panel_h = reference_layout_size(self)
        row_h = int(max(self._ROW_MIN_HEIGHT, min(self._ROW_MAX_HEIGHT, panel_h * 0.24)))
        self._charts_scroll.setMaximumHeight(row_h)
        self._samples_scroll.setMaximumHeight(row_h)
        self._thumb_side = int(max(120, min(280, min(panel_w * 0.24, row_h * 0.78))))

    def _load_file_preview(self, name: str, kind: str) -> str:
        raw_text = self._http_get_text(f"{self._artifacts_path}/{name}")
        if kind == "json":
            try:
                parsed = json.loads(raw_text)
            except Exception:
                return raw_text
            return json.dumps(parsed, indent=2, ensure_ascii=True, default=str)
        return raw_text

    def _load_metrics(self, by_name: dict[str, dict[str, Any]]) -> dict[str, str]:
        out: dict[str, str] = {}
        if "metrics.json" in by_name:
            try:
                text = self._http_get_text(f"{self._artifacts_path}/metrics.json")
                data = json.loads(text)
                if isinstance(data, dict):
                    # YOLO-style flat metrics
                    if data.get("final_model_name"):
                        out["final_model_name"] = str(data.get("final_model_name"))
                    for key in ("map50", "map50_95", "precision", "recall", "fitness"):
                        if key in data and key not in out:
                            out[key] = str(data.get(key))
                    # Tabular-style nested metrics
                    nested = data.get("metrics") if isinstance(data.get("metrics"), dict) else None
                    if nested:
                        for key in (
                            "task",
                            "final_val_metric",
                            "best_val_loss",
                            "num_features",
                            "rows",
                            "num_classes",
                            "train_examples",
                            "val_examples",
                            "dry_run",
                        ):
                            if key in nested and key not in out:
                                out[key] = str(nested.get(key))
                        # Provide a friendly label depending on task.
                        task = str(nested.get("task") or "")
                        if "final_val_metric" in nested and "final_val_metric" in out:
                            if task == "regression":
                                out.setdefault("val_mae", out.get("final_val_metric", ""))
                            elif task == "classification":
                                out.setdefault("val_acc", out.get("final_val_metric", ""))
                    for key in ("base_model", "ollama_base_model", "adapter_path", "modelfile"):
                        if key in data and key not in out:
                            out[key] = str(data.get(key))
                    run_info = data.get("run") if isinstance(data.get("run"), dict) else None
                    if run_info:
                        for key in ("epochs", "imgsz", "batch"):
                            if key in run_info and key not in out:
                                out[key] = str(run_info.get(key))
            except Exception:
                pass
        if "results.csv" in by_name:
            try:
                text = self._http_get_text(f"{self._artifacts_path}/results.csv")
                reader = csv.DictReader(io.StringIO(text))
                rows = list(reader)
                if rows:
                    last = rows[-1]
                    for raw_key, raw_val in last.items():
                        if raw_key is None:
                            continue
                        key = str(raw_key).strip()
                        if not key:
                            continue
                        if "mAP50" in key or "loss" in key.lower() or "precision" in key.lower() or "recall" in key.lower():
                            label = f"csv:{key}"
                            if label not in out:
                                out[label] = str(raw_val).strip()
            except Exception:
                pass
        return out

    def _load_tabular_history_points(self, by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        if "metrics.json" not in by_name or not self._artifacts_path:
            return []
        try:
            text = self._http_get_text(f"{self._artifacts_path}/metrics.json")
            data = json.loads(text)
        except Exception:
            return []
        if not isinstance(data, dict):
            return []
        history = data.get("history")
        if not isinstance(history, list) or not history:
            return []
        points: list[dict[str, Any]] = []
        # BasicCNN template history has keys: epoch, train_loss, val_loss, val_mae/val_acc
        for row in history:
            if not isinstance(row, dict):
                continue
            try:
                ep = int(float(row.get("epoch") or 0)) - 1
            except Exception:
                ep = len(points)
            point = {"event": "epoch", "epoch": ep}
            for k in ("train_loss", "val_loss", "val_mae", "val_acc"):
                if k in row:
                    point[k] = row.get(k)
            points.append(point)
        # Attach epochs/progress so the live label uses reasonable numbers.
        total = len(points)
        for i, p in enumerate(points):
            p["epochs"] = total
            p["progress"] = (float(i + 1) / float(max(1, total))) * 100.0
        return points
