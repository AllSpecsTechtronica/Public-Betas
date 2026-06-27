from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional
import urllib.error
import urllib.request

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...config import ROOT_DIR
from .collapsible_section import CollapsibleSection
from .csv_table_editor import CsvTableEditorDialog

# Accepted ingest formats (mirrors service.SUPPORTED_TABULAR_SUFFIXES).
_TABULAR_SUFFIXES = (
    ".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".pq", ".json", ".jsonl", ".ndjson",
)
_FILE_DIALOG_FILTER = (
    "Tabular data (*.csv *.tsv *.xlsx *.xls *.parquet *.pq *.json *.jsonl *.ndjson)"
)

_QUOTED_RE = re.compile(r"'([^']+)'")


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        detail = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        detail = ""
    if detail:
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict) and parsed.get("detail"):
                detail = str(parsed["detail"])
        except Exception:
            pass
    if len(detail) > 300:
        detail = detail[:300] + "..."
    return f"HTTP {exc.code}" + (f" {detail}" if detail else "")


class _CsvDropZone(QLabel):
    """Drop target that accepts a single tabular file in any supported format."""

    filesDropped = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(72)
        self.setText("Drop a tabular file here  (.csv .tsv .xlsx .parquet .json .jsonl)")
        self._apply_style(active=False)

    def _apply_style(self, *, active: bool) -> None:
        if active:
            border = "rgba(146, 208, 218, 0.92)"
            bg = "rgba(146, 208, 218, 0.10)"
        else:
            border = "rgba(146, 208, 218, 0.56)"
            bg = "rgba(146, 208, 218, 0.04)"
        self.setStyleSheet(
            "QLabel {"
            f"border: 1px dashed {border};"
            f"background: {bg};"
            "padding: 10px;"
            "font-size: 10px;"
            "border-radius: 0px;"
            "}"
        )

    @staticmethod
    def _tabular_paths(event) -> list[str]:  # type: ignore[no-untyped-def]
        out: list[str] = []
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local and Path(local).suffix.lower() in _TABULAR_SUFFIXES:
                out.append(local)
        return out

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls() and self._tabular_paths(event):
            event.acceptProposedAction()
            self._apply_style(active=True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # type: ignore[override]
        self._apply_style(active=False)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        self._apply_style(active=False)
        paths = self._tabular_paths(event)
        if paths:
            event.acceptProposedAction()
            self.filesDropped.emit(paths)
        else:
            event.ignore()


class _TabularUploadWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object)  # uploaded list[(slug, payload)], errors list[str]

    def __init__(self, *, base_url: str, files: list[str], parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._base_url = str(base_url or "").rstrip("/")
        self._files = list(files or [])

    def run(self) -> None:
        from .dataset_panel import _multipart_upload

        url = f"{self._base_url}/database/upload_tabular"
        uploaded: list[tuple[str, dict]] = []
        errors: list[str] = []
        valid: list[Path] = []

        self.progress.emit(f"Queued {len(self._files)} file(s) for tabular upload.")
        for raw in self._files:
            path = Path(str(raw or "")).expanduser()
            if not path.exists():
                errors.append(f"{path.name or raw}: file not found")
                continue
            if not path.is_file():
                errors.append(f"{path.name}: not a file")
                continue
            if path.suffix.lower() not in _TABULAR_SUFFIXES:
                errors.append(f"{path.name}: unsupported extension {path.suffix}")
                continue
            valid.append(path)

        if not valid:
            self.progress.emit("No supported tabular files found.")
            self.finished.emit(uploaded, errors)
            return

        for index, path in enumerate(valid, start=1):
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            self.progress.emit(
                f"[{index}/{len(valid)}] Uploading {path.name} ({_fmt_bytes(size)})..."
            )
            try:
                payload = _multipart_upload(url, files={"file": path}, timeout=180.0)
            except urllib.error.HTTPError as exc:
                errors.append(f"{path.name}: {_http_error_detail(exc)}")
                self.progress.emit(f"[{index}/{len(valid)}] Upload failed for {path.name}.")
                continue
            except urllib.error.URLError as exc:
                reason = getattr(exc, "reason", exc)
                errors.append(f"{path.name}: connection failed: {reason}")
                self.progress.emit(f"[{index}/{len(valid)}] Service connection failed.")
                continue
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
                self.progress.emit(f"[{index}/{len(valid)}] Upload failed: {exc}")
                continue

            payload = dict(payload or {})
            slug = str(payload.get("slug") or "").strip()
            rel = str(payload.get("path") or "").strip()
            if not slug:
                errors.append(f"{path.name}: upload response did not include a dataset slug")
                continue
            uploaded.append((slug, payload))
            fmt = str(payload.get("source_format") or path.suffix.lstrip("."))
            target = rel or payload.get("filename") or f"{slug}.csv"
            self.progress.emit(
                f"[{index}/{len(valid)}] Stored '{slug}' ({fmt} -> {target})."
            )

        self.finished.emit(uploaded, errors)


class _JsonHttpWorker(QObject):
    """Generic GET/POST JSON worker so the panel can drive the tabular endpoints."""

    finished = pyqtSignal(str, object, str)  # tag, payload(dict|None), error

    def __init__(
        self,
        *,
        base_url: str,
        tag: str,
        method: str,
        path: str,
        body: Optional[dict] = None,
        timeout: float = 120.0,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._base_url = str(base_url or "").rstrip("/")
        self._tag = str(tag)
        self._method = str(method or "GET").upper()
        self._path = str(path)
        self._body = body
        self._timeout = float(timeout)

    def run(self) -> None:
        url = self._base_url + self._path
        data = None
        headers = {"Content-Type": "application/json"}
        if self._body is not None:
            data = json.dumps(self._body).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=data, method=self._method, headers=headers)
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
            payload = json.loads(raw) if raw else {}
            self.finished.emit(self._tag, payload, "")
        except urllib.error.HTTPError as exc:
            self.finished.emit(self._tag, None, _http_error_detail(exc))
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            self.finished.emit(self._tag, None, f"connection failed: {reason}")
        except Exception as exc:
            self.finished.emit(self._tag, None, str(exc))


class _IssueRow(QFrame):
    """One profile issue with severity, message, and an optional one-click fix."""

    fixRequested = pyqtSignal(object)  # op dict

    _SEV_COLOR = {
        "critical": "rgba(220, 90, 90, 0.95)",
        "warning": "rgba(214, 178, 92, 0.95)",
        "info": "rgba(146, 208, 218, 0.85)",
    }

    def __init__(self, issue: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        severity = str(issue.get("severity") or "info").lower()
        category = str(issue.get("category") or "").lower()
        message = str(issue.get("message") or "").strip()

        row = QHBoxLayout(self)
        row.setContentsMargins(2, 2, 2, 2)
        row.setSpacing(6)

        dot = QLabel("●")  # filled circle
        dot.setStyleSheet(f"color: {self._SEV_COLOR.get(severity, self._SEV_COLOR['info'])};")
        row.addWidget(dot)

        label = QLabel(f"[{severity.upper()}] {message}")
        label.setWordWrap(True)
        label.setStyleSheet("font-size: 10px; color: rgba(214,222,218,0.92);")
        row.addWidget(label, stretch=1)

        op = self._fix_op_for(category, message)
        if op is not None:
            btn = QPushButton(op.pop("_label", "Fix"))
            btn.setToolTip(op.get("_tip", ""))
            op.pop("_tip", None)
            btn.clicked.connect(lambda: self.fixRequested.emit(op))
            row.addWidget(btn)

    @staticmethod
    def _fix_op_for(category: str, message: str) -> Optional[dict]:
        """Map a profile issue to a safe one-click transform op (or None)."""
        if category == "duplicate":
            return {
                "_label": "Drop duplicates",
                "_tip": "Remove exact-duplicate rows.",
                "op": "drop_duplicate_rows",
            }
        if category == "missing":
            match = _QUOTED_RE.search(message)
            if match:
                col = match.group(1)
                return {
                    "_label": "Impute",
                    "_tip": f"Fill missing values in '{col}' (median for numeric, mode otherwise).",
                    "op": "impute_missing",
                    "columns": [col],
                    "strategy": "median",
                }
        # leakage and anything else: informational only.
        return None


class CollectTabularPanel(QFrame):
    """Onboard + prepare tabular datasets: multi-format ingest, profile-driven fixes,
    reproducible splits, and a transform history with undo.

    Keeps the original public surface (tabularDatasetUploaded, errorRaised,
    note_library_handoff) so the host window wiring is unchanged.
    """

    tabularDatasetUploaded = pyqtSignal(str)  # slug
    errorRaised = pyqtSignal(str)

    def __init__(self, *, base_url: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._base_url = str(base_url or "").rstrip("/")
        self._last_csv_path: Optional[Path] = None
        self._active_slug = ""
        self._columns: list[str] = []
        self._upload_thread: Optional[QThread] = None
        self._upload_worker: Optional[_TabularUploadWorker] = None
        self._json_thread: Optional[QThread] = None
        self._json_worker: Optional[_JsonHttpWorker] = None
        self._pending: list[tuple[str, str, str, Optional[dict]]] = []
        self._csv_windows: list[CsvTableEditorDialog] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        info = QLabel(
            "Onboard a tabular dataset (CSV, TSV, Excel, Parquet, JSON, JSONL). After upload, "
            "use Data Health to fix issues, create a reproducible train/val/test split, then set "
            "the label/feature columns in the Dataset Library to make it train-ready."
        )
        info.setWordWrap(True)
        info.setObjectName("stageInfo")
        layout.addWidget(info)

        layout.addWidget(self._build_ingest_section())
        layout.addWidget(self._build_health_section())
        layout.addWidget(self._build_target_section())
        layout.addWidget(self._build_split_section())
        layout.addWidget(self._build_score_section())
        layout.addWidget(self._build_history_section())

        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 10px; color: rgba(147,161,161,0.85);")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(96)
        self._log.document().setMaximumBlockCount(120)
        self._log.setPlaceholderText("Upload progress and service responses will appear here.")
        self._log.setStyleSheet(
            "QPlainTextEdit {"
            "font-size: 10px;"
            "color: rgba(214,222,218,0.90);"
            "background: rgba(7,11,12,0.62);"
            "border: 1px solid rgba(77,128,138,0.42);"
            "padding: 4px;"
            "}"
        )
        layout.addWidget(self._log)
        layout.addStretch(1)

        self._set_active_controls_enabled(False)

    # ------------------------------------------------------------------ #
    # Section builders
    # ------------------------------------------------------------------ #

    def _build_ingest_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Ingest", expanded=True)
        body = section.body_layout()

        self._drop = _CsvDropZone()
        self._drop.filesDropped.connect(self._upload_files)
        body.addWidget(self._drop)

        btn_row = QHBoxLayout()
        self._upload_btn = QPushButton("Upload File…")
        self._upload_btn.clicked.connect(self._pick_and_upload)
        btn_row.addWidget(self._upload_btn)
        self._folder_btn = QPushButton("Import Folder…")
        self._folder_btn.setToolTip("Batch-import every supported tabular file in a folder.")
        self._folder_btn.clicked.connect(self._pick_and_import_folder)
        btn_row.addWidget(self._folder_btn)
        self._edit_btn = QPushButton("Edit…")
        self._edit_btn.setToolTip("Open the active dataset in a CSV editor window.")
        self._edit_btn.clicked.connect(self._edit_last)
        btn_row.addWidget(self._edit_btn)
        self._visualize_btn = QPushButton("Visualize…")
        self._visualize_btn.setToolTip("Open the active dataset in a CSV visualization window.")
        self._visualize_btn.clicked.connect(self._visualize_last)
        btn_row.addWidget(self._visualize_btn)
        btn_row.addStretch(1)
        body.addLayout(btn_row)
        return section

    def _build_health_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Data Health", expanded=True)
        body = section.body_layout()

        header = QHBoxLayout()
        self._quality_label = QLabel("Quality: —")
        self._quality_label.setStyleSheet("font-size: 11px; font-weight: 600;")
        header.addWidget(self._quality_label)
        header.addStretch(1)
        self._refresh_health_btn = QPushButton("Refresh")
        self._refresh_health_btn.clicked.connect(self._refresh_profile)
        header.addWidget(self._refresh_health_btn)
        self._fix_dups_btn = QPushButton("Drop dup rows")
        self._fix_dups_btn.clicked.connect(lambda: self._apply_ops([{"op": "drop_duplicate_rows"}]))
        header.addWidget(self._fix_dups_btn)
        self._fix_constant_btn = QPushButton("Drop constant cols")
        self._fix_constant_btn.clicked.connect(lambda: self._apply_ops([{"op": "drop_constant_columns"}]))
        header.addWidget(self._fix_constant_btn)
        body.addLayout(header)

        self._issues_host = QWidget()
        self._issues_layout = QVBoxLayout(self._issues_host)
        self._issues_layout.setContentsMargins(0, 0, 0, 0)
        self._issues_layout.setSpacing(2)
        body.addWidget(self._issues_host)

        self._issues_empty = QLabel("Upload or select a dataset, then Refresh to scan for issues.")
        self._issues_empty.setStyleSheet("font-size: 10px; color: rgba(147,161,161,0.7);")
        self._issues_empty.setWordWrap(True)
        self._issues_layout.addWidget(self._issues_empty)
        return section

    def _build_target_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Target & Readiness", expanded=False)
        body = section.body_layout()

        row = QHBoxLayout()
        row.addWidget(QLabel("label column"))
        self._label_combo = QComboBox()
        self._label_combo.setMinimumWidth(160)
        row.addWidget(self._label_combo)
        self._analyze_btn = QPushButton("Analyze")
        self._analyze_btn.clicked.connect(self._analyze_target)
        row.addWidget(self._analyze_btn)
        self._balance_btn = QPushButton("Balance classes")
        self._balance_btn.setToolTip("Oversample minority classes for the selected label column.")
        self._balance_btn.clicked.connect(self._balance_classes)
        row.addWidget(self._balance_btn)
        row.addStretch(1)
        body.addLayout(row)

        self._readiness_label = QLabel("Pick a label column and Analyze to gate train-readiness.")
        self._readiness_label.setWordWrap(True)
        self._readiness_label.setStyleSheet("font-size: 11px; font-weight: 600;")
        body.addWidget(self._readiness_label)

        self._target_detail = QPlainTextEdit()
        self._target_detail.setReadOnly(True)
        self._target_detail.setMaximumHeight(110)
        self._target_detail.setStyleSheet(
            "QPlainTextEdit { font-size: 10px; color: rgba(214,222,218,0.88);"
            "background: rgba(7,11,12,0.5); border: 1px solid rgba(77,128,138,0.35); padding: 4px; }"
        )
        body.addWidget(self._target_detail)
        return section

    def _build_split_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Train / Val / Test Split", expanded=False)
        body = section.body_layout()

        row = QHBoxLayout()
        row.addWidget(QLabel("val"))
        self._val_spin = QDoubleSpinBox()
        self._val_spin.setRange(0.0, 0.9)
        self._val_spin.setSingleStep(0.05)
        self._val_spin.setValue(0.2)
        row.addWidget(self._val_spin)
        row.addWidget(QLabel("test"))
        self._test_spin = QDoubleSpinBox()
        self._test_spin.setRange(0.0, 0.9)
        self._test_spin.setSingleStep(0.05)
        self._test_spin.setValue(0.0)
        row.addWidget(self._test_spin)
        row.addWidget(QLabel("seed"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 999_999)
        self._seed_spin.setValue(42)
        row.addWidget(self._seed_spin)
        row.addStretch(1)
        body.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("stratify by"))
        self._stratify_combo = QComboBox()
        self._stratify_combo.addItem("(none)", "")
        self._stratify_combo.setMinimumWidth(160)
        row2.addWidget(self._stratify_combo)
        self._write_col_chk = QCheckBox("Write split column")
        row2.addWidget(self._write_col_chk)
        row2.addStretch(1)
        self._split_btn = QPushButton("Create Split")
        self._split_btn.clicked.connect(self._create_split)
        row2.addWidget(self._split_btn)
        body.addLayout(row2)
        return section

    def _build_score_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Batch Score", expanded=False)
        body = section.body_layout()

        info = QLabel(
            "Run a trained tabular model over this dataset. Give a scenario (uses its latest/"
            "prod model) or a model.pkl path. Results are written as a new dataset."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 10px; color: rgba(147,161,161,0.8);")
        body.addWidget(info)

        row = QHBoxLayout()
        row.addWidget(QLabel("scenario"))
        self._score_scenario = QLineEdit()
        self._score_scenario.setPlaceholderText("scenario name")
        row.addWidget(self._score_scenario, stretch=1)
        row.addWidget(QLabel("version"))
        self._score_version = QLineEdit()
        self._score_version.setPlaceholderText("prod / candidate / vN")
        self._score_version.setMaximumWidth(120)
        row.addWidget(self._score_version)
        body.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("or model.pkl"))
        self._score_model_path = QLineEdit()
        self._score_model_path.setPlaceholderText("path to model.pkl (overrides scenario)")
        row2.addWidget(self._score_model_path, stretch=1)
        self._score_btn = QPushButton("Score")
        self._score_btn.clicked.connect(self._run_score)
        row2.addWidget(self._score_btn)
        body.addLayout(row2)

        self._score_detail = QLabel("")
        self._score_detail.setWordWrap(True)
        self._score_detail.setStyleSheet("font-size: 10px; color: rgba(214,222,218,0.9);")
        body.addWidget(self._score_detail)
        return section

    def _build_history_section(self) -> CollapsibleSection:
        section = CollapsibleSection("Transform History", expanded=False)
        body = section.body_layout()

        row = QHBoxLayout()
        self._refresh_history_btn = QPushButton("Refresh")
        self._refresh_history_btn.clicked.connect(self._refresh_history)
        row.addWidget(self._refresh_history_btn)
        self._undo_btn = QPushButton("Undo Last")
        self._undo_btn.setToolTip("Restore the most recent backup (single-level undo).")
        self._undo_btn.clicked.connect(self._undo_last)
        row.addWidget(self._undo_btn)
        row.addStretch(1)
        body.addLayout(row)

        self._history_view = QPlainTextEdit()
        self._history_view.setReadOnly(True)
        self._history_view.setMaximumHeight(96)
        self._history_view.setPlaceholderText("No transforms recorded yet.")
        self._history_view.setStyleSheet(
            "QPlainTextEdit { font-size: 10px; color: rgba(214,222,218,0.88);"
            "background: rgba(7,11,12,0.5); border: 1px solid rgba(77,128,138,0.35); padding: 4px; }"
        )
        body.addWidget(self._history_view)
        return section

    # ------------------------------------------------------------------ #
    # Upload
    # ------------------------------------------------------------------ #

    def _pick_and_upload(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Upload tabular dataset", "", _FILE_DIALOG_FILTER
        )
        files = [f for f in files if str(f or "").strip()]
        if files:
            self._upload_files(files)

    def _pick_and_import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Import tabular folder", "")
        folder = str(folder or "").strip()
        if not folder:
            return
        self._append_log(f"Scanning {folder} for tabular files...")
        self._run_json(
            "import_folder", "POST", "/database/import_tabular_folder",
            {"source_path": folder, "recursive": True},
        )

    def _upload_files(self, files: list[str]) -> None:
        if self._upload_thread is not None:
            self._append_log("Upload already in progress; wait for the current upload to finish.")
            return
        paths = [str(f or "").strip() for f in files if str(f or "").strip()]
        if not paths:
            self._set_status("No tabular files selected.")
            return

        self._set_uploading(True)
        self._set_status(f"Uploading {len(paths)} tabular file(s)...")

        thread = QThread(self)
        worker = _TabularUploadWorker(base_url=self._base_url, files=paths)
        worker.moveToThread(thread)
        worker.progress.connect(self._append_log)
        worker.finished.connect(self._on_upload_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._clear_upload_worker(thread))
        thread.started.connect(worker.run)
        self._upload_thread = thread
        self._upload_worker = worker
        thread.start()

    def _on_upload_finished(self, uploaded_obj: object, errors_obj: object) -> None:
        uploaded = list(uploaded_obj or []) if isinstance(uploaded_obj, list) else []
        errors = list(errors_obj or []) if isinstance(errors_obj, list) else []
        self._set_uploading(False)

        if uploaded:
            slug, payload = uploaded[-1]
            self._set_active_dataset(str(slug or "").strip(), payload)

        if uploaded and not errors:
            names = ", ".join(str(s) for s, _ in uploaded[:4])
            if len(uploaded) > 4:
                names += "..."
            self._set_status(f"Uploaded {len(uploaded)} dataset(s): {names}.")
        elif uploaded or errors:
            parts: list[str] = []
            if uploaded:
                parts.append(f"Uploaded {len(uploaded)}")
            if errors:
                parts.append("Errors: " + " | ".join(str(e) for e in errors[:3]))
            self._set_status(". ".join(parts))
        else:
            self._set_status("No tabular datasets uploaded.")

        if errors:
            for err in errors[:5]:
                self._append_log(f"[ERROR] {err}")
            self.errorRaised.emit("; ".join(str(e) for e in errors[:3]))

        if uploaded:
            self.tabularDatasetUploaded.emit(str(uploaded[-1][0]))

    # ------------------------------------------------------------------ #
    # Active dataset + generic JSON calls
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_slug(value: str) -> str:
        """Reduce a slug-or-path to a bare dataset slug.

        Library selection can hand us a relative path like
        'mlops/datasets/foo.csv'; the REST routes take a slug segment, so a path
        would build an invalid URL (and collide with /database/{slug}/{name:path}).
        """
        name = str(value or "").strip()
        if not name:
            return ""
        if "/" in name or "\\" in name or name.lower().endswith((".csv", ".tsv")):
            name = Path(name).stem
        return name.strip()

    def set_active_dataset(self, slug: str) -> None:
        """Public entry: bind the panel to an existing tabular dataset slug."""
        name = self._normalize_slug(slug)
        if not name:
            return
        if name == self._active_slug:
            return  # already bound; avoid redundant refresh storms
        self._set_active_dataset(name, None)

    def _set_active_dataset(self, slug: str, payload: Optional[dict]) -> None:
        self._active_slug = self._normalize_slug(slug)
        if payload:
            self._last_csv_path = self._resolve_local_csv(str((payload or {}).get("path") or ""))
        else:
            self._last_csv_path = None
        self._set_active_controls_enabled(bool(self._active_slug))
        if self._active_slug:
            self._append_log(f"Active dataset: {self._active_slug}")
            self._refresh_profile()
            self._refresh_history()

    def _run_json(self, tag: str, method: str, path: str, body: Optional[dict] = None) -> None:
        # Serialize requests through a small queue instead of dropping concurrent
        # calls. Idempotent reads (profile/history/target) are de-duplicated so a
        # rapid bind sequence does not stack redundant refreshes.
        if tag in ("profile", "history", "target"):
            self._pending = [item for item in self._pending if item[0] != tag]
        self._pending.append((tag, method, path, body))
        self._pump_queue()

    def _pump_queue(self) -> None:
        if self._json_thread is not None or not self._pending:
            return
        tag, method, path, body = self._pending.pop(0)
        thread = QThread(self)
        worker = _JsonHttpWorker(
            base_url=self._base_url, tag=tag, method=method, path=path, body=body
        )
        worker.moveToThread(thread)
        worker.finished.connect(self._on_json_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._clear_json_worker(thread))
        thread.started.connect(worker.run)
        self._json_thread = thread
        self._json_worker = worker
        self._set_busy(True)
        thread.start()

    def _on_json_finished(self, tag: str, payload: object, error: str) -> None:
        self._set_busy(False)
        if error:
            self._append_log(f"[ERROR] {tag}: {error}")
            self.errorRaised.emit(error)
            return
        data = payload if isinstance(payload, dict) else {}
        handler: Optional[Callable[[dict], None]] = {
            "profile": self._render_profile,
            "transform": self._after_transform,
            "split": self._after_split,
            "history": self._render_history,
            "undo": self._after_undo,
            "import_folder": self._after_import_folder,
            "target": self._render_target,
            "score": self._after_score,
        }.get(tag)
        if handler is not None:
            handler(data)

    # ------------------------------------------------------------------ #
    # Profile / fixes
    # ------------------------------------------------------------------ #

    def _refresh_profile(self) -> None:
        if not self._active_slug:
            return
        self._run_json("profile", "GET", f"/database/{self._active_slug}/tabular_profile?max_rows=5000")

    def _render_profile(self, data: dict) -> None:
        score = data.get("quality_score")
        n_samples = data.get("n_samples")
        n_features = data.get("n_features")
        if isinstance(score, (int, float)):
            pct = int(round(float(score) * 100))
            self._quality_label.setText(
                f"Quality: {pct}%   ({n_samples} rows x {n_features} cols)"
            )
        else:
            self._quality_label.setText("Quality: —")

        # Columns power the stratify combo.
        cols = data.get("num_cols", []) + data.get("cat_cols", [])
        self._columns = [str(c) for c in cols if str(c).strip()]
        cur = self._stratify_combo.currentData()
        self._stratify_combo.blockSignals(True)
        self._stratify_combo.clear()
        self._stratify_combo.addItem("(none)", "")
        for col in self._columns:
            self._stratify_combo.addItem(col, col)
        idx = self._stratify_combo.findData(cur)
        if idx >= 0:
            self._stratify_combo.setCurrentIndex(idx)
        self._stratify_combo.blockSignals(False)

        # Label combo (Target section) — default to the last column (common target spot).
        cur_label = self._label_combo.currentData()
        self._label_combo.blockSignals(True)
        self._label_combo.clear()
        for col in self._columns:
            self._label_combo.addItem(col, col)
        lidx = self._label_combo.findData(cur_label)
        if lidx >= 0:
            self._label_combo.setCurrentIndex(lidx)
        elif self._columns:
            self._label_combo.setCurrentIndex(self._label_combo.count() - 1)
        self._label_combo.blockSignals(False)

        # Render issues.
        self._clear_issues()
        issues = data.get("issues") if isinstance(data.get("issues"), list) else []
        if not issues:
            lbl = QLabel("No issues detected.")
            lbl.setStyleSheet("font-size: 10px; color: rgba(133,187,101,0.9);")
            self._issues_layout.addWidget(lbl)
            return
        for issue in issues[:40]:
            if not isinstance(issue, dict):
                continue
            row = _IssueRow(issue)
            row.fixRequested.connect(self._apply_single_op)
            self._issues_layout.addWidget(row)

    def _clear_issues(self) -> None:
        while self._issues_layout.count():
            item = self._issues_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _apply_single_op(self, op: object) -> None:
        if isinstance(op, dict):
            self._apply_ops([op])

    def _apply_ops(self, ops: list[dict]) -> None:
        if not self._active_slug:
            self._set_status("Select a dataset first.")
            return
        if not ops:
            return
        self._append_log(f"Applying {len(ops)} transform op(s)...")
        self._run_json("transform", "POST", f"/database/{self._active_slug}/tabular_transform", {"ops": ops})

    def _after_transform(self, data: dict) -> None:
        before = data.get("before") or {}
        after = data.get("after") or {}
        self._append_log(
            f"Transform done: {before.get('rows', '?')}x{before.get('cols', '?')} -> "
            f"{after.get('rows', '?')}x{after.get('cols', '?')} (rev {data.get('revision', '?')})."
        )
        self._set_status("Transform applied.")
        self._refresh_profile()
        self._refresh_history()

    # ------------------------------------------------------------------ #
    # Target & readiness
    # ------------------------------------------------------------------ #

    def _selected_label(self) -> str:
        return str(self._label_combo.currentData() or self._label_combo.currentText() or "").strip()

    def _analyze_target(self) -> None:
        if not self._active_slug:
            self._set_status("Select a dataset first.")
            return
        label = self._selected_label()
        if not label:
            self._set_status("Choose a label column to analyze.")
            return
        self._append_log(f"Analyzing target '{label}'...")
        self._run_json(
            "target", "GET",
            f"/database/{self._active_slug}/tabular_target?label_col={label}",
        )

    def _render_target(self, data: dict) -> None:
        task = str(data.get("task") or "unknown")
        readiness = data.get("readiness") if isinstance(data.get("readiness"), dict) else {}
        ready = bool(readiness.get("ready"))
        blockers = readiness.get("blockers") if isinstance(readiness.get("blockers"), list) else []
        warnings = readiness.get("warnings") if isinstance(readiness.get("warnings"), list) else []

        verdict = "READY to train" if ready else "NOT train-ready"
        color = "rgba(133,187,101,0.95)" if ready else "rgba(220,90,90,0.95)"
        self._readiness_label.setText(f"{verdict}  —  task: {task}")
        self._readiness_label.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {color};")

        lines: list[str] = []
        cb = data.get("class_balance") if isinstance(data.get("class_balance"), dict) else {}
        if cb:
            lines.append(
                f"classes: {cb.get('n_classes', '?')}  majority={cb.get('majority', '?')} "
                f"minority={cb.get('minority', '?')} ratio={cb.get('imbalance_ratio', '?')}"
            )
        classes = data.get("classes") if isinstance(data.get("classes"), list) else []
        if classes:
            top = "  ".join(f"{c.get('value')}={c.get('count')}" for c in classes[:6])
            lines.append(f"top: {top}")
        leakage = data.get("leakage") if isinstance(data.get("leakage"), list) else []
        if leakage:
            lines.append("leakage: " + ", ".join(str(it.get("feature")) for it in leakage[:6]))
        for b in blockers:
            lines.append(f"[BLOCKER] {b}")
        for w in warnings:
            lines.append(f"[warn] {w}")
        self._target_detail.setPlainText("\n".join(lines) if lines else "No issues.")
        self._set_status(verdict)

    def _balance_classes(self) -> None:
        if not self._active_slug:
            self._set_status("Select a dataset first.")
            return
        label = self._selected_label()
        if not label:
            self._set_status("Choose a label column to balance.")
            return
        self._apply_ops([{"op": "balance_classes", "label_col": label, "strategy": "oversample"}])

    # ------------------------------------------------------------------ #
    # Split
    # ------------------------------------------------------------------ #

    def _create_split(self) -> None:
        if not self._active_slug:
            self._set_status("Select a dataset first.")
            return
        body = {
            "val_frac": float(self._val_spin.value()),
            "test_frac": float(self._test_spin.value()),
            "seed": int(self._seed_spin.value()),
            "stratify_col": str(self._stratify_combo.currentData() or ""),
            "write_column": bool(self._write_col_chk.isChecked()),
        }
        self._append_log("Creating split...")
        self._run_json("split", "POST", f"/database/{self._active_slug}/tabular_split", body)

    def _after_split(self, data: dict) -> None:
        counts = data.get("counts") or {}
        strat = " (stratified)" if data.get("stratified") else ""
        msg = (
            f"Split{strat}: train={counts.get('train', '?')}, "
            f"val={counts.get('val', '?')}, test={counts.get('test', '?')}."
        )
        self._append_log(msg)
        self._set_status(msg)
        if data.get("wrote_column"):
            self._refresh_history()

    # ------------------------------------------------------------------ #
    # Batch score
    # ------------------------------------------------------------------ #

    def _run_score(self) -> None:
        if not self._active_slug:
            self._set_status("Select a dataset to score.")
            return
        scenario = self._score_scenario.text().strip()
        model_path = self._score_model_path.text().strip()
        if not scenario and not model_path:
            self._set_status("Provide a scenario or a model.pkl path.")
            return
        body = {
            "scenario": scenario,
            "version": self._score_version.text().strip(),
            "model_path": model_path,
            "write_dataset": True,
        }
        self._append_log("Scoring dataset...")
        self._run_json("score", "POST", f"/database/{self._active_slug}/tabular_score", body)

    def _after_score(self, data: dict) -> None:
        n = data.get("n_rows", "?")
        task = data.get("task", "?")
        written = str(data.get("written_slug") or "")
        sample = data.get("sample") if isinstance(data.get("sample"), list) else []
        preview = ", ".join(str(s) for s in sample[:8])
        msg = f"Scored {n} row(s) (task={task})."
        if written:
            msg += f" Wrote '{written}'."
        self._score_detail.setText(msg + (f"\nfirst: {preview}" if preview else ""))
        self._append_log(msg)
        self._set_status(msg)
        if written:
            self.tabularDatasetUploaded.emit(written)

    # ------------------------------------------------------------------ #
    # History / undo
    # ------------------------------------------------------------------ #

    def _refresh_history(self) -> None:
        if not self._active_slug:
            return
        self._run_json("history", "GET", f"/database/{self._active_slug}/tabular_history")

    def _render_history(self, data: dict) -> None:
        entries = data.get("entries") if isinstance(data.get("entries"), list) else []
        if not entries:
            self._history_view.setPlainText("No transforms recorded yet.")
        else:
            lines = []
            for e in entries[-20:]:
                if not isinstance(e, dict):
                    continue
                rev = e.get("revision", "?")
                action = e.get("action", "?")
                at = str(e.get("at", ""))[:19].replace("T", " ")
                if action == "transform":
                    ops = e.get("ops") if isinstance(e.get("ops"), list) else []
                    names = ", ".join(str(o.get("op")) for o in ops if isinstance(o, dict))
                    lines.append(f"r{rev}  {at}  {names}")
                else:
                    lines.append(f"r{rev}  {at}  {action}")
            self._history_view.setPlainText("\n".join(lines))
        self._undo_btn.setEnabled(bool(data.get("can_undo")) and bool(self._active_slug))

    def _undo_last(self) -> None:
        if not self._active_slug:
            return
        self._append_log("Undoing last transform...")
        self._run_json("undo", "POST", f"/database/{self._active_slug}/tabular_undo", {})

    def _after_undo(self, data: dict) -> None:
        after = data.get("after") or {}
        self._append_log(
            f"Reverted to {after.get('rows', '?')}x{after.get('cols', '?')} (rev {data.get('revision', '?')})."
        )
        self._set_status("Undo complete.")
        self._refresh_profile()
        self._refresh_history()

    def _after_import_folder(self, data: dict) -> None:
        imported = data.get("imported") if isinstance(data.get("imported"), list) else []
        found = data.get("found", 0)
        errors = data.get("errors") if isinstance(data.get("errors"), list) else []
        self._append_log(f"Imported {len(imported)} of {found} file(s) from folder.")
        for err in errors[:5]:
            self._append_log(f"[ERROR] {err}")
        if imported:
            last = imported[-1] if isinstance(imported[-1], dict) else {}
            self._set_active_dataset(str(last.get("slug") or "").strip(), last)
            self._set_status(f"Folder import: {len(imported)} dataset(s) added.")
            self.tabularDatasetUploaded.emit(str(last.get("slug") or ""))
        else:
            self._set_status(f"Folder import found {found} file(s) but imported none.")
        if errors:
            self.errorRaised.emit("; ".join(str(e) for e in errors[:3]))

    # ------------------------------------------------------------------ #
    # CSV editor windows
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_local_csv(rel: str) -> Optional[Path]:
        rel = str(rel or "").strip()
        if not rel:
            return None
        path = Path(rel)
        if not path.is_absolute():
            path = (Path(ROOT_DIR) / path).resolve()
        return path if path.is_file() else None

    def _edit_last(self) -> None:
        self._open_last_csv_window("editor")

    def _visualize_last(self) -> None:
        self._open_last_csv_window("visualize")

    def _open_last_csv_window(self, mode: str) -> None:
        if self._last_csv_path is None:
            self._set_status("Upload or select a dataset before opening it.")
            return
        dlg = CsvTableEditorDialog(csv_path=self._last_csv_path, parent=None)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        if mode == "visualize":
            dlg.show_visualization()
            self._append_log(f"Opened visualization window for {self._last_csv_path.name}.")
        else:
            self._append_log(f"Opened editor window for {self._last_csv_path.name}.")
        self._csv_windows.append(dlg)
        dlg.destroyed.connect(lambda _obj=None, window=dlg: self._forget_csv_window(window))
        dlg.show()
        try:
            dlg.raise_()
            dlg.activateWindow()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Host bridge + state helpers
    # ------------------------------------------------------------------ #

    def note_library_handoff(self, slug: str, *, ok: bool, detail: str = "") -> None:
        name = str(slug or "").strip()
        if ok:
            self._append_log(f"Dataset Library selected '{name}' for schema/label setup.")
            self._set_status(f"Uploaded '{name}'. Data Health, Split, and History are available.")
            return
        msg = detail or "Dataset Library selection failed."
        self._append_log(f"[ERROR] {msg}")
        self._set_status(f"Uploaded '{name}', but library handoff failed: {msg}")

    def _append_log(self, message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        self._log.appendPlainText(text)

    def _set_status(self, text: str) -> None:
        self._status.setText(str(text or ""))

    def _set_uploading(self, uploading: bool) -> None:
        self._upload_btn.setEnabled(not uploading)
        self._folder_btn.setEnabled(not uploading)
        self._drop.setEnabled(not uploading)

    def _set_busy(self, busy: bool) -> None:
        for btn in (
            self._refresh_health_btn, self._fix_dups_btn, self._fix_constant_btn,
            self._analyze_btn, self._balance_btn, self._score_btn,
            self._split_btn, self._refresh_history_btn, self._undo_btn,
        ):
            btn.setEnabled((not busy) and bool(self._active_slug))

    def _set_active_controls_enabled(self, enabled: bool) -> None:
        for btn in (
            self._edit_btn, self._visualize_btn, self._refresh_health_btn,
            self._fix_dups_btn, self._fix_constant_btn, self._analyze_btn,
            self._balance_btn, self._score_btn, self._split_btn,
            self._refresh_history_btn, self._undo_btn,
        ):
            btn.setEnabled(enabled)

    def _clear_upload_worker(self, thread: QThread) -> None:
        if self._upload_thread is thread:
            self._upload_thread = None
            self._upload_worker = None

    def _clear_json_worker(self, thread: QThread) -> None:
        if self._json_thread is thread:
            self._json_thread = None
            self._json_worker = None
        self._pump_queue()

    def _forget_csv_window(self, window: CsvTableEditorDialog) -> None:
        try:
            self._csv_windows.remove(window)
        except ValueError:
            pass
