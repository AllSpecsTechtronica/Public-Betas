"""PyQt6 panel for 3D model generation.

Product direction: native TRELLIS.2 Apple MLX (see ``mlops/three_d/trellis2-apple-main``)
is the primary image-to-mesh path on macOS (Apple Silicon / MPS). Cloud TRELLIS.2 on Hugging Face
remains an auxiliary backup. ComfyUI-Trellis2 is kept as an optional legacy adapter. Separately: replication / COLMAP, true Gaussian splatting through
Nerfstudio Splatfacto, and DepthAnything heightfield GLB (Core ML).

Implemented today:
- Quick samples: each queued single-image job also copies the input into ``~/.trellis2/cvops_quick_samples``
  for thumbnail reuse from the 3D Gen panel (no file dialog).
- Native Apple MLX (TRELLIS.2): call local ``trellis2-apple-main/api_server.py`` at ``http://127.0.0.1:8082``.
- Local ComfyUI (TRELLIS.2): optional legacy workflow bridge against ``http://127.0.0.1:8188``.
- Replication inputs: image folder or video (ffmpeg frame extraction), optional COLMAP import or
  ``colmap automatic_reconstructor``, workspace under ``~/.gaussian_splat/jobs/<id>/``.
- Cloud (Hugging Face): TRELLIS.2 image-to-3D via the hosted Space (mesh/GLB), optional backup.
- Local depth (DepthAnything): Core ML depth to heightfield GLB from
  ``Insight_assets/models/DepthAnythingModelSmall/*.mlpackage`` (macOS).
- Local (CUDA): reserved for future native TRELLIS on Linux/Windows+NVIDIA.

Threading model: heavy work runs in a daemon thread. Progress is persisted via JobStore; a QTimer polls
status.json on the main thread.
"""

from __future__ import annotations

import logging
import json
import html
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import threading
import time
from typing import Optional
from types import SimpleNamespace

from PyQt6.QtCore import QSize, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QIcon, QPixmap, QStandardItemModel
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .collapsible_section import CollapsibleSection
from .cvops_theme import repolish

log = logging.getLogger(__name__)

_OS_LABEL = {"darwin": "macOS", "windows": "Windows", "linux": "Linux"}
_THUMB_SIZE = 120
_QUICK_SAMPLE_ICON = 72
_QUICK_SAMPLE_MAX_FILES = 100
_QUICK_SAMPLE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})


class _AssetBrowserWorker(QThread):
    """Fetch the 3D asset list off the main thread to avoid blocking during training."""

    ready = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, db_root: Path, parent=None) -> None:
        super().__init__(parent)
        self._db_root = db_root

    def run(self) -> None:
        try:
            from mlops.three_d import ThreeDAssetStore  # noqa: PLC0415
            assets = ThreeDAssetStore(self._db_root).list_assets()
            self.ready.emit(list(assets))
        except Exception as exc:
            self.failed.emit(str(exc))


class _QuickSamplesWorker(QThread):
    """Scan the quick-samples directory off the main thread; emits resolved path strings."""

    ready = pyqtSignal(list)

    def __init__(self, dir_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._dir_path = dir_path

    def run(self) -> None:
        try:
            paths = sorted(
                (p for p in self._dir_path.iterdir() if p.is_file() and p.suffix.lower() in _QUICK_SAMPLE_SUFFIXES),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            self.ready.emit([str(p.resolve()) for p in paths[:_QUICK_SAMPLE_MAX_FILES]])
        except Exception:
            self.ready.emit([])


class ThreeDPanel(QWidget):
    """Native PyQt6 port of the Streamlit ``three_d_panel.render()`` function."""

    generationCompleted = pyqtSignal(str)   # glb_path
    errorRaised = pyqtSignal(str)
    replicationWorkspaceReady = pyqtSignal(str)
    _replication_finished = pyqtSignal(bool, str)
    _replication_status = pyqtSignal(str, str, float)
    _nerfstudio_prepare_finished = pyqtSignal(bool, str, object)
    _nerfstudio_prepare_status = pyqtSignal(str, str, float)

    def __init__(self, *, preloaded: Optional[dict] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        if preloaded is not None:
            # Fast path: caps/store/depth were already resolved by the deferred-page
            # preload step, so no blocking imports or GPU-detection happen here.
            self._caps = preloaded["caps"]
            self._store = preloaded["store"]
            self._defaults = preloaded["defaults"]
            self._depth_mlpackage = preloaded["depth_mlpackage"]
            self._preloaded_apple_ok: Optional[bool] = preloaded.get("apple_ok")
        else:
            # Fallback: synchronous init (used when panel is built without preload).
            from mlops.trellis2 import DEFAULT_PARAMS, JobStore, detect  # noqa: PLC0415
            from mlops.trellis2.depth_anything_local import resolve_depth_mlpackage  # noqa: PLC0415
            self._caps = detect()
            self._store = JobStore()
            self._defaults = DEFAULT_PARAMS
            self._depth_mlpackage = resolve_depth_mlpackage()
            self._preloaded_apple_ok = None

        self._three_d_db_root = Path(__file__).resolve().parents[4] / "database" / "3D"

        self._image_path: Optional[Path] = None
        self._repl_media_path: Optional[Path] = None
        self._repl_colmap_sparse: Optional[Path] = None
        self._asset_source_path: Optional[Path] = None
        self._last_replication_workspace = ""
        self._last_replication_depth_assist_dir = ""
        self._replication_log_lines: list[str] = []
        self._repl_render_requested = False
        self._repl_prep_busy = False
        self._ns_prepare_busy = False
        self._active_job_id: Optional[str] = None

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1000)
        self._poll_timer.timeout.connect(self._poll_job)

        self._build_ui()
        self._replication_finished.connect(self._on_replication_finished_slot)
        self._replication_status.connect(self._on_replication_status_slot)
        self._nerfstudio_prepare_finished.connect(self._on_nerfstudio_prepare_finished_slot)
        self._nerfstudio_prepare_status.connect(self._on_nerfstudio_prepare_status_slot)
        self._refresh_caps_banner()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_section_title(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("threeDSectionTitle")
        return lab

    def _build_comfy_local_block(self) -> QWidget:
        from mlops.three_d.comfy_local import (  # noqa: PLC0415
            default_comfy_base_url,
            list_bundled_workflows,
        )

        cell = QFrame()
        cell.setObjectName("opsCell")
        cell.setFrameShape(QFrame.Shape.StyledPanel)
        vl = QVBoxLayout(cell)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(8)
        url_row = QHBoxLayout()
        url_lbl = QLabel("ComfyUI URL")
        url_lbl.setProperty("muted", True)
        repolish(url_lbl)
        url_row.addWidget(url_lbl)
        self._comfy_url_edit = QLineEdit()
        self._comfy_url_edit.setPlaceholderText(default_comfy_base_url())
        self._comfy_url_edit.setClearButtonEnabled(True)
        url_row.addWidget(self._comfy_url_edit, stretch=1)
        vl.addLayout(url_row)
        wf_row = QHBoxLayout()
        wf_lbl = QLabel("Workflow")
        wf_lbl.setProperty("muted", True)
        repolish(wf_lbl)
        wf_row.addWidget(wf_lbl)
        self._comfy_workflow_combo = QComboBox()
        self._comfy_workflow_combo.setMinimumWidth(220)
        for p in list_bundled_workflows():
            self._comfy_workflow_combo.addItem(p.name, p)
        if self._comfy_workflow_combo.count() == 0:
            self._comfy_workflow_combo.addItem("(no JSON in example_workflows — add ComfyUI-Trellis2-main)", None)
            self._comfy_workflow_combo.model().item(0).setEnabled(False)
        else:
            for i in range(self._comfy_workflow_combo.count()):
                data = self._comfy_workflow_combo.itemData(i)
                if data is not None and Path(data).name == "MeshOnly.json":
                    self._comfy_workflow_combo.setCurrentIndex(i)
                    break
        wf_row.addWidget(self._comfy_workflow_combo, stretch=1)
        vl.addLayout(wf_row)
        ping_row = QHBoxLayout()
        ping_btn = QPushButton("Test ComfyUI connection")
        ping_btn.setProperty("buttonRole", "secondary")
        repolish(ping_btn)
        ping_btn.clicked.connect(self._on_comfy_test_connection)
        ping_row.addWidget(ping_btn)
        self._comfy_conn_status = QLabel("")
        self._comfy_conn_status.setWordWrap(True)
        self._comfy_conn_status.setProperty("muted", True)
        repolish(self._comfy_conn_status)
        ping_row.addWidget(self._comfy_conn_status, stretch=1)
        vl.addLayout(ping_row)
        tools_row = QHBoxLayout()
        tools_row.setSpacing(8)
        open_repo_btn = QPushButton("Open Trellis2 repo")
        open_repo_btn.setProperty("buttonRole", "secondary")
        repolish(open_repo_btn)
        open_repo_btn.clicked.connect(self._on_open_comfy_repo)
        tools_row.addWidget(open_repo_btn)
        open_readme_btn = QPushButton("Open setup README")
        open_readme_btn.setProperty("buttonRole", "secondary")
        repolish(open_readme_btn)
        open_readme_btn.clicked.connect(self._on_open_comfy_setup_readme)
        tools_row.addWidget(open_readme_btn)
        copy_url_btn = QPushButton("Copy default URL")
        copy_url_btn.setProperty("buttonRole", "secondary")
        repolish(copy_url_btn)
        copy_url_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(default_comfy_base_url())
        )
        tools_row.addWidget(copy_url_btn)
        tools_row.addStretch()
        vl.addLayout(tools_row)
        return cell

    def _nav_make_anchor(self, nav_id: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("threeDNavAnchor")
        frame.setProperty("navId", nav_id)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        self._nav_anchors[nav_id] = frame
        return frame, lay

    def _nav_fill_outline(self) -> None:
        from PyQt6.QtGui import QFont, QFontDatabase  # noqa: PLC0415

        entries: list[tuple[str, str, str, str]] = [
            (
                "overview",
                "Overview",
                "Workspace",
                "Product copy + roadmap. Parent: 3D Gen tab (no upstream job).",
            ),
            (
                "catalog",
                "Catalog / Assets",
                "Workspace",
                "Inherits: mlops.three_d.ThreeDAssetStore, database/3D. Downstream: nerfstudio prep (separate).",
            ),
            (
                "pipeline_mode",
                "Pipeline mode",
                "Pipeline",
                "Chooses branch: single mesh vs replication. Mutually exclusive downstream rails.",
            ),
            (
                "pipeline_single",
                "Single source",
                "Pipeline",
                "Feeds: ~/.trellis2/jobs/<id> via trellis2 JobStore; backends Comfy | HF | depth share SamplingParams.",
            ),
            (
                "pipeline_repl",
                "Replication",
                "Pipeline",
                "Feeds: ~/.gaussian_splat/jobs/<id> via mlops.gaussian_splat.replication (non-Trellis path).",
            ),
            (
                "runtime_host",
                "Host",
                "Runtime",
                "Inherits: mlops.trellis2.detect() caps for this process.",
            ),
            (
                "runtime_sampling",
                "Sampling",
                "Runtime",
                "Inherits: mlops.trellis2.SamplingParams (same struct as HF gradio client sparse stage).",
            ),
            (
                "runtime_run",
                "Run pipeline",
                "Runtime",
                "Selects backend id → _start_job_thread (same JobStore persistence as Streamlit-era trellis2).",
            ),
            (
                "output_status",
                "Job status",
                "Output",
                "Reads: JobStore status.json for active job id (polled from main thread).",
            ),
            (
                "output_result",
                "Result",
                "Output",
                "GLB + optional HTML preview paths written by completed job.",
            ),
            (
                "output_history",
                "Recent jobs",
                "Output",
                "Lists recent JobStore rows (same root ~/.trellis2/jobs).",
            ),
        ]
        self._nav_tree.clear()
        mono = QFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        if mono.pointSize() > 0:
            mono.setPointSize(max(9, mono.pointSize() - 1))
        self._nav_tree.setFont(mono)
        groups: dict[str, QTreeWidgetItem] = {}
        for nid, label, group, tip in entries:
            parent = groups.get(group)
            if parent is None:
                parent = QTreeWidgetItem([group])
                parent.setData(0, Qt.ItemDataRole.UserRole, "")
                parent.setExpanded(True)
                self._nav_tree.addTopLevelItem(parent)
                groups[group] = parent
            it = QTreeWidgetItem([label])
            it.setData(0, Qt.ItemDataRole.UserRole, nid)
            it.setToolTip(0, tip)
            parent.addChild(it)
        self._nav_tree.expandAll()

    def _nav_apply_breadcrumb(self, nav_id: str) -> None:
        crumbs = {
            "overview": "3D › Workspace › Overview",
            "catalog": "3D › Workspace › Catalog › Asset staging",
            "pipeline_mode": "3D › Workspace › Pipeline › Mode",
            "pipeline_single": "3D › Workspace › Pipeline › Single › Source image",
            "pipeline_repl": "3D › Workspace › Pipeline › Replication",
            "runtime_host": "3D › Workspace › Runtime › Host",
            "runtime_sampling": "3D › Workspace › Runtime › Sampling",
            "runtime_run": "3D › Workspace › Runtime › Run",
            "output_status": "3D › Workspace › Output › Job status",
            "output_result": "3D › Workspace › Output › Result",
            "output_history": "3D › Workspace › Output › Recent jobs",
        }
        self._nav_breadcrumb.setText(crumbs.get(nav_id, "3D › Workspace"))

    def _on_nav_outline_item(self, item: QTreeWidgetItem | None) -> None:
        if item is None:
            return
        nid = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(nid, str):
            return
        if not nid:
            return
        w = getattr(self, "_nav_anchors", {}).get(nid)
        if w is not None:
            self._controls_scroll.ensureWidgetVisible(w)
        self._nav_apply_breadcrumb(nid)

    def _nav_sync_outline_branch_styles(self) -> None:
        from PyQt6.QtGui import QBrush, QColor  # noqa: PLC0415

        is_single = self._mode_combo.currentData() == "single"
        self._nav_context.setText(
            "Active pipeline branch: Single mesh (trellis2 / Comfy)"
            if is_single
            else "Active pipeline branch: Replication (gaussian_splat prep)"
        )
        muted = QBrush(QColor(118, 122, 128))
        normal = QBrush()
        root = self._nav_tree.invisibleRootItem()
        for i in range(root.childCount()):
            group = root.child(i)
            for j in range(group.childCount()):
                it = group.child(j)
                nid = it.data(0, Qt.ItemDataRole.UserRole)
                if nid == "pipeline_single":
                    it.setForeground(0, normal if is_single else muted)
                elif nid == "pipeline_repl":
                    it.setForeground(0, muted if is_single else normal)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        inner = QWidget()
        inner.setObjectName("threeDPanel")
        root = QVBoxLayout(inner)
        root.setContentsMargins(4, 0, 12, 0)
        root.setSpacing(14)

        title = QLabel("3D generation (Gaussian-first)")
        title.setProperty("isTitle", True)
        repolish(title)
        root.addWidget(title)
        self._mesh_blocker_note = QLabel(
            "Native TRELLIS.2 Apple MLX is available via mlops/three_d/trellis2-apple-main. "
            "Start its API server on 127.0.0.1:8082, then use Pipeline mode = Single source and "
            "Backend = Native Trellis2 Apple (MLX) for image-to-GLB generation.\n\n"
            "ComfyUI-Trellis2 remains as a legacy bridge, but the direct MLX API is now the macOS path. "
            "Replication is still the route for multi-view Gaussian workspace prep."
        )
        self._mesh_blocker_note.setWordWrap(True)
        self._mesh_blocker_note.setProperty("state", "success")
        repolish(self._mesh_blocker_note)
        root.addWidget(self._mesh_blocker_note)

        workspace = QSplitter(Qt.Orientation.Horizontal)
        workspace.setObjectName("threeDWorkspace")
        workspace.setChildrenCollapsible(False)
        workspace.setHandleWidth(4)
        root.addWidget(workspace, stretch=1)

        self._nav_anchors = {}

        controls_column = QWidget()
        controls_column.setObjectName("threeDControlsColumn")
        cc_lay = QVBoxLayout(controls_column)
        cc_lay.setContentsMargins(0, 0, 0, 0)
        cc_lay.setSpacing(6)

        self._nav_breadcrumb = QLabel("3D › Workspace › Overview")
        self._nav_breadcrumb.setObjectName("threeDBreadcrumb")
        self._nav_breadcrumb.setWordWrap(True)
        repolish(self._nav_breadcrumb)
        cc_lay.addWidget(self._nav_breadcrumb)

        self._nav_context = QLabel("Active pipeline branch: Single mesh")
        self._nav_context.setObjectName("threeDNavContext")
        self._nav_context.setProperty("muted", True)
        self._nav_context.setWordWrap(True)
        repolish(self._nav_context)
        cc_lay.addWidget(self._nav_context)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(8)
        self._nav_tree = QTreeWidget()
        self._nav_tree.setObjectName("threeDNavOutline")
        self._nav_tree.setHeaderHidden(True)
        self._nav_tree.setRootIsDecorated(True)
        self._nav_tree.setUniformRowHeights(True)
        self._nav_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._nav_tree.setMinimumWidth(180)
        self._nav_tree.setMaximumWidth(260)
        nav_row.addWidget(self._nav_tree)

        controls_scroll = QScrollArea()
        controls_scroll.setObjectName("threeDControlsRail")
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setMinimumWidth(320)
        self._controls_scroll = controls_scroll

        controls_inner = QWidget()
        controls_inner.setObjectName("threeDControlsInner")
        controls_root = QVBoxLayout(controls_inner)
        controls_root.setContentsMargins(0, 0, 10, 0)
        controls_root.setSpacing(12)

        ov_fr, ov_lay = self._nav_make_anchor("overview")
        roadmap = CollapsibleSection("Local 3D roadmap (Gaussian splatting)", expanded=False)
        roadmap_body = QLabel(
            "Local 3D pipeline targets:\n"
            "[Replication] Multi-view or video captures with pose recovery, then splat optimization.\n"
            "[Point clouds] Fused depth / MVS clouds as .ply (interchange before or beside splats).\n"
            "[Gaussian splatting] Primary radiance output for modeling and viewing (training/export wired next).\n"
            "DepthAnything + Gaussian is the active default path in this tab."
        )
        roadmap_body.setWordWrap(True)
        roadmap_body.setProperty("muted", True)
        repolish(roadmap_body)
        roadmap.body_layout().addWidget(roadmap_body)
        ov_lay.addWidget(roadmap)
        controls_root.addWidget(ov_fr)

        cat_fr, cat_lay = self._nav_make_anchor("catalog")
        controls_title = QLabel("Objects and configs")
        controls_title.setProperty("isTitle", True)
        repolish(controls_title)
        cat_lay.addWidget(controls_title)

        self._asset_shell = self._build_cvops_asset_shell()
        cat_lay.addWidget(self._asset_shell)

        self._asset_browser = self._build_asset_browser()
        cat_lay.addWidget(self._asset_browser)
        controls_root.addWidget(cat_fr)

        pm_fr, pm_lay = self._nav_make_anchor("pipeline_mode")
        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Pipeline mode")
        mode_lbl.setProperty("muted", True)
        repolish(mode_lbl)
        mode_row.addWidget(mode_lbl)
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Replication (Gaussian · image/video + COLMAP)", "replication")
        self._mode_combo.addItem("Single image (mesh / depth)", "single")
        self._mode_combo.currentIndexChanged.connect(self._on_pipeline_mode_changed)
        mode_row.addWidget(self._mode_combo, stretch=1)
        pm_wrap = QWidget()
        pm_wrap.setLayout(mode_row)
        pm_lay.addWidget(pm_wrap)
        controls_root.addWidget(pm_fr)

        ps_fr, ps_lay = self._nav_make_anchor("pipeline_single")
        self._single_block = QWidget()
        _single_lay = QVBoxLayout(self._single_block)
        _single_lay.setContentsMargins(0, 0, 0, 0)
        _single_lay.setSpacing(8)
        _single_lay.addWidget(self._make_section_title("Source image"))

        img_cell = QFrame()
        img_cell.setObjectName("opsCell")
        img_cell.setFrameShape(QFrame.Shape.StyledPanel)
        img_row = QHBoxLayout(img_cell)
        img_row.setContentsMargins(8, 8, 8, 8)
        img_row.setSpacing(12)

        self._preview_thumb = QLabel()
        self._preview_thumb.setObjectName("overlayPreview")
        self._preview_thumb.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
        self._preview_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_thumb.setText("No image")
        img_row.addWidget(self._preview_thumb)

        img_info = QVBoxLayout()
        img_info.setSpacing(6)
        self._select_btn = QPushButton("Select image…")
        self._select_btn.setProperty("buttonRole", "secondary")
        repolish(self._select_btn)
        self._select_btn.clicked.connect(self._on_select_image)
        self._filename_label = QLabel("No file selected")
        self._filename_label.setProperty("muted", True)
        self._filename_label.setWordWrap(True)
        img_info.addWidget(self._select_btn)
        img_info.addWidget(self._filename_label)
        img_info.addStretch()
        img_row.addLayout(img_info, stretch=1)

        _single_lay.addWidget(img_cell)

        quick_section = CollapsibleSection("Quick samples", expanded=True)
        quick_hint = QLabel(
            "Each time you run Generate 3D, the source image is copied here so you can pick it again "
            "without the file dialog — useful for retries and backend comparisons."
        )
        quick_hint.setWordWrap(True)
        quick_hint.setProperty("muted", True)
        repolish(quick_hint)
        quick_section.body_layout().addWidget(quick_hint)

        qs_row = QHBoxLayout()
        qs_row.setSpacing(8)
        self._quick_samples_open_btn = QPushButton("Open folder")
        self._quick_samples_open_btn.setProperty("buttonRole", "secondary")
        repolish(self._quick_samples_open_btn)
        self._quick_samples_open_btn.clicked.connect(self._on_open_quick_samples_folder)
        qs_row.addWidget(self._quick_samples_open_btn)
        self._quick_samples_remove_btn = QPushButton("Remove selected")
        self._quick_samples_remove_btn.setProperty("buttonRole", "secondary")
        repolish(self._quick_samples_remove_btn)
        self._quick_samples_remove_btn.setEnabled(False)
        self._quick_samples_remove_btn.clicked.connect(self._on_remove_quick_sample_clicked)
        qs_row.addWidget(self._quick_samples_remove_btn)
        self._quick_samples_refresh_btn = QPushButton("Refresh")
        self._quick_samples_refresh_btn.setProperty("buttonRole", "secondary")
        repolish(self._quick_samples_refresh_btn)
        self._quick_samples_refresh_btn.clicked.connect(self._refresh_quick_samples_list)
        qs_row.addWidget(self._quick_samples_refresh_btn)
        qs_row.addStretch()
        quick_section.body_layout().addLayout(qs_row)

        self._quick_samples_list = QListWidget()
        self._quick_samples_list.setObjectName("threeDQuickSamples")
        self._quick_samples_list.setViewMode(QListWidget.ViewMode.IconMode)
        self._quick_samples_list.setFlow(QListView.Flow.LeftToRight)
        self._quick_samples_list.setWrapping(False)
        self._quick_samples_list.setMovement(QListWidget.Movement.Static)
        self._quick_samples_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._quick_samples_list.setSpacing(8)
        self._quick_samples_list.setIconSize(QSize(_QUICK_SAMPLE_ICON, _QUICK_SAMPLE_ICON))
        self._quick_samples_list.setMinimumHeight(130)
        self._quick_samples_list.setMaximumHeight(200)
        self._quick_samples_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._quick_samples_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._quick_samples_list.itemSelectionChanged.connect(self._on_quick_sample_selection_changed)
        self._quick_samples_list.itemClicked.connect(self._on_quick_sample_item_clicked)
        self._quick_samples_list.itemActivated.connect(self._on_quick_sample_item_activated)
        quick_section.body_layout().addWidget(self._quick_samples_list)

        _single_lay.addWidget(quick_section)
        ps_lay.addWidget(self._single_block)
        controls_root.addWidget(ps_fr)

        pr_fr, pr_lay = self._nav_make_anchor("pipeline_repl")
        self._replication_block = self._build_replication_block()
        pr_lay.addWidget(self._replication_block)
        controls_root.addWidget(pr_fr)

        rh_fr, rh_lay = self._nav_make_anchor("runtime_host")
        rh_lay.addWidget(self._make_section_title("Host"))

        # Host status
        host_cell = QFrame()
        host_cell.setObjectName("opsCell")
        host_cell.setFrameShape(QFrame.Shape.StyledPanel)
        hl_host = QVBoxLayout(host_cell)
        hl_host.setContentsMargins(8, 6, 8, 6)
        self._caps_label = QLabel()
        self._caps_label.setWordWrap(True)
        hl_host.addWidget(self._caps_label)
        rh_lay.addWidget(host_cell)
        controls_root.addWidget(rh_fr)

        rs_fr, rs_lay = self._nav_make_anchor("runtime_sampling")
        # Advanced parameters (collapsed by default)
        self._params_section = CollapsibleSection("Sampling parameters", expanded=False)
        params_inset = QFrame()
        params_inset.setObjectName("threeDInset")
        params_inset.setFrameShape(QFrame.Shape.StyledPanel)
        pil = QVBoxLayout(params_inset)
        pil.setContentsMargins(6, 6, 6, 6)
        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setFrameShape(QFrame.Shape.NoFrame)
        params_scroll.setMaximumHeight(280)
        params_widget = QWidget()
        self._params_form = QFormLayout(params_widget)
        self._params_form.setContentsMargins(4, 4, 4, 4)
        self._params_form.setHorizontalSpacing(12)
        self._params_form.setVerticalSpacing(6)
        params_scroll.setWidget(params_widget)
        pil.addWidget(params_scroll)
        self._params_section.body_layout().addWidget(params_inset)
        self._build_params_form()
        rs_lay.addWidget(self._params_section)
        controls_root.addWidget(rs_fr)

        rr_fr, rr_lay = self._nav_make_anchor("runtime_run")
        rr_lay.addWidget(self._make_section_title("Run pipeline"))

        self._run_inset = QFrame()
        self._run_inset.setObjectName("threeDInset")
        self._run_inset.setFrameShape(QFrame.Shape.StyledPanel)
        run_outer = QVBoxLayout(self._run_inset)
        run_outer.setContentsMargins(10, 10, 10, 10)
        run_outer.setSpacing(8)

        gen_row = QHBoxLayout()
        gen_row.setSpacing(12)
        self._generate_btn = QPushButton("Generate 3D")
        self._generate_btn.setProperty("isPrimary", True)
        repolish(self._generate_btn)
        self._generate_btn.setEnabled(False)
        self._generate_btn.clicked.connect(self._on_generate_clicked)
        gen_row.addWidget(self._generate_btn)

        backend_lbl = QLabel("Backend")
        backend_lbl.setProperty("muted", True)
        repolish(backend_lbl)
        gen_row.addWidget(backend_lbl)
        self._backend_combo = QComboBox()
        self._backend_combo.setMinimumWidth(220)
        self._backend_combo.addItem("Local Gaussian splatting", "gaussian_local")
        if self._preloaded_apple_ok is not None:
            apple_ok = self._preloaded_apple_ok
        else:
            try:
                from mlops.three_d.trellis2_apple import available as _apple_available  # noqa: PLC0415
                apple_ok, _apple_msg = _apple_available()
            except Exception:
                apple_ok = False
        if self._caps.os == "darwin" and apple_ok:
            self._backend_combo.addItem("Native Trellis2 Apple (MLX)", "apple_mlx")
        if self._depth_mlpackage is not None:
            self._backend_combo.addItem(
                "Local depth (DepthAnything CoreML)", "depth_local"
            )
        apple_idx = self._backend_combo.findData("apple_mlx")
        self._backend_combo.setCurrentIndex(apple_idx if apple_idx >= 0 else 0)
        gen_row.addWidget(self._backend_combo)
        gen_row.addStretch()
        run_outer.addLayout(gen_row)
        rr_lay.addWidget(self._run_inset)
        controls_root.addWidget(rr_fr)

        os_fr, os_lay = self._nav_make_anchor("output_status")
        # Progress
        self._progress_cell = QFrame()
        self._progress_cell.setObjectName("opsCell")
        self._progress_cell.setFrameShape(QFrame.Shape.StyledPanel)
        pc_lay = QVBoxLayout(self._progress_cell)
        pc_lay.setContentsMargins(8, 8, 8, 8)
        pc_lay.setSpacing(8)
        pc_lay.addWidget(self._make_section_title("Job status"))
        self._stage_label = QLabel("Idle — select an image, then generate.")
        self._stage_label.setWordWrap(True)
        pc_lay.addWidget(self._stage_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        self._progress_bar.setTextVisible(True)
        pc_lay.addWidget(self._progress_bar)
        os_lay.addWidget(self._progress_cell)
        controls_root.addWidget(os_fr)

        or_fr, or_lay = self._nav_make_anchor("output_result")
        # Output (hidden until completed)
        self._output_cell = QFrame()
        self._output_cell.setObjectName("opsCell")
        self._output_cell.setFrameShape(QFrame.Shape.StyledPanel)
        self._output_cell.setVisible(False)
        oc_lay = QVBoxLayout(self._output_cell)
        oc_lay.setContentsMargins(8, 8, 8, 8)
        oc_lay.setSpacing(8)
        oc_lay.addWidget(self._make_section_title("Result"))
        self._output_thumb = QLabel()
        self._output_thumb.setObjectName("previewThumb")
        self._output_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._output_thumb.setMinimumHeight(140)
        self._output_thumb.setMaximumHeight(220)
        self._output_thumb.setScaledContents(False)
        oc_lay.addWidget(self._output_thumb)
        out_btns = QHBoxLayout()
        out_btns.setSpacing(8)
        self._dl_btn = QPushButton("Save artifact as…")
        self._dl_btn.setProperty("buttonRole", "secondary")
        repolish(self._dl_btn)
        self._dl_btn.clicked.connect(self._on_download_glb)
        out_btns.addWidget(self._dl_btn)
        self._html_btn = QPushButton("Open HTML preview")
        self._html_btn.setProperty("buttonRole", "secondary")
        repolish(self._html_btn)
        self._html_btn.clicked.connect(self._on_view_html)
        out_btns.addWidget(self._html_btn)
        out_btns.addStretch()
        oc_lay.addLayout(out_btns)
        or_lay.addWidget(self._output_cell)
        controls_root.addWidget(or_fr)

        oh_fr, oh_lay = self._nav_make_anchor("output_history")
        # Job history (collapsed)
        self._history_section = CollapsibleSection("Recent jobs", expanded=False)
        self._history_table = QTableWidget(0, 4)
        self._history_table.setHorizontalHeaderLabels(["Job", "Status", "Detail", ""])
        hdr = self._history_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._history_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._history_table.setAlternatingRowColors(True)
        self._history_table.verticalHeader().setVisible(False)
        self._history_table.setMaximumHeight(220)
        self._history_table.setShowGrid(False)
        self._history_section.body_layout().addWidget(self._history_table)
        oh_lay.addWidget(self._history_section)
        controls_root.addWidget(oh_fr)

        controls_root.addStretch()
        controls_scroll.setWidget(controls_inner)
        self._nav_tree.blockSignals(True)
        self._nav_fill_outline()
        self._nav_tree.blockSignals(False)
        self._nav_tree.itemSelectionChanged.connect(
            lambda: self._on_nav_outline_item(self._nav_tree.currentItem())
        )
        first_group = self._nav_tree.topLevelItem(0)
        if first_group is not None and first_group.childCount() > 0:
            self._nav_tree.setCurrentItem(first_group.child(0))
        nav_row.addWidget(controls_scroll, stretch=1)
        cc_lay.addLayout(nav_row)
        workspace.addWidget(controls_column)

        self._viewer_panel = self._build_viewer_panel()
        workspace.addWidget(self._viewer_panel)
        workspace.setStretchFactor(0, 0)
        workspace.setStretchFactor(1, 1)
        workspace.setSizes([520, 900])

        outer.addWidget(inner, stretch=1)

        repl_idx = self._mode_combo.findData("replication")
        if repl_idx >= 0:
            self._mode_combo.setCurrentIndex(repl_idx)
        self._on_pipeline_mode_changed(self._mode_combo.currentIndex())
        self._nav_sync_outline_branch_styles()
        self._refresh_asset_shell()
        self._refresh_asset_browser()
        self._refresh_quick_samples_list()

    def _quick_samples_dir(self) -> Path:
        d = (self._store.root.parent / "cvops_quick_samples").expanduser()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _prune_quick_samples(self) -> None:
        root = self._quick_samples_dir()
        files = [
            p
            for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in _QUICK_SAMPLE_SUFFIXES
        ]
        if len(files) <= _QUICK_SAMPLE_MAX_FILES:
            return
        files.sort(key=lambda p: p.stat().st_mtime)
        for p in files[: len(files) - _QUICK_SAMPLE_MAX_FILES]:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    def _save_quick_sample_for_job(self, image_path: Path, job_id: str) -> Optional[Path]:
        root = self._quick_samples_dir()
        stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in image_path.stem)[:96]
        suffix = image_path.suffix.lower() or ".png"
        if suffix not in _QUICK_SAMPLE_SUFFIXES:
            suffix = ".png"
        out = root / f"{job_id}__{stem}{suffix}"
        try:
            shutil.copy2(image_path, out)
        except OSError as exc:
            log.warning("Could not save quick sample %s: %s", out, exc)
            return None
        self._prune_quick_samples()
        return out

    def _select_quick_sample_path(self, path: Path) -> None:
        if not hasattr(self, "_quick_samples_list"):
            return
        try:
            want = str(path.resolve())
        except OSError:
            want = str(path)
        self._quick_samples_list.blockSignals(True)
        for i in range(self._quick_samples_list.count()):
            it = self._quick_samples_list.item(i)
            if it is None:
                continue
            raw = it.data(Qt.ItemDataRole.UserRole)
            if isinstance(raw, str) and raw == want:
                self._quick_samples_list.setCurrentRow(i)
                break
        self._quick_samples_list.blockSignals(False)
        self._on_quick_sample_selection_changed()

    def _refresh_quick_samples_list(self) -> None:
        if not hasattr(self, "_quick_samples_list"):
            return
        worker = getattr(self, "_quick_samples_worker", None)
        if worker is not None:
            try:
                worker.ready.disconnect()
            except Exception:
                pass
            self._quick_samples_worker = None

        current_path = ""
        if self._image_path is not None:
            try:
                if self._image_path.is_file():
                    current_path = str(self._image_path.resolve())
            except OSError:
                current_path = str(self._image_path)

        worker = _QuickSamplesWorker(self._quick_samples_dir(), parent=self)
        worker.ready.connect(lambda paths: self._on_quick_samples_ready(paths, current_path))
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._quick_samples_worker = worker

    def _on_quick_samples_ready(self, path_strs: list, current_path: str) -> None:
        self._quick_samples_worker = None
        if not hasattr(self, "_quick_samples_list"):
            return
        icon_sz = QSize(_QUICK_SAMPLE_ICON, _QUICK_SAMPLE_ICON)
        self._quick_samples_list.blockSignals(True)
        self._quick_samples_list.clear()
        select_row = -1
        for i, path_str in enumerate(path_strs):
            p = Path(path_str)
            pix = QPixmap(path_str)
            it = QListWidgetItem()
            it.setData(Qt.ItemDataRole.UserRole, path_str)
            label = p.name
            if len(label) > 28:
                label = label[:25] + "..."
            it.setText(label)
            if not pix.isNull():
                it.setIcon(
                    QIcon(
                        pix.scaled(
                            icon_sz,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                )
            it.setToolTip(f"{p.name}\n{p}")
            self._quick_samples_list.addItem(it)
            if current_path and path_str == current_path:
                select_row = i
        if select_row >= 0:
            self._quick_samples_list.setCurrentRow(select_row)
        self._quick_samples_list.blockSignals(False)
        self._on_quick_sample_selection_changed()

    def _on_quick_sample_selection_changed(self) -> None:
        if not hasattr(self, "_quick_samples_remove_btn"):
            return
        it = self._quick_samples_list.currentItem()
        self._quick_samples_remove_btn.setEnabled(it is not None)

    def _quick_sample_path_from_item(self, item: QListWidgetItem | None) -> Optional[Path]:
        if item is None:
            return None
        raw = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(raw, str) or not raw:
            return None
        p = Path(raw)
        return p if p.is_file() else None

    def _on_quick_sample_item_clicked(self, item: QListWidgetItem) -> None:
        p = self._quick_sample_path_from_item(item)
        if p is not None:
            self._set_single_image_source(p)

    def _on_quick_sample_item_activated(self, item: QListWidgetItem) -> None:
        self._on_quick_sample_item_clicked(item)

    def _on_open_quick_samples_folder(self) -> None:
        d = self._quick_samples_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(d.resolve())))

    def _on_remove_quick_sample_clicked(self) -> None:
        item = self._quick_samples_list.currentItem()
        p = self._quick_sample_path_from_item(item)
        if p is None:
            return
        reply = QMessageBox.question(
            self,
            "Remove quick sample",
            f"Delete this file?\n{p.name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            p.unlink()
        except OSError as exc:
            self.errorRaised.emit(f"Could not delete quick sample: {exc}")
            return
        if self._image_path is not None and self._image_path.resolve() == p.resolve():
            self._image_path = None
            self._filename_label.setText("No file selected")
            self._preview_thumb.clear()
            self._preview_thumb.setText("No image")
            if self._mode_combo.currentData() == "single":
                self._generate_btn.setEnabled(False)
        self._refresh_quick_samples_list()

    def _build_viewer_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("threeDViewerPanel")
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("3D Viewport")
        title.setProperty("isTitle", True)
        repolish(title)
        header.addWidget(title, stretch=0)
        header.addStretch(1)

        self._viewer_status = QLabel("No render loaded")
        self._viewer_status.setObjectName("wsStatus")
        self._viewer_status.setProperty("state", "connecting")
        repolish(self._viewer_status)
        header.addWidget(self._viewer_status)
        layout.addLayout(header)

        canvas = QFrame()
        canvas.setObjectName("threeDRenderCanvas")
        canvas.setFrameShape(QFrame.Shape.StyledPanel)
        canvas.setMinimumHeight(460)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        canvas_layout = QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(18, 18, 18, 18)
        canvas_layout.setSpacing(12)

        self._viewer_placeholder = QLabel(
            "Render space\n\n"
            "Generated objects, nerfstudio previews, splats, meshes, and area scans will render here. "
            "The left rail owns object selection, source inputs, presets, and pipeline configuration."
        )
        self._viewer_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._viewer_placeholder.setWordWrap(True)
        self._viewer_placeholder.setProperty("muted", True)
        repolish(self._viewer_placeholder)
        canvas_layout.addWidget(self._viewer_placeholder, stretch=1)
        layout.addWidget(canvas, stretch=1)

        footer = QHBoxLayout()
        self._viewer_asset_label = QLabel("Active asset: none")
        self._viewer_asset_label.setWordWrap(True)
        self._viewer_asset_label.setProperty("muted", True)
        repolish(self._viewer_asset_label)
        footer.addWidget(self._viewer_asset_label, stretch=1)

        open_preview = QPushButton("Open preview artifact")
        open_preview.setProperty("buttonRole", "secondary")
        open_preview.setEnabled(False)
        repolish(open_preview)
        footer.addWidget(open_preview)
        layout.addLayout(footer)
        return panel

    def _build_cvops_asset_shell(self) -> QWidget:
        section = CollapsibleSection("CV Ops 3D asset shell", expanded=False)

        shell = QFrame()
        shell.setObjectName("opsCell")
        shell.setFrameShape(QFrame.Shape.StyledPanel)
        outer = QVBoxLayout(shell)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self._asset_type_combo = QComboBox()
        self._asset_type_combo.addItem("Single object", "object")
        self._asset_type_combo.addItem("Scene", "scene")
        self._asset_type_combo.addItem("Area / city block", "area")
        self._asset_type_combo.currentIndexChanged.connect(self._on_asset_shell_changed)
        form.addRow("3D asset type", self._asset_type_combo)

        self._asset_name_edit = QLineEdit()
        self._asset_name_edit.setPlaceholderText("Example: Market Street block A, fire hydrant 01")
        self._asset_name_edit.textChanged.connect(self._on_asset_shell_changed)
        form.addRow("Name", self._asset_name_edit)

        self._asset_description_edit = QPlainTextEdit()
        self._asset_description_edit.setPlaceholderText(
            "Capture intent, source assumptions, scale notes, or reconstruction constraints."
        )
        self._asset_description_edit.setMaximumHeight(84)
        self._asset_description_edit.textChanged.connect(self._on_asset_shell_changed)
        form.addRow("Description", self._asset_description_edit)

        self._asset_preset_combo = QComboBox()
        self._asset_preset_combo.addItem("Object turntable / isolated asset", "object")
        self._asset_preset_combo.addItem("Small scene / room / street corner", "small_scene")
        self._asset_preset_combo.addItem("1x1 standard city block", "city_block_1x1")
        self._asset_preset_combo.currentIndexChanged.connect(self._on_asset_shell_changed)
        form.addRow("Preset", self._asset_preset_combo)

        scale_row = QHBoxLayout()
        scale_row.setSpacing(8)
        self._asset_width_m = QDoubleSpinBox()
        self._asset_width_m.setRange(0.01, 10000.0)
        self._asset_width_m.setDecimals(2)
        self._asset_width_m.setSuffix(" m W")
        self._asset_width_m.setValue(80.0)
        self._asset_width_m.valueChanged.connect(self._on_asset_shell_changed)
        scale_row.addWidget(self._asset_width_m)
        self._asset_depth_m = QDoubleSpinBox()
        self._asset_depth_m.setRange(0.01, 10000.0)
        self._asset_depth_m.setDecimals(2)
        self._asset_depth_m.setSuffix(" m D")
        self._asset_depth_m.setValue(80.0)
        self._asset_depth_m.valueChanged.connect(self._on_asset_shell_changed)
        scale_row.addWidget(self._asset_depth_m)
        self._asset_height_m = QDoubleSpinBox()
        self._asset_height_m.setRange(0.01, 10000.0)
        self._asset_height_m.setDecimals(2)
        self._asset_height_m.setSuffix(" m H")
        self._asset_height_m.setValue(30.0)
        self._asset_height_m.valueChanged.connect(self._on_asset_shell_changed)
        scale_row.addWidget(self._asset_height_m)
        form.addRow("Target bounds", scale_row)

        self._asset_pipeline_combo = QComboBox()
        self._asset_pipeline_combo.addItem("nerfstudio nerfacto", "nerfacto")
        self._asset_pipeline_combo.addItem("nerfstudio splatfacto", "splatfacto")
        self._asset_pipeline_combo.addItem("Prepare only / no training", "prepare_only")
        self._asset_pipeline_combo.currentIndexChanged.connect(self._on_asset_shell_changed)
        form.addRow("Pipeline target", self._asset_pipeline_combo)

        self._asset_source_combo = QComboBox()
        self._asset_source_combo.addItem("Image folder", "image_folder")
        self._asset_source_combo.addItem("Video file", "video_file")
        self._asset_source_combo.addItem("Single image", "single_image")
        self._asset_source_combo.currentIndexChanged.connect(self._on_asset_source_kind_changed)
        form.addRow("Source kind", self._asset_source_combo)

        outer.addLayout(form)

        source_row = QHBoxLayout()
        source_row.setSpacing(8)
        self._asset_pick_source_btn = QPushButton("Choose folder…")
        self._asset_pick_source_btn.setProperty("buttonRole", "secondary")
        repolish(self._asset_pick_source_btn)
        self._asset_pick_source_btn.clicked.connect(self._on_pick_asset_source)
        source_row.addWidget(self._asset_pick_source_btn)
        self._asset_source_label = QLabel("No source selected.")
        self._asset_source_label.setWordWrap(True)
        self._asset_source_label.setProperty("muted", True)
        repolish(self._asset_source_label)
        source_row.addWidget(self._asset_source_label, stretch=1)
        outer.addLayout(source_row)

        target_cell = QFrame()
        target_cell.setObjectName("threeDInset")
        target_cell.setFrameShape(QFrame.Shape.StyledPanel)
        target_layout = QVBoxLayout(target_cell)
        target_layout.setContentsMargins(8, 8, 8, 8)
        target_layout.setSpacing(6)
        self._asset_target_label = QLabel("")
        self._asset_target_label.setWordWrap(True)
        target_layout.addWidget(self._asset_target_label)
        self._asset_manifest_preview = QPlainTextEdit()
        self._asset_manifest_preview.setObjectName("artifactPreview")
        self._asset_manifest_preview.setReadOnly(True)
        self._asset_manifest_preview.setMaximumHeight(220)
        target_layout.addWidget(self._asset_manifest_preview)
        outer.addWidget(target_cell)

        actions = QHBoxLayout()
        self._asset_stage_btn = QPushButton("Stage asset shell (storage next)")
        self._asset_stage_btn.setProperty("isPrimary", True)
        self._asset_stage_btn.setEnabled(False)
        repolish(self._asset_stage_btn)
        self._asset_stage_btn.clicked.connect(self._on_stage_asset_shell_clicked)
        actions.addWidget(self._asset_stage_btn)
        self._asset_open_db_btn = QPushButton("Open database/3D")
        self._asset_open_db_btn.setProperty("buttonRole", "secondary")
        repolish(self._asset_open_db_btn)
        self._asset_open_db_btn.clicked.connect(self._on_open_three_d_database)
        actions.addWidget(self._asset_open_db_btn)
        actions.addStretch()
        outer.addLayout(actions)

        self._asset_shell_status = QLabel("Storage layer is not wired yet.")
        self._asset_shell_status.setWordWrap(True)
        self._asset_shell_status.setProperty("muted", True)
        repolish(self._asset_shell_status)
        outer.addWidget(self._asset_shell_status)

        section.body_layout().addWidget(shell)
        return section

    def _build_asset_browser(self) -> QWidget:
        section = CollapsibleSection("Staged 3D assets", expanded=False)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._asset_refresh_btn = QPushButton("Refresh")
        self._asset_refresh_btn.setProperty("buttonRole", "secondary")
        repolish(self._asset_refresh_btn)
        self._asset_refresh_btn.clicked.connect(self._refresh_asset_browser)
        row.addWidget(self._asset_refresh_btn)
        self._asset_open_selected_btn = QPushButton("Open selected")
        self._asset_open_selected_btn.setProperty("buttonRole", "secondary")
        self._asset_open_selected_btn.setEnabled(False)
        repolish(self._asset_open_selected_btn)
        self._asset_open_selected_btn.clicked.connect(self._on_open_selected_asset)
        row.addWidget(self._asset_open_selected_btn)
        self._asset_prepare_btn = QPushButton("Prepare dataset")
        self._asset_prepare_btn.setProperty("buttonRole", "secondary")
        self._asset_prepare_btn.setEnabled(False)
        repolish(self._asset_prepare_btn)
        self._asset_prepare_btn.clicked.connect(self._on_prepare_selected_asset)
        row.addWidget(self._asset_prepare_btn)
        row.addStretch()
        layout.addLayout(row)

        self._asset_list = QListWidget()
        self._asset_list.setAlternatingRowColors(True)
        self._asset_list.setMinimumHeight(160)
        self._asset_list.itemSelectionChanged.connect(self._on_asset_selection_changed)
        self._asset_list.itemActivated.connect(lambda _item: self._on_open_selected_asset())
        layout.addWidget(self._asset_list)

        self._asset_browser_status = QLabel("No staged assets loaded.")
        self._asset_browser_status.setWordWrap(True)
        self._asset_browser_status.setProperty("muted", True)
        repolish(self._asset_browser_status)
        layout.addWidget(self._asset_browser_status)

        section.body_layout().addWidget(body)
        return section

    # ------------------------------------------------------------------ #
    # CV Ops 3D asset shell
    # ------------------------------------------------------------------ #

    @staticmethod
    def _slugify_asset_name(value: str) -> str:
        from mlops.three_d import slugify_asset_name

        return slugify_asset_name(value)

    def _asset_shell_payload(self) -> dict[str, object]:
        asset_type = str(self._asset_type_combo.currentData() or "object")
        name = self._asset_name_edit.text().strip()
        slug = self._slugify_asset_name(name)
        bucket = {
            "object": "objects",
            "scene": "scenes",
            "area": "areas",
        }.get(asset_type, "objects")
        target_root = self._three_d_db_root / bucket / slug
        return {
            "version": 1,
            "status": "draft",
            "asset_type": asset_type,
            "name": name,
            "slug": slug,
            "description": self._asset_description_edit.toPlainText().strip(),
            "preset": str(self._asset_preset_combo.currentData() or ""),
            "pipeline_target": str(self._asset_pipeline_combo.currentData() or ""),
            "source": {
                "kind": str(self._asset_source_combo.currentData() or ""),
                "path": str(self._asset_source_path or ""),
            },
            "database": {
                "sector_path": "/3D",
                "root": str(self._three_d_db_root),
                "target_path": str(target_root),
                "folders": ["inputs", "nerfstudio", "outputs"],
            },
            "bounds_meters": {
                "width": round(float(self._asset_width_m.value()), 3),
                "depth": round(float(self._asset_depth_m.value()), 3),
                "height": round(float(self._asset_height_m.value()), 3),
            },
            "nerfstudio": {
                "command_plan": self._planned_nerfstudio_commands(target_root),
                "wired": False,
            },
        }

    def _planned_nerfstudio_commands(self, target_root: Path) -> list[str]:
        pipeline = str(self._asset_pipeline_combo.currentData() or "nerfacto")
        if pipeline == "prepare_only":
            return []
        data_dir = target_root / "nerfstudio"
        return [
            f"ns-process-data images --data {target_root / 'inputs'} --output-dir {data_dir}",
            f"ns-train {pipeline} --data {data_dir}",
        ]

    def _refresh_asset_shell(self) -> None:
        payload = self._asset_shell_payload()
        target = str(dict(payload.get("database") or {}).get("target_path") or "")
        self._asset_target_label.setText(
            "Target draft path:\n"
            f"{target}\n"
            "Expected layout: inputs/ + nerfstudio/ + outputs/ + manifest.json"
        )
        self._asset_manifest_preview.setPlainText(json.dumps(payload, indent=2))
        has_name = bool(str(payload.get("name") or "").strip())
        self._asset_stage_btn.setEnabled(has_name)
        if hasattr(self, "_viewer_asset_label"):
            asset_type = str(payload.get("asset_type") or "object")
            name = str(payload.get("name") or "").strip() or "unnamed draft"
            preset = str(payload.get("preset") or "")
            self._viewer_asset_label.setText(
                f"Active asset: {name} [{asset_type}] · preset: {preset} · target: {target}"
            )

    def _refresh_asset_browser(self) -> None:
        worker = getattr(self, "_asset_browser_worker", None)
        if worker is not None:
            try:
                worker.ready.disconnect()
                worker.failed.disconnect()
            except Exception:
                pass
            self._asset_browser_worker = None

        selected_root = ""
        if hasattr(self, "_asset_list"):
            current = self._asset_list.currentItem()
            if current is not None:
                selected_root = str(current.data(Qt.ItemDataRole.UserRole) or "")
            self._asset_browser_status.setText("[Loading] scanning 3D assets…")

        worker = _AssetBrowserWorker(self._three_d_db_root, parent=self)
        worker.ready.connect(lambda assets: self._on_asset_browser_ready(assets, selected_root))
        worker.failed.connect(lambda err: self._asset_browser_status.setText(f"3D asset list failed: {err}"))
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._asset_browser_worker = worker

    def _on_asset_browser_ready(self, assets: list, selected_root: str) -> None:
        self._asset_browser_worker = None
        if not hasattr(self, "_asset_list"):
            return
        self._asset_list.blockSignals(True)
        self._asset_list.clear()
        restore_row = -1
        for i, entry in enumerate(assets):
            manifest = dict(entry.get("manifest") or {})
            name = str(manifest.get("name") or entry.get("slug") or "unnamed")
            asset_type = str(entry.get("asset_type") or manifest.get("asset_type") or "asset")
            preset = str(manifest.get("preset") or "")
            root = str(entry.get("root") or "")
            label = f"{name}\n[{asset_type}] {preset}\n{root}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, root)
            item.setData(Qt.ItemDataRole.UserRole + 1, entry)
            self._asset_list.addItem(item)
            if root and root == selected_root:
                restore_row = i
        if restore_row >= 0:
            self._asset_list.setCurrentRow(restore_row)
        elif self._asset_list.count() > 0:
            self._asset_list.setCurrentRow(0)
        self._asset_list.blockSignals(False)
        self._asset_browser_status.setText(
            f"{len(assets)} staged 3D asset{'s' if len(assets) != 1 else ''} under database/3D."
        )
        self._on_asset_selection_changed()

    def _selected_asset_entry(self) -> dict[str, object]:
        item = self._asset_list.currentItem()
        if item is None:
            return {}
        entry = item.data(Qt.ItemDataRole.UserRole + 1)
        return dict(entry) if isinstance(entry, dict) else {}

    def _on_asset_selection_changed(self) -> None:
        entry = self._selected_asset_entry()
        self._asset_open_selected_btn.setEnabled(bool(entry))
        self._asset_prepare_btn.setEnabled(bool(entry) and not self._ns_prepare_busy)
        if not entry:
            return
        manifest = dict(entry.get("manifest") or {})
        root = str(entry.get("root") or "")
        name = str(manifest.get("name") or entry.get("slug") or "unnamed")
        asset_type = str(manifest.get("asset_type") or entry.get("asset_type") or "asset")
        preset = str(manifest.get("preset") or "")
        if hasattr(self, "_viewer_asset_label"):
            self._viewer_asset_label.setText(
                f"Selected asset: {name} [{asset_type}] · preset: {preset} · root: {root}"
            )
        if hasattr(self, "_viewer_status"):
            self._viewer_status.setText(str(manifest.get("status") or "draft"))
            self._viewer_status.setProperty("state", "connected")
            repolish(self._viewer_status)
        if hasattr(self, "_viewer_placeholder"):
            source = dict(manifest.get("source") or {})
            count = len(list(source.get("input_paths") or []))
            ns = dict(manifest.get("nerfstudio") or {})
            dataset = str(ns.get("dataset_path") or "")
            prep = str(ns.get("prepare_status") or "not prepared")
            self._viewer_placeholder.setText(
                "Staged 3D asset selected\n\n"
                f"{name}\n{root}\n\n"
                f"Inputs materialized: {count}\n"
                f"Nerfstudio: {prep}\n"
                f"{dataset}\n\n"
                "Render output will attach here when nerfstudio jobs are wired."
            )

    def _on_open_selected_asset(self) -> None:
        entry = self._selected_asset_entry()
        root = str(entry.get("root") or "")
        if not root or not Path(root).exists():
            self.errorRaised.emit("Selected 3D asset folder was not found.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(root).resolve())))

    def _on_prepare_selected_asset(self) -> None:
        if self._ns_prepare_busy:
            return
        entry = self._selected_asset_entry()
        root = str(entry.get("root") or "")
        if not root or not Path(root).exists():
            self.errorRaised.emit("Select a staged 3D asset first.")
            return

        asset_root = Path(root)
        project_root = Path(__file__).resolve().parents[4]

        def on_status(stage: str, message: str, progress: float = -1.0) -> None:
            self._nerfstudio_prepare_status.emit(stage, message, progress)

        def _run() -> None:
            try:
                from mlops.three_d import prepare_nerfstudio_dataset

                result = prepare_nerfstudio_dataset(
                    asset_root,
                    project_root=project_root,
                    on_status=on_status,
                )
                self._nerfstudio_prepare_finished.emit(True, str(asset_root), result.to_dict())
            except Exception as exc:
                self._nerfstudio_prepare_finished.emit(False, str(asset_root), {"error": str(exc)})

        self._ns_prepare_busy = True
        self._asset_prepare_btn.setEnabled(False)
        self._stage_label.setText("Nerfstudio — preparing dataset…")
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(3)
        if hasattr(self, "_viewer_status"):
            self._viewer_status.setText("Preparing dataset")
            self._viewer_status.setProperty("state", "connecting")
            repolish(self._viewer_status)
        threading.Thread(
            target=_run,
            daemon=True,
            name=f"nerfstudio-prepare-{asset_root.name}",
        ).start()

    def _on_nerfstudio_prepare_status_slot(self, stage: str, message: str, progress: float) -> None:
        self._stage_label.setText(f"{stage} — {message}")
        if progress >= 0.0:
            self._progress_bar.setVisible(True)
            self._progress_bar.setValue(int(max(0.0, min(1.0, progress)) * 100))
        if hasattr(self, "_viewer_placeholder"):
            self._viewer_placeholder.setText(
                "Preparing nerfstudio dataset\n\n"
                f"{stage}\n{message}"
            )

    def _on_nerfstudio_prepare_finished_slot(self, ok: bool, asset_root: str, payload: object) -> None:
        self._ns_prepare_busy = False
        self._asset_prepare_btn.setEnabled(bool(self._selected_asset_entry()))
        result = dict(payload) if isinstance(payload, dict) else {}
        if ok:
            dataset_path = str(result.get("dataset_path") or "")
            log_path = str(result.get("log_path") or "")
            self._stage_label.setText(f"Nerfstudio dataset ready — {dataset_path}")
            self._progress_bar.setVisible(True)
            self._progress_bar.setValue(100)
            if hasattr(self, "_viewer_status"):
                self._viewer_status.setText("Dataset ready")
                self._viewer_status.setProperty("state", "connected")
                repolish(self._viewer_status)
            if hasattr(self, "_viewer_placeholder"):
                self._viewer_placeholder.setText(
                    "Nerfstudio dataset ready\n\n"
                    f"Asset: {asset_root}\n"
                    f"Dataset: {dataset_path}\n"
                    f"Log: {log_path}\n\n"
                    "Next phase: launch nerfstudio training and stream preview artifacts."
                )
        else:
            error = str(result.get("error") or "nerfstudio dataset preparation failed")
            self._stage_label.setText(f"Nerfstudio prepare failed — {error}")
            self._progress_bar.setVisible(False)
            if hasattr(self, "_viewer_status"):
                self._viewer_status.setText("Prepare failed")
                self._viewer_status.setProperty("state", "warning")
                repolish(self._viewer_status)
            self.errorRaised.emit(error)
        self._refresh_asset_browser()

    def _on_asset_shell_changed(self, *_args) -> None:
        asset_type = str(self._asset_type_combo.currentData() or "object")
        preset = str(self._asset_preset_combo.currentData() or "")
        if asset_type == "area" and preset != "city_block_1x1":
            idx = self._asset_preset_combo.findData("city_block_1x1")
            if idx >= 0:
                self._asset_preset_combo.blockSignals(True)
                self._asset_preset_combo.setCurrentIndex(idx)
                self._asset_preset_combo.blockSignals(False)
        if str(self._asset_preset_combo.currentData() or "") == "city_block_1x1":
            self._asset_width_m.blockSignals(True)
            self._asset_depth_m.blockSignals(True)
            self._asset_height_m.blockSignals(True)
            if self._asset_width_m.value() < 1.0:
                self._asset_width_m.setValue(80.0)
            if self._asset_depth_m.value() < 1.0:
                self._asset_depth_m.setValue(80.0)
            if self._asset_height_m.value() < 1.0:
                self._asset_height_m.setValue(30.0)
            self._asset_width_m.blockSignals(False)
            self._asset_depth_m.blockSignals(False)
            self._asset_height_m.blockSignals(False)
        self._refresh_asset_shell()

    def _on_asset_source_kind_changed(self, _index: int) -> None:
        kind = str(self._asset_source_combo.currentData() or "image_folder")
        if kind == "video_file":
            self._asset_pick_source_btn.setText("Choose video…")
        elif kind == "single_image":
            self._asset_pick_source_btn.setText("Choose image…")
        else:
            self._asset_pick_source_btn.setText("Choose folder…")
        self._refresh_asset_shell()

    def _on_pick_asset_source(self) -> None:
        kind = str(self._asset_source_combo.currentData() or "image_folder")
        if kind == "image_folder":
            d = QFileDialog.getExistingDirectory(self, "Select 3D source image folder")
            if not d:
                return
            self._asset_source_path = Path(d)
        elif kind == "video_file":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select 3D source video",
                "",
                "Video (*.mp4 *.mov *.mkv *.avi *.webm *.m4v)",
            )
            if not path:
                return
            self._asset_source_path = Path(path)
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select 3D source image",
                "",
                "Images (*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp)",
            )
            if not path:
                return
            self._asset_source_path = Path(path)
        self._asset_source_label.setText(str(self._asset_source_path))
        self._refresh_asset_shell()

    def _on_stage_asset_shell_clicked(self) -> None:
        from mlops.three_d import ThreeDAssetStore

        payload = self._asset_shell_payload()
        store = ThreeDAssetStore(self._three_d_db_root)
        try:
            result = store.create_draft(payload)
        except FileExistsError as exc:
            self.errorRaised.emit(str(exc))
            self._asset_shell_status.setText(
                "Draft already exists. Change the name or wire overwrite support later."
            )
            return
        except Exception as exc:
            self.errorRaised.emit(f"3D asset staging failed: {exc}")
            return

        self._asset_shell_status.setText(
            "Draft asset staged.\n"
            f"Root: {result.root}\n"
            f"Manifest: {result.manifest_path}"
        )
        if hasattr(self, "_viewer_status"):
            self._viewer_status.setText("Draft staged")
            self._viewer_status.setProperty("state", "connected")
            repolish(self._viewer_status)
        if hasattr(self, "_viewer_placeholder"):
            self._viewer_placeholder.setText(
                "Draft 3D asset staged\n\n"
                f"{result.root}\n\n"
                "Next phase: attach nerfstudio dataset preparation and render previews here."
            )
        self._asset_manifest_preview.setPlainText(json.dumps(result.manifest, indent=2))
        self._refresh_asset_browser()

    def _on_open_three_d_database(self) -> None:
        self._three_d_db_root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._three_d_db_root.resolve())))

    def _build_replication_block(self) -> QWidget:
        block = QWidget()
        outer = QVBoxLayout(block)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)
        outer.addWidget(self._make_section_title("Replication inputs"))

        hint = QLabel(
            "Stage a folder of images or extract frames from video with ffmpeg, then optionally "
            "import an existing COLMAP sparse reconstruction or run ``colmap automatic_reconstructor``. "
            "Workspaces are written under ~/.gaussian_splat/jobs/<id>/ with images/, "
            "replication_manifest.json, and sparse/ when calibration is used."
        )
        hint.setWordWrap(True)
        hint.setProperty("muted", True)
        repolish(hint)
        outer.addWidget(hint)
        handoff = QLabel(
            "Gaussian handoff status: workspace prep is implemented and stable enough for capture staging. "
            "Training/export/render wiring is still pending; use this rail to build per-object datasets on the fly."
        )
        handoff.setWordWrap(True)
        handoff.setProperty("state", "warning")
        repolish(handoff)
        outer.addWidget(handoff)
        depth_assist = QLabel(
            "DepthAnything assist (Core ML): optionally generate quick depth preview meshes from sampled "
            "staged frames to validate object capture quality before Gaussian training."
        )
        depth_assist.setWordWrap(True)
        depth_assist.setProperty("muted", True)
        repolish(depth_assist)
        outer.addWidget(depth_assist)

        cell = QFrame()
        cell.setObjectName("opsCell")
        cell.setFrameShape(QFrame.Shape.StyledPanel)
        cl = QVBoxLayout(cell)
        cl.setContentsMargins(8, 8, 8, 8)
        cl.setSpacing(8)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Media"))
        self._repl_source_combo = QComboBox()
        self._repl_source_combo.addItem("Image folder", "image_folder")
        self._repl_source_combo.addItem("Single image", "single_image")
        self._repl_source_combo.addItem("Video file", "video_file")
        self._repl_source_combo.currentIndexChanged.connect(self._on_repl_source_kind_changed)
        row1.addWidget(self._repl_source_combo, stretch=1)
        self._repl_pick_btn = QPushButton("Choose folder…")
        self._repl_pick_btn.setProperty("buttonRole", "secondary")
        repolish(self._repl_pick_btn)
        self._repl_pick_btn.clicked.connect(self._on_repl_pick_media)
        row1.addWidget(self._repl_pick_btn)
        self._repl_pick_db_btn = QPushButton("From DB video…")
        self._repl_pick_db_btn.setProperty("buttonRole", "secondary")
        repolish(self._repl_pick_db_btn)
        self._repl_pick_db_btn.clicked.connect(self._on_repl_pick_media_from_database)
        row1.addWidget(self._repl_pick_db_btn)
        cl.addLayout(row1)

        self._repl_path_label = QLabel("No folder or video selected.")
        self._repl_path_label.setWordWrap(True)
        self._repl_path_label.setProperty("muted", True)
        cl.addWidget(self._repl_path_label)

        cal_row = QHBoxLayout()
        cal_row.addWidget(QLabel("Calibration"))
        self._repl_calibration_combo = QComboBox()
        self._repl_calibration_combo.addItem("None (images only)", "none")
        self._repl_calibration_combo.addItem("Import COLMAP sparse model", "import_sparse")
        self._repl_calibration_combo.addItem("Run COLMAP automatic reconstruction", "run_colmap")
        self._repl_calibration_combo.currentIndexChanged.connect(self._on_repl_calibration_changed)
        cal_row.addWidget(self._repl_calibration_combo, stretch=1)
        cl.addLayout(cal_row)

        row_cs = QHBoxLayout()
        self._repl_colmap_btn = QPushButton("Choose COLMAP sparse…")
        self._repl_colmap_btn.setProperty("buttonRole", "secondary")
        repolish(self._repl_colmap_btn)
        self._repl_colmap_btn.setEnabled(False)
        self._repl_colmap_btn.clicked.connect(self._on_repl_pick_colmap_sparse)
        row_cs.addWidget(self._repl_colmap_btn)
        self._repl_colmap_label = QLabel("No sparse model selected.")
        self._repl_colmap_label.setWordWrap(True)
        self._repl_colmap_label.setProperty("muted", True)
        row_cs.addWidget(self._repl_colmap_label, stretch=1)
        cl.addLayout(row_cs)

        vid_form = QFormLayout()
        self._repl_video_fps = QDoubleSpinBox()
        self._repl_video_fps.setRange(0.05, 60.0)
        self._repl_video_fps.setDecimals(2)
        self._repl_video_fps.setSingleStep(0.25)
        self._repl_video_fps.setValue(1.0)
        vid_form.addRow("Video extract FPS", self._repl_video_fps)

        self._repl_max_frames = QSpinBox()
        self._repl_max_frames.setRange(10, 100_000)
        self._repl_max_frames.setValue(300)
        vid_form.addRow("Max frames (video)", self._repl_max_frames)
        self._repl_depth_assist_count = QSpinBox()
        self._repl_depth_assist_count.setRange(1, 64)
        self._repl_depth_assist_count.setValue(8)
        vid_form.addRow("Depth assist samples", self._repl_depth_assist_count)
        cl.addLayout(vid_form)

        self._repl_symlink_chk = QCheckBox("Symlink images from folder (faster; not used for video)")
        self._repl_symlink_chk.setChecked(False)
        cl.addWidget(self._repl_symlink_chk)
        self._repl_depth_assist_chk = QCheckBox(
            "Generate DepthAnything preview meshes from sampled staged images"
        )
        self._repl_depth_assist_chk.setChecked(True)
        self._repl_depth_assist_chk.setEnabled(self._depth_mlpackage is not None)
        cl.addWidget(self._repl_depth_assist_chk)

        act_row = QHBoxLayout()
        self._prep_workspace_btn = QPushButton("Prepare replication workspace")
        self._prep_workspace_btn.setProperty("buttonRole", "secondary")
        repolish(self._prep_workspace_btn)
        self._prep_workspace_btn.clicked.connect(self._on_prepare_replication_workspace)
        act_row.addWidget(self._prep_workspace_btn)
        self._run_gaussian_btn = QPushButton("Run 3D render (Gaussian)")
        self._run_gaussian_btn.setProperty("isPrimary", True)
        repolish(self._run_gaussian_btn)
        self._run_gaussian_btn.clicked.connect(self._on_run_gaussian_render_clicked)
        act_row.addWidget(self._run_gaussian_btn)
        self._open_ws_btn = QPushButton("Open last workspace folder")
        self._open_ws_btn.setProperty("buttonRole", "secondary")
        repolish(self._open_ws_btn)
        self._open_ws_btn.setEnabled(False)
        self._open_ws_btn.clicked.connect(self._on_open_replication_workspace)
        act_row.addWidget(self._open_ws_btn)
        act_row.addStretch()
        cl.addLayout(act_row)

        self._repl_status_label = QLabel("")
        self._repl_status_label.setWordWrap(True)
        self._repl_status_label.setProperty("muted", True)
        cl.addWidget(self._repl_status_label)

        self._repl_log_view = QPlainTextEdit()
        self._repl_log_view.setReadOnly(True)
        self._repl_log_view.setMaximumBlockCount(500)
        self._repl_log_view.setMinimumHeight(150)
        self._repl_log_view.setPlaceholderText("COLMAP / Gaussian training log will stream here.")
        self._repl_log_view.setProperty("muted", True)
        cl.addWidget(self._repl_log_view)

        outer.addWidget(cell)
        block.setVisible(False)
        return block

    def _params_section_label(self, text: str) -> None:
        lab = QLabel(text)
        lab.setProperty("muted", True)
        repolish(lab)
        self._params_form.addRow(lab)

    def _build_params_form(self) -> None:
        d = self._defaults

        self._params_section_label("General")

        self._p_randomize_seed = QCheckBox()
        self._p_randomize_seed.setChecked(d.randomize_seed)
        self._params_form.addRow("Randomize seed", self._p_randomize_seed)

        self._p_seed = QSpinBox()
        self._p_seed.setRange(0, 2_147_483_647)
        self._p_seed.setValue(int(d.seed))
        self._params_form.addRow("Seed", self._p_seed)

        self._p_resolution = QComboBox()
        for r in ("512", "768", "1024"):
            self._p_resolution.addItem(r, r)
        self._p_resolution.setCurrentText(d.resolution)
        self._params_form.addRow("Input resolution", self._p_resolution)

        self._p_decimation = QSpinBox()
        self._p_decimation.setRange(1_000, 2_000_000)
        self._p_decimation.setSingleStep(10_000)
        self._p_decimation.setValue(int(d.decimation_target))
        self._params_form.addRow("Mesh decimation target", self._p_decimation)

        self._p_texture_size = QComboBox()
        for ts in (1024, 2048, 4096):
            self._p_texture_size.addItem(str(ts), ts)
        self._p_texture_size.setCurrentText(str(int(d.texture_size)))
        self._params_form.addRow("Texture size", self._p_texture_size)

        def _dspin(val: float, lo: float, hi: float, step: float = 0.1) -> QDoubleSpinBox:
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setDecimals(2)
            w.setValue(val)
            return w

        self._params_section_label("Sparse structure")

        self._p_ss_g_strength = _dspin(d.ss_guidance_strength, 0.0, 15.0)
        self._params_form.addRow("Guidance strength", self._p_ss_g_strength)

        self._p_ss_g_rescale = _dspin(d.ss_guidance_rescale, 0.0, 1.0, 0.05)
        self._params_form.addRow("Guidance rescale", self._p_ss_g_rescale)

        self._p_ss_steps = QSpinBox()
        self._p_ss_steps.setRange(1, 50)
        self._p_ss_steps.setValue(int(d.ss_sampling_steps))
        self._params_form.addRow("Sampling steps", self._p_ss_steps)

        self._p_ss_rescale_t = _dspin(d.ss_rescale_t, 0.0, 10.0)
        self._params_form.addRow("Rescale T", self._p_ss_rescale_t)

        self._params_section_label("Shape SLAT")

        self._p_shape_g_strength = _dspin(d.shape_slat_guidance_strength, 0.0, 15.0)
        self._params_form.addRow("Guidance strength", self._p_shape_g_strength)

        self._p_shape_g_rescale = _dspin(d.shape_slat_guidance_rescale, 0.0, 1.0, 0.05)
        self._params_form.addRow("Guidance rescale", self._p_shape_g_rescale)

        self._p_shape_steps = QSpinBox()
        self._p_shape_steps.setRange(1, 50)
        self._p_shape_steps.setValue(int(d.shape_slat_sampling_steps))
        self._params_form.addRow("Sampling steps", self._p_shape_steps)

        self._p_shape_rescale_t = _dspin(d.shape_slat_rescale_t, 0.0, 10.0)
        self._params_form.addRow("Rescale T", self._p_shape_rescale_t)

        self._params_section_label("Texture SLAT")

        self._p_tex_g_strength = _dspin(d.tex_slat_guidance_strength, 0.0, 15.0)
        self._params_form.addRow("Guidance strength", self._p_tex_g_strength)

        self._p_tex_g_rescale = _dspin(d.tex_slat_guidance_rescale, 0.0, 1.0, 0.05)
        self._params_form.addRow("Guidance rescale", self._p_tex_g_rescale)

        self._p_tex_steps = QSpinBox()
        self._p_tex_steps.setRange(1, 50)
        self._p_tex_steps.setValue(int(d.tex_slat_sampling_steps))
        self._params_form.addRow("Sampling steps", self._p_tex_steps)

        self._p_tex_rescale_t = _dspin(d.tex_slat_rescale_t, 0.0, 10.0)
        self._params_form.addRow("Rescale T", self._p_tex_rescale_t)

    # ------------------------------------------------------------------ #
    # Capability banner
    # ------------------------------------------------------------------ #

    def _refresh_caps_banner(self) -> None:
        caps = self._caps
        os_label = _OS_LABEL.get(caps.os, caps.os)
        if self._depth_mlpackage is not None:
            depth_hint = (
                f" DepthAnything Core ML: {self._depth_mlpackage.name} "
                '(backend "Local depth").'
            )
        else:
            depth_hint = (
                " No DepthAnything .mlpackage — place one under "
                "Insight_assets/models/DepthAnythingModelSmall/ "
                "or set INSIGHT_DEPTH_ANYTHING_MLPACKAGE."
            )

        if caps.os == "darwin":
            text = (
                f"{os_label}: Gaussian + DepthAnything local path is active. "
                f"On-device depth preview meshes are available when the Core ML bundle is present.{depth_hint}"
            )
            self._caps_label.setProperty("state", "success")
        else:
            text = (
                f"{os_label}: Gaussian replication workspace path is active. "
                f"Depth local support depends on Core ML availability (macOS-specific).{depth_hint}"
            )
            self._caps_label.setProperty("state", "warning")
        self._caps_label.setText(text)
        repolish(self._caps_label)

    # ------------------------------------------------------------------ #
    # Replication (multi-view / COLMAP prep)
    # ------------------------------------------------------------------ #

    def _on_pipeline_mode_changed(self, _index: int) -> None:
        is_single = self._mode_combo.currentData() == "single"
        self._single_block.setVisible(is_single)
        self._replication_block.setVisible(not is_single)
        self._params_section.setVisible(True)
        self._run_inset.setVisible(is_single)
        if is_single:
            self._stage_label.setText("Idle — select an image, then generate.")
            self._generate_btn.setEnabled(self._image_path is not None)
        else:
            self._stage_label.setText("Idle — choose media and prepare a replication workspace.")
            self._generate_btn.setEnabled(False)

        ps_w = self._nav_anchors.get("pipeline_single")
        pr_w = self._nav_anchors.get("pipeline_repl")
        if ps_w is not None:
            ps_w.setVisible(is_single)
        if pr_w is not None:
            pr_w.setVisible(not is_single)
        for nid in ("runtime_sampling", "runtime_run"):
            w = self._nav_anchors.get(nid)
            if w is not None:
                w.setVisible(is_single)
        self._nav_sync_outline_branch_styles()
        if not is_single:
            self._on_repl_source_kind_changed(self._repl_source_combo.currentIndex())

    def _on_repl_source_kind_changed(self, _index: int) -> None:
        sk = self._repl_source_combo.currentData()
        if sk == "video_file":
            self._repl_pick_btn.setText("Choose video…")
        elif sk == "single_image":
            self._repl_pick_btn.setText("Choose image…")
        else:
            self._repl_pick_btn.setText("Choose folder…")

    def _on_repl_calibration_changed(self, _index: int) -> None:
        cal = self._repl_calibration_combo.currentData()
        self._repl_colmap_btn.setEnabled(cal == "import_sparse")

    def _on_repl_pick_media(self) -> None:
        sk = self._repl_source_combo.currentData()
        if sk == "image_folder":
            d = QFileDialog.getExistingDirectory(self, "Select image folder")
            if not d:
                return
            self._repl_media_path = Path(d)
        elif sk == "single_image":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select image",
                "",
                "Images (*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp)",
            )
            if not path:
                return
            self._repl_media_path = Path(path)
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select video",
                "",
                "Video (*.mp4 *.mov *.mkv *.avi *.webm *.m4v)",
            )
            if not path:
                return
            self._repl_media_path = Path(path)
        self._repl_path_label.setText(str(self._repl_media_path))

    def _on_repl_pick_media_from_database(self) -> None:
        root = Path(__file__).resolve().parents[4]
        db_root = root / "database"
        start_dir = str(db_root if db_root.is_dir() else root)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video from database",
            start_dir,
            "Video (*.mp4 *.mov *.mkv *.avi *.webm *.m4v)",
        )
        if not path:
            return
        idx = self._repl_source_combo.findData("video_file")
        if idx >= 0:
            self._repl_source_combo.blockSignals(True)
            self._repl_source_combo.setCurrentIndex(idx)
            self._repl_source_combo.blockSignals(False)
            self._on_repl_source_kind_changed(self._repl_source_combo.currentIndex())
        self._repl_media_path = Path(path)
        self._repl_path_label.setText(str(self._repl_media_path))

    def _on_repl_pick_colmap_sparse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "COLMAP sparse model folder (often sparse/0)")
        if not d:
            return
        self._repl_colmap_sparse = Path(d)
        self._repl_colmap_label.setText(str(self._repl_colmap_sparse))

    def _on_replication_status_slot(self, stage: str, message: str, progress: float) -> None:
        self._stage_label.setText(f"{stage} — {message}")
        if progress >= 0.0:
            self._progress_bar.setVisible(True)
            self._progress_bar.setValue(int(max(0.0, min(1.0, progress)) * 100))
        self._append_replication_log(stage, message, progress)

    def _reset_replication_log(self) -> None:
        self._replication_log_lines = []
        if hasattr(self, "_repl_log_view"):
            self._repl_log_view.clear()

    def _append_replication_log(self, stage: str, message: str, progress: float = -1.0) -> None:
        msg = " ".join(str(message or "").split())
        if not msg:
            return
        pct = f" {int(max(0.0, min(1.0, progress)) * 100):3d}%" if progress >= 0.0 else "    "
        line = f"{time.strftime('%H:%M:%S')} {pct} [{stage}] {msg}"
        self._replication_log_lines.append(line)
        self._replication_log_lines = self._replication_log_lines[-500:]
        if hasattr(self, "_repl_log_view"):
            self._repl_log_view.appendPlainText(line)
            bar = self._repl_log_view.verticalScrollBar()
            bar.setValue(bar.maximum())

    def _on_replication_finished_slot(self, ok: bool, message: str) -> None:
        self._repl_prep_busy = False
        self._prep_workspace_btn.setEnabled(True)
        self._run_gaussian_btn.setEnabled(True)
        if self._mode_combo.currentData() == "single":
            self._generate_btn.setEnabled(self._image_path is not None)
        if ok:
            self._last_replication_workspace = message
            depth_dir = self._last_replication_depth_assist_dir
            extra = ""
            if depth_dir:
                extra = f"\nDepth assist:\n{depth_dir}"
            self._repl_status_label.setText(f"Workspace:\n{message}{extra}")
            self._open_ws_btn.setEnabled(True)
            self._stage_label.setText("Replication workspace ready.")
            self._progress_bar.setValue(100)
            self._progress_bar.setVisible(True)
            if self._repl_render_requested:
                if not self._show_replication_render_output():
                    self.errorRaised.emit(
                        "Gaussian run completed workspace prep, but no preview render artifact was found."
                    )
            self.replicationWorkspaceReady.emit(message)
        else:
            self._progress_bar.setVisible(False)
            self.errorRaised.emit(message)
        self._repl_render_requested = False

    def _on_run_gaussian_render_clicked(self) -> None:
        self._repl_render_requested = True
        self._on_prepare_replication_workspace()

    def _on_open_replication_workspace(self) -> None:
        p = self._last_replication_workspace
        if not p or not Path(p).is_dir():
            self.errorRaised.emit("No workspace folder available.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(p).resolve())))

    def _on_prepare_replication_workspace(self) -> None:
        if self._repl_prep_busy:
            return
        from mlops.gaussian_splat.replication import (  # noqa: PLC0415
            new_job_id,
            prepare_replication_workspace,
        )
        from mlops.gaussian_splat.true_gaussian import run_true_gaussian_pipeline  # noqa: PLC0415

        if self._repl_media_path is None:
            self.errorRaised.emit("Select an image folder or video file first.")
            return
        cal = str(self._repl_calibration_combo.currentData() or "none")
        if cal == "import_sparse" and self._repl_colmap_sparse is None:
            self.errorRaised.emit("Choose a COLMAP sparse model folder for import.")
            return

        workspace = Path(__file__).resolve().parents[4] / ".gaussian_splat" / "jobs" / new_job_id()
        source_kind = str(self._repl_source_combo.currentData() or "image_folder")
        media_path = self._repl_media_path
        colmap_sparse = self._repl_colmap_sparse
        symlink = self._repl_symlink_chk.isChecked()
        vfps = float(self._repl_video_fps.value())
        vmax = int(self._repl_max_frames.value())
        depth_assist_enabled = (
            self._repl_depth_assist_chk.isChecked()
            and self._depth_mlpackage is not None
        )
        depth_assist_samples = int(self._repl_depth_assist_count.value())

        def on_status(stage: str, message: str, progress: float = -1.0) -> None:
            self._replication_status.emit(stage, message, progress)

        self._repl_prep_busy = True
        self._prep_workspace_btn.setEnabled(False)
        self._run_gaussian_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        self._stage_label.setText("Preparing replication workspace…")
        self._reset_replication_log()
        self._append_replication_log(
            "start",
            "Starting Gaussian render workflow" if self._repl_render_requested else "Preparing workspace only",
            0.0,
        )

        panel = self

        def _run() -> None:
            try:
                prep_cal = "none" if panel._repl_render_requested and cal == "run_colmap" else cal
                manifest = prepare_replication_workspace(
                    workspace,
                    source_kind=source_kind,  # type: ignore[arg-type]
                    source_path=media_path,
                    calibration=prep_cal,  # type: ignore[arg-type]
                    colmap_sparse_user=colmap_sparse if prep_cal == "import_sparse" else None,
                    video_fps=vfps,
                    video_max_frames=vmax,
                    prefer_symlink=symlink,
                    on_status=on_status,
                )
                depth_assist_dir = ""
                if depth_assist_enabled:
                    depth_assist_dir = panel._generate_depth_assist_previews(
                        workspace=workspace,
                        image_paths=[Path(p) for p in manifest.image_paths],
                        max_samples=depth_assist_samples,
                        on_status=on_status,
                    )
                panel._last_replication_depth_assist_dir = depth_assist_dir
                if panel._repl_render_requested:
                    run_true_gaussian_pipeline(workspace, on_status=on_status)
                panel._replication_finished.emit(True, str(workspace))
            except Exception as exc:
                panel._replication_finished.emit(False, str(exc))

        threading.Thread(target=_run, daemon=True, name="replication-prep").start()

    def _show_replication_render_output(self) -> bool:
        workspace_raw = self._last_replication_workspace
        if workspace_raw:
            workspace = Path(workspace_raw)
            manifest_path = workspace / "gaussian_run_manifest.json"
            splat_path = workspace / "nerfstudio" / "gaussian_export" / "splat.ply"
            if splat_path.is_file():
                config_path = ""
                train_log = ""
                export_log = ""
                if manifest_path.is_file():
                    try:
                        data = json.loads(manifest_path.read_text(encoding="utf-8"))
                        config_path = str(data.get("train_config_path") or "")
                        train_log = str(data.get("train_log_path") or "")
                        export_log = str(data.get("export_log_path") or "")
                    except Exception:
                        pass
                self._stage_label.setText("Gaussian splat ready.")
                if hasattr(self, "_viewer_status"):
                    self._viewer_status.setText("Gaussian splat ready")
                    self._viewer_status.setProperty("state", "connected")
                    repolish(self._viewer_status)
                if hasattr(self, "_viewer_placeholder"):
                    self._viewer_placeholder.setText(
                        "True Gaussian Splat Result\n\n"
                        f"Workspace: {workspace}\n"
                        f"Splat PLY: {splat_path}\n"
                        f"Train config: {config_path or '(missing)'}\n"
                        f"Train log: {train_log or '(missing)'}\n"
                        f"Export log: {export_log or '(missing)'}"
                    )
                self._output_cell.setVisible(True)
                self._output_thumb.clear()
                self._output_thumb.setText("Gaussian splat exported as PLY.")
                self._dl_btn.setProperty("_artifact_path", str(splat_path))
                self._html_btn.setProperty("_html_path", "")
                return True

        depth_root = self._last_replication_depth_assist_dir
        if not depth_root:
            return False
        root = Path(depth_root)
        if not root.is_dir():
            return False
        glbs = sorted(root.glob("**/output.glb"))
        if not glbs:
            return False
        map_html = self._write_gaussian_map_artifacts(root, glbs)
        # Prefer latest artifact for the main preview card.
        glb = glbs[-1]
        preview = glb.parent / "preview.png"
        item_html = glb.parent / "preview.html"
        pseudo = SimpleNamespace(
            glb_path=str(glb),
            preview_path=str(preview) if preview.is_file() else "",
            preview_html_path=(
                str(map_html)
                if map_html is not None and map_html.is_file()
                else (str(item_html) if item_html.is_file() else "")
            ),
        )
        self._stage_label.setText("Gaussian preview render ready.")
        self._show_output(pseudo)
        if hasattr(self, "_viewer_status"):
            self._viewer_status.setText("Gaussian previews ready")
            self._viewer_status.setProperty("state", "connected")
            repolish(self._viewer_status)
        if hasattr(self, "_viewer_placeholder"):
            lines: list[str] = []
            for idx, g in enumerate(glbs, start=1):
                p = g.parent / "preview.png"
                src_name = g.parent.name
                if src_name.startswith(tuple(str(i).zfill(3) for i in range(1, 1000))):
                    src_name = src_name.split("_", 1)[-1]
                lines.append(
                    f"{idx}. {src_name}\n"
                    f"   GLB: {g.name}\n"
                    f"   Preview: {p.name if p.is_file() else '(missing)'}"
                )
            self._viewer_placeholder.setText(
                "Depth Assist Results (Gaussian path)\n\n"
                f"Workspace: {self._last_replication_workspace}\n"
                f"Depth assist dir: {depth_root}\n"
                f"3D map: {map_html if map_html is not None else '(not written)'}\n"
                f"Artifacts: {len(glbs)}\n\n"
                + "\n\n".join(lines)
            )
        return True

    def _write_gaussian_map_artifacts(self, depth_root: Path, glbs: list[Path]) -> Optional[Path]:
        if not glbs:
            return None
        map_dir = depth_root / "map"
        map_dir.mkdir(parents=True, exist_ok=True)
        items: list[dict[str, str]] = []
        for i, g in enumerate(glbs, start=1):
            sample_dir = g.parent
            p = sample_dir / "preview.png"
            h = sample_dir / "preview.html"
            items.append(
                {
                    "index": str(i),
                    "sample": sample_dir.name,
                    "glb_path": str(g),
                    "preview_path": str(p) if p.is_file() else "",
                    "preview_html_path": str(h) if h.is_file() else "",
                }
            )
        manifest_path = map_dir / "gaussian_map.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "workspace": self._last_replication_workspace,
                    "depth_assist_root": str(depth_root),
                    "items": items,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        rows = []
        for it in items:
            preview_tag = ""
            if it["preview_path"]:
                preview_tag = (
                    f'<img src="{html.escape(it["preview_path"])}" '
                    'style="max-width:180px;max-height:120px;border:1px solid #333;border-radius:6px;" />'
                )
            rows.append(
                "<tr>"
                f"<td>{html.escape(it['index'])}</td>"
                f"<td>{html.escape(it['sample'])}</td>"
                f'<td><a href="{html.escape(it["glb_path"])}">GLB</a></td>'
                f'<td><a href="{html.escape(it["preview_html_path"])}">Preview HTML</a></td>'
                f"<td>{preview_tag}</td>"
                "</tr>"
            )
        index_path = map_dir / "index.html"
        index_path.write_text(
            "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
            "<title>Gaussian 3D Map</title>"
            "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;background:#111;color:#ddd;padding:16px}"
            "table{border-collapse:collapse;width:100%}th,td{border:1px solid #333;padding:8px;vertical-align:top}"
            "a{color:#8fc7ff}th{background:#1c1c1c;text-align:left}</style></head><body>"
            "<h2>Gaussian 3D Map</h2>"
            f"<p>Workspace: <code>{html.escape(self._last_replication_workspace)}</code></p>"
            f"<p>Depth root: <code>{html.escape(str(depth_root))}</code></p>"
            "<table><thead><tr><th>#</th><th>Sample</th><th>GLB</th><th>HTML</th><th>Preview</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            f"<p>Manifest: <code>{html.escape(str(manifest_path))}</code></p>"
            "</body></html>",
            encoding="utf-8",
        )
        return index_path

    def _generate_depth_assist_previews(
        self,
        *,
        workspace: Path,
        image_paths: list[Path],
        max_samples: int,
        on_status,
    ) -> str:
        from mlops.trellis2.depth_anything_local import generate_depth_glb  # noqa: PLC0415

        if self._depth_mlpackage is None:
            return ""
        if not image_paths:
            return ""
        out_root = workspace / "depth_assist"
        out_root.mkdir(parents=True, exist_ok=True)
        total = min(max(1, int(max_samples)), len(image_paths))
        if len(image_paths) <= total:
            picked = image_paths
        else:
            step = max(1, len(image_paths) // total)
            picked = image_paths[::step][:total]
        for idx, p in enumerate(picked, start=1):
            prog = 0.82 + 0.16 * (idx / max(total, 1))
            on_status("depth_assist", f"Depth preview {idx}/{total}: {p.name}", prog)
            sample_dir = out_root / f"{idx:03d}_{p.stem}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            generate_depth_glb(
                mlpackage_path=self._depth_mlpackage,
                image_path=p,
                out_dir=sample_dir,
                params=self._defaults,
            )
        return str(out_root)

    def _on_comfy_test_connection(self) -> None:
        from mlops.three_d.comfy_local import comfy_ping, default_comfy_base_url  # noqa: PLC0415

        raw = self._comfy_url_edit.text().strip()
        url = raw or default_comfy_base_url()
        ok, msg = comfy_ping(url)
        self._comfy_conn_status.setText(msg)
        if not ok:
            self.errorRaised.emit(msg)

    def _default_comfy_root(self) -> Path:
        return (Path.home() / "ComfyUI").expanduser()

    def _discover_comfy_root(self) -> Optional[Path]:
        candidates = [
            (Path(__file__).resolve().parents[4] / "mlops" / "three_d" / "ComfyUI-master"),
            (Path(__file__).resolve().parents[4] / "mlops" / "three_d" / "ComfyUI"),
            self._default_comfy_root(),
            (Path.home() / "Applications" / "ComfyUI").expanduser(),
            (Path.home() / "dev" / "ComfyUI").expanduser(),
            (Path.home() / "Developer" / "ComfyUI").expanduser(),
            (Path(__file__).resolve().parents[4] / "ComfyUI"),
            (Path(__file__).resolve().parents[4] / "mlops" / "ComfyUI"),
        ]
        for c in candidates:
            if (c / "main.py").is_file():
                return c
        return None

    def _comfy_launch_config(self) -> tuple[Optional[Path], list[str], str]:
        from mlops.three_d.comfy_local import default_comfy_base_url  # noqa: PLC0415

        root_raw = os.environ.get("CVOPS_COMFY_ROOT", "").strip()
        if root_raw:
            root = Path(root_raw).expanduser()
        else:
            root = self._discover_comfy_root()
        cmd_raw = os.environ.get("CVOPS_COMFY_CMD", "").strip()
        if cmd_raw:
            argv = shlex.split(cmd_raw)
            launch_root = root
        else:
            base = default_comfy_base_url()
            port = "8188"
            try:
                from urllib.parse import urlparse  # noqa: PLC0415

                parsed = urlparse(base)
                if parsed.port is not None:
                    port = str(parsed.port)
            except Exception:
                pass
            if root is not None and (root / "main.py").is_file():
                argv = ["python3", "main.py", "--listen", "127.0.0.1", "--port", port]
                if self._caps.os == "darwin":
                    argv.append("--cpu")
                launch_root = root
            else:
                argv = ["comfyui", "--listen", "127.0.0.1", "--port", port]
                if self._caps.os == "darwin":
                    argv.append("--cpu")
                launch_root = None
        pretty = " ".join(argv)
        return launch_root, argv, pretty

    def _autostart_enabled(self) -> bool:
        raw = os.environ.get("CVOPS_COMFY_AUTOSTART", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _maybe_autostart_comfy(self) -> None:
        from mlops.three_d.comfy_local import comfy_ping, default_comfy_base_url  # noqa: PLC0415

        if not self._autostart_enabled():
            return
        base = (self._comfy_url_edit.text().strip() if hasattr(self, "_comfy_url_edit") else "") or default_comfy_base_url()
        ok, msg = comfy_ping(base)
        if ok:
            self._comfy_conn_status.setText("ComfyUI already running.")
            return
        root, argv, pretty = self._comfy_launch_config()
        self._comfy_conn_status.setText(f"Starting ComfyUI: {pretty}")
        self._stage_label.setText("Starting local ComfyUI runtime…")
        try:
            self._comfy_proc = subprocess.Popen(
                argv,
                cwd=str(root) if root is not None else None,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            hint = (
                " Set CVOPS_COMFY_ROOT to your ComfyUI folder or CVOPS_COMFY_CMD "
                "to your launch command."
            )
            self._comfy_conn_status.setText(f"ComfyUI launch failed: {exc}.{hint}")
            return

        def _probe() -> None:
            deadline = time.time() + 90.0
            last = ""
            ok2 = False
            while time.time() < deadline:
                ok_local, m = comfy_ping(base)
                last = m
                if ok_local:
                    ok2 = True
                    break
                time.sleep(1.5)
            final = "ComfyUI ready." if ok2 else f"ComfyUI started but not reachable yet: {last}"
            QTimer.singleShot(0, lambda: self._on_autostart_probe_done(ok2, final))

        threading.Thread(target=_probe, daemon=True, name="comfy-autostart-probe").start()

    def _on_autostart_probe_done(self, ok: bool, message: str) -> None:
        self._comfy_conn_status.setText(message)
        if ok:
            self._stage_label.setText("Idle — select an image, then generate.")

    def _comfy_repo_root(self) -> Path:
        return Path(__file__).resolve().parents[4] / "mlops" / "three_d" / "ComfyUI-Trellis2-main"

    def _on_open_comfy_repo(self) -> None:
        repo = self._comfy_repo_root()
        if not repo.is_dir():
            self.errorRaised.emit(f"ComfyUI-Trellis2 repo not found: {repo}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(repo.resolve())))

    def _on_open_comfy_setup_readme(self) -> None:
        readme = self._comfy_repo_root() / "README.md"
        if not readme.is_file():
            self.errorRaised.emit(f"Setup README not found: {readme}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(readme.resolve())))

    # ------------------------------------------------------------------ #
    # Collect SamplingParams from form widgets
    # ------------------------------------------------------------------ #

    def _collect_params(self):  # noqa: ANN201
        from mlops.trellis2 import SamplingParams  # noqa: PLC0415
        return SamplingParams(
            seed=self._p_seed.value(),
            randomize_seed=self._p_randomize_seed.isChecked(),
            resolution=self._p_resolution.currentData(),
            ss_guidance_strength=self._p_ss_g_strength.value(),
            ss_guidance_rescale=self._p_ss_g_rescale.value(),
            ss_sampling_steps=self._p_ss_steps.value(),
            ss_rescale_t=self._p_ss_rescale_t.value(),
            shape_slat_guidance_strength=self._p_shape_g_strength.value(),
            shape_slat_guidance_rescale=self._p_shape_g_rescale.value(),
            shape_slat_sampling_steps=self._p_shape_steps.value(),
            shape_slat_rescale_t=self._p_shape_rescale_t.value(),
            tex_slat_guidance_strength=self._p_tex_g_strength.value(),
            tex_slat_guidance_rescale=self._p_tex_g_rescale.value(),
            tex_slat_sampling_steps=self._p_tex_steps.value(),
            tex_slat_rescale_t=self._p_tex_rescale_t.value(),
            decimation_target=self._p_decimation.value(),
            texture_size=int(self._p_texture_size.currentData()),
        )

    # ------------------------------------------------------------------ #
    # Slot: select image
    # ------------------------------------------------------------------ #

    def _set_single_image_source(self, path: Path) -> None:
        path = path.expanduser()
        try:
            path = path.resolve()
        except OSError:
            pass
        if not path.is_file():
            self.errorRaised.emit(f"Image not found: {path}")
            return
        self._image_path = path
        self._filename_label.setText(path.name)
        pix = QPixmap(str(path))
        if not pix.isNull():
            pix = pix.scaled(
                _THUMB_SIZE,
                _THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._preview_thumb.setPixmap(pix)
            self._preview_thumb.setText("")
        else:
            self._preview_thumb.clear()
            self._preview_thumb.setText("Preview failed")
        if self._mode_combo.currentData() == "single":
            self._generate_btn.setEnabled(True)
        self._highlight_quick_sample_for_current_image()

    def _highlight_quick_sample_for_current_image(self) -> None:
        if not hasattr(self, "_quick_samples_list") or self._image_path is None:
            return
        try:
            want = str(self._image_path.resolve())
        except OSError:
            want = str(self._image_path)
        self._quick_samples_list.blockSignals(True)
        found = False
        for i in range(self._quick_samples_list.count()):
            it = self._quick_samples_list.item(i)
            if it is None:
                continue
            raw = it.data(Qt.ItemDataRole.UserRole)
            if isinstance(raw, str) and raw == want:
                self._quick_samples_list.setCurrentRow(i)
                found = True
                break
        if not found:
            self._quick_samples_list.clearSelection()
        self._quick_samples_list.blockSignals(False)
        self._on_quick_sample_selection_changed()

    def _on_select_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.webp)",
        )
        if not path:
            return
        self._set_single_image_source(Path(path))

    # ------------------------------------------------------------------ #
    # Slot: generate
    # ------------------------------------------------------------------ #

    def _on_generate_clicked(self) -> None:
        if self._mode_combo.currentData() != "single":
            return
        if self._image_path is None:
            return

        backend = self._backend_combo.currentData() or "gaussian_local"
        if backend == "gaussian_local":
            self._start_gaussian_single_image_run()
            return
        if backend == "apple_mlx":
            try:
                from mlops.three_d.trellis2_apple import available as _apple_available  # noqa: PLC0415

                ok, msg = _apple_available()
            except Exception as exc:
                ok, msg = False, str(exc)
            if not ok:
                self.errorRaised.emit(msg)
                return
        elif backend == "depth_local":
            from mlops.trellis2.depth_anything_local import depth_bundle_available as _depth_ok

            ok, msg = _depth_ok()
            if not ok or self._depth_mlpackage is None:
                self.errorRaised.emit(msg)
                return
        else:
            self.errorRaised.emit(f"Unsupported backend in this mode: {backend}")
            return

        params = self._collect_params()
        params_dict = dict(params.as_dict())
        job = self._store.create(backend=backend, params=params_dict)
        job_dir = self._store.dir(job.job_id)

        # Copy image into job directory.
        suffix = self._image_path.suffix.lower() or ".png"
        dest_image = job_dir / f"input{suffix}"
        try:
            shutil.copy2(self._image_path, dest_image)
        except Exception as exc:
            self.errorRaised.emit(f"Could not copy image: {exc}")
            return

        qs = self._save_quick_sample_for_job(dest_image, job.job_id)
        self._refresh_quick_samples_list()
        if qs is not None:
            self._select_quick_sample_path(qs)

        self._active_job_id = job.job_id
        self._generate_btn.setEnabled(False)
        self._output_cell.setVisible(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._stage_label.setText("Queued — starting…")

        self._start_job_thread(job, dest_image, params)
        self._poll_timer.start()
        self._refresh_history()

    def _start_gaussian_single_image_run(self) -> None:
        if self._image_path is None:
            self.errorRaised.emit("Select an image first.")
            return
        self.errorRaised.emit(
            "True Gaussian splatting needs overlapping multi-view input, not a single depth estimate. "
            "Use Replication with an image folder/video, or choose Local depth for a single-image depth mesh."
        )

    def _start_job_thread(self, job, image_path: Path, params) -> None:  # noqa: ANN001
        store = self._store

        def _run() -> None:
            from mlops.trellis2 import Trellis2Client, Trellis2Error  # noqa: PLC0415
            from mlops.trellis2.depth_anything_local import (  # noqa: PLC0415
                generate_depth_glb,
                resolve_depth_mlpackage,
            )

            def on_status(stage: str, message: str, progress: float = -1.0) -> None:
                j = store.load(job.job_id)
                if j is None:
                    return
                j.stage = stage
                j.message = message
                if progress >= 0.0:
                    j.progress = max(0.0, min(1.0, progress))
                from mlops.trellis2 import JobStatus  # noqa: PLC0415
                j.status = JobStatus.RUNNING
                store.save(j)

            try:
                from mlops.trellis2 import JobStatus  # noqa: PLC0415
                j = store.load(job.job_id)
                if j is None:
                    return
                j.status = JobStatus.RUNNING
                j.input_path = str(image_path)
                store.save(j)

                if job.backend == "apple_mlx":
                    from mlops.three_d.trellis2_apple import (  # noqa: PLC0415
                        default_api_base_url,
                        run_trellis2_apple_job,
                    )

                    result = run_trellis2_apple_job(
                        base_url=str((job.params or {}).get("apple_api_url") or "").strip()
                        or default_api_base_url(),
                        image_path=image_path,
                        out_dir=store.dir(job.job_id),
                        params=params,
                        on_status=on_status,
                    )
                elif job.backend == "depth_local":
                    mp = resolve_depth_mlpackage()
                    if mp is None:
                        raise Trellis2Error("DepthAnything Core ML bundle not found.")
                    result = generate_depth_glb(
                        mlpackage_path=mp,
                        image_path=image_path,
                        out_dir=store.dir(job.job_id),
                        params=params,
                        on_status=on_status,
                    )
                elif job.backend == "comfy_local":
                    from mlops.three_d.comfy_local import (  # noqa: PLC0415
                        default_comfy_base_url,
                        run_comfy_trellis_job,
                    )

                    wf_raw = (job.params or {}).get("comfy_workflow") or ""
                    wf_path = Path(str(wf_raw))
                    if not wf_path.is_file():
                        raise Trellis2Error("Missing comfy_workflow path in job record.")
                    base = str((job.params or {}).get("comfy_url") or "").strip()
                    if not base:
                        base = default_comfy_base_url()
                    result = run_comfy_trellis_job(
                        comfy_base_url=base,
                        workflow_path=wf_path,
                        image_path=image_path,
                        out_dir=store.dir(job.job_id),
                        params=params,
                        on_status=on_status,
                        job_prefix=str(job.job_id),
                    )
                else:
                    client = Trellis2Client()
                    result = client.generate(
                        image_path=image_path,
                        params=params,
                        out_dir=store.dir(job.job_id),
                        on_status=on_status,
                    )

                j = store.load(job.job_id) or j
                j.status = JobStatus.COMPLETED
                j.stage = "done"
                j.message = "complete"
                j.preview_path = result.get("preview_path", "")
                j.preview_html_path = result.get("preview_html_path", "")
                j.glb_path = result.get("glb_path", "")
                j.seed = result.get("seed", "")
                store.save(j)
            except Trellis2Error as exc:
                log.exception("trellis2 cloud job failed")
                from mlops.trellis2 import JobStatus  # noqa: PLC0415
                j = store.load(job.job_id)
                if j is not None:
                    j.status = JobStatus.FAILED
                    j.stage = "error"
                    j.error = str(exc)
                    store.save(j)
            except Exception as exc:
                log.exception("unexpected trellis2 failure")
                from mlops.trellis2 import JobStatus  # noqa: PLC0415
                j = store.load(job.job_id)
                if j is not None:
                    j.status = JobStatus.FAILED
                    j.stage = "error"
                    j.error = f"unexpected: {exc!r}"
                    store.save(j)

        threading.Thread(
            target=_run,
            name=f"trellis2-{job.job_id}",
            daemon=True,
        ).start()

    # ------------------------------------------------------------------ #
    # Polling
    # ------------------------------------------------------------------ #

    def _poll_job(self) -> None:
        if self._active_job_id is None:
            self._poll_timer.stop()
            return

        from mlops.trellis2 import JobStatus  # noqa: PLC0415
        fresh = self._store.load(self._active_job_id)
        if fresh is None:
            self._poll_timer.stop()
            return

        if fresh.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            stage = (fresh.stage or "running").replace("_", " ").title()
            self._stage_label.setText(f"{stage} — {fresh.message or 'working…'}")
            pct = int(max(0.0, min(1.0, fresh.progress)) * 100)
            self._progress_bar.setValue(pct)

        elif fresh.status == JobStatus.COMPLETED:
            self._poll_timer.stop()
            self._stage_label.setText("Completed — artifacts ready.")
            self._progress_bar.setValue(100)
            self._show_output(fresh)
            self._generate_btn.setEnabled(self._image_path is not None)
            self.generationCompleted.emit(fresh.glb_path or "")
            self._refresh_history()

        elif fresh.status == JobStatus.FAILED:
            self._poll_timer.stop()
            self._stage_label.setText(f"Failed — {fresh.error}")
            self._progress_bar.setVisible(False)
            self._generate_btn.setEnabled(self._image_path is not None)
            self.errorRaised.emit(fresh.error or "job failed")
            self._refresh_history()

    # ------------------------------------------------------------------ #
    # Output display
    # ------------------------------------------------------------------ #

    def _show_output(self, job) -> None:  # noqa: ANN001
        self._output_cell.setVisible(True)
        self._output_thumb.clear()
        if job.preview_path and Path(job.preview_path).exists():
            pix = QPixmap(job.preview_path)
            if not pix.isNull():
                pix = pix.scaledToHeight(180, Qt.TransformationMode.SmoothTransformation)
                self._output_thumb.setPixmap(pix)
                self._output_thumb.setText("")
            else:
                self._output_thumb.setText("Preview file could not be loaded.")
        else:
            self._output_thumb.setText("No static preview was produced for this run.")
        self._dl_btn.setProperty("_artifact_path", job.glb_path or "")
        self._html_btn.setProperty("_html_path", job.preview_html_path or "")

    def _on_download_glb(self) -> None:
        artifact_path = self._dl_btn.property("_artifact_path") or ""
        if not artifact_path or not Path(artifact_path).exists():
            self.errorRaised.emit("3D artifact file not found.")
            return
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save 3D artifact",
            Path(artifact_path).name,
            "3D artifact (*.glb *.ply);;All files (*)",
        )
        if not dest:
            return
        try:
            shutil.copy2(artifact_path, dest)
        except Exception as exc:
            self.errorRaised.emit(f"Copy failed: {exc}")

    def _on_view_html(self) -> None:
        html_path = self._html_btn.property("_html_path") or ""
        if html_path and Path(html_path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(html_path))
        else:
            self.errorRaised.emit("HTML preview file not found.")

    # ------------------------------------------------------------------ #
    # History table
    # ------------------------------------------------------------------ #

    def _refresh_history(self) -> None:
        from mlops.trellis2 import JobStatus  # noqa: PLC0415
        jobs = self._store.list_recent(limit=8)
        self._history_table.setRowCount(0)
        for row, j in enumerate(jobs):
            self._history_table.insertRow(row)
            self._history_table.setItem(row, 0, QTableWidgetItem(j.job_id))
            self._history_table.setItem(row, 1, QTableWidgetItem(j.status.value))
            detail = j.message or j.error or j.stage or ""
            self._history_table.setItem(row, 2, QTableWidgetItem(detail))
            if j.status == JobStatus.COMPLETED:
                open_btn = QPushButton("Load")
                open_btn.setProperty("buttonRole", "secondary")
                repolish(open_btn)
                open_btn.setProperty("_job_id", j.job_id)
                open_btn.clicked.connect(self._on_history_open)
                self._history_table.setCellWidget(row, 3, open_btn)
        self._history_table.resizeColumnsToContents()

    def _on_history_open(self) -> None:
        btn = self.sender()
        if btn is None:
            return
        job_id = btn.property("_job_id") or ""
        if not job_id:
            return
        job = self._store.load(job_id)
        if job is None:
            return
        self._active_job_id = job_id
        self._stage_label.setText("Completed — artifacts ready.")
        self._progress_bar.setValue(100)
        self._progress_bar.setVisible(True)
        self._show_output(job)
