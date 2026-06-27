from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional
import urllib.error
import urllib.parse
import urllib.request

from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImageReader, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
    QDialog,
)

from ..config import CVOPS_BASE_URL, MODELS_DIR
from .theme import beacon_title_tag_css, current_color_scheme, text_css, text_hex, theme_hex, theme_rgba


def _gallery_dark() -> bool:
    return current_color_scheme() in ("tactical", "solarized_dark", "material_dark", "wear_marathon")


def _section_style() -> str:
    border = theme_rgba("accent_dark", 0.14 if _gallery_dark() else 0.18)
    bg = theme_rgba("panel", 0.66 if _gallery_dark() else 0.90)
    return (
        f"QFrame#gallerySection {{ background: {bg}; border: 1px solid {border}; border-radius: 12px; }}"
    )


def _heading_style() -> str:
    if current_color_scheme() == "beacon":
        return beacon_title_tag_css(font_size=10, padding="3px 8px")
    title = theme_rgba("strip_soft", 0.84) if _gallery_dark() else theme_rgba("accent_dark", 0.82)
    return f"font-size: 12px; font-weight: 700; color: {title}; letter-spacing: 0.2px;"


def _hint_style() -> str:
    return f"font-size: 10px; color: {text_css(0.58)}; letter-spacing: 0.12px; line-height: 1.45;"


def _body_style() -> str:
    return f"font-size: 10px; color: {text_css(0.72)}; line-height: 1.45;"


def _input_style() -> str:
    return (
        f"QLineEdit {{ background: {theme_rgba('panel', 0.74)}; "
        f"border: 1px solid {theme_rgba('accent_dark', 0.20)}; color: {text_hex()}; "
        f"padding: 6px 10px; font-size: 10px; border-radius: 6px; }}"
        f"QLineEdit:focus {{ border: 1px solid {theme_rgba('accent_dark', 0.44)}; }}"
    )


def _toggle_style(*, selected: bool) -> str:
    bg = theme_rgba("strip_soft", 0.14 if selected else 0.03)
    border = theme_rgba("strip_soft", 0.56 if selected else 0.22)
    fg = text_hex() if selected else text_css(0.78)
    return (
        f"QPushButton {{ background: {bg}; color: {fg}; border: 1px solid {border}; "
        f"border-radius: 7px; padding: 5px 12px; font-size: 10px; font-weight: 600; }}"
    )


def _row_action_style() -> str:
    return (
        f"QPushButton {{ font-size: 9px; font-weight: 600; padding: 5px 10px; border-radius: 6px; "
        f"border: 1px solid {theme_rgba('accent_dark', 0.22)}; background: {theme_rgba('panel', 0.70)}; "
        f"color: {text_css(0.86)}; }}"
        f"QPushButton:hover {{ background: {theme_rgba('strip_soft', 0.10)}; "
        f"border-color: {theme_rgba('strip_soft', 0.38)}; color: {text_hex()}; }}"
    )


def _destructive_btn_style() -> str:
    line = theme_rgba("privacy_warn", 0.58)
    return (
        f"QPushButton {{ color: {text_hex()}; background: {theme_rgba('privacy_warn', 0.22)}; "
        f"border: 1px solid {line}; border-radius: 6px; padding: 5px 10px; font-size: 9px; font-weight: 700; }}"
        f"QPushButton:hover {{ background: {theme_rgba('privacy_warn', 0.30)}; }}"
    )


def _row_frame_style() -> str:
    return (
        f"QFrame#galleryListRow {{ background: {theme_rgba('panel', 0.56)}; "
        f"border: 1px solid {theme_rgba('accent_dark', 0.12)}; border-radius: 9px; }}"
    )


def _thumb_style(*, placeholder: bool = False) -> str:
    extra = ""
    if placeholder:
        extra = f" color: {text_css(0.46)}; font-size: 9px; font-weight: 600;"
    return (
        f"QLabel#galleryRowThumb {{ border: 1px solid {theme_rgba('accent_dark', 0.16)}; "
        f"border-radius: 8px; background: {theme_rgba('panel', 0.70)};{extra} }}"
    )


def _metric_card_style() -> str:
    return (
        f"QFrame#galleryMetricCard {{ background: {theme_rgba('panel', 0.72)}; "
        f"border: 1px solid {theme_rgba('accent_dark', 0.14)}; border-radius: 9px; }}"
    )


def _dialog_style() -> str:
    return f"QDialog {{ background: {theme_rgba('panel', 0.97)}; color: {text_hex()}; }}"


def _pill_caption_style() -> str:
    return (
        f"font-size: 9px; color: {theme_hex('accent_dark')}; letter-spacing: 0.55px; "
        f"font-weight: 600; text-transform: uppercase;"
    )


def _progress_style() -> str:
    return (
        f"QProgressBar {{ border: none; border-radius: 4px; background: {theme_rgba('strip_soft', 0.10)}; "
        f"font-size: 9px; color: {text_css(0.85)}; min-height: 14px; text-align: center; }}"
        f"QProgressBar::chunk {{ background: {theme_rgba('strip_soft', 0.86)}; border-radius: 4px; }}"
    )


def _grid_thumb_style() -> str:
    return (
        f"QLabel#galleryGridThumb {{ border: 1px solid {theme_rgba('accent_dark', 0.14)}; "
        f"border-radius: 8px; background: {theme_rgba('panel', 0.72)}; }}"
    )


def _similarity_result_card_style() -> str:
    return (
        f"QFrame#galleryResultCard {{ background: {theme_rgba('panel', 0.68)}; "
        f"border: 1px solid {theme_rgba('accent_dark', 0.14)}; border-radius: 10px; }}"
    )


def _sheet_cell_style() -> str:
    return (
        f"QFrame#gallerySheetCell {{ background: {theme_rgba('panel', 0.62)}; "
        f"border: 1px solid {theme_rgba('accent_dark', 0.14)}; border-radius: 8px; }}"
        f"QFrame#gallerySheetCell:hover {{ background: {theme_rgba('strip_soft', 0.08)}; "
        f"border-color: {theme_rgba('strip_soft', 0.30)}; }}"
    )


def _label_badge_style(*, present: bool) -> str:
    color = theme_rgba("strip_soft", 0.90) if present else theme_rgba("privacy_warn", 0.86)
    bg = theme_rgba("strip_soft", 0.14) if present else theme_rgba("privacy_warn", 0.14)
    return (
        f"QLabel {{ color: {color}; background: {bg}; "
        f"border: 1px solid {color}; border-radius: 5px; padding: 2px 6px; "
        "font-size: 9px; font-weight: 800; }}"
    )


_FILE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.webp)"


def _load_thumb(source_path: str, thumb_b64: str = "", *, size: int = 56) -> QPixmap:
    pix = QPixmap()
    if source_path and os.path.exists(source_path):
        pix.load(source_path)
    elif thumb_b64:
        try:
            pix.loadFromData(base64.b64decode(thumb_b64))
        except Exception:
            pix = QPixmap()
    if pix.isNull():
        return QPixmap()
    return pix.scaled(
        size,
        size,
        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
        Qt.TransformationMode.SmoothTransformation,
    )


def _label_candidate_paths(source_path: str) -> list[Path]:
    src = Path(str(source_path or "")).expanduser()
    if not src.name:
        return []
    candidates = [src.with_suffix(".txt")]
    parts = list(src.parts)
    lower_parts = [p.lower() for p in parts]
    for idx, part in enumerate(lower_parts):
        if part == "images":
            replaced = list(parts)
            replaced[idx] = "labels"
            candidates.append(Path(*replaced).with_suffix(".txt"))
    if len(parts) >= 3 and parts[-3].lower() == "images":
        candidates.append(Path(*parts[:-3], "labels", parts[-2], src.with_suffix(".txt").name))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = path.as_posix()
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _label_status(source_path: str) -> tuple[bool, str]:
    for path in _label_candidate_paths(source_path):
        try:
            if path.is_file() and path.stat().st_size >= 0:
                return True, str(path)
        except Exception:
            continue
    candidates = _label_candidate_paths(source_path)
    return False, str(candidates[0]) if candidates else ""


class PersonRow(QFrame):
    delete_requested = pyqtSignal(str)
    rename_requested = pyqtSignal(str, str)
    view_requested = pyqtSignal(str)

    def __init__(self, entry: dict[str, Any], *, editable: bool = True, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._name = str(entry.get("name", ""))
        self.setObjectName("galleryListRow")
        self.setStyleSheet(_row_frame_style())
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)

        thumb_label = QLabel()
        thumb_label.setObjectName("galleryRowThumb")
        thumb_label.setFixedSize(44, 44)
        thumb_label.setScaledContents(True)
        thumb_label.setStyleSheet(_thumb_style())
        pix = _load_thumb(str(entry.get("source_path", "")), size=88)
        if not pix.isNull():
            thumb_label.setPixmap(pix)
        else:
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_label.setText("No photo")
            thumb_label.setStyleSheet(_thumb_style(placeholder=True))
        row.addWidget(thumb_label)

        info = QVBoxLayout()
        info.setSpacing(2)
        group_name = str(entry.get("group_name", "") or "")
        if group_name:
            grp = QLabel(group_name.upper())
            grp.setStyleSheet(_pill_caption_style())
            info.addWidget(grp)
        name_label = QLabel(self._name)
        nf = name_label.font()
        nf.setBold(True)
        name_label.setFont(nf)
        name_label.setStyleSheet(f"font-size: 11px; color: {text_hex()};")
        count_label = QLabel(f"{int(entry.get('embedding_count', 0) or 0)} face photo(s)")
        count_label.setStyleSheet(_hint_style())
        info.addWidget(name_label)
        info.addWidget(count_label)
        row.addLayout(info, stretch=1)

        view_btn = QPushButton("View")
        view_btn.setMinimumWidth(56)
        view_btn.setToolTip("Review photos for this person")
        view_btn.setStyleSheet(_row_action_style())
        view_btn.clicked.connect(lambda: self.view_requested.emit(self._name))
        row.addWidget(view_btn)
        if editable:
            ren_btn = QPushButton("Rename")
            ren_btn.setMinimumWidth(56)
            ren_btn.setToolTip("Rename this person")
            ren_btn.setStyleSheet(_row_action_style())
            ren_btn.clicked.connect(self._on_rename)
            del_btn = QPushButton("Delete")
            del_btn.setMinimumWidth(56)
            del_btn.setToolTip("Remove this person from the gallery")
            del_btn.setStyleSheet(_destructive_btn_style())
            del_btn.clicked.connect(lambda: self.delete_requested.emit(self._name))
            row.addWidget(ren_btn)
            row.addWidget(del_btn)

    def _on_rename(self) -> None:
        new_name, ok = QInputDialog.getText(
            self, "Rename Person", f"New name for '{self._name}':", text=self._name
        )
        if ok and new_name.strip() and new_name.strip() != self._name:
            self.rename_requested.emit(self._name, new_name.strip())


class SimilarityRow(QFrame):
    delete_requested = pyqtSignal(int)
    find_requested = pyqtSignal(int)
    view_requested = pyqtSignal(str)

    def __init__(self, entry: dict[str, Any], *, editable: bool = True, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._item_id = int(entry.get("item_id", 0) or 0)
        self._source_path = str(entry.get("source_path", "") or "")
        self.setObjectName("galleryListRow")
        self.setStyleSheet(_row_frame_style())
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(10)

        thumb_label = QLabel()
        thumb_label.setObjectName("galleryRowThumb")
        thumb_label.setFixedSize(52, 52)
        thumb_label.setScaledContents(True)
        thumb_label.setStyleSheet(_thumb_style())
        pix = _load_thumb(self._source_path, str(entry.get("thumb_png_b64", "")), size=104)
        if not pix.isNull():
            thumb_label.setPixmap(pix)
        else:
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_label.setText("No image")
            thumb_label.setStyleSheet(_thumb_style(placeholder=True))
        row.addWidget(thumb_label)

        info = QVBoxLayout()
        info.setSpacing(2)
        batch = str(entry.get("batch_label", "") or "")
        if batch:
            batch_label = QLabel(batch.upper())
            batch_label.setStyleSheet(_pill_caption_style())
            info.addWidget(batch_label)
        name = QLabel(str(entry.get("display_name", "Image") or "Image"))
        nf = name.font()
        nf.setBold(True)
        name.setFont(nf)
        name.setStyleSheet(f"font-size: 11px; color: {text_hex()};")
        source_name = Path(self._source_path).name if self._source_path else "Imported image"
        detail = QLabel(source_name)
        detail.setStyleSheet(_hint_style())
        info.addWidget(name)
        info.addWidget(detail)
        row.addLayout(info, stretch=1)

        find_btn = QPushButton("Similar")
        find_btn.setMinimumWidth(56)
        find_btn.setToolTip("Find visually similar images")
        find_btn.setStyleSheet(_row_action_style())
        find_btn.clicked.connect(lambda: self.find_requested.emit(self._item_id))
        view_btn = QPushButton("View")
        view_btn.setMinimumWidth(56)
        view_btn.setToolTip("Open this image")
        view_btn.setStyleSheet(_row_action_style())
        view_btn.clicked.connect(lambda: self.view_requested.emit(self._source_path))
        row.addWidget(find_btn)
        row.addWidget(view_btn)
        if editable:
            del_btn = QPushButton("Delete")
            del_btn.setMinimumWidth(56)
            del_btn.setToolTip("Remove from similar-images gallery")
            del_btn.setStyleSheet(_destructive_btn_style())
            del_btn.clicked.connect(lambda: self.delete_requested.emit(self._item_id))
            row.addWidget(del_btn)


class SimilarityPreviewCell(QFrame):
    delete_requested = pyqtSignal(int)
    find_requested = pyqtSignal(int)
    view_requested = pyqtSignal(str)
    more_requested = pyqtSignal(dict)

    def __init__(self, entry: dict[str, Any], *, editable: bool = True, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._entry = dict(entry)
        self._item_id = int(entry.get("item_id", 0) or 0)
        self._source_path = str(entry.get("source_path", "") or "")
        label_present, label_path = _label_status(self._source_path)
        self._entry["has_label"] = bool(entry.get("has_label", label_present))
        self._entry["label_path"] = str(entry.get("label_path") or label_path)
        self.setObjectName("gallerySheetCell")
        self.setStyleSheet(_sheet_cell_style())
        self.setFixedWidth(174)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        top = QHBoxLayout()
        batch = str(entry.get("batch_label", "") or "").strip() or "gallery"
        batch_label = QLabel(batch)
        batch_label.setToolTip(batch)
        batch_label.setStyleSheet(_pill_caption_style())
        top.addWidget(batch_label, stretch=1)
        label_badge = QLabel("TAG" if self._entry["has_label"] else "NO TAG")
        label_badge.setToolTip(
            f"Label file: {self._entry['label_path']}" if self._entry["has_label"] else "No label file found"
        )
        label_badge.setStyleSheet(_label_badge_style(present=bool(self._entry["has_label"])))
        top.addWidget(label_badge)
        root.addLayout(top)

        thumb_label = QLabel()
        thumb_label.setObjectName("galleryGridThumb")
        thumb_label.setFixedSize(158, 104)
        thumb_label.setScaledContents(True)
        thumb_label.setStyleSheet(_grid_thumb_style())
        pix = _load_thumb(self._source_path, str(entry.get("thumb_png_b64", "")), size=260)
        if not pix.isNull():
            thumb_label.setPixmap(pix)
        else:
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_label.setText("No image")
        root.addWidget(thumb_label)

        title = QLabel(str(entry.get("display_name", "") or Path(self._source_path).stem or "Image"))
        title.setToolTip(title.text())
        title.setWordWrap(False)
        title.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {text_hex()};")
        root.addWidget(title)

        source_name = Path(self._source_path).name if self._source_path else "Imported image"
        source = QLabel(source_name)
        source.setToolTip(self._source_path)
        source.setStyleSheet(_hint_style())
        root.addWidget(source)

        actions = QHBoxLayout()
        actions.setSpacing(4)
        for text, callback, tip in (
            ("More", lambda: self.more_requested.emit(dict(self._entry)), "Show provenance and label details"),
            ("Similar", lambda: self.find_requested.emit(self._item_id), "Find visually similar images"),
            ("View", lambda: self.view_requested.emit(self._source_path), "Open this image"),
        ):
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.setStyleSheet(_row_action_style())
            btn.clicked.connect(callback)
            actions.addWidget(btn)
        root.addLayout(actions)

        if editable:
            del_btn = QPushButton("Delete")
            del_btn.setToolTip("Remove from similar-images gallery")
            del_btn.setStyleSheet(_destructive_btn_style())
            del_btn.clicked.connect(lambda: self.delete_requested.emit(self._item_id))
            root.addWidget(del_btn)


class ProvenanceDialog(QDialog):
    def __init__(self, entry: dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gallery Provenance")
        self.resize(680, 460)
        self.setStyleSheet(_dialog_style())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel(str(entry.get("display_name", "") or Path(str(entry.get("source_path", ""))).stem or "Image"))
        title.setStyleSheet(f"font-size: 14px; font-weight: 800; color: {text_hex()};")
        layout.addWidget(title)

        body = QHBoxLayout()
        body.setSpacing(14)
        img = QLabel()
        img.setObjectName("galleryGridThumb")
        img.setFixedSize(220, 180)
        img.setScaledContents(True)
        img.setStyleSheet(_grid_thumb_style())
        pix = _load_thumb(str(entry.get("source_path", "")), str(entry.get("thumb_png_b64", "")), size=440)
        if not pix.isNull():
            img.setPixmap(pix)
        body.addWidget(img)

        info = QVBoxLayout()
        info.setSpacing(8)
        source_path = str(entry.get("source_path", "") or "")
        has_label = bool(entry.get("has_label", False))
        label_path = str(entry.get("label_path", "") or "")
        rows = [
            ("Gallery ID", str(entry.get("item_id", "") or "")),
            ("Batch", str(entry.get("batch_label", "") or "")),
            ("Source", source_path),
            ("Label", "present" if has_label else "missing"),
            ("Label path", label_path),
        ]
        for label, value in rows:
            name = QLabel(label.upper())
            name.setStyleSheet(_pill_caption_style())
            value_lbl = QLabel(value or "-")
            value_lbl.setWordWrap(True)
            value_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            value_lbl.setStyleSheet(_body_style())
            info.addWidget(name)
            info.addWidget(value_lbl)
        body.addLayout(info, stretch=1)
        layout.addLayout(body, stretch=1)

        close_btn = QPushButton("Close")
        close_btn.setMinimumWidth(88)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)


class ImageViewDialog(QDialog):
    def __init__(self, title: str, image_paths: list[str], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(780, 540)
        self.setStyleSheet(_dialog_style())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QLabel(f"{title}  ·  {len(image_paths)} image(s)")
        header.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {text_hex()}; padding-bottom: 4px;")
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        grid = QGridLayout(inner)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(12)
        cols = 5
        for idx, path in enumerate(image_paths):
            lbl = QLabel()
            lbl.setObjectName("galleryGridThumb")
            lbl.setFixedSize(120, 120)
            lbl.setScaledContents(True)
            lbl.setStyleSheet(_grid_thumb_style())
            pix = _load_thumb(path, size=240)
            if not pix.isNull():
                lbl.setPixmap(pix)
            grid.addWidget(lbl, idx // cols, idx % cols)
        scroll.setWidget(inner)
        layout.addWidget(scroll, stretch=1)

        close_btn = QPushButton("Close")
        close_btn.setMinimumWidth(88)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)


class SimilarityResultsDialog(QDialog):
    def __init__(self, source_path: str, results: list[dict[str, Any]], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Similar Images")
        self.resize(800, 560)
        self.setStyleSheet(_dialog_style())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        source_name = Path(source_path).name if source_path else "Selected image"
        header = QLabel(f"Similar to {source_name}")
        header.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {text_hex()};")
        layout.addWidget(header)

        if not results:
            empty = QLabel("No similar images found yet.")
            empty.setStyleSheet(f"font-size: 11px; color: {text_css(0.65)}; padding: 24px 0;")
            layout.addWidget(empty)
        else:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
            inner = QWidget()
            inner.setStyleSheet("background: transparent;")
            grid = QGridLayout(inner)
            grid.setContentsMargins(4, 4, 4, 4)
            grid.setSpacing(14)
            cols = 3
            for idx, result in enumerate(results):
                card = QFrame()
                card.setObjectName("galleryResultCard")
                card.setStyleSheet(_similarity_result_card_style())
                card_layout = QVBoxLayout(card)
                card_layout.setContentsMargins(12, 10, 12, 10)
                card_layout.setSpacing(8)
                img = QLabel()
                img.setObjectName("galleryGridThumb")
                img.setFixedSize(200, 150)
                img.setScaledContents(True)
                img.setStyleSheet(_grid_thumb_style())
                pix = _load_thumb(str(result.get("source_path", "")), size=400)
                if not pix.isNull():
                    img.setPixmap(pix)
                card_layout.addWidget(img)
                title = QLabel(str(result.get("display_name", "Image") or "Image"))
                tf = title.font()
                tf.setBold(True)
                title.setFont(tf)
                title.setStyleSheet(f"font-size: 11px; color: {text_hex()};")
                card_layout.addWidget(title)
                batch = str(result.get("batch_label", "") or "").strip()
                if batch:
                    batch_label = QLabel(batch)
                    batch_label.setStyleSheet(_hint_style())
                    card_layout.addWidget(batch_label)
                similarity = float(result.get("similarity", 0.0) or 0.0)
                score = QLabel(f"Match {int(round(similarity * 100))}%")
                score.setStyleSheet(f"font-size: 10px; color: {theme_hex('accent_dark')}; font-weight: 600;")
                card_layout.addWidget(score)
                grid.addWidget(card, idx // cols, idx % cols)
            scroll.setWidget(inner)
            layout.addWidget(scroll, stretch=1)

        close_btn = QPushButton("Close")
        close_btn.setMinimumWidth(88)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)


class GalleryPanel(QWidget):
    def __init__(
        self,
        send_message: Callable[[dict[str, Any]], None],
        get_gallery_images: Callable[[str], list[str]],
        read_only: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._send = send_message
        self._get_images = get_gallery_images
        self._read_only = bool(read_only)
        self._threshold = 42
        self._people: list[dict[str, Any]] = []
        self._similarity_items: list[dict[str, Any]] = []
        self._source_kind = "folder"
        self._mode = "face"
        self._similarity_enabled = False
        self._similarity_error = ""
        self._burst_identity = ""
        self._burst_group = "attendance"

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self._scroll.viewport().setStyleSheet("background: transparent;")
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        cl = QVBoxLayout(content)
        cl.setContentsMargins(14, 14, 14, 14)
        cl.setSpacing(14)

        intro = QFrame()
        intro.setObjectName("gallerySection")
        intro.setStyleSheet(_section_style())
        intro_layout = QVBoxLayout(intro)
        intro_layout.setContentsMargins(16, 14, 16, 14)
        intro_layout.setSpacing(8)
        intro_title = QLabel("Gallery")
        intro_title.setStyleSheet(_heading_style())
        intro_layout.addWidget(intro_title)
        intro_copy = QLabel(
            "Add photos once, then either recognize people or find visually similar images. "
            "The gallery stays local on this device."
        )
        intro_copy.setWordWrap(True)
        intro_copy.setStyleSheet(_body_style())
        intro_layout.addWidget(intro_copy)
        cl.addWidget(intro)

        self._build_add_media_section(cl)
        self._build_burst_review_section(cl)
        self._build_people_settings_section(cl)
        self._build_people_section(cl)
        self._build_similarity_section(cl)

        cl.addStretch(1)
        self._scroll.setWidget(content)
        root.addWidget(self._scroll, stretch=1)
        if self._read_only:
            self._add_btn.setEnabled(False)
            self._source_image_btn.setEnabled(False)
            self._source_folder_btn.setEnabled(False)
            self._mode_similarity_btn.setEnabled(False)
            self._mode_face_btn.setEnabled(False)
            self._name_edit.setEnabled(False)
            self._group_edit.setEnabled(False)
            self._auto_btn.setEnabled(False)
            self._rebuild_btn.setEnabled(False)
            self._thr_slider.setEnabled(False)
            self._burst_more_btn.setEnabled(False)
            self._add_copy.setText("Read-only mode: shared Insight gallery is visible for review only.")
        self._refresh_mode_ui()

    def _build_add_media_section(self, parent_layout: QVBoxLayout) -> None:
        add = QFrame()
        add.setObjectName("gallerySection")
        add.setStyleSheet(_section_style())
        layout = QVBoxLayout(add)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        title = QLabel("Add media")
        title.setStyleSheet(_heading_style())
        layout.addWidget(title)

        self._add_copy = QLabel("")
        self._add_copy.setWordWrap(True)
        self._add_copy.setStyleSheet(_hint_style())
        layout.addWidget(self._add_copy)

        source_row = QHBoxLayout()
        source_row.setSpacing(8)
        self._source_image_btn = QPushButton("Image")
        self._source_image_btn.setCheckable(True)
        self._source_image_btn.clicked.connect(lambda: self._set_source_kind("image"))
        self._source_folder_btn = QPushButton("Folder")
        self._source_folder_btn.setCheckable(True)
        self._source_folder_btn.clicked.connect(lambda: self._set_source_kind("folder"))
        source_row.addWidget(self._source_image_btn)
        source_row.addWidget(self._source_folder_btn)
        source_row.addStretch(1)
        layout.addLayout(source_row)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        self._mode_similarity_btn = QPushButton("Find Similar Images")
        self._mode_similarity_btn.setCheckable(True)
        self._mode_similarity_btn.clicked.connect(lambda: self._set_mode("similarity"))
        self._mode_face_btn = QPushButton("Recognize Faces")
        self._mode_face_btn.setCheckable(True)
        self._mode_face_btn.clicked.connect(lambda: self._set_mode("face"))
        mode_row.addWidget(self._mode_similarity_btn, stretch=1)
        mode_row.addWidget(self._mode_face_btn, stretch=1)
        layout.addLayout(mode_row)

        self._similarity_hint = QLabel("")
        self._similarity_hint.setWordWrap(True)
        self._similarity_hint.setStyleSheet(_hint_style())
        layout.addWidget(self._similarity_hint)

        self._face_fields = QWidget()
        face_layout = QVBoxLayout(self._face_fields)
        face_layout.setContentsMargins(0, 0, 0, 0)
        face_layout.setSpacing(6)

        name_row = QHBoxLayout()
        name_lbl = QLabel("Person")
        name_lbl.setStyleSheet(f"font-size: 10px; color: {text_css(0.62)}; min-width: 48px;")
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. John Doe")
        self._name_edit.setStyleSheet(_input_style())
        name_row.addWidget(name_lbl)
        name_row.addWidget(self._name_edit, stretch=1)
        face_layout.addLayout(name_row)

        group_row = QHBoxLayout()
        group_lbl = QLabel("Group")
        group_lbl.setStyleSheet(f"font-size: 10px; color: {text_css(0.62)}; min-width: 48px;")
        self._group_edit = QLineEdit()
        self._group_edit.setPlaceholderText("optional")
        self._group_edit.setStyleSheet(_input_style())
        group_row.addWidget(group_lbl)
        group_row.addWidget(self._group_edit, stretch=1)
        face_layout.addLayout(group_row)
        layout.addWidget(self._face_fields)

        self._add_btn = QPushButton("Add to gallery")
        self._add_btn.setMinimumHeight(34)
        self._add_btn.clicked.connect(self._on_add_media)
        layout.addWidget(self._add_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFixedHeight(16)
        self._progress_bar.setStyleSheet(_progress_style())
        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet(_hint_style())
        self._progress_bar.hide()
        self._progress_label.hide()
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._progress_label)
        parent_layout.addWidget(add)

    def _build_burst_review_section(self, parent_layout: QVBoxLayout) -> None:
        self._burst_review = QFrame()
        self._burst_review.setObjectName("gallerySection")
        self._burst_review.setStyleSheet(_section_style())
        bl = QVBoxLayout(self._burst_review)
        bl.setContentsMargins(16, 14, 16, 14)
        bl.setSpacing(10)
        title = QLabel("Quick capture review")
        title.setStyleSheet(_heading_style())
        bl.addWidget(title)
        self._burst_header = QLabel("Face samples are ready to review")
        self._burst_header.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {text_hex()};")
        bl.addWidget(self._burst_header)
        self._burst_copy = QLabel(
            "Face samples captured from the live view appear here so you can review them before adding more photos."
        )
        self._burst_copy.setWordWrap(True)
        self._burst_copy.setStyleSheet(_body_style())
        bl.addWidget(self._burst_copy)

        metrics = QGridLayout()
        metrics.setContentsMargins(0, 0, 0, 0)
        metrics.setHorizontalSpacing(8)
        metrics.setVerticalSpacing(8)
        self._burst_requested = self._make_metric_card("Requested")
        self._burst_captured = self._make_metric_card("Captured")
        self._burst_added = self._make_metric_card("Added")
        self._burst_total = self._make_metric_card("Total")
        metrics.addWidget(self._burst_requested[0], 0, 0)
        metrics.addWidget(self._burst_captured[0], 0, 1)
        metrics.addWidget(self._burst_added[0], 1, 0)
        metrics.addWidget(self._burst_total[0], 1, 1)
        bl.addLayout(metrics)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self._burst_view_btn = QPushButton("Review samples")
        self._burst_view_btn.setStyleSheet(_row_action_style())
        self._burst_view_btn.clicked.connect(self._review_burst_identity)
        self._burst_more_btn = QPushButton("Add more photos")
        self._burst_more_btn.setStyleSheet(_row_action_style())
        self._burst_more_btn.clicked.connect(self._add_more_for_burst_identity)
        actions.addWidget(self._burst_view_btn)
        actions.addWidget(self._burst_more_btn)
        actions.addStretch(1)
        bl.addLayout(actions)
        self._burst_review.hide()
        parent_layout.addWidget(self._burst_review)

    def _build_people_settings_section(self, parent_layout: QVBoxLayout) -> None:
        settings = QFrame()
        settings.setObjectName("gallerySection")
        settings.setStyleSheet(_section_style())
        sl = QVBoxLayout(settings)
        sl.setContentsMargins(16, 14, 16, 14)
        sl.setSpacing(8)
        settings_title = QLabel("Face matching")
        settings_title.setStyleSheet(_heading_style())
        sl.addWidget(settings_title)

        thr_row = QHBoxLayout()
        thr_lbl = QLabel("Match threshold")
        thr_lbl.setStyleSheet(f"font-size: 10px; color: {text_css(0.72)};")
        self._thr_slider = QSlider(Qt.Orientation.Horizontal)
        self._thr_slider.setRange(40, 98)
        self._thr_slider.setValue(self._threshold)
        self._thr_val = QLabel(f"{self._threshold}%")
        self._thr_val.setStyleSheet(f"font-size: 10px; color: {text_hex()}; min-width: 36px; font-weight: 600;")
        self._thr_slider.valueChanged.connect(self._on_threshold_changed)
        thr_row.addWidget(thr_lbl)
        thr_row.addWidget(self._thr_slider, stretch=1)
        thr_row.addWidget(self._thr_val)
        sl.addLayout(thr_row)

        hint = QLabel(
            "Lower values find more matches. Higher values are stricter. Use this only for face recognition."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(_hint_style())
        sl.addWidget(hint)

        auto_row = QHBoxLayout()
        self._auto_btn = QPushButton("Auto-Recognize Faces")
        self._auto_btn.setCheckable(True)
        self._auto_btn.setChecked(True)
        self._auto_btn.clicked.connect(lambda checked: self._send({"type": "set_recognition_auto", "enabled": checked}))
        self._rebuild_btn = QPushButton("Refresh Face Library")
        self._rebuild_btn.clicked.connect(lambda: self._send({"type": "rebuild_gallery_index"}))
        auto_row.addWidget(self._auto_btn)
        auto_row.addWidget(self._rebuild_btn)
        auto_row.addStretch(1)
        sl.addLayout(auto_row)
        parent_layout.addWidget(settings)

    def _build_people_section(self, parent_layout: QVBoxLayout) -> None:
        section = QFrame()
        section.setObjectName("gallerySection")
        section.setStyleSheet(_section_style())
        layout = QVBoxLayout(section)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        title = QLabel("People")
        title.setStyleSheet(_heading_style())
        self._people_stats = QLabel("No people added yet.")
        self._people_stats.setStyleSheet(_hint_style() + "padding: 2px 0;")
        layout.addWidget(title)
        layout.addWidget(self._people_stats)
        self._people_inner = QWidget()
        self._people_layout = QVBoxLayout(self._people_inner)
        self._people_layout.setContentsMargins(0, 0, 0, 0)
        self._people_layout.setSpacing(8)
        self._people_empty = QLabel("No people added yet.\nUse Add Media above and choose Recognize Faces.")
        self._people_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._people_empty.setWordWrap(True)
        self._people_empty.setStyleSheet(f"color: {text_css(0.48)}; font-size: 10px; padding: 20px;")
        self._people_layout.addWidget(self._people_empty)
        layout.addWidget(self._people_inner)
        parent_layout.addWidget(section)

    def _build_similarity_section(self, parent_layout: QVBoxLayout) -> None:
        section = QFrame()
        section.setObjectName("gallerySection")
        section.setStyleSheet(_section_style())
        layout = QVBoxLayout(section)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        title = QLabel("Similar images")
        title.setStyleSheet(_heading_style())
        self._similarity_stats = QLabel("No images added yet.")
        self._similarity_stats.setStyleSheet(_hint_style() + "padding: 2px 0;")
        layout.addWidget(title)
        layout.addWidget(self._similarity_stats)
        self._similarity_inner = QWidget()
        self._similarity_layout = QGridLayout(self._similarity_inner)
        self._similarity_layout.setContentsMargins(0, 0, 0, 0)
        self._similarity_layout.setHorizontalSpacing(8)
        self._similarity_layout.setVerticalSpacing(8)
        self._similarity_empty = QLabel("No similarity images yet.\nUse Add Media above and choose Find Similar Images.")
        self._similarity_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._similarity_empty.setWordWrap(True)
        self._similarity_empty.setStyleSheet(f"color: {text_css(0.48)}; font-size: 10px; padding: 20px;")
        self._similarity_layout.addWidget(self._similarity_empty, 0, 0)
        layout.addWidget(self._similarity_inner)
        parent_layout.addWidget(section)

    def _set_source_kind(self, source_kind: str) -> None:
        self._source_kind = source_kind
        self._refresh_mode_ui()

    def _set_mode(self, mode: str) -> None:
        if mode == "similarity" and not self._similarity_enabled:
            QMessageBox.information(
                self,
                "Find Similar Images Unavailable",
                self._similarity_error or f"Add a local similarity model to {MODELS_DIR} to enable this mode.",
            )
            return
        self._mode = mode
        self._refresh_mode_ui()

    def _refresh_mode_ui(self) -> None:
        self._source_image_btn.setChecked(self._source_kind == "image")
        self._source_folder_btn.setChecked(self._source_kind == "folder")
        self._mode_similarity_btn.setChecked(self._mode == "similarity")
        self._mode_face_btn.setChecked(self._mode == "face")
        self._mode_similarity_btn.setEnabled(self._similarity_enabled and not self._read_only)
        for button in (self._source_image_btn, self._source_folder_btn, self._mode_similarity_btn, self._mode_face_btn, self._auto_btn):
            button.setStyleSheet(_toggle_style(selected=button.isChecked()))
        self._face_fields.setVisible(self._mode == "face")
        noun = "image" if self._source_kind == "image" else "folder"
        if self._mode == "face":
            self._add_copy.setText(
                f"Choose a {noun}, add a person name, and we will build a face-matching library for that person."
            )
            self._add_btn.setText(f"Add {noun} for face recognition")
        else:
            self._add_copy.setText(
                f"Choose a {noun} and we will add whole images for visual similarity search."
            )
            self._add_btn.setText(f"Add {noun} to similar images")
        self._similarity_hint.setText(
            "" if self._similarity_enabled else (self._similarity_error or "Add a local similarity model to enable this mode.")
        )

    def _on_add_media(self) -> None:
        if self._read_only:
            QMessageBox.information(self, "Read-Only Gallery", "Gallery mutations are disabled in CV Ops phase 1.")
            return
        if self._mode == "face" and not self._name_edit.text().strip():
            QMessageBox.warning(self, "Person Required", "Enter a person name before adding photos.")
            return
        if self._mode == "similarity" and not self._similarity_enabled:
            QMessageBox.information(
                self,
                "Find Similar Images Unavailable",
                self._similarity_error or f"Add a local similarity model to {MODELS_DIR} to enable this mode.",
            )
            return
        if self._source_kind == "folder":
            path = QFileDialog.getExistingDirectory(self, "Choose a folder of photos")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Choose a photo", "", _FILE_FILTER)
        if not path:
            return
        payload = {
            "type": "ingest_gallery_media",
            "mode": self._mode,
            "source_kind": self._source_kind,
            "path": path,
        }
        if self._mode == "face":
            payload["identity"] = self._name_edit.text().strip()
            payload["group"] = self._group_edit.text().strip()
        self._send(payload)
        if self._mode == "face" and self._source_kind == "folder":
            self._name_edit.clear()

    def _on_threshold_changed(self, value: int) -> None:
        self._threshold = value
        self._thr_val.setText(f"{value}%")
        if self._read_only:
            return
        self._send({"type": "set_recognition_threshold", "threshold": value / 100.0})

    def _on_delete_person(self, name: str) -> None:
        if self._read_only:
            return
        reply = QMessageBox.question(
            self,
            "Delete Person",
            f"Delete all photos and face matches for '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._send({"type": "delete_gallery_identity", "identity": name})

    def _on_rename_person(self, old_name: str, new_name: str) -> None:
        if self._read_only:
            return
        self._send({"type": "rename_gallery_identity", "old_name": old_name, "new_name": new_name})

    def _on_view_person(self, name: str) -> None:
        dlg = ImageViewDialog(f"People: {name}", self._get_images(name), self)
        dlg.exec()

    def _on_view_similarity_image(self, path: str) -> None:
        dlg = ImageViewDialog("Similar Images", [path] if path else [], self)
        dlg.exec()

    def _on_more_similarity_image(self, entry: dict[str, Any]) -> None:
        dlg = ProvenanceDialog(entry, self)
        dlg.exec()

    def _on_delete_similarity_item(self, item_id: int) -> None:
        if self._read_only:
            return
        reply = QMessageBox.question(
            self,
            "Delete Image",
            "Delete this image from Similar Images?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._send({"type": "delete_similarity_item", "item_id": item_id})

    def apply_gallery_state(self, payload: dict[str, Any]) -> None:
        self._people = list(payload.get("people", payload.get("identities", [])))
        self._similarity_items = list(payload.get("similarity_items", []))
        self._similarity_enabled = bool(payload.get("similarity_enabled", False))
        self._similarity_error = str(payload.get("similarity_error", "") or "")
        self._people_stats.setText(
            f"{int(payload.get('identity_count', 0) or 0)} people  ·  {int(payload.get('image_count', 0) or 0)} face photo(s)"
        )
        self._similarity_stats.setText(
            f"{int(payload.get('similarity_item_count', len(self._similarity_items)) or 0)} image(s) ready for matching"
        )
        self._rebuild_people()
        self._rebuild_similarity()
        self._refresh_mode_ui()
        if not self._burst_identity:
            return
        self._update_burst_total()

    def apply_ingest_progress(self, payload: dict[str, Any]) -> None:
        current = int(payload.get("current", 0) or 0)
        total = int(payload.get("total", 1) or 1)
        name = str(payload.get("file", "") or "")
        mode = str(payload.get("mode", "face") or "face")
        pct = int(current / total * 100) if total > 0 else 0
        label = "Finding faces" if mode == "face" else "Adding images"
        self._progress_bar.setValue(pct)
        self._progress_bar.setFormat(f"{label}: {current}/{total}")
        self._progress_label.setText(name)
        self._progress_bar.show()
        self._progress_label.show()

    def apply_ingest_result(self, payload: dict[str, Any]) -> None:
        self._progress_bar.hide()
        self._progress_label.hide()
        added = int(payload.get("added", 0) or 0)
        errors = [str(error) for error in payload.get("errors", [])[:8]]
        mode = str(payload.get("mode", "face") or "face")
        if not errors:
            return
        title = "Add Photos Warning" if mode == "face" else "Add Images Warning"
        QMessageBox.warning(
            self,
            title,
            f"{added} item(s) added.\n\n" + "\n".join(errors),
        )

    def apply_similarity_search_result(self, payload: dict[str, Any]) -> None:
        dlg = SimilarityResultsDialog(
            str(payload.get("source_path", "") or ""),
            list(payload.get("results", [])),
            self,
        )
        dlg.exec()

    def apply_burst_enroll_result(self, payload: dict[str, Any]) -> None:
        identity = str(payload.get("identity", "") or "").strip()
        if not identity:
            return
        requested = int(payload.get("requested", 0) or 0)
        captured = int(payload.get("captured", 0) or 0)
        added = int(payload.get("added", 0) or 0)
        duration_sec = float(payload.get("duration_sec", 0.0) or 0.0)
        group = str(payload.get("group", "") or "").strip() or "attendance"
        self._burst_identity = identity
        self._burst_group = group
        self._burst_header.setText(f"{identity} is ready to review")
        if added > 0:
            self._burst_copy.setText(
                f"{added} usable face photo(s) were added in {duration_sec:.1f}s. Review them or add more photos from disk."
            )
        else:
            self._burst_copy.setText(
                "The quick capture finished, but no usable face photos were added. Try clearer photos or add them from disk."
            )
        self._burst_requested[1].setText(str(requested))
        self._burst_captured[1].setText(str(captured))
        self._burst_added[1].setText(str(added))
        self._update_burst_total()
        self._burst_review.show()
        self._name_edit.setText(identity)
        if not self._group_edit.text().strip():
            self._group_edit.setText(group)
        self._scroll.verticalScrollBar().setValue(0)

    def _make_metric_card(self, label: str) -> tuple[QFrame, QLabel]:
        card = QFrame()
        card.setObjectName("galleryMetricCard")
        card.setStyleSheet(_metric_card_style())
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        heading = QLabel(label.upper())
        heading.setStyleSheet(
            f"font-size: 9px; color: {text_css(0.55)}; letter-spacing: 0.55px; font-weight: 600; text-transform: uppercase;"
        )
        value = QLabel("0")
        value.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {text_hex()};")
        layout.addWidget(heading)
        layout.addWidget(value)
        return card, value

    def _sample_count_for_identity(self, identity: str) -> int:
        for entry in self._people:
            if str(entry.get("name", "")).strip() == identity:
                return int(entry.get("embedding_count", 0) or 0)
        return 0

    def _update_burst_total(self) -> None:
        total = self._sample_count_for_identity(self._burst_identity)
        self._burst_total[1].setText(str(total))
        self._burst_view_btn.setEnabled(total > 0)

    def _review_burst_identity(self) -> None:
        if self._burst_identity:
            self._on_view_person(self._burst_identity)

    def _add_more_for_burst_identity(self) -> None:
        if self._read_only:
            return
        if not self._burst_identity:
            return
        self._mode = "face"
        self._source_kind = "folder"
        self._name_edit.setText(self._burst_identity)
        if not self._group_edit.text().strip():
            self._group_edit.setText(self._burst_group)
        self._refresh_mode_ui()
        self._on_add_media()

    def _clear_layout_rows(self, layout, empty_label: QLabel) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget and widget is not empty_label:
                widget.deleteLater()

    def _rebuild_people(self) -> None:
        self._clear_layout_rows(self._people_layout, self._people_empty)
        if not self._people:
            self._people_layout.addWidget(self._people_empty)
            self._people_empty.show()
            return
        self._people_empty.hide()
        for entry in self._people:
            row = PersonRow(entry, editable=not self._read_only)
            if not self._read_only:
                row.delete_requested.connect(self._on_delete_person)
                row.rename_requested.connect(self._on_rename_person)
            row.view_requested.connect(self._on_view_person)
            self._people_layout.addWidget(row)

    def _rebuild_similarity(self) -> None:
        self._clear_layout_rows(self._similarity_layout, self._similarity_empty)
        if not self._similarity_items:
            self._similarity_layout.addWidget(self._similarity_empty, 0, 0)
            self._similarity_empty.show()
            return
        self._similarity_empty.hide()
        cols = 5
        for idx, entry in enumerate(self._similarity_items):
            row = SimilarityPreviewCell(entry, editable=not self._read_only)
            if not self._read_only:
                row.delete_requested.connect(self._on_delete_similarity_item)
            row.find_requested.connect(lambda item_id: self._send({"type": "find_similar_gallery_item", "item_id": item_id}))
            row.view_requested.connect(self._on_view_similarity_image)
            row.more_requested.connect(self._on_more_similarity_image)
            self._similarity_layout.addWidget(row, idx // cols, idx % cols)
        self._similarity_layout.setColumnStretch(cols, 1)


class DatabaseGalleryPanel(QWidget):
    """Database-backed replacement for the legacy Gallery tab interior."""

    def __init__(
        self,
        *,
        send_message: Callable[[dict[str, Any]], None],
        get_gallery_images: Callable[[str], list[str]],
        read_only: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._send = send_message
        self._get_gallery_images = get_gallery_images
        self._read_only = read_only
        self._base_url = str(CVOPS_BASE_URL).rstrip("/")
        self._datasets: list[str] = []
        self._categories: dict[str, str] = {}
        self._payload: dict[str, Any] = {}
        self._images: list[dict[str, Any]] = []
        self._folders: list[dict[str, Any]] = []
        self._selected_dataset = ""
        self._selected_folder = ""
        self._suppress_events = False
        self._preview_limit = 72

        self._build_ui()
        QTimer.singleShot(0, self.refresh)

    def _build_ui(self) -> None:
        self.setObjectName("databaseGalleryRoot")
        self.setStyleSheet(self._stylesheet())

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        header = QFrame()
        header.setObjectName("databaseGalleryPanel")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 10, 12, 10)
        header_layout.setSpacing(10)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("Database Gallery")
        title.setObjectName("databaseGalleryTitle")
        subtitle = QLabel("Browse the same folders used by CV Ops add-to-database.")
        subtitle.setObjectName("databaseGalleryHint")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box, 1)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setObjectName("databaseGalleryButton")
        self._refresh_btn.clicked.connect(self.refresh)
        header_layout.addWidget(self._refresh_btn)
        root.addWidget(header)

        controls = QFrame()
        controls.setObjectName("databaseGalleryPanel")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._dataset_combo = QComboBox()
        self._dataset_combo.setEditable(True)
        self._dataset_combo.setMinimumWidth(190)
        self._dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        row.addWidget(self._field_box("Database", self._dataset_combo), 1)

        self._folder_combo = QComboBox()
        self._folder_combo.setEditable(True)
        self._folder_combo.setMinimumWidth(230)
        self._folder_combo.currentTextChanged.connect(self._on_folder_changed)
        row.addWidget(self._field_box("Target folder", self._folder_combo), 1)

        self._new_folder_edit = QLineEdit()
        self._new_folder_edit.setPlaceholderText("New folder path")
        self._new_folder_btn = QPushButton("Use Folder")
        self._new_folder_btn.setObjectName("databaseGalleryButton")
        self._new_folder_btn.clicked.connect(self._use_new_folder)
        new_folder_box = QFrame()
        new_folder_layout = QVBoxLayout(new_folder_box)
        new_folder_layout.setContentsMargins(0, 0, 0, 0)
        new_folder_layout.setSpacing(4)
        label = QLabel("Make or select")
        label.setObjectName("databaseGalleryFieldLabel")
        new_folder_layout.addWidget(label)
        inner = QHBoxLayout()
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(6)
        inner.addWidget(self._new_folder_edit, 1)
        inner.addWidget(self._new_folder_btn)
        new_folder_layout.addLayout(inner)
        row.addWidget(new_folder_box, 1)
        controls_layout.addLayout(row)

        self._status = QLabel("Loading database folders...")
        self._status.setObjectName("databaseGalleryHint")
        self._status.setWordWrap(True)
        controls_layout.addWidget(self._status)
        root.addWidget(controls)

        body = QHBoxLayout()
        body.setSpacing(10)

        folder_panel = QFrame()
        folder_panel.setObjectName("databaseGalleryPanel")
        folder_layout = QVBoxLayout(folder_panel)
        folder_layout.setContentsMargins(12, 12, 12, 12)
        folder_layout.setSpacing(8)
        folder_header = QLabel("Folders")
        folder_header.setObjectName("databaseGallerySectionTitle")
        folder_layout.addWidget(folder_header)
        self._folder_scroll = QScrollArea()
        self._folder_scroll.setWidgetResizable(True)
        self._folder_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._folder_holder = QWidget()
        self._folder_layout = QVBoxLayout(self._folder_holder)
        self._folder_layout.setContentsMargins(0, 0, 0, 0)
        self._folder_layout.setSpacing(6)
        self._folder_scroll.setWidget(self._folder_holder)
        folder_layout.addWidget(self._folder_scroll, 1)
        body.addWidget(folder_panel, 1)

        content_panel = QFrame()
        content_panel.setObjectName("databaseGalleryPanel")
        content_layout = QVBoxLayout(content_panel)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(8)
        self._content_title = QLabel("Contents")
        self._content_title.setObjectName("databaseGallerySectionTitle")
        content_layout.addWidget(self._content_title)
        self._content_scroll = QScrollArea()
        self._content_scroll.setWidgetResizable(True)
        self._content_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._content_holder = QWidget()
        self._content_grid = QGridLayout(self._content_holder)
        self._content_grid.setContentsMargins(0, 0, 0, 0)
        self._content_grid.setHorizontalSpacing(8)
        self._content_grid.setVerticalSpacing(8)
        self._content_scroll.setWidget(self._content_holder)
        content_layout.addWidget(self._content_scroll, 1)
        body.addWidget(content_panel, 3)
        root.addLayout(body, 1)

        self._legacy_status = QLabel("")
        self._legacy_status.setObjectName("databaseGalleryHint")
        self._legacy_status.setWordWrap(True)
        root.addWidget(self._legacy_status)

    def _field_box(self, label_text: str, widget: QWidget) -> QFrame:
        box = QFrame()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        label = QLabel(label_text)
        label.setObjectName("databaseGalleryFieldLabel")
        layout.addWidget(label)
        layout.addWidget(widget)
        return box

    def _stylesheet(self) -> str:
        panel_border = theme_rgba("accent_dark", 0.20 if _gallery_dark() else 0.24)
        panel_bg = theme_rgba("panel", 0.30 if _gallery_dark() else 0.20)
        input_bg = theme_rgba("panel", 0.62 if _gallery_dark() else 0.78)
        button_bg = theme_rgba("strip_soft", 0.16)
        button_border = theme_rgba("strip_soft", 0.34)
        return f"""
            QWidget#databaseGalleryRoot {{
                background: transparent;
            }}
            QFrame#databaseGalleryPanel {{
                background: {panel_bg};
                border: 1px solid {panel_border};
                border-radius: 6px;
            }}
            QLabel#databaseGalleryTitle {{
                color: {text_hex()};
                font-size: 18px;
                font-weight: 800;
                letter-spacing: 0px;
            }}
            QLabel#databaseGallerySectionTitle {{
                color: {text_hex()};
                font-size: 12px;
                font-weight: 800;
                letter-spacing: 0px;
            }}
            QLabel#databaseGalleryFieldLabel {{
                color: {text_css(0.58)};
                font-size: 9px;
                font-weight: 700;
                letter-spacing: 0.4px;
                text-transform: uppercase;
            }}
            QLabel#databaseGalleryHint {{
                color: {text_css(0.64)};
                font-size: 10px;
                line-height: 1.35;
            }}
            QComboBox, QLineEdit {{
                background: {input_bg};
                color: {text_hex()};
                border: 1px solid {theme_rgba("accent_dark", 0.22)};
                border-radius: 5px;
                padding: 6px 8px;
                font-size: 10px;
                min-height: 20px;
            }}
            QComboBox QAbstractItemView {{
                background: {theme_hex("panel")};
                color: {text_hex()};
                selection-background-color: {theme_rgba("strip_soft", 0.28)};
                border: 1px solid {theme_rgba("accent_dark", 0.24)};
            }}
            QPushButton#databaseGalleryButton, QPushButton#databaseGalleryFolder {{
                background: {button_bg};
                color: {text_hex()};
                border: 1px solid {button_border};
                border-radius: 5px;
                padding: 6px 10px;
                font-size: 10px;
                font-weight: 700;
                text-align: left;
            }}
            QPushButton#databaseGalleryFolder:checked {{
                background: {theme_rgba("strip_soft", 0.34)};
                border: 1px solid {theme_rgba("strip_soft", 0.58)};
            }}
            QFrame#databaseGalleryImageCell {{
                background: {theme_rgba("panel", 0.42)};
                border: 1px solid {theme_rgba("accent_dark", 0.16)};
                border-radius: 5px;
            }}
        """

    def _http_json(self, path: str, *, timeout: float = 2.5) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
        payload = json.loads(raw.decode("utf-8") or "{}")
        return payload if isinstance(payload, dict) else {}

    def refresh(self) -> None:
        try:
            payload = self._http_json("/database")
        except Exception as exc:
            self._datasets = []
            self._categories = {}
            self._payload = {}
            self._images = []
            self._folders = []
            self._set_status(f"CV Ops database is unavailable at {self._base_url}: {exc}")
            self._populate_dataset_combo([])
            self._populate_folder_combo([])
            self._render_folders()
            self._render_contents()
            return

        categories = payload.get("categories") if isinstance(payload.get("categories"), dict) else {}
        datasets = [str(name) for name in payload.get("datasets", []) if str(name).strip()]
        image_datasets = [
            name for name in datasets if str(categories.get(name, "image") or "image").lower() == "image"
        ]
        self._datasets = image_datasets or datasets
        self._categories = {str(k): str(v) for k, v in categories.items()}
        current = self._dataset_combo.currentText().strip() or self._selected_dataset
        self._populate_dataset_combo(self._datasets, preferred=current)
        self._selected_dataset = self._dataset_combo.currentText().strip()
        self._load_selected_dataset()

    def _populate_dataset_combo(self, names: list[str], *, preferred: str = "") -> None:
        self._suppress_events = True
        self._dataset_combo.blockSignals(True)
        self._dataset_combo.clear()
        for name in names:
            self._dataset_combo.addItem(name)
        if preferred and preferred not in names:
            self._dataset_combo.addItem(preferred)
        if preferred:
            idx = self._dataset_combo.findText(preferred)
            if idx >= 0:
                self._dataset_combo.setCurrentIndex(idx)
            else:
                self._dataset_combo.setEditText(preferred)
        elif names:
            self._dataset_combo.setCurrentIndex(0)
        self._dataset_combo.blockSignals(False)
        self._suppress_events = False

    def _on_dataset_changed(self, text: str) -> None:
        if self._suppress_events:
            return
        self._selected_dataset = text.strip()
        self._load_selected_dataset()

    def _load_selected_dataset(self) -> None:
        slug = self._selected_dataset.strip()
        if not slug:
            self._payload = {}
            self._images = []
            self._folders = []
            self._set_status("Choose a database folder or type a new database name.")
            self._populate_folder_combo([])
            self._render_folders()
            self._render_contents()
            return

        encoded = urllib.parse.quote(slug, safe="")
        try:
            payload = self._http_json(f"/database/{encoded}", timeout=4.0)
        except urllib.error.HTTPError as exc:
            self._payload = {"slug": slug, "images": [], "folders": []}
            self._images = []
            self._folders = []
            self._set_status(f"{slug} is ready as a new database name. It will exist after the first add.")
            self._populate_folder_combo([])
            self._render_folders()
            self._render_contents()
            return
        except Exception as exc:
            self._payload = {}
            self._images = []
            self._folders = []
            self._set_status(f"Could not load {slug}: {exc}")
            self._populate_folder_combo([])
            self._render_folders()
            self._render_contents()
            return

        self._payload = payload
        self._images = [item for item in payload.get("images", []) if isinstance(item, dict)]
        self._folders = [item for item in payload.get("folders", []) if isinstance(item, dict)]
        self._populate_folder_combo(self._folder_options())
        self._render_folders()
        self._render_contents()
        folder_count = len(self._folders)
        image_count = len(self._images)
        self._set_status(f"{slug}: {folder_count} folder(s), {image_count} image file(s).")

    def _populate_folder_combo(self, folders: list[str]) -> None:
        previous = self._selected_folder
        self._suppress_events = True
        self._folder_combo.blockSignals(True)
        self._folder_combo.clear()
        self._folder_combo.addItem("Root")
        for folder in folders:
            clean = folder.strip("/")
            if clean:
                self._folder_combo.addItem(clean)
        if previous:
            idx = self._folder_combo.findText(previous)
            if idx >= 0:
                self._folder_combo.setCurrentIndex(idx)
            else:
                self._folder_combo.addItem(previous)
                self._folder_combo.setCurrentText(previous)
        else:
            self._folder_combo.setCurrentIndex(0)
        self._folder_combo.blockSignals(False)
        self._suppress_events = False

    def _on_folder_changed(self, text: str) -> None:
        if self._suppress_events:
            return
        self._selected_folder = self._normalize_folder_text(text)
        self._render_folders()
        self._render_contents()

    def _use_new_folder(self) -> None:
        folder = self._normalize_folder_text(self._new_folder_edit.text())
        if not folder:
            return
        if self._folder_combo.findText(folder) < 0:
            self._folder_combo.addItem(folder)
        self._folder_combo.setCurrentText(folder)
        self._selected_folder = folder
        self._new_folder_edit.clear()
        self._render_folders()
        self._render_contents()

    def _normalize_folder_text(self, text: str) -> str:
        clean = str(text or "").strip().strip("/")
        return "" if clean.lower() in {"", "root", "."} else clean

    def _folder_options(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for folder in self._folders:
            path = str(folder.get("path") or "").strip().strip("/")
            if path and path.lower() not in seen:
                seen.add(path.lower())
                out.append(path)
        for item in self._images:
            folder = self._image_target_folder(item)
            while folder:
                key = folder.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(folder)
                folder = str(Path(folder).parent).replace("\\", "/")
                if folder in {".", ""}:
                    break
        return sorted(out, key=lambda value: value.lower())

    def _image_rel_path(self, item: dict[str, Any]) -> str:
        for key in ("relative_path", "display_name", "name", "path"):
            value = str(item.get(key) or "").strip()
            if value:
                return value.replace("\\", "/")
        return ""

    def _image_target_folder(self, item: dict[str, Any]) -> str:
        rel = str(item.get("relative_path") or item.get("display_name") or "").replace("\\", "/").strip("/")
        if "/" not in rel:
            return ""
        return rel.rsplit("/", 1)[0]

    def _folder_items(self, folder: str) -> list[dict[str, Any]]:
        folder_l = folder.strip("/").lower()
        items: list[dict[str, Any]] = []
        for item in self._images:
            item_folder = self._image_target_folder(item).strip("/").lower()
            if folder_l:
                if item_folder != folder_l and not item_folder.startswith(f"{folder_l}/"):
                    continue
            elif item_folder:
                continue
            items.append(item)
        return items

    def _render_folders(self) -> None:
        self._clear_layout(self._folder_layout)
        root_items = self._folder_items("")
        root_btn = self._folder_button("Root", len(root_items), "")
        self._folder_layout.addWidget(root_btn)
        for folder in self._folder_options():
            count = len(self._folder_items(folder))
            self._folder_layout.addWidget(self._folder_button(folder, count, folder))
        self._folder_layout.addStretch(1)

    def _folder_button(self, label: str, count: int, folder: str) -> QPushButton:
        text = f"{label}\n{count} image(s)"
        btn = QPushButton(text)
        btn.setObjectName("databaseGalleryFolder")
        btn.setCheckable(True)
        btn.setChecked(self._selected_folder == folder)
        btn.clicked.connect(lambda _checked=False, value=folder: self._select_folder(value))
        return btn

    def _select_folder(self, folder: str) -> None:
        self._selected_folder = folder
        display = folder or "Root"
        if self._folder_combo.findText(display) < 0 and folder:
            self._folder_combo.addItem(folder)
        self._folder_combo.setCurrentText(display)
        self._render_folders()
        self._render_contents()

    def _render_contents(self) -> None:
        self._clear_layout(self._content_grid)
        folder = self._selected_folder
        items = self._folder_items(folder)
        display = folder or "Root"
        total = len(items)
        shown = items[: self._preview_limit]
        self._content_title.setText(f"{display} contents · showing {len(shown)} of {total}")
        if not shown:
            empty = QLabel("No images in this folder yet.")
            empty.setObjectName("databaseGalleryHint")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._content_grid.addWidget(empty, 0, 0)
            return
        cols = 4
        for idx, item in enumerate(shown):
            self._content_grid.addWidget(self._image_cell(item), idx // cols, idx % cols)
        self._content_grid.setColumnStretch(cols, 1)

    def _image_cell(self, item: dict[str, Any]) -> QFrame:
        cell = QFrame()
        cell.setObjectName("databaseGalleryImageCell")
        layout = QVBoxLayout(cell)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        image = QLabel()
        image.setFixedSize(150, 110)
        image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = self._pixmap_for_item(item)
        if not pixmap.isNull():
            image.setPixmap(
                pixmap.scaled(
                    image.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            image.setText("Preview unavailable")
            image.setObjectName("databaseGalleryHint")
        layout.addWidget(image, alignment=Qt.AlignmentFlag.AlignCenter)
        caption = QLabel(str(item.get("display_name") or item.get("name") or Path(str(item.get("path") or "")).name))
        caption.setObjectName("databaseGalleryHint")
        caption.setWordWrap(True)
        layout.addWidget(caption)
        view_btn = QPushButton("View")
        view_btn.setObjectName("databaseGalleryButton")
        view_btn.clicked.connect(lambda _checked=False, entry=item: self._open_image(entry))
        layout.addWidget(view_btn)
        return cell

    def _pixmap_for_item(self, item: dict[str, Any]) -> QPixmap:
        pixmap = QPixmap()
        path = Path(str(item.get("path") or ""))
        if path.exists():
            reader = QImageReader(str(path))
            reader.setAutoTransform(True)
            reader.setScaledSize(QSize(160, 160))
            image = reader.read()
            if not image.isNull():
                return QPixmap.fromImage(image)
        slug = self._selected_dataset.strip()
        rel = self._image_rel_path(item)
        if not slug or not rel:
            return pixmap
        try:
            payload = self._http_json(
                f"/database/{urllib.parse.quote(slug, safe='')}/thumb/{urllib.parse.quote(rel, safe='')}",
                timeout=2.0,
            )
            thumb = str(payload.get("thumb_b64") or "")
            if thumb:
                pixmap.loadFromData(base64.b64decode(thumb))
        except Exception:
            pass
        return pixmap

    def _open_image(self, item: dict[str, Any]) -> None:
        path = str(item.get("path") or "").strip()
        if not path:
            QMessageBox.information(self, "Image Preview", "This item does not expose a local image path.")
            return
        if not Path(path).exists():
            QMessageBox.warning(self, "Image Preview", f"Image path does not exist:\n{path}")
            return
        dlg = ImageViewDialog(Path(path).name, [path], self)
        dlg.exec()

    def _clear_layout(self, layout: QGridLayout | QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)  # type: ignore[arg-type]

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def apply_gallery_state(self, payload: dict[str, Any]) -> None:
        people = payload.get("people", []) if isinstance(payload, dict) else []
        similarity = payload.get("similarity_items", []) if isinstance(payload, dict) else []
        self._legacy_status.setText(
            f"Recognition gallery stream: {len(people or [])} people, {len(similarity or [])} similarity item(s)."
        )

    def apply_ingest_progress(self, payload: dict[str, Any]) -> None:
        identity = str(payload.get("identity") or payload.get("name") or "").strip()
        current = int(payload.get("processed", payload.get("current", 0)) or 0)
        total = int(payload.get("total", 0) or 0)
        detail = f" for {identity}" if identity else ""
        if total > 0:
            self._legacy_status.setText(f"Legacy gallery ingest{detail}: {current}/{total}.")
        else:
            self._legacy_status.setText(f"Legacy gallery ingest{detail} is running.")

    def apply_ingest_result(self, payload: dict[str, Any]) -> None:
        added = int(payload.get("added", 0) or 0)
        errors = [str(error) for error in payload.get("errors", [])[:4]]
        message = f"Legacy gallery ingest added {added} item(s)."
        if errors:
            message += " " + " ".join(errors)
        self._legacy_status.setText(message)

    def apply_similarity_search_result(self, payload: dict[str, Any]) -> None:
        dlg = SimilarityResultsDialog(
            str(payload.get("source_path", "") or ""),
            list(payload.get("results", [])),
            self,
        )
        dlg.exec()

    def apply_burst_enroll_result(self, payload: dict[str, Any]) -> None:
        identity = str(payload.get("identity", "") or "").strip()
        added = int(payload.get("added", 0) or 0)
        if identity:
            self._legacy_status.setText(f"Quick capture for {identity}: {added} face photo(s) added.")


GalleryPanel = DatabaseGalleryPanel
