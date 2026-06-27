"""Choose destination library dataset when promoting a QA detection group."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


class ScrapePromoteDatasetDialog(QDialog):
    """Destination + target class; detection group is chosen from the parent menu."""

    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        group_summary: str,
        match_count: int,
        source_slug: str,
        existing_slugs: list[str],
        default_promoted_label: str,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Promote to dataset")
        self.resize(480, 360)
        self._target_slug = ""
        self._is_new = True
        self._promoted_class = ""

        vl = QVBoxLayout(self)
        intro = QLabel(
            f"Copy {match_count} staged image(s) from `{source_slug}` matching:\n"
            f"{group_summary}\n\n"
            "Images go to the target dataset's staged/ folder. "
            "Every hand-drawn box on those images is reassigned to the class you set below "
            "(bounding boxes stay the same — only the YOLO class index changes)."
        )
        intro.setWordWrap(True)
        vl.addWidget(intro)

        form = QFormLayout()
        self._final_class_edit = QLineEdit(default_promoted_label)
        self._final_class_edit.setPlaceholderText("e.g. person, pedestrian, face")
        self._final_class_edit.setToolTip(
            "Defaults from the detection group when it is a single label. "
            "Edit to match your target dataset taxonomy without relabeling each box."
        )
        form.addRow("Class in target dataset:", self._final_class_edit)

        dest = QWidget()
        dv = QVBoxLayout(dest)
        dv.setContentsMargins(0, 0, 0, 0)
        self._bg = QButtonGroup(self)
        self._radio_new = QRadioButton("New dataset slug")
        self._radio_exist = QRadioButton("Existing dataset")
        self._radio_new.setChecked(True)
        self._bg.addButton(self._radio_new)
        self._bg.addButton(self._radio_exist)
        dv.addWidget(self._radio_new)
        self._new_slug = QLineEdit()
        self._new_slug.setPlaceholderText("e.g. scrap_faces_subset")
        dv.addWidget(self._new_slug)
        dv.addWidget(self._radio_exist)
        self._exist_combo = QComboBox()
        self._exist_combo.setEditable(False)
        for s in existing_slugs:
            self._exist_combo.addItem(s)
        dv.addWidget(self._exist_combo)
        form.addRow("Destination:", dest)
        vl.addLayout(form)

        self._radio_new.toggled.connect(self._sync_dest_enabled)
        self._radio_exist.toggled.connect(self._sync_dest_enabled)
        self._sync_dest_enabled()

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        vl.addWidget(bb)

    def _sync_dest_enabled(self) -> None:
        new = self._radio_new.isChecked()
        self._new_slug.setEnabled(new)
        self._exist_combo.setEnabled(not new)

    def _on_accept(self) -> None:
        cls_name = str(self._final_class_edit.text() or "").strip()
        if not cls_name:
            QMessageBox.warning(
                self,
                "Promote to dataset",
                "Enter the class name to use for promoted boxes in the target dataset.",
            )
            return
        if self._radio_new.isChecked():
            slug = self._new_slug.text().strip()
            if not slug:
                QMessageBox.warning(self, "Promote to dataset", "Enter a dataset slug.")
                return
        else:
            if self._exist_combo.count() == 0:
                QMessageBox.warning(
                    self,
                    "Promote to dataset",
                    "No other library datasets exist yet. Create a new slug first.",
                )
                return
            slug = self._exist_combo.currentText()
        self._target_slug = slug
        self._is_new = self._radio_new.isChecked()
        self._promoted_class = cls_name
        self.accept()

    def result(self) -> tuple[str, bool, str]:
        return self._target_slug, self._is_new, self._promoted_class
