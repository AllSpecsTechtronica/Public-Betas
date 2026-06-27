from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


@dataclass
class ModelFeedbackResult:
    issue_type: str
    severity: str
    notes: str
    recommendation: str


class ModelFeedbackDialog(QDialog):
    """Collects Console-level feedback for a run or weights artifact."""

    def __init__(self, *, payload: dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._payload = dict(payload or {})
        self.result_payload: Optional[ModelFeedbackResult] = None

        weights_path = str(self._payload.get("weights_path") or "").strip()
        weight_name = Path(weights_path).name if weights_path else ""
        self.setWindowTitle(f"Flag model feedback - {weight_name or 'Console run'}")
        self.resize(620, 460)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        title = QLabel("<b>Run / Model Feedback</b>")
        title.setTextFormat(Qt.TextFormat.RichText)
        outer.addWidget(title)

        context_form = QFormLayout()
        context_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        context_form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        for label, value in self._context_rows():
            context_form.addRow(label, self._context_label(value))
        outer.addLayout(context_form)

        edit_form = QFormLayout()
        edit_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._issue = QComboBox()
        for item in (
            "False positives",
            "False negatives",
            "Wrong labels",
            "Poor confidence / calibration",
            "Regression",
            "Bad run / weights",
            "Other",
        ):
            self._issue.addItem(item)
        edit_form.addRow("Issue", self._issue)

        self._severity = QComboBox()
        for item in ("medium", "low", "high", "blocking"):
            self._severity.addItem(item)
        edit_form.addRow("Severity", self._severity)

        outer.addLayout(edit_form)

        self._notes = QPlainTextEdit()
        self._notes.setPlaceholderText("What happened, what data exposed it, and what should be checked next.")
        self._notes.setMinimumHeight(110)
        outer.addWidget(QLabel("Notes"))
        outer.addWidget(self._notes, stretch=1)

        self._recommendation = QPlainTextEdit()
        self._recommendation.setPlaceholderText("Optional: dataset fix, category to index, retrain target, threshold change.")
        self._recommendation.setMaximumHeight(90)
        outer.addWidget(QLabel("Suggested training action"))
        outer.addWidget(self._recommendation)

        self._status = QLabel("")
        self._status.setObjectName("feedbackStatus")
        self._status.setStyleSheet("border: none; color: rgba(203,75,22,0.92);")
        outer.addWidget(self._status)

        row = QHBoxLayout()
        row.addStretch(1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Save feedback")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        outer.addLayout(row)

    def _context_rows(self) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        context = str(self._payload.get("context") or "").strip()
        job_id = str(self._payload.get("job_id") or "").strip()
        scenario = str(self._payload.get("scenario") or "").strip()
        weights = str(self._payload.get("weights_path") or "").strip()
        result_path = str(self._payload.get("result_path") or "").strip()
        run_dir = str(self._payload.get("run_dir") or "").strip()
        summary = str(self._payload.get("summary") or "").strip()
        if context:
            rows.append(("Context", context))
        if job_id:
            rows.append(("Job", job_id))
        if scenario:
            rows.append(("Scenario", scenario))
        if weights:
            rows.append(("Weights", weights))
        if result_path:
            rows.append(("Result", result_path))
        elif run_dir:
            rows.append(("Run", run_dir))
        if summary:
            rows.append(("Summary", summary))
        return rows[:6]

    @staticmethod
    def _context_label(value: str) -> QLabel:
        label = QLabel(value)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setWordWrap(True)
        return label

    def _accept(self) -> None:
        notes = self._notes.toPlainText().strip()
        if len(notes) < 4:
            self._status.setText("Add a short note before saving.")
            return
        self.result_payload = ModelFeedbackResult(
            issue_type=str(self._issue.currentText() or ""),
            severity=str(self._severity.currentText() or ""),
            notes=notes,
            recommendation=self._recommendation.toPlainText().strip(),
        )
        self.accept()
