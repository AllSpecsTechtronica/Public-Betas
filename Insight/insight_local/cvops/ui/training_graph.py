from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from .cvops_theme import cvops_qcolor


class TrainingGraphWidget(QWidget):
    """Lightweight line graph for live training telemetry."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[dict[str, Any]] = []
        self.setMinimumHeight(72)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_points(self, points: list[dict[str, Any]]) -> None:
        self._points = list(points or [])
        self.update()

    def clear(self) -> None:
        self._points = []
        self.update()

    def refresh_theme_styles(self) -> None:
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        rect = self.rect().adjusted(10, 10, -10, -10)
        if rect.width() < 8 or rect.height() < 8:
            return
        plot = QRectF(rect.left() + 42, rect.top() + 12, rect.width() - 56, rect.height() - 34)
        if plot.width() <= 1.0 or plot.height() <= 1.0:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(rect, cvops_qcolor("bg_panel", 235))
        p.setPen(QPen(cvops_qcolor("line_light", 72), 1.0))
        for i in range(5):
            y = plot.top() + (plot.height() * i / 4.0)
            p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))

        if len(self._points) < 1:
            p.setPen(cvops_qcolor("text_iron", 150))
            p.drawText(plot, Qt.AlignmentFlag.AlignCenter, "No live training data yet")
            return

        epoch_points = [pt for pt in self._points if str(pt.get("event") or "") in {"epoch", "completed"}]
        if not epoch_points:
            p.setPen(cvops_qcolor("text_iron", 150))
            p.drawText(plot, Qt.AlignmentFlag.AlignCenter, "Waiting for epoch metrics...")
            return

        min_epoch = min(int(pt.get("epoch") or 0) for pt in epoch_points)
        max_epoch = max(int(pt.get("epoch") or 0) for pt in epoch_points)
        span = max(1, max_epoch - min_epoch)

        def _x(epoch: int) -> float:
            return plot.left() + ((epoch - min_epoch) / span) * plot.width()

        metric_vals: list[tuple[int, float]] = []
        metric_label = ""
        loss_vals = []
        for pt in epoch_points:
            # Prefer CV mAP50, else tabular val_acc / val_mae.
            mv = pt.get("map50")
            if mv is not None:
                metric_label = "mAP50"
                try:
                    metric_vals.append((int(pt.get("epoch") or 0), float(mv)))
                except Exception:
                    pass
            else:
                av = pt.get("val_acc")
                if av is not None:
                    metric_label = "val_acc"
                    try:
                        metric_vals.append((int(pt.get("epoch") or 0), float(av)))
                    except Exception:
                        pass
                else:
                    ev = pt.get("val_mae")
                    if ev is not None:
                        metric_label = "val_mae"
                        try:
                            metric_vals.append((int(pt.get("epoch") or 0), float(ev)))
                        except Exception:
                            pass
            lv = pt.get("train_loss")
            try:
                if lv is not None:
                    loss_vals.append((int(pt.get("epoch") or 0), float(lv)))
            except Exception:
                pass

        if metric_vals:
            vals_only = [v for _e, v in metric_vals]
            vmin = min(vals_only)
            vmax = max(vals_only)
            span_v = max(1e-9, vmax - vmin)
            invert = metric_label == "val_mae"  # lower is better
            metric_path = QPainterPath()
            for i, (ep, val) in enumerate(metric_vals):
                norm = (val - vmin) / span_v
                if invert:
                    norm = 1.0 - norm
                norm = max(0.0, min(1.0, float(norm)))
                y = plot.bottom() - (norm * plot.height())
                if i == 0:
                    metric_path.moveTo(QPointF(_x(ep), y))
                else:
                    metric_path.lineTo(QPointF(_x(ep), y))
            p.setPen(QPen(cvops_qcolor("accent_select", 230), 2.1))
            p.drawPath(metric_path)

        if loss_vals:
            max_loss = max(v for _, v in loss_vals)
            scale = max(0.0001, max_loss)
            loss_path = QPainterPath()
            for i, (ep, val) in enumerate(loss_vals):
                norm = max(0.0, min(1.0, val / scale))
                y = plot.bottom() - (norm * plot.height())
                if i == 0:
                    loss_path.moveTo(QPointF(_x(ep), y))
                else:
                    loss_path.lineTo(QPointF(_x(ep), y))
            loss_pen = QPen(cvops_qcolor("accent_active", 150), 1.8)
            loss_pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(loss_pen)
            p.drawPath(loss_path)

        font = QFont("JetBrains Mono", 9)
        if not font.exactMatch():
            font = QFont("IBM Plex Mono", 9)
        font.setStyleHint(QFont.StyleHint.Monospace)
        p.setFont(font)
        p.setPen(cvops_qcolor("text_signal", 190))
        p.drawText(int(plot.left()), int(rect.bottom()) - 3, f"Epoch {min_epoch}")
        p.drawText(int(plot.right()) - 70, int(rect.bottom()) - 3, f"Epoch {max_epoch}")
        p.setPen(cvops_qcolor("accent_select", 230))
        p.drawText(int(plot.left()), int(rect.top()) + 8, metric_label or "metric")
        p.setPen(cvops_qcolor("accent_active", 150))
        p.drawText(int(plot.left()) + 58, int(rect.top()) + 8, "train loss")
