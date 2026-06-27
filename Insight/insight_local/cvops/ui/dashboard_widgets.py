from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ...ui.theme import text_qcolor, theme_hex


class DashboardOverviewWidget(QWidget):
    """Native visual summary for the embedded dashboard tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dashboardOverview")
        self.setMinimumHeight(156)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._service: dict[str, Any] = {
            "running": False,
            "url": "",
            "embed_supported": False,
            "detail": "Dashboard stopped",
        }
        self._health: dict[str, Any] = {
            "status": "waiting",
            "queued": 0,
            "running": 0,
            "done": 0,
            "failed": 0,
            "slots_free": "",
            "max_workers": "",
        }
        self._scenarios: dict[str, int] = {
            "total": 0,
            "ready": 0,
            "partial": 0,
            "failed": 0,
        }
        self._jobs: dict[str, int] = {}

    def set_service(
        self,
        *,
        running: bool,
        url: str,
        embed_supported: bool,
        detail: str,
    ) -> None:
        self._service = {
            "running": bool(running),
            "url": str(url or ""),
            "embed_supported": bool(embed_supported),
            "detail": str(detail or ""),
        }
        self.update()

    def set_health(
        self,
        *,
        status: str,
        queued: int,
        running: int,
        done: int,
        failed: int,
        slots_free: object = "",
        max_workers: object = "",
    ) -> None:
        self._health = {
            "status": str(status or "unknown"),
            "queued": max(0, int(queued or 0)),
            "running": max(0, int(running or 0)),
            "done": max(0, int(done or 0)),
            "failed": max(0, int(failed or 0)),
            "slots_free": slots_free,
            "max_workers": max_workers,
        }
        self.update()

    def set_scenarios(self, *, total: int, ready: int, partial: int, failed: int) -> None:
        self._scenarios = {
            "total": max(0, int(total or 0)),
            "ready": max(0, int(ready or 0)),
            "partial": max(0, int(partial or 0)),
            "failed": max(0, int(failed or 0)),
        }
        self.update()

    def set_jobs(self, counts: dict[str, int]) -> None:
        self._jobs = {str(k): max(0, int(v or 0)) for k, v in dict(counts or {}).items()}
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        if rect.width() < 80 or rect.height() < 80:
            return

        panel = self._color("panel", 0.90)
        border = self._color("accent_dark", 0.24)
        accent = self._color("accent_dark", 0.95)
        ok = self._color("strip_soft", 0.90)
        warn = self._color("privacy_warn", 0.95)
        muted = text_qcolor(0.58)
        text = text_qcolor(0.94)
        drift = self._color("strip_soft", 0.90)

        p.setPen(QPen(border, 1.0))
        p.setBrush(panel)
        p.drawRoundedRect(rect, 8, 8)

        body = rect.adjusted(10, 9, -10, -9)
        card_gap = 8.0
        card_h = min(68.0, max(58.0, body.height() * 0.48))
        card_w = (body.width() - card_gap * 2.0) / 3.0
        cards = [
            QRectF(body.left(), body.top(), card_w, card_h),
            QRectF(body.left() + card_w + card_gap, body.top(), card_w, card_h),
            QRectF(body.left() + (card_w + card_gap) * 2.0, body.top(), card_w, card_h),
        ]

        service_color = accent if bool(self._service.get("running")) else muted
        service_value = "RUNNING" if bool(self._service.get("running")) else "STOPPED"
        embed = "embedded" if bool(self._service.get("embed_supported")) else "browser fallback"
        self._draw_card(
            p,
            cards[0],
            "Dashboard",
            service_value,
            embed,
            service_color,
            text,
            muted,
            border,
        )

        queued = int(self._health.get("queued") or 0)
        running = int(self._health.get("running") or 0)
        done = int(self._health.get("done") or 0)
        failed = int(self._health.get("failed") or 0)
        active = queued + running
        workers = self._workers_text()
        self._draw_card(
            p,
            cards[1],
            "Jobs",
            str(active),
            f"{done} done / {failed} issue",
            accent if active else ok,
            text,
            muted,
            border,
            detail=workers,
        )

        total = int(self._scenarios.get("total") or 0)
        ready = int(self._scenarios.get("ready") or 0)
        partial = int(self._scenarios.get("partial") or 0)
        scenario_failed = int(self._scenarios.get("failed") or 0)
        self._draw_card(
            p,
            cards[2],
            "Scenarios",
            f"{ready}/{total}",
            f"{partial} partial / {scenario_failed} issue",
            ok if scenario_failed == 0 else warn,
            text,
            muted,
            border,
        )

        bar_top = cards[0].bottom() + 12.0
        bar_h = max(26.0, body.bottom() - bar_top - 2.0)
        left_bar = QRectF(body.left(), bar_top, (body.width() - card_gap) / 2.0, bar_h)
        right_bar = QRectF(left_bar.right() + card_gap, bar_top, left_bar.width(), bar_h)

        job_segments = {
            "queued": queued,
            "running": running,
            "done": done,
            "issue": failed,
        }
        if sum(job_segments.values()) <= 0 and self._jobs:
            job_segments = {
                "queued": int(self._jobs.get("queued", 0)),
                "running": int(self._jobs.get("running", 0)),
                "done": int(self._jobs.get("done", 0)),
                "issue": int(self._jobs.get("error", 0)),
            }
        self._draw_stacked_bar(
            p,
            left_bar,
            "Job mix",
            job_segments,
            {
                "queued": muted,
                "running": accent,
                "done": ok,
                "issue": warn,
            },
            text,
            border,
        )
        self._draw_stacked_bar(
            p,
            right_bar,
            "Scenario readiness",
            {
                "ready": ready,
                "partial": partial,
                "issue": scenario_failed,
            },
            {
                "ready": ok,
                "partial": drift,
                "issue": warn,
            },
            text,
            border,
        )
        p.end()

    @staticmethod
    def _color(role: str, alpha: float) -> QColor:
        color = QColor(theme_hex(role))
        color.setAlphaF(max(0.0, min(1.0, float(alpha))))
        return color

    def _workers_text(self) -> str:
        free = self._health.get("slots_free", "")
        total = self._health.get("max_workers", "")
        if free != "" and total != "":
            return f"{free}/{total} workers free"
        return str(self._health.get("status") or "workers unknown")

    def _font(self, size: int, *, bold: bool = False, mono: bool = False) -> QFont:
        font = QFont("Menlo" if mono else "Arial", size)
        if mono:
            font.setStyleHint(QFont.StyleHint.Monospace)
        if bold:
            font.setWeight(QFont.Weight.DemiBold)
        return font

    def _draw_card(
        self,
        p: QPainter,
        rect: QRectF,
        title: str,
        value: str,
        subtitle: str,
        accent: QColor,
        text: QColor,
        muted: QColor,
        border: QColor,
        *,
        detail: str = "",
    ) -> None:
        fill = self._color("input_fill", 0.72)
        p.setPen(QPen(border, 1.0))
        p.setBrush(fill)
        p.drawRoundedRect(rect, 7, 7)

        p.setFont(self._font(8, bold=True))
        p.setPen(muted)
        p.drawText(rect.adjusted(9, 7, -9, -7), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, title)

        p.setFont(self._font(18, bold=True, mono=True))
        p.setPen(accent)
        p.drawText(rect.adjusted(9, 21, -9, -5), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, value)

        p.setFont(self._font(8))
        p.setPen(text)
        p.drawText(rect.adjusted(9, 45, -9, -5), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, subtitle)
        if detail:
            p.setPen(muted)
            p.drawText(rect.adjusted(9, 45, -9, -5), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, detail)

    def _draw_stacked_bar(
        self,
        p: QPainter,
        rect: QRectF,
        title: str,
        segments: dict[str, int],
        colors: dict[str, QColor],
        text: QColor,
        border: QColor,
    ) -> None:
        p.setFont(self._font(8, bold=True))
        p.setPen(text)
        p.drawText(rect.adjusted(0, 0, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, title)

        bar = QRectF(rect.left(), rect.top() + 16.0, rect.width(), min(12.0, rect.height() - 18.0))
        p.setPen(QPen(border, 1.0))
        p.setBrush(self._color("input_fill", 0.58))
        p.drawRoundedRect(bar, 5, 5)

        total = sum(max(0, int(v or 0)) for v in segments.values())
        if total <= 0:
            return
        x = bar.left()
        for label, value in segments.items():
            count = max(0, int(value or 0))
            if count <= 0:
                continue
            width = max(1.0, bar.width() * (count / total))
            seg = QRectF(x, bar.top(), min(width, bar.right() - x), bar.height())
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(colors.get(label, text))
            p.drawRoundedRect(seg, 5, 5)
            x += width

        p.setFont(self._font(7, mono=True))
        p.setPen(text_qcolor(0.62))
        legend = "  ".join(f"{k}:{v}" for k, v in segments.items() if int(v or 0) > 0)
        p.drawText(rect.adjusted(0, 30, 0, 0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, legend)
