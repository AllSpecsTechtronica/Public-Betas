from __future__ import annotations

from typing import Optional, Sequence

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QStyleOptionGraphicsItem,
    QWidget,
)

# Aurora-leaning flow palette. Tones mirror the scenario readiness states the
# CatalogPanel computes so the diagram reads the same as the rest of CV Ops.
_TONE_COLORS = {
    "ok": "#38d26f",
    "active": "#24d8ff",
    "warning": "#d9b54a",
    "error": "#ff5a52",
    "idle": "#8da0a6",
}
_BG = "#070b0d"
_CARD = "#10171a"
_BORDER = "#1d4e58"
_TEXT_BRIGHT = "#f4f7f8"
_TEXT_DIM = "#cdd6d9"
_TEXT_MUTED = "#8da0a6"

_NODE_WIDTH = 320
_NODE_X = 26
_TOP_MARGIN = 56
_NODE_GAP = 30  # vertical space between a node bottom and the next node top
_LINE_HEIGHT = 16


# A single ecosystem node: rounded card with a tone accent rail, an index,
# a title, an uppercase state line, and a stack of detail lines. Painted
# natively so it lives inside Qt's scene graph (no QWebEngine / HTML).
class _FlowNode(QGraphicsItem):
    def __init__(
        self,
        index: int,
        title: str,
        state: str,
        lines: Sequence[str],
        tone: str,
    ) -> None:
        super().__init__()
        self._index = index
        self._title = title
        self._state = state
        self._lines = [str(line) for line in lines if str(line or "").strip()]
        self._tone = tone if tone in _TONE_COLORS else "idle"
        self._width = _NODE_WIDTH
        self._height = self._calc_height()

    def _calc_height(self) -> int:
        return 48 + len(self._lines) * _LINE_HEIGHT + 12

    def node_height(self) -> int:
        return self._height

    def boundingRect(self) -> QRectF:  # type: ignore[override]
        return QRectF(0, 0, self._width, self._height)

    def paint(  # type: ignore[override]
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(_TONE_COLORS[self._tone])

        body = QRectF(0, 0, self._width, self._height)
        path = QPainterPath()
        path.addRoundedRect(body, 8.0, 8.0)
        painter.fillPath(path, QColor(_CARD))
        pen = QPen(QColor(_BORDER))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawPath(path)

        # Tone accent rail down the left edge.
        rail = QPainterPath()
        rail.addRoundedRect(QRectF(0, 0, 4.0, self._height), 2.0, 2.0)
        painter.fillPath(rail, color)

        # Index badge.
        painter.setPen(color)
        idx_font = QFont()
        idx_font.setPointSize(13)
        idx_font.setBold(True)
        painter.setFont(idx_font)
        painter.drawText(
            QRectF(12, 8, 38, 22),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{self._index:02d}",
        )

        # Title.
        painter.setPen(QColor(_TEXT_BRIGHT))
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(
            QRectF(52, 8, self._width - 64, 18),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._title,
        )

        # State (uppercase, tone-colored).
        painter.setPen(color)
        state_font = QFont()
        state_font.setPointSize(8)
        painter.setFont(state_font)
        painter.drawText(
            QRectF(52, 27, self._width - 64, 14),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            str(self._state or "").upper(),
        )

        # Detail lines.
        painter.setPen(QColor(_TEXT_DIM))
        line_font = QFont()
        line_font.setPointSize(9)
        painter.setFont(line_font)
        y = 48.0
        for line in self._lines:
            painter.drawText(
                QRectF(14, y, self._width - 24, _LINE_HEIGHT),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                line,
            )
            y += _LINE_HEIGHT


# Native Qt ecosystem diagram for a single scenario: a progressive top-to-bottom
# chain of nodes (Scenario -> Dataset -> Model -> Guard -> Run Config ->
# Training -> Review) connected by arrows. Replaces the earlier HTML/QTextBrowser
# render with a real scene-graph surface that can be zoomed and scrolled.
class ScenarioFlowView(QGraphicsView):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("scenarioFlowView")
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setBackgroundBrush(QColor(_BG))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self._scenario = ""
        self._dataset = ""
        self.set_flow("", "", [])

    def set_flow(
        self,
        scenario: str,
        dataset: str,
        steps: Sequence[tuple[str, str, Sequence[str], str]],
    ) -> None:
        self._scenario = str(scenario or "")
        self._dataset = str(dataset or "")
        self._scene.clear()

        if not steps:
            self._draw_placeholder()
            return

        self._draw_header()

        nodes: list[_FlowNode] = []
        y = float(_TOP_MARGIN)
        for idx, (title, state, lines, tone) in enumerate(steps, start=1):
            node = _FlowNode(idx, str(title), str(state), lines, str(tone))
            node.setPos(_NODE_X, y)
            self._scene.addItem(node)
            nodes.append(node)
            y += node.node_height() + _NODE_GAP

        # Connector arrows between consecutive nodes.
        for first, second in zip(nodes, nodes[1:]):
            top = first.pos().y() + first.node_height()
            bottom = second.pos().y()
            self._draw_connector(_NODE_X + _NODE_WIDTH / 2.0, top, bottom)

        bottom = y + 10
        self._scene.setSceneRect(0, 0, _NODE_WIDTH + 2 * _NODE_X, bottom)

    def _draw_header(self) -> None:
        title = self._scene.addText("Scenario Flow", QFont("", 12, QFont.Weight.Bold))
        title.setDefaultTextColor(QColor(_TEXT_BRIGHT))
        title.setPos(_NODE_X - 4, 8)

        subtitle_text = (
            f"{self._scenario or 'Selected scenario'}  |  dataset "
            f"{self._dataset or 'Unlinked'}"
        )
        subtitle = self._scene.addText(subtitle_text, QFont("", 9))
        subtitle.setDefaultTextColor(QColor(_TEXT_MUTED))
        subtitle.setPos(_NODE_X - 4, 30)

    def _draw_connector(self, x: float, top: float, bottom: float) -> None:
        pen = QPen(QColor(_BORDER))
        pen.setWidthF(1.5)
        head = bottom - 7.0
        self._scene.addLine(x, top, x, head, pen)

        arrow = QPolygonF(
            [
                QPointF(x - 5.0, head),
                QPointF(x + 5.0, head),
                QPointF(x, bottom),
            ]
        )
        self._scene.addPolygon(arrow, QPen(Qt.PenStyle.NoPen), QColor(_BORDER))

    def _draw_placeholder(self) -> None:
        text = self._scene.addText(
            "Select a scenario to build its flow.",
            QFont("", 10),
        )
        text.setDefaultTextColor(QColor(_TEXT_MUTED))
        text.setPos(_NODE_X, _TOP_MARGIN)
        self._scene.setSceneRect(0, 0, _NODE_WIDTH + 2 * _NODE_X, 160)
