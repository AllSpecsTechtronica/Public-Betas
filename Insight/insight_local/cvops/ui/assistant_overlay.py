from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import QPoint, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...ui.theme import text_css, theme_rgba
from .cvops_theme import repolish
from .notes_ai_keys import DEFAULT_ASSISTANT_NAME, assistant_display_name
from .notes_ai_workspace import NotesAiWorkspace


class _AssistantResizeHandle(QFrame):
    def __init__(self, owner: "AssistantOverlayWindow") -> None:
        super().__init__(owner)
        self._owner = owner
        self.setObjectName("cvOpsAssistantResizeHandle")
        self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        self.setFixedSize(18, 18)
        self.setToolTip("Resize assistant card")
        self.setStyleSheet(
            "QFrame#cvOpsAssistantResizeHandle {"
            f" background: {theme_rgba('accent_dark', 0.22)};"
            f" border: 1px solid {theme_rgba('accent_dark', 0.36)};"
            " border-radius: 4px;"
            "}"
            "QFrame#cvOpsAssistantResizeHandle:hover {"
            f" background: {theme_rgba('accent_dark', 0.34)};"
            "}"
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._owner.begin_user_resize(event.globalPosition().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._owner.continue_user_resize(event.globalPosition().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._owner.end_user_resize()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class AssistantOverlayWindow(QWidget):
    """In-window assistant card backed by the Notes AI workspace."""

    closed = pyqtSignal()

    def __init__(
        self,
        *,
        workspace_provider: Callable[[], Optional[NotesAiWorkspace]],
        workspace_restorer: Callable[[], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("CV Ops Assistant")
        self.setObjectName("cvOpsAssistantOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMinimumSize(420, 560)

        self._workspace_provider = workspace_provider
        self._workspace_restorer = workspace_restorer
        self._workspace: Optional[NotesAiWorkspace] = None
        self._workspace_mounted = False
        self._workspace_error_connected = False
        self._workspace_name_connected = False
        self._scratch_chat_id = ""
        self._user_size = QSize(520, 720)
        self._resize_origin = QPoint()
        self._resize_start_size = QSize()
        self._last_anchor_rect = QRect()
        self._last_anchor_margin = 12
        # Interactive resize is throttled: mouse-move events fire far faster than
        # a relayout of the embedded Notes workspace (QTextBrowser re-wrap +
        # translucent recompositing) can keep up, which otherwise floods the UI
        # thread and beachballs. We coalesce moves to ~60fps and freeze the
        # heavy workspace's repaints until the drag ends.
        self._pending_resize_size: Optional[QSize] = None
        self._resize_active = False
        self._resize_throttle = QTimer(self)
        self._resize_throttle.setSingleShot(True)
        self._resize_throttle.setInterval(16)
        self._resize_throttle.timeout.connect(self._apply_pending_resize)
        self.resize(self._user_size)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(0)

        card = QFrame()
        card.setObjectName("cvOpsAssistantCard")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        card.setStyleSheet(
            "QFrame#cvOpsAssistantCard {"
            f" background: {theme_rgba('panel', 0.96)};"
            f" border: 1px solid {theme_rgba('accent_dark', 0.34)};"
            " border-radius: 6px;"
            "}"
        )
        root.addWidget(card, stretch=1)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QFrame()
        header.setObjectName("cvOpsAssistantHeader")
        header.setFixedHeight(30)
        header.setStyleSheet(
            "QFrame#cvOpsAssistantHeader {"
            " background: transparent;"
            " border: none;"
            "}"
        )
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)

        title = QLabel(assistant_display_name().upper())
        title.setObjectName("cvOpsAssistantTitle")
        self._title = title
        title.setStyleSheet(
            "QLabel#cvOpsAssistantTitle {"
            f" color: {text_css(0.92)};"
            " background: transparent;"
            " border: none;"
            " padding: 0 2px;"
            " font-size: 10px;"
            " font-weight: 700;"
            " letter-spacing: 0.06em;"
            "}"
        )
        header_row.addWidget(title)

        self._status = QLabel("Scratch chat")
        self._status.setWordWrap(False)
        self._status.setStyleSheet(f"font-size: 10px; color: {text_css(0.72)}; border: none;")
        header_row.addWidget(self._status, stretch=1)

        self._new_btn = QPushButton("New")
        self._new_btn.setToolTip("Start a fresh disposable assistant question.")
        self._new_btn.clicked.connect(self._new_scratch)
        header_row.addWidget(self._new_btn)

        self._discard_btn = QPushButton("Discard")
        self._discard_btn.setToolTip("Delete this scratch chat from the Notes chat store.")
        self._discard_btn.clicked.connect(self._discard_scratch)
        header_row.addWidget(self._discard_btn)

        self._keep_btn = QPushButton("Keep")
        self._keep_btn.setProperty("isPrimary", True)
        self._keep_btn.setToolTip("Keep this scratch question as a normal Notes AI chat.")
        self._keep_btn.clicked.connect(self._keep_scratch)
        repolish(self._keep_btn)
        header_row.addWidget(self._keep_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        header_row.addWidget(close_btn)
        layout.addWidget(header)

        self._card_layout = layout
        self._workspace_insert_index = layout.count()
        self._mount_workspace()

        grip_row = QHBoxLayout()
        grip_row.setContentsMargins(0, 0, 0, 0)
        grip_row.setSpacing(0)
        grip_row.addWidget(_AssistantResizeHandle(self))
        grip_row.addStretch(1)
        layout.addLayout(grip_row)

        self._new_scratch()

    def _assistant_name(self) -> str:
        if self._workspace is not None:
            getter = getattr(self._workspace, "assistant_name", None)
            if callable(getter):
                return str(getter() or "").strip() or DEFAULT_ASSISTANT_NAME
        return assistant_display_name()

    def _sync_assistant_name(self, name: str = "") -> None:
        chosen = str(name or "").strip() or self._assistant_name()
        self._title.setText(chosen.upper())
        self.setWindowTitle(f"CV Ops {chosen}")
        self._new_btn.setToolTip(f"Start a fresh disposable {chosen} question.")
        parent = self.parentWidget()
        btn = getattr(parent, "_ai_assistant_btn", None)
        if isinstance(btn, QPushButton):
            btn.setText(chosen)
            btn.setToolTip(f"Open {chosen} for quick CV Ops questions.")

    def show_for_parent(self, parent: Optional[QWidget]) -> None:
        if parent is not None and self.parentWidget() is None:
            self.setParent(parent)
        self._mount_workspace()
        self._sync_assistant_name()
        if not self._scratch_chat_id:
            self._new_scratch()
        self.show()
        self.raise_()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def place_in_parent(self, anchor_rect: QRect, *, margin: int = 12) -> None:
        if anchor_rect.width() <= 0 or anchor_rect.height() <= 0:
            return
        margin = max(0, int(margin))
        self._last_anchor_rect = QRect(anchor_rect)
        self._last_anchor_margin = margin

        available_width = max(1, anchor_rect.width() - (margin * 2))
        available_height = max(1, anchor_rect.height() - (margin * 2))
        min_width = min(self.minimumWidth(), available_width)
        min_height = min(self.minimumHeight(), available_height)
        width = max(min_width, min(self._user_size.width(), available_width))
        height = max(min_height, min(self._user_size.height(), available_height))

        min_x = anchor_rect.left() + margin
        min_y = anchor_rect.top() + margin
        max_y = anchor_rect.bottom() - height - margin
        x = anchor_rect.right() - width - margin
        y = max(min_y, max_y)
        self.resize(width, height)
        self.move(max(min_x, x), max(min_y, y))
        self.raise_()

    def begin_user_resize(self, global_pos: QPoint) -> None:
        self._resize_origin = QPoint(global_pos)
        self._resize_start_size = QSize(self._user_size)
        self._resize_active = True
        # Suppress the workspace's expensive content relayout/repaint for the
        # duration of the drag; the outer card still tracks the cursor live, and
        # the chat reflows once on release. This is what keeps the drag smooth
        # instead of beachballing on every mouse-move.
        try:
            if self._workspace is not None:
                self._workspace.setUpdatesEnabled(False)
        except Exception:
            pass

    def continue_user_resize(self, global_pos: QPoint) -> None:
        if self._resize_start_size.isEmpty():
            return
        delta = global_pos - self._resize_origin
        requested = QSize(
            self._resize_start_size.width() - delta.x(),
            self._resize_start_size.height() + delta.y(),
        )
        requested = self._clamp_requested_size(requested)
        self._user_size = requested
        # Coalesce: remember the latest target and apply it at most once per
        # ~16ms tick rather than on every mouse-move event.
        self._pending_resize_size = requested
        if not self._resize_throttle.isActive():
            self._resize_throttle.start()

    def _apply_pending_resize(self) -> None:
        size = self._pending_resize_size
        self._pending_resize_size = None
        if size is None:
            return
        if self._last_anchor_rect.isValid():
            self.place_in_parent(self._last_anchor_rect, margin=self._last_anchor_margin)
        else:
            self.resize(size)

    def end_user_resize(self) -> None:
        self._resize_throttle.stop()
        # Flush the final size, then re-enable and repaint the workspace once.
        self._apply_pending_resize()
        self._resize_origin = QPoint()
        self._resize_start_size = QSize()
        if self._resize_active:
            self._resize_active = False
            try:
                if self._workspace is not None:
                    self._workspace.setUpdatesEnabled(True)
                    self._workspace.update()
            except Exception:
                pass

    def _clamp_requested_size(self, requested: QSize) -> QSize:
        width = max(self.minimumWidth(), requested.width())
        height = max(self.minimumHeight(), requested.height())
        if self._last_anchor_rect.isValid():
            margin = self._last_anchor_margin
            width = min(width, max(self.minimumWidth(), self._last_anchor_rect.width() - (margin * 2)))
            height = min(height, max(self.minimumHeight(), self._last_anchor_rect.height() - (margin * 2)))
        return QSize(width, height)

    def _mount_workspace(self) -> bool:
        if self._workspace_mounted and self._workspace is not None:
            return True
        workspace = self._workspace_provider()
        if workspace is None:
            self._status.setText("Notes AI workspace is not available.")
            return False
        self._workspace = workspace
        if not self._workspace_error_connected:
            workspace.errorRaised.connect(self._on_workspace_error)
            self._workspace_error_connected = True
        if not self._workspace_name_connected:
            workspace.assistantNameChanged.connect(self._sync_assistant_name)
            self._workspace_name_connected = True
        self._sync_assistant_name()
        workspace.set_compact_overlay_mode(True)
        self._card_layout.insertWidget(self._workspace_insert_index, workspace, stretch=1)
        self._workspace_mounted = True
        workspace.show()
        workspace.focus_chats_sidebar_mode()
        return True

    def _restore_workspace(self) -> None:
        workspace = self._workspace
        if workspace is None or not self._workspace_mounted:
            return
        try:
            workspace.setUpdatesEnabled(True)
            if self._workspace_error_connected:
                try:
                    workspace.errorRaised.disconnect(self._on_workspace_error)
                except Exception:
                    pass
                self._workspace_error_connected = False
            if self._workspace_name_connected:
                try:
                    workspace.assistantNameChanged.disconnect(self._sync_assistant_name)
                except Exception:
                    pass
                self._workspace_name_connected = False
            self._card_layout.removeWidget(workspace)
            workspace.setParent(None)
        finally:
            self._workspace_mounted = False
            self._workspace = None
            self._workspace_restorer()

    def _new_scratch(self) -> None:
        if not self._mount_workspace() or self._workspace is None:
            return
        if self._workspace.is_ai_busy():
            QMessageBox.information(self, self._assistant_name(), "Wait for the current reply to finish.")
            return
        self._scratch_chat_id = self._workspace.start_scratch_chat(f"Scratch {self._assistant_name()} question")
        self._status.setText("Throwaway question")

    def _discard_scratch(self) -> None:
        if self._workspace is None:
            return
        if self._workspace.is_ai_busy():
            QMessageBox.information(self, self._assistant_name(), "Wait for the current reply to finish.")
            return
        cid = self._scratch_chat_id
        if not cid:
            self._new_scratch()
            return
        self._scratch_chat_id = self._workspace.discard_chat_without_prompt(
            cid,
            replacement_title=f"Scratch {self._assistant_name()} question",
        )
        self._status.setText("Discarded. New scratch ready.")

    def _keep_scratch(self) -> None:
        if self._workspace is None:
            return
        cid = self._scratch_chat_id
        if not cid:
            self._status.setText("No scratch chat to keep.")
            return
        if self._workspace.chat_message_count(cid) <= 0:
            self._status.setText("Ask a question before keeping the chat.")
            return
        title = self._workspace.suggested_chat_title(cid, f"{self._assistant_name()} chat")
        if self._workspace.keep_chat_without_prompt(cid, title=title):
            self._scratch_chat_id = ""
            self._status.setText("Kept in Notes AI chats.")
        else:
            self._status.setText("Could not keep this chat.")

    def _on_workspace_error(self, message: str) -> None:
        self._status.setText(str(message or f"{self._assistant_name()} error")[:140])

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._restore_workspace()
        self.closed.emit()
        super().closeEvent(event)
