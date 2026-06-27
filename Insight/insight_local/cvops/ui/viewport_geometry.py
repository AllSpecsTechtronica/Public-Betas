from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow, QScrollArea, QWidget


def reference_layout_size(widget: QWidget | None) -> tuple[int, int]:
    """Return (width, height) for responsive layout tied to what the user actually sees.

    When `widget` lives inside a QScrollArea, its own height is often the full scroll
    content height, so window resize does not change it. Prefer the scroll area's
    viewport size, then the main window's central widget, then the widget's size.
    """
    if widget is None:
        return 800, 600
    w = max(1, int(widget.width()))
    h = max(1, int(widget.height()))
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QScrollArea):
            vp = parent.viewport()
            if vp is not None:
                return max(1, int(vp.width())), max(1, int(vp.height()))
        parent = parent.parentWidget()
    win = widget.window()
    if isinstance(win, QMainWindow):
        cw = win.centralWidget()
        if cw is not None:
            return max(1, int(cw.width())), max(1, int(cw.height()))
    return w, h
