from __future__ import annotations
import base64

import numpy as np
from PyQt6.QtGui import QImage, QPixmap


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    if bgr is None or bgr.size == 0:
        return QImage()
    # Qt can display BGR planes directly, so we skip the BGR→RGB cvtColor.
    # np.ascontiguousarray guarantees the backing buffer layout QImage expects
    # and also hands ownership to the numpy array we return-bind via .copy().
    buf = np.ascontiguousarray(bgr)
    h, w, ch = buf.shape
    return QImage(buf.data, w, h, ch * w, QImage.Format.Format_BGR888).copy()


def pixmap_from_b64_jpeg(b64: str) -> QPixmap:
    if not b64:
        return QPixmap()
    raw = base64.b64decode(b64)
    img = QImage()
    img.loadFromData(raw, "JPEG")
    return QPixmap.fromImage(img)


def pixmap_from_b64_png(b64: str) -> QPixmap:
    if not b64:
        return QPixmap()
    raw = base64.b64decode(b64)
    img = QImage()
    img.loadFromData(raw, "PNG")
    return QPixmap.fromImage(img)
