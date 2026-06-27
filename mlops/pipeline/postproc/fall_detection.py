from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np


def _to_int_bbox(bbox: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1 = int(round(float(bbox[0])))
        y1 = int(round(float(bbox[1])))
        x2 = int(round(float(bbox[2])))
        y2 = int(round(float(bbox[3])))
    except Exception:
        return None
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _encode_overlay(image_bgr: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ""
    return base64.b64encode(encoded).decode("ascii")


def run(image_bgr: np.ndarray, raw_detections: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    hyperparams = config.get("hyperparams") if isinstance(config, dict) else {}
    threshold_raw = (hyperparams or {}).get("alert_confidence", 0.35)
    try:
        threshold = float(threshold_raw)
    except Exception:
        threshold = 0.35

    overlay = image_bgr.copy()
    h, w = overlay.shape[:2]

    detections: list[dict[str, Any]] = []
    for det in raw_detections:
        if not isinstance(det, dict):
            continue
        label = str(det.get("label") or det.get("class") or "")
        conf_raw = det.get("confidence", det.get("score"))
        try:
            conf = float(conf_raw)
        except Exception:
            conf = 0.0
        bbox = _to_int_bbox(det.get("bbox") or det.get("box"), w, h)
        if bbox is None or conf < threshold:
            continue
        x1, y1, x2, y2 = bbox
        color = (45, 65, 235)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"{label or 'fall'} {conf:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        detections.append(
            {
                "label": label or "fall",
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
            }
        )

    flagged = len(detections) > 0
    banner = "FALL DETECTED" if flagged else "NO FALL DETECTED"
    banner_color = (45, 65, 235) if flagged else (40, 160, 65)
    cv2.putText(
        overlay,
        banner,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        banner_color,
        2,
        cv2.LINE_AA,
    )

    return {
        "signal": {
            "flag": flagged,
            "summary": f"{len(detections)} fall event(s) above {threshold:.2f}" if flagged else "no fall events",
            "metrics": {
                "fall_count": len(detections),
                "alert_confidence": threshold,
            },
        },
        "detections": detections,
        "overlay_image": _encode_overlay(overlay),
    }
