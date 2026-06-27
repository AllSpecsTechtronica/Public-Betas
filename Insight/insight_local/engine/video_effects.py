from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from ..config import heatmap_category


# ──────────────────────────────────────────────────────────────────────
# Flat accent colors for non-human categories (BGR)
# ──────────────────────────────────────────────────────────────────────

_CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "plant":     (60, 210, 80),
    "animal":    (190, 200, 0),
    "tech":      (255, 80, 180),
    "inorganic": (255, 150, 60),
}

# Human thermal gradient stops: (normalised_dist_from_center, BGR)
# dist 0.0 = bbox center (hottest), 1.0 = bbox perimeter (coolest)
_HUMAN_STOPS: list[tuple[float, tuple[int, int, int]]] = [
    (0.00, (0,   80,  255)),   # hot red-orange
    (0.22, (0,   140, 255)),   # orange
    (0.44, (0,   220, 255)),   # yellow
    (0.64, (30,  230, 160)),   # yellow-green
    (0.82, (190, 200, 0)),     # cyan
    (1.00, (220, 100, 20)),    # cool blue
]


def _build_depth_mask(
    gray_roi: np.ndarray,
    kernel_dim: int,
) -> Optional[np.ndarray]:
    """Return per-pixel depth t in [0,1] where 0=deep interior (hot), 1=silhouette (cool).

    Uses Canny contour fill + distance transform so the gradient follows the
    actual object shape rather than the rectangular bounding box geometry.
    Returns None when the ROI is too featureless to produce a meaningful depth range.
    """
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    canny   = cv2.Canny(blurred, 40, 120)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_dim, kernel_dim))
    closed  = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 30]

    interior = np.zeros(gray_roi.shape[:2], dtype=np.uint8)
    if contours:
        cv2.drawContours(interior, contours, -1, 255, thickness=-1)
    else:
        # [TIER-1 FALLBACK] no recognizable contours — treat entire ROI as interior
        interior[:] = 255

    dist_raw = cv2.distanceTransform(interior, cv2.DIST_L2, cv2.DIST_MASK_5)
    dist_max = float(dist_raw.max())
    if dist_max < 2.0:
        # [TIER-2 FALLBACK] no meaningful depth range — caller will use radial fallback
        return None

    # Invert so t=0 at deepest point (hot), t=1 at outer silhouette (cool)
    return (1.0 - np.clip(dist_raw / dist_max, 0.0, 1.0)).astype(np.float32)


def _paint_human_thermal_edges(
    color_overlay: np.ndarray,
    edge_pixels: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    gray_frame: Optional[np.ndarray] = None,
) -> None:
    """Color edge pixels inside a human bbox with a depth-aware warm->cool gradient.

    When gray_frame is provided the gradient follows the actual object silhouette
    via contour fill + distance transform. Falls back to the original radial
    bbox-center distance when the depth mask cannot be computed.
    """
    roi = edge_pixels[y1:y2, x1:x2]
    ys, xs = np.nonzero(roi)
    if len(ys) == 0:
        return

    bh = y2 - y1
    bw = x2 - x1

    depth_map = None
    if gray_frame is not None and bh >= 8 and bw >= 8:
        kd = max(5, min(21, int(min(bh, bw) * 0.08)))
        if kd % 2 == 0:
            kd += 1
        depth_map = _build_depth_mask(gray_frame[y1:y2, x1:x2], kd)

    if depth_map is not None:
        dist = depth_map[ys, xs]
    else:
        # [RADIAL FALLBACK] original bbox-center distance
        cx = bw / 2.0
        cy = bh / 2.0
        half_diag = max(1.0, (cx ** 2 + cy ** 2) ** 0.5)
        dx = (xs.astype(np.float32) - cx) / half_diag
        dy = (ys.astype(np.float32) - cy) / half_diag
        dist = np.clip(np.sqrt(dx * dx + dy * dy), 0.0, 1.0)

    colors = np.zeros((len(ys), 3), dtype=np.float32)
    for i in range(len(_HUMAN_STOPS) - 1):
        t0, c0 = _HUMAN_STOPS[i]
        t1, c1 = _HUMAN_STOPS[i + 1]
        band = (dist >= t0) & (dist <= t1)
        if not np.any(band):
            continue
        a = ((dist[band] - t0) / max(t1 - t0, 1e-6)).reshape(-1, 1)
        c0a = np.array(c0, dtype=np.float32)
        c1a = np.array(c1, dtype=np.float32)
        colors[band] = c0a + (c1a - c0a) * a

    color_overlay[ys + y1, xs + x1] = np.clip(colors, 0, 255).astype(np.uint8)


def _paint_category_depth_edges(
    color_overlay: np.ndarray,
    edge_pixels: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    base_color: tuple[int, int, int],
    gray_frame: Optional[np.ndarray] = None,
) -> None:
    """Color non-human category edges with depth-modulated intensity.

    Interior edges (deep in shape) render at full saturation; outer silhouette
    edges fade to ~40% so the color hugs the object contour rather than filling
    the entire bbox uniformly.
    """
    roi = edge_pixels[y1:y2, x1:x2]
    ys, xs = np.nonzero(roi)
    if len(ys) == 0:
        return

    bh = y2 - y1
    bw = x2 - x1
    depth_map = None
    if gray_frame is not None and bh >= 8 and bw >= 8:
        kd = max(5, min(21, int(min(bh, bw) * 0.08)))
        if kd % 2 == 0:
            kd += 1
        depth_map = _build_depth_mask(gray_frame[y1:y2, x1:x2], kd)

    if depth_map is not None:
        # t=0 deep interior -> 100% intensity; t=1 outer silhouette -> 40%
        depth_t   = depth_map[ys, xs]
        intensity = 0.4 + 0.6 * (1.0 - depth_t)
        bc        = np.array(base_color, dtype=np.float32)
        colors    = np.clip(bc[None, :] * intensity[:, None], 0, 255).astype(np.uint8)
        color_overlay[ys + y1, xs + x1] = colors
    else:
        color_overlay[y1:y2, x1:x2][roi > 0] = base_color


# ──────────────────────────────────────────────────────────────────────
# Hairline edge background
# ──────────────────────────────────────────────────────────────────────

def apply_hairline_edge_background(
    frame: np.ndarray,
    overlays: list,
    *,
    thermal: bool = False,
    blend_alpha: float = 0.15,
) -> np.ndarray:
    """Apply subtle horizontal edge scan overlay.

    Normal mode: all edges red.
    Thermal mode: edges inside each detection bbox get category coloring.
    Human bboxes use a contour-depth gradient (hot at interior edges,
    cool at the outer silhouette) so the shading hugs the object shape.
    Non-human bboxes use depth-modulated accent color intensity.
    """
    try:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=3)
        sobel_x = np.uint8(np.clip(np.abs(sobel_x), 0, 255))

        _, thresh = cv2.threshold(sobel_x, 80, 255, cv2.THRESH_BINARY)
        scan_mask = np.zeros_like(thresh)
        scan_mask[::4, :] = 255
        scan_mask = cv2.dilate(
            scan_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2)),
        )
        edge_pixels = cv2.bitwise_and(thresh, scan_mask)

        color_overlay = np.zeros_like(frame)
        color_overlay[edge_pixels > 0] = (0, 0, 255)

        if thermal and overlays:
            for overlay in overlays:
                bbox_norm = overlay.get("bbox_norm") if isinstance(overlay, dict) else None
                if not isinstance(bbox_norm, (list, tuple)) or len(bbox_norm) < 4:
                    continue
                try:
                    x1 = max(0, min(w - 1, int(float(bbox_norm[0]) * w)))
                    y1 = max(0, min(h - 1, int(float(bbox_norm[1]) * h)))
                    x2 = max(0, min(w,     int(float(bbox_norm[2]) * w)))
                    y2 = max(0, min(h,     int(float(bbox_norm[3]) * h)))
                except (TypeError, ValueError):
                    continue
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue

                cat = heatmap_category(str(overlay.get("label", "") or ""))
                if cat == "human":
                    _paint_human_thermal_edges(
                        color_overlay, edge_pixels, x1, y1, x2, y2,
                        gray_frame=gray,
                    )
                else:
                    color = _CATEGORY_COLORS.get(cat, _CATEGORY_COLORS["inorganic"])
                    _paint_category_depth_edges(
                        color_overlay, edge_pixels, x1, y1, x2, y2,
                        color, gray_frame=gray,
                    )

        ba = float(max(0.01, min(1.0, blend_alpha)))
        return cv2.addWeighted(frame, 1.0 - ba, color_overlay, ba, 0)
    except Exception:
        return frame
