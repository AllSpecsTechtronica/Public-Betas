# Base Edge Background Overlay
# ═══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HairlineEdgeBackgroundConfig:
    """Config for the always-on hairline edge background overlay."""

    sobel_ksize: int = 3
    threshold: int = 80
    scanline_stride: int = 4
    scanline_dilate_kernel: tuple[int, int] = (1, 2)
    overlay_color: tuple[int, int, int] = (0, 0, 255)
    frame_weight: float = 0.85
    overlay_weight: float = 0.15


class HairlineEdgeBackground:
    """Applies a subtle horizontal edge scan overlay to a BGR frame."""

    def __init__(self, config: HairlineEdgeBackgroundConfig | None = None):
        self.config = config or HairlineEdgeBackgroundConfig()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=self.config.sobel_ksize)
            sobel_x = np.abs(sobel_x)
            sobel_x = np.uint8(np.clip(sobel_x, 0, 255))

            _, thresh = cv2.threshold(sobel_x, self.config.threshold, 255, cv2.THRESH_BINARY)
            thin_edges = thresh.copy()

            scan_mask = np.zeros_like(thin_edges)
            scan_mask[:: self.config.scanline_stride, :] = 255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.config.scanline_dilate_kernel)
            scan_mask = cv2.dilate(scan_mask, kernel)
            masked_edges = cv2.bitwise_and(thin_edges, scan_mask)

            red_overlay = np.zeros_like(frame)
            red_overlay[masked_edges > 0] = self.config.overlay_color
            return cv2.addWeighted(
                frame,
                self.config.frame_weight,
                red_overlay,
                self.config.overlay_weight,
                0,
            )
        except Exception:
            return frame
