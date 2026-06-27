from collections import deque
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


@dataclass
class HeatmapHazeConfig:
    alpha: float = 0.8
    decay: float = 0.6
    update_interval: int = 2
    max_objects: int = 12
    cool_color: tuple[int, int, int] = (0, 255, 255)
    warm_color: tuple[int, int, int] = (0, 0, 255)
    blur: int = 7
    mask_blur: int = 9
    motion_threshold: float = 6.0
    motion_decay: float = 0.2
    motion_cooldown_frames: int = 2
    mask_weight: float = 0.25
    overlay_blend: float = 0.15
    intensity: float = 4.0
    stabilizer_window: int = 4
    stabilizer_threshold: float = 10.0


class HeatmapHaze:
    def __init__(self, config: HeatmapHazeConfig, enabled: bool = False):
        self.config = config
        self.enabled = bool(enabled)

        self.heatmap = None
        self.heatmap_color = None
        self._heatmap_update_counter = 0
        self._heatmap_prev_gray = None
        self._heatmap_motion_cooldown = 0
        self._prev_overlay = None
        self._stabilizer_deltas = deque(maxlen=max(1, int(self.config.stabilizer_window)))

    def clear_buffers(self) -> None:
        if self.heatmap is not None:
            self.heatmap.fill(0)
        if self.heatmap_color is not None:
            self.heatmap_color.fill(0)
        self._prev_overlay = None

    def update_from_detections(
        self,
        frame,
        detections,
        *,
        compute_background_edges: Callable,
        min_contour_area: float,
        foreground_mask,
        min_foreground_ratio: float,
        max_foreground_ratio: float,
        roi_morph_iterations: int,
        roi_morph_kernel,
    ) -> None:
        if not self.enabled:
            return
        if frame is None or frame.size == 0:
            return

        h, w = frame.shape[:2]
        if self.heatmap is None or self.heatmap.shape != (h, w):
            self.heatmap = np.zeros((h, w), dtype=np.float32)
            self._heatmap_prev_gray = None
        if self.heatmap_color is None or self.heatmap_color.shape != (h, w, 3):
            self.heatmap_color = np.zeros((h, w, 3), dtype=np.float32)

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = None

        if gray is not None and self._heatmap_prev_gray is not None and self._heatmap_prev_gray.shape == gray.shape:
            try:
                # Stabilize against global camera motion by measuring median pixel delta.
                delta = np.median(cv2.absdiff(gray, self._heatmap_prev_gray))
                self._stabilizer_deltas.append(delta)
                if len(self._stabilizer_deltas) == self._stabilizer_deltas.maxlen:
                    median_delta = float(np.median(self._stabilizer_deltas))
                    if median_delta > self.config.stabilizer_threshold:
                        self.clear_buffers()
                        self._heatmap_motion_cooldown = max(1, int(self.config.motion_cooldown_frames))
                        self._heatmap_prev_gray = gray
                        return
            except Exception:
                pass

        if self._heatmap_motion_cooldown > 0:
            self.heatmap *= float(self.config.motion_decay)
            self.heatmap_color *= float(self.config.motion_decay)
            self._heatmap_motion_cooldown -= 1
            if gray is not None:
                self._heatmap_prev_gray = gray
            return

        if gray is not None and self._heatmap_prev_gray is not None and gray.shape == self._heatmap_prev_gray.shape:
            motion_level = float(np.mean(cv2.absdiff(gray, self._heatmap_prev_gray)))
            if motion_level > self.config.motion_threshold:
                self.clear_buffers()
                self._heatmap_motion_cooldown = max(1, int(self.config.motion_cooldown_frames))
                self._heatmap_prev_gray = gray
                return

        if gray is not None:
            self._heatmap_prev_gray = gray

        self.heatmap *= float(self.config.decay)
        self.heatmap_color *= float(self.config.decay)

        self._heatmap_update_counter = (self._heatmap_update_counter + 1) % max(1, int(self.config.update_interval))
        if self._heatmap_update_counter != 0:
            return

        dets = sorted(detections or [], key=lambda d: d.get("confidence", 0), reverse=True)[: max(1, int(self.config.max_objects))]
        for d in dets:
            try:
                x1, y1, x2, y2 = d.get("bbox", [0, 0, 0, 0])
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if x2 - x1 <= 2 or y2 - y1 <= 2:
                    continue

                roi_frame = frame[y1:y2, x1:x2]
                if roi_frame.size == 0:
                    continue

                edges = compute_background_edges(roi_frame)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
                if not contours:
                    continue

                mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                for cnt in contours:
                    cv2.drawContours(mask, [cnt], -1, 255, -1)

                if foreground_mask is not None:
                    fg_roi = foreground_mask[y1:y2, x1:x2]
                    if fg_roi.size > 0:
                        fg_ratio = float(np.mean(fg_roi > 0))
                        if min_foreground_ratio <= fg_ratio <= max_foreground_ratio:
                            mask = cv2.bitwise_and(mask, fg_roi)

                if roi_morph_iterations > 0:
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, roi_morph_kernel, iterations=roi_morph_iterations)

                roi_heat = self.heatmap[y1:y2, x1:x2]
                mask_norm = mask.astype(np.float32) / 255.0

                if self.config.mask_blur and self.config.mask_blur > 0:
                    k = int(self.config.mask_blur)
                    if k % 2 == 0:
                        k += 1
                    mask_norm = cv2.GaussianBlur(mask_norm, (k, k), 0)

                if self.config.intensity and self.config.intensity != 1.0:
                    mask_norm = np.clip(mask_norm * float(self.config.intensity), 0.0, 1.0)

                blend_w = float(self.config.mask_weight)
                roi_heat[:] = roi_heat * (1.0 - blend_w) + mask_norm * blend_w

                roi_h, roi_w = mask.shape
                if roi_h <= 0 or roi_w <= 0:
                    continue

                cy = roi_h * 0.5
                cx = roi_w * 0.5
                yy, xx = np.ogrid[:roi_h, :roi_w]
                dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                max_dist = max(1.0, np.sqrt(cy**2 + cx**2))
                t = 1.0 - np.clip(dist / max_dist, 0.0, 1.0)

                cool = np.array(self.config.cool_color, dtype=np.float32)
                warm = np.array(self.config.warm_color, dtype=np.float32)
                color_map = cool + (warm - cool) * t[..., None]
                color_map *= mask_norm[..., None]

                roi_color = self.heatmap_color[y1:y2, x1:x2]
                try:
                    blended = cv2.addWeighted(roi_color, 1.0 - blend_w, color_map, blend_w, 0)
                    np.copyto(roi_color, blended)
                except Exception:
                    roi_color[:] = roi_color * (1.0 - blend_w) + color_map * blend_w
            except Exception:
                continue

        if self.config.blur and self.config.blur > 0:
            k = int(self.config.blur)
            if k % 2 == 0:
                k += 1
            self.heatmap = cv2.GaussianBlur(self.heatmap, (k, k), 0)
            self.heatmap_color = cv2.GaussianBlur(self.heatmap_color, (k, k), 0)

    def apply(self, frame):
        if not self.enabled or self.heatmap_color is None:
            return frame
        if self.heatmap_color.shape[:2] != frame.shape[:2]:
            return frame

        overlay = np.clip(self.heatmap_color, 0.0, 255.0).astype(np.uint8)
        if overlay.max() < 5:
            if self.heatmap is None:
                self._prev_overlay = None
                return frame

            heat_u8 = np.clip(self.heatmap * 255.0, 0, 255).astype(np.uint8)
            if heat_u8.max() < 5:
                self._prev_overlay = None
                return frame

            try:
                heat_norm = heat_u8.astype(np.float32) / 255.0
                cool = np.array(self.config.cool_color, dtype=np.float32)
                warm = np.array(self.config.warm_color, dtype=np.float32)
                overlay = (cool + (warm - cool) * heat_norm[..., None]).astype(np.uint8)
            except Exception:
                self._prev_overlay = None
                return frame

        if self._prev_overlay is not None and self._prev_overlay.shape == overlay.shape:
            try:
                overlay = cv2.addWeighted(
                    self._prev_overlay,
                    1.0 - self.config.overlay_blend,
                    overlay,
                    self.config.overlay_blend,
                    0,
                )
            except Exception:
                overlay = (
                    self._prev_overlay * (1.0 - self.config.overlay_blend)
                    + overlay * self.config.overlay_blend
                ).astype(np.uint8)

        self._prev_overlay = overlay
        return cv2.addWeighted(frame, 1.0, overlay, float(self.config.alpha), 0)
