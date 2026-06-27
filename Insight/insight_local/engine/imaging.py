from __future__ import annotations
import base64
import cv2
import numpy as np

from ..config import PREVIEW_QUALITY, FOCUS_MAX_DIM, CLASS_PRIORITY
from ..runtime_profile import pick_torch_device


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pick_device() -> str:
    return pick_torch_device()


def render_status_frame(message: str, width: int = 1280, height: int = 720) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    gradient = np.linspace(22, 3, height, dtype=np.uint8)
    frame[:, :, 0] = gradient[:, None]
    frame[:, :, 1] = (gradient[:, None] * 2.5).astype(np.uint8)
    frame[:, :, 2] = (gradient[:, None] * 4.5).astype(np.uint8)
    cv2.putText(frame, "INSIGHT", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 240, 190), 3, cv2.LINE_AA)
    wrapped: list[str] = []
    words = message.split()
    line: list[str] = []
    for word in words:
        candidate = " ".join(line + [word])
        if len(candidate) > 48 and line:
            wrapped.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        wrapped.append(" ".join(line))
    y = 180
    for text in wrapped[:6]:
        cv2.putText(frame, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (190, 255, 255), 2, cv2.LINE_AA)
        y += 48
    return frame


def resize_for_encoding(image: np.ndarray, max_dim: int) -> np.ndarray:
    if image is None or image.size == 0:
        return image
    height, width = image.shape[:2]
    largest_dim = max(height, width)
    if largest_dim <= max_dim:
        return image
    scale = max_dim / float(largest_dim)
    return cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)


def encode_jpeg_base64(image: np.ndarray, max_dim: int, quality: int = PREVIEW_QUALITY) -> str:
    image = resize_for_encoding(image, max_dim)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return base64.b64encode(encoded).decode("ascii")


def encode_png_base64(image: np.ndarray, max_dim: int) -> str:
    image = resize_for_encoding(image, max_dim)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return ""
    return base64.b64encode(encoded).decode("ascii")


def crop_with_padding(
    frame: np.ndarray, bbox: tuple[int, int, int, int], pad_ratio: float = 0.12,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
    pad_x, pad_y = int(box_w * pad_ratio), int(box_h * pad_ratio)
    crop_x1, crop_y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    crop_x2, crop_y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    inner_bbox = (x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1)
    return crop, (crop_x1, crop_y1, crop_x2, crop_y2), inner_bbox


def build_focus_silhouette(crop: np.ndarray, inner_bbox: tuple[int, int, int, int]) -> str:
    if crop is None or crop.size == 0:
        return ""
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    ix1, iy1, ix2, iy2 = inner_bbox
    ix1, iy1 = max(0, ix1), max(0, iy1)
    ix2, iy2 = min(crop.shape[1], ix2), min(crop.shape[0], iy2)
    if ix2 - ix1 < 4 or iy2 - iy1 < 4:
        return ""
    roi = crop[iy1:iy2, ix1:ix2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 135)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 40]
    if not contours:
        return ""
    roi_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(roi_mask, [largest], -1, 255, thickness=-1)
    roi_mask = cv2.GaussianBlur(roi_mask, (9, 9), 0)
    mask[iy1:iy2, ix1:ix2] = roi_mask
    overlay = np.zeros((crop.shape[0], crop.shape[1], 4), dtype=np.uint8)
    overlay[:, :, 0] = 255
    overlay[:, :, 1] = 235
    overlay[:, :, 2] = 70
    overlay[:, :, 3] = (mask.astype(np.float32) * 0.72).astype(np.uint8)
    return encode_png_base64(overlay, FOCUS_MAX_DIM)


def downsample_gray(image: np.ndarray, side: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)


def center_of_bbox(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def class_priority(label: str) -> float:
    return CLASS_PRIORITY.get(label, 0.45)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def describe_track_event(label: str, tag: str) -> str:
    nice_label = label.replace("_", " ")
    if tag == "new":
        return f"{nice_label} entered scene"
    if tag == "moving":
        return f"{nice_label} moving"
    if tag == "persistent":
        return f"{nice_label} persisted in view"
    if tag == "focused":
        return f"Focused {nice_label}"
    return f"{nice_label} updated"
