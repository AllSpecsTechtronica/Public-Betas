"""Lightweight foreground segmentation for optional TRELLIS enhancement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops


@dataclass
class SegmentResult:
    crop_path: Path
    bbox: tuple[int, int, int, int]
    confidence: float


def segment_foreground(image_path: Path, out_dir: Path, max_objects: int = 1) -> list[SegmentResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from rembg import remove  # type: ignore[import-not-found]
    except Exception:
        return _center_crop(image_path, out_dir, max_objects=max_objects)

    image = Image.open(image_path).convert("RGBA")
    cutout = remove(image)
    alpha = cutout.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return _center_crop(image_path, out_dir, max_objects=max_objects)
    crop = cutout.crop(_pad_bbox(bbox, image.size, pad=0.08))
    path = out_dir / "crop.png"
    crop.save(path)
    coverage = _bbox_coverage(bbox, image.size)
    return [SegmentResult(path, bbox, confidence=max(0.35, min(0.9, coverage * 3.0)))][:max_objects]


def _center_crop(image_path: Path, out_dir: Path, max_objects: int) -> list[SegmentResult]:
    if max_objects <= 0:
        return []
    image = Image.open(image_path).convert("RGBA")
    w, h = image.size
    side = int(min(w, h) * 0.78)
    left = max(0, (w - side) // 2)
    top = max(0, (h - side) // 2)
    bbox = (left, top, min(w, left + side), min(h, top + side))
    crop = image.crop(bbox)
    if crop.getbbox() is None:
        bg = Image.new("RGBA", crop.size, (0, 0, 0, 0))
        crop = ImageChops.lighter(crop, bg)
    path = out_dir / "crop.png"
    crop.save(path)
    return [SegmentResult(path, bbox, confidence=0.35)]


def _pad_bbox(bbox: tuple[int, int, int, int], size: tuple[int, int], pad: float) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    w, h = size
    bw = right - left
    bh = bottom - top
    px = int(bw * pad)
    py = int(bh * pad)
    return (max(0, left - px), max(0, top - py), min(w, right + px), min(h, bottom + py))


def _bbox_coverage(bbox: tuple[int, int, int, int], size: tuple[int, int]) -> float:
    left, top, right, bottom = bbox
    w, h = size
    if w <= 0 or h <= 0:
        return 0.0
    return max(0.0, ((right - left) * (bottom - top)) / float(w * h))
