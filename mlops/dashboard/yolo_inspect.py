from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class YoloLabelSummary:
    label_files_scanned: int
    objects_total: int
    class_counts: dict[int, int]
    parse_errors: list[str]


def _iter_label_files(labels_root: Path, *, max_files: int | None) -> Iterable[Path]:
    base = labels_root.resolve()
    if not base.exists():
        return []
    out: list[Path] = []
    try:
        for p in base.rglob("*.txt"):
            if p.name.startswith("."):
                continue
            out.append(p)
            if max_files is not None and len(out) >= max_files:
                break
    except Exception:
        return out
    return out


def summarize_yolo_labels(labels_root: Path, *, max_files: int = 25_000) -> YoloLabelSummary:
    """Summarize YOLO labels as class-id counts.

    Expected label format per line: "<class_id> <xc> <yc> <w> <h>"
    """
    counter: Counter[int] = Counter()
    objects_total = 0
    parse_errors: list[str] = []
    label_files = list(_iter_label_files(labels_root, max_files=max_files))
    for label_path in label_files:
        try:
            raw = label_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            if len(parse_errors) < 25:
                parse_errors.append(f"{label_path}: {exc}")
            continue
        for i, line in enumerate(raw.splitlines(), start=1):
            ln = line.strip()
            if not ln:
                continue
            parts = ln.split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except Exception:
                if len(parse_errors) < 25:
                    parse_errors.append(f"{label_path}:{i}: invalid class id: {parts[0]!r}")
                continue
            counter[class_id] += 1
            objects_total += 1
    return YoloLabelSummary(
        label_files_scanned=len(label_files),
        objects_total=objects_total,
        class_counts=dict(counter),
        parse_errors=parse_errors,
    )


@dataclass(frozen=True)
class YoloPairCheck:
    images_scanned: int
    labels_scanned: int
    images_total_est: int
    labels_total_est: int
    images_missing_labels: int
    labels_missing_images: int
    examples: list[str]
    truncated: bool


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_yolo_pairs(
    dataset_root: Path,
    *,
    max_examples: int = 30,
    max_images: int = 200_000,
    max_labels: int = 200_000,
) -> YoloPairCheck:
    """Check image/label pairing under YOLO-style dataset roots.

    Large datasets can be expensive to scan; this function caps how many files it
    inspects (defaults: 200k images/labels). When capped, totals are estimates
    based on scanned items only.
    """
    base = dataset_root.resolve()
    images_root = base / "images"
    labels_root = base / "labels"

    def _rel_no_ext(p: Path, root: Path) -> str:
        rel = p.relative_to(root).as_posix()
        return str(Path(rel).with_suffix(""))

    image_keys: set[str] = set()
    label_keys: set[str] = set()
    examples: list[str] = []
    truncated = False

    images_scanned = 0
    if images_root.exists():
        try:
            for p in images_root.rglob("*"):
                if not p.is_file() or p.name.startswith("."):
                    continue
                if p.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                image_keys.add(_rel_no_ext(p, images_root))
                images_scanned += 1
                if images_scanned >= max_images:
                    truncated = True
                    break
        except Exception:
            pass

    labels_scanned = 0
    if labels_root.exists():
        try:
            for p in labels_root.rglob("*.txt"):
                if not p.is_file() or p.name.startswith("."):
                    continue
                label_keys.add(_rel_no_ext(p, labels_root))
                labels_scanned += 1
                if labels_scanned >= max_labels:
                    truncated = True
                    break
        except Exception:
            pass

    missing_labels = sorted(image_keys - label_keys)
    missing_images = sorted(label_keys - image_keys)
    for key in missing_labels[: max_examples // 2]:
        examples.append(f"image missing label: {key}")
    for key in missing_images[: max_examples - len(examples)]:
        examples.append(f"label missing image: {key}")

    return YoloPairCheck(
        images_scanned=images_scanned,
        labels_scanned=labels_scanned,
        images_total_est=images_scanned,
        labels_total_est=labels_scanned,
        images_missing_labels=len(missing_labels),
        labels_missing_images=len(missing_images),
        examples=examples,
        truncated=truncated,
    )
