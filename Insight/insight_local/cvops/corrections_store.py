"""Local JSONL store for human-flagged detection corrections.

Each correction records (a) the frame snapshot, (b) what the model produced,
(c) what the human says is correct. Stored under assets/corrections/<video>/
so corrections are namespaced per video and trivial to inspect by hand.

Correction layout on disk:
    assets/corrections/<video_stem>/
        corrections.jsonl       # one JSON object per line (append-only)
        <correction_id>.jpg     # frame snapshot, RGB JPEG

Export to YOLO dataset:
    export_yolo_dataset([...], dest, classes=[...])
        dest/
            data.yaml
            images/<correction_id>.jpg
            labels/<correction_id>.txt   # YOLO format: class_id cx cy w h (norm)
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional


_STORE_ROOT = Path(__file__).resolve().parents[2] / "assets" / "corrections"


def _safe_stem(name: str) -> str:
    """Filesystem-safe directory name derived from a video filename."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()) or "untitled"
    return s[:128]


def get_store_root() -> Path:
    """Return the corrections store root, creating it on first call."""
    _STORE_ROOT.mkdir(parents=True, exist_ok=True)
    return _STORE_ROOT


def video_dir(video_path: str) -> Path:
    """Per-video subdirectory inside the store."""
    stem = _safe_stem(Path(video_path).stem) if video_path else "untitled"
    d = get_store_root() / stem
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Correction:
    """A single human-flagged correction.

    `model_detections` and `ground_truth` are lists of dicts with keys:
        label (str), x1, y1, x2, y2 (pixels in original frame coords),
        and for model_detections also `conf` (float). Pixel coords let us
        reconstitute the YOLO normalized format on export.
    """

    id: str
    created_at: float  # unix ts
    video_path: str
    frame_ts_ms: int
    frame_w: int
    frame_h: int
    model_path: str
    model_classes: list[str]
    model_detections: list[dict]
    ground_truth: list[dict]
    image_filename: str  # relative to video_dir(video_path)
    notes: str = ""
    kind: str = ""  # informational summary: "fn" / "fp" / "relabel" / "mixed"

    @staticmethod
    def new(
        *,
        video_path: str,
        frame_ts_ms: int,
        frame_w: int,
        frame_h: int,
        model_path: str,
        model_classes: list[str],
        model_detections: list[dict],
        ground_truth: list[dict],
        notes: str = "",
        kind: str = "",
    ) -> "Correction":
        cid = uuid.uuid4().hex[:16]
        return Correction(
            id=cid,
            created_at=time.time(),
            video_path=str(video_path),
            frame_ts_ms=int(frame_ts_ms),
            frame_w=int(frame_w),
            frame_h=int(frame_h),
            model_path=str(model_path),
            model_classes=[str(c) for c in model_classes],
            model_detections=list(model_detections or []),
            ground_truth=list(ground_truth or []),
            image_filename=f"{cid}.jpg",
            notes=str(notes or ""),
            kind=str(kind or ""),
        )


def _jsonl_path(vdir: Path) -> Path:
    return vdir / "corrections.jsonl"


def append_correction(c: Correction) -> Path:
    """Persist a correction. Caller is responsible for writing the JPEG
    snapshot to image_path(c) BEFORE calling this so a reader never sees a
    record without a backing image.
    """
    vdir = video_dir(c.video_path)
    line = json.dumps(asdict(c), ensure_ascii=False, separators=(",", ":"))
    path = _jsonl_path(vdir)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path


def image_path(c: Correction) -> Path:
    return video_dir(c.video_path) / c.image_filename


def load_corrections(video_path: Optional[str] = None) -> list[Correction]:
    """Load corrections for one video, or all videos when video_path is None.

    Records that fail to parse are skipped silently — the JSONL is
    append-only, but a partially-written tail line shouldn't poison the
    whole list.
    """
    out: list[Correction] = []
    roots: list[Path] = []
    if video_path:
        roots.append(video_dir(video_path))
    else:
        if _STORE_ROOT.exists():
            roots = [p for p in _STORE_ROOT.iterdir() if p.is_dir()]
    for d in roots:
        f = _jsonl_path(d)
        if not f.exists():
            continue
        for raw in f.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                out.append(Correction(**obj))
            except Exception:
                continue
    out.sort(key=lambda c: c.created_at)
    return out


def delete_correction(c: Correction) -> None:
    """Remove a correction from the JSONL and delete its snapshot.

    Implemented as a rewrite-without-this-id of the JSONL, since the file is
    otherwise append-only. Volume is expected to be small (hundreds, not
    millions), so the cost is fine.
    """
    vdir = video_dir(c.video_path)
    f = _jsonl_path(vdir)
    if f.exists():
        keep: list[str] = []
        for raw in f.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                if obj.get("id") == c.id:
                    continue
            except Exception:
                pass
            keep.append(raw)
        f.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    img = image_path(c)
    if img.exists():
        try:
            img.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# YOLO export
# ---------------------------------------------------------------------------


def _to_yolo_line(cls_id: int, box: dict, frame_w: int, frame_h: int) -> Optional[str]:
    if frame_w <= 0 or frame_h <= 0:
        return None
    try:
        x1 = float(box["x1"]); y1 = float(box["y1"])
        x2 = float(box["x2"]); y2 = float(box["y2"])
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    cx = ((x1 + x2) / 2.0) / frame_w
    cy = ((y1 + y2) / 2.0) / frame_h
    bw = (x2 - x1) / frame_w
    bh = (y2 - y1) / frame_h
    return f"{int(cls_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def export_yolo_dataset(
    corrections: Iterable[Correction],
    dest: Path,
    classes: Optional[list[str]] = None,
) -> dict:
    """Export the given corrections as a YOLOv8-style dataset folder.

    If `classes` is None, the union of all labels found in ground_truth (in
    first-seen order) is used. Returns a small summary dict for the caller
    to surface in the UI.
    """
    dest = Path(dest)
    images_dir = dest / "images"
    labels_dir = dest / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    items = list(corrections)
    if classes is None:
        seen: list[str] = []
        for c in items:
            for b in c.ground_truth:
                lbl = str(b.get("label", "")).strip()
                if lbl and lbl not in seen:
                    seen.append(lbl)
        classes = seen
    cls_index = {name: i for i, name in enumerate(classes)}

    written = 0
    skipped = 0
    for c in items:
        src = image_path(c)
        if not src.exists():
            skipped += 1
            continue
        gt_lines: list[str] = []
        for b in c.ground_truth:
            lbl = str(b.get("label", "")).strip()
            if lbl not in cls_index:
                continue
            line = _to_yolo_line(cls_index[lbl], b, c.frame_w, c.frame_h)
            if line:
                gt_lines.append(line)
        if not gt_lines:
            skipped += 1
            continue
        shutil.copy2(src, images_dir / src.name)
        (labels_dir / f"{c.id}.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
        written += 1

    yaml_lines = [
        f"path: {dest.resolve()}",
        "train: images",
        "val: images",
        f"nc: {len(classes)}",
        "names:",
    ]
    for i, name in enumerate(classes):
        yaml_lines.append(f"  {i}: {name}")
    (dest / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    return {
        "written": written,
        "skipped": skipped,
        "classes": list(classes),
        "dest": str(dest.resolve()),
    }
