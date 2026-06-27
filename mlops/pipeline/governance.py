from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .registry import DATASET_IMAGE_SUFFIXES, MLOPS_ROOT

try:
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover - optional at import time
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]


DATASET_REGISTRY_DIR = MLOPS_ROOT / "dataset_registry"
_YOLO_SPLIT_DIRS = ("train", "valid", "val", "test")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _hash_text(parts: list[str]) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode("utf-8", errors="replace"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _iter_dataset_files(dataset_path: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(dataset_path.rglob("*")):
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        files.append(p)
    return files


def _existing_dirs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen or not path.is_dir():
            continue
        seen.add(resolved)
        out.append(path)
    return out


def _yolo_image_dirs(dataset_path: Path) -> list[Path]:
    return _existing_dirs(
        [dataset_path / "images"]
        + [dataset_path / split / "images" for split in _YOLO_SPLIT_DIRS]
    )


def _yolo_label_dirs(dataset_path: Path) -> list[Path]:
    return _existing_dirs(
        [dataset_path / "labels"]
        + [dataset_path / split / "labels" for split in _YOLO_SPLIT_DIRS]
    )


def _matching_yolo_roots(dataset_path: Path) -> list[tuple[str, Path, Path]]:
    roots: list[tuple[str, Path, Path]] = []
    seen: set[tuple[Path, Path]] = set()

    def _add(split: str, images: Path, labels: Path) -> None:
        if not (images.is_dir() and labels.is_dir()):
            return
        try:
            key = (images.resolve(), labels.resolve())
        except Exception:
            key = (images, labels)
        if key in seen:
            return
        seen.add(key)
        roots.append((split, images, labels))

    _add("root", dataset_path / "images", dataset_path / "labels")
    for split in _YOLO_SPLIT_DIRS:
        _add(split, dataset_path / split / "images", dataset_path / split / "labels")
    return roots


_MAX_FULL_FILE_HASH_BYTES = 64 * 1024 * 1024


def file_content_sha256(path: Path) -> str:
    """Stable SHA-256 over file bytes (full read for files up to 64 MiB, else head+size+tail)."""
    hasher = hashlib.sha256()
    try:
        size = int(path.stat().st_size)
    except Exception:
        return "missing"
    try:
        if size <= _MAX_FULL_FILE_HASH_BYTES:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        with path.open("rb") as handle:
            hasher.update(handle.read(4 * 1024 * 1024))
            hasher.update(str(size).encode("utf-8", errors="replace"))
            if size > 8 * 1024 * 1024:
                try:
                    handle.seek(max(0, size - 4 * 1024 * 1024))
                    hasher.update(handle.read(4 * 1024 * 1024))
                except Exception:
                    pass
        return hasher.hexdigest()
    except Exception:
        return "unreadable"


def infer_dataset_lineage(dataset_path: Path) -> dict[str, Any]:
    """Heuristic lineage evidence for the dataset library.

    This intentionally returns useful nodes even when there are no transform
    edges, so dashboard lineage sections have something concrete to render for
    every dataset shape.
    """
    edges: list[dict[str, str]] = []
    nodes: list[dict[str, str]] = []
    root = dataset_path.resolve()

    def _add_node(node_id: str, label: str, evidence: str) -> None:
        if any(n.get("id") == node_id for n in nodes):
            return
        nodes.append({"id": node_id, "label": label, "evidence": evidence})

    _add_node("dataset_root", "Dataset root", str(root))
    if (dataset_path / "raw").is_dir():
        _add_node("raw", "Raw inputs", "raw/ directory present")
        _add_node("curated", "Curated dataset", "dataset contains curated assets")
        edges.append({"from": "raw", "to": "curated", "evidence": "raw/ directory present"})

    yolo_roots = _matching_yolo_roots(dataset_path)
    if yolo_roots:
        _add_node("curated", "Curated dataset", "YOLO image/label assets present")
        for split, images, labels in yolo_roots:
            node_id = f"yolo_{split}"
            _add_node(node_id, f"YOLO {split}", f"{images.name}/ and {labels.name}/")
            edges.append({
                "from": "curated",
                "to": node_id,
                "evidence": f"{split}: images={images.relative_to(dataset_path).as_posix()} labels={labels.relative_to(dataset_path).as_posix()}",
            })
    elif (dataset_path / "images").is_dir():
        _add_node("curated", "Curated dataset", "image assets present")
        _add_node("images_only", "Images only", "images/ without labels/")
        edges.append({"from": "curated", "to": "images_only", "evidence": "images/ without labels/"})

    csv_files = [p for p in dataset_path.glob("*.csv") if p.is_file()]
    if csv_files:
        _add_node("tabular_csv", "Tabular CSV", f"{len(csv_files)} CSV file(s)")
        edges.append({"from": "dataset_root", "to": "tabular_csv", "evidence": f"{len(csv_files)} CSV file(s) at root"})

    jsonl_files = [p for p in dataset_path.glob("*.jsonl") if p.is_file()]
    if jsonl_files:
        _add_node("instruction_jsonl", "Instruction JSONL", f"{len(jsonl_files)} JSONL file(s)")
        edges.append({
            "from": "dataset_root",
            "to": "instruction_jsonl",
            "evidence": f"{len(jsonl_files)} JSONL file(s) at root",
        })

    audio_exts = {".wav", ".aiff", ".aif", ".flac", ".mp3", ".m4a", ".ogg"}
    audio_count = 0
    try:
        audio_count = sum(1 for p in dataset_path.rglob("*") if p.is_file() and p.suffix.lower() in audio_exts)
    except Exception:
        audio_count = 0
    if audio_count:
        _add_node("audio_files", "Audio files", f"{audio_count} audio file(s)")
        edges.append({"from": "dataset_root", "to": "audio_files", "evidence": f"{audio_count} audio file(s)"})

    return {"nodes": nodes, "edges": edges, "dataset_root": str(root)}


def _average_hash_int(image_l: Any) -> int:
    pixels = list(image_l.getdata())
    if not pixels:
        return 0
    avg = sum(pixels) / float(len(pixels))
    bits = 0
    for i, p in enumerate(pixels):
        if float(p) >= avg:
            bits |= 1 << i
    return bits


def _pil_has_gps_exif(img: Any) -> bool:
    try:
        exif = img.getexif()
        if exif is None:
            return False
        # Standard EXIF tag for GPS IFD pointer
        return 34853 in exif and exif.get(34853) is not None
    except Exception:
        return False


def audit_dataset_media(dataset_path: Path, *, max_images: int = 500) -> dict[str, Any]:
    """Image integrity (PIL.verify), average-hash near-duplicates, EXIF GPS presence."""
    out: dict[str, Any] = {
        "status": "skipped",
        "corrupt_images": [],
        "corrupt_count": 0,
        "duplicate_groups": [],
        "duplicate_cluster_count": 0,
        "gps_exif_image_count": 0,
        "images_scanned": 0,
        "note": "",
    }
    if Image is None:
        out["status"] = "skipped"
        out["note"] = "Pillow not available; install pillow for media audit."
        return out
    image_dirs = _yolo_image_dirs(dataset_path)
    if not image_dirs:
        out["note"] = "no YOLO images directory"
        return out
    candidates = [
        p
        for image_root in image_dirs
        for p in image_root.rglob("*")
        if p.is_file() and p.suffix.lower() in DATASET_IMAGE_SUFFIXES
    ]
    candidates.sort(key=lambda p: str(p))
    if len(candidates) > max_images:
        candidates = candidates[:max_images]
        out["note"] = f"capped scan at {max_images} images"
    hash_to_paths: dict[int, list[str]] = {}
    corrupt: list[str] = []
    gps_count = 0
    scanned = 0
    for path in candidates:
        rel = ""
        try:
            rel = path.relative_to(dataset_path).as_posix()
        except Exception:
            rel = path.name
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            corrupt.append(rel)
            continue
        try:
            with Image.open(path) as img2:
                if ImageOps is not None:
                    img2 = ImageOps.exif_transpose(img2)
                gray = img2.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
                hval = _average_hash_int(gray)
                hash_to_paths.setdefault(hval, []).append(rel)
                if _pil_has_gps_exif(img2):
                    gps_count += 1
        except Exception:
            corrupt.append(rel)
            continue
        scanned += 1
    dup_groups = [paths for paths in hash_to_paths.values() if len(paths) > 1]
    out.update(
        {
            "status": "ok",
            "corrupt_images": corrupt[:200],
            "corrupt_count": len(corrupt),
            "duplicate_groups": dup_groups[:40],
            "duplicate_cluster_count": len(dup_groups),
            "gps_exif_image_count": gps_count,
            "images_scanned": scanned,
        }
    )
    return out


def scrub_image_exif(path: Path) -> dict[str, Any]:
    """Rewrite a raster image without GPS EXIF; applies EXIF orientation then strips metadata."""
    if Image is None or ImageOps is None:
        raise RuntimeError("Pillow is required for EXIF scrub")
    if path.suffix.lower() not in DATASET_IMAGE_SUFFIXES:
        raise ValueError("unsupported image type for scrub")
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        ext = path.suffix.lower().lstrip(".")
        fmt_map = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP", "bmp": "BMP"}
        pil_fmt = fmt_map.get(ext, (img.format or "PNG").upper())
        save_kw: dict[str, Any] = {}
        if pil_fmt == "JPEG":
            save_kw["quality"] = 92
            save_kw["optimize"] = True
        tmp = path.with_suffix(path.suffix + ".scrub_tmp")
        img.save(tmp, format=pil_fmt, **save_kw)
    tmp.replace(path)
    return {"path": str(path), "status": "scrubbed"}


def validate_dataset_contract(dataset_path: Path, classes: list[str]) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    image_dirs = _yolo_image_dirs(dataset_path)
    label_dirs = _yolo_label_dirs(dataset_path)
    matching_roots = _matching_yolo_roots(dataset_path)
    if not image_dirs:
        issues.append("missing YOLO images directory (expected images/ or train/images)")
    if not label_dirs:
        issues.append("missing YOLO labels directory (expected labels/ or train/labels)")
    if image_dirs and label_dirs and not matching_roots:
        issues.append("image/label directory layout mismatch")
    if not classes:
        issues.append("no classes configured for scenario")
    if len(set(c.lower() for c in classes)) != len(classes):
        warnings.append("class list contains case-insensitive duplicates")
    status = "ok" if not issues else "failed"
    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "layout": "yolo",
        "splits": [split for split, _images, _labels in matching_roots],
    }


def evaluate_dataset_quality(dataset_path: Path, classes: list[str]) -> dict[str, Any]:
    image_count = 0
    label_count = 0
    empty_label_files = 0
    invalid_label_lines = 0
    class_counts: dict[int, int] = {}
    bbox_areas: list[float] = []

    image_paths: set[Path] = set()
    for image_root in _yolo_image_dirs(dataset_path):
        image_paths.update(
            p.resolve()
            for p in image_root.rglob("*")
            if p.is_file() and p.suffix.lower() in DATASET_IMAGE_SUFFIXES
        )
    image_count = len(image_paths)

    label_paths: set[Path] = set()
    for label_root in _yolo_label_dirs(dataset_path):
        label_paths.update(p.resolve() for p in label_root.rglob("*.txt") if p.is_file())
    if label_paths:
        label_files = sorted(label_paths, key=lambda p: str(p))
        label_count = len(label_files)
        for label_path in label_files:
            text = ""
            try:
                text = label_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                invalid_label_lines += 1
                continue
            nonempty = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if not nonempty:
                empty_label_files += 1
                continue
            for ln in nonempty:
                parts = ln.split()
                if len(parts) < 5:
                    invalid_label_lines += 1
                    continue
                cls = _safe_float(parts[0])
                w = _safe_float(parts[3])
                h = _safe_float(parts[4])
                if cls is None or w is None or h is None:
                    invalid_label_lines += 1
                    continue
                cls_idx = int(cls)
                class_counts[cls_idx] = class_counts.get(cls_idx, 0) + 1
                if w <= 0.0 or h <= 0.0:
                    invalid_label_lines += 1
                    continue
                bbox_areas.append(float(w * h))

    class_named_counts = {
        (classes[idx] if 0 <= idx < len(classes) else f"class_{idx}"): count
        for idx, count in sorted(class_counts.items(), key=lambda kv: kv[0])
    }
    area_small = sum(1 for a in bbox_areas if a < 0.02)
    area_medium = sum(1 for a in bbox_areas if 0.02 <= a < 0.15)
    area_large = sum(1 for a in bbox_areas if a >= 0.15)

    quality_score = 100.0
    if image_count > 0 and label_count == 0:
        quality_score -= 45.0
    if invalid_label_lines:
        quality_score -= min(30.0, invalid_label_lines * 0.25)
    if empty_label_files:
        quality_score -= min(15.0, empty_label_files * 0.15)
    quality_score = max(0.0, round(quality_score, 2))

    return {
        "images": image_count,
        "label_files": label_count,
        "empty_label_files": empty_label_files,
        "invalid_label_lines": invalid_label_lines,
        "class_instance_counts": class_named_counts,
        "bbox_area_buckets": {"small": area_small, "medium": area_medium, "large": area_large},
        "quality_score": quality_score,
    }


def create_dataset_snapshot(dataset_name: str, dataset_path: Path, classes: list[str]) -> dict[str, Any]:
    files = _iter_dataset_files(dataset_path)
    file_rows: list[dict[str, Any]] = []
    fingerprint_parts: list[str] = []
    total_bytes = 0
    for p in files:
        try:
            stat = p.stat()
            size = int(stat.st_size)
            mtime_ns = int(stat.st_mtime_ns)
        except Exception:
            size = 0
            mtime_ns = 0
        rel = p.relative_to(dataset_path).as_posix()
        content_sha = file_content_sha256(p)
        total_bytes += size
        file_rows.append({"path": rel, "size": size, "mtime_ns": mtime_ns, "content_sha256": content_sha})
        fingerprint_parts.append(f"{rel}|{size}|{content_sha}")

    contract = validate_dataset_contract(dataset_path, classes)
    quality = evaluate_dataset_quality(dataset_path, classes)
    media_audit = audit_dataset_media(dataset_path)
    quality = {**quality, "media_audit": media_audit}
    lineage = infer_dataset_lineage(dataset_path)
    classes_key = "|".join(classes)
    snapshot_hash = _hash_text(
        [str(dataset_name), classes_key, *fingerprint_parts, json.dumps(contract, sort_keys=True)]
    )
    snapshot_id = f"{dataset_name}:{snapshot_hash[:16]}"
    return {
        "snapshot_id": snapshot_id,
        "dataset": dataset_name,
        "created_at": _utc_now(),
        "dataset_path": str(dataset_path.resolve()),
        "total_files": len(file_rows),
        "total_bytes": total_bytes,
        "classes": list(classes),
        "contract": contract,
        "quality": quality,
        "files": file_rows,
        "snapshot_hash": snapshot_hash,
        "fingerprint_mode": "content_sha256",
        "lineage": lineage,
    }


def persist_dataset_snapshot(snapshot: dict[str, Any]) -> Path:
    dataset = str(snapshot.get("dataset") or "unknown")
    snapshot_id = str(snapshot.get("snapshot_id") or "")
    if not snapshot_id:
        raise ValueError("snapshot_id is required")
    dataset_dir = DATASET_REGISTRY_DIR / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / f"{snapshot_id.replace(':', '__')}.json"
    if not path.exists():
        path.write_text(
            json.dumps(snapshot, indent=2, ensure_ascii=True, default=str),
            encoding="utf-8",
        )
    return path


def load_dataset_snapshot(snapshot_id: str) -> dict[str, Any] | None:
    wanted = str(snapshot_id or "").strip()
    if not wanted:
        return None
    if not DATASET_REGISTRY_DIR.exists():
        return None
    needle = f"{wanted.replace(':', '__')}.json"
    for p in DATASET_REGISTRY_DIR.rglob(needle):
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(data, dict):
            return data
    return None


def dataset_drift_report(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    cur_q = current.get("quality") if isinstance(current.get("quality"), dict) else {}
    base_q = baseline.get("quality") if isinstance(baseline.get("quality"), dict) else {}

    def _class_dist(q: dict[str, Any]) -> dict[str, float]:
        counts = q.get("class_instance_counts")
        if not isinstance(counts, dict):
            return {}
        total = float(sum(float(v or 0.0) for v in counts.values()))
        if total <= 0.0:
            return {}
        return {str(k): float(v or 0.0) / total for k, v in counts.items()}

    cur_dist = _class_dist(cur_q)
    base_dist = _class_dist(base_q)
    keys = sorted(set(cur_dist.keys()) | set(base_dist.keys()))

    # Jensen-Shannon divergence (bounded [0,1] for log2)
    def _kld(p: dict[str, float], q: dict[str, float]) -> float:
        out = 0.0
        for k in keys:
            pv = float(p.get(k, 0.0))
            qv = float(q.get(k, 0.0))
            if pv <= 0.0 or qv <= 0.0:
                continue
            out += pv * math.log2(pv / qv)
        return out

    m = {k: (float(cur_dist.get(k, 0.0)) + float(base_dist.get(k, 0.0))) / 2.0 for k in keys}
    jsd = 0.5 * _kld(cur_dist, m) + 0.5 * _kld(base_dist, m)

    cur_images = int(cur_q.get("images") or 0)
    base_images = int(base_q.get("images") or 0)
    cur_quality = float(cur_q.get("quality_score") or 0.0)
    base_quality = float(base_q.get("quality_score") or 0.0)
    return {
        "current_snapshot_id": str(current.get("snapshot_id") or ""),
        "baseline_snapshot_id": str(baseline.get("snapshot_id") or ""),
        "image_count_delta": cur_images - base_images,
        "quality_score_delta": round(cur_quality - base_quality, 4),
        "class_distribution_jsd": round(float(jsd), 6),
        "drift_level": (
            "high" if jsd >= 0.22 else "medium" if jsd >= 0.12 else "low"
        ),
    }
