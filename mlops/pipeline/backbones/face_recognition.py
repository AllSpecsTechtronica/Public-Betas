"""face_recognition.py — Face recognition backbone: build a face gallery artifact.

This "training" mode builds a local face-gallery database from an ImageFolder-
style dataset, intended for use with the Insight local recognition worker.

Dataset layout (recommended):
  database/<slug>/
    train/
      alice/
        img1.jpg
      bob/
        img2.jpg
    val/...

The training run writes:
  mlops/models/<scenario>/vN/gallery.db
  mlops/models/<scenario>/vN/metrics.json

And updates the scenario YAML `weights` field to point at the latest gallery.db,
so scenario status can reflect readiness.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any
import threading
import sys
from threading import Event

import yaml

from ..backbone import BackboneBase, BackboneCell, BackboneContext, CellResult
from ..registry import REPO_ROOT


_SPLIT_DIR_ALIASES: list[tuple[str, str]] = [
    ("train", "train"),
    ("val", "val"),
    ("valid", "val"),
    ("test", "test"),
]


def _next_run_dir(models_root: Path) -> Path:
    runs = [p for p in models_root.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
    if not runs:
        return models_root / "v1"
    latest = max(int(p.name[1:]) for p in runs)
    return models_root / f"v{latest + 1}"


def _resolve_models_root(scenario_name: str) -> Path:
    return (REPO_ROOT / "mlops" / "models" / scenario_name).resolve()


_SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_latest_gallery_db(scenario_name: str) -> Path | None:
    """Return the gallery.db from the highest-numbered vN run that has one."""
    models_root = _resolve_models_root(scenario_name)
    if not models_root.exists():
        return None
    runs = [p for p in models_root.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
    if not runs:
        return None
    # Walk from highest vN downward — the highest dir may be empty (just created).
    for run_dir in sorted(runs, key=lambda p: int(p.name[1:]), reverse=True):
        gallery = run_dir / "gallery.db"
        if gallery.exists():
            return gallery
    return None


def _iter_identity_folders(dataset_root: Path) -> list[tuple[str, str, Path]]:
    """Return list of (split, identity_name, folder_path)."""
    base = dataset_root.resolve()
    split_dirs: list[tuple[str, Path]] = []
    for dirname, canon in _SPLIT_DIR_ALIASES:
        p = base / dirname
        if p.is_dir():
            split_dirs.append((canon, p))
    if not split_dirs:
        split_dirs = [("root", base)]

    out: list[tuple[str, str, Path]] = []
    for split, root in split_dirs:
        try:
            entries = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
        except Exception:
            entries = []
        for ident_dir in entries:
            name = ident_dir.name.strip()
            if not name or name.startswith("."):
                continue
            out.append((split, name, ident_dir))
    return out


def _update_scenario_weights_yaml(config_path: Path, weights_ref: str) -> None:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario config must be a mapping: {config_path}")
    raw["weights"] = str(weights_ref or "").strip()
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _ensure_insight_local_importable() -> None:
    try:
        import insight_local  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    insight_root = (REPO_ROOT / "Insight").resolve()
    if str(insight_root) not in sys.path:
        sys.path.insert(0, str(insight_root))


class _BuildGalleryCell(BackboneCell):
    name = "Build Face Gallery"
    description = "Ingest identity folders and build a face recognition gallery database"

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        _ensure_insight_local_importable()
        from insight_local.engine.gallery_db import GalleryDB  # local import: heavy deps
        from .. import registry as reg

        cfg = ctx.scenario_config
        bcfg = dict(getattr(cfg, "backbone_config", {}) or {})

        dataset_root = Path(str(getattr(cfg, "dataset_path", ""))).resolve()
        if not dataset_root.exists() or not dataset_root.is_dir():
            raise RuntimeError(f"Dataset folder missing: {dataset_root}")

        fmt = reg.detect_library_dataset_format(dataset_root)
        if fmt not in {reg.LIBRARY_DATASET_FORMAT_IMAGEFOLDER, reg.LIBRARY_DATASET_FORMAT_FACE_CSV}:
            raise RuntimeError(
                f"Face recognition backbone expects ImageFolder or face CSV dataset format; got: {fmt}"
            )

        models_root = _resolve_models_root(cfg.name)
        models_root.mkdir(parents=True, exist_ok=True)
        run_dir = _next_run_dir(models_root)
        run_dir.mkdir(parents=True, exist_ok=True)
        gallery_db_path = run_dir / "gallery.db"

        identities: list[tuple[str, str, Path]] = []
        csv_rows: list[dict[str, str]] = []
        if fmt == reg.LIBRARY_DATASET_FORMAT_IMAGEFOLDER:
            identities = _iter_identity_folders(dataset_root)
            if not identities:
                raise RuntimeError("No identity folders found (expected split/identity/ images).")
        else:
            # face_csv: require a (id,label) CSV and an image directory (Faces/ or Original Images/).
            import csv as _csv
            csv_path = dataset_root / "Dataset.csv"
            if not csv_path.exists():
                # best-effort: first .csv in root.
                try:
                    for p in sorted(dataset_root.iterdir(), key=lambda x: x.name.lower()):
                        if p.is_file() and p.suffix.lower() == ".csv":
                            csv_path = p
                            break
                except Exception:
                    pass
            if not csv_path.exists():
                raise RuntimeError("face_csv dataset is missing a root-level CSV (expected Dataset.csv).")
            try:
                with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                    reader = _csv.DictReader(f)
                    for row in reader:
                        if not isinstance(row, dict):
                            continue
                        fid = str(row.get("id") or "").strip()
                        label = str(row.get("label") or "").strip()
                        if fid and label:
                            csv_rows.append({"id": fid, "label": label})
                        if len(csv_rows) >= 500000:
                            break
            except Exception as exc:
                raise RuntimeError(f"Unable to read CSV: {csv_path} ({exc})") from exc
            if not csv_rows:
                raise RuntimeError("CSV had no usable rows (expected columns: id,label).")

        max_identities = bcfg.get("max_identities")
        try:
            max_identities_i = int(max_identities) if max_identities is not None else None
        except Exception:
            max_identities_i = None
        max_samples = bcfg.get("max_samples")
        try:
            max_samples_i = int(max_samples) if max_samples is not None else None
        except Exception:
            max_samples_i = None

        if fmt == reg.LIBRARY_DATASET_FORMAT_IMAGEFOLDER:
            if max_identities_i is not None and max_identities_i > 0:
                identities = identities[: max_identities_i]
            print(f"dataset: {dataset_root.name}  identities: {len(identities)}  run: {run_dir.name}")
        else:
            print(f"dataset: {dataset_root.name}  samples: {len(csv_rows)}  run: {run_dir.name}")

        gallery = GalleryDB(gallery_db_path)
        errors_total: list[str] = []
        skipped_total = 0
        last_progress_log = 0.0

        def _progress_cb(i: int, n: int, fname: str) -> None:
            nonlocal last_progress_log
            now = time.perf_counter()
            is_checkpoint = i == 0 or (i + 1) == n or (i + 1) % 5 == 0
            if not is_checkpoint and (now - last_progress_log) < 4.0:
                return
            last_progress_log = now
            print(f"  - {fname}  ({i + 1}/{n})")

        try:
            if fmt == reg.LIBRARY_DATASET_FORMAT_IMAGEFOLDER:
                for split, identity, folder in identities:
                    print(f"[{split}] ingest: {identity}  ({folder})")
                    _added, errs = gallery.ingest_folder(
                        folder,
                        identity_name=identity,
                        group_name=split,
                        progress_cb=_progress_cb,
                    )
                    if errs:
                        skipped_total += len(errs)
                        errors_total.extend(str(e) for e in errs[:5])
            else:
                # Index candidate image directories by filename for fast lookup.
                img_dirs = [
                    dataset_root / "Faces",
                    dataset_root / "Faces" / "Faces",
                    dataset_root / "Original Images",
                    dataset_root / "Original Images" / "Original Images",
                ]
                file_map: dict[str, Path] = {}
                import os as _os

                for base in img_dirs:
                    if not base.is_dir():
                        continue
                    for root, _dirs, files in _os.walk(base):
                        for fn in files:
                            low = fn.lower()
                            if low.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")) and fn not in file_map:
                                file_map[fn] = Path(root) / fn
                        if len(file_map) >= 500000:
                            break
                    if len(file_map) >= 500000:
                        break

                seen_identities: set[str] = set()
                last_row_log = 0.0
                for i, row in enumerate(csv_rows):
                    if max_samples_i is not None and max_samples_i > 0 and i >= max_samples_i:
                        break
                    fid = row["id"]
                    label = row["label"]
                    if max_identities_i is not None and max_identities_i > 0:
                        if label not in seen_identities and len(seen_identities) >= max_identities_i:
                            continue
                        seen_identities.add(label)
                    p = file_map.get(fid)
                    if p is None:
                        skipped_total += 1
                        if len(errors_total) < 8:
                            errors_total.append(f"missing image: {fid}")
                        continue
                    ok, msg = gallery.ingest_single(p, identity_name=label, group_name="root")
                    if not ok:
                        skipped_total += 1
                        if len(errors_total) < 8:
                            errors_total.append(f"{fid}: {msg}")
                    now = time.perf_counter()
                    if (i + 1) == 1 or (i + 1) % 50 == 0 or (now - last_row_log) >= 4.0:
                        last_row_log = now
                        print(f"  - rows: {i + 1}/{len(csv_rows)}  identities: {len(seen_identities) or '?'}")

            matrix_n = int(gallery.build_matrix() or 0)
            stats = gallery.get_stats()
        finally:
            try:
                gallery.close()
            except Exception:
                pass

        payload = {
            "trained_at": time.time(),
            "run": run_dir.name,
            "dataset": str(dataset_root),
            "dataset_slug": str(getattr(cfg, "dataset", "")),
            "identity_count": int(getattr(stats, "identity_count", 0) or 0),
            "image_count": int(getattr(stats, "image_count", 0) or 0),
            "matrix_rows": matrix_n,
            "groups": list(getattr(stats, "group_names", []) or []),
            "skipped_images": int(skipped_total),
            "error_count": int(len(errors_total)),
            "errors_sample": errors_total[:8],
            "backbone_type": "face_recognition",
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # Update scenario YAML to reference this run's weights.
        try:
            weights_ref = run_dir.resolve().relative_to(REPO_ROOT).as_posix() + "/gallery.db"
        except Exception:
            weights_ref = str(gallery_db_path)
        _update_scenario_weights_yaml(Path(str(getattr(cfg, "config_path"))), weights_ref)

        summary = f"Built gallery with {payload['identity_count']} identities ({payload['image_count']} images)."
        if skipped_total:
            summary += f" Skipped {skipped_total} unusable images."
        print(summary)
        return CellResult(
            cell_name=self.name,
            status="done",
            output="",
            elapsed_ms=0,
            data={
                "result_path": str(run_dir),
                "weights_path": str(gallery_db_path),
                "model_version": run_dir.name,
                "summary": summary,
                "metrics": payload,
            },
        )


class _IncrementalUpdateCell(BackboneCell):
    """Copy the latest gallery.db into a new vN+1 dir and ingest only images
    whose source paths are not already tracked in the database.

    Dataset layout is identical to the full-build cell (ImageFolder):
      dataset/<split>/<identity>/img1.jpg ...

    To add a new person or new photos: drop them into the dataset folder and
    run a train job on a scenario that has ``backbone_config.incremental: true``.
    The cell skips any file already recorded in ``identities.source_path``.
    """

    name = "Incremental Gallery Update"
    description = "Ingest only new identity images into the existing gallery"

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        _ensure_insight_local_importable()
        from insight_local.engine.gallery_db import GalleryDB
        from .. import registry as reg

        cfg = ctx.scenario_config
        bcfg = dict(getattr(cfg, "backbone_config", {}) or {})

        dataset_root = Path(str(getattr(cfg, "dataset_path", ""))).resolve()
        if not dataset_root.exists() or not dataset_root.is_dir():
            raise RuntimeError(f"Dataset folder missing: {dataset_root}")

        fmt = reg.detect_library_dataset_format(dataset_root)
        if fmt != reg.LIBRARY_DATASET_FORMAT_IMAGEFOLDER:
            raise RuntimeError(
                "Incremental update only supports ImageFolder dataset format."
            )

        models_root = _resolve_models_root(cfg.name)
        models_root.mkdir(parents=True, exist_ok=True)
        run_dir = _next_run_dir(models_root)
        run_dir.mkdir(parents=True, exist_ok=True)
        gallery_db_path = run_dir / "gallery.db"

        # [base] Copy latest gallery.db so we build on top of prior work.
        latest_db = _resolve_latest_gallery_db(cfg.name)
        if latest_db is not None and latest_db.parent != run_dir:
            base_version = latest_db.parent.name
            print(f"[incremental] base={base_version} -> {run_dir.name}  ({latest_db})")
            shutil.copy2(str(latest_db), str(gallery_db_path))
            for wal_suffix in ("-wal", "-shm"):
                src_extra = Path(str(latest_db) + wal_suffix)
                if src_extra.exists():
                    shutil.copy2(str(src_extra), str(gallery_db_path) + wal_suffix)
        else:
            base_version = None
            print(f"[incremental] no prior gallery found — building fresh: {run_dir.name}")

        gallery = GalleryDB(gallery_db_path)

        # [dedup] Collect every source path already in the DB to skip re-ingestion.
        known_sources: set[str] = gallery.get_all_source_paths()
        print(f"[incremental] already-known images: {len(known_sources)}")

        identities = _iter_identity_folders(dataset_root)
        if not identities:
            try:
                gallery.close()
            except Exception:
                pass
            raise RuntimeError("No identity folders found in dataset.")

        added_total = 0
        skipped_known = 0
        skipped_bad = 0
        errors_total: list[str] = []
        new_identity_names: set[str] = set()
        last_progress_log = 0.0

        def _progress_cb(i: int, n: int, fname: str) -> None:
            nonlocal last_progress_log
            now = time.perf_counter()
            if i != 0 and (i + 1) != n and (i + 1) % 5 != 0 and (now - last_progress_log) < 4.0:
                return
            last_progress_log = now
            print(f"  - {fname}  ({i + 1}/{n})")

        for split, identity, folder in identities:
            all_images = sorted(
                p for p in folder.iterdir() if p.suffix.lower() in _SUPPORTED_IMAGE_EXTS
            )
            new_images = [p for p in all_images if str(p) not in known_sources]
            skipped_known += len(all_images) - len(new_images)
            if not new_images:
                continue

            print(f"[{split}] update: {identity}  +{len(new_images)} new / {len(all_images)} total")
            new_identity_names.add(identity)
            for idx, img_path in enumerate(new_images):
                _progress_cb(idx, len(new_images), img_path.name)
                ok, msg = gallery.ingest_single(img_path, identity_name=identity, group_name=split)
                if ok:
                    added_total += 1
                else:
                    skipped_bad += 1
                    if len(errors_total) < 8:
                        errors_total.append(f"{img_path.name}: {msg}")

        if added_total > 0:
            matrix_n = int(gallery.build_matrix() or 0)
            print(f"[incremental] matrix rebuilt: {matrix_n} rows")
        else:
            print("[incremental] no new images — gallery unchanged")

        stats = gallery.get_stats()
        try:
            gallery.close()
        except Exception:
            pass

        payload = {
            "trained_at": time.time(),
            "run": run_dir.name,
            "base_version": base_version,
            "dataset": str(dataset_root),
            "dataset_slug": str(getattr(cfg, "dataset", "")),
            "identity_count": int(getattr(stats, "identity_count", 0) or 0),
            "image_count": int(getattr(stats, "image_count", 0) or 0),
            "new_images_added": added_total,
            "new_identities": sorted(new_identity_names),
            "skipped_already_known": skipped_known,
            "skipped_unusable": skipped_bad,
            "error_count": len(errors_total),
            "errors_sample": errors_total[:8],
            "backbone_type": "face_recognition",
            "incremental": True,
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        try:
            weights_ref = run_dir.resolve().relative_to(REPO_ROOT).as_posix() + "/gallery.db"
        except Exception:
            weights_ref = str(gallery_db_path)
        _update_scenario_weights_yaml(Path(str(getattr(cfg, "config_path"))), weights_ref)

        if added_total > 0:
            summary = (
                f"Incremental update: +{added_total} images "
                f"({len(new_identity_names)} identities touched), "
                f"{payload['identity_count']} total identities."
            )
        else:
            summary = (
                f"No new images found. Gallery unchanged: "
                f"{payload['identity_count']} identities / {payload['image_count']} images."
            )
        if skipped_bad:
            summary += f" Skipped {skipped_bad} unusable."
        print(summary)

        return CellResult(
            cell_name=self.name,
            status="done",
            output="",
            elapsed_ms=0,
            data={
                "result_path": str(run_dir),
                "weights_path": str(gallery_db_path),
                "model_version": run_dir.name,
                "summary": summary,
                "metrics": payload,
            },
        )


class _InferNotSupportedCell(BackboneCell):
    name = "Recognize Face"
    description = "Run face recognition against the trained gallery database"

    def _emit_log(self, ctx: BackboneContext, phase: str, message: str) -> None:
        try:
            ctx.cell_callback(
                {
                    "type": "log",
                    "phase": str(phase or "").strip(),
                    "message": str(message or "").strip(),
                    "ts": round(time.time(), 3),
                }
            )
        except Exception:
            pass

    def _run_with_heartbeats(
        self,
        ctx: BackboneContext,
        *,
        phase: str,
        label: str,
        interval_s: float = 2.0,
        fn,
    ):
        done = Event()
        box: dict[str, Any] = {"result": None, "exc": None}
        started = time.perf_counter()

        def _worker() -> None:
            try:
                box["result"] = fn()
            except Exception as exc:
                box["exc"] = exc
            finally:
                done.set()

        t = threading.Thread(target=_worker, daemon=True, name=f"FaceRec-{label}")
        t.start()
        while not done.wait(timeout=max(0.25, float(interval_s))):
            elapsed = round(time.perf_counter() - started, 2)
            self._emit_log(ctx, phase, f"{label}: still running ({elapsed}s elapsed)")
        if box["exc"] is not None:
            raise box["exc"]
        return box["result"]

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        _ensure_insight_local_importable()
        from insight_local.engine.gallery_db import GalleryDB
        from insight_local.engine.recognizer import cosine_search, decide_identity, prototype_search
        from .. import registry as reg

        if ctx.image_bgr is None:
            raise RuntimeError("No image provided for face recognition")

        cfg = ctx.scenario_config
        bcfg = dict(getattr(cfg, "backbone_config", {}) or {})
        threshold = float(bcfg.get("threshold", 0.72))
        margin_threshold = float(bcfg.get("margin_threshold", 0.045))
        top_k = int(bcfg.get("top_k", 5))
        self._emit_log(ctx, "config", f"threshold={threshold:.3f}  margin_threshold={margin_threshold:.3f}  top_k={top_k}")

        # Pick gallery db path based on requested version (vN) if provided.
        version = str(ctx.payload.get("version") or "").strip()
        gallery_path = Path(str(getattr(cfg, "weights_path", ""))).resolve()
        if version:
            run_dir = reg.resolve_scenario_run_dir(cfg.name, version)
            if run_dir is not None:
                candidate = (run_dir / "gallery.db").resolve()
                if candidate.exists():
                    gallery_path = candidate

        if not gallery_path.exists() or not gallery_path.is_file():
            raise RuntimeError(f"Gallery DB not found: {gallery_path}")

        self._emit_log(ctx, "gallery", f"opening gallery: {gallery_path}")
        gallery = self._run_with_heartbeats(
            ctx,
            phase="gallery",
            label="load_gallery",
            fn=lambda: _get_cached_gallery(gallery_path),
        )
        self._emit_log(ctx, "face", "extracting face embedding")
        sample = self._run_with_heartbeats(
            ctx,
            phase="face",
            label="extract_face",
            fn=lambda: gallery.extract_face(ctx.image_bgr),
        )
        if sample is None:
            msg = "No usable face detected"
            self._emit_log(ctx, "face", msg)
            print(msg)
            return CellResult(cell_name=self.name, status="error", output=msg, elapsed_ms=0, data={})

        matrix = gallery.matrix
        profile_matrix = gallery.profile_matrix
        if (
            matrix is None
            or matrix.shape[0] == 0
            or profile_matrix is None
            or profile_matrix.shape[0] == 0
        ):
            msg = "Gallery empty (no embeddings). Run training first."
            self._emit_log(ctx, "gallery", msg)
            print(msg)
            return CellResult(cell_name=self.name, status="error", output=msg, elapsed_ms=0, data={})

        self._emit_log(ctx, "search", f"running cosine search (matrix_rows={int(matrix.shape[0])})")
        matches = cosine_search(
            query=sample.feature,
            gallery_matrix=matrix,
            identity_labels=gallery.matrix_labels,
            group_labels=gallery.matrix_groups,
            source_paths=gallery.matrix_sources,
            top_k=top_k,
            threshold=threshold,
        )
        self._emit_log(ctx, "search", "running prototype search")
        profile_matches = prototype_search(
            query=sample.feature,
            profile_matrix=profile_matrix,
            identity_labels=gallery.profile_labels,
            group_labels=gallery.profile_groups,
            source_paths=gallery.profile_sources,
            sample_counts=gallery.profile_sample_counts,
            top_k=3,
        )
        self._emit_log(ctx, "decision", "deciding identity")
        identity, confidence, decision = decide_identity(
            raw_matches=matches,
            profile_matches=profile_matches,
            threshold=threshold,
            margin_threshold=max(0.01, min(0.2, float(margin_threshold))),
        )

        summary = f"identity={identity}  conf={confidence:.3f}"
        if decision.get("reason"):
            summary += f"  ({decision.get('reason')})"
        self._emit_log(ctx, "result", summary)
        print(summary)
        detections = [
            {"label": m.identity, "confidence": float(m.similarity), "track_id": "", "bbox": []}
            for m in matches[:top_k]
        ]
        metrics = {
            "identity": identity,
            "confidence": float(confidence),
            "threshold": float(threshold),
            "face_quality": float(sample.quality),
            "face_detection_score": float(sample.detection_score),
            "decision": dict(decision),
        }
        return CellResult(
            cell_name=self.name,
            status="done",
            output="",
            elapsed_ms=0,
            data={
                "model_version": str(version or ""),
                "weights_path": str(gallery_path),
                "summary": summary,
                "signal": {"flag": False, "summary": summary, "metrics": metrics},
                "detections": detections,
            },
        )


_GALLERY_CACHE_LOCK = threading.Lock()
_GALLERY_CACHE: dict[str, tuple[int, Any, float]] = {}
_GALLERY_CACHE_CAP = 3


def _get_cached_gallery(path: Path):
    _ensure_insight_local_importable()
    from insight_local.engine.gallery_db import GalleryDB

    p = path.resolve()
    key = str(p)
    try:
        mtime_ns = int(p.stat().st_mtime_ns)
    except Exception:
        mtime_ns = 0
    now = time.time()
    with _GALLERY_CACHE_LOCK:
        hit = _GALLERY_CACHE.get(key)
        if hit is not None and hit[0] == mtime_ns:
            _GALLERY_CACHE[key] = (hit[0], hit[1], now)
            return hit[1]
        # Evict LRU
        if len(_GALLERY_CACHE) >= _GALLERY_CACHE_CAP:
            victim = min(_GALLERY_CACHE.items(), key=lambda kv: kv[1][2])[0]
            old = _GALLERY_CACHE.pop(victim, None)
            if old is not None:
                try:
                    old[1].close()
                except Exception:
                    pass
        gallery = GalleryDB(p, read_only=True)
        try:
            gallery.build_matrix()
        except Exception:
            pass
        _GALLERY_CACHE[key] = (mtime_ns, gallery, now)
        return gallery


class FaceRecognitionBackbone(BackboneBase):
    backbone_type = "face_recognition"

    def __init__(self, config: Any) -> None:
        self._config = config
        self._job_type = "infer"

    @property
    def cells(self) -> list[BackboneCell]:
        if self._job_type == "train":
            bcfg = dict(getattr(self._config, "backbone_config", {}) or {})
            if bcfg.get("incremental"):
                return [_IncrementalUpdateCell()]
            return [_BuildGalleryCell()]
        return [_InferNotSupportedCell()]

    def run(self, ctx: BackboneContext) -> dict[str, Any]:
        self._job_type = ctx.job_type
        return super().run(ctx)

    def _build_result(self, ctx: BackboneContext, cell_results: list[CellResult]) -> dict[str, Any]:
        error = ""
        merged: dict[str, Any] = {}
        for r in cell_results:
            if r.status == "error":
                error = r.output or f"Cell '{r.cell_name}' failed"
            if isinstance(r.data, dict):
                merged.update(r.data)

        weights = str(merged.get("weights_path") or "")
        result_path = str(merged.get("result_path") or "")
        summary = str(merged.get("summary") or ("completed" if not error else "failed"))
        signal = merged.get("signal") if isinstance(merged.get("signal"), dict) else None
        if signal is None:
            signal = {"flag": bool(error), "summary": summary, "metrics": merged.get("metrics") or {}}
        detections = merged.get("detections") if isinstance(merged.get("detections"), list) else []

        return {
            "scenario": ctx.scenario_config.name,
            "model_version": str(merged.get("model_version") or ""),
            "weights": weights,
            "result_path": result_path,
            "summary": summary,
            "detections": detections,
            "overlay_image": "",
            "elapsed_ms": sum(r.elapsed_ms for r in cell_results),
            "signal": signal,
            "error": error,
            "artifact_policy": "path_only" if ctx.job_type == "train" else "inline_overlay_optional",
            "backbone_data": {k: v for k, v in merged.items() if k not in {"weights_path", "result_path", "metrics"}},
        }
