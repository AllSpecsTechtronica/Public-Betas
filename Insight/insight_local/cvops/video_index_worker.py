from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    from .detection_backends import (
        YuNetFaceDetectorBackend,
        extract_yolo_detections,
        is_yunet_face_detector_model,
    )
except ImportError:
    from detection_backends import (  # type: ignore
        YuNetFaceDetectorBackend,
        extract_yolo_detections,
        is_yunet_face_detector_model,
    )


_EMIT_INTERVAL_S = 0.1
_FALLBACK_HEATMAP_CATEGORIES: dict[str, set[str]] = {
    "human": {"person", "face"},
    "plant": {"potted plant"},
    "animal": {
        "dog", "cat", "bird", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe",
    },
    "tech": {
        "laptop", "cell phone", "keyboard", "mouse", "remote", "tv",
        "microwave", "oven", "toaster",
    },
}
_FALLBACK_LABEL_TO_CATEGORY = {
    label: category
    for category, labels in _FALLBACK_HEATMAP_CATEGORIES.items()
    for label in labels
}
_FALLBACK_QUICK_FILTER_TERMS = {
    "people": ("human", "person"),
    "animals": ("animal",),
    "tech": ("tech",),
    "objects": ("inorganic", "plant", "object"),
}
_HEATMAP_CATEGORY_FUNC: Any = None
_MATCHES_DETECTION_VIEW_FUNC: Any = None
_HELPERS_LOADED = False


def _emit(kind: str, **payload: Any) -> None:
    payload["type"] = kind
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _prepare_runtime_env() -> None:
    cache_root = Path(tempfile.gettempdir()) / "insight-video-index"
    for key, child in (
        ("MPLCONFIGDIR", "matplotlib"),
        ("YOLO_CONFIG_DIR", "ultralytics"),
        ("XDG_CACHE_HOME", "cache"),
    ):
        path = cache_root / child
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception:
            continue
        os.environ.setdefault(key, str(path))


def _load_filter_helpers() -> None:
    global _HEATMAP_CATEGORY_FUNC, _MATCHES_DETECTION_VIEW_FUNC, _HELPERS_LOADED
    if _HELPERS_LOADED:
        return
    _HELPERS_LOADED = True
    try:
        insight_root = Path(__file__).resolve().parents[2]
        if str(insight_root) not in sys.path:
            sys.path.insert(0, str(insight_root))
        from insight_local.config import heatmap_category  # type: ignore
        from insight_local.filtering import matches_detection_view  # type: ignore

        _HEATMAP_CATEGORY_FUNC = heatmap_category
        _MATCHES_DETECTION_VIEW_FUNC = matches_detection_view
    except Exception:
        _HEATMAP_CATEGORY_FUNC = None
        _MATCHES_DETECTION_VIEW_FUNC = None


def _heatmap_category(label: str) -> str:
    _load_filter_helpers()
    if _HEATMAP_CATEGORY_FUNC is not None:
        try:
            return str(_HEATMAP_CATEGORY_FUNC(label))
        except Exception:
            pass
    return _FALLBACK_LABEL_TO_CATEGORY.get(str(label or "").strip().lower(), "inorganic")


def _matches_detection_view(query: str, active_filters: set[str], *parts: object) -> bool:
    _load_filter_helpers()
    if _MATCHES_DETECTION_VIEW_FUNC is not None:
        try:
            return bool(_MATCHES_DETECTION_VIEW_FUNC(query, active_filters, *parts))
        except Exception:
            pass
    haystack = " ".join(str(part or "").strip().lower() for part in parts if str(part or "").strip())
    if query:
        tokens = [part for part in str(query).lower().replace(",", " ").split() if part]
        if tokens and not any(token in haystack for token in tokens):
            return False
    if not active_filters:
        return True
    for key in active_filters:
        for term in _FALLBACK_QUICK_FILTER_TERMS.get(key, (key,)):
            if term in haystack:
                return True
    return False


def _parse_categories(raw: str) -> set[str]:
    return {
        part.strip().lower()
        for part in str(raw or "").replace(";", ",").split(",")
        if part.strip()
    }


def _filter_detections(
    detections: list[dict[str, Any]],
    *,
    label_filter: str,
    categories: set[str],
) -> list[dict[str, Any]]:
    if not label_filter and not categories:
        return detections
    out: list[dict[str, Any]] = []
    for det in detections:
        label = str(det.get("label") or "")
        category = _heatmap_category(label)
        if _matches_detection_view(label_filter, categories, label, category):
            out.append(det)
    return out


def _model_supports_to(model_path: str) -> bool:
    return Path(model_path).suffix.lower() in {".pt", ".torchscript"}


def _resolve_auto_device() -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run_index(
    video_path: str,
    model_path: str,
    sample_every: int,
    device: str,
    *,
    label_filter: str = "",
    categories: set[str] | None = None,
    export_frames_dir: str = "",
    export_file_prefix: str = "idx",
) -> int:
    _prepare_runtime_env()
    categories = set(categories or set())
    export_root: Path | None = None
    export_prefix_safe = re.sub(r"[^a-zA-Z0-9._-]", "_", str(export_file_prefix or "idx").strip()).strip("._-")[
        :72
    ] or "idx"
    raw_export = str(export_frames_dir or "").strip()
    if raw_export:
        export_root = Path(raw_export).expanduser().resolve()
        try:
            export_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            _emit("failed", message=f"cannot create export directory {export_root}: {exc}")
            return 1
    frames_saved = 0
    try:
        import cv2  # type: ignore
    except Exception as exc:
        _emit("failed", message=f"opencv unavailable: {exc}")
        return 1

    resolved_device = device.strip() or _resolve_auto_device()
    backend_kind = "yunet_face" if is_yunet_face_detector_model(model_path) else "yolo"
    model = None
    face_backend: YuNetFaceDetectorBackend | None = None
    supports_to = False
    if backend_kind == "yunet_face":
        _emit("status", message=f"Loading face detector {Path(model_path).name} (OpenCV YuNet)...")
        try:
            face_backend = YuNetFaceDetectorBackend(model_path)
        except Exception as exc:
            _emit("failed", message=f"face detector load failed: {exc}")
            return 1
    else:
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            _emit("failed", message=f"ultralytics unavailable: {exc}")
            return 1
        _emit("status", message=f"Loading model {Path(model_path).name} on {resolved_device}...")
        try:
            model = YOLO(model_path)
        except Exception as exc:
            _emit("failed", message=f"model load failed: {exc}")
            return 1

        supports_to = _model_supports_to(model_path)
        if supports_to:
            try:
                model.to(resolved_device)
            except Exception as exc:
                _emit(
                    "failed",
                    message=(
                        f"could not move model to {resolved_device}: {exc} "
                        "falling back requires re-running with CPU"
                    ),
                )
                return 1

    _emit("status", message="Opening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _emit("failed", message=f"cannot open video: {video_path}")
        return 1

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(sample_every))
        count = 0
        frame_idx = 0
        batch_buf: list[tuple[int, list[dict[str, Any]]]] = []
        last_batch_emit = time.perf_counter()
        last_progress_emit = 0.0
        # Corrupt-footage tracking: spans (start_ms, end_ms) that fail to decode.
        corrupt_buf: list[tuple[int, int]] = []
        last_corrupt_emit = 0.0
        consecutive_fail = 0
        _MAX_CONSEC_FAIL = 240  # give up after a long unbroken run of bad frames

        _emit(
            "status",
            message=f"Scanning {total} frames @ {fps:.1f} fps (sample every {step})...",
        )
        _emit("progress", frame_idx=0, total=total)

        while True:
            if step > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                # Distinguish real EOF from a corrupt/undecodable frame. When we
                # know there are more frames ahead, treat it as a bad span:
                # record it, seek past it, and keep scanning the good parts so
                # one glitch doesn't abort the whole pass.
                if total and frame_idx < total - step:
                    bad_start = int((frame_idx / fps) * 1000.0) if fps > 0 else 0
                    frame_idx += step
                    bad_end = int((frame_idx / fps) * 1000.0) if fps > 0 else bad_start
                    corrupt_buf.append((bad_start, bad_end))
                    consecutive_fail += 1
                    now = time.perf_counter()
                    if corrupt_buf and (now - last_corrupt_emit) >= _EMIT_INTERVAL_S:
                        _emit("corrupt_batch", ranges=corrupt_buf)
                        corrupt_buf = []
                        last_corrupt_emit = now
                    if (now - last_progress_emit) >= _EMIT_INTERVAL_S:
                        _emit("progress", frame_idx=frame_idx, total=total)
                        last_progress_emit = now
                    if consecutive_fail > _MAX_CONSEC_FAIL or frame_idx >= total:
                        break
                    # Force the decoder to resync at the next keyframe.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    continue
                break
            consecutive_fail = 0

            ts_ms = int((frame_idx / fps) * 1000.0) if fps > 0 else 0
            try:
                if face_backend is not None:
                    detections = face_backend.predict(frame)
                else:
                    predict_kwargs: dict[str, Any] = {"verbose": False}
                    if supports_to:
                        predict_kwargs["device"] = resolved_device
                    results = model.predict(frame, **predict_kwargs)
                    fh, fw = frame.shape[:2]
                    detections = extract_yolo_detections(results, fw, fh)
            except Exception as exc:
                _emit("failed", message=f"inference failed at frame {frame_idx}: {exc}")
                return 1

            detections = _filter_detections(
                detections,
                label_filter=label_filter,
                categories=categories,
            )
            if detections:
                if export_root is not None:
                    try:
                        out_path = export_root / f"{export_prefix_safe}_{frame_idx:08d}.jpg"
                        if cv2.imwrite(
                            str(out_path),
                            frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                        ):
                            frames_saved += 1
                    except Exception as exc:
                        _emit("failed", message=f"frame export failed at frame {frame_idx}: {exc}")
                        return 1
                batch_buf.append((ts_ms, detections))
                count += len(detections)

            now = time.perf_counter()
            if batch_buf and (now - last_batch_emit) >= _EMIT_INTERVAL_S:
                _emit("detections_batch", batch=batch_buf)
                batch_buf = []
                last_batch_emit = now
            if (now - last_progress_emit) >= _EMIT_INTERVAL_S:
                _emit("progress", frame_idx=frame_idx, total=total)
                last_progress_emit = now

            frame_idx += step
            if total and frame_idx >= total:
                break

        if batch_buf:
            _emit("detections_batch", batch=batch_buf)
        if corrupt_buf:
            _emit("corrupt_batch", ranges=corrupt_buf)
        _emit("progress", frame_idx=frame_idx, total=total)
        _emit("finished", total_events=count)
        if export_root is not None:
            _emit(
                "export_summary",
                saved=frames_saved,
                directory=str(export_root),
            )
        return 0
    finally:
        cap.release()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video indexing worker")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--device", default="")
    parser.add_argument("--label-filter", default="")
    parser.add_argument("--categories", default="")
    parser.add_argument("--export-frames-dir", default="")
    parser.add_argument("--export-file-prefix", default="idx")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _prepare_runtime_env()
    args = _parse_args(argv)
    return run_index(
        video_path=str(args.video),
        model_path=str(args.model),
        sample_every=int(args.sample_every),
        device=str(args.device or ""),
        label_filter=str(args.label_filter or "").strip().lower(),
        categories=_parse_categories(str(args.categories or "")),
        export_frames_dir=str(args.export_frames_dir or ""),
        export_file_prefix=str(args.export_file_prefix or "idx"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
