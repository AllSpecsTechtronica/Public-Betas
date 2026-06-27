"""Replication inputs: image folders, video frame extraction, COLMAP-compatible layouts."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".JPG", ".JPEG", ".PNG"}
)
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"})

SourceKind = Literal["image_folder", "video_file", "single_image"]
CalibrationKind = Literal["none", "import_sparse", "run_colmap"]


def list_images_in_folder(folder: Path) -> list[Path]:
    """Sorted list of image paths under ``folder`` (non-recursive)."""
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS:
            out.append(p)
    return out


def validate_image_folder(folder: Path) -> tuple[bool, str]:
    imgs = list_images_in_folder(folder)
    if not imgs:
        return False, f"No images ({', '.join(sorted(IMAGE_EXTENSIONS))}) in folder: {folder}"
    return True, ""


def validate_video_file(path: Path) -> tuple[bool, str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        return False, f"Video not found: {path}"
    if path.suffix.lower() not in {e.lower() for e in VIDEO_EXTENSIONS}:
        return False, f"Unsupported video extension (expected one of {sorted(VIDEO_EXTENSIONS)}): {path}"
    return True, ""


def validate_single_image(path: Path) -> tuple[bool, str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        return False, f"Image not found: {path}"
    if path.suffix not in IMAGE_EXTENSIONS:
        return False, (
            f"Unsupported image extension (expected one of {sorted(IMAGE_EXTENSIONS)}): {path}"
        )
    return True, ""


def find_ffmpeg() -> Path | None:
    import shutil as _shutil

    exe = _shutil.which("ffmpeg")
    return Path(exe) if exe else None


def extract_video_frames(
    video: Path,
    out_dir: Path,
    *,
    fps: float = 1.0,
    max_frames: int = 300,
    on_status: Callable[[str, str, float], None] | None = None,
) -> list[Path]:
    """Extract PNG frames with ffmpeg. Returns sorted frame paths (capped at ``max_frames``)."""
    ff = find_ffmpeg()
    if ff is None:
        raise RuntimeError("ffmpeg not found on PATH — install ffmpeg to extract video frames.")

    video = video.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.png")
    fps = max(0.05, float(fps))
    cmd = [
        str(ff),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-frames:v",
        str(max(1, int(max_frames))),
        pattern,
    ]
    if on_status:
        on_status("ffmpeg", f"Extracting frames at {fps:.2f} fps (max {max_frames})…", 0.1)
    log.info("running ffmpeg: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {err[:800]}")

    frames = sorted(out_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("ffmpeg produced no frames — check the video file.")
    if on_status:
        on_status("ffmpeg", f"Extracted {len(frames)} frame(s).", 0.35)
    return frames


def _has_colmap_model_files(d: Path) -> bool:
    """COLMAP sparse reconstruction folder (usually …/sparse/0)."""
    if not d.is_dir():
        return False
    cam_bin = (d / "cameras.bin").is_file()
    cam_txt = (d / "cameras.txt").is_file()
    img_bin = (d / "images.bin").is_file()
    img_txt = (d / "images.txt").is_file()
    return (cam_bin or cam_txt) and (img_bin or img_txt)


def resolve_colmap_sparse_dir(user_path: Path) -> Path | None:
    """Return directory containing cameras/images binaries or txt, or None."""
    user_path = user_path.expanduser().resolve()
    if not user_path.exists():
        return None
    if user_path.is_file():
        return None
    if _has_colmap_model_files(user_path):
        return user_path
    # Often users select …/sparse instead of …/sparse/0
    sub0 = user_path / "0"
    if _has_colmap_model_files(sub0):
        return sub0
    return None


@dataclass
class ReplicationManifest:
    version: int = 1
    source_kind: str = ""
    source_path: str = ""
    workspace: str = ""
    images_dir: str = ""
    image_paths: list[str] = field(default_factory=list)
    calibration: str = "none"
    colmap_sparse_imported: str = ""
    colmap_sparse_import_source: str = ""
    colmap_automatic_ran: bool = False
    colmap_log: str = ""
    ffmpeg_path: str = ""
    notes: str = ""

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ReplicationManifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            version=int(data.get("version") or 1),
            source_kind=str(data.get("source_kind") or ""),
            source_path=str(data.get("source_path") or ""),
            workspace=str(data.get("workspace") or ""),
            images_dir=str(data.get("images_dir") or ""),
            image_paths=list(data.get("image_paths") or []),
            calibration=str(data.get("calibration") or "none"),
            colmap_sparse_imported=str(data.get("colmap_sparse_imported") or ""),
            colmap_sparse_import_source=str(data.get("colmap_sparse_import_source") or ""),
            colmap_automatic_ran=bool(data.get("colmap_automatic_ran")),
            colmap_log=str(data.get("colmap_log") or ""),
            ffmpeg_path=str(data.get("ffmpeg_path") or ""),
            notes=str(data.get("notes") or ""),
        )


def _copy_or_link(src: Path, dest: Path, *, prefer_symlink: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        if prefer_symlink:
            dest.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def populate_workspace_images(
    workspace: Path,
    images: list[Path],
    *,
    prefer_symlink: bool = False,
    on_status: Callable[[str, str, float], None] | None = None,
) -> list[Path]:
    """Copy or symlink images into ``workspace/images`` with stable names."""
    img_dir = workspace / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    total = max(len(images), 1)
    for i, src in enumerate(sorted(images, key=lambda p: str(p).lower())):
        dest = img_dir / f"frame_{i + 1:06d}{src.suffix.lower() or '.png'}"
        _copy_or_link(src, dest, prefer_symlink=prefer_symlink)
        out.append(dest)
        if on_status and i % 25 == 0:
            on_status("images", f"Staged {i + 1}/{len(images)} images…", 0.15 + 0.25 * (i / total))
    return out


def prepare_replication_workspace(
    workspace: Path,
    *,
    source_kind: SourceKind,
    source_path: Path,
    calibration: CalibrationKind,
    colmap_sparse_user: Path | None,
    video_fps: float,
    video_max_frames: int,
    prefer_symlink: bool = False,
    on_status: Callable[[str, str, float], None] | None = None,
) -> ReplicationManifest:
    """Create ``workspace/images``, optional COLMAP import/run. Returns manifest."""
    from mlops.gaussian_splat.colmap_pipe import (  # noqa: PLC0415
        copy_sparse_into_workspace,
        find_colmap,
        run_automatic_reconstructor,
    )

    workspace = workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    source_path = source_path.expanduser().resolve()

    manifest = ReplicationManifest(
        workspace=str(workspace),
        images_dir=str(workspace / "images"),
        source_kind=source_kind,
        source_path=str(source_path),
        ffmpeg_path=str(find_ffmpeg() or ""),
    )

    if source_kind == "image_folder":
        ok, err = validate_image_folder(source_path)
        if not ok:
            raise ValueError(err)
        imgs = list_images_in_folder(source_path)
        manifest.image_paths = [str(p) for p in populate_workspace_images(
            workspace, imgs, prefer_symlink=prefer_symlink, on_status=on_status
        )]
    elif source_kind == "single_image":
        ok, err = validate_single_image(source_path)
        if not ok:
            raise ValueError(err)
        manifest.image_paths = [str(p) for p in populate_workspace_images(
            workspace,
            [source_path],
            prefer_symlink=False,
            on_status=on_status,
        )]
    elif source_kind == "video_file":
        ok, err = validate_video_file(source_path)
        if not ok:
            raise ValueError(err)
        frames = extract_video_frames(
            source_path,
            workspace / "_video_frames_raw",
            fps=video_fps,
            max_frames=video_max_frames,
            on_status=on_status,
        )
        manifest.image_paths = [str(p) for p in populate_workspace_images(
            workspace, frames, prefer_symlink=False, on_status=on_status
        )]
    else:
        raise ValueError(f"Unknown source_kind: {source_kind}")

    manifest.calibration = calibration

    if calibration == "import_sparse":
        if colmap_sparse_user is None:
            raise ValueError("COLMAP sparse folder is required for import_sparse calibration.")
        resolved = resolve_colmap_sparse_dir(colmap_sparse_user)
        if resolved is None:
            raise ValueError(
                "Not a COLMAP sparse model folder (need cameras/images .bin or .txt). "
                f"Got: {colmap_sparse_user}"
            )
        sparse_dest = workspace / "sparse" / "0"
        copy_sparse_into_workspace(resolved, sparse_dest)
        manifest.colmap_sparse_imported = str(sparse_dest)
        manifest.colmap_sparse_import_source = str(colmap_sparse_user.resolve())
        if on_status:
            on_status("colmap_import", f"Imported sparse model into {sparse_dest}", 0.75)

    elif calibration == "run_colmap":
        exe = find_colmap()
        if exe is None:
            raise RuntimeError(
                "COLMAP executable not found on PATH. Install COLMAP or choose import sparse / none."
            )
        run_automatic_reconstructor(
            workspace,
            image_dir_name="images",
            colmap_executable=exe,
            on_status=on_status,
        )
        manifest.colmap_automatic_ran = True
        if on_status:
            on_status("colmap", "COLMAP automatic reconstruction finished.", 0.95)

    manifest.write(workspace / "replication_manifest.json")
    if on_status:
        on_status("done", f"Workspace ready: {workspace}", 1.0)
    return manifest


def new_job_id() -> str:
    return f"repl_{uuid.uuid4().hex[:12]}"
