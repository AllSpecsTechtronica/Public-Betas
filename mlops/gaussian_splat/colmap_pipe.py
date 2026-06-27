"""COLMAP discovery, sparse import, and automatic reconstruction wrapper."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

_STATUS = Callable[[str, str, float], None]


def find_colmap() -> Path | None:
    import shutil as _shutil

    exe = _shutil.which("colmap")
    return Path(exe) if exe else None


def copy_sparse_into_workspace(src_sparse_model_dir: Path, dest_sparse_model_dir: Path) -> None:
    """Copy COLMAP model files (binaries or text) into workspace sparse/0."""
    src_sparse_model_dir = src_sparse_model_dir.expanduser().resolve()
    dest_sparse_model_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "cameras.txt",
        "images.txt",
        "points3D.txt",
    ):
        p = src_sparse_model_dir / name
        if p.is_file():
            shutil.copy2(p, dest_sparse_model_dir / name)


def run_automatic_reconstructor(
    workspace: Path,
    *,
    image_dir_name: str = "images",
    colmap_executable: Path | None = None,
    on_status: _STATUS | None = None,
    gpu_index: str = "-1",
) -> None:
    """Run ``colmap automatic_reconstructor`` on ``workspace`` / ``image_dir_name``."""
    exe = colmap_executable or find_colmap()
    if exe is None:
        raise RuntimeError("COLMAP not found on PATH.")

    workspace = workspace.expanduser().resolve()
    image_path = workspace / image_dir_name
    if not image_path.is_dir():
        raise FileNotFoundError(f"Image directory missing: {image_path}")

    cmd = [
        str(exe),
        "automatic_reconstructor",
        "--workspace_path",
        str(workspace),
        "--image_path",
        str(image_path),
    ]
    if _supports_option(exe, "automatic_reconstructor", "--gpu_index"):
        cmd.extend(["--gpu_index", str(gpu_index)])
    log.info("running colmap: %s", " ".join(cmd))
    if on_status:
        on_status("colmap", "Running COLMAP automatic_reconstructor (this may take a long time)…", 0.55)

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=86400,
        cwd=str(workspace),
    )
    tail = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()
    if proc.returncode != 0:
        raise RuntimeError(
            "COLMAP automatic_reconstructor failed "
            f"(exit {proc.returncode}): {tail[-4000:]}"
        )
    log.info("colmap finished ok")


def _supports_option(exe: Path, command: str, option: str) -> bool:
    try:
        proc = subprocess.run(
            [str(exe), command, "-h"],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return False
    return option in ((proc.stdout or "") + (proc.stderr or ""))
