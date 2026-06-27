"""Prepare staged CV Ops 3D assets as nerfstudio datasets."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


StatusCallback = Callable[[str, str, float], None]


@dataclass(frozen=True)
class NerfstudioPrepareResult:
    asset_root: Path
    dataset_path: Path
    manifest_path: Path
    command: list[str]
    log_path: Path
    returncode: int
    manifest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_root": str(self.asset_root),
            "dataset_path": str(self.dataset_path),
            "manifest_path": str(self.manifest_path),
            "command": list(self.command),
            "log_path": str(self.log_path),
            "returncode": self.returncode,
            "manifest": dict(self.manifest),
        }


def prepare_nerfstudio_dataset(
    asset_root: Path,
    *,
    project_root: Path | None = None,
    on_status: StatusCallback | None = None,
) -> NerfstudioPrepareResult:
    """Run ``ns-process-data`` for a staged 3D asset.

    The asset must already contain ``manifest.json`` and source material under
    ``inputs/source`` or a source path in the manifest.
    """

    asset_root = Path(asset_root).expanduser().resolve()
    project_root = (project_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    manifest_path = asset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"3D asset manifest not found: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    source = dict(manifest.get("source") or {})
    source_kind = str(source.get("kind") or "image_folder")
    data_path = _resolve_data_path(asset_root, source_kind, source)
    dataset_path = asset_root / "nerfstudio" / "dataset"
    log_path = asset_root / "nerfstudio" / "prepare.log"
    dataset_path.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    runner = _resolve_ns_process_data(project_root)
    subcommand = "video" if source_kind == "video_file" else "images"
    command = [
        *runner.command_prefix,
        subcommand,
        "--data",
        str(data_path),
        "--output-dir",
        str(dataset_path),
    ]

    started_at = time.time()
    manifest["status"] = "preparing_nerfstudio"
    ns = dict(manifest.get("nerfstudio") or {})
    ns.update(
        {
            "prepare_status": "running",
            "dataset_path": str(dataset_path),
            "prepare_command": command,
            "prepare_log_path": str(log_path),
            "prepare_started_at": started_at,
            "runner": runner.label,
            "wired": True,
        }
    )
    manifest["nerfstudio"] = ns
    _write_manifest(manifest_path, manifest)

    if on_status:
        on_status("nerfstudio", f"Preparing dataset with {runner.label}", 0.05)

    env = os.environ.copy()
    if runner.pythonpath:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{runner.pythonpath}{os.pathsep}{existing}" if existing else runner.pythonpath
        )

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        proc = subprocess.Popen(
            command,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        line_count = 0
        for line in proc.stdout:
            log.write(line)
            log.flush()
            line_count += 1
            if on_status and line_count % 10 == 0:
                on_status("nerfstudio", line.strip()[-240:] or "processing", 0.35)
        returncode = int(proc.wait())

    finished_at = time.time()
    manifest = _load_manifest(manifest_path)
    ns = dict(manifest.get("nerfstudio") or {})
    ns.update(
        {
            "prepare_status": "complete" if returncode == 0 else "failed",
            "prepare_returncode": returncode,
            "prepare_finished_at": finished_at,
            "transforms_path": str(dataset_path / "transforms.json"),
        }
    )
    manifest["nerfstudio"] = ns
    manifest["status"] = "nerfstudio_ready" if returncode == 0 else "nerfstudio_prepare_failed"
    manifest["updated_at"] = finished_at
    _write_manifest(manifest_path, manifest)

    if returncode != 0:
        tail = _tail_text(log_path)
        raise RuntimeError(
            "nerfstudio dataset preparation failed "
            f"(exit {returncode}). Log: {log_path}\n{tail}"
        )

    if on_status:
        on_status("nerfstudio", f"Dataset ready: {dataset_path}", 1.0)

    return NerfstudioPrepareResult(
        asset_root=asset_root,
        dataset_path=dataset_path,
        manifest_path=manifest_path,
        command=command,
        log_path=log_path,
        returncode=returncode,
        manifest=manifest,
    )


@dataclass(frozen=True)
class _Runner:
    command_prefix: list[str]
    label: str
    pythonpath: str = ""


def _resolve_ns_process_data(project_root: Path) -> _Runner:
    exe = shutil.which("ns-process-data")
    if exe:
        return _Runner([exe], "ns-process-data")

    script = project_root / "nerfstudio" / "nerfstudio" / "scripts" / "process_data.py"
    package_root = project_root / "nerfstudio"
    if script.is_file():
        return _Runner(
            [sys.executable, str(script)],
            "local nerfstudio checkout",
            pythonpath=str(package_root),
        )
    raise FileNotFoundError(
        "ns-process-data was not found on PATH and local nerfstudio checkout was not found."
    )


def _resolve_data_path(asset_root: Path, source_kind: str, source: dict[str, Any]) -> Path:
    materialized = [Path(str(p)) for p in list(source.get("input_paths") or []) if str(p)]
    if source_kind == "video_file":
        for path in materialized:
            if path.is_file():
                return path
        source_path = Path(str(source.get("path") or "")).expanduser()
        if source_path.is_file():
            return source_path
        raise FileNotFoundError("No video source found for nerfstudio preparation.")

    source_dir = asset_root / "inputs" / "source"
    if source_dir.is_dir() and any(source_dir.iterdir()):
        return source_dir
    source_path = Path(str(source.get("path") or "")).expanduser()
    if source_path.is_dir():
        return source_path
    if source_path.is_file():
        return source_path.parent
    raise FileNotFoundError("No image source folder found for nerfstudio preparation.")


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest: {path}")
    return data


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _tail_text(path: Path, *, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[-max_chars:]
