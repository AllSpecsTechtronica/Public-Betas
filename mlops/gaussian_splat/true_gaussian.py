"""Nerfstudio Splatfacto training/export for CV Ops Gaussian workspaces."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from mlops.gaussian_splat.replication import ReplicationManifest, resolve_colmap_sparse_dir

StatusCallback = Callable[[str, str, float], None]


@dataclass(frozen=True)
class GaussianRunResult:
    workspace: Path
    dataset_path: Path
    train_output_dir: Path
    export_dir: Path
    splat_path: Path
    train_config_path: Path
    prepare_log_path: Path
    train_log_path: Path
    export_log_path: Path
    commands: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        for key in (
            "workspace",
            "dataset_path",
            "train_output_dir",
            "export_dir",
            "splat_path",
            "train_config_path",
            "prepare_log_path",
            "train_log_path",
            "export_log_path",
        ):
            data[key] = str(data[key])
        return data


@dataclass(frozen=True)
class _Runner:
    command_prefix: list[str]
    label: str
    pythonpath: str = ""


@dataclass(frozen=True)
class _ResourceProfile:
    cpu_count: int
    memory_gb: float
    worker_threads: int
    max_image_size: int
    max_num_features: int
    matcher_block_size: int
    torch_threads: int

    def summary(self) -> str:
        return (
            f"{self.cpu_count} CPU cores, {self.memory_gb:.1f} GB RAM, "
            f"{self.worker_threads} COLMAP threads, max image {self.max_image_size}px, "
            f"{self.max_num_features} SIFT features"
        )


def run_true_gaussian_pipeline(
    workspace: Path,
    *,
    max_num_iterations: int | None = None,
    project_root: Path | None = None,
    on_status: StatusCallback | None = None,
) -> GaussianRunResult:
    """Train and export a true 3D Gaussian splat from a replication workspace.

    This is intentionally not a depth-estimation fallback. It requires at least
    two staged images and a working Nerfstudio/COLMAP runtime.
    """

    workspace = workspace.expanduser().resolve()
    project_root = (project_root or Path(__file__).resolve().parents[2]).expanduser().resolve()
    manifest_path = workspace / "replication_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Replication manifest not found: {manifest_path}")
    manifest = ReplicationManifest.load(manifest_path)
    image_paths = [Path(p) for p in manifest.image_paths]
    if len(image_paths) < 2:
        raise RuntimeError(
            "True Gaussian splatting needs overlapping multi-view input. "
            "Provide an image folder or video with at least two frames; single-image depth meshes "
            "are available only from the Local depth backend."
        )

    iterations = _resolve_iterations(max_num_iterations)
    dataset_path = workspace / "nerfstudio" / "dataset"
    train_output_dir = workspace / "nerfstudio" / "outputs"
    export_dir = workspace / "nerfstudio" / "gaussian_export"
    logs_dir = workspace / "nerfstudio" / "logs"
    dataset_path.mkdir(parents=True, exist_ok=True)
    train_output_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    commands: dict[str, list[str]] = {}
    resources = _resolve_resource_profile()
    env = _runner_env(project_root, resources)
    started_at = time.time()
    status = _EtaStatus(on_status, started_at=started_at)

    status("resources", resources.summary(), 0.02)
    status("gaussian_prepare", "Preparing Nerfstudio dataset from staged images", 0.05)
    prepare_runner = _resolve_ns_process_data(project_root)
    if _colmap_without_cuda(project_root):
        gpu_args = ["0"]
    else:
        gpu_args = ["1"]
    _stage_nerfstudio_images(
        workspace / "images",
        dataset_path / "images",
        max_image_size=resources.max_image_size,
        on_status=status,
    )
    sparse = _workspace_sparse_dir(workspace)
    if sparse is not None:
        ns_sparse = dataset_path / "colmap" / "sparse" / "0"
        _copy_sparse(sparse, ns_sparse)
        sparse = ns_sparse
    elif _resolve_colmap(project_root) is None:
        raise RuntimeError(
            "COLMAP not found on PATH. Install COLMAP, or import an existing COLMAP sparse model "
            "before running true Gaussian splatting."
        )
    else:
        sparse = _run_colmap_reconstruction(
            dataset_path,
            project_root=project_root,
            logs_dir=logs_dir,
            commands=commands,
            use_gpu=gpu_args[0],
            resources=resources,
            on_status=status,
        )

    colmap_model_rel = sparse.relative_to(dataset_path)
    prepare_cmd = [
        *prepare_runner.command_prefix,
        "images",
        "--data",
        str(workspace / "images"),
        "--output-dir",
        str(dataset_path),
        "--skip-colmap",
        "--skip-image-processing",
        "--colmap-model-path",
        str(colmap_model_rel),
    ]
    commands["prepare"] = prepare_cmd
    prepare_log = logs_dir / "prepare.log"
    _run_logged(
        prepare_cmd,
        prepare_log,
        cwd=project_root,
        env=env,
        on_status=status,
        progress=0.25,
        stage="nerfstudio_prepare",
    )

    transforms_path = dataset_path / "transforms.json"
    if not transforms_path.is_file():
        raise RuntimeError(
            "Nerfstudio dataset preparation did not produce transforms.json. "
            f"Check log: {prepare_log}"
        )

    status("gaussian_train", f"Training Splatfacto for {iterations} iterations", 0.35)
    train_runner = _resolve_ns_train(project_root)
    device_type = _resolve_device_type()
    train_cmd = [
        *train_runner.command_prefix,
        "splatfacto",
        "--output-dir",
        str(train_output_dir),
        "--max-num-iterations",
        str(iterations),
        "--vis",
        "viewer",
        "--viewer.quit-on-train-completion",
        "True",
        "--machine.device-type",
        device_type,
        "--machine.num-devices",
        "1",
        "--pipeline.datamanager.cache-images",
        "cpu",
        "--pipeline.datamanager.images-on-gpu",
        "False",
        "--pipeline.datamanager.masks-on-gpu",
        "False",
        "--data",
        str(dataset_path),
    ]
    commands["train"] = train_cmd
    train_log = logs_dir / "train.log"
    _run_logged(
        train_cmd,
        train_log,
        cwd=project_root,
        env=env,
        on_status=status,
        progress=0.75,
        stage="splatfacto_train",
        progress_range=(0.35, 0.88),
        max_iterations=iterations,
        progress_parser="train",
    )

    config_path = _latest_config(train_output_dir)
    if config_path is None:
        raise RuntimeError(f"Splatfacto training did not write config.yml under {train_output_dir}")

    status("gaussian_export", "Exporting trained Gaussian splat PLY", 0.88)
    export_script = Path(__file__).with_name("nerfstudio_gaussian_export.py")
    splat_path = export_dir / "splat.ply"
    export_cmd = [
        sys.executable,
        str(export_script),
        "--load-config",
        str(config_path),
        "--output-path",
        str(splat_path),
    ]
    commands["export"] = export_cmd
    export_log = logs_dir / "export.log"
    _run_logged(
        export_cmd,
        export_log,
        cwd=project_root,
        env=env,
        on_status=status,
        progress=0.97,
        stage="gaussian_export",
    )
    if not splat_path.is_file():
        raise RuntimeError(f"Gaussian export did not produce {splat_path}. Check log: {export_log}")
    preview_html = export_dir / "preview.html"
    _write_ply_preview(preview_html, splat_path)

    result = GaussianRunResult(
        workspace=workspace,
        dataset_path=dataset_path,
        train_output_dir=train_output_dir,
        export_dir=export_dir,
        splat_path=splat_path,
        train_config_path=config_path,
        prepare_log_path=prepare_log,
        train_log_path=train_log,
        export_log_path=export_log,
        commands=commands,
    )
    _write_run_manifest(workspace / "gaussian_run_manifest.json", result)
    status("gaussian_done", f"Gaussian splat ready: {splat_path}", 1.0)
    return result


class _EtaStatus:
    def __init__(self, callback: StatusCallback | None, *, started_at: float) -> None:
        self.callback = callback
        self.started_at = started_at
        self.last_progress = 0.0

    def __call__(self, stage: str, message: str, progress: float = -1.0) -> None:
        effective_progress = progress
        if progress >= 0.0:
            self.last_progress = max(self.last_progress, min(0.999, progress))
            effective_progress = self.last_progress
        eta = self.eta_text(effective_progress)
        out = f"{message} | ETA {eta}" if eta else message
        if self.callback:
            self.callback(stage, out, effective_progress)

    def eta_text(self, progress: float = -1.0) -> str:
        p = progress if progress >= 0.0 else self.last_progress
        if p <= 0.02 or p >= 0.999:
            return ""
        elapsed = max(0.1, time.time() - self.started_at)
        remaining = elapsed * (1.0 - p) / p
        return _format_duration(remaining)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _resolve_iterations(value: int | None) -> int:
    if value is not None:
        return max(100, int(value))
    raw = os.environ.get("CVOPS_GAUSSIAN_MAX_ITERATIONS", "").strip()
    if raw:
        try:
            return max(100, int(raw))
        except ValueError:
            pass
    return 7000


def _resolve_device_type() -> str:
    raw = os.environ.get("CVOPS_GAUSSIAN_DEVICE", "auto").strip().lower()
    if raw in {"cpu", "cuda", "mps"}:
        return raw
    if platform.system().lower() == "darwin":
        return "cpu"
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _resolve_resource_profile() -> _ResourceProfile:
    cpu_count = max(1, os.cpu_count() or 1)
    memory_gb = _system_memory_gb()
    default_threads = max(1, min(cpu_count // 2, 8))
    if memory_gb <= 16:
        default_threads = min(default_threads, 2)
        default_image_size = 1200
        default_features = 4096
        default_block = 20
    elif memory_gb <= 32:
        default_threads = min(default_threads, 4)
        default_image_size = 1600
        default_features = 6144
        default_block = 30
    else:
        default_image_size = 2200
        default_features = 8192
        default_block = 50
    worker_threads = _env_int("CVOPS_GAUSSIAN_THREADS", default_threads, min_value=1, max_value=cpu_count)
    max_image_size = _env_int("CVOPS_GAUSSIAN_MAX_IMAGE_SIZE", default_image_size, min_value=256, max_value=8192)
    max_num_features = _env_int("CVOPS_GAUSSIAN_MAX_FEATURES", default_features, min_value=512, max_value=32768)
    matcher_block_size = _env_int("CVOPS_GAUSSIAN_MATCH_BLOCK", default_block, min_value=5, max_value=100)
    torch_threads = _env_int("CVOPS_GAUSSIAN_TORCH_THREADS", worker_threads, min_value=1, max_value=cpu_count)
    return _ResourceProfile(
        cpu_count=cpu_count,
        memory_gb=memory_gb,
        worker_threads=worker_threads,
        max_image_size=max_image_size,
        max_num_features=max_num_features,
        matcher_block_size=matcher_block_size,
        torch_threads=torch_threads,
    )


def _system_memory_gb() -> float:
    try:
        proc = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            return max(1.0, int(proc.stdout.strip()) / (1024 ** 3))
    except Exception:
        pass
    return 16.0


def _env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            return max(min_value, min(max_value, int(raw)))
        except ValueError:
            pass
    return max(min_value, min(max_value, int(default)))


def _resolve_ns_process_data(project_root: Path) -> _Runner:
    exe = shutil.which("ns-process-data")
    if exe:
        return _Runner([exe], "ns-process-data")
    script = project_root / "nerfstudio" / "nerfstudio" / "scripts" / "process_data.py"
    package_root = project_root / "nerfstudio"
    if script.is_file():
        return _Runner(
            [sys.executable, str(script)],
            "local nerfstudio process_data",
            pythonpath=str(package_root),
        )
    raise FileNotFoundError("ns-process-data was not found and local Nerfstudio checkout is missing.")


def _resolve_ns_train(project_root: Path) -> _Runner:
    exe = shutil.which("ns-train")
    if exe:
        return _Runner([exe], "ns-train")
    script = project_root / "nerfstudio" / "nerfstudio" / "scripts" / "train.py"
    package_root = project_root / "nerfstudio"
    if script.is_file():
        return _Runner(
            [sys.executable, str(script)],
            "local nerfstudio train",
            pythonpath=str(package_root),
        )
    raise FileNotFoundError("ns-train was not found and local Nerfstudio checkout is missing.")


def _resolve_ns_export(project_root: Path) -> _Runner:
    exe = shutil.which("ns-export")
    if exe:
        return _Runner([exe], "ns-export")
    script = project_root / "nerfstudio" / "nerfstudio" / "scripts" / "exporter.py"
    package_root = project_root / "nerfstudio"
    if script.is_file():
        return _Runner(
            [sys.executable, str(script)],
            "local nerfstudio export",
            pythonpath=str(package_root),
        )
    raise FileNotFoundError("ns-export was not found and local Nerfstudio checkout is missing.")


def _runner_env(project_root: Path, resources: _ResourceProfile | None = None) -> dict[str, str]:
    env = os.environ.copy()
    path_parts = []
    local_bin = project_root / ".tools" / "bin"
    if local_bin.is_dir():
        path_parts.append(str(local_bin))
    homebrew_bin = Path("/opt/homebrew/bin")
    if homebrew_bin.is_dir():
        path_parts.append(str(homebrew_bin))
    if path_parts:
        existing_path = env.get("PATH", "")
        path_parts.append(existing_path)
        env["PATH"] = os.pathsep.join(p for p in path_parts if p)
    package_root = project_root / "nerfstudio"
    if package_root.is_dir():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{package_root}{os.pathsep}{existing}" if existing else str(package_root)
    if resources is not None:
        thread_count = str(resources.torch_threads)
        env["OMP_NUM_THREADS"] = thread_count
        env["MKL_NUM_THREADS"] = thread_count
        env["VECLIB_MAXIMUM_THREADS"] = thread_count
        env["NUMEXPR_NUM_THREADS"] = thread_count
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env


def _resolve_colmap(project_root: Path) -> str | None:
    local_colmap = project_root / ".tools" / "bin" / "colmap"
    if local_colmap.exists():
        return str(local_colmap)
    path = os.environ.get("PATH", "")
    extra = []
    local_bin = project_root / ".tools" / "bin"
    if local_bin.is_dir():
        extra.append(str(local_bin))
    homebrew_bin = Path("/opt/homebrew/bin")
    if homebrew_bin.is_dir():
        extra.append(str(homebrew_bin))
    if extra:
        path = os.pathsep.join([*extra, path])
    return shutil.which("colmap", path=path)


def _colmap_without_cuda(project_root: Path) -> bool:
    exe = _resolve_colmap(project_root)
    if not exe:
        return False
    try:
        proc = subprocess.run([exe, "-h"], capture_output=True, text=True, timeout=20)
    except Exception:
        return False
    text = (proc.stdout or "") + (proc.stderr or "")
    return "without CUDA" in text


def _run_logged(
    command: list[str],
    log_path: Path,
    *,
    cwd: Path,
    env: dict[str, str],
    on_status: StatusCallback | None,
    progress: float,
    stage: str = "gaussian",
    progress_range: tuple[float, float] | None = None,
    max_iterations: int | None = None,
    progress_parser: str = "",
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(env)
    cache_dir = log_path.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    started = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        if on_status:
            on_status(stage, "$ " + " ".join(command), progress)
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        last_emit = 0.0
        progress_events_seen = 0
        for i, line in enumerate(proc.stdout, start=1):
            log.write(line)
            log.flush()
            cleaned = _strip_ansi(line.strip())
            now = time.time()
            line_progress = _progress_from_line(cleaned, progress_range, max_iterations, progress_parser)
            if line_progress is None and progress_parser == "colmap_matching" and max_iterations:
                if "feature_matching.cc:217]" in cleaned and "] in " in cleaned:
                    progress_events_seen += 1
                    lo, hi = progress_range or (progress, progress)
                    line_progress = lo + (hi - lo) * min(1.0, progress_events_seen / max(1, max_iterations))
            emit_progress = line_progress if line_progress is not None else progress
            urgent = _urgent_log_line(cleaned)
            important = _important_log_line(cleaned)
            should_emit = (
                urgent
                or line_progress is not None
                or now - last_emit > 2.0
                or i % 18 == 0
                or (important and now - last_emit > 0.9)
            )
            if on_status and cleaned and should_emit:
                on_status(stage, cleaned[-320:], emit_progress)
                last_emit = now
        returncode = int(proc.wait())
    if returncode != 0:
        tail = _tail_text(log_path)
        raise RuntimeError(
            f"{stage} command failed after {time.time() - started:.1f}s "
            f"(exit {returncode}). Log: {log_path}\n{tail}"
        )


def _important_log_line(line: str) -> bool:
    needles = (
        "Processed file",
        "Processing block",
        "Feature extraction",
        "Feature matching",
        "Finding good initial image pair",
        "Colmap matched",
        "All DONE",
        "Saving checkpoints",
        "Saving config",
        "Traceback",
        "Error",
        "Failed",
        "Training",
        "Export",
    )
    return any(n in line for n in needles)


def _urgent_log_line(line: str) -> bool:
    needles = ("Traceback", "RuntimeError", "Error", "Failed", "Exception")
    return any(n in line for n in needles)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def _progress_from_line(
    line: str,
    progress_range: tuple[float, float] | None,
    max_iterations: int | None,
    parser: str = "",
) -> float | None:
    if progress_range is None:
        return None
    lo, hi = progress_range
    if parser == "colmap_features":
        match = re.search(r"Processed file \[(\d+)/(\d+)\]", line)
        if not match:
            return None
        current = int(match.group(1))
        total = max(1, int(match.group(2)))
        return lo + (hi - lo) * min(1.0, current / total)
    if parser == "colmap_matching":
        match = re.search(r"Processing block \[(\d+)/(\d+),\s*(\d+)/(\d+)\]", line)
        if not match:
            return None
        row = int(match.group(1))
        rows = max(1, int(match.group(2)))
        col = int(match.group(3))
        cols = max(1, int(match.group(4)))
        current = ((row - 1) * cols) + col
        total = rows * cols
        return lo + (hi - lo) * min(1.0, current / max(1, total))
    if parser != "train" or not max_iterations:
        return None
    if any(token in line for token in ("TrainerConfig", "max_steps=", "Trainer.train_iteration", "get_train_loss_dict")):
        return None
    match = re.search(r"(?:^|\b)(?:step|iter(?:ation)?)\s*[:= ]+\s*(\d+)\b", line, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"\b(\d+)\s*/\s*(\d+)\b", line)
        if not match:
            return None
        total = int(match.group(2))
        if total > max_iterations * 2:
            return None
        current = int(match.group(1))
        return lo + (hi - lo) * min(1.0, current / max(1, total))
    current = min(max_iterations, int(match.group(1)))
    return lo + (hi - lo) * (current / max(1, max_iterations))


def _workspace_sparse_dir(workspace: Path) -> Path | None:
    for candidate in (workspace / "sparse" / "0", workspace / "sparse"):
        resolved = resolve_colmap_sparse_dir(candidate)
        if resolved is not None:
            return resolved
    return None


def _copy_sparse(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "cameras.txt",
        "images.txt",
        "points3D.txt",
    ):
        p = src / name
        if p.is_file():
            shutil.copy2(p, dest / name)


def _stage_nerfstudio_images(
    src_dir: Path,
    dest_dir: Path,
    *,
    max_image_size: int,
    on_status: StatusCallback | None,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    images = [p for p in sorted(src_dir.iterdir()) if p.is_file()]
    for idx, p in enumerate(images, start=1):
        dest = dest_dir / p.name
        if _resize_image_for_colmap(p, dest, max_image_size=max_image_size):
            pass
        else:
            shutil.copy2(p, dest)
        if on_status and (idx == 1 or idx % 10 == 0 or idx == len(images)):
            on_status("resources", f"Staged {idx}/{len(images)} images at <= {max_image_size}px", 0.07)


def _resize_image_for_colmap(src: Path, dest: Path, *, max_image_size: int) -> bool:
    try:
        from PIL import Image, ImageOps  # noqa: PLC0415

        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            longest = max(w, h)
            if longest > max_image_size:
                scale = max_image_size / float(longest)
                size = (max(1, int(w * scale)), max(1, int(h * scale)))
                im = im.resize(size, Image.Resampling.LANCZOS)
            im.save(dest, quality=92)
        return True
    except Exception:
        return False


def _run_colmap_reconstruction(
    dataset_path: Path,
    *,
    project_root: Path,
    logs_dir: Path,
    commands: dict[str, list[str]],
    use_gpu: str,
    resources: _ResourceProfile,
    on_status: StatusCallback | None,
) -> Path:
    exe = _resolve_colmap(project_root)
    if exe is None:
        raise RuntimeError("COLMAP not found on PATH.")
    colmap_dir = dataset_path / "colmap"
    sparse_root = colmap_dir / "sparse"
    sparse_root.mkdir(parents=True, exist_ok=True)
    database_path = colmap_dir / "database.db"
    if database_path.exists():
        database_path.unlink()

    feature_cmd = [
        exe,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(dataset_path / "images"),
        "--ImageReader.single_camera",
        "0",
        "--ImageReader.camera_model",
        "OPENCV",
        "--FeatureExtraction.use_gpu",
        use_gpu,
        "--FeatureExtraction.num_threads",
        str(resources.worker_threads),
        "--FeatureExtraction.max_image_size",
        str(resources.max_image_size),
        "--SiftExtraction.max_num_features",
        str(resources.max_num_features),
    ]
    match_cmd = [
        exe,
        "exhaustive_matcher",
        "--database_path",
        str(database_path),
        "--FeatureMatching.use_gpu",
        use_gpu,
        "--FeatureMatching.num_threads",
        str(resources.worker_threads),
        "--ExhaustiveMatching.block_size",
        str(resources.matcher_block_size),
    ]
    mapper_cmd = [
        exe,
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(dataset_path / "images"),
        "--output_path",
        str(sparse_root),
        "--Mapper.num_threads",
        str(resources.worker_threads),
        "--Mapper.ba_use_gpu",
        use_gpu,
    ]
    commands["colmap_feature"] = feature_cmd
    commands["colmap_match"] = match_cmd
    commands["colmap_mapper"] = mapper_cmd
    if on_status:
        on_status("colmap", "Extracting SIFT features", 0.09)
    _run_logged(
        feature_cmd,
        logs_dir / "colmap_feature.log",
        cwd=project_root,
        env=_runner_env(project_root, resources),
        on_status=on_status,
        progress=0.12,
        stage="colmap_features",
        progress_range=(0.09, 0.16),
        progress_parser="colmap_features",
    )
    if on_status:
        on_status("colmap", "Matching image pairs", 0.16)
    num_images = len([p for p in (dataset_path / "images").iterdir() if p.is_file()])
    match_blocks = max(1, (num_images + resources.matcher_block_size - 1) // resources.matcher_block_size)
    _run_logged(
        match_cmd,
        logs_dir / "colmap_match.log",
        cwd=project_root,
        env=_runner_env(project_root, resources),
        on_status=on_status,
        progress=0.18,
        stage="colmap_matching",
        progress_range=(0.16, 0.22),
        max_iterations=match_blocks * match_blocks,
        progress_parser="colmap_matching",
    )
    if on_status:
        on_status("colmap", "Mapping sparse reconstruction", 0.22)
    _run_logged(
        mapper_cmd,
        logs_dir / "colmap_mapper.log",
        cwd=project_root,
        env=_runner_env(project_root, resources),
        on_status=on_status,
        progress=0.23,
        stage="colmap_mapper",
    )

    candidates = sorted(p for p in sparse_root.iterdir() if resolve_colmap_sparse_dir(p) is not None)
    if not candidates:
        raise RuntimeError(
            "COLMAP did not produce a valid sparse model. The input images likely are not an overlapping "
            "multi-view capture of one subject."
        )
    return candidates[0]


def _write_ply_preview(path: Path, splat_path: Path) -> None:
    rel = splat_path.name
    path.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="ViewPort" content="width=device-width, initial-scale=1" />
  <title>Gaussian PLY Preview</title>
  <style>
    html, body { margin: 0; height: 100%; background: #101113; color: #eee; font-family: system-ui, sans-serif; }
    #viewer { position: fixed; inset: 0; }
    #hud { position: fixed; left: 12px; bottom: 12px; background: rgba(0,0,0,.55); padding: 8px 10px; border-radius: 6px; font-size: 13px; }
  </style>
</head>
<body>
  <div id="viewer"></div>
  <div id="hud">Drag to orbit. Scroll to zoom.</div>
  <script type="importmap">
    {"imports":{"three":"https://unpkg.com/three@0.164.1/build/three.module.js","three/addons/":"https://unpkg.com/three@0.164.1/examples/jsm/"}}
  </script>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
    const root = document.getElementById('viewer');
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    root.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101113);
    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 1000);
    camera.position.set(0, 0, 3);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    scene.add(new THREE.AmbientLight(0xffffff, 1));
    new PLYLoader().load('""" + rel + """', (geometry) => {
      geometry.computeBoundingSphere();
      const center = geometry.boundingSphere.center;
      geometry.translate(-center.x, -center.y, -center.z);
      const radius = Math.max(geometry.boundingSphere.radius, 0.001);
      const material = new THREE.PointsMaterial({ size: radius * 0.01, color: 0xd8e7ff, vertexColors: geometry.hasAttribute('color') });
      scene.add(new THREE.Points(geometry, material));
      camera.position.set(0, 0, radius * 3.0);
      controls.update();
    });
    function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
    animate();
    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def _latest_config(output_dir: Path) -> Path | None:
    configs = sorted(output_dir.glob("**/config.yml"), key=lambda p: p.stat().st_mtime)
    return configs[-1] if configs else None


def _write_run_manifest(path: Path, result: GaussianRunResult) -> None:
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def _tail_text(path: Path, *, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[-max_chars:]
