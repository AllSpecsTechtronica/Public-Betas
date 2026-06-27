from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path

from .config import LOCKED_INFERENCE_IMAGE_SIZE, is_apple_silicon, resolve_model_path
from .runtime_profile import RuntimeProfile, profile_runtime


_RUNTIME_EXPORT_LOCK = threading.Lock()


def _auto_export_enabled() -> bool:
    raw = str(os.environ.get("INSIGHT_AUTO_EXPORT_RUNTIME", "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _variant_exists(path: Path) -> bool:
    if path.suffix.lower() == ".mlpackage":
        return path.is_dir()
    return path.is_file()


def apple_variant_paths(model_path: Path) -> list[Path]:
    resolved = Path(model_path).expanduser().resolve()
    if resolved.suffix.lower() != ".pt":
        return [resolved]
    return [
        resolved.with_suffix(".mlpackage"),
        resolved.with_suffix(".onnx"),
        resolved,
    ]


def _variant_path(model_path: Path, suffix: str) -> Path:
    return model_path if suffix == ".pt" else model_path.with_suffix(suffix)


def runtime_variant_paths(model_path: Path, runtime: RuntimeProfile | None = None) -> list[Path]:
    resolved = Path(model_path).expanduser().resolve()
    if resolved.suffix.lower() != ".pt":
        return [resolved]
    runtime = runtime or profile_runtime()
    candidates: list[Path] = []
    seen: set[Path] = set()
    for suffix in runtime.preferred_model_suffixes:
        candidate = _variant_path(resolved, suffix)
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates


def preferred_runtime_model_path(model_path: Path, runtime: RuntimeProfile | None = None) -> Path:
    candidates = runtime_variant_paths(model_path, runtime=runtime)
    for candidate in candidates:
        if _variant_exists(candidate):
            return candidate.resolve()
    return Path(model_path).expanduser().resolve()


def export_runtime_variants(
    model_path: Path,
    *,
    runtime: RuntimeProfile | None = None,
    image_size: int = LOCKED_INFERENCE_IMAGE_SIZE,
    export_onnx: bool = True,
    export_coreml: bool = False,
    export_engine: bool = False,
    half: bool = True,
    simplify: bool = True,
) -> list[Path]:
    from ultralytics import YOLO

    runtime = runtime or profile_runtime()
    source_model = Path(model_path).expanduser().resolve()
    if source_model.suffix.lower() != ".pt":
        raise ValueError("Runtime export requires a .pt checkpoint as the source model.")
    if not source_model.exists():
        raise FileNotFoundError(f"Model file not found: {source_model}")

    model = YOLO(str(source_model))
    outputs: list[Path] = []

    if export_onnx:
        exported = model.export(
            format="onnx",
            imgsz=image_size,
            half=half,
            simplify=simplify,
        )
        outputs.append(Path(exported).expanduser().resolve())

    if export_engine:
        export_kwargs = {
            "format": "engine",
            "imgsz": image_size,
            "half": half,
        }
        if runtime.has_cuda:
            export_kwargs["device"] = 0
        exported = model.export(**export_kwargs)
        outputs.append(Path(exported).expanduser().resolve())

    if export_coreml:
        exported = model.export(
            format="coreml",
            imgsz=image_size,
            half=half,
        )
        outputs.append(Path(exported).expanduser().resolve())

    return outputs


def ensure_runtime_model(
    model_path: Path,
    *,
    runtime: RuntimeProfile | None = None,
    image_size: int = LOCKED_INFERENCE_IMAGE_SIZE,
    half: bool = True,
    simplify: bool = True,
    auto_export: bool | None = None,
) -> Path:
    runtime = runtime or profile_runtime()
    resolved = Path(model_path).expanduser().resolve()
    if resolved.suffix.lower() != ".pt":
        return preferred_runtime_model_path(resolved, runtime=runtime)
    if auto_export is None:
        auto_export = _auto_export_enabled()

    needs_coreml = runtime.is_apple_silicon and not _variant_exists(resolved.with_suffix(".mlpackage"))
    needs_onnx = (
        runtime.is_apple_silicon
        or (runtime.system in {"Windows", "Linux"} and runtime.has_cuda)
    ) and not _variant_exists(resolved.with_suffix(".onnx"))
    if auto_export and (needs_coreml or needs_onnx):
        with _RUNTIME_EXPORT_LOCK:
            needs_coreml = runtime.is_apple_silicon and not _variant_exists(resolved.with_suffix(".mlpackage"))
            needs_onnx = (
                runtime.is_apple_silicon
                or (runtime.system in {"Windows", "Linux"} and runtime.has_cuda)
            ) and not _variant_exists(resolved.with_suffix(".onnx"))
            if needs_coreml or needs_onnx:
                export_runtime_variants(
                    resolved,
                    runtime=runtime,
                    image_size=image_size,
                    export_onnx=needs_onnx,
                    export_coreml=needs_coreml,
                    export_engine=False,
                    half=half,
                    simplify=simplify,
                )
    return preferred_runtime_model_path(resolved, runtime=runtime)


def export_apple_variants(
    model_path: Path,
    *,
    image_size: int = LOCKED_INFERENCE_IMAGE_SIZE,
    export_onnx: bool = True,
    export_coreml: bool = True,
    half: bool = True,
    simplify: bool = True,
) -> list[Path]:
    return export_runtime_variants(
        model_path,
        image_size=image_size,
        export_onnx=export_onnx,
        export_coreml=export_coreml,
        export_engine=False,
        half=half,
        simplify=simplify,
    )


def ensure_apple_runtime_model(
    model_path: Path,
    *,
    image_size: int = LOCKED_INFERENCE_IMAGE_SIZE,
    half: bool = True,
    simplify: bool = True,
) -> Path:
    resolved = Path(model_path).expanduser().resolve()
    if not is_apple_silicon():
        return resolved
    return ensure_runtime_model(
        resolved,
        image_size=image_size,
        half=half,
        simplify=simplify,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export runtime-friendly YOLO variants for the current host."
    )
    parser.add_argument("--model", type=str, default="yolo26n")
    parser.add_argument("--imgsz", type=int, default=LOCKED_INFERENCE_IMAGE_SIZE)
    parser.add_argument("--no-onnx", action="store_true")
    parser.add_argument("--no-coreml", action="store_true")
    parser.add_argument("--engine", action="store_true")
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args(argv)

    runtime = profile_runtime()
    source_model = resolve_model_path(args.model)
    outputs = export_runtime_variants(
        source_model,
        runtime=runtime,
        image_size=args.imgsz,
        export_onnx=not args.no_onnx,
        export_coreml=runtime.is_apple_silicon and not args.no_coreml,
        export_engine=args.engine,
        half=not args.no_half,
        simplify=not args.no_simplify,
    )
    for output in outputs:
        print(output)
    return 0
