"""Semantic carve: folder of images + text query -> labeled ImageFolder dataset.

This is the "general vision" counterpart to the text-oriented archive engine.
It embeds a source folder with CLIP, ranks images against a natural-language
query (zero-shot, no training), and materializes a binary ImageFolder dataset
(``<class>`` / ``not_<class>``) under the library dataset registry, where it is
auto-discovered as an ``imagefolder_classification`` dataset and becomes
trainable in a few clicks.

The embedding step is injected (any object exposing ``embed_images`` and
``embed_texts`` like :class:`clip_embed.ClipEmbedder`) so the carve/select/
materialize logic is unit-testable without loading CLIP.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import numpy as np

from .clip_embed import list_images


class Embedder(Protocol):
    def embed_images(
        self, paths, *, batch_size: int = ..., progress_cb=...
    ) -> tuple[np.ndarray, list[Path]]: ...

    def embed_texts(self, prompts: list[str]) -> np.ndarray: ...


def sanitize_class_name(name: str) -> str:
    safe = re.sub(r"[^a-z0-9_]+", "_", str(name or "").strip().lower()).strip("_")
    return safe or "class"


@dataclass
class CarveIndex:
    """Embedded images for one source folder; reusable across queries."""

    folder: str
    paths: list[Path]
    feats: np.ndarray  # (N, D) L2-normalized
    created_at: float = field(default_factory=time.time)

    def __len__(self) -> int:
        return len(self.paths)


def build_index(
    folder: Path,
    embedder: Embedder,
    *,
    max_images: int = 0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> CarveIndex:
    folder = Path(folder).expanduser()
    paths = list_images(folder)
    if max_images and len(paths) > max_images:
        paths = paths[:max_images]
    feats, kept = embedder.embed_images(paths, progress_cb=progress_cb)
    return CarveIndex(folder=str(folder), paths=kept, feats=feats)


def query_scores(index: CarveIndex, query: str, embedder: Embedder) -> np.ndarray:
    """Cosine of every indexed image to the query prompt (zero-shot)."""
    if len(index) == 0:
        return np.zeros((0,), dtype=np.float32)
    prompt = f"a photo of {str(query or '').strip()}"
    tvec = embedder.embed_texts([prompt])  # (1, D)
    return (index.feats @ tvec[0]).astype(np.float32)


def select(
    scores: np.ndarray,
    *,
    threshold: float,
    max_positive: int = 0,
    max_negative: int = 0,
    negative_margin: float = 0.10,
) -> tuple[list[int], list[int]]:
    """Indices of positives (>= threshold) and negatives (clearly below it).

    Negatives are drawn from the lowest-scoring images so the ``not_<class>``
    class is unambiguous; capped to roughly balance the positives.
    """
    n = int(scores.shape[0])
    order = np.argsort(-scores)  # high -> low
    pos = [int(i) for i in order if scores[i] >= threshold]
    if max_positive:
        pos = pos[:max_positive]

    neg_cut = threshold - negative_margin
    neg_pool = [int(i) for i in order[::-1] if scores[i] < neg_cut]
    cap = max_negative or max(1, len(pos))
    neg = neg_pool[:cap]
    return pos, neg


def materialize_imagefolder(
    *,
    registry_dir: Path,
    slug: str,
    class_name: str,
    positive_paths: list[Path],
    negative_paths: list[Path],
    negative_class: str = "",
    copy: bool = True,
    query: str = "",
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Write ``<registry>/<slug>/<class>/`` + ``<neg_class>/`` ImageFolder dirs.

    Returns a summary dict. Files are copied (default) or hard-linked. Filenames
    are de-collided so two source images with the same name both survive.
    """
    slug = re.sub(r"[^a-z0-9_-]+", "_", str(slug or "").strip().lower()).strip("_")
    if not slug:
        raise ValueError("slug is required")
    pos_class = sanitize_class_name(class_name)
    neg_class = sanitize_class_name(negative_class or f"not_{pos_class}")
    if neg_class == pos_class:
        neg_class = f"not_{pos_class}"

    ds_root = Path(registry_dir).expanduser() / slug
    if ds_root.exists():
        raise FileExistsError(f"dataset already exists: {ds_root}")

    def _place(paths: list[Path], cls: str) -> int:
        out = ds_root / cls
        out.mkdir(parents=True, exist_ok=True)
        seen: dict[str, int] = {}
        placed = 0
        for src in paths:
            src = Path(src)
            if not src.is_file():
                continue
            stem, suffix = src.stem, src.suffix.lower() or ".jpg"
            n = seen.get(src.name, 0)
            seen[src.name] = n + 1
            name = f"{stem}{suffix}" if n == 0 else f"{stem}_{n}{suffix}"
            dst = out / name
            try:
                if copy:
                    shutil.copy2(src, dst)
                else:
                    dst.hardlink_to(src)
                placed += 1
            except Exception:
                try:
                    shutil.copy2(src, dst)
                    placed += 1
                except Exception:
                    pass
        return placed

    n_pos = _place(positive_paths, pos_class)
    n_neg = _place(negative_paths, neg_class)

    classes = [pos_class, neg_class]
    (ds_root / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    manifest = {
        "format": "imagefolder_classification",
        "slug": slug,
        "classes": classes,
        "counts": {pos_class: n_pos, neg_class: n_neg},
        "carve": {"query": query, "threshold": threshold},
        "created_at": time.time(),
        "source": "semantic_carve",
    }
    (ds_root / "carve_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["dataset_path"] = str(ds_root)
    return manifest
