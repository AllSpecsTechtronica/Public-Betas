"""Shared CLIP image/text embedding helpers for CV Ops.

Used by the semantic-carve feature (turn a folder of images into a labeled
ImageFolder dataset via a text query) and by the read-only archive probes under
``Insight/tools``. Kept dependency-light and lazy: ``open_clip`` and ``torch``
are only imported when an embedder is actually built, so importing this module
never pulls the heavy stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# Image suffixes CLIP/PIL can decode in this corpus.
EMBED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ClipEmbedder:
    """Lazy CLIP wrapper: ``embed_images(paths)`` and ``embed_texts(strings)``
    both return L2-normalized float32 arrays so cosine == dot product."""

    def __init__(self, device: str = "auto") -> None:
        import open_clip
        import torch

        self._torch = torch
        self.device = pick_device(device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        self._model = model.eval().to(self.device)
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

    def embed_images(
        self,
        paths: Iterable[Path],
        *,
        batch_size: int = 32,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[np.ndarray, list[Path]]:
        from PIL import Image

        torch = self._torch
        paths = list(paths)
        feats: list[np.ndarray] = []
        kept: list[Path] = []
        batch, batch_paths = [], []

        def _flush() -> None:
            if not batch:
                return
            with torch.no_grad():
                x = torch.stack(batch).to(self.device)
                f = self._model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy().astype(np.float32))
            kept.extend(batch_paths)
            batch.clear()
            batch_paths.clear()

        for i, p in enumerate(paths):
            try:
                img = Image.open(p).convert("RGB")
                batch.append(self._preprocess(img))
                batch_paths.append(p)
            except Exception:
                continue
            if len(batch) >= batch_size:
                _flush()
                if progress_cb:
                    progress_cb(len(kept), len(paths))
        _flush()
        if progress_cb:
            progress_cb(len(kept), len(paths))
        if not feats:
            return np.zeros((0, 512), dtype=np.float32), []
        return np.concatenate(feats, axis=0), kept

    def embed_texts(self, prompts: list[str]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            toks = self._tokenizer(prompts).to(self.device)
            tf = self._model.encode_text(toks)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        return tf.cpu().numpy().astype(np.float32)


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and not p.name.startswith("._") and p.suffix.lower() in EMBED_IMAGE_SUFFIXES
    )
