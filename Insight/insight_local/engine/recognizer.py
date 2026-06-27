from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..runtime_profile import pick_torch_device


@dataclass
class RecognitionMatch:
    identity: str
    group_name: str
    similarity: float
    source_path: str


@dataclass
class IdentityMatch:
    identity: str
    group_name: str
    similarity: float
    source_path: str
    sample_count: int


@dataclass
class SimilaritySearchMatch:
    item_id: int
    display_name: str
    batch_label: str
    similarity: float
    source_path: str


class MobileNetV3Embedder:
    """
    Wraps MobileNetV3-Small as a frozen feature extractor.
    Weights download once (~9 MB) on first call to embed().
    All operations are thread-safe via an internal lock.
    """

    INPUT_SIZE = 224
    EMBEDDING_DIM = 576

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, weights_path: Optional[Path] = None) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._device = "cpu"
        self._ready = False
        self._load_error: Optional[str] = None
        self._weights_path = Path(weights_path).expanduser().resolve() if weights_path else None

    def _load(self) -> None:
        """Lazy-load model; called once under lock."""
        try:
            import torch
            import torchvision.models as tv_models

            self._device = self._pick_device()
            if self._weights_path is None or not self._weights_path.exists():
                raise FileNotFoundError(
                    f"Similarity model not found: {self._weights_path or 'unset'}"
                )

            full_model = tv_models.mobilenet_v3_small(weights=None)
            state_dict = torch.load(str(self._weights_path), map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                state_dict = state_dict["state_dict"]
            if not isinstance(state_dict, dict):
                raise RuntimeError("Similarity model weights are not a valid state_dict")
            full_model.load_state_dict(state_dict)
            full_model.eval()

            # Remove classifier, keep features + adaptive pool
            class _FeatureExtractor(torch.nn.Module):
                def __init__(self, base):
                    super().__init__()
                    self.features = base.features
                    self.avgpool = base.avgpool

                def forward(self, x):
                    x = self.features(x)
                    x = self.avgpool(x)
                    return x.flatten(1)

            self._model = _FeatureExtractor(full_model).to(self._device)
            for p in self._model.parameters():
                p.requires_grad_(False)

            self._torch = torch
            self._ready = True
        except Exception as exc:
            self._load_error = str(exc)
            self._ready = False

    @staticmethod
    def _pick_device() -> str:
        return pick_torch_device()

    def ensure_loaded(self) -> bool:
        """Returns True if model is ready. Thread-safe."""
        if self._ready:
            return True
        with self._lock:
            if not self._ready and self._load_error is None:
                self._load()
        return self._ready

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def embed(self, bgr_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Convert BGR crop to L2-normalised 576-dim embedding.
        Returns None if model not ready or image too small.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return None
        if not self.ensure_loaded():
            return None

        import torch

        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32) / 255.0

        mean = np.array(self._IMAGENET_MEAN, dtype=np.float32)
        std = np.array(self._IMAGENET_STD, dtype=np.float32)
        arr = (arr - mean) / std

        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self._device)

        with torch.no_grad(), self._lock:
            feat = self._model(tensor)

        vec = feat.cpu().numpy()[0].astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return None
        return vec / norm

    def embed_batch(self, bgr_crops: list[np.ndarray]) -> list[Optional[np.ndarray]]:
        """Embed multiple crops; returns list of embeddings (None for failures)."""
        return [self.embed(c) for c in bgr_crops]

    @property
    def weights_path(self) -> Optional[Path]:
        return self._weights_path


def cosine_search(
    query: np.ndarray,
    gallery_matrix: np.ndarray,
    identity_labels: list[str],
    group_labels: list[str],
    source_paths: list[str],
    top_k: int = 5,
    threshold: float = 0.72,
) -> list[RecognitionMatch]:
    """
    Cosine similarity search against a pre-built gallery matrix.

    gallery_matrix: (N, 576) float32, each row is L2-normalised.
    Returns top_k matches above threshold, sorted descending by similarity.
    """
    if gallery_matrix is None or gallery_matrix.shape[0] == 0:
        return []

    sims = gallery_matrix @ query  # cosine similarity (both L2-normalised)
    top_indices = np.argsort(sims)[::-1][:top_k]

    results: list[RecognitionMatch] = []
    for idx in top_indices:
        sim = float(sims[idx])
        if sim < threshold:
            break
        results.append(
            RecognitionMatch(
                identity=identity_labels[idx],
                group_name=group_labels[idx],
                similarity=round(sim, 4),
                source_path=source_paths[idx],
            )
        )
    return results


def aggregate_matches(matches: list[RecognitionMatch]) -> tuple[str, float]:
    """
    Aggregate top-k matches into a single identity verdict via weighted vote.
    Returns (identity_name, confidence) or ("unknown", 0.0).
    """
    if not matches:
        return "unknown", 0.0
    votes: dict[str, float] = {}
    for m in matches:
        votes[m.identity] = votes.get(m.identity, 0.0) + m.similarity
    best = max(votes, key=lambda k: votes[k])
    total = sum(votes.values())
    confidence = votes[best] / total if total > 0 else 0.0
    return best, round(confidence, 4)


def prototype_search(
    query: np.ndarray,
    profile_matrix: np.ndarray,
    identity_labels: list[str],
    group_labels: list[str],
    source_paths: list[str],
    sample_counts: list[int],
    top_k: int = 3,
) -> list[IdentityMatch]:
    """
    Identity-level search against one prototype embedding per enrolled identity.
    """
    if profile_matrix is None or profile_matrix.shape[0] == 0:
        return []

    sims = profile_matrix @ query
    top_indices = np.argsort(sims)[::-1][:top_k]
    results: list[IdentityMatch] = []
    for idx in top_indices:
        results.append(
            IdentityMatch(
                identity=identity_labels[idx],
                group_name=group_labels[idx],
                similarity=round(float(sims[idx]), 4),
                source_path=source_paths[idx],
                sample_count=int(sample_counts[idx]),
            )
        )
    return results


def similarity_search(
    query: np.ndarray,
    gallery_matrix: np.ndarray,
    item_ids: list[int],
    display_names: list[str],
    batch_labels: list[str],
    source_paths: list[str],
    *,
    exclude_item_id: int = 0,
    top_k: int = 12,
) -> list[SimilaritySearchMatch]:
    if gallery_matrix is None or gallery_matrix.shape[0] == 0:
        return []

    sims = gallery_matrix @ query
    top_indices = np.argsort(sims)[::-1]

    results: list[SimilaritySearchMatch] = []
    for idx in top_indices:
        item_id = int(item_ids[idx])
        if exclude_item_id and item_id == exclude_item_id:
            continue
        results.append(
            SimilaritySearchMatch(
                item_id=item_id,
                display_name=display_names[idx],
                batch_label=batch_labels[idx],
                similarity=round(float(sims[idx]), 4),
                source_path=source_paths[idx],
            )
        )
        if len(results) >= top_k:
            break
    return results


def decide_identity(
    raw_matches: list[RecognitionMatch],
    profile_matches: list[IdentityMatch],
    threshold: float,
    margin_threshold: float,
) -> tuple[str, float, dict[str, float | int | str | bool]]:
    """
    Combine sample-level and identity-level evidence into a single decision.
    """
    if not raw_matches or not profile_matches:
        return "unknown", 0.0, {
            "accepted": False,
            "vote_share": 0.0,
            "prototype_similarity": 0.0,
            "margin": 0.0,
            "support_count": 0,
            "reason": "no_match",
        }

    voted_identity, vote_share = aggregate_matches(raw_matches)
    best_profile = profile_matches[0]
    second_similarity = profile_matches[1].similarity if len(profile_matches) > 1 else 0.0
    margin = round(float(best_profile.similarity - second_similarity), 4)
    support_count = sum(1 for match in raw_matches if match.identity == voted_identity)
    top_similarity = float(raw_matches[0].similarity)

    accepted = (
        voted_identity == best_profile.identity
        and best_profile.similarity >= threshold
        and top_similarity >= threshold
        and margin >= margin_threshold
        and (
            len(raw_matches) == 1
            or vote_share >= 0.55
            or support_count >= 2
            or best_profile.sample_count <= 1
        )
    )

    confidence = (
        (best_profile.similarity * 0.5)
        + (top_similarity * 0.25)
        + (float(vote_share) * 0.25)
    )
    confidence = round(max(0.0, min(1.0, confidence)), 4)

    reason = "accepted"
    if voted_identity != best_profile.identity:
        reason = "prototype_vote_mismatch"
    elif best_profile.similarity < threshold or top_similarity < threshold:
        reason = "below_threshold"
    elif margin < margin_threshold:
        reason = "ambiguous_margin"
    elif len(raw_matches) > 1 and vote_share < 0.55 and support_count < 2 and best_profile.sample_count > 1:
        reason = "weak_support"

    return (
        voted_identity if accepted else "unknown",
        confidence if accepted else 0.0,
        {
            "accepted": accepted,
            "vote_share": round(float(vote_share), 4),
            "prototype_similarity": round(float(best_profile.similarity), 4),
            "margin": margin,
            "support_count": support_count,
            "reason": reason,
        },
    )
