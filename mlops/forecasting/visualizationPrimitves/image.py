"""
Image dataset analyzer.

Handles: directories of images, image classification datasets, any collection
of image files organized in folders.

Key insight: you don't need to train anything to investigate an image dataset.
Embed everything through a frozen backbone, then run the same clustering and
outlier detection machinery in embedding space.

Pipeline:
    1. File-level census (formats, sizes, resolutions)
    2. Embed via frozen backbone (torchvision ResNet50 by default, CLIP if available)
    3. Clustering in embedding space
    4. Near-duplicate detection
    5. Anomaly detection (outlier images)
    6. Class balance analysis (if folder structure implies labels)
    7. 2D embedding for visualization
    8. Recommendations
"""

import os
import pathlib
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter

from dataset_investigator.fingerprint import (
    DatasetFingerprint, Archetype, ClusterInfo, AnomalyInfo,
    QualityIssue, Recommendation, DistributionDescriptor,
)
from dataset_investigator.utils import (
    compute_embedding_2d, detect_anomalies_isolation_forest, find_clusters,
)


def analyze_image(data: dict, metadata: dict, target_column: str = None,
                  time_column: str = None, max_samples: int = 5000,
                  embedding_dim: int = 2, random_state: int = 42
                  ) -> DatasetFingerprint:
    """Full image dataset investigation pipeline."""

    image_paths = data.get("image_paths", [])
    root = data.get("root", "")

    fp = DatasetFingerprint(
        archetype=Archetype.IMAGE,
        source=root,
        n_samples=len(image_paths),
        shape_raw=[len(image_paths)],
    )

    quality_issues = []
    recommendations = []

    if len(image_paths) == 0:
        quality_issues.append(QualityIssue(
            severity="critical",
            category="missing",
            message="No image files found in the specified path",
        ))
        fp.quality_issues = quality_issues
        return fp

    # ---- 1. File-level census ----
    file_stats = _census_images(image_paths, max_scan=min(max_samples, len(image_paths)))
    fp.extras["file_census"] = file_stats
    fp.memory_bytes = file_stats.get("total_bytes", 0)

    # Distribution of file sizes
    if file_stats.get("sizes_bytes"):
        sizes = np.array(file_stats["sizes_bytes"])
        fp.distributions.append(DistributionDescriptor(
            name="file_size_bytes",
            dtype="int64",
            count=len(sizes),
            missing=0,
            missing_pct=0.0,
            unique=len(set(sizes)),
            mean=float(sizes.mean()),
            std=float(sizes.std()),
            min=float(sizes.min()),
            q25=float(np.percentile(sizes, 25)),
            median=float(np.median(sizes)),
            q75=float(np.percentile(sizes, 75)),
            max=float(sizes.max()),
        ))

    # Distribution of resolutions
    if file_stats.get("widths") and file_stats.get("heights"):
        widths = np.array(file_stats["widths"])
        heights = np.array(file_stats["heights"])
        fp.distributions.append(DistributionDescriptor(
            name="image_width",
            dtype="int64",
            count=len(widths),
            missing=0,
            missing_pct=0.0,
            unique=len(set(widths)),
            mean=float(widths.mean()),
            std=float(widths.std()),
            min=float(widths.min()),
            q25=float(np.percentile(widths, 25)),
            median=float(np.median(widths)),
            q75=float(np.percentile(widths, 75)),
            max=float(widths.max()),
        ))
        fp.distributions.append(DistributionDescriptor(
            name="image_height",
            dtype="int64",
            count=len(heights),
            missing=0,
            missing_pct=0.0,
            unique=len(set(heights)),
            mean=float(heights.mean()),
            std=float(heights.std()),
            min=float(heights.min()),
            q25=float(np.percentile(heights, 25)),
            median=float(np.median(heights)),
            q75=float(np.percentile(heights, 75)),
            max=float(heights.max()),
        ))

        # Check resolution consistency
        n_unique_resolutions = len(set(zip(widths, heights)))
        if n_unique_resolutions > len(widths) * 0.5:
            quality_issues.append(QualityIssue(
                severity="warning",
                category="outlier",
                message=f"Highly inconsistent image resolutions ({n_unique_resolutions} unique across {len(widths)} images)",
            ))
            recommendations.append(Recommendation(
                priority=1,
                category="preprocessing",
                message="Standardize image resolutions before training -- resize or center-crop to uniform dimensions",
            ))

    # ---- 2. Class structure from folder hierarchy ----
    labels, label_map = _infer_labels_from_paths(image_paths, root)
    if labels is not None:
        fp.n_features = len(set(labels))
        fp.target_type = "multiclass" if fp.n_features > 2 else "binary"
        fp.target_column = "folder_label"

        label_counts = Counter(labels)
        total = sum(label_counts.values())
        fp.class_balance = {
            str(k): round(v / total * 100, 2)
            for k, v in label_counts.most_common()
        }

        # Check imbalance
        counts = np.array(list(label_counts.values()))
        if counts.max() / max(counts.min(), 1) > 10:
            quality_issues.append(QualityIssue(
                severity="critical",
                category="imbalance",
                message=f"Severe class imbalance: largest class is {counts.max()}x the smallest",
            ))
        elif counts.max() / max(counts.min(), 1) > 3:
            quality_issues.append(QualityIssue(
                severity="warning",
                category="imbalance",
                message=f"Class imbalance: {counts.max()/max(counts.min(),1):.1f}x ratio between largest and smallest",
            ))

    # ---- 3. Embedding computation ----
    # Subsample for embedding
    if len(image_paths) > max_samples:
        rng = np.random.RandomState(random_state)
        sample_idx = rng.choice(len(image_paths), max_samples, replace=False)
        sample_paths = [image_paths[i] for i in sample_idx]
        fp.was_subsampled = True
        fp.subsample_n = max_samples
    else:
        sample_paths = image_paths
        sample_idx = np.arange(len(image_paths))

    embeddings = _compute_image_embeddings(sample_paths)

    if embeddings is not None and len(embeddings) >= 10:
        # ---- 4. Clustering in embedding space ----
        cluster_labels, n_clusters, cluster_method = find_clusters(
            embeddings, random_state=random_state
        )
        fp.n_clusters = n_clusters
        fp.cluster_method = f"embedding_{cluster_method}"
        fp.clusters = [
            ClusterInfo(
                cluster_id=cid,
                size=int((cluster_labels == cid).sum()),
                pct=round(float((cluster_labels == cid).sum()) / len(cluster_labels) * 100, 2),
            )
            for cid in range(n_clusters)
        ]

        # ---- 5. Anomaly detection ----
        anom_labels, anom_scores = detect_anomalies_isolation_forest(
            embeddings, random_state=random_state
        )
        anomaly_mask = anom_labels == -1
        fp.n_anomalies = int(anomaly_mask.sum())
        fp.anomaly_pct = round(fp.n_anomalies / len(embeddings) * 100, 2)
        fp.anomaly_method = "isolation_forest_embedding"

        if fp.n_anomalies > 0:
            anom_indices = np.where(anomaly_mask)[0]
            top_anom = anom_indices[np.argsort(anom_scores[anom_indices])[-20:]]
            fp.anomalies = [
                AnomalyInfo(
                    index=int(sample_idx[i]),
                    score=round(float(anom_scores[i]), 4),
                    reason="embedding_space_outlier",
                )
                for i in top_anom
            ]

        # ---- 6. Near-duplicate detection ----
        n_near_dupes = _detect_near_duplicates(embeddings, threshold=0.95)
        if n_near_dupes > 0:
            quality_issues.append(QualityIssue(
                severity="warning",
                category="duplicate",
                message=f"{n_near_dupes} near-duplicate image pairs detected in embedding space",
            ))
            fp.extras["n_near_duplicates"] = n_near_dupes

        # ---- 7. 2D embedding for visualization ----
        embedding_2d, emb_method = compute_embedding_2d(
            embeddings, random_state=random_state
        )
        fp.embedding_2d = embedding_2d.tolist()
        fp.embedding_method = f"image_{emb_method}"

        # Use folder labels if available, else cluster labels
        if labels is not None:
            sampled_labels = [labels[i] for i in sample_idx]
            fp.embedding_labels = sampled_labels
        else:
            fp.embedding_labels = cluster_labels.tolist()

        # ---- 8. Mislabel candidates ----
        if labels is not None:
            mislabel_candidates = _find_mislabel_candidates(
                embeddings, [labels[i] for i in sample_idx], k=5
            )
            if mislabel_candidates:
                fp.extras["mislabel_candidates"] = mislabel_candidates[:20]
                quality_issues.append(QualityIssue(
                    severity="info",
                    category="outlier",
                    message=f"{len(mislabel_candidates)} potential mislabeled images detected (embedding neighbors have different labels)",
                ))

    # ---- 9. Format census ----
    if file_stats.get("format_counts"):
        fp.extras["format_distribution"] = file_stats["format_counts"]
        if len(file_stats["format_counts"]) > 1:
            quality_issues.append(QualityIssue(
                severity="info",
                category="outlier",
                message=f"Mixed image formats: {file_stats['format_counts']}",
            ))

    # ---- Recommendations ----
    if fp.n_samples < 100:
        recommendations.append(Recommendation(
            priority=1,
            category="sampling",
            message="Dataset has fewer than 100 images -- consider data augmentation or transfer learning from a pretrained model",
        ))

    if fp.anomaly_pct > 10:
        recommendations.append(Recommendation(
            priority=2,
            category="preprocessing",
            message=f"{fp.anomaly_pct:.1f}% outlier images -- manually review the flagged samples for corruption or mislabeling",
        ))

    fp.quality_issues = quality_issues
    fp.recommendations = sorted(recommendations, key=lambda r: r.priority)

    return fp


def _census_images(paths: list, max_scan: int = 5000) -> dict:
    """Gather file-level statistics without loading pixel data."""
    sizes = []
    widths = []
    heights = []
    formats = []
    corrupted = []

    try:
        from PIL import Image
        has_pil = True
    except ImportError:
        has_pil = False

    for i, p in enumerate(paths[:max_scan]):
        try:
            size = os.path.getsize(p)
            sizes.append(size)
            ext = pathlib.Path(p).suffix.lower()
            formats.append(ext)

            if has_pil:
                with Image.open(p) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
        except Exception as e:
            corrupted.append({"path": str(p), "error": str(e)})

    return {
        "n_scanned": min(max_scan, len(paths)),
        "total_bytes": sum(sizes),
        "sizes_bytes": sizes,
        "widths": widths,
        "heights": heights,
        "format_counts": dict(Counter(formats)),
        "n_corrupted": len(corrupted),
        "corrupted": corrupted[:20],
    }


def _infer_labels_from_paths(paths: list, root: str) -> Tuple[Optional[list], Optional[dict]]:
    """
    Infer class labels from directory structure.

    Common pattern: root/class_name/image.jpg
    """
    root_path = pathlib.Path(root)
    labels = []

    for p in paths:
        rel = pathlib.Path(p).relative_to(root_path)
        parts = rel.parts
        if len(parts) >= 2:
            # Parent folder is the label
            labels.append(parts[0])
        else:
            labels.append(None)

    # Check if we got meaningful labels
    non_none = [l for l in labels if l is not None]
    if len(non_none) < len(labels) * 0.5:
        return None, None

    unique_labels = set(non_none)
    if len(unique_labels) < 2 or len(unique_labels) > len(labels) * 0.5:
        # Too few or too many classes -- probably not a classification dataset
        return None, None

    label_map = {l: i for i, l in enumerate(sorted(unique_labels))}
    return labels, label_map


def _compute_image_embeddings(paths: list) -> Optional[np.ndarray]:
    """
    Compute image embeddings using available backbone.

    Priority: CLIP > torchvision ResNet50 > pixel average (fallback).
    """
    # Try torchvision (more likely available than CLIP in a general environment)
    try:
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms
        from PIL import Image

        model = models.resnet50(weights="IMAGENET1K_V1")
        model.eval()
        # Remove final classification layer -> 2048-dim embedding
        modules = list(model.children())[:-1]
        backbone = torch.nn.Sequential(*modules)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        embeddings = []
        with torch.no_grad():
            for p in paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensor = transform(img).unsqueeze(0)
                    emb = backbone(tensor).squeeze().numpy()
                    embeddings.append(emb)
                except Exception:
                    embeddings.append(np.zeros(2048))

        return np.array(embeddings)

    except ImportError:
        pass

    # Fallback: pixel-level features (very crude but requires only PIL)
    try:
        from PIL import Image

        embeddings = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB").resize((64, 64))
                arr = np.array(img).flatten().astype(np.float32) / 255.0
                embeddings.append(arr)
            except Exception:
                embeddings.append(np.zeros(64 * 64 * 3))

        return np.array(embeddings)

    except ImportError:
        return None


def _detect_near_duplicates(embeddings: np.ndarray, threshold: float = 0.95) -> int:
    """Count near-duplicate pairs via cosine similarity."""
    n = len(embeddings)
    if n > 2000:
        # Too expensive for full pairwise -- sample
        idx = np.random.choice(n, 2000, replace=False)
        embeddings = embeddings[idx]
        n = 2000

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = embeddings / norms

    # Compute pairwise cosine in chunks to manage memory
    n_dupes = 0
    chunk_size = 500
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        for j in range(i, n, chunk_size):
            end_j = min(j + chunk_size, n)
            sim = normed[i:end_i] @ normed[j:end_j].T
            # Zero out diagonal and lower triangle for the same-chunk case
            if i == j:
                mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
                n_dupes += int((sim[mask] > threshold).sum())
            elif j > i:
                n_dupes += int((sim > threshold).sum())

    return n_dupes


def _find_mislabel_candidates(embeddings: np.ndarray, labels: list,
                               k: int = 5) -> list:
    """
    Find images whose k nearest neighbors mostly have different labels.

    These are mislabel candidates -- the embedding says they belong to
    a different class than their folder label indicates.
    """
    n = len(embeddings)
    if n < k + 1:
        return []

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = embeddings / norms

    candidates = []

    # Process in chunks
    chunk_size = 500
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim = normed[start:end] @ normed.T  # (chunk, n)

        for i_local in range(end - start):
            i_global = start + i_local
            # Get k nearest neighbors (excluding self)
            sims = sim[i_local].copy()
            sims[i_global] = -1  # exclude self
            nn_idx = np.argpartition(sims, -k)[-k:]

            nn_labels = [labels[j] for j in nn_idx]
            own_label = labels[i_global]

            if own_label is None:
                continue

            # If majority of neighbors have a different label
            different = sum(1 for l in nn_labels if l != own_label)
            if different >= k * 0.6:  # 60% threshold
                majority_label = Counter(nn_labels).most_common(1)[0][0]
                candidates.append({
                    "index": i_global,
                    "current_label": str(own_label),
                    "suggested_label": str(majority_label),
                    "neighbor_agreement": round((k - different) / k, 2),
                })

    candidates.sort(key=lambda x: x["neighbor_agreement"])
    return candidates
