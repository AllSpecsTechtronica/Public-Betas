"""
Shared utilities for all analyzers.

Statistics, embedding computation, subsampling, and common math.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple

from dataset_investigator.fingerprint import DistributionDescriptor


def describe_distribution(series: pd.Series, name: str = None) -> DistributionDescriptor:
    """
    Compute a full distribution descriptor for a pandas Series.
    Handles both numeric and categorical data.
    """
    name = name or str(series.name) or "unnamed"
    count = len(series)
    missing = int(series.isna().sum())
    missing_pct = round(missing / max(count, 1) * 100, 2)
    unique = int(series.nunique())

    desc = DistributionDescriptor(
        name=name,
        dtype=str(series.dtype),
        count=count,
        missing=missing,
        missing_pct=missing_pct,
        unique=unique,
    )

    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        if len(clean) > 0:
            desc.mean = _safe_float(clean.mean())
            desc.std = _safe_float(clean.std())
            desc.min = _safe_float(clean.min())
            desc.q25 = _safe_float(clean.quantile(0.25))
            desc.median = _safe_float(clean.median())
            desc.q75 = _safe_float(clean.quantile(0.75))
            desc.max = _safe_float(clean.max())
            desc.skewness = _safe_float(clean.skew())
            desc.kurtosis = _safe_float(clean.kurtosis())
    else:
        # Categorical / object
        top = series.value_counts().head(10)
        desc.top_values = [
            {"value": str(v), "count": int(c), "pct": round(c / max(count - missing, 1) * 100, 2)}
            for v, c in top.items()
        ]

    return desc


def compute_correlation_matrix(df: pd.DataFrame, max_features: int = 50) -> Tuple[
    Optional[List[List[float]]], Optional[List[str]], Optional[List[Dict[str, Any]]]
]:
    """
    Compute correlation matrix for numeric columns.

    Returns (matrix, feature_names, high_correlation_pairs).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return None, None, None

    # Subsample columns if too many
    if len(numeric_cols) > max_features:
        # Keep columns with highest variance
        variances = df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = variances.head(max_features).index.tolist()

    corr = df[numeric_cols].corr()

    # Extract high-correlation pairs (|r| > 0.7, excluding diagonal)
    high_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.7 and not np.isnan(r):
                high_pairs.append({
                    "feature_1": numeric_cols[i],
                    "feature_2": numeric_cols[j],
                    "correlation": round(float(r), 4),
                })

    high_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    matrix = corr.values.tolist()
    # Replace NaN with None for JSON serialization
    matrix = [[None if np.isnan(v) else round(v, 4) for v in row] for row in matrix]

    return matrix, numeric_cols, high_pairs


def compute_embedding_2d(X: np.ndarray, method: str = "auto",
                         n_samples_max: int = 10000, random_state: int = 42
                         ) -> Tuple[np.ndarray, str]:
    """
    Compute 2D embedding for visualization.

    Uses PCA for speed on large datasets, UMAP when available and n < threshold.
    Falls back gracefully.
    """
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 2:
        return np.zeros((X.shape[0], 2)), "none"

    # Subsample if needed
    n = X.shape[0]
    if n > n_samples_max:
        idx = np.random.RandomState(random_state).choice(n, n_samples_max, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
        idx = np.arange(n)

    # Handle NaN/inf
    X_clean = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    stds = X_clean.std(axis=0)
    stds[stds == 0] = 1.0
    X_norm = (X_clean - X_clean.mean(axis=0)) / stds

    if method == "auto":
        # Try UMAP first if dataset is manageable
        if n <= n_samples_max:
            try:
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=random_state, n_neighbors=min(15, n - 1))
                embedding = reducer.fit_transform(X_norm)
                return _expand_embedding(embedding, idx, n), "umap"
            except ImportError:
                pass

        # Fall back to PCA (always available via numpy)
        method = "pca"

    if method == "pca":
        embedding = _pca_2d(X_norm)
        return _expand_embedding(embedding, idx, n), "pca"

    return np.zeros((n, 2)), "none"


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """Simple PCA via SVD -- no sklearn dependency for the core path."""
    X_centered = X - X.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        return U[:, :2] * S[:2]
    except np.linalg.LinAlgError:
        return np.zeros((X.shape[0], 2))


def _expand_embedding(embedding: np.ndarray, idx: np.ndarray, n: int) -> np.ndarray:
    """If we subsampled, expand back to full size (zeros for non-sampled)."""
    if len(idx) == n:
        return embedding
    full = np.zeros((n, 2))
    full[idx] = embedding
    return full


def detect_anomalies_isolation_forest(X: np.ndarray, contamination: float = 0.05,
                                       max_samples: int = 10000, random_state: int = 42
                                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies using Isolation Forest.

    Returns (labels, scores) where labels are -1 for anomalies, 1 for normal.
    Scores are anomaly scores (higher = more anomalous).
    """
    from sklearn.ensemble import IsolationForest

    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n = X_clean.shape[0]
    if n < 10:
        return np.ones(n), np.zeros(n)

    clf = IsolationForest(
        contamination=contamination,
        max_samples=min(max_samples, n),
        random_state=random_state,
        n_jobs=-1,
    )
    labels = clf.fit_predict(X_clean)
    scores = -clf.decision_function(X_clean)  # negate so higher = more anomalous

    return labels, scores


def find_clusters(X: np.ndarray, max_clusters: int = 10,
                  random_state: int = 42) -> Tuple[np.ndarray, int, str]:
    """
    Find natural clusters using KMeans with silhouette-based k selection.

    Returns (labels, n_clusters, method).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n = X_clean.shape[0]
    if n < 10:
        return np.zeros(n, dtype=int), 1, "none"

    # Standardize
    stds = X_clean.std(axis=0)
    stds[stds == 0] = 1.0
    X_norm = (X_clean - X_clean.mean(axis=0)) / stds

    # If dataset is large, subsample for k selection
    if n > 5000:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, 5000, replace=False)
        X_sample = X_norm[idx]
    else:
        X_sample = X_norm

    # Try k from 2 to max_clusters, pick best silhouette
    best_k = 2
    best_score = -1
    k_range = range(2, min(max_clusters + 1, len(X_sample)))

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=5, max_iter=100)
        labs = km.fit_predict(X_sample)
        if len(set(labs)) < 2:
            continue
        score = silhouette_score(X_sample, labs, sample_size=min(2000, len(X_sample)))
        if score > best_score:
            best_score = score
            best_k = k

    # Fit on full data with best k
    km_final = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    labels = km_final.fit_predict(X_norm)

    return labels, best_k, "kmeans_silhouette"


def subsample_data(data, max_samples: int, random_state: int = 42):
    """Subsample DataFrame or array if larger than max_samples."""
    if isinstance(data, pd.DataFrame):
        if len(data) <= max_samples:
            return data, False
        return data.sample(n=max_samples, random_state=random_state).reset_index(drop=True), True
    elif isinstance(data, np.ndarray):
        if data.shape[0] <= max_samples:
            return data, False
        rng = np.random.RandomState(random_state)
        idx = rng.choice(data.shape[0], max_samples, replace=False)
        return data[idx], True
    return data, False


def _safe_float(v) -> Optional[float]:
    """Convert to float, returning None for non-finite values."""
    try:
        f = float(v)
        if np.isfinite(f):
            return round(f, 6)
        return None
    except (TypeError, ValueError):
        return None
