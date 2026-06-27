"""
Tabular data analyzer.

Handles: CSV, DataFrames, any row-is-observation / column-is-feature structure.
This is the most general archetype and the fallback for ambiguous data.

Pipeline:
    1. Distribution profiling per column
    2. Correlation structure
    3. Clustering (KMeans with silhouette selection)
    4. Anomaly detection (Isolation Forest)
    5. 2D embedding (PCA/UMAP)
    6. Target analysis if target column specified
    7. Duplicate detection
    8. Recommendations
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from dataset_investigator.fingerprint import (
    DatasetFingerprint, Archetype, ClusterInfo, AnomalyInfo,
    QualityIssue, Recommendation, TemporalDecomposition,
)
from dataset_investigator.utils import (
    describe_distribution, compute_correlation_matrix, compute_embedding_2d,
    detect_anomalies_isolation_forest, find_clusters, subsample_data,
)


def analyze_tabular(data, metadata: dict, target_column: str = None,
                    time_column: str = None, max_samples: int = 50000,
                    embedding_dim: int = 2, random_state: int = 42
                    ) -> DatasetFingerprint:
    """Full tabular investigation pipeline."""

    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    fp = DatasetFingerprint(
        archetype=Archetype.TABULAR,
        source=metadata.get("source", ""),
        n_samples=len(df),
        n_features=len(df.columns),
        shape_raw=list(df.shape),
        memory_bytes=metadata.get("memory_bytes") or int(df.memory_usage(deep=True).sum()),
    )

    # Subsample if necessary
    df_work, was_subsampled = subsample_data(df, max_samples, random_state)
    fp.was_subsampled = was_subsampled
    fp.subsample_n = len(df_work) if was_subsampled else None

    quality_issues = []
    recommendations = []

    # ---- 1. Distribution profiling ----
    distributions = []
    for col in df_work.columns:
        distributions.append(describe_distribution(df_work[col], name=str(col)))
    fp.distributions = distributions

    # Flag high-missing columns
    for dist in distributions:
        if dist.missing_pct > 50:
            quality_issues.append(QualityIssue(
                severity="critical",
                category="missing",
                message=f"Column '{dist.name}' has {dist.missing_pct:.1f}% missing values",
                affected_columns=[dist.name],
            ))
        elif dist.missing_pct > 10:
            quality_issues.append(QualityIssue(
                severity="warning",
                category="missing",
                message=f"Column '{dist.name}' has {dist.missing_pct:.1f}% missing values",
                affected_columns=[dist.name],
            ))

    # ---- 2. Duplicate detection ----
    n_dupes = int(df_work.duplicated().sum())
    if n_dupes > 0:
        dupe_pct = round(n_dupes / len(df_work) * 100, 2)
        quality_issues.append(QualityIssue(
            severity="warning" if dupe_pct < 5 else "critical",
            category="duplicate",
            message=f"{n_dupes} duplicate rows ({dupe_pct}%)",
            affected_rows=n_dupes,
        ))

    # ---- 3. Correlation structure ----
    corr_matrix, corr_features, high_pairs = compute_correlation_matrix(df_work)
    fp.correlation_matrix = corr_matrix
    fp.correlation_features = corr_features
    fp.high_correlation_pairs = high_pairs

    if high_pairs:
        # Flag potential multicollinearity
        extreme_pairs = [p for p in high_pairs if abs(p["correlation"]) > 0.95]
        if extreme_pairs:
            cols = set()
            for p in extreme_pairs:
                cols.add(p["feature_1"])
                cols.add(p["feature_2"])
            quality_issues.append(QualityIssue(
                severity="warning",
                category="leakage",
                message=f"{len(extreme_pairs)} feature pairs with |r| > 0.95 -- possible redundancy or leakage",
                affected_columns=list(cols),
            ))

    # ---- 4. Clustering ----
    numeric_df = df_work.select_dtypes(include=[np.number])
    if len(numeric_df.columns) >= 2 and len(numeric_df) >= 10:
        X_numeric = numeric_df.fillna(0).values
        cluster_labels, n_clusters, cluster_method = find_clusters(
            X_numeric, random_state=random_state
        )
        fp.n_clusters = n_clusters
        fp.cluster_method = cluster_method

        clusters = []
        for cid in range(n_clusters):
            mask = cluster_labels == cid
            size = int(mask.sum())
            centroid = X_numeric[mask].mean(axis=0).tolist()
            # Find representative sample (closest to centroid)
            if size > 0:
                dists = np.linalg.norm(X_numeric[mask] - centroid, axis=1)
                rep_idx = np.where(mask)[0][np.argsort(dists)[:3]].tolist()
            else:
                rep_idx = []
            clusters.append(ClusterInfo(
                cluster_id=cid,
                size=size,
                pct=round(size / len(df_work) * 100, 2),
                centroid=centroid[:10],  # cap centroid length for serialization
                representative_indices=rep_idx,
            ))
        fp.clusters = clusters

        # ---- 5. Anomaly detection ----
        anom_labels, anom_scores = detect_anomalies_isolation_forest(
            X_numeric, random_state=random_state
        )
        anomaly_mask = anom_labels == -1
        fp.n_anomalies = int(anomaly_mask.sum())
        fp.anomaly_pct = round(fp.n_anomalies / len(df_work) * 100, 2)
        fp.anomaly_method = "isolation_forest"

        # Store top anomalies
        if fp.n_anomalies > 0:
            anom_indices = np.where(anomaly_mask)[0]
            top_anom = anom_indices[np.argsort(anom_scores[anom_indices])[-20:]]
            fp.anomalies = [
                AnomalyInfo(
                    index=int(i),
                    score=round(float(anom_scores[i]), 4),
                    reason="isolation_forest_outlier",
                )
                for i in top_anom
            ]

        # ---- 6. 2D Embedding ----
        embedding, emb_method = compute_embedding_2d(
            X_numeric, random_state=random_state
        )
        fp.embedding_2d = embedding.tolist()
        fp.embedding_method = emb_method
        fp.embedding_labels = cluster_labels.tolist()

    # ---- 7. Target analysis ----
    if target_column and target_column in df_work.columns:
        fp.target_column = target_column
        target_series = df_work[target_column]
        fp.target_distribution = describe_distribution(target_series, name=target_column)

        if pd.api.types.is_numeric_dtype(target_series):
            unique_vals = target_series.nunique()
            if unique_vals <= 2:
                fp.target_type = "binary"
            elif unique_vals <= 20:
                fp.target_type = "multiclass"
            else:
                fp.target_type = "continuous"
        else:
            unique_vals = target_series.nunique()
            fp.target_type = "binary" if unique_vals <= 2 else "multiclass"

        # Class balance for classification
        if fp.target_type in ("binary", "multiclass"):
            vc = target_series.value_counts(normalize=True)
            fp.class_balance = {str(k): round(float(v) * 100, 2) for k, v in vc.items()}

            # Check imbalance
            if vc.max() > 0.9:
                quality_issues.append(QualityIssue(
                    severity="critical",
                    category="imbalance",
                    message=f"Severe class imbalance: majority class is {vc.max()*100:.1f}%",
                    affected_columns=[target_column],
                ))
            elif vc.max() > 0.7:
                quality_issues.append(QualityIssue(
                    severity="warning",
                    category="imbalance",
                    message=f"Class imbalance: majority class is {vc.max()*100:.1f}%",
                    affected_columns=[target_column],
                ))

        # Feature importance via mutual information (fast, model-free)
        if fp.target_type and len(numeric_df.columns) >= 2:
            fp.feature_importance = _compute_feature_importance(
                df_work, target_column, fp.target_type
            )

    # ---- 8. Temporal analysis ----
    fp.temporal = _check_temporal(df_work, time_column)

    # ---- 9. Recommendations ----
    recommendations.extend(_generate_recommendations(fp, quality_issues))

    fp.quality_issues = quality_issues
    fp.recommendations = sorted(recommendations, key=lambda r: r.priority)

    return fp


def _compute_feature_importance(df: pd.DataFrame, target_col: str,
                                target_type: str) -> list:
    """Compute feature importance via mutual information."""
    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
        if len(numeric_cols) == 0:
            return []

        X = df[numeric_cols].fillna(0).values
        y = df[target_col]

        if target_type == "continuous":
            mi = mutual_info_regression(X, y, random_state=42)
        else:
            y_encoded = pd.factorize(y)[0]
            mi = mutual_info_classif(X, y_encoded, random_state=42)

        importance = sorted(
            [{"feature": col, "importance": round(float(score), 4)}
             for col, score in zip(numeric_cols, mi)],
            key=lambda x: x["importance"],
            reverse=True,
        )
        return importance[:30]  # top 30

    except ImportError:
        return []


def _check_temporal(df: pd.DataFrame, time_column: str = None) -> TemporalDecomposition:
    """Check for temporal structure in the data."""
    td = TemporalDecomposition()

    # Find time column
    time_col = time_column
    if time_col is None:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                time_col = str(col)
                break
        if time_col is None:
            # Try common names
            for candidate in ["timestamp", "time", "date", "datetime", "created_at", "ts"]:
                for col in df.columns:
                    if str(col).lower().strip() == candidate:
                        try:
                            pd.to_datetime(df[col].head(20))
                            time_col = str(col)
                            break
                        except (ValueError, TypeError):
                            pass
                if time_col:
                    break

    if time_col is None:
        return td

    td.has_time_axis = True
    td.time_column = time_col

    try:
        time_series = pd.to_datetime(df[time_col])
        td.time_range = {
            "start": str(time_series.min()),
            "end": str(time_series.max()),
            "duration": str(time_series.max() - time_series.min()),
        }

        # Sampling rate
        diffs = time_series.diff().dropna()
        if len(diffs) > 0:
            median_diff = diffs.median()
            if median_diff.total_seconds() > 0:
                td.sampling_rate_hz = round(1.0 / median_diff.total_seconds(), 4)
                # Check regularity (CV of intervals)
                cv = diffs.dt.total_seconds().std() / max(diffs.dt.total_seconds().mean(), 1e-10)
                td.is_regular = cv < 0.1
    except (ValueError, TypeError):
        pass

    return td


def _generate_recommendations(fp: DatasetFingerprint, issues: list) -> list:
    """Generate actionable recommendations based on findings."""
    recs = []
    priority = 1

    # Missing data
    critical_missing = [i for i in issues if i.category == "missing" and i.severity == "critical"]
    if critical_missing:
        cols = []
        for i in critical_missing:
            if i.affected_columns:
                cols.extend(i.affected_columns)
        recs.append(Recommendation(
            priority=priority,
            category="preprocessing",
            message="Drop or impute columns with >50% missing values before training",
            details=f"Affected columns: {', '.join(cols[:10])}",
        ))
        priority += 1

    # Duplicates
    dupe_issues = [i for i in issues if i.category == "duplicate"]
    if dupe_issues:
        recs.append(Recommendation(
            priority=priority,
            category="preprocessing",
            message="Remove duplicate rows to prevent data leakage in train/test splits",
        ))
        priority += 1

    # High correlation
    leakage_issues = [i for i in issues if i.category == "leakage"]
    if leakage_issues:
        recs.append(Recommendation(
            priority=priority,
            category="feature_engineering",
            message="Review highly correlated features -- consider dropping redundant ones or investigating leakage",
        ))
        priority += 1

    # Class imbalance
    imbalance_issues = [i for i in issues if i.category == "imbalance"]
    if imbalance_issues:
        recs.append(Recommendation(
            priority=priority,
            category="sampling",
            message="Address class imbalance via stratified sampling, SMOTE, or class weights",
        ))
        priority += 1

    # Anomalies
    if fp.anomaly_pct > 5:
        recs.append(Recommendation(
            priority=priority,
            category="preprocessing",
            message=f"{fp.anomaly_pct:.1f}% anomalies detected -- review before training or use robust loss functions",
        ))
        priority += 1

    # General
    if fp.n_features > 100:
        recs.append(Recommendation(
            priority=priority,
            category="feature_engineering",
            message="High dimensionality -- consider PCA, feature selection, or regularized models",
        ))
        priority += 1

    return recs
