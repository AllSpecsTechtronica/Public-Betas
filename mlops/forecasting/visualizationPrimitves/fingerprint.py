"""
DatasetFingerprint - the universal investigation report schema.

Every analyzer populates the same structure regardless of archetype.
Fully serializable to JSON for storage in cvLayer's SQLite catalog.
"""

import json
import enum
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime


class Archetype(enum.Enum):
    TABULAR = "tabular"
    SEQUENCE = "sequence"
    IMAGE = "image"
    LOG = "log"


@dataclass
class DistributionDescriptor:
    """Statistical summary of a single feature or channel."""
    name: str
    dtype: str
    count: int
    missing: int
    missing_pct: float
    unique: int
    # Numeric stats (None for non-numeric)
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q25: Optional[float] = None
    median: Optional[float] = None
    q75: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    # Categorical stats (None for numeric)
    top_values: Optional[List[Dict[str, Any]]] = None


@dataclass
class ClusterInfo:
    """A discovered grouping in the data."""
    cluster_id: int
    size: int
    pct: float
    centroid: Optional[List[float]] = None
    label: Optional[str] = None  # auto-generated descriptive label
    representative_indices: Optional[List[int]] = None


@dataclass
class AnomalyInfo:
    """A flagged anomalous sample."""
    index: int
    score: float  # higher = more anomalous
    reason: str


@dataclass
class TemporalDecomposition:
    """Time-axis analysis results."""
    has_time_axis: bool = False
    time_column: Optional[str] = None
    time_range: Optional[Dict[str, str]] = None  # {start, end, duration}
    sampling_rate_hz: Optional[float] = None
    is_regular: Optional[bool] = None
    stationarity_pvalue: Optional[float] = None
    is_stationary: Optional[bool] = None
    n_changepoints: int = 0
    changepoint_indices: Optional[List[int]] = None
    dominant_frequencies_hz: Optional[List[float]] = None
    trend_direction: Optional[str] = None  # "increasing", "decreasing", "flat"
    seasonality_periods: Optional[List[float]] = None


@dataclass
class QualityIssue:
    """A specific data quality problem."""
    severity: str  # "critical", "warning", "info"
    category: str  # "missing", "duplicate", "imbalance", "outlier", "drift", "leakage"
    message: str
    affected_columns: Optional[List[str]] = None
    affected_rows: Optional[int] = None


@dataclass
class Recommendation:
    """Actionable suggestion before training."""
    priority: int  # 1 = do this first
    category: str  # "preprocessing", "feature_engineering", "sampling", "architecture"
    message: str
    details: Optional[str] = None


@dataclass
class DatasetFingerprint:
    """
    Universal investigation report.

    This is the single output type for all archetype analyzers.
    Serializable to JSON. Storable in cvLayer's catalog as a
    first-class entity attached to dataset nodes.
    """
    # -- Identity --
    archetype: Archetype
    source: str = ""  # file path, URL, or description
    investigated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # -- Shape --
    n_samples: int = 0
    n_features: int = 0  # columns for tabular, channels for sequence, HxW for image
    shape_raw: Optional[List[int]] = None
    memory_bytes: Optional[int] = None
    was_subsampled: bool = False
    subsample_n: Optional[int] = None

    # -- Distributions --
    distributions: List[DistributionDescriptor] = field(default_factory=list)

    # -- Correlation structure --
    correlation_matrix: Optional[List[List[float]]] = None  # dense, feature-indexed
    correlation_features: Optional[List[str]] = None
    high_correlation_pairs: Optional[List[Dict[str, Any]]] = None  # [{f1, f2, corr}]

    # -- Clusters --
    n_clusters: int = 0
    clusters: List[ClusterInfo] = field(default_factory=list)
    cluster_method: Optional[str] = None

    # -- Anomalies --
    n_anomalies: int = 0
    anomaly_pct: float = 0.0
    anomalies: List[AnomalyInfo] = field(default_factory=list)
    anomaly_method: Optional[str] = None

    # -- Temporal --
    temporal: TemporalDecomposition = field(default_factory=TemporalDecomposition)

    # -- Target analysis (if target_column specified) --
    target_column: Optional[str] = None
    target_type: Optional[str] = None  # "continuous", "binary", "multiclass"
    target_distribution: Optional[DistributionDescriptor] = None
    class_balance: Optional[Dict[str, float]] = None  # class -> pct
    feature_importance: Optional[List[Dict[str, Any]]] = None  # [{feature, importance}]

    # -- 2D embedding for visualization --
    embedding_2d: Optional[List[List[float]]] = None  # [[x, y], ...] per sample
    embedding_method: Optional[str] = None
    embedding_labels: Optional[List[Any]] = None  # cluster or class labels per point

    # -- Quality --
    quality_score: float = 0.0  # 0-1 composite
    quality_issues: List[QualityIssue] = field(default_factory=list)

    # -- Recommendations --
    recommendations: List[Recommendation] = field(default_factory=list)

    # -- Archetype-specific extras --
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dict (JSON-safe)."""
        d = asdict(self)
        d["archetype"] = self.archetype.value
        return d

    def to_json(self, indent=2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetFingerprint":
        """Reconstruct from dict."""
        d = d.copy()
        d["archetype"] = Archetype(d["archetype"])
        # Reconstruct nested dataclasses
        d["distributions"] = [DistributionDescriptor(**dd) for dd in d.get("distributions", [])]
        d["clusters"] = [ClusterInfo(**c) for c in d.get("clusters", [])]
        d["anomalies"] = [AnomalyInfo(**a) for a in d.get("anomalies", [])]
        d["temporal"] = TemporalDecomposition(**d.get("temporal", {}))
        d["quality_issues"] = [QualityIssue(**q) for q in d.get("quality_issues", [])]
        d["recommendations"] = [Recommendation(**r) for r in d.get("recommendations", [])]
        if d.get("target_distribution"):
            d["target_distribution"] = DistributionDescriptor(**d["target_distribution"])
        return cls(**d)
