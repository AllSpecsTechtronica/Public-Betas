"""
Log data analyzer.

Handles: system logs, application logs, any semi-structured text with
timestamps and repetitive structural patterns.

Key insight: log data becomes tabular once you extract templates.
The Drain algorithm identifies structural templates, then each log line
becomes a (timestamp, template_id, parameter_values) tuple.
From there, the tabular/temporal machinery applies.

Pipeline:
    1. Timestamp extraction and parsing
    2. Log level detection
    3. Template extraction (simplified Drain)
    4. Event frequency time-series construction
    5. Anomaly detection on event frequency patterns
    6. Temporal analysis (burst detection, periodicity)
    7. Parameter analysis per template
    8. Recommendations
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter, defaultdict

from dataset_investigator.fingerprint import (
    DatasetFingerprint, Archetype, ClusterInfo, AnomalyInfo,
    QualityIssue, Recommendation, TemporalDecomposition,
    DistributionDescriptor,
)
from dataset_investigator.utils import (
    describe_distribution, compute_embedding_2d,
    detect_anomalies_isolation_forest, find_clusters,
)


# Common timestamp patterns
TIMESTAMP_PATTERNS = [
    # ISO 8601
    (r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)", "%Y-%m-%dT%H:%M:%S"),
    # Common syslog
    (r"([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})", "%b %d %H:%M:%S"),
    # Apache/nginx
    (r"\[(\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4})\]", "%d/%b/%Y:%H:%M:%S %z"),
    # Simple date-time
    (r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", "%Y-%m-%d %H:%M:%S"),
    # Epoch seconds (10 digits)
    (r"(\b\d{10}\b)", "epoch_s"),
    # Epoch milliseconds (13 digits)
    (r"(\b\d{13}\b)", "epoch_ms"),
]

LOG_LEVELS = {"TRACE", "DEBUG", "INFO", "WARN", "WARNING", "ERROR", "FATAL", "CRITICAL", "NOTICE", "SEVERE"}
LOG_LEVEL_PATTERN = re.compile(r"\b(" + "|".join(LOG_LEVELS) + r")\b", re.IGNORECASE)


def analyze_log(data, metadata: dict, target_column: str = None,
                time_column: str = None, max_samples: int = 50000,
                embedding_dim: int = 2, random_state: int = 42
                ) -> DatasetFingerprint:
    """Full log investigation pipeline."""

    # Normalize to list of raw lines
    if isinstance(data, pd.DataFrame):
        # Find the text column
        text_col = _find_text_column(data)
        if text_col is None:
            # Fall back: concatenate all columns
            lines = data.astype(str).apply(lambda row: " ".join(row), axis=1).tolist()
        else:
            lines = data[text_col].astype(str).tolist()
    elif isinstance(data, (list, np.ndarray)):
        lines = [str(l) for l in data]
    else:
        lines = str(data).split("\n")

    fp = DatasetFingerprint(
        archetype=Archetype.LOG,
        source=metadata.get("source", ""),
        n_samples=len(lines),
        shape_raw=[len(lines)],
    )

    quality_issues = []
    recommendations = []

    if len(lines) == 0:
        quality_issues.append(QualityIssue(
            severity="critical",
            category="missing",
            message="No log lines found",
        ))
        fp.quality_issues = quality_issues
        return fp

    # Subsample if needed
    was_subsampled = False
    if len(lines) > max_samples:
        # Take contiguous block from the middle for temporal coherence
        start = (len(lines) - max_samples) // 2
        lines = lines[start:start + max_samples]
        was_subsampled = True
    fp.was_subsampled = was_subsampled
    fp.subsample_n = len(lines) if was_subsampled else None

    # ---- 1. Timestamp extraction ----
    timestamps, ts_pattern_used = _extract_timestamps(lines)
    has_timestamps = timestamps is not None and len(timestamps) > len(lines) * 0.3

    td = TemporalDecomposition(has_time_axis=has_timestamps)

    if has_timestamps:
        valid_ts = [t for t in timestamps if t is not None and pd.notna(t)]
        if valid_ts:
            td.time_column = "extracted_timestamp"
            td.time_range = {
                "start": str(min(valid_ts)),
                "end": str(max(valid_ts)),
                "duration": str(max(valid_ts) - min(valid_ts)),
            }
            # Estimate sampling rate
            ts_series = pd.Series(valid_ts).sort_values()
            diffs = ts_series.diff().dropna().dt.total_seconds()
            if len(diffs) > 0 and diffs.median() > 0:
                td.sampling_rate_hz = round(1.0 / diffs.median(), 4)
                td.is_regular = (diffs.std() / max(diffs.mean(), 1e-10)) < 0.5

        fp.extras["timestamp_coverage"] = round(len(valid_ts) / len(lines) * 100, 2)
        fp.extras["timestamp_pattern"] = ts_pattern_used

    fp.temporal = td

    # ---- 2. Log level detection ----
    levels = _extract_log_levels(lines)
    level_counts = Counter(levels)

    if None in level_counts:
        unlabeled = level_counts.pop(None)
        fp.extras["unlabeled_lines"] = unlabeled

    if level_counts:
        total_labeled = sum(level_counts.values())
        fp.extras["log_level_distribution"] = {
            k: {"count": v, "pct": round(v / total_labeled * 100, 2)}
            for k, v in level_counts.most_common()
        }

        # Flag high error rates
        error_count = sum(v for k, v in level_counts.items()
                          if k in {"ERROR", "FATAL", "CRITICAL", "SEVERE"})
        if total_labeled > 0:
            error_pct = error_count / total_labeled * 100
            if error_pct > 20:
                quality_issues.append(QualityIssue(
                    severity="critical",
                    category="outlier",
                    message=f"High error rate: {error_pct:.1f}% of log lines are ERROR/FATAL/CRITICAL",
                ))
            elif error_pct > 5:
                quality_issues.append(QualityIssue(
                    severity="warning",
                    category="outlier",
                    message=f"Elevated error rate: {error_pct:.1f}% of log lines are ERROR/FATAL/CRITICAL",
                ))

    # ---- 3. Template extraction (simplified Drain) ----
    templates, template_ids = _extract_templates(lines, max_templates=200)

    fp.n_features = len(templates)
    fp.extras["n_templates"] = len(templates)
    fp.extras["top_templates"] = [
        {"id": tid, "template": tpl, "count": cnt}
        for tid, (tpl, cnt) in enumerate(
            sorted(templates.items(), key=lambda x: x[1], reverse=True)[:20]
        )
    ]

    # Template frequency distribution
    template_counts = Counter(template_ids)
    count_values = list(template_counts.values())
    if count_values:
        fp.distributions.append(DistributionDescriptor(
            name="template_frequency",
            dtype="int64",
            count=len(count_values),
            missing=0,
            missing_pct=0.0,
            unique=len(set(count_values)),
            mean=float(np.mean(count_values)),
            std=float(np.std(count_values)),
            min=float(np.min(count_values)),
            q25=float(np.percentile(count_values, 25)),
            median=float(np.median(count_values)),
            q75=float(np.percentile(count_values, 75)),
            max=float(np.max(count_values)),
            skewness=float(pd.Series(count_values).skew()) if len(count_values) > 2 else None,
            kurtosis=float(pd.Series(count_values).kurtosis()) if len(count_values) > 2 else None,
        ))

    # Line length distribution
    line_lengths = [len(l) for l in lines]
    fp.distributions.append(DistributionDescriptor(
        name="line_length",
        dtype="int64",
        count=len(line_lengths),
        missing=0,
        missing_pct=0.0,
        unique=len(set(line_lengths)),
        mean=float(np.mean(line_lengths)),
        std=float(np.std(line_lengths)),
        min=float(np.min(line_lengths)),
        q25=float(np.percentile(line_lengths, 25)),
        median=float(np.median(line_lengths)),
        q75=float(np.percentile(line_lengths, 75)),
        max=float(np.max(line_lengths)),
    ))

    # ---- 4. Event frequency time-series ----
    if has_timestamps and len(templates) > 1:
        freq_features, window_timestamps = _build_frequency_timeseries(
            timestamps, template_ids, len(templates),
        )

        if freq_features is not None and len(freq_features) >= 10:
            # ---- 5. Anomaly detection on frequency patterns ----
            anom_labels, anom_scores = detect_anomalies_isolation_forest(
                freq_features, random_state=random_state
            )
            anomaly_mask = anom_labels == -1
            fp.n_anomalies = int(anomaly_mask.sum())
            fp.anomaly_pct = round(fp.n_anomalies / len(freq_features) * 100, 2)
            fp.anomaly_method = "isolation_forest_event_frequency"

            if fp.n_anomalies > 0:
                anom_indices = np.where(anomaly_mask)[0]
                top_anom = anom_indices[np.argsort(anom_scores[anom_indices])[-20:]]
                fp.anomalies = [
                    AnomalyInfo(
                        index=int(i),
                        score=round(float(anom_scores[i]), 4),
                        reason="anomalous_event_frequency_window",
                    )
                    for i in top_anom
                ]

            # Clustering on frequency windows
            cluster_labels, n_clusters, cluster_method = find_clusters(
                freq_features, random_state=random_state
            )
            fp.n_clusters = n_clusters
            fp.cluster_method = f"event_frequency_{cluster_method}"
            fp.clusters = [
                ClusterInfo(
                    cluster_id=cid,
                    size=int((cluster_labels == cid).sum()),
                    pct=round(float((cluster_labels == cid).sum()) / len(cluster_labels) * 100, 2),
                )
                for cid in range(n_clusters)
            ]

            # 2D embedding
            embedding, emb_method = compute_embedding_2d(
                freq_features, random_state=random_state
            )
            fp.embedding_2d = embedding.tolist()
            fp.embedding_method = f"log_frequency_{emb_method}"
            fp.embedding_labels = cluster_labels.tolist()

    # ---- 6. Burst detection ----
    if has_timestamps:
        bursts = _detect_bursts(timestamps)
        if bursts:
            fp.extras["bursts"] = bursts[:10]
            if len(bursts) > 3:
                quality_issues.append(QualityIssue(
                    severity="info",
                    category="drift",
                    message=f"{len(bursts)} log burst periods detected -- may indicate incident events or batch processes",
                ))

    # ---- 7. Quality checks ----
    # Multiline log entries (lines without timestamps that follow timestamped lines)
    if has_timestamps:
        no_ts_count = sum(1 for t in timestamps if t is None)
        if no_ts_count > len(lines) * 0.3:
            quality_issues.append(QualityIssue(
                severity="info",
                category="missing",
                message=f"{no_ts_count} lines ({no_ts_count/len(lines)*100:.1f}%) lack timestamps -- likely multiline entries or stack traces",
            ))

    # Template concentration
    if templates:
        top_template_pct = max(template_counts.values()) / len(lines) * 100
        if top_template_pct > 50:
            quality_issues.append(QualityIssue(
                severity="info",
                category="imbalance",
                message=f"Single template accounts for {top_template_pct:.1f}% of all lines -- log may be dominated by heartbeat/health-check messages",
            ))
            recommendations.append(Recommendation(
                priority=2,
                category="preprocessing",
                message="Consider filtering dominant heartbeat templates before anomaly analysis",
            ))

    # ---- 8. Recommendations ----
    if not has_timestamps:
        recommendations.append(Recommendation(
            priority=1,
            category="preprocessing",
            message="No timestamps detected -- temporal analysis unavailable. Add structured timestamps for richer investigation.",
        ))

    if len(templates) > 100:
        recommendations.append(Recommendation(
            priority=2,
            category="preprocessing",
            message=f"{len(templates)} unique templates -- consider tuning template extraction depth or grouping similar templates",
        ))

    if fp.n_anomalies > 0:
        recommendations.append(Recommendation(
            priority=1,
            category="preprocessing",
            message=f"{fp.n_anomalies} anomalous time windows detected -- investigate for incident correlation",
        ))

    fp.quality_issues = quality_issues
    fp.recommendations = sorted(recommendations, key=lambda r: r.priority)

    return fp


def _find_text_column(df: pd.DataFrame) -> Optional[str]:
    """Find the primary text column in a DataFrame."""
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(string_cols) == 0:
        return None
    if len(string_cols) == 1:
        return string_cols[0]
    # Pick the one with highest average string length
    avg_lengths = {col: df[col].astype(str).str.len().mean() for col in string_cols}
    return max(avg_lengths, key=avg_lengths.get)


def _extract_timestamps(lines: list) -> Tuple[Optional[list], Optional[str]]:
    """Extract timestamps from log lines using pattern matching."""
    for pattern_str, fmt in TIMESTAMP_PATTERNS:
        pattern = re.compile(pattern_str)
        matches = 0
        timestamps = []

        # Test on first 100 lines
        for line in lines[:100]:
            m = pattern.search(line)
            if m:
                matches += 1

        # If >30% match, use this pattern for all lines
        if matches > min(30, len(lines[:100]) * 0.3):
            for line in lines:
                m = pattern.search(line)
                if m:
                    ts_str = m.group(1)
                    try:
                        if fmt == "epoch_s":
                            ts = pd.Timestamp(int(ts_str), unit="s")
                        elif fmt == "epoch_ms":
                            ts = pd.Timestamp(int(ts_str), unit="ms")
                        else:
                            ts = pd.to_datetime(ts_str, format=fmt, errors="coerce")
                        timestamps.append(ts)
                    except (ValueError, OverflowError):
                        timestamps.append(None)
                else:
                    timestamps.append(None)
            return timestamps, pattern_str

    return None, None


def _extract_log_levels(lines: list) -> list:
    """Extract log level from each line."""
    levels = []
    for line in lines:
        m = LOG_LEVEL_PATTERN.search(line[:100])  # only check first 100 chars
        if m:
            levels.append(m.group(1).upper())
        else:
            levels.append(None)
    return levels


def _extract_templates(lines: list, max_templates: int = 200) -> Tuple[Dict[str, int], list]:
    """
    Simplified Drain-style log template extraction.

    Strategy: tokenize each line, replace variable tokens (numbers, IPs,
    paths, UUIDs) with wildcards, then group by the resulting template string.
    """
    # Patterns for variable tokens
    var_patterns = [
        (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "<IP>"),
        (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<UUID>"),
        (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
        (re.compile(r'(?<=[=:,\s])/[\w/._-]+'), "<PATH>"),
        (re.compile(r"\b\d+\.\d+\b"), "<NUM>"),
        (re.compile(r"\b\d+\b"), "<NUM>"),
    ]

    templates = Counter()
    template_ids = []
    template_map = {}  # template_str -> template_id
    next_id = 0

    for line in lines:
        # Strip timestamp prefix (first match of any timestamp pattern)
        stripped = line.strip()

        # Apply variable replacement
        for pat, repl in var_patterns:
            stripped = pat.sub(repl, stripped)

        # Collapse whitespace
        stripped = re.sub(r"\s+", " ", stripped).strip()

        # Truncate very long templates
        if len(stripped) > 200:
            stripped = stripped[:200]

        templates[stripped] += 1

        if stripped not in template_map:
            if len(template_map) < max_templates:
                template_map[stripped] = next_id
                next_id += 1
            else:
                # Overflow bucket
                template_map[stripped] = max_templates
        template_ids.append(template_map[stripped])

    return dict(templates), template_ids


def _build_frequency_timeseries(timestamps: list, template_ids: list,
                                  n_templates: int,
                                  n_windows: int = 100) -> Tuple[Optional[np.ndarray], Optional[list]]:
    """
    Build a time-series of event frequencies per template.

    Divides the time range into n_windows bins, counts template occurrences
    per bin. Result is (n_windows, n_template_features) matrix.
    """
    valid = [(ts, tid) for ts, tid in zip(timestamps, template_ids)
             if ts is not None and not (hasattr(ts, 'isnull') and ts.isnull()) and pd.notna(ts)]
    if len(valid) < 20:
        return None, None

    valid.sort(key=lambda x: x[0])
    ts_list = [v[0] for v in valid]
    tid_list = [v[1] for v in valid]

    t_start = ts_list[0]
    t_end = ts_list[-1]
    duration = (t_end - t_start).total_seconds()

    if duration <= 0:
        return None, None

    # Cap template features to avoid huge sparse matrices
    n_template_features = min(n_templates + 1, 50)

    window_duration = duration / n_windows
    features = np.zeros((n_windows, n_template_features + 1))  # +1 for total count
    window_timestamps = []

    for w in range(n_windows):
        w_start = t_start + pd.Timedelta(seconds=w * window_duration)
        w_end = t_start + pd.Timedelta(seconds=(w + 1) * window_duration)
        window_timestamps.append(w_start)

        for ts, tid in valid:
            if w_start <= ts < w_end:
                if tid < n_template_features:
                    features[w, tid] += 1
                features[w, -1] += 1  # total count

    return features, window_timestamps


def _detect_bursts(timestamps: list, threshold_factor: float = 5.0) -> list:
    """
    Detect burst periods (abnormally high log rate).

    A burst is a window where the event rate exceeds threshold_factor
    times the median rate.
    """
    valid_ts = sorted([t for t in timestamps if t is not None and pd.notna(t)])
    if len(valid_ts) < 20:
        return []

    # Compute inter-event times
    diffs = []
    for i in range(1, len(valid_ts)):
        d = (valid_ts[i] - valid_ts[i-1]).total_seconds()
        diffs.append(d)

    if not diffs:
        return []

    diffs = np.array(diffs)
    median_gap = np.median(diffs)

    if median_gap <= 0:
        return []

    # Find burst starts: where gap drops below median / threshold_factor
    burst_threshold = median_gap / threshold_factor
    bursts = []
    in_burst = False
    burst_start = None
    burst_count = 0

    for i, d in enumerate(diffs):
        if d < burst_threshold:
            if not in_burst:
                in_burst = True
                burst_start = i
                burst_count = 1
            else:
                burst_count += 1
        else:
            if in_burst and burst_count >= 5:
                bursts.append({
                    "start_index": burst_start,
                    "end_index": i,
                    "n_events": burst_count,
                    "start_time": str(valid_ts[burst_start]),
                    "end_time": str(valid_ts[i]),
                    "avg_rate_hz": round(burst_count / max(
                        (valid_ts[i] - valid_ts[burst_start]).total_seconds(), 0.001
                    ), 2),
                })
            in_burst = False
            burst_count = 0

    # Close trailing burst
    if in_burst and burst_count >= 5:
        bursts.append({
            "start_index": burst_start,
            "end_index": len(diffs),
            "n_events": burst_count,
            "start_time": str(valid_ts[burst_start]),
            "end_time": str(valid_ts[-1]),
            "avg_rate_hz": round(burst_count / max(
                (valid_ts[-1] - valid_ts[burst_start]).total_seconds(), 0.001
            ), 2),
        })

    return bursts
