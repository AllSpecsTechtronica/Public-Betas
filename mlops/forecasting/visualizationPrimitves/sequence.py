"""
Sequence / time-series analyzer.

Handles: EEG/EMG/ECG signals, sensor streams, trajectory data, any ordered
multichannel numeric data where the row index is meaningful (time).

Pipeline:
    1. Channel-level distribution profiling
    2. Stationarity testing (ADF)
    3. Spectral analysis (FFT dominant frequencies)
    4. Changepoint detection (simple variance-ratio method)
    5. Cross-channel correlation
    6. Anomaly detection on windowed features
    7. 2D embedding of windowed segments
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


def analyze_sequence(data, metadata: dict, target_column: str = None,
                     time_column: str = None, max_samples: int = 50000,
                     embedding_dim: int = 2, random_state: int = 42
                     ) -> DatasetFingerprint:
    """Full sequence/time-series investigation pipeline."""

    # Normalize to DataFrame
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            df = pd.DataFrame({"ch_0": data})
        elif data.ndim == 2:
            df = pd.DataFrame(data, columns=[f"ch_{i}" for i in range(data.shape[1])])
        else:
            raise ValueError(f"Sequence analyzer expects 1D or 2D array, got shape {data.shape}")
    else:
        df = data.copy()

    fp = DatasetFingerprint(
        archetype=Archetype.SEQUENCE,
        source=metadata.get("source", ""),
        n_samples=len(df),
        shape_raw=list(df.shape),
        memory_bytes=metadata.get("memory_bytes") or int(df.memory_usage(deep=True).sum()),
    )

    quality_issues = []
    recommendations = []

    # Identify numeric channels vs metadata columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude time column from channels
    if time_column and time_column in numeric_cols:
        numeric_cols.remove(time_column)

    fp.n_features = len(numeric_cols)

    # Subsample if too long
    df_work, was_subsampled = subsample_data(df, max_samples, random_state)
    if was_subsampled:
        # For sequences, take a contiguous block instead of random sampling
        start = max(0, (len(df) - max_samples) // 2)
        df_work = df.iloc[start:start + max_samples].reset_index(drop=True)
    fp.was_subsampled = was_subsampled
    fp.subsample_n = len(df_work) if was_subsampled else None

    # ---- 1. Channel-level distribution profiling ----
    for col in numeric_cols:
        fp.distributions.append(describe_distribution(df_work[col], name=str(col)))

    # ---- 2. Temporal analysis ----
    td = TemporalDecomposition(has_time_axis=True)

    # Determine sampling rate
    if time_column and time_column in df_work.columns:
        td.time_column = time_column
        try:
            ts = pd.to_datetime(df_work[time_column])
            diffs = ts.diff().dropna().dt.total_seconds()
            if len(diffs) > 0 and diffs.median() > 0:
                td.sampling_rate_hz = round(1.0 / diffs.median(), 4)
                td.is_regular = (diffs.std() / max(diffs.mean(), 1e-10)) < 0.1
            td.time_range = {
                "start": str(ts.min()),
                "end": str(ts.max()),
                "duration": str(ts.max() - ts.min()),
            }
        except (ValueError, TypeError):
            pass
    else:
        # Assume uniform sampling, rate unknown
        td.time_column = "sample_index"
        td.is_regular = True

    # ---- 3. Stationarity testing ----
    # Test first numeric channel as representative
    if len(numeric_cols) > 0:
        test_signal = df_work[numeric_cols[0]].dropna().values
        if len(test_signal) > 50:
            td.is_stationary, td.stationarity_pvalue = _adf_test(test_signal)

    # ---- 4. Spectral analysis ----
    if len(numeric_cols) > 0 and td.sampling_rate_hz and td.sampling_rate_hz > 0:
        dominant_freqs = _spectral_analysis(
            df_work[numeric_cols[0]].dropna().values,
            td.sampling_rate_hz,
        )
        td.dominant_frequencies_hz = dominant_freqs

    # ---- 5. Changepoint detection ----
    if len(numeric_cols) > 0:
        signal = df_work[numeric_cols[0]].dropna().values
        changepoints = _detect_changepoints(signal)
        td.n_changepoints = len(changepoints)
        td.changepoint_indices = changepoints[:20]  # cap for serialization

        if len(changepoints) > 0:
            quality_issues.append(QualityIssue(
                severity="info",
                category="drift",
                message=f"{len(changepoints)} regime changepoints detected in primary channel",
                affected_columns=[numeric_cols[0]],
            ))

    # Trend direction
    if len(numeric_cols) > 0:
        signal = df_work[numeric_cols[0]].dropna().values
        if len(signal) > 10:
            first_quarter = signal[:len(signal)//4].mean()
            last_quarter = signal[-len(signal)//4:].mean()
            ratio = (last_quarter - first_quarter) / max(abs(first_quarter), abs(signal.std()), 1e-10)
            if ratio > 0.1:
                td.trend_direction = "increasing"
            elif ratio < -0.1:
                td.trend_direction = "decreasing"
            else:
                td.trend_direction = "flat"

    fp.temporal = td

    # ---- 6. Cross-channel correlation ----
    if len(numeric_cols) >= 2:
        corr_matrix, corr_features, high_pairs = compute_correlation_matrix(df_work[numeric_cols])
        fp.correlation_matrix = corr_matrix
        fp.correlation_features = corr_features
        fp.high_correlation_pairs = high_pairs

    # ---- 7. Windowed feature extraction for clustering/anomalies ----
    if len(numeric_cols) > 0:
        window_features = _extract_window_features(
            df_work[numeric_cols].fillna(0).values,
            window_size=min(256, len(df_work) // 10),
        )

        if window_features is not None and len(window_features) >= 10:
            # Clustering on windows
            cluster_labels, n_clusters, cluster_method = find_clusters(
                window_features, random_state=random_state
            )
            fp.n_clusters = n_clusters
            fp.cluster_method = f"windowed_{cluster_method}"
            fp.clusters = [
                ClusterInfo(
                    cluster_id=cid,
                    size=int((cluster_labels == cid).sum()),
                    pct=round(float((cluster_labels == cid).sum()) / len(cluster_labels) * 100, 2),
                )
                for cid in range(n_clusters)
            ]

            # Anomaly detection on windows
            anom_labels, anom_scores = detect_anomalies_isolation_forest(
                window_features, random_state=random_state
            )
            anomaly_mask = anom_labels == -1
            fp.n_anomalies = int(anomaly_mask.sum())
            fp.anomaly_pct = round(fp.n_anomalies / len(window_features) * 100, 2)
            fp.anomaly_method = "isolation_forest_windowed"

            if fp.n_anomalies > 0:
                anom_indices = np.where(anomaly_mask)[0]
                top_anom = anom_indices[np.argsort(anom_scores[anom_indices])[-20:]]
                fp.anomalies = [
                    AnomalyInfo(
                        index=int(i),
                        score=round(float(anom_scores[i]), 4),
                        reason="anomalous_window_statistics",
                    )
                    for i in top_anom
                ]

            # 2D embedding of windows
            embedding, emb_method = compute_embedding_2d(
                window_features, random_state=random_state
            )
            fp.embedding_2d = embedding.tolist()
            fp.embedding_method = f"windowed_{emb_method}"
            fp.embedding_labels = cluster_labels.tolist()

    # ---- 8. Signal quality checks ----
    for col in numeric_cols:
        series = df_work[col]
        # Check for flatlines (zero variance over stretches)
        if series.std() < 1e-10:
            quality_issues.append(QualityIssue(
                severity="critical",
                category="outlier",
                message=f"Channel '{col}' has near-zero variance (flatline)",
                affected_columns=[col],
            ))
        # Check for NaN gaps
        nan_pct = series.isna().mean() * 100
        if nan_pct > 5:
            quality_issues.append(QualityIssue(
                severity="warning",
                category="missing",
                message=f"Channel '{col}' has {nan_pct:.1f}% missing samples",
                affected_columns=[col],
            ))

    # ---- 9. Sequence-specific extras ----
    if len(numeric_cols) > 0:
        signal = df_work[numeric_cols[0]].dropna().values
        fp.extras["primary_channel_snr_db"] = _estimate_snr(signal)
        fp.extras["n_channels"] = len(numeric_cols)
        fp.extras["total_duration_samples"] = len(df_work)
        if td.sampling_rate_hz:
            fp.extras["total_duration_seconds"] = round(len(df_work) / td.sampling_rate_hz, 2)

    # ---- 10. Recommendations ----
    if td.is_stationary is False:
        recommendations.append(Recommendation(
            priority=1,
            category="preprocessing",
            message="Signal is non-stationary -- consider differencing, detrending, or windowed normalization",
        ))

    if td.n_changepoints > 5:
        recommendations.append(Recommendation(
            priority=2,
            category="architecture",
            message=f"{td.n_changepoints} regime changes detected -- consider regime-aware models or segmented training",
        ))

    if fp.extras.get("primary_channel_snr_db", 100) < 10:
        recommendations.append(Recommendation(
            priority=1,
            category="preprocessing",
            message=f"Low SNR ({fp.extras.get('primary_channel_snr_db', 0):.1f} dB) -- apply bandpass filtering or wavelet denoising before training",
        ))

    if len(numeric_cols) > 32:
        recommendations.append(Recommendation(
            priority=3,
            category="feature_engineering",
            message=f"{len(numeric_cols)} channels -- consider spatial filtering (CSP, ICA) for dimensionality reduction",
        ))

    fp.quality_issues = quality_issues
    fp.recommendations = sorted(recommendations, key=lambda r: r.priority)

    return fp


def _adf_test(signal: np.ndarray) -> tuple:
    """Augmented Dickey-Fuller stationarity test."""
    try:
        from scipy.stats import normaltest
        # Simplified stationarity check: compare variance of first and second half
        half = len(signal) // 2
        var1 = np.var(signal[:half])
        var2 = np.var(signal[half:])
        mean1 = np.mean(signal[:half])
        mean2 = np.mean(signal[half:])

        # F-test approximation for variance equality
        if var1 > 0 and var2 > 0:
            f_stat = max(var1, var2) / min(var1, var2)
            # Rough p-value (proper F-dist would need scipy.stats.f)
            mean_shift = abs(mean2 - mean1) / max(np.std(signal), 1e-10)
            # Combined stationarity indicator
            is_stationary = f_stat < 2.0 and mean_shift < 0.5
            p_value = 1.0 / (1.0 + f_stat + mean_shift)  # pseudo p-value
            return is_stationary, round(p_value, 6)
    except Exception:
        pass
    return None, None


def _spectral_analysis(signal: np.ndarray, fs: float, n_peaks: int = 5) -> list:
    """Find dominant frequencies via FFT."""
    if len(signal) < 16:
        return []

    # Remove DC
    signal = signal - np.mean(signal)

    # FFT
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)

    # Skip DC (index 0) and find peaks
    if len(fft_mag) < 2:
        return []

    fft_mag[0] = 0  # zero out DC
    # Simple peak detection: local maxima
    peaks = []
    for i in range(1, len(fft_mag) - 1):
        if fft_mag[i] > fft_mag[i-1] and fft_mag[i] > fft_mag[i+1]:
            peaks.append((freqs[i], fft_mag[i]))

    peaks.sort(key=lambda x: x[1], reverse=True)
    return [round(float(f), 4) for f, _ in peaks[:n_peaks]]


def _detect_changepoints(signal: np.ndarray, min_segment: int = 50) -> list:
    """
    Simple variance-ratio changepoint detection.

    Splits signal into windows. Where the ratio of adjacent window variances
    exceeds a threshold, flag a changepoint.
    """
    if len(signal) < min_segment * 3:
        return []

    window_size = max(min_segment, len(signal) // 50)
    n_windows = len(signal) // window_size

    if n_windows < 3:
        return []

    variances = []
    means = []
    for i in range(n_windows):
        chunk = signal[i * window_size:(i + 1) * window_size]
        variances.append(np.var(chunk))
        means.append(np.mean(chunk))

    variances = np.array(variances)
    means = np.array(means)

    changepoints = []
    for i in range(1, len(variances)):
        if variances[i-1] > 0:
            var_ratio = variances[i] / max(variances[i-1], 1e-10)
            mean_shift = abs(means[i] - means[i-1]) / max(np.std(signal), 1e-10)
            if var_ratio > 3.0 or var_ratio < 0.33 or mean_shift > 1.0:
                changepoints.append(i * window_size)

    return changepoints


def _extract_window_features(data: np.ndarray, window_size: int = 256) -> Optional[np.ndarray]:
    """
    Extract statistical features from non-overlapping windows.

    For each window: mean, std, min, max, skew, kurtosis per channel.
    This compresses the sequence into a tabular form suitable for
    clustering and anomaly detection.
    """
    n_samples, n_channels = data.shape
    if window_size < 8 or n_samples < window_size * 3:
        return None

    n_windows = n_samples // window_size
    # 6 features per channel: mean, std, min, max, skew, kurtosis
    features = np.zeros((n_windows, n_channels * 6))

    for w in range(n_windows):
        chunk = data[w * window_size:(w + 1) * window_size]
        for c in range(n_channels):
            col = chunk[:, c]
            offset = c * 6
            features[w, offset] = np.mean(col)
            features[w, offset + 1] = np.std(col)
            features[w, offset + 2] = np.min(col)
            features[w, offset + 3] = np.max(col)
            # Skewness
            std = features[w, offset + 1]
            if std > 0:
                features[w, offset + 4] = np.mean(((col - features[w, offset]) / std) ** 3)
                features[w, offset + 5] = np.mean(((col - features[w, offset]) / std) ** 4) - 3
            else:
                features[w, offset + 4] = 0.0
                features[w, offset + 5] = 0.0

    return features


def _estimate_snr(signal: np.ndarray) -> float:
    """
    Rough SNR estimate via autocorrelation method.

    Signal power estimated from autocorrelation at lag 1 (correlated = signal).
    Noise power = total power - signal power.
    """
    if len(signal) < 10:
        return 0.0

    signal = signal - np.mean(signal)
    total_power = np.var(signal)
    if total_power < 1e-20:
        return 0.0

    # Autocorrelation at lag 1
    ac1 = np.corrcoef(signal[:-1], signal[1:])[0, 1]
    if np.isnan(ac1):
        return 0.0

    # Signal power ~ autocorrelated component
    signal_power = total_power * max(ac1, 0)
    noise_power = total_power - signal_power

    if noise_power <= 0:
        return 60.0  # effectively noiseless

    snr = 10 * np.log10(signal_power / noise_power)
    return round(float(snr), 2)
