"""
Universal quality scoring.

Computes a composite 0-1 quality score from the fingerprint's quality issues.
The score is archetype-aware: different issue types carry different weights
depending on what matters for that data modality.

Scoring philosophy: start at 1.0, subtract penalties for each issue found.
This means a perfect dataset (no issues) scores 1.0, and a catastrophically
broken one approaches 0.0.
"""

from dataset_investigator.fingerprint import DatasetFingerprint, Archetype


# Penalty weights by (severity, category)
# Tuned to produce intuitive scores: a dataset with one critical issue
# should score ~0.7, one with multiple criticals should be below 0.5.
SEVERITY_WEIGHTS = {
    "critical": 0.15,
    "warning": 0.05,
    "info": 0.01,
}

# Bonus penalties for specific categories (on top of severity)
CATEGORY_PENALTIES = {
    "missing": 0.02,
    "duplicate": 0.02,
    "leakage": 0.05,
    "imbalance": 0.03,
    "outlier": 0.01,
    "drift": 0.02,
}

# Archetype-specific amplifiers: multiply penalty for categories
# that are especially dangerous for that archetype.
ARCHETYPE_AMPLIFIERS = {
    Archetype.TABULAR: {
        "leakage": 2.0,
        "duplicate": 1.5,
        "missing": 1.5,
    },
    Archetype.SEQUENCE: {
        "drift": 2.0,
        "missing": 2.0,  # gaps in time series are severe
        "outlier": 1.5,
    },
    Archetype.IMAGE: {
        "duplicate": 2.0,
        "imbalance": 2.0,
        "outlier": 1.5,
    },
    Archetype.LOG: {
        "missing": 1.0,  # missing timestamps are noted but less severe
        "drift": 1.5,
        "imbalance": 1.5,
    },
}


def compute_quality_score(fingerprint: DatasetFingerprint) -> DatasetFingerprint:
    """
    Compute composite quality score from accumulated quality issues.

    Modifies the fingerprint in-place and returns it.
    """
    score = 1.0
    archetype = fingerprint.archetype
    amplifiers = ARCHETYPE_AMPLIFIERS.get(archetype, {})

    for issue in fingerprint.quality_issues:
        # Base penalty from severity
        base = SEVERITY_WEIGHTS.get(issue.severity, 0.01)

        # Category bonus penalty
        cat_penalty = CATEGORY_PENALTIES.get(issue.category, 0.0)

        # Archetype amplifier
        amp = amplifiers.get(issue.category, 1.0)

        total_penalty = (base + cat_penalty) * amp
        score -= total_penalty

    # Additional structural penalties
    # Very small datasets
    if fingerprint.n_samples < 50:
        score -= 0.15
    elif fingerprint.n_samples < 200:
        score -= 0.05

    # Very high anomaly rate
    if fingerprint.anomaly_pct > 15:
        score -= 0.10
    elif fingerprint.anomaly_pct > 8:
        score -= 0.05

    # No features (degenerate dataset)
    if fingerprint.n_features == 0:
        score -= 0.20

    # Clamp to [0.0, 1.0]
    fingerprint.quality_score = round(max(0.0, min(1.0, score)), 3)

    return fingerprint
