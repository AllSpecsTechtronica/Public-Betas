"""
dataset_investigator - Universal dataset investigation engine for MLOps.

Public API:
    investigate(data, **kwargs) -> DatasetFingerprint

Accepts file paths, directories, DataFrames, or numpy arrays.
Auto-detects archetype (TABULAR, SEQUENCE, IMAGE, LOG) and runs
the appropriate analytical pipeline.

Output is a serializable DatasetFingerprint suitable for storage
in cvLayer's catalog and rendering in the Workbench UI.
"""

from dataset_investigator.fingerprint import DatasetFingerprint, Archetype
from dataset_investigator.detector import detect_archetype, load_data
from dataset_investigator.quality import compute_quality_score

__version__ = "0.1.0"


def investigate(data, target_column=None, time_column=None, archetype_hint=None,
                max_samples=50000, embedding_dim=2, random_state=42):
    """
    Universal dataset investigation entry point.

    Parameters
    ----------
    data : str, pathlib.Path, pd.DataFrame, or np.ndarray
        The dataset to investigate. Can be:
        - File path to CSV, Parquet, JSON, or directory of images
        - pandas DataFrame
        - numpy array (2D for tabular/sequence, 3D+ for images)
    target_column : str, optional
        Column name to treat as prediction target (tabular/log).
    time_column : str, optional
        Column name containing timestamps. Auto-detected if absent.
    archetype_hint : str or Archetype, optional
        Force archetype detection. One of 'tabular', 'sequence', 'image', 'log'.
    max_samples : int
        Maximum samples to analyze. Larger datasets are subsampled.
    embedding_dim : int
        Dimensionality for visualization embeddings (2 or 3).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    DatasetFingerprint
        Serializable investigation report.
    """
    # -- Step 1: Load raw data into a canonical form --
    canonical, metadata = load_data(data)

    # -- Step 2: Detect archetype --
    if archetype_hint is not None:
        if isinstance(archetype_hint, str):
            archetype = Archetype[archetype_hint.upper()]
        else:
            archetype = archetype_hint
    else:
        archetype = detect_archetype(canonical, metadata)

    # -- Step 3: Route to archetype-specific analyzer --
    analyzer_kwargs = dict(
        target_column=target_column,
        time_column=time_column,
        max_samples=max_samples,
        embedding_dim=embedding_dim,
        random_state=random_state,
    )

    if archetype == Archetype.TABULAR:
        from dataset_investigator.analyzers.tabular import analyze_tabular
        fingerprint = analyze_tabular(canonical, metadata, **analyzer_kwargs)

    elif archetype == Archetype.SEQUENCE:
        from dataset_investigator.analyzers.sequence import analyze_sequence
        fingerprint = analyze_sequence(canonical, metadata, **analyzer_kwargs)

    elif archetype == Archetype.IMAGE:
        from dataset_investigator.analyzers.image import analyze_image
        fingerprint = analyze_image(canonical, metadata, **analyzer_kwargs)

    elif archetype == Archetype.LOG:
        from dataset_investigator.analyzers.log import analyze_log
        fingerprint = analyze_log(canonical, metadata, **analyzer_kwargs)

    else:
        raise ValueError(f"Unknown archetype: {archetype}")

    # -- Step 4: Compute universal quality score --
    fingerprint = compute_quality_score(fingerprint)

    return fingerprint
