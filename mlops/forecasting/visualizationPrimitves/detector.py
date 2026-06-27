"""
Archetype detection and canonical data loading.

Detection is structural, not heuristic. We test physical properties
of the data rather than guessing from file extensions.
"""

import os
import pathlib
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from dataset_investigator.fingerprint import Archetype

# Image extensions we recognize
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}


def load_data(data) -> Tuple[Any, Dict[str, Any]]:
    """
    Load data into a canonical form.

    Returns
    -------
    canonical : pd.DataFrame, np.ndarray, or dict
        The loaded data in its natural representation.
    metadata : dict
        Loading metadata: source path, original format, etc.
    """
    metadata = {"source": str(data), "original_type": type(data).__name__}

    # -- Already a DataFrame --
    if isinstance(data, pd.DataFrame):
        metadata["format"] = "dataframe"
        return data, metadata

    # -- Numpy array --
    if isinstance(data, np.ndarray):
        metadata["format"] = "ndarray"
        metadata["shape"] = list(data.shape)
        metadata["dtype"] = str(data.dtype)
        return data, metadata

    # -- String or Path -> file system --
    path = pathlib.Path(data)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Directory -> check for images
    if path.is_dir():
        image_files = _find_image_files(path)
        if image_files:
            metadata["format"] = "image_directory"
            metadata["n_files"] = len(image_files)
            metadata["image_paths"] = image_files
            return {"image_paths": image_files, "root": str(path)}, metadata
        else:
            # Directory of CSVs or other files
            data_files = list(path.glob("*.csv")) + list(path.glob("*.parquet"))
            if data_files:
                # Load first file as representative
                return _load_file(data_files[0], metadata)
            raise ValueError(f"Directory contains no recognizable data files: {path}")

    # Single file
    return _load_file(path, metadata)


def _load_file(path: pathlib.Path, metadata: dict) -> Tuple[Any, Dict]:
    """Load a single file based on extension."""
    ext = path.suffix.lower()
    metadata["format"] = ext.lstrip(".")

    if ext == ".csv":
        df = pd.read_csv(path, low_memory=False)
        metadata["memory_bytes"] = df.memory_usage(deep=True).sum()
        return df, metadata

    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t", low_memory=False)
        metadata["memory_bytes"] = df.memory_usage(deep=True).sum()
        return df, metadata

    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        metadata["memory_bytes"] = df.memory_usage(deep=True).sum()
        return df, metadata

    elif ext == ".json":
        df = pd.read_json(path)
        metadata["memory_bytes"] = df.memory_usage(deep=True).sum()
        return df, metadata

    elif ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        metadata["shape"] = list(arr.shape)
        metadata["dtype"] = str(arr.dtype)
        return arr, metadata

    elif ext == ".npz":
        npz = np.load(path, allow_pickle=False)
        # Return the first array if single, else dict
        keys = list(npz.keys())
        if len(keys) == 1:
            arr = npz[keys[0]]
            metadata["shape"] = list(arr.shape)
            metadata["dtype"] = str(arr.dtype)
            return arr, metadata
        else:
            metadata["array_keys"] = keys
            return {k: npz[k] for k in keys}, metadata

    elif ext in IMAGE_EXTENSIONS:
        # Single image file -- wrap in list for consistency
        metadata["format"] = "image_directory"
        metadata["n_files"] = 1
        metadata["image_paths"] = [str(path)]
        return {"image_paths": [str(path)], "root": str(path.parent)}, metadata

    elif ext in (".log", ".txt"):
        # Read as text lines -> DataFrame with single column
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
        df = pd.DataFrame({"line": [l.rstrip("\n") for l in lines]})
        metadata["n_lines"] = len(lines)
        return df, metadata

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _find_image_files(directory: pathlib.Path, max_depth=3) -> list:
    """Recursively find image files up to max_depth."""
    image_files = []
    for root, dirs, files in os.walk(directory):
        depth = len(pathlib.Path(root).relative_to(directory).parts)
        if depth >= max_depth:
            dirs.clear()
            continue
        for f in files:
            if pathlib.Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                image_files.append(os.path.join(root, f))
    return sorted(image_files)


def detect_archetype(canonical, metadata: dict) -> Archetype:
    """
    Determine dataset archetype from structural properties.

    Decision tree (physics, not heuristics):
    1. If data contains image paths -> IMAGE
    2. If numpy array with ndim >= 3 -> IMAGE
    3. If numpy array with ndim == 2, narrow width -> SEQUENCE
    4. If DataFrame:
       a. Has dominant string column with repetitive templates -> LOG
       b. Has monotonic/datetime index or specified time col, mostly numeric -> SEQUENCE
       c. Otherwise -> TABULAR
    """
    # -- Image directory --
    if isinstance(canonical, dict) and "image_paths" in canonical:
        return Archetype.IMAGE

    # -- Numpy arrays --
    if isinstance(canonical, np.ndarray):
        if canonical.ndim >= 3:
            return Archetype.IMAGE
        if canonical.ndim == 2 and canonical.shape[1] < 256:
            # Multichannel signal: rows are time steps, columns are channels
            # 256 is a generous upper bound (EEG systems max ~256 channels)
            return Archetype.SEQUENCE
        return Archetype.TABULAR

    # -- DataFrame analysis --
    if isinstance(canonical, pd.DataFrame):
        df = canonical

        # Test for LOG archetype:
        # A log dataset has at least one string column where many rows
        # share similar structural templates (repetitive patterns).
        if _is_log_data(df):
            return Archetype.LOG

        # Test for SEQUENCE archetype:
        # Predominantly numeric with a time-like index or column.
        if _is_sequence_data(df):
            return Archetype.SEQUENCE

        return Archetype.TABULAR

    return Archetype.TABULAR


def _is_log_data(df: pd.DataFrame) -> bool:
    """
    Detect log-structured data.

    Heuristic: if a string column exists where the ratio of unique values
    to total rows is low AND the strings contain common structural tokens
    (timestamps, log levels, IPs, paths), it's log data.
    """
    string_cols = df.select_dtypes(include=["object", "string"]).columns

    if len(string_cols) == 0:
        return False

    # Check if there's a single dominant text column (like raw log lines)
    if len(df.columns) <= 3 and len(string_cols) >= 1:
        for col in string_cols:
            sample = df[col].dropna().head(100)
            if len(sample) < 10:
                continue
            # Log lines typically contain timestamps, levels, or structured prefixes
            log_indicators = 0
            for line in sample:
                line_str = str(line)
                # Check for common log patterns
                has_timestamp = any(c in line_str for c in [":", "T", "Z"]) and any(c.isdigit() for c in line_str[:20])
                has_level = any(lvl in line_str.upper() for lvl in ["ERROR", "WARN", "INFO", "DEBUG", "FATAL", "TRACE"])
                has_brackets = "[" in line_str and "]" in line_str
                if has_timestamp or has_level or has_brackets:
                    log_indicators += 1
            if log_indicators / len(sample) > 0.5:
                return True

    return False


def _is_sequence_data(df: pd.DataFrame) -> bool:
    """
    Detect time-series / sequence data.

    Conditions (must satisfy at least one time indicator + mostly numeric):
    - Has a datetime column or datetime index
    - Has a monotonically increasing integer index with mostly numeric columns
    - Column names suggest channels (ch1, ch2, ... or sensor_0, sensor_1, ...)
    """
    numeric_ratio = len(df.select_dtypes(include=[np.number]).columns) / max(len(df.columns), 1)

    if numeric_ratio < 0.6:
        return False

    # Check for datetime index
    if isinstance(df.index, pd.DatetimeIndex):
        return True

    # Check for datetime columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return True
        # Try parsing string columns as dates (sample only)
        if df[col].dtype == object:
            sample = df[col].dropna().head(20)
            try:
                pd.to_datetime(sample)
                return True
            except (ValueError, TypeError):
                pass

    # Check for monotonic numeric index with high numeric ratio
    if df.index.is_monotonic_increasing and numeric_ratio > 0.8 and len(df) > 100:
        # Could be a signal with sample-index as rows
        return True

    # Check channel-like column names
    col_str = " ".join(str(c).lower() for c in df.columns)
    channel_indicators = ["ch", "channel", "sensor", "electrode", "eeg", "emg", "ecg", "acc", "gyro"]
    if any(ind in col_str for ind in channel_indicators):
        return True

    return False
