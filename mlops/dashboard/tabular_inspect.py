from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TabularSummary:
    rows: int
    cols: int
    dtypes: dict[str, str]
    missing_by_col: dict[str, int]
    nunique_by_col: dict[str, int]
    duplicate_rows: int


@dataclass(frozen=True)
class TabularQualityChecks:
    constant_cols: list[str]
    high_missing_cols: list[str]
    high_cardinality_cols: list[str]


def load_csv(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def summarize_frame(df: pd.DataFrame) -> TabularSummary:
    dtypes = {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
    missing = {str(k): int(v) for k, v in df.isna().sum().to_dict().items()}
    nunique = {str(k): int(v) for k, v in df.nunique(dropna=True).to_dict().items()}
    try:
        duplicate_rows = int(df.duplicated().sum())
    except Exception:
        duplicate_rows = 0
    return TabularSummary(
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        dtypes=dtypes,
        missing_by_col=missing,
        nunique_by_col=nunique,
        duplicate_rows=duplicate_rows,
    )


def quality_checks(
    df: pd.DataFrame,
    *,
    missing_frac_threshold: float = 0.5,
    cardinality_frac_threshold: float = 0.2,
    min_rows_for_cardinality: int = 200,
) -> TabularQualityChecks:
    rows = int(df.shape[0])
    constant_cols: list[str] = []
    high_missing_cols: list[str] = []
    high_cardinality_cols: list[str] = []

    if rows <= 0:
        return TabularQualityChecks([], [], [])

    nunique = df.nunique(dropna=True)
    missing = df.isna().mean(numeric_only=False)

    for col in df.columns:
        try:
            if int(nunique[col]) <= 1:
                constant_cols.append(str(col))
        except Exception:
            pass
        try:
            if float(missing[col]) >= float(missing_frac_threshold):
                high_missing_cols.append(str(col))
        except Exception:
            pass

    if rows >= int(min_rows_for_cardinality):
        for col in df.columns:
            try:
                frac = float(nunique[col]) / float(rows)
            except Exception:
                continue
            # High-cardinality is most interesting for non-numeric columns.
            try:
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
            except Exception:
                is_numeric = False
            if (not is_numeric) and frac >= float(cardinality_frac_threshold):
                high_cardinality_cols.append(str(col))

    constant_cols.sort(key=lambda s: s.lower())
    high_missing_cols.sort(key=lambda s: s.lower())
    high_cardinality_cols.sort(key=lambda s: s.lower())
    return TabularQualityChecks(
        constant_cols=constant_cols,
        high_missing_cols=high_missing_cols,
        high_cardinality_cols=high_cardinality_cols,
    )
