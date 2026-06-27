"""sklearn_knn_baseline.py — KNN baseline for tabular "relationship" modeling.

K-Nearest Neighbors is often a good quick check for whether local neighborhood
structure (relationships between samples) is predictive.

Config (backbone_config):
  - dataset_csv: str (required)
  - label_col: str (default: "label")
  - feature_cols: list[str] (optional)
  - val_split: float (default: 0.2)
  - seed: int (default: 1337)
  - n_neighbors: int (default: 15)
  - weights: "uniform" | "distance" (default: "distance")
  - metric: str (default: "minkowski")  # "euclidean", "manhattan", ...

Artifacts:
  - mlops/models/<scenario>/vN/model.pkl
  - mlops/models/<scenario>/vN/metrics.json
"""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from mlops.pipeline.backbone import BackboneContext
from mlops.pipeline.registry import REPO_ROOT
from mlops.algos.schema_prompt import raise_label_col_missing


def _resolve_path(path_value: str) -> Path:
    p = Path(str(path_value or "").strip())
    if not p:
        raise ValueError("dataset_csv is required")
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _next_run_dir(models_root: Path) -> Path:
    runs = [p for p in models_root.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
    if not runs:
        return models_root / "v1"
    latest = max(int(p.name[1:]) for p in runs)
    return models_root / f"v{latest + 1}"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _safe_stratify_targets(y: np.ndarray) -> Optional[np.ndarray]:
    """Return y for stratify if valid; otherwise None (requires >=2 samples per class)."""
    try:
        y_int = y.astype(np.int64, copy=False)
    except Exception:
        return None
    if y_int.size < 4:
        return None
    try:
        counts = np.bincount(y_int)
    except Exception:
        return None
    nonzero = counts[counts > 0]
    if nonzero.size < 2:
        return None
    if int(nonzero.min()) < 2:
        return None
    return y_int


def _infer_task(label_series: pd.Series) -> str:
    numeric = pd.to_numeric(label_series, errors="coerce")
    numeric_ratio = float(numeric.notna().mean()) if len(label_series) else 0.0
    nunique = int(label_series.nunique(dropna=True)) if len(label_series) else 0
    unique_ratio = float(nunique / max(1, len(label_series)))
    if numeric_ratio >= 0.95 and (nunique >= 20 or unique_ratio >= 0.10):
        return "regression"
    return "classification"


def _load_xy(
    *,
    dataset_csv: Path,
    label_col: str,
    feature_cols: Optional[list[str]],
) -> tuple[np.ndarray, pd.Series, list[str]]:
    df = pd.read_csv(dataset_csv)
    if label_col not in df.columns:
        raise_label_col_missing(
            dataset_csv=dataset_csv,
            attempted_label_col=label_col,
            columns=[str(c) for c in df.columns],
        )
    if feature_cols:
        feats = df[feature_cols]
    else:
        feats = df.drop(columns=[label_col])
    feats = feats.select_dtypes(include=[np.number]).copy()
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = feats.to_numpy(dtype=np.float32, copy=True)
    return x, df[label_col], list(feats.columns)


def run(ctx: BackboneContext, prev: list[Any]) -> dict[str, Any]:
    cfg = ctx.scenario_config
    bcfg = getattr(cfg, "backbone_config", {}) or {}

    dataset_csv = _resolve_path(str(bcfg.get("dataset_csv") or "").strip())
    label_col = str(bcfg.get("label_col") or "label").strip()
    feature_cols_raw = bcfg.get("feature_cols")
    feature_cols = [str(c) for c in feature_cols_raw] if isinstance(feature_cols_raw, list) else None
    task = str(bcfg.get("task") or "").strip().lower()

    seed = int(bcfg.get("seed") or 1337)
    _set_seed(seed)

    val_split = float(bcfg.get("val_split") or 0.2)
    val_split = min(max(val_split, 0.05), 0.5)

    x, y_series, used_cols = _load_xy(dataset_csv=dataset_csv, label_col=label_col, feature_cols=feature_cols)
    if task not in {"classification", "regression"}:
        task = _infer_task(y_series)
    label_classes: list[str] = []
    if task == "regression":
        y_num = pd.to_numeric(y_series, errors="coerce").astype(np.float32)
        if int(y_num.isna().sum()) > 0:
            y_num = y_num.fillna(float(y_num.median() if y_num.notna().any() else 0.0))
        y = y_num.to_numpy(dtype=np.float32, copy=False)
    else:
        enc = LabelEncoder()
        y = enc.fit_transform(y_series.astype(str).fillna("").to_numpy()).astype(np.int64, copy=False)
        label_classes = [str(c) for c in enc.classes_.tolist()]
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=val_split,
        random_state=seed,
        stratify=_safe_stratify_targets(y) if task == "classification" else None,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    n_neighbors = int(bcfg.get("n_neighbors") or 15)
    weights = str(bcfg.get("weights") or "distance").strip().lower() or "distance"
    metric = str(bcfg.get("metric") or "minkowski").strip() or "minkowski"

    if task == "regression":
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric)
        knn.fit(x_train, y_train)
        pred = knn.predict(x_val) if y_val.size else np.array([], dtype=np.float32)
        val_metric = float(np.mean(np.abs(pred - y_val))) if y_val.size else 0.0  # MAE
    else:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
        knn.fit(x_train, y_train)
        val_metric = float((knn.predict(x_val) == y_val).mean()) if y_val.size else 0.0  # ACC

    models_root = (REPO_ROOT / "mlops" / "models" / str(cfg.name)).resolve()
    run_dir = _next_run_dir(models_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "model.pkl"
    with artifact.open("wb") as f:
        pickle.dump(
            {
                "model": knn,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "task": task,
                "label_classes": label_classes,
                "feature_cols": used_cols,
                "label_col": label_col,
                "config": dict(bcfg),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    metrics = {
        "task": task,
        "val_metric": float(val_metric),
        "num_classes": int(len(label_classes)),
        "num_features": int(x.shape[1]),
        "rows": int(x.shape[0]),
        "n_neighbors": int(n_neighbors),
        "weights": weights,
        "metric": metric,
    }
    (run_dir / "metrics.json").write_text(json.dumps({"metrics": metrics}, indent=2), encoding="utf-8")

    rel_artifact = str(artifact.relative_to(REPO_ROOT)).replace("\\", "/")
    if task == "regression":
        summary = f"trained KNN(regression) → {run_dir.name} (val_mae={metrics['val_metric']:.4f}, k={n_neighbors})"
    else:
        summary = f"trained KNN(classification) → {run_dir.name} (val_acc={metrics['val_metric']:.4f}, k={n_neighbors})"
    print(summary)
    return {
        "output": summary,
        "data": {
            "model_version": run_dir.name,
            "weights_path": rel_artifact,
            "signal": {"flag": False, "summary": summary, "metrics": metrics},
        },
    }
