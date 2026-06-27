"""sklearn_logreg_baseline.py — simple baseline cell for tabular classification.

Reads `dataset_csv` + `label_col` from backbone_config and trains a LogisticRegression
model. Saves a pickle artifact under `mlops/models/<scenario>/vN/model.pkl`.
"""

from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, list[str]]:
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
    enc = LabelEncoder()
    y = enc.fit_transform(df[label_col].astype(str).fillna("").to_numpy()).astype(np.int64, copy=False)
    return x, y, enc, list(feats.columns)


def run(ctx: BackboneContext, prev: list[Any]) -> dict[str, Any]:
    cfg = ctx.scenario_config
    bcfg = getattr(cfg, "backbone_config", {}) or {}

    dataset_csv = _resolve_path(str(bcfg.get("dataset_csv") or "").strip())
    label_col = str(bcfg.get("label_col") or "label").strip()
    feature_cols_raw = bcfg.get("feature_cols")
    feature_cols = [str(c) for c in feature_cols_raw] if isinstance(feature_cols_raw, list) else None
    seed = int(bcfg.get("seed") or 1337)
    _set_seed(seed)

    task = str(bcfg.get("task") or "").strip().lower()
    if task not in {"", "classification", "regression"}:
        task = ""

    val_split = float(bcfg.get("val_split") or 0.2)
    val_split = min(max(val_split, 0.05), 0.5)

    # Infer task from labels for user friendliness.
    if not task:
        try:
            label_series = pd.read_csv(dataset_csv, usecols=[label_col])[label_col]
            task = _infer_task(label_series)
        except Exception:
            task = "classification"
    if task == "regression":
        raise ValueError(
            "LogisticRegression baseline is classification-only. "
            "Set backbone_config.task='classification' or use a regression-capable algo "
            "(e.g. basic_cnn_for_editing.py, sklearn_knn_baseline.py, sklearn_random_forest_baseline.py)."
        )

    x, y, enc, used_cols = _load_xy(dataset_csv=dataset_csv, label_col=label_col, feature_cols=feature_cols)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=val_split,
        random_state=seed,
        stratify=_safe_stratify_targets(y),
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    max_iter = int(bcfg.get("max_iter") or 2000)
    c_value = float(bcfg.get("C") or 1.0)

    model = LogisticRegression(
        max_iter=max_iter,
        C=c_value,
        n_jobs=None,
        multi_class="auto",
        solver="lbfgs",
    )
    model.fit(x_train, y_train)
    val_acc = float((model.predict(x_val) == y_val).mean()) if y_val.size else 0.0

    models_root = (REPO_ROOT / "mlops" / "models" / str(cfg.name)).resolve()
    run_dir = _next_run_dir(models_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "model.pkl"
    with artifact.open("wb") as f:
        pickle.dump(
            {
                "model": model,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "label_classes": [str(c) for c in enc.classes_.tolist()],
                "feature_cols": used_cols,
                "label_col": label_col,
                "config": dict(bcfg),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    metrics = {
        "val_acc": val_acc,
        "num_classes": int(len(enc.classes_)),
        "num_features": int(x.shape[1]),
        "rows": int(x.shape[0]),
    }
    (run_dir / "metrics.json").write_text(json.dumps({"metrics": metrics}, indent=2), encoding="utf-8")

    rel_artifact = str(artifact.relative_to(REPO_ROOT)).replace("\\", "/")
    summary = f"trained LogisticRegression → {run_dir.name} (val_acc={val_acc:.4f})"
    print(summary)
    return {
        "output": summary,
        "data": {
            "model_version": run_dir.name,
            "weights_path": rel_artifact,
            "signal": {"flag": False, "summary": summary, "metrics": metrics},
        },
    }
