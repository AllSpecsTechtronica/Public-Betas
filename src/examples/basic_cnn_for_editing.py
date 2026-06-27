"""basic_cnn_for_editing.py — default editable algo cell for tabular datasets.

This file is intentionally dataset-agnostic and safe to import (no top-level
training side effects). It is meant to be edited in-place and used as a
torch_tabular execution cell.

Scenario YAML example:

  backbone_type: torch_tabular
  backbone_config:
    dataset_csv: database/my_tabular.csv
    label_col: label
    cells:
      - path: basic_cnn_for_editing.py

Expected CSV shape:
  - One row per example
  - One label column (classification)
  - Remaining columns are numeric features (or specify feature_cols)
"""

from __future__ import annotations
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_stratify_targets(y: np.ndarray) -> Optional[np.ndarray]:
    """Return y for stratify if valid; otherwise None.

    Stratified splitting requires at least 2 samples per class.
    """
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
    raw = label_series
    # If mostly numeric and high-cardinality, treat as regression.
    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_ratio = float(numeric.notna().mean()) if len(raw) else 0.0
    nunique = int(raw.nunique(dropna=True)) if len(raw) else 0
    unique_ratio = float(nunique / max(1, len(raw)))
    if numeric_ratio >= 0.95 and (nunique >= 20 or unique_ratio >= 0.10):
        return "regression"
    return "classification"


class _GpuCycleCounter:
    """Approximate GPU cycle counter using CUDA event timing.

    We record elapsed GPU time via CUDA events, then convert to an estimated cycle
    count using the device's reported core clock rate (kHz). This is not a
    hardware performance counter (CUPTI), but it's lightweight and works in
    typical PyTorch environments.
    """

    def __init__(self, device: torch.device) -> None:
        self._enabled = bool(device.type == "cuda" and torch.cuda.is_available())
        self._start: Optional[torch.cuda.Event] = None
        self._end: Optional[torch.cuda.Event] = None
        self._elapsed_ms: Optional[float] = None
        self._device_index: Optional[int] = None
        self._device_name: str = ""
        self._clock_khz: Optional[int] = None

        if self._enabled:
            try:
                self._device_index = int(torch.cuda.current_device())
                props = torch.cuda.get_device_properties(self._device_index)
                self._device_name = str(getattr(props, "name", "") or "")
                clock = getattr(props, "clock_rate", None)
                self._clock_khz = int(clock) if clock is not None else None
            except Exception:
                self._device_index = None
                self._device_name = ""
                self._clock_khz = None

    def start(self) -> None:
        if not self._enabled:
            return
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        self._start.record()

    def stop(self) -> None:
        if not self._enabled or self._start is None or self._end is None:
            return
        self._end.record()
        torch.cuda.synchronize()
        try:
            self._elapsed_ms = float(self._start.elapsed_time(self._end))
        except Exception:
            self._elapsed_ms = None

    def result(self) -> Optional[dict[str, Any]]:
        if not self._enabled or self._elapsed_ms is None:
            return None
        elapsed_s = self._elapsed_ms / 1000.0
        cycles_est: Optional[int] = None
        if self._clock_khz is not None and self._clock_khz > 0:
            cycles_est = int(round(elapsed_s * float(self._clock_khz) * 1000.0))
        return {
            "device_index": self._device_index,
            "device_name": self._device_name,
            "elapsed_ms": float(self._elapsed_ms),
            "clock_khz": self._clock_khz,
            "cycles_est": cycles_est,
        }


class BasicTabularCNN(nn.Module):
    """Simple 1D CNN over feature vector treated as a sequence."""

    def __init__(self, num_features: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(64)
        # Adaptive pooling so it works for any feature count.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        x = x.unsqueeze(1)  # (batch, 1, features)
        x = F.gelu(self.conv1(x))
        x = self.dropout(x)
        x = F.gelu(self.conv2(x))
        x = x.transpose(1, 2)  # (batch, features, channels)
        x = self.norm(x).transpose(1, 2)  # (batch, channels, features)
        x = self.pool(x).squeeze(-1)  # (batch, channels)
        return self.head(x)


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
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature_cols missing from CSV: {missing}")
        feats = df[feature_cols]
    else:
        feats = df.drop(columns=[label_col])

    # Keep only numeric features by default; users can one-hot upstream if needed.
    feats = feats.select_dtypes(include=[np.number]).copy()
    if feats.shape[1] == 0:
        raise ValueError("No numeric feature columns found. Provide numeric features or pre-process the CSV.")

    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = feats.to_numpy(dtype=np.float32, copy=True)
    y_series = df[label_col]
    return x, y_series, list(feats.columns)


def run(ctx: BackboneContext, prev: list[Any]) -> dict[str, Any]:
    """Entry point used by the torch_tabular cell runner."""

    cfg = ctx.scenario_config
    bcfg = getattr(cfg, "backbone_config", {}) or {}

    dataset_csv_value = str(bcfg.get("dataset_csv") or "").strip()
    label_col = str(bcfg.get("label_col") or "label").strip()
    feature_cols_raw = bcfg.get("feature_cols")
    feature_cols = [str(c) for c in feature_cols_raw] if isinstance(feature_cols_raw, list) else None
    task = str(bcfg.get("task") or "").strip().lower()

    seed = int(bcfg.get("seed") or 1337)
    _set_seed(seed)

    val_split = float(bcfg.get("val_split") or 0.2)
    val_split = min(max(val_split, 0.05), 0.5)

    epochs = int(bcfg.get("epochs") or 20)
    batch_size = int(bcfg.get("batch_size") or 128)
    lr = float(bcfg.get("lr") or 1e-3)
    weight_decay = float(bcfg.get("weight_decay") or 1e-4)
    dropout = float(bcfg.get("dropout") or 0.2)

    dataset_csv = _resolve_path(dataset_csv_value)
    x, y_series, used_feature_cols = _load_xy(
        dataset_csv=dataset_csv,
        label_col=label_col,
        feature_cols=feature_cols,
    )
    if task not in {"classification", "regression"}:
        task = _infer_task(y_series)

    if task == "regression":
        y_num = pd.to_numeric(y_series, errors="coerce").astype(np.float32)
        if int(y_num.isna().sum()) > 0:
            # Fill missing targets with median; caller can clean upstream if desired.
            y_num = y_num.fillna(float(y_num.median() if y_num.notna().any() else 0.0))
        y = y_num.to_numpy(dtype=np.float32, copy=False)
        out_dim = 1
        label_classes: list[str] = []
    else:
        y_raw = y_series.astype(str).fillna("").to_numpy()
        enc = LabelEncoder()
        y = enc.fit_transform(y_raw).astype(np.int64, copy=False)
        out_dim = int(len(enc.classes_))
        label_classes = [str(c) for c in enc.classes_.tolist()]

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=val_split,
        random_state=seed,
        stratify=_safe_stratify_targets(y) if task == "classification" else None,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    x_val = scaler.transform(x_val).astype(np.float32, copy=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicTabularCNN(num_features=x_train.shape[1], out_dim=out_dim, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn: Any
    if task == "regression":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    if task == "regression":
        train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).view(-1, 1))
        val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val).view(-1, 1))
    else:
        train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    def _eval() -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        abs_err = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                total_loss += float(loss.item()) * int(yb.shape[0])
                if task == "regression":
                    abs_err += float(torch.abs(logits - yb).sum().item())
                else:
                    pred = torch.argmax(logits, dim=1)
                    correct += int((pred == yb).sum().item())
                total += int(yb.shape[0])
        if task == "regression":
            mae = abs_err / max(1, total)
            return (total_loss / max(1, total)), float(mae)
        return (total_loss / max(1, total)), (correct / max(1, total))

    best_val_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    gpu_cycles = _GpuCycleCounter(device)
    gpu_cycles.start()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * int(yb.shape[0])
            seen += int(yb.shape[0])

        train_loss = running / max(1, seen)
        val_loss, val_metric = _eval()
        if task == "regression":
            history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss, "val_mae": float(val_metric)})
            print(f"[epoch {epoch}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mae={val_metric:.4f}")
        else:
            history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss, "val_acc": float(val_metric)})
            print(f"[epoch {epoch}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_metric:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    gpu_cycles.stop()
    gpu_cycles_result = gpu_cycles.result()
    if gpu_cycles_result is not None:
        cyc = gpu_cycles_result.get("cycles_est")
        clk = gpu_cycles_result.get("clock_khz")
        ms = gpu_cycles_result.get("elapsed_ms")
        dev = gpu_cycles_result.get("device_name") or "cuda"
        if cyc is not None and clk is not None and ms is not None:
            print(f"[gpu] {dev}: {ms:.2f} ms  ~{cyc} cycles  (clock_rate={clk} kHz)")
        elif ms is not None:
            print(f"[gpu] {dev}: {ms:.2f} ms")

    models_root = (REPO_ROOT / "mlops" / "models" / str(cfg.name)).resolve()
    run_dir = _next_run_dir(models_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "weights.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "task": task,
            "label_classes": label_classes,
            "feature_cols": used_feature_cols,
            "label_col": label_col,
            "config": dict(bcfg),
        },
        weights_path,
    )

    metrics = {
        "best_val_loss": float(best_val_loss),
        "task": task,
        "final_val_metric": float(
            history[-1].get("val_mae", history[-1].get("val_acc", 0.0))
        ) if history else 0.0,
        "num_classes": int(len(label_classes)),
        "num_features": int(x_train.shape[1]),
        "rows": int(x.shape[0]),
    }
    if gpu_cycles_result is not None:
        metrics["gpu_elapsed_ms"] = float(gpu_cycles_result.get("elapsed_ms") or 0.0)
        if gpu_cycles_result.get("cycles_est") is not None:
            metrics["gpu_cycles_est"] = int(gpu_cycles_result["cycles_est"])
        if gpu_cycles_result.get("clock_khz") is not None:
            metrics["gpu_clock_khz"] = int(gpu_cycles_result["clock_khz"])
        if gpu_cycles_result.get("device_name"):
            metrics["gpu_device_name"] = str(gpu_cycles_result["device_name"])
    (run_dir / "metrics.json").write_text(json.dumps({"metrics": metrics, "history": history}, indent=2), encoding="utf-8")
    if task == "classification" and label_classes:
        (run_dir / "label_map.json").write_text(json.dumps({"classes": list(label_classes)}, indent=2), encoding="utf-8")

    rel_weights = str(weights_path.relative_to(REPO_ROOT)).replace("\\", "/")
    if task == "regression":
        summary = f"trained BasicTabularCNN(regression) → {run_dir.name} (val_mae={metrics['final_val_metric']:.4f})"
    else:
        summary = f"trained BasicTabularCNN(classification) → {run_dir.name} (val_acc={metrics['final_val_metric']:.4f})"

    return {
        "output": summary,
        "data": {
            "model_version": run_dir.name,
            "weights_path": rel_weights,
            "signal": {"flag": False, "summary": summary, "metrics": metrics},
            "backbone_data": {
                "run_dir": str(run_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
                "dataset_csv": str(dataset_csv),
                "label_col": label_col,
                "feature_cols": used_feature_cols,
                "seed": seed,
            },
        },
    }
