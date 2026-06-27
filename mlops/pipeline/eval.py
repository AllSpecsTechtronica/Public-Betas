from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from .governance import dataset_drift_report, load_dataset_snapshot
from .model_registry import resolve_alias
from .registry import MLOPS_ROOT, get_scenario_config, resolve_scenario_run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario evaluation suite")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--run", required=True, help="Run version, e.g. v1")
    parser.add_argument(
        "--baseline",
        default="prod",
        help="Baseline run version or alias (prod/candidate/latest/none). Default: prod",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=8,
        help="Robustness sample size from val split",
    )
    parser.add_argument("--save", action="store_true", help="Write eval_report.json into run dir")
    parser.add_argument(
        "--emit-artifacts",
        action="store_true",
        help="Run val() and write confusion_matrix.json, per_class_ap.json, and error_samples.json into the run dir",
    )
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _extract_metric(payload: dict[str, Any], key: str) -> float | None:
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    direct = _to_float(payload.get(key))
    if direct is not None:
        return direct
    mapped = _to_float(metrics.get(key))
    if mapped is not None:
        return mapped
    aliases = {
        "map50": ("metrics/mAP50(B)", "mAP50"),
        "precision": ("metrics/precision(B)",),
        "recall": ("metrics/recall(B)",),
    }
    for alt in aliases.get(key, ()):
        v = _to_float(metrics.get(alt))
        if v is not None:
            return v
    return None


def _thresholds_from_config(cfg: dict[str, Any]) -> dict[str, float]:
    hyper = cfg.get("hyperparams") if isinstance(cfg.get("hyperparams"), dict) else {}
    th = hyper.get("eval_thresholds") if isinstance(hyper.get("eval_thresholds"), dict) else {}
    return {
        "map50_min": float(th.get("map50_min", 0.15)),
        "precision_min": float(th.get("precision_min", 0.05)),
        "recall_min": float(th.get("recall_min", 0.05)),
        "max_map50_regression": float(th.get("max_map50_regression", 0.03)),
    }


def _read_results_csv(run_dir: Path) -> list[dict[str, Any]]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [{str(k).strip(): v for k, v in row.items()} for row in reader if isinstance(row, dict)]
    except Exception:
        return []


def _dataset_slice_metrics(dataset_path: Path, classes: list[str]) -> dict[str, Any]:
    labels = [p for p in (dataset_path / "labels" / "val").rglob("*.txt") if p.is_file()]
    if not labels:
        labels = [p for p in (dataset_path / "labels").rglob("*.txt") if p.is_file()]
    class_counts: dict[int, int] = {}
    image_counts: dict[int, int] = {}
    area_small = 0
    area_medium = 0
    area_large = 0
    for path in labels:
        present_in_image: set[int] = set()
        try:
            lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        except Exception:
            continue
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            cls = _to_float(parts[0])
            w = _to_float(parts[3])
            h = _to_float(parts[4])
            if cls is None or w is None or h is None:
                continue
            idx = int(cls)
            class_counts[idx] = class_counts.get(idx, 0) + 1
            present_in_image.add(idx)
            area = w * h
            if area < 0.02:
                area_small += 1
            elif area < 0.15:
                area_medium += 1
            else:
                area_large += 1
        for idx in present_in_image:
            image_counts[idx] = image_counts.get(idx, 0) + 1
    named_instances = {
        (classes[idx] if 0 <= idx < len(classes) else f"class_{idx}"): count
        for idx, count in sorted(class_counts.items(), key=lambda kv: kv[0])
    }
    named_images = {
        (classes[idx] if 0 <= idx < len(classes) else f"class_{idx}"): count
        for idx, count in sorted(image_counts.items(), key=lambda kv: kv[0])
    }
    max_instances = max(class_counts.values()) if class_counts else 0
    min_instances = min(class_counts.values()) if class_counts else 0
    imbalance_ratio = float(max_instances / max(1, min_instances)) if class_counts else 0.0
    return {
        "label_files_analyzed": len(labels),
        "class_instance_counts": named_instances,
        "class_image_counts": named_images,
        "class_imbalance_ratio": round(imbalance_ratio, 4),
        "bbox_area_buckets": {"small": area_small, "medium": area_medium, "large": area_large},
    }


def _xyxy_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(a[0]), float(a[1]), float(a[2]), float(a[3]))
    bx1, by1, bx2, by2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def _yolo_norm_to_xyxy(
    cls: int,
    xc: float,
    yc: float,
    w: float,
    h: float,
) -> tuple[int, tuple[float, float, float, float]]:
    x1 = xc - w / 2.0
    y1 = yc - h / 2.0
    x2 = xc + w / 2.0
    y2 = yc + h / 2.0
    return cls, (x1, y1, x2, y2)


def _read_yolo_label_file(path: Path) -> tuple[list[int], list[tuple[float, float, float, float]]]:
    classes: list[int] = []
    boxes: list[tuple[float, float, float, float]] = []
    try:
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    except Exception:
        return classes, boxes
    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            c = int(float(parts[0]))
            xc, yc, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except Exception:
            continue
        cls, bb = _yolo_norm_to_xyxy(c, xc, yc, w, h)
        classes.append(cls)
        boxes.append(bb)
    return classes, boxes


def _greedy_match_iou(
    pred_cls: list[int],
    pred_xyxy: list[tuple[float, float, float, float]],
    gt_cls: list[int],
    gt_xyxy: list[tuple[float, float, float, float]],
    *,
    iou_thresh: float = 0.5,
) -> tuple[int, int, float]:
    """Return false_positives, false_negatives, mean_iou of matched same-class pairs."""
    gt_used = [False] * len(gt_xyxy)
    matched_ious: list[float] = []
    for pi, pbox in enumerate(pred_xyxy):
        best_j = -1
        best_iou = 0.0
        for gj, gbox in enumerate(gt_xyxy):
            if gt_used[gj]:
                continue
            if pred_cls[pi] != gt_cls[gj]:
                continue
            iou_v = _xyxy_iou(pbox, gbox)
            if iou_v > best_iou:
                best_iou = iou_v
                best_j = gj
        if best_j >= 0 and best_iou >= iou_thresh:
            gt_used[best_j] = True
            matched_ious.append(best_iou)
    fp = len(pred_xyxy) - len(matched_ious)
    fn = sum(1 for u in gt_used if not u)
    mean_iou = float(sum(matched_ious) / max(1, len(matched_ious)))
    return fp, fn, mean_iou


def _write_validation_sidecars(
    cfg: Any,
    run_dir: Path,
    weights_path: Path,
    *,
    sample_limit: int = 180,
    top_errors: int = 40,
) -> dict[str, str]:
    """Run ``model.val`` and emit confusion / per-class / error sample JSON next to the run."""
    data_yaml = run_dir / "data.generated.yaml"
    if not data_yaml.is_file():
        raise SystemExit(f"data yaml missing: {data_yaml}")
    hp = cfg.hyperparams if isinstance(cfg.hyperparams, dict) else {}
    imgsz = int(hp.get("imgsz", 640) or 640)
    model = YOLO(str(weights_path))
    det_metrics = model.val(data=str(data_yaml), imgsz=imgsz, plots=True, verbose=False)

    cm = getattr(det_metrics, "confusion_matrix", None)
    cm_path = run_dir / "confusion_matrix.json"
    cm_written = False
    if cm is not None and getattr(cm, "matrix", None) is not None:
        try:
            mat = cm.matrix
            names_map = getattr(cm, "names", {}) or {}
            if isinstance(names_map, dict):
                class_names = [str(names_map.get(i, f"class_{i}")) for i in range(len(names_map))]
            else:
                class_names = []
            payload = {
                "version": 1,
                "scenario": cfg.name,
                "run": run_dir.name,
                "class_names": class_names,
                "matrix": mat.tolist() if hasattr(mat, "tolist") else mat,
            }
            cm_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
            cm_written = True
        except Exception:
            cm_written = False

    per_path = run_dir / "per_class_ap.json"
    rows: list[dict[str, Any]] = []
    names = det_metrics.names if isinstance(det_metrics.names, dict) else {}
    try:
        nc = int(det_metrics.box.nc) if getattr(det_metrics.box, "nc", None) else len(names)
    except Exception:
        nc = len(names)
    for i in range(nc):
        try:
            p, r, ap50, ap = det_metrics.class_result(i)
        except Exception:
            p, r, ap50, ap = 0.0, 0.0, 0.0, 0.0
        ap75 = None
        try:
            row_ap = det_metrics.box.all_ap[i]
            ap75 = float(row_ap[5])
        except Exception:
            ap75 = None
        cname = str(names.get(i, f"class_{i}"))
        rows.append(
            {
                "class_id": i,
                "name": cname,
                "precision": float(p),
                "recall": float(r),
                "ap50": float(ap50),
                "ap50_95": float(ap),
                "ap75": ap75,
            }
        )
    per_path.write_text(json.dumps({"version": 1, "classes": rows}, indent=2, ensure_ascii=True), encoding="utf-8")

    val_images = [p for p in (cfg.dataset_path / "images" / "val").rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    if not val_images:
        val_images = [
            p
            for p in (cfg.dataset_path / "images").rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    val_images = sorted(val_images)[: max(1, int(sample_limit))]
    errors: list[dict[str, Any]] = []
    for img_path in val_images:
        label_path = cfg.dataset_path / "labels" / "val" / f"{img_path.stem}.txt"
        if not label_path.is_file():
            try:
                alt = cfg.dataset_path / "labels" / img_path.relative_to(cfg.dataset_path / "images")
                alt_txt = alt.with_suffix(".txt")
                if alt_txt.is_file():
                    label_path = alt_txt
            except Exception:
                pass
        gt_cls, gt_boxes = _read_yolo_label_file(label_path)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        try:
            results = model.predict(source=img, imgsz=imgsz, verbose=False, conf=0.25)
        except Exception:
            continue
        pred_cls: list[int] = []
        pred_boxes_norm: list[tuple[float, float, float, float]] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for bi in range(len(xyxy)):
                x1, y1, x2, y2 = [float(v) for v in xyxy[bi]]
                pred_cls.append(int(clss[bi]))
                pred_boxes_norm.append((x1 / max(iw, 1), y1 / max(ih, 1), x2 / max(iw, 1), y2 / max(ih, 1)))
        fp, fn, mean_iou = _greedy_match_iou(pred_cls, pred_boxes_norm, gt_cls, gt_boxes, iou_thresh=0.5)
        score = int(fp + fn)
        errors.append(
            {
                "score": score,
                "mean_match_iou": round(mean_iou, 6),
                "fp": fp,
                "fn": fn,
                "num_pred": len(pred_cls),
                "num_gt": len(gt_cls),
                "image": (
                    str(img_path.relative_to(cfg.dataset_path))
                    if cfg.dataset_path in img_path.parents
                    else str(img_path.name)
                ),
                "pred_boxes_norm": [
                    {"cls": int(pred_cls[k]), "xyxy": [round(v, 6) for v in pred_boxes_norm[k]]}
                    for k in range(len(pred_cls))
                ],
                "gt_boxes_norm": [
                    {"cls": int(gt_cls[k]), "xyxy": [round(v, 6) for v in gt_boxes[k]]} for k in range(len(gt_cls))
                ],
            }
        )

    errors.sort(key=lambda d: (-int(d.get("score") or 0), -float(d.get("mean_match_iou") or 0.0)))
    err_path = run_dir / "error_samples.json"
    err_path.write_text(
        json.dumps(
            {
                "version": 1,
                "scenario": cfg.name,
                "run": run_dir.name,
                "top_n": min(top_errors, len(errors)),
                "samples": errors[:top_errors],
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    out: dict[str, str] = {}
    if cm_written:
        out["confusion_matrix_json"] = str(cm_path)
    out["per_class_ap_json"] = str(per_path)
    out["error_samples_json"] = str(err_path)
    return out


def _robustness_metrics(
    *,
    weights_path: Path,
    dataset_path: Path,
    sample_size: int,
) -> dict[str, Any]:
    images = [p for p in (dataset_path / "images" / "val").rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    if not images:
        images = [p for p in (dataset_path / "images").rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    if not images or not weights_path.exists():
        return {"sampled_images": 0, "retention": {}, "status": "skipped"}

    random.seed(42)
    sample = images if len(images) <= sample_size else random.sample(images, sample_size)
    model = YOLO(str(weights_path))

    def _count(img) -> int:
        try:
            results = model.predict(source=img, verbose=False, stream=False)
            out = 0
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is not None:
                    out += len(boxes)
            return int(out)
        except Exception:
            return 0

    base_total = 0
    dark_total = 0
    blur_total = 0
    noise_total = 0
    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            continue
        base = _count(img)
        base_total += base
        dark = cv2.convertScaleAbs(img, alpha=0.72, beta=-12)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        noise = np.random.normal(0.0, 8.0, img.shape).astype("float32")
        noisy = np.clip(img.astype("float32") + noise, 0, 255).astype("uint8")
        dark_total += _count(dark)
        blur_total += _count(blur)
        noise_total += _count(noisy)

    denom = max(1, base_total)
    return {
        "sampled_images": len(sample),
        "base_detections": base_total,
        "retention": {
            "darkened": round(dark_total / denom, 4),
            "blurred": round(blur_total / denom, 4),
            "noisy": round(noise_total / denom, 4),
        },
        "status": "ok",
    }


def main() -> int:
    args = _parse_args()
    cfg = get_scenario_config(args.scenario)
    run_dir = resolve_scenario_run_dir(cfg.name, args.run)
    if run_dir is None:
        raise SystemExit(f"Run not found: scenario={cfg.name} run={args.run}")
    metrics_path = Path(run_dir) / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"Metrics not found: {metrics_path}")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid metrics payload: {metrics_path}")

    thresholds = _thresholds_from_config(cfg.raw)
    current = {
        "map50": _extract_metric(payload, "map50"),
        "precision": _extract_metric(payload, "precision"),
        "recall": _extract_metric(payload, "recall"),
    }
    threshold_checks = {
        "map50_min": {
            "required": thresholds["map50_min"],
            "actual": current["map50"],
            "passed": current["map50"] is not None and current["map50"] >= thresholds["map50_min"],
        },
        "precision_min": {
            "required": thresholds["precision_min"],
            "actual": current["precision"],
            "passed": current["precision"] is not None and current["precision"] >= thresholds["precision_min"],
        },
        "recall_min": {
            "required": thresholds["recall_min"],
            "actual": current["recall"],
            "passed": current["recall"] is not None and current["recall"] >= thresholds["recall_min"],
        },
    }

    baseline_ref = str(args.baseline or "prod").strip().lower()
    baseline_metrics: dict[str, Any] = {}
    baseline_source = ""
    if baseline_ref and baseline_ref != "none":
        if baseline_ref in {"prod", "candidate"}:
            aliased = resolve_alias(cfg.name, baseline_ref)
            if isinstance(aliased, dict):
                artifacts = aliased.get("artifacts") if isinstance(aliased.get("artifacts"), dict) else {}
                p = Path(str(artifacts.get("metrics_path") or ""))
                if p.exists():
                    try:
                        baseline_metrics = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        baseline_metrics = {}
                baseline_source = f"alias:{baseline_ref}"
        elif baseline_ref == "latest":
            run_v = f"v{max(int(Path(run_dir).name[1:]) - 1, 1)}"
            baseline_dir = resolve_scenario_run_dir(cfg.name, run_v)
            if baseline_dir is not None:
                p = Path(baseline_dir) / "metrics.json"
                if p.exists():
                    try:
                        baseline_metrics = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        baseline_metrics = {}
                baseline_source = f"run:{run_v}"
        else:
            baseline_dir = resolve_scenario_run_dir(cfg.name, baseline_ref)
            if baseline_dir is not None:
                p = Path(baseline_dir) / "metrics.json"
                if p.exists():
                    try:
                        baseline_metrics = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        baseline_metrics = {}
                baseline_source = f"run:{baseline_ref}"

    baseline_map50 = _extract_metric(baseline_metrics, "map50") if baseline_metrics else None
    regression = {
        "baseline_source": baseline_source,
        "baseline_map50": baseline_map50,
        "current_map50": current["map50"],
        "max_allowed_drop": thresholds["max_map50_regression"],
        "delta_map50": (
            None
            if baseline_map50 is None or current["map50"] is None
            else round(float(current["map50"] - baseline_map50), 6)
        ),
        "passed": True,
    }
    if regression["delta_map50"] is not None:
        regression["passed"] = bool(regression["delta_map50"] >= -thresholds["max_map50_regression"])

    slices = _dataset_slice_metrics(cfg.dataset_path, cfg.classes)
    robustness = _robustness_metrics(
        weights_path=Path(str(payload.get("weights") or (Path(run_dir) / "weights.pt"))),
        dataset_path=cfg.dataset_path,
        sample_size=max(1, int(args.sample_size)),
    )

    drift = {}
    current_snapshot_id = str(payload.get("dataset_snapshot_id") or "")
    current_snapshot = load_dataset_snapshot(current_snapshot_id)
    baseline_snapshot_id = str((baseline_metrics or {}).get("dataset_snapshot_id") or "")
    baseline_snapshot = load_dataset_snapshot(baseline_snapshot_id) if baseline_snapshot_id else None
    if isinstance(current_snapshot, dict) and isinstance(baseline_snapshot, dict):
        drift = dataset_drift_report(current_snapshot, baseline_snapshot)

    passed = all(bool(v.get("passed")) for v in threshold_checks.values()) and bool(regression["passed"])
    report = {
        "version": 1,
        "scenario": cfg.name,
        "run": Path(run_dir).name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": thresholds,
        "current_metrics": current,
        "threshold_checks": threshold_checks,
        "regression": regression,
        "slice_metrics": slices,
        "robustness": robustness,
        "dataset_drift": drift,
        "results_rows": len(_read_results_csv(Path(run_dir))),
        "status": "passed" if passed else "failed",
    }
    if args.save:
        out_path = Path(run_dir) / "eval_report.json"
        out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
        report["saved_to"] = str(out_path)
    if args.emit_artifacts:
        weights = Path(str(payload.get("weights") or (Path(run_dir) / "weights.pt")))
        if not weights.is_file():
            raise SystemExit(f"Weights not found for artifact emit: {weights}")
        sidecars = _write_validation_sidecars(cfg, Path(run_dir), weights)
        report["eval_artifacts"] = sidecars
    print(json.dumps(report, ensure_ascii=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
