from __future__ import annotations

import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Insight"))

from mlops.pipeline import train as mlops_train
from mlops.pipeline import registry as mlops_registry


class TrainingDeviceTests(unittest.TestCase):
    def test_quality_stop_hyperparam_validation(self) -> None:
        self.assertTrue(mlops_registry._coerce_hyperparam_value("quality_stop_enabled", "true"))
        self.assertEqual(
            mlops_registry._coerce_hyperparam_value("quality_stop_metric", "map50_95"),
            "map50_95",
        )
        self.assertEqual(
            mlops_registry._coerce_hyperparam_value("quality_stop_threshold", "0.9"),
            0.9,
        )
        with self.assertRaises(ValueError):
            mlops_registry._coerce_hyperparam_value("quality_stop_metric", "accuracy")
        with self.assertRaises(ValueError):
            mlops_registry._coerce_hyperparam_value("quality_stop_threshold", "1.5")

    def test_extract_training_point_preserves_map50_and_adds_map50_95(self) -> None:
        point = mlops_train._extract_training_point(
            {
                "epoch": "8",
                "metrics/mAP50(B)": "0.928",
                "metrics/mAP50-95(B)": "0.900",
                "metrics/precision(B)": "0.954",
                "metrics/recall(B)": "0.916",
            },
            epochs=20,
            run_dir=Path("/tmp/run"),
        )

        self.assertIsNotNone(point)
        assert point is not None
        self.assertEqual(point["map50"], 0.928)
        self.assertEqual(point["map50_95"], 0.900)
        self.assertEqual(point["precision"], 0.954)
        self.assertEqual(point["recall"], 0.916)

    def test_extract_training_point_keeps_legacy_map50_fallback(self) -> None:
        point = mlops_train._extract_training_point(
            {"epoch": "1", "metrics/mAP50-95(B)": "0.812"},
            epochs=20,
            run_dir=Path("/tmp/run"),
        )

        self.assertIsNotNone(point)
        assert point is not None
        self.assertEqual(point["map50"], 0.812)
        self.assertEqual(point["map50_95"], 0.812)

    def test_quality_stop_evaluator_requires_min_epoch_and_consecutive_hits(self) -> None:
        config = mlops_train._quality_stop_config(
            {
                "quality_stop_enabled": True,
                "quality_stop_metric": "map50_95",
                "quality_stop_threshold": 0.90,
                "quality_stop_min_epochs": 5,
                "quality_stop_consecutive_epochs": 2,
            }
        )
        state: dict[str, object] = {}

        before_min = mlops_train._evaluate_quality_stop(state, {"epoch": 3, "map50_95": 0.95}, config)
        self.assertFalse(before_min["should_stop"])
        self.assertEqual(before_min["consecutive_epochs"], 0)

        first_hit = mlops_train._evaluate_quality_stop(state, {"epoch": 4, "map50_95": 0.91}, config)
        self.assertFalse(first_hit["should_stop"])
        self.assertEqual(first_hit["consecutive_epochs"], 1)

        second_hit = mlops_train._evaluate_quality_stop(state, {"epoch": 5, "map50_95": 0.92}, config)
        self.assertTrue(second_hit["should_stop"])
        self.assertEqual(second_hit["consecutive_epochs"], 2)

    def test_quality_stop_evaluator_does_not_stop_on_single_spike(self) -> None:
        config = mlops_train._quality_stop_config(
            {
                "quality_stop_enabled": True,
                "quality_stop_threshold": 0.90,
                "quality_stop_min_epochs": 5,
                "quality_stop_consecutive_epochs": 2,
            }
        )
        state: dict[str, object] = {}

        first_hit = mlops_train._evaluate_quality_stop(state, {"epoch": 4, "map50_95": 0.91}, config)
        miss = mlops_train._evaluate_quality_stop(state, {"epoch": 5, "map50_95": 0.89}, config)

        self.assertFalse(first_hit["should_stop"])
        self.assertFalse(miss["should_stop"])
        self.assertEqual(miss["consecutive_epochs"], 0)

    def test_quality_stop_evaluator_disabled_noops(self) -> None:
        config = mlops_train._quality_stop_config({"quality_stop_enabled": False})
        state: dict[str, object] = {}

        decision = mlops_train._evaluate_quality_stop(state, {"epoch": 10, "map50_95": 0.99}, config)

        self.assertFalse(decision["should_stop"])
        self.assertEqual(decision["consecutive_epochs"], 0)

    def test_quality_stop_evaluator_rapid_clear_relaxes_min_epoch(self) -> None:
        config = mlops_train._quality_stop_config(
            {
                "quality_stop_enabled": True,
                "quality_stop_metric": "map50_95",
                "quality_stop_threshold": 0.90,
                "quality_stop_min_epochs": 5,
                "quality_stop_consecutive_epochs": 2,
                "quality_stop_rapid_clear_enabled": True,
                "quality_stop_rapid_clear_loss_ratio": 0.35,
                "quality_stop_rapid_clear_metric_margin": 0.03,
            }
        )
        state: dict[str, object] = {}

        warmup = mlops_train._evaluate_quality_stop(
            state,
            {"epoch": 0, "map50_95": 0.80, "train_loss": 4.0},
            config,
        )
        first_fast_hit = mlops_train._evaluate_quality_stop(
            state,
            {"epoch": 1, "map50_95": 0.96, "train_loss": 1.0},
            config,
        )
        second_fast_hit = mlops_train._evaluate_quality_stop(
            state,
            {"epoch": 2, "map50_95": 0.97, "train_loss": 0.9},
            config,
        )

        self.assertFalse(warmup["should_stop"])
        self.assertFalse(first_fast_hit["should_stop"])
        self.assertTrue(first_fast_hit["rapid_clear_triggered"])
        self.assertEqual(first_fast_hit["effective_min_epochs"], 1)
        self.assertTrue(second_fast_hit["should_stop"])
        self.assertEqual(second_fast_hit["mode"], "threshold")

    def test_quality_stop_evaluator_stops_on_post_peak_regression(self) -> None:
        config = mlops_train._quality_stop_config(
            {
                "quality_stop_enabled": True,
                "quality_stop_metric": "map50_95",
                "quality_stop_threshold": 0.90,
                "quality_stop_min_epochs": 5,
                "quality_regression_enabled": True,
                "quality_regression_abs_tolerance": 0.05,
                "quality_regression_rel_tolerance": 0.15,
                "quality_regression_consecutive_epochs": 1,
            }
        )
        state: dict[str, object] = {}

        for epoch, value in enumerate((0.82, 0.91, 0.96, 0.97, 0.79)):
            decision = mlops_train._evaluate_quality_stop(
                state,
                {"epoch": epoch, "map50_95": value, "train_loss": max(0.5, 4.0 - epoch)},
                config,
            )

        self.assertTrue(decision["should_stop"])
        self.assertEqual(decision["mode"], "regression")
        self.assertEqual(decision["peak_epoch"], 3)
        self.assertAlmostEqual(float(decision["peak_value"]), 0.97, places=6)
        self.assertEqual(decision["recommended_max_epochs"], 4)
        self.assertGreater(float(decision["abs_drop"]), 0.05)

    def test_resolve_training_device_from_system_specs(self) -> None:
        self.assertEqual(
            mlops_train._resolve_training_device({"accelerator": "cuda", "gpu_count": 1}),
            "0",
        )
        self.assertEqual(
            mlops_train._resolve_training_device({"accelerator": "mps", "gpu_count": 1}),
            "mps",
        )
        self.assertEqual(
            mlops_train._resolve_training_device({"accelerator": "cpu", "gpu_count": 0}),
            "cpu",
        )

    def test_run_training_passes_explicit_device_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            run_dir = tmp / "models" / "demo" / "v1"
            weights_dir = run_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            best_pt = weights_dir / "best.pt"
            last_pt = weights_dir / "last.pt"
            best_pt.write_bytes(b"best")
            last_pt.write_bytes(b"last")
            config_path = tmp / "scenarios" / "demo.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("name: demo\n", encoding="utf-8")
            data_yaml = run_dir / "data.generated.yaml"
            data_yaml.write_text("train: images/train\nval: images/val\n", encoding="utf-8")
            snapshot_path = run_dir / "dataset.snapshot.json"
            snapshot_path.write_text("{}", encoding="utf-8")
            target_weights = tmp / "weights" / "demo.pt"
            target_weights.parent.mkdir(parents=True, exist_ok=True)

            cfg = SimpleNamespace(
                name="demo",
                dataset="demo_ds",
                dataset_path=tmp / "dataset",
                classes=["a"],
                hyperparams={"epochs": 1, "imgsz": 640},
                weights_path=target_weights,
                base_model="base.pt",
                config_path=config_path,
            )

            train_calls: list[dict[str, object]] = []

            class _FakeYOLO:
                def __init__(self, _model: str) -> None:
                    pass

                def add_callback(self, *_args, **_kwargs) -> None:
                    return None

                def train(self, **kwargs):
                    train_calls.append(dict(kwargs))
                    return SimpleNamespace(save_dir=run_dir, results_dict={})

            guard = {
                "status": "ok",
                "summary": "test guard",
                "model_scale": "n",
                "requested_hyperparams": {"epochs": 1, "imgsz": 640},
                "effective_hyperparams": {"epochs": 1, "imgsz": 640, "batch": 1, "workers": 0},
                "adjustments": [],
                "system_specs": {"accelerator": "mps", "gpu_count": 1},
            }

            with (
                mock.patch.object(mlops_train, "get_scenario_config", return_value=cfg),
                mock.patch.object(mlops_train, "validate_trainer_name", return_value="ultralytics_yolo"),
                mock.patch.object(mlops_train, "_find_latest_resume_checkpoint", return_value=(run_dir, last_pt)),
                mock.patch.object(mlops_train, "_build_data_yaml", return_value=data_yaml),
                mock.patch.object(mlops_train, "_resolve_base_model", return_value=("base.pt", "")),
                mock.patch.object(
                    mlops_train,
                    "create_dataset_snapshot",
                    return_value={"snapshot_id": "snap-1", "contract": {"status": "ok"}},
                ),
                mock.patch.object(mlops_train, "persist_dataset_snapshot", return_value=snapshot_path),
                mock.patch.object(mlops_train, "build_training_guard", return_value=guard),
                mock.patch.object(mlops_train, "_capture_training_logs", side_effect=lambda _cb: contextlib.nullcontext()),
                mock.patch.object(mlops_train, "capture_environment_fingerprint", return_value={}),
                mock.patch.object(mlops_train, "create_repro_manifest", return_value={"replay_command": "python train.py"}),
                mock.patch.object(mlops_train, "register_model_version", return_value={}),
                mock.patch.object(mlops_train, "forecast_run", return_value={}),
                mock.patch.object(mlops_train, "render_forecast", return_value=""),
                mock.patch.object(mlops_train, "_extract_last_results_csv_row", return_value={}),
                mock.patch.object(mlops_train, "YOLO", _FakeYOLO),
            ):
                summary = mlops_train.run_training("demo", resume=True)

            self.assertEqual(summary["resumed_from"], str(last_pt))
            self.assertEqual(len(train_calls), 1)
            self.assertEqual(train_calls[0]["device"], "mps")
            self.assertEqual(train_calls[0]["resume"], str(last_pt))


if __name__ == "__main__":
    unittest.main()
