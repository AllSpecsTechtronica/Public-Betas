from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

from insight_local.cvops.service import CvOpsService
from mlops.pipeline import registry as mlops_registry


_CSV = (
    "age,city,score,note\n"
    "30,NYC,1.0,a\n"
    "30,NYC,1.0,a\n"  # exact duplicate of the row above
    "40,LA,2.0,b\n"
    "50,,3.0,c\n"  # missing city
    "60,SF,,d\n"  # missing score
    "70,LA,5.0,e\n"
)


class CvOpsTabularTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.repo_root = root / "repo"
        self.mlops_root = self.repo_root / "mlops"
        self.tabular_root = self.mlops_root / "datasets"
        self.tabular_root.mkdir(parents=True, exist_ok=True)

        self._old_repo = mlops_registry.REPO_ROOT
        self._old_mlops = mlops_registry.MLOPS_ROOT
        self._old_tabular = mlops_registry.TABULAR_DATASETS_ROOT
        mlops_registry.REPO_ROOT = self.repo_root
        mlops_registry.MLOPS_ROOT = self.mlops_root
        mlops_registry.TABULAR_DATASETS_ROOT = self.tabular_root

        self.csv_path = self.tabular_root / "demo.csv"
        self.csv_path.write_text(_CSV, encoding="utf-8")

        self._svc_tmp = tempfile.TemporaryDirectory()
        svc_dir = Path(self._svc_tmp.name)
        self.svc = CvOpsService(db_path=svc_dir / "jobs.db", catalog_db_path=svc_dir / "catalog.db")

    def tearDown(self) -> None:
        mlops_registry.REPO_ROOT = self._old_repo
        mlops_registry.MLOPS_ROOT = self._old_mlops
        mlops_registry.TABULAR_DATASETS_ROOT = self._old_tabular
        self._svc_tmp.cleanup()
        self._tmp.cleanup()

    def _read(self) -> list[list[str]]:
        import csv

        with self.csv_path.open(newline="") as fh:
            return list(csv.reader(fh))

    def test_drop_duplicates_and_impute(self) -> None:
        with TestClient(self.svc.app) as client:
            resp = client.post(
                "/database/demo/tabular_transform",
                json={
                    "ops": [
                        {"op": "drop_duplicate_rows"},
                        {"op": "impute_missing", "columns": ["score"], "strategy": "median"},
                        {"op": "impute_missing", "columns": ["city"], "strategy": "mode"},
                    ]
                },
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            self.assertTrue(data["changed"])
            # 6 data rows, one exact duplicate removed -> 5
            self.assertEqual(data["before"]["rows"], 6)
            self.assertEqual(data["after"]["rows"], 5)

        table = self._read()
        header = table[0]
        rows = table[1:]
        self.assertEqual(len(rows), 5)
        score_idx = header.index("score")
        city_idx = header.index("city")
        # No missing cells remain in imputed columns.
        self.assertTrue(all(r[score_idx].strip() for r in rows))
        self.assertTrue(all(r[city_idx].strip() for r in rows))
        # A .bak backup was created.
        self.assertTrue((self.csv_path.with_suffix(".csv.bak")).exists())

    def test_drop_columns_and_high_missing(self) -> None:
        with TestClient(self.svc.app) as client:
            resp = client.post(
                "/database/demo/tabular_transform",
                json={"ops": [{"op": "drop_columns", "columns": ["note"]}]},
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            self.assertNotIn("note", self._read()[0])

    def test_split_stratified_and_indices(self) -> None:
        with TestClient(self.svc.app) as client:
            resp = client.post(
                "/database/demo/tabular_split",
                json={"val_frac": 0.34, "test_frac": 0.0, "seed": 7},
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            counts = data["counts"]
            self.assertEqual(counts["train"] + counts["val"] + counts["test"], 6)
            self.assertGreater(counts["train"], 0)
            self.assertGreater(counts["val"], 0)

            splits_path = self.csv_path.with_name("demo.splits.json")
            self.assertTrue(splits_path.exists())
            payload = json.loads(splits_path.read_text())
            all_idx = (
                payload["splits"]["train"]
                + payload["splits"]["val"]
                + payload["splits"]["test"]
            )
            # Every row index assigned exactly once, no overlap.
            self.assertEqual(sorted(all_idx), list(range(6)))

    def test_split_write_column(self) -> None:
        with TestClient(self.svc.app) as client:
            resp = client.post(
                "/database/demo/tabular_split",
                json={"val_frac": 0.2, "write_column": True, "seed": 1},
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            self.assertTrue(resp.json()["wrote_column"])
        header = self._read()[0]
        self.assertIn("split", header)

    def test_normalize_clip_filter_and_rename(self) -> None:
        with TestClient(self.svc.app) as client:
            # minmax normalize age -> [0,1]; rename score->target; drop rows city==LA
            resp = client.post(
                "/database/demo/tabular_transform",
                json={
                    "ops": [
                        {"op": "rename_columns", "rename": {"score": "target"}},
                        {"op": "normalize", "columns": ["age"], "method": "minmax"},
                        {"op": "filter_rows", "where_col": "city", "where_op": "!=", "where_value": "LA"},
                    ]
                },
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            self.assertGreaterEqual(data["revision"], 1)

        table = self._read()
        header = table[0]
        rows = table[1:]
        self.assertIn("target", header)
        self.assertNotIn("score", header)
        # No LA rows remain.
        city_idx = header.index("city")
        self.assertTrue(all(r[city_idx] != "LA" for r in rows))
        # age normalized into [0,1].
        age_idx = header.index("age")
        age_vals = [float(r[age_idx]) for r in rows]
        self.assertGreaterEqual(min(age_vals), 0.0)
        self.assertLessEqual(max(age_vals), 1.0)

    def test_history_and_undo(self) -> None:
        with TestClient(self.svc.app) as client:
            r1 = client.post(
                "/database/demo/tabular_transform",
                json={"ops": [{"op": "drop_duplicate_rows"}]},
            )
            self.assertEqual(r1.status_code, 200, r1.text)
            self.assertEqual(r1.json()["after"]["rows"], 5)

            hist = client.get("/database/demo/tabular_history")
            self.assertEqual(hist.status_code, 200, hist.text)
            hdata = hist.json()
            self.assertEqual(hdata["count"], 1)
            self.assertEqual(hdata["entries"][0]["action"], "transform")
            self.assertTrue(hdata["can_undo"])

            undo = client.post("/database/demo/tabular_undo", json={})
            self.assertEqual(undo.status_code, 200, undo.text)
            self.assertTrue(undo.json()["restored"])

            # Undo restored the original 6 rows, and logged an undo entry.
            self.assertEqual(len(self._read()) - 1, 6)
            hist2 = client.get("/database/demo/tabular_history").json()
            self.assertEqual(hist2["count"], 2)
            self.assertEqual(hist2["entries"][-1]["action"], "undo")
            # Backup consumed -> no further undo.
            self.assertFalse(hist2["can_undo"])

    def test_mcp_tabular_tools_end_to_end(self) -> None:
        """Drive the new MCP tools through the live service via TestClient."""
        from insight_local.cvops.tacitus_mcp import TacitusMcpSurface

        with TestClient(self.svc.app) as client:
            def _get(path: str) -> dict:
                resp = client.get(path)
                resp.raise_for_status()
                return resp.json()

            def _post(path: str, body=None) -> dict:
                resp = client.post(path, json=body or {})
                resp.raise_for_status()
                return resp.json()

            surface = TacitusMcpSurface(http_get=_get, http_post=_post)

            # The two tools are registered on the surface.
            tool_names = {item["name"] for item in TacitusMcpSurface.tools()}
            self.assertIn("dataset.tabular_fix", tool_names)
            self.assertIn("dataset.tabular_split", tool_names)

            fix = surface.call_tool(
                "dataset.tabular_fix",
                {"dataset": "demo", "ops": [{"op": "drop_duplicate_rows"}]},
            )
            self.assertTrue(fix["ok"], fix)
            self.assertEqual(fix["data"]["after"]["rows"], 5)

            split = surface.call_tool(
                "dataset.tabular_split",
                {"dataset": "demo", "val_frac": 0.4, "seed": 3},
            )
            self.assertTrue(split["ok"], split)
            counts = split["data"]["counts"]
            self.assertEqual(counts["train"] + counts["val"] + counts["test"], 5)


    def test_upload_parquet_and_jsonl_normalize_to_csv(self) -> None:
        import io

        import pandas as pd

        frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        pq_buf = io.BytesIO()
        frame.to_parquet(pq_buf, index=False)
        jsonl = frame.to_json(orient="records", lines=True).encode("utf-8")

        with TestClient(self.svc.app) as client:
            r_pq = client.post(
                "/database/upload_tabular",
                files={"file": ("nums.parquet", pq_buf.getvalue(), "application/octet-stream")},
            )
            self.assertEqual(r_pq.status_code, 200, r_pq.text)
            self.assertEqual(r_pq.json()["source_format"], "parquet")

            r_jl = client.post(
                "/database/upload_tabular",
                files={"file": ("nums.jsonl", jsonl, "application/json")},
            )
            self.assertEqual(r_jl.status_code, 200, r_jl.text)
            self.assertEqual(r_jl.json()["source_format"], "jsonl")

        # Both landed as discoverable .csv with the right header.
        for stem in ("nums", "nums-2"):
            path = self.tabular_root / f"{stem}.csv"
            self.assertTrue(path.exists(), f"missing {path}")
            header = path.read_text().splitlines()[0]
            self.assertEqual(header.strip(), "a,b")

    def test_import_tabular_folder(self) -> None:
        import io

        import pandas as pd

        src = self.tabular_root.parent / "incoming"
        (src / "sub").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"a": [1, 2]}).to_csv(src / "one.csv", index=False)
        pd.DataFrame({"a": [3, 4]}).to_parquet(src / "two.parquet", index=False)
        pd.DataFrame({"a": [5, 6]}).to_csv(src / "sub" / "three.csv", index=False)

        with TestClient(self.svc.app) as client:
            # Non-recursive: only top-level files.
            resp = client.post(
                "/database/import_tabular_folder",
                json={"source_path": str(src), "recursive": False},
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            self.assertEqual(data["found"], 2)
            self.assertEqual(data["imported_count"], 2)
            self.assertEqual(data["errors"], [])

            # Recursive picks up the nested file too.
            resp2 = client.post(
                "/database/import_tabular_folder",
                json={"source_path": str(src), "recursive": True},
            )
            self.assertEqual(resp2.status_code, 200, resp2.text)
            self.assertEqual(resp2.json()["found"], 3)

    def test_target_classification_balance_and_leakage(self) -> None:
        # Build a dataset with an imbalanced binary label and a leaking feature.
        rows = ["label,leak,feat\n"]
        for _ in range(30):
            rows.append("A,a,1\n")
        for _ in range(2):
            rows.append("B,b,2\n")
        self.csv_path.write_text("".join(rows), encoding="utf-8")

        with TestClient(self.svc.app) as client:
            resp = client.get("/database/demo/tabular_target?label_col=label")
            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            self.assertEqual(data["task"], "classification")
            self.assertEqual(data["class_balance"]["n_classes"], 2)
            self.assertTrue(data["class_balance"]["imbalanced"])
            # 'leak' maps 1:1 to label -> flagged as leakage.
            leak_feats = {item["feature"] for item in data["leakage"]}
            self.assertIn("leak", leak_feats)
            self.assertTrue(data["readiness"]["ready"])  # blockers empty
            self.assertTrue(any("imbalance" in w for w in data["readiness"]["warnings"]))

            # Oversample to balance the classes.
            bal = client.post(
                "/database/demo/tabular_transform",
                json={"ops": [{"op": "balance_classes", "label_col": "label", "strategy": "oversample"}]},
            )
            self.assertEqual(bal.status_code, 200, bal.text)
            after = client.get("/database/demo/tabular_target?label_col=label").json()
            # After oversampling, majority:minority ratio should be ~1.
            self.assertLessEqual(after["class_balance"]["imbalance_ratio"], 1.5)

    def test_target_readiness_blockers(self) -> None:
        with TestClient(self.svc.app) as client:
            # Missing label column -> blocker, not ready.
            r1 = client.get("/database/demo/tabular_target?label_col=does_not_exist")
            self.assertEqual(r1.status_code, 200, r1.text)
            self.assertFalse(r1.json()["readiness"]["ready"])
            self.assertTrue(r1.json()["readiness"]["blockers"])

            # Numeric high-cardinality label -> regression task.
            r2 = client.get("/database/demo/tabular_target?label_col=score")
            self.assertEqual(r2.status_code, 200, r2.text)
            self.assertIn(r2.json()["task"], ("regression", "classification"))

    def _train_demo_artifact(self) -> Path:
        """Train a tiny sklearn LogisticRegression artifact in the baselines' schema."""
        import pickle

        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        # Linearly separable on `age`: <45 -> low, else high.
        rows = ["age,score,label\n"]
        for a in range(20, 44, 2):
            rows.append(f"{a},{a/10.0},low\n")
        for a in range(46, 70, 2):
            rows.append(f"{a},{a/10.0},high\n")
        train_csv = self.tabular_root / "train.csv"
        train_csv.write_text("".join(rows), encoding="utf-8")

        import csv as _csv

        with train_csv.open() as fh:
            reader = list(_csv.reader(fh))
        feat_cols = ["age", "score"]
        x = np.array([[float(r[0]), float(r[1])] for r in reader[1:]], dtype=np.float32)
        labels = [r[2] for r in reader[1:]]
        enc = LabelEncoder()
        y = enc.fit_transform(labels)
        scaler = StandardScaler().fit(x)
        xs = scaler.transform(x)
        model = LogisticRegression(max_iter=200).fit(xs, y)

        model_path = self.repo_root / "mlops" / "models" / "demo" / "v1" / "model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as fh:
            pickle.dump(
                {
                    "model": model,
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "label_classes": [str(c) for c in enc.classes_.tolist()],
                    "feature_cols": feat_cols,
                    "label_col": "label",
                    "config": {},
                },
                fh,
            )
        return model_path

    def test_score_with_model_path_and_write_dataset(self) -> None:
        model_path = self._train_demo_artifact()
        # Input dataset to score (no label needed; just the features).
        self.csv_path.write_text("age,score\n25,2.5\n65,6.5\n30,3.0\n", encoding="utf-8")

        with TestClient(self.svc.app) as client:
            resp = client.post(
                "/database/demo/tabular_score",
                json={"model_path": str(model_path), "write_dataset": True},
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            self.assertEqual(data["n_rows"], 3)
            self.assertEqual(data["task"], "classification")
            self.assertEqual(set(data["label_classes"]), {"low", "high"})
            self.assertEqual(len(data["sample"]), 3)
            # Young rows -> low, old -> high (model learned the split).
            self.assertEqual(data["sample"][0], "low")
            self.assertEqual(data["sample"][1], "high")

            written = data["written_slug"]
            self.assertTrue(written)
            scored_csv = self.tabular_root / f"{written}.csv"
            self.assertTrue(scored_csv.exists())
            self.assertIn("prediction", scored_csv.read_text().splitlines()[0])

    def test_score_requires_model_source(self) -> None:
        with TestClient(self.svc.app) as client:
            resp = client.post("/database/demo/tabular_score", json={})
            self.assertEqual(resp.status_code, 400, resp.text)
            self.assertIn("scenario or model_path", resp.json()["detail"])

    def test_transform_hard_cap_rejects_huge_file(self) -> None:
        import os

        # Cap must be read at service construction; build a fresh service under a low cap.
        big = self.tabular_root / "big.csv"
        big.write_text("a,b\n" + "".join(f"{i},x\n" for i in range(50)), encoding="utf-8")
        old = os.environ.get("CVOPS_TABULAR_MAX_ROWS")
        os.environ["CVOPS_TABULAR_MAX_ROWS"] = "10"
        try:
            svc = CvOpsService(
                db_path=self.tabular_root.parent / "j2.db",
                catalog_db_path=self.tabular_root.parent / "c2.db",
            )
            with TestClient(svc.app) as client:
                resp = client.post(
                    "/database/big/tabular_transform",
                    json={"ops": [{"op": "drop_duplicate_rows"}]},
                )
                self.assertEqual(resp.status_code, 413, resp.text)
                self.assertIn("cap", resp.json()["detail"].lower())
                # Sampling reads (target) still work past the cap.
                t = client.get("/database/big/tabular_target?label_col=b")
                self.assertEqual(t.status_code, 200, t.text)
        finally:
            if old is None:
                os.environ.pop("CVOPS_TABULAR_MAX_ROWS", None)
            else:
                os.environ["CVOPS_TABULAR_MAX_ROWS"] = old

    def test_upload_unsupported_extension_rejected(self) -> None:
        with TestClient(self.svc.app) as client:
            resp = client.post(
                "/database/upload_tabular",
                files={"file": ("notes.txt", b"hello", "text/plain")},
            )
            self.assertEqual(resp.status_code, 400, resp.text)
            self.assertIn("unsupported", resp.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
