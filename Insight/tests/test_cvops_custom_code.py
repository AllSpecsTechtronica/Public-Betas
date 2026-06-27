from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

from fastapi.testclient import TestClient

from insight_local.cvops.service import CvOpsService
from mlops.pipeline import registry as mlops_registry
from mlops.pipeline.backbone import BackboneContext
from mlops.pipeline.backbones import get_backbone


class CustomCodeBackboneTests(unittest.TestCase):
    def test_custom_code_ctx_and_two_cells(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo_root = tmp
            mlops_root = repo_root / "mlops"
            (mlops_root / "scenarios").mkdir(parents=True)
            (mlops_root / "registry.json").write_text('{"version": 1, "scenarios": []}', encoding="utf-8")

            old_repo = mlops_registry.REPO_ROOT
            old_mlops = mlops_registry.MLOPS_ROOT
            old_registry = mlops_registry.REGISTRY_PATH
            try:
                mlops_registry.REPO_ROOT = repo_root
                mlops_registry.MLOPS_ROOT = mlops_root
                mlops_registry.REGISTRY_PATH = mlops_root / "registry.json"

                c1 = mlops_root / "cell_one.py"
                c1.write_text(
                    "def run(ctx, prev):\n"
                    "    assert ctx.datasets is not None\n"
                    "    assert ctx.active_cell is not None\n"
                    "    return {'data': {'cell': 1}}\n",
                    encoding="utf-8",
                )
                c2 = mlops_root / "cell_two.py"
                c2.write_text(
                    "def run(ctx, prev):\n"
                    "    p = ctx.active_cell.get('pasted_data_dir', '')\n"
                    "    assert 'draft/data' in p.replace('\\\\', '/')\n"
                    "    return {'data': {'cell': 2}}\n",
                    encoding="utf-8",
                )
                # Use absolute paths so cell loading does not depend on REPO_ROOT rebinding.
                rel1 = str(c1.resolve())
                rel2 = str(c2.resolve())

                mlops_registry.create_scenario_profile(
                    name="cc_demo",
                    display_name="CC Demo",
                    description="test",
                    dataset="",
                    backbone_type="custom_code",
                    backbone_config={
                        "cells": [
                            {
                                "id": "a1",
                                "name": "first",
                                "path": rel1,
                                "entry": "run",
                                "datasets": [
                                    {
                                        "name": "p",
                                        "kind": "inline_text",
                                        "path": "mlops/custom_cells/cc_demo/draft/data/a1/sample.csv",
                                        "format": "csv",
                                        "mode": "managed_copy",
                                    }
                                ],
                            },
                            {"id": "a2", "name": "second", "path": rel2, "entry": "run", "datasets": []},
                        ],
                        "datasets": [
                            {"name": "primary", "kind": "file", "path": rel1, "format": "py", "mode": "reference"}
                        ],
                    },
                )
                cfg = mlops_registry.get_scenario_config("cc_demo")
                pasted = (
                    mlops_root / "custom_cells" / "cc_demo" / "draft" / "data" / "a1"
                )
                pasted.mkdir(parents=True, exist_ok=True)
                (pasted / "sample.csv").write_text("x,y\n1,2\n", encoding="utf-8")

                bb = get_backbone("custom_code", cfg)
                results: list[dict] = []

                def _cb(payload: dict) -> None:
                    results.append(payload)

                out = bb.run(
                    BackboneContext(
                        scenario_config=cfg,
                        job_id="job-cc",
                        job_type="train",
                        image_bgr=None,
                        payload={},
                        cell_callback=_cb,
                    )
                )
                self.assertFalse(out.get("error"), out)
                rp = str(out.get("result_path") or "").strip()
                self.assertTrue(rp)
                run_root = (repo_root / rp).resolve()
                self.assertTrue((run_root / "metrics.json").exists())
            finally:
                mlops_registry.REPO_ROOT = old_repo
                mlops_registry.MLOPS_ROOT = old_mlops
                mlops_registry.REGISTRY_PATH = old_registry


class CustomCellsApiTests(unittest.TestCase):
    def test_custom_cells_put_get_promote_patches_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo_root = tmp / "repo"
            database_root = repo_root / "database"
            mlops_root = repo_root / "mlops"
            (mlops_root / "scenarios").mkdir(parents=True)
            (mlops_root / "algos").mkdir(parents=True)
            (mlops_root / "registry.json").write_text('{"version": 1, "scenarios": []}', encoding="utf-8")

            old_repo = mlops_registry.REPO_ROOT
            old_mlops = mlops_registry.MLOPS_ROOT
            old_db = mlops_registry.DATABASE_ROOT
            old_registry = mlops_registry.REGISTRY_PATH
            try:
                mlops_registry.REPO_ROOT = repo_root
                mlops_registry.MLOPS_ROOT = mlops_root
                mlops_registry.DATABASE_ROOT = database_root
                mlops_registry.REGISTRY_PATH = mlops_root / "registry.json"

                mlops_registry.create_scenario_profile(
                    name="cc_api",
                    display_name="CC API",
                    description="t",
                    dataset="",
                    backbone_type="custom_code",
                    backbone_config={"cells": [], "datasets": []},
                )

                svc = CvOpsService(db_path=tmp / "jobs.db", catalog_db_path=tmp / "catalog.db")
                with TestClient(svc.app) as client:
                    put = client.put(
                        "/scenarios/cc_api/custom_cells",
                        json={
                            "cells": [
                                {
                                    "id": "c1",
                                    "name": "TrainCell",
                                    "code": "def run(ctx, prev):\n    return {'data': {}}\n",
                                    "entry": "run",
                                    "datasets": [],
                                    "pasted_files": [
                                        {"name": "data/blob.txt", "content": "hello", "format": "text"},
                                    ],
                                }
                            ],
                            "scenario_datasets": [
                                {"name": "primary", "kind": "database", "path": "database/x", "format": "yolo", "mode": "reference"}
                            ],
                        },
                    )
                    self.assertEqual(put.status_code, 200, put.text)
                    got = client.get("/scenarios/cc_api/custom_cells")
                    self.assertEqual(got.status_code, 200, got.text)
                    body = got.json()
                    self.assertEqual(len(body.get("cells") or []), 1)
                    self.assertIn("def run", str(body["cells"][0].get("code") or ""))
                    self.assertEqual(body["cells"][0]["pasted_files"][0]["name"], "data/blob.txt")
                    self.assertEqual(body["cells"][0]["pasted_files"][0]["content"], "hello")
                    self.assertTrue(
                        (
                            repo_root
                            / "mlops/custom_cells/cc_api/draft/data/c1/data/blob.txt"
                        ).is_file()
                    )

                    pr = client.post(
                        "/scenarios/cc_api/custom_cells/promote",
                        json={"template_name": "MyTpl"},
                    )
                    self.assertEqual(pr.status_code, 200, pr.text)
                    cfg = mlops_registry.get_scenario_config("cc_api")
                    cells = (cfg.backbone_config or {}).get("cells") or []
                    self.assertEqual(len(cells), 1)
                    self.assertTrue(str(cells[0].get("path") or "").startswith("mlops/algos/"))
            finally:
                mlops_registry.REPO_ROOT = old_repo
                mlops_registry.MLOPS_ROOT = old_mlops
                mlops_registry.DATABASE_ROOT = old_db
                mlops_registry.REGISTRY_PATH = old_registry


class CustomCodeStatusTests(unittest.TestCase):
    def test_status_trained_when_metrics_in_latest_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo_root = tmp
            mlops_root = repo_root / "mlops"
            (mlops_root / "scenarios").mkdir(parents=True)
            (mlops_root / "registry.json").write_text('{"version": 1, "scenarios": []}', encoding="utf-8")

            old_repo = mlops_registry.REPO_ROOT
            old_mlops = mlops_registry.MLOPS_ROOT
            old_registry = mlops_registry.REGISTRY_PATH
            try:
                mlops_registry.REPO_ROOT = repo_root
                mlops_registry.MLOPS_ROOT = mlops_root
                mlops_registry.REGISTRY_PATH = mlops_root / "registry.json"
                mlops_registry.create_scenario_profile(
                    name="cc_stat",
                    display_name="S",
                    description="",
                    dataset="",
                    backbone_type="custom_code",
                    backbone_config={"cells": []},
                )
                run_dir = mlops_root / "models" / "cc_stat" / "v1"
                run_dir.mkdir(parents=True)
                (run_dir / "metrics.json").write_text(
                    json.dumps({"trained_at": "2026-01-01T00:00:00Z"}),
                    encoding="utf-8",
                )
                st = mlops_registry.get_scenario_status("cc_stat")
                self.assertEqual(st.get("status"), "trained")
            finally:
                mlops_registry.REPO_ROOT = old_repo
                mlops_registry.MLOPS_ROOT = old_mlops
                mlops_registry.REGISTRY_PATH = old_registry


try:
    from PyQt6.QtWidgets import QApplication
except Exception:  # pragma: no cover
    QApplication = None  # type: ignore[assignment,misc]


@unittest.skipIf(QApplication is None, "PyQt6 is not available")
class CustomCodeQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_new_scenario_custom_code_backbone_visible(self) -> None:
        from insight_local.cvops.ui.new_scenario_dialog import NewScenarioDialog

        def _get(path: str) -> dict:
            return {
                "datasets": [],
                "categories": {},
                "tabular_datasets": [],
            }

        dlg = NewScenarioDialog(http_get=_get, http_post=lambda _p, _b: {"name": "x"}, models=[])
        try:
            dlg._set_backbone_type("custom_code")
            dlg._on_backbone_changed()
            self.assertEqual(dlg._current_backbone(), "custom_code")
            dlg.show()
            dlg._name.setText("test_cc_scenario")
            dlg._on_dataset_changed()
            self._app.processEvents()
            self.assertFalse(dlg._backbone_config_edit.isHidden())
            self.assertTrue(dlg._tabular_rows[0][1].isHidden())
            self.assertTrue(dlg._create_btn.isEnabled())
        finally:
            dlg.deleteLater()

    def test_catalog_custom_code_training_payload(self) -> None:
        from insight_local.cvops.ui.catalog_panel import CatalogPanel

        def _get(path: str) -> dict:
            if path.endswith("/custom_cells"):
                return {
                    "cells": [
                        {
                            "id": "z1",
                            "name": "Z",
                            "path": "mlops/custom_cells/cc_ui/draft/cell_z1.py",
                            "entry": "run",
                            "datasets": [],
                            "code": "def run(ctx, prev):\n    return None\n",
                        }
                    ],
                    "scenario_datasets": [{"name": "primary", "kind": "folder", "path": "database/d", "format": "yolo", "mode": "reference"}],
                }
            return {}

        panel = CatalogPanel(
            base_url="http://127.0.0.1:8787",
            http_get=_get,
            http_post=lambda _p, _b: {},
            http_delete=lambda _p: {},
            http_get_text=lambda _p: "",
            http_put=lambda _p, _b: {},
        )
        try:
            panel.apply_scenarios(
                [
                    {
                        "name": "cc_ui",
                        "display_name": "CC",
                        "description": "",
                        "status": "empty",
                        "backbone_type": "custom_code",
                        "dataset": "",
                        "dataset_count": 0,
                        "backbone_config": {},
                        "training_guard": {},
                    }
                ]
            )
            panel._select_by_name("cc_ui")
            payload = panel._training_payload("custom_code")
            self.assertIsNotNone(payload)
            ov = payload.get("backbone_config_override") if isinstance(payload, dict) else None
            self.assertIsInstance(ov, dict)
            self.assertEqual(len(ov.get("cells") or []), 1)
            self.assertEqual(ov["cells"][0]["path"], "mlops/custom_cells/cc_ui/draft/cell_z1.py")
            self.assertEqual(len(ov.get("datasets") or []), 1)
        finally:
            panel.deleteLater()
