from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlops.pipeline.governance import evaluate_dataset_quality, validate_dataset_contract


class MlopsGovernanceTests(unittest.TestCase):
    def test_split_first_yolo_dataset_contract_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset = Path(td) / "SplitFirstYolo"
            image = dataset / "train" / "images" / "sample.jpg"
            label = dataset / "train" / "labels" / "sample.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"fake-image")
            label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

            contract = validate_dataset_contract(dataset, ["Animals"])
            self.assertEqual(contract["status"], "ok")
            self.assertEqual(contract["splits"], ["train"])

            quality = evaluate_dataset_quality(dataset, ["Animals"])
            self.assertEqual(quality["images"], 1)
            self.assertEqual(quality["label_files"], 1)
            self.assertEqual(quality["class_instance_counts"], {"Animals": 1})


if __name__ == "__main__":
    unittest.main()
