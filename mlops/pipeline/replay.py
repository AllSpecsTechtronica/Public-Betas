from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print replay command from a reproducibility manifest")
    parser.add_argument("--manifest", required=True, help="Path to repro_manifest.json")
    return parser.parse_args()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"unable to load manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    return payload


def main() -> int:
    args = _parse_args()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    manifest = _load_manifest(manifest_path)
    scenario = str(manifest.get("scenario") or "").strip()
    hyper = manifest.get("hyperparams") if isinstance(manifest.get("hyperparams"), dict) else {}
    if not scenario:
        raise SystemExit("manifest missing scenario")

    epochs = hyper.get("epochs")
    imgsz = hyper.get("imgsz")
    save_period = hyper.get("save_period")
    seed = hyper.get("seed")
    deterministic = hyper.get("deterministic")
    cmd = ["python", "-m", "mlops.pipeline.train", "--scenario", scenario]
    if isinstance(epochs, int):
        cmd.extend(["--epochs", str(epochs)])
    if isinstance(imgsz, int):
        cmd.extend(["--imgsz", str(imgsz)])
    if isinstance(save_period, int):
        cmd.extend(["--save-period", str(save_period)])
    # Preserve resume behavior from the original system.
    if bool(manifest.get("resumed_from")):
        cmd.append("--resume")
    else:
        cmd.append("--no-resume")
    if isinstance(seed, int):
        cmd.extend(["--seed", str(seed)])
    if deterministic is False:
        cmd.append("--non-deterministic")

    print("[replay] manifest:", str(manifest_path))
    print("[replay] command:")
    print(" ".join(shlex.quote(part) for part in cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

