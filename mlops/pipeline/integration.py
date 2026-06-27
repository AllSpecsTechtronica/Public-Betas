from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EVENTS_PATH = REPO_ROOT / "mlops" / "integration" / "events.jsonl"


def append_integration_event(event: dict[str, Any]) -> Path:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=True) + "\n")
    return EVENTS_PATH
