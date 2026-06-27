from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonStateStore:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.snapshot_path = self.state_dir / "snapshot.json"
        self.recovery_log_path = self.state_dir / "recovery_log.jsonl"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def verify_writable(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        probe_path = self.state_dir / ".probe"
        probe_path.write_text(str(time.time()), encoding="utf-8")
        probe_path.unlink(missing_ok=True)

    def load_snapshot(self) -> dict[str, Any]:
        if not self.snapshot_path.exists():
            return {}
        try:
            return json.loads(self.snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(snapshot)
        payload["saved_at"] = round(time.time(), 3)
        tmp_path = self.snapshot_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.snapshot_path)

    def append_recovery_event(self, event: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(event)
        payload.setdefault("ts", round(time.time(), 3))
        with self.recovery_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")

    def load_recovery_log(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.recovery_log_path.exists():
            return []
        events: list[dict[str, Any]] = []
        try:
            with self.recovery_log_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return []
        return events[-limit:]
