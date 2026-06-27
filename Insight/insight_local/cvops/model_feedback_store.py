"""Append-only store for Console run/model feedback.

Console feedback is intentionally separate from Range corrections:
it is tied to a training run or weights file, not to a specific video frame.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_STORE_ROOT = Path(__file__).resolve().parents[2] / "assets" / "model_feedback"

_PRIMARY_FIELDS = {
    "source",
    "weights_path",
    "run_dir",
    "result_path",
    "job_id",
    "scenario",
    "context",
}


def get_store_root() -> Path:
    """Return the model feedback store root, creating it on first call."""
    _STORE_ROOT.mkdir(parents=True, exist_ok=True)
    return _STORE_ROOT


def feedback_path() -> Path:
    return get_store_root() / "console_model_feedback.jsonl"


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=True)
        return value
    except Exception:
        return str(value)


@dataclass
class ConsoleModelFeedback:
    """A single human feedback note against a Console run/model artifact."""

    id: str
    created_at: float
    source: str
    weights_path: str
    run_dir: str = ""
    result_path: str = ""
    job_id: str = ""
    scenario: str = ""
    context: str = ""
    issue_type: str = ""
    severity: str = ""
    notes: str = ""
    recommendation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new(
        *,
        payload: dict[str, Any],
        issue_type: str,
        severity: str,
        notes: str,
        recommendation: str = "",
    ) -> "ConsoleModelFeedback":
        clean = {str(k): _json_safe(v) for k, v in dict(payload or {}).items()}
        metadata = {k: v for k, v in clean.items() if k not in _PRIMARY_FIELDS}
        return ConsoleModelFeedback(
            id=uuid.uuid4().hex[:16],
            created_at=time.time(),
            source=str(clean.get("source") or "console"),
            weights_path=str(clean.get("weights_path") or ""),
            run_dir=str(clean.get("run_dir") or ""),
            result_path=str(clean.get("result_path") or ""),
            job_id=str(clean.get("job_id") or ""),
            scenario=str(clean.get("scenario") or ""),
            context=str(clean.get("context") or ""),
            issue_type=str(issue_type or ""),
            severity=str(severity or ""),
            notes=str(notes or ""),
            recommendation=str(recommendation or ""),
            metadata=metadata,
        )


def append_feedback(feedback: ConsoleModelFeedback) -> Path:
    path = feedback_path()
    line = json.dumps(asdict(feedback), ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path


def load_feedback() -> list[ConsoleModelFeedback]:
    path = feedback_path()
    if not path.exists():
        return []
    out: list[ConsoleModelFeedback] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            out.append(ConsoleModelFeedback(**obj))
        except Exception:
            continue
    out.sort(key=lambda item: item.created_at)
    return out
