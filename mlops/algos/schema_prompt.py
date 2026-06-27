"""schema_prompt.py — structured configuration prompts for CSV schema mismatches.

Cells can raise a ValueError whose message contains a JSON payload prefixed with
`__CVOPS_PROMPT__:`. The Insight UI can detect this and offer an interactive fix
(e.g., choosing `label_col`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


PROMPT_PREFIX = "__CVOPS_PROMPT__:"


def _norm(s: str) -> str:
    return str(s or "").strip().lower()


def suggest_label_columns(columns: Iterable[str]) -> list[str]:
    cols = [str(c) for c in columns if str(c)]
    norm_to_orig: dict[str, str] = {}
    for c in cols:
        n = _norm(c)
        if n and n not in norm_to_orig:
            norm_to_orig[n] = c

    preferred = [
        "label",
        "labels",
        "target",
        "y",
        "class",
        "category",
        "price",
        "outcome",
        "response",
    ]
    out: list[str] = []
    for key in preferred:
        if key in norm_to_orig:
            out.append(norm_to_orig[key])
    # Heuristic fallback: last column is often the target.
    if cols:
        out.append(cols[-1])
    # De-dup while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for c in out:
        if c in seen:
            continue
        seen.add(c)
        deduped.append(c)
    return deduped[:10]


def raise_label_col_missing(
    *,
    dataset_csv: Path,
    attempted_label_col: str,
    columns: list[str],
) -> None:
    payload: dict[str, Any] = {
        "kind": "label_col_missing",
        "attempted_label_col": str(attempted_label_col or ""),
        "dataset_csv": str(dataset_csv),
        "columns": [str(c) for c in columns],
        "suggested": suggest_label_columns(columns),
    }
    raise ValueError(PROMPT_PREFIX + json.dumps(payload, ensure_ascii=True, separators=(",", ":")))

