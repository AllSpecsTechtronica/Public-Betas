from __future__ import annotations

import re
from typing import Iterable

_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_ALIASES = {
    "people": "person",
    "persons": "person",
    "humans": "human",
    "cats": "cat",
    "dogs": "dog",
    "vehicles": "vehicle",
}
_QUICK_FILTER_TERMS = {
    "people": ("human", "person"),
    "animals": ("animal",),
    "tech": ("tech",),
    "objects": ("inorganic", "plant", "object"),
}


def _normalize_token(token: str) -> str:
    cleaned = _TOKEN_RE.sub(" ", str(token).strip().lower()).strip()
    if not cleaned:
        return ""
    cleaned = _ALIASES.get(cleaned, cleaned)
    if cleaned.endswith("s") and len(cleaned) > 3:
        singular = cleaned[:-1]
        cleaned = _ALIASES.get(singular, singular)
    return cleaned


def normalized_query_tokens(query: str) -> list[str]:
    seen: list[str] = []
    for raw in _TOKEN_RE.sub(" ", str(query or "").lower()).split():
        token = _normalize_token(raw)
        if token and token not in seen:
            seen.append(token)
    return seen


def matches_detection_filter(query: str, *parts: object) -> bool:
    tokens = normalized_query_tokens(query)
    if not tokens:
        return True
    haystack = " ".join(
        _normalize_token(str(part))
        for part in parts
        if str(part or "").strip()
    )
    if not haystack:
        return False
    return any(token in haystack for token in tokens)


def matches_quick_filters(active_filters: set[str] | tuple[str, ...] | list[str], *parts: object) -> bool:
    enabled = {str(item).strip().lower() for item in active_filters if str(item).strip()}
    if not enabled:
        return True
    haystack = " ".join(
        _normalize_token(str(part))
        for part in parts
        if str(part or "").strip()
    )
    if not haystack:
        return False
    for key in enabled:
        for term in _QUICK_FILTER_TERMS.get(key, (key,)):
            if term in haystack:
                return True
    return False


def matches_detection_view(query: str, active_filters: set[str] | tuple[str, ...] | list[str], *parts: object) -> bool:
    return matches_detection_filter(query, *parts) and matches_quick_filters(active_filters, *parts)


def filter_items(query: str, items: Iterable[object], extractor) -> list[object]:
    return [item for item in items if matches_detection_filter(query, *extractor(item))]
