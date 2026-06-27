from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .registry import MLOPS_ROOT


MODEL_REGISTRY_PATH = MLOPS_ROOT / "model_registry.json"
# Champion-challenger alias ladder: a passing gate stages a challenger
# (``staging``); a human promotes the champion (``prod``). ``candidate`` is the
# most recently registered run.
_ALIASES = {"candidate", "staging", "prod"}
_STATUSES = {"active", "deprecated", "archived", "partial", "canceled", "interrupted", "error"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_registry() -> dict[str, Any]:
    return {"version": 1, "models": {}}


def _load_registry() -> dict[str, Any]:
    if not MODEL_REGISTRY_PATH.exists():
        return _empty_registry()
    try:
        payload = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _empty_registry()
    if not isinstance(payload, dict):
        return _empty_registry()
    if int(payload.get("version") or 0) != 1:
        return _empty_registry()
    if not isinstance(payload.get("models"), dict):
        payload["models"] = {}
    return payload


def _save_registry(payload: dict[str, Any]) -> None:
    MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_REGISTRY_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )


def _ensure_scenario(payload: dict[str, Any], scenario: str) -> dict[str, Any]:
    models = payload.setdefault("models", {})
    node = models.setdefault(scenario, {})
    if not isinstance(node.get("versions"), list):
        node["versions"] = []
    if not isinstance(node.get("aliases"), dict):
        node["aliases"] = {}
    if not isinstance(node.get("alias_history"), dict):
        node["alias_history"] = {}
    if not isinstance(node.get("events"), list):
        node["events"] = []
    return node


def _version_id(scenario: str, run_version: str) -> str:
    return f"{scenario}:{run_version}"


def _record_alias_pointer(node: dict[str, Any], alias: str, version_id: str) -> None:
    """Point ``alias`` at ``version_id`` and append it to the alias's ordered
    pointer history (skipping consecutive duplicates) so reverts have a trail."""
    node.setdefault("aliases", {})[alias] = version_id
    history = node.setdefault("alias_history", {})
    seq = history.setdefault(alias, [])
    if not seq or str(seq[-1]) != str(version_id):
        seq.append(version_id)


def _version_exists(node: dict[str, Any], version_id: str) -> bool:
    wanted = str(version_id or "")
    return any(
        str(v.get("version_id") or "") == wanted
        for v in node.get("versions") or []
        if isinstance(v, dict)
    )


def register_model_version(
    *,
    scenario: str,
    run_version: str,
    artifacts: dict[str, Any],
    lineage: dict[str, Any],
    metrics: dict[str, Any],
    initial_status: str = "active",
    set_candidate: bool = True,
    ci_cd: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = str(initial_status or "active").strip().lower()
    if status not in _STATUSES:
        status = "active"
    run_version = str(run_version or "").strip()
    if not run_version:
        raise ValueError("run_version is required")
    payload = _load_registry()
    node = _ensure_scenario(payload, scenario)
    versions = node["versions"]
    vid = _version_id(scenario, run_version)
    existing = next((v for v in versions if str(v.get("version_id") or "") == vid), None)
    if existing is None:
        entry = {
            "version_id": vid,
            "scenario": scenario,
            "run_version": run_version,
            "created_at": _utc_now(),
            "status": status,
            "artifacts": dict(artifacts or {}),
            "lineage": dict(lineage or {}),
            "metrics": dict(metrics or {}),
            "ci_cd": dict(ci_cd or {}),
            "lifecycle": [],
        }
        versions.append(entry)
    else:
        # Keep version immutable; only merge additive metadata if missing.
        entry = existing
        if not isinstance(entry.get("artifacts"), dict):
            entry["artifacts"] = {}
        if not isinstance(entry.get("lineage"), dict):
            entry["lineage"] = {}
        if not isinstance(entry.get("metrics"), dict):
            entry["metrics"] = {}
        if not isinstance(entry.get("ci_cd"), dict):
            entry["ci_cd"] = {}
        for key, value in dict(artifacts or {}).items():
            entry["artifacts"].setdefault(key, value)
        for key, value in dict(lineage or {}).items():
            entry["lineage"].setdefault(key, value)
        for key, value in dict(metrics or {}).items():
            entry["metrics"].setdefault(key, value)
        for key, value in dict(ci_cd or {}).items():
            entry["ci_cd"][key] = value

    if set_candidate:
        _record_alias_pointer(node, "candidate", vid)
    node["events"].append(
        {"at": _utc_now(), "event": "register", "version_id": vid, "status": status}
    )
    _save_registry(payload)
    return dict(entry)


def get_model_version(scenario: str, version_id: str) -> dict[str, Any] | None:
    payload = _load_registry()
    node = _ensure_scenario(payload, scenario)
    wanted = str(version_id or "").strip()
    if not wanted:
        return None
    for entry in node.get("versions") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("version_id") or "") == wanted:
            return dict(entry)
    return None


def list_model_versions(scenario: str) -> list[dict[str, Any]]:
    payload = _load_registry()
    node = _ensure_scenario(payload, scenario)
    versions = [dict(v) for v in node.get("versions") or [] if isinstance(v, dict)]
    versions.sort(key=lambda v: str(v.get("created_at") or ""), reverse=True)
    return versions


def resolve_alias(scenario: str, alias: str) -> dict[str, Any] | None:
    wanted = str(alias or "").strip().lower()
    if wanted not in _ALIASES:
        return None
    payload = _load_registry()
    node = _ensure_scenario(payload, scenario)
    vid = str((node.get("aliases") or {}).get(wanted) or "")
    if not vid:
        return None
    for entry in node.get("versions") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("version_id") or "") == vid:
            return dict(entry)
    return None


def set_alias(scenario: str, alias: str, version_id: str) -> dict[str, Any]:
    target_alias = str(alias or "").strip().lower()
    if target_alias not in _ALIASES:
        raise ValueError(f"alias must be one of: {sorted(_ALIASES)}")
    payload = _load_registry()
    node = _ensure_scenario(payload, scenario)
    versions = [v for v in node.get("versions") or [] if isinstance(v, dict)]
    if not any(str(v.get("version_id") or "") == version_id for v in versions):
        raise ValueError(f"version_id not found for scenario '{scenario}': {version_id}")
    _record_alias_pointer(node, target_alias, version_id)
    node["events"].append(
        {"at": _utc_now(), "event": "alias_set", "alias": target_alias, "version_id": version_id}
    )
    _save_registry(payload)
    return {"scenario": scenario, "alias": target_alias, "version_id": version_id}


def update_version_ci_cd(
    scenario: str,
    version_id: str,
    ci_cd: dict[str, Any],
    *,
    lifecycle_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _load_registry()
    node = _ensure_scenario(payload, scenario)
    wanted = str(version_id or "").strip()
    if not wanted:
        raise ValueError("version_id is required")
    for entry in node.get("versions") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("version_id") or "") != wanted:
            continue
        current = entry.get("ci_cd")
        if not isinstance(current, dict):
            current = {}
        current.update(dict(ci_cd or {}))
        entry["ci_cd"] = current
        if lifecycle_event:
            lifecycle = entry.setdefault("lifecycle", [])
            if isinstance(lifecycle, list):
                lifecycle.append(dict(lifecycle_event))
        node["events"].append(
            {
                "at": _utc_now(),
                "event": "ci_cd_update",
                "version_id": wanted,
                "gate_status": str(current.get("gate_status") or ""),
            }
        )
        _save_registry(payload)
        return dict(entry)
    raise ValueError(f"version_id not found for scenario '{scenario}': {version_id}")


def promote_version(
    scenario: str,
    version_id: str,
    *,
    alias: str = "prod",
    actor: str = "",
    reason: str = "",
    ci_cd: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_alias = str(alias or "prod").strip().lower()
    if target_alias not in _ALIASES:
        raise ValueError(f"alias must be one of: {sorted(_ALIASES)}")
    payload = _load_registry()
    node = _ensure_scenario(payload, scenario)
    wanted = str(version_id or "").strip()
    if not wanted:
        raise ValueError("version_id is required")
    for entry in node.get("versions") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("version_id") or "") != wanted:
            continue
        _record_alias_pointer(node, target_alias, wanted)
        current = entry.get("ci_cd")
        if not isinstance(current, dict):
            current = {}
        current.update(dict(ci_cd or {}))
        entry["ci_cd"] = current
        lifecycle = entry.setdefault("lifecycle", [])
        event = {
            "at": _utc_now(),
            "status": "promoted",
            "alias": target_alias,
            "actor": str(actor or ""),
            "reason": str(reason or ""),
        }
        if isinstance(lifecycle, list):
            lifecycle.append(event)
        node["events"].append(
            {
                "at": event["at"],
                "event": "promoted",
                "alias": target_alias,
                "version_id": wanted,
                "actor": str(actor or ""),
                "reason": str(reason or ""),
            }
        )
        _save_registry(payload)
        return dict(entry)
    raise ValueError(f"version_id not found for scenario '{scenario}': {version_id}")


def alias_history(scenario: str, alias: str) -> list[str]:
    """Ordered list of version_ids an alias has pointed to (oldest first)."""
    target = str(alias or "").strip().lower()
    if target not in _ALIASES:
        raise ValueError(f"alias must be one of: {sorted(_ALIASES)}")
    payload = _load_registry()
    node = _ensure_scenario(payload, scenario)
    seq = (node.get("alias_history") or {}).get(target) or []
    return [str(v) for v in seq if str(v)]


def revert_alias(
    scenario: str,
    alias: str,
    *,
    actor: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Repoint ``alias`` to the most recent prior version it pointed to.

    A first-class, one-call rollback. Safe no-op when there is no prior pointer
    (or no surviving prior version): the alias is left untouched and
    ``reverted`` is ``False``.
    """
    target = str(alias or "").strip().lower()
    if target not in _ALIASES:
        raise ValueError(f"alias must be one of: {sorted(_ALIASES)}")
    payload = _load_registry()
    node = _ensure_scenario(payload, scenario)
    seq = list((node.get("alias_history") or {}).get(target) or [])
    current = str((node.get("aliases") or {}).get(target) or "")

    # Walk back from the entry before the current pointer to the most recent
    # prior version that still exists in the registry.
    prior = ""
    for vid in reversed(seq[:-1]) if len(seq) >= 2 else []:
        if str(vid) != current and _version_exists(node, str(vid)):
            prior = str(vid)
            break

    if not prior:
        return {
            "scenario": scenario,
            "alias": target,
            "version_id": current,
            "from_version_id": current,
            "reverted": False,
        }

    _record_alias_pointer(node, target, prior)
    now = _utc_now()
    node["events"].append(
        {
            "at": now,
            "event": "reverted",
            "alias": target,
            "version_id": prior,
            "from_version_id": current,
            "actor": str(actor or ""),
            "reason": str(reason or ""),
        }
    )
    _save_registry(payload)
    return {
        "scenario": scenario,
        "alias": target,
        "version_id": prior,
        "from_version_id": current,
        "reverted": True,
    }


def version_id_for_run(scenario: str, run_version: str) -> str | None:
    vid = _version_id(scenario, str(run_version or "").strip())
    payload = _load_registry()
    node = _ensure_scenario(payload, scenario)
    for entry in node.get("versions") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("version_id") or "") == vid:
            return vid
    return None


def deprecate_version(scenario: str, version_id: str, *, reason: str = "") -> dict[str, Any]:
    payload = _load_registry()
    node = _ensure_scenario(payload, scenario)
    for entry in node.get("versions") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("version_id") or "") != version_id:
            continue
        entry["status"] = "deprecated"
        lifecycle = entry.setdefault("lifecycle", [])
        if isinstance(lifecycle, list):
            lifecycle.append({"at": _utc_now(), "status": "deprecated", "reason": str(reason or "")})
        node["events"].append(
            {
                "at": _utc_now(),
                "event": "deprecated",
                "version_id": version_id,
                "reason": str(reason or ""),
            }
        )
        _save_registry(payload)
        return dict(entry)
    raise ValueError(f"version_id not found for scenario '{scenario}': {version_id}")
