"""Notes vault AI provider keys and static model ids for the chat catalog.

Cloud provider API keys are stored in the operating system keyring (macOS
Keychain, Windows Credential Locker, Linux Secret Service) via the ``keyring``
package -- never in plaintext on disk. Everything else (assistant name, voice
profile, system prompt, local model paths) lives in ``ai_settings.json`` under
the cvops state dir. If no OS keyring backend is available (e.g. a headless
Linux box with no Secret Service), the keys fall back to ``ai_settings.json``
so the app still works; ``keyring_available()`` reports which mode is active.
Legacy installs that wrote plaintext keys into the JSON are migrated into the
keyring automatically on first load and stripped from the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..paths import CVOPS_STATE_DIR

_AI_SETTINGS_FILE = "ai_settings.json"

KEY_OPENAI = "openai_api_key"
KEY_ANTHROPIC = "anthropic_api_key"
KEY_GROK = "grok_api_key"
KEY_GEMINI = "gemini_api_key"
KEY_ASSISTANT_NAME = "assistant_name"
KEY_LOCAL_GGUF_MODELS = "local_gguf_models"
KEY_VOICE_PROFILE = "voice_profile"
KEY_SYSTEM_PROMPT = "system_prompt"
DEFAULT_ASSISTANT_NAME = "Tacitus"

# API-key fields are secrets: stored in the OS keyring, not ai_settings.json.
SECRET_KEYS: tuple[str, ...] = (KEY_OPENAI, KEY_ANTHROPIC, KEY_GROK, KEY_GEMINI)

# Service name (the "where") under which provider keys are filed in the keyring.
_KEYRING_SERVICE = "insight-cvops-ai"

try:  # keyring is an optional dependency; degrade to JSON if it is missing.
    import keyring as _keyring
except Exception:  # pragma: no cover - exercised only when keyring is absent
    _keyring = None


def keyring_available() -> bool:
    """True when a usable OS keyring backend is present for secret storage.

    The ``fail``/``null`` fallback backends keyring selects on a box with no real
    credential store are treated as unavailable so callers fall back to JSON.
    """
    if _keyring is None:
        return False
    try:
        backend = _keyring.get_keyring()
    except Exception:
        return False
    name = f"{type(backend).__module__}.{type(backend).__name__}".lower()
    return "fail" not in name and "null" not in name


def _keyring_get(name: str) -> str:
    if _keyring is None:
        return ""
    try:
        return str(_keyring.get_password(_KEYRING_SERVICE, name) or "")
    except Exception:
        return ""


def _keyring_set(name: str, value: str) -> bool:
    """Store ``value`` for ``name`` (or delete the entry when empty). Returns
    True on success; False means the keyring rejected the write."""
    if _keyring is None:
        return False
    try:
        if value:
            _keyring.set_password(_KEYRING_SERVICE, name, value)
        else:
            try:
                _keyring.delete_password(_KEYRING_SERVICE, name)
            except Exception:
                pass  # nothing stored for this provider yet -> nothing to clear
        return True
    except Exception:
        return False
# Upper bound on the global system prompt to keep ai_settings.json sane and avoid
# blowing past provider context limits with a single field.
SYSTEM_PROMPT_MAX_CHARS = 8000

# --------------------------------------------------------------------------- #
# Voice profile (the "voice maker"): a base system voice + a restrained ffmpeg
# effect chain, designed in AI settings and applied whenever the assistant reads
# a message aloud. Kept as plain dicts here (Qt-free) so the schema, presets, and
# clamping are unit-testable. Numeric fields are bounded by _VOICE_FIELD_BOUNDS.
# --------------------------------------------------------------------------- #

# field -> (min, max) for clamping on load/save. ``base_voice`` (str) and
# ``comms_bandpass`` (bool) are handled separately.
_VOICE_FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    # Core character
    "rate_wpm": (90.0, 320.0),       # say words-per-minute; lower = more measured
    "pitch_semitones": (-12.0, 12.0),  # ffmpeg pitch shift
    "warmth_db": (0.0, 12.0),        # low-shelf bump for body
    "high_cut_hz": (0.0, 20000.0),   # gentle de-harsh lowpass; 0 = off
    "room": (0.0, 1.0),              # subtle echo/room mix; 0 = off
    # Fine tuning — naturalness / smoothing (all neutral at their low end)
    "low_cut_hz": (0.0, 400.0),      # highpass to clear rumble/plosives; 0 = off
    "presence_db": (-12.0, 12.0),    # ~3 kHz clarity: + = articulate, - = softer
    "air_db": (0.0, 12.0),           # high-shelf "breath" on top; 0 = off
    "sibilance": (0.0, 1.0),         # de-esser intensity (tames harsh S/SH); 0 = off
    "smoothing": (0.0, 1.0),         # loudness leveling for an even delivery; 0 = off
    "depth": (0.0, 1.0),             # subtle chorus richness so it's not thin; 0 = off
}


def default_voice_profile() -> dict[str, Any]:
    """The "Tacitus" voice: an en_IN base (Rishi) shaped warm, measured, grounded.

    Tuned by ear — a touch lower, generous low-mid warmth, softened highs, and a
    faint sense of room. ``base_voice`` falls back to the platform default if
    Rishi is not installed on a given machine.
    """
    return {
        "name": "Tacitus",
        "base_voice": "Rishi",     # en_IN; "" = platform default if missing
        "rate_wpm": 200.0,         # steady, unhurried but not slow
        "pitch_semitones": -2.2,   # slightly lower than natural
        "warmth_db": 11.8,         # strong low-mid body
        "high_cut_hz": 14465.0,    # gently soften the very top
        "room": 0.29,             # a clear-but-subtle sense of space
        "comms_bandpass": False,
        # Fine tuning: smooth and humanize without changing the character above.
        "low_cut_hz": 80.0,        # clear sub-rumble for a cleaner low end
        "presence_db": 1.5,        # a little clarity so words stay crisp
        "air_db": 2.0,             # gentle breath up top
        "sibilance": 0.45,         # tame the synthetic "s" harshness
        "smoothing": 0.4,          # even out the delivery
        "depth": 0.12,             # faint richness so it isn't thin
    }


# Selectable starting points in the designer. "Custom" is implied (any edit).
VOICE_PRESETS: dict[str, dict[str, Any]] = {
    "Tacitus": default_voice_profile(),
    "Minimal": {
        "name": "Minimal",
        "base_voice": "",
        "rate_wpm": 175.0,
        "pitch_semitones": 0.0,
        "warmth_db": 1.5,
        "high_cut_hz": 0.0,
        "room": 0.0,
        "comms_bandpass": False,
        "low_cut_hz": 60.0,
        "presence_db": 0.0,
        "air_db": 0.0,
        "sibilance": 0.2,
        "smoothing": 0.25,
        "depth": 0.0,
    },
    "Spartan-comms": {
        "name": "Spartan-comms",
        "base_voice": "",
        "rate_wpm": 160.0,
        "pitch_semitones": -4.0,
        "warmth_db": 4.0,
        "high_cut_hz": 3200.0,
        "room": 0.18,
        "comms_bandpass": True,
        "low_cut_hz": 200.0,
        "presence_db": 2.0,
        "air_db": 0.0,
        "sibilance": 0.3,
        "smoothing": 0.5,
        "depth": 0.0,
    },
}


def _normalized_voice_profile(value: Any) -> dict[str, Any]:
    """Merge ``value`` over the Tacitus default, clamping numbers and dropping
    unknown keys. Always returns a complete, valid profile dict."""
    out = default_voice_profile()
    if not isinstance(value, dict):
        return out
    name = str(value.get("name") or "").strip()
    if name:
        out["name"] = name[:60]
    base = str(value.get("base_voice") or "").strip()
    out["base_voice"] = base[:80]
    out["comms_bandpass"] = bool(value.get("comms_bandpass", out["comms_bandpass"]))
    for field, (lo, hi) in _VOICE_FIELD_BOUNDS.items():
        raw = value.get(field, out[field])
        try:
            num = float(raw)
        except (TypeError, ValueError):
            num = float(out[field])
        out[field] = max(lo, min(hi, num))
    return out

OPENAI_MODEL_IDS = (
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "o3-mini",
    "o1-mini",
)

ANTHROPIC_MODEL_IDS = (
    # Dateless Claude 4.x ids (see Anthropic "Model IDs" / streaming docs). Older 3.x snapshot ids
    # such as claude-3-opus-20240229 are removed from the API for new keys and return HTTP 404.
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    # Still available on many accounts as pinned snapshots:
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
)

GROK_MODEL_IDS = (
    "grok-2-latest",
    "grok-2-vision-latest",
)

GEMINI_MODEL_IDS = (
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
)

OLLAMA_DEFAULT_MODELS = ("gemma3:4b", "gemma2:4b", "llama3.2")


def ai_settings_path() -> Path:
    return (Path(CVOPS_STATE_DIR) / "notes" / _AI_SETTINGS_FILE).resolve()


def default_ai_settings() -> dict[str, Any]:
    return {
        KEY_OPENAI: "",
        KEY_ANTHROPIC: "",
        KEY_GROK: "",
        KEY_GEMINI: "",
        KEY_ASSISTANT_NAME: DEFAULT_ASSISTANT_NAME,
        KEY_LOCAL_GGUF_MODELS: [],
        KEY_VOICE_PROFILE: default_voice_profile(),
        KEY_SYSTEM_PROMPT: "",
    }


def _normalized_system_prompt(value: Any) -> str:
    return str(value or "").strip()[:SYSTEM_PROMPT_MAX_CHARS]


def _normalized_gguf_paths(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in value:
        path = str(item or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def load_ai_settings() -> dict[str, Any]:
    out = default_ai_settings()
    path = ai_settings_path()
    raw: dict[str, Any] = {}
    if path.is_file():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                raw = parsed
        except Exception:
            raw = {}

    # Non-secret fields come straight from the JSON file.
    for k in out:
        if k in SECRET_KEYS:
            continue
        v = raw.get(k)
        if k == KEY_LOCAL_GGUF_MODELS:
            out[k] = _normalized_gguf_paths(v)
        elif k == KEY_VOICE_PROFILE:
            out[k] = _normalized_voice_profile(v)
        elif k == KEY_SYSTEM_PROMPT:
            out[k] = _normalized_system_prompt(v)
        elif isinstance(v, str):
            out[k] = v.strip()

    # Secrets come from the OS keyring when available; otherwise from JSON.
    use_keyring = keyring_available()
    migrated = False
    for k in SECRET_KEYS:
        legacy = str(raw.get(k) or "").strip()  # plaintext from an older install
        if use_keyring:
            value = _keyring_get(k)
            if not value and legacy and _keyring_set(k, legacy):
                value = legacy  # migrate legacy plaintext key into the keyring
                migrated = True
            out[k] = value
        else:
            out[k] = legacy

    # Rewrite the file once to drop any plaintext secrets we just migrated.
    if migrated:
        save_ai_settings(out)
    return out


def assistant_display_name(settings: dict[str, Any] | None = None) -> str:
    source = settings if isinstance(settings, dict) else load_ai_settings()
    name = str(source.get(KEY_ASSISTANT_NAME) or "").strip()
    return name or DEFAULT_ASSISTANT_NAME


def local_gguf_models(settings: dict[str, Any] | None = None) -> list[str]:
    source = settings if isinstance(settings, dict) else load_ai_settings()
    return _normalized_gguf_paths(source.get(KEY_LOCAL_GGUF_MODELS))


def system_prompt(settings: dict[str, Any] | None = None) -> str:
    """The global system prompt applied to every model/provider (may be empty)."""
    source = settings if isinstance(settings, dict) else load_ai_settings()
    return _normalized_system_prompt(source.get(KEY_SYSTEM_PROMPT))


def voice_profile(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """The active voice profile (always a complete, clamped dict)."""
    source = settings if isinstance(settings, dict) else load_ai_settings()
    return _normalized_voice_profile(source.get(KEY_VOICE_PROFILE))


def save_ai_settings(settings: dict[str, Any]) -> None:
    path = ai_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    base = default_ai_settings()
    use_keyring = keyring_available()
    for k in base:
        v = settings.get(k)
        if k == KEY_LOCAL_GGUF_MODELS:
            base[k] = _normalized_gguf_paths(v)
        elif k == KEY_VOICE_PROFILE:
            base[k] = _normalized_voice_profile(v)
        elif k == KEY_SYSTEM_PROMPT:
            base[k] = _normalized_system_prompt(v)
        elif k in SECRET_KEYS:
            secret = str(v).strip() if v is not None else ""
            if use_keyring and _keyring_set(k, secret):
                base[k] = ""  # secret lives in the keyring, never in the file
            else:
                base[k] = secret  # no keyring backend -> fall back to JSON
        else:
            base[k] = str(v).strip() if v is not None else ""
    path.write_text(json.dumps(base, indent=2), encoding="utf-8")


def model_catalog_entries(
    settings: dict[str, Any],
    *,
    ollama_installed: list[str] | None = None,
    local_gguf_installed: list[str] | None = None,
) -> list[tuple[str, str]]:
    """(display label, route key ``provider:model_id``) for the model combobox.

    ``ollama_installed`` should be tags returned by ``discover_ollama_model_tags`` (``ollama list``
    plus HTTP ``/api/tags``); they are listed first, then defaults for any gap.
    """
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    ordered: list[str] = []
    for m in ollama_installed or ():
        tag = str(m).strip()
        if tag and tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    for m in OLLAMA_DEFAULT_MODELS:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    for m in ordered:
        rows.append((m, f"ollama:{m}"))
    gguf_seen: set[str] = set()
    gguf_paths = local_gguf_installed if local_gguf_installed is not None else local_gguf_models(settings)
    for raw_path in gguf_paths:
        path = str(raw_path or "").strip()
        if not path or path in gguf_seen:
            continue
        gguf_seen.add(path)
        rows.append((f"{Path(path).name} (GGUF)", f"ollama:{path}"))
    if settings.get(KEY_OPENAI):
        for m in OPENAI_MODEL_IDS:
            rows.append((f"{m} (OpenAI)", f"openai:{m}"))
    if settings.get(KEY_ANTHROPIC):
        for m in ANTHROPIC_MODEL_IDS:
            rows.append((f"{m} (Anthropic)", f"anthropic:{m}"))
    if settings.get(KEY_GROK):
        for m in GROK_MODEL_IDS:
            rows.append((f"{m} (Grok)", f"grok:{m}"))
    if settings.get(KEY_GEMINI):
        for m in GEMINI_MODEL_IDS:
            rows.append((f"{m} (Gemini)", f"gemini:{m}"))
    return rows


def parse_route_key(route: str) -> tuple[str, str]:
    """Split ``provider:rest``; unknown shapes become Ollama with full string as model."""
    route = str(route or "").strip()
    if ":" not in route:
        return "ollama", route
    prov, _, rest = route.partition(":")
    prov = prov.strip().lower()
    rest = rest.strip()
    if prov in ("openai", "anthropic", "gemini", "grok", "ollama") and rest:
        return prov, rest
    return "ollama", route
