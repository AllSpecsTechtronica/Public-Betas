# Lazy OpenAI client import — avoids loading ai_integration at process import time
# (helps slow/timeout-prone volumes and keeps startup working if that module fails).

_openai_client = None
_import_error = None


def get_openai_client():
    """Return the shared openai_client singleton, or None if import failed."""
    global _openai_client, _import_error
    if _openai_client is not None:
        return _openai_client
    if _import_error is not None:
        return None
    try:
        from ai_integration.openai_client import openai_client as _client

        _openai_client = _client
        return _openai_client
    except Exception as e:
        _import_error = e
        from utils.debug import debug_print

        debug_print(f"[OPENAI_BRIDGE] Could not import openai_client: {e}")
        return None
