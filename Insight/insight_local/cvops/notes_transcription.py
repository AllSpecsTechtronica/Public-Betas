from __future__ import annotations

import re
import json
import os
import shutil
import subprocess
import tempfile
import wave
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

TRANSCRIBABLE_AUDIO_NOTE_SUFFIXES = frozenset(
    {
        ".aac",
        ".aif",
        ".aiff",
        ".flac",
        ".m4a",
        ".mov",
        ".mp3",
        ".mp4",
        ".ogg",
        ".wav",
        ".webm",
    }
)
DEFAULT_TRANSCRIPTION_PROVIDER = "vosk"
DEFAULT_VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CVOPS_STATE_DIR = _REPO_ROOT / "state" / "insight_local" / "cvops"


def is_transcribable_audio_note(path: str | Path) -> bool:
    return Path(path).suffix.lower() in TRANSCRIBABLE_AUDIO_NOTE_SUFFIXES


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_filename_stem(value: str, fallback: str = "audio_note") -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "").strip()).strip("_")
    return (safe or fallback)[:80]


def transcript_filename_for_source(source_path: str | Path, *, created_at: Optional[str] = None) -> str:
    stamp = created_at or _utc_now_compact()
    stamp = re.sub(r"[^0-9A-Za-z_-]+", "_", stamp).strip("_")[:32]
    stem = _safe_filename_stem(Path(source_path).stem)
    return f"transcript_{stem}_{stamp}.md"


def _block_start_end(block: dict[str, Any]) -> tuple[float, float]:
    raw_region = block.get("raw_region") if isinstance(block.get("raw_region"), dict) else {}
    metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
    start = raw_region.get("start_sec", metadata.get("start_sec", 0.0))
    end = raw_region.get("end_sec", metadata.get("end_sec", start))
    try:
        start_f = float(start or 0.0)
    except Exception:
        start_f = 0.0
    try:
        end_f = float(end or start_f)
    except Exception:
        end_f = start_f
    return max(0.0, start_f), max(0.0, end_f)


def _fmt_seconds(value: float) -> str:
    total_ms = int(round(max(0.0, float(value)) * 1000.0))
    minutes, rem_ms = divmod(total_ms, 60_000)
    seconds, millis = divmod(rem_ms, 1000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def _source_display(path: Path, notes_root: Optional[Path]) -> str:
    if notes_root is None:
        return path.name
    try:
        return path.resolve().relative_to(notes_root.resolve()).as_posix()
    except Exception:
        return path.name


def transcript_text_from_blocks(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = str(block.get("text") or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def _looks_like_vosk_model(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "am").is_dir() and ((path / "conf").is_dir() or (path / "graph").is_dir())


def _candidate_vosk_model_paths() -> list[Path]:
    out: list[Path] = []
    for env_name in ("CVOPS_VOSK_MODEL", "VOSK_MODEL_PATH"):
        raw = str(os.environ.get(env_name) or "").strip()
        if raw:
            out.append(Path(raw).expanduser())

    roots = [
        _CVOPS_STATE_DIR / "models" / "vosk",
        _CVOPS_STATE_DIR / DEFAULT_VOSK_MODEL_NAME,
        _REPO_ROOT / "models" / "vosk",
        _REPO_ROOT / "models" / DEFAULT_VOSK_MODEL_NAME,
        _REPO_ROOT / "mlops" / "models" / "vosk",
        _REPO_ROOT / "mlops" / "models" / DEFAULT_VOSK_MODEL_NAME,
        _REPO_ROOT / "assets" / "vosk",
    ]
    out.extend(roots)
    for root in roots:
        if not root.is_dir():
            continue
        try:
            out.extend(child for child in sorted(root.iterdir()) if child.is_dir())
        except Exception:
            continue
    return out


def find_vosk_model_path() -> Optional[Path]:
    for candidate in _candidate_vosk_model_paths():
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        if _looks_like_vosk_model(resolved):
            return resolved
    return None


def vosk_setup_hint() -> str:
    return (
        "Install Vosk with `pip install vosk`, download a Vosk model, then place the extracted "
        f"`{DEFAULT_VOSK_MODEL_NAME}` folder under `{(_CVOPS_STATE_DIR / 'models' / 'vosk').as_posix()}` "
        "or set `CVOPS_VOSK_MODEL` to the extracted model folder."
    )


def _wav_is_vosk_ready(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getcomptype() == "NONE"
    except Exception:
        return False


@contextmanager
def _audio_as_vosk_wav(path: Path) -> Iterator[Path]:
    if path.suffix.lower() == ".wav" and _wav_is_vosk_ready(path):
        yield path
        return

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("decode_unavailable")
    with tempfile.TemporaryDirectory(prefix="cvops_vosk_") as tmp:
        out = Path(tmp) / "audio_16k_mono.wav"
        cmd = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not out.is_file():
            raise RuntimeError("decode_failed")
        yield out


def _avg_word_confidence(words: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for word in words:
        try:
            values.append(float(word.get("conf")))
        except Exception:
            continue
    if not values:
        return 0.68
    return max(0.0, min(1.0, sum(values) / len(values)))


def _vosk_result_block(result: dict[str, Any], *, index: int, source_file_id: str) -> Optional[dict[str, Any]]:
    text = str(result.get("text") or "").strip()
    if not text:
        return None
    words_raw = result.get("result") if isinstance(result.get("result"), list) else []
    words = [dict(item) for item in words_raw if isinstance(item, dict)]
    start_sec = 0.0
    end_sec = 0.0
    if words:
        try:
            start_sec = float(words[0].get("start") or 0.0)
        except Exception:
            start_sec = 0.0
        try:
            end_sec = float(words[-1].get("end") or start_sec)
        except Exception:
            end_sec = start_sec
    return {
        "block_id": f"block-vosk-{source_file_id}-{index}",
        "provider": "vosk",
        "capability": "ok",
        "block_kind": "audio_transcript",
        "source_file_id": source_file_id,
        "page_index": None,
        "segment_id": f"audio-{index}",
        "text": text,
        "confidence": _avg_word_confidence(words),
        "raw_region": {"kind": "audio_segment", "start_sec": max(0.0, start_sec), "end_sec": max(0.0, end_sec)},
        "metadata": {"words": words[:500], "engine": "vosk"},
    }


def _transcribe_vosk(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        from vosk import KaldiRecognizer, Model, SetLogLevel  # type: ignore[import-not-found]
    except Exception:
        return [], "dependency_unavailable"

    model_path = find_vosk_model_path()
    if model_path is None:
        return [], "model_unavailable"

    try:
        SetLogLevel(-1)
    except Exception:
        pass

    try:
        model = Model(str(model_path))
        blocks: list[dict[str, Any]] = []
        with _audio_as_vosk_wav(path) as wav_path:
            with wave.open(str(wav_path), "rb") as wf:
                rec = KaldiRecognizer(model, wf.getframerate())
                rec.SetWords(True)
                while True:
                    data = wf.readframes(4000)
                    if not data:
                        break
                    if rec.AcceptWaveform(data):
                        raw = json.loads(rec.Result())
                        block = _vosk_result_block(raw, index=len(blocks), source_file_id=path.name)
                        if block is not None:
                            blocks.append(block)
                raw = json.loads(rec.FinalResult())
                block = _vosk_result_block(raw, index=len(blocks), source_file_id=path.name)
                if block is not None:
                    blocks.append(block)
        return blocks, "ok"
    except RuntimeError as exc:
        token = str(exc)
        if token in {"decode_unavailable", "decode_failed"}:
            return [], token
        return [], "failed"
    except Exception:
        return [], "failed"


def _transcribe_whisper(path: Path, provider: str) -> tuple[list[dict[str, Any]], str]:
    from .archive_engine import _transcribe_audio

    return _transcribe_audio(path, provider)


def _transcribe_with_provider(path: Path, provider: str) -> tuple[list[dict[str, Any]], str]:
    provider_label = str(provider or DEFAULT_TRANSCRIPTION_PROVIDER).strip().lower()
    if provider_label in {"", "local", "offline", "vosk"}:
        return _transcribe_vosk(path)
    if provider_label in {"whisper", "whisper_tiny"}:
        return _transcribe_whisper(path, provider_label)
    return [], "capability_unavailable"


def format_transcript_markdown(payload: dict[str, Any], *, notes_root: Optional[str | Path] = None) -> str:
    source_path = Path(str(payload.get("source_path") or ""))
    blocks = payload.get("blocks") if isinstance(payload.get("blocks"), list) else []
    text = str(payload.get("text") or transcript_text_from_blocks(blocks)).strip()
    created_at = str(payload.get("created_at") or _utc_now_iso())
    provider = str(payload.get("provider") or "audio_asr")
    model = str(payload.get("model") or DEFAULT_TRANSCRIPTION_PROVIDER)
    capability = str(payload.get("capability") or "")
    notes_root_path = Path(notes_root) if notes_root is not None else None

    lines = [
        f"# Transcript: {source_path.name or 'audio note'}",
        "",
        f"- Source: `{_source_display(source_path, notes_root_path)}`",
        f"- Provider: {provider} / {model}",
        f"- Capability: {capability or 'ok'}",
        f"- Created: {created_at}",
        "",
    ]
    if text:
        lines.extend(["## Text", "", text, ""])

    segment_lines: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        segment_text = str(block.get("text") or "").strip()
        if not segment_text:
            continue
        start, end = _block_start_end(block)
        segment_lines.append(f"- [{_fmt_seconds(start)} - {_fmt_seconds(end)}] {segment_text}")
    if segment_lines:
        lines.extend(["## Segments", "", *segment_lines, ""])
    return "\n".join(lines).rstrip() + "\n"


def transcribe_audio_note(path: str | Path, *, provider: str = DEFAULT_TRANSCRIPTION_PROVIDER) -> dict[str, Any]:
    audio_path = Path(path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio note not found: {audio_path}")
    if not is_transcribable_audio_note(audio_path):
        raise ValueError(f"unsupported audio note type: {audio_path.suffix or audio_path.name}")

    provider_label = str(provider or DEFAULT_TRANSCRIPTION_PROVIDER).strip().lower()
    blocks, capability = _transcribe_with_provider(audio_path, provider_label)
    normalized_blocks = [dict(block) for block in blocks if isinstance(block, dict)]
    return {
        "source_path": str(audio_path),
        "source_name": audio_path.name,
        "provider": "audio_asr",
        "model": provider_label or DEFAULT_TRANSCRIPTION_PROVIDER,
        "capability": str(capability or ""),
        "blocks": normalized_blocks,
        "text": transcript_text_from_blocks(normalized_blocks),
        "created_at": _utc_now_iso(),
    }
