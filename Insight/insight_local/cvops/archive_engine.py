from __future__ import annotations

import hashlib
import json
import math
import re
import time
import zipfile
from collections import defaultdict
from copy import deepcopy
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from .archive_store import ArchiveStore, PROCESSABLE_IMAGE_SUFFIXES, classify_media_family


PHASE_SEQUENCE = (
    "archive_phase0",
    "archive_phase1",
    "archive_phase2",
    "archive_phase3",
    "archive_phase4",
    "archive_phase5",
)

_PHASE_GOALS = (
    {
        "phase": "archive_phase0",
        "index": 0,
        "label": "Phase 0 — Assembly",
        "summary": "Reconstruct physical objects from flat files using filename patterns, sequential pairing, and manual overrides.",
    },
    {
        "phase": "archive_phase1",
        "index": 1,
        "label": "Phase 1 — Content Classification",
        "summary": "Classify media type, era bucket, and content complexity to route later extraction branches.",
    },
    {
        "phase": "archive_phase2",
        "index": 2,
        "label": "Phase 2 — Text Extraction",
        "summary": "Preserve raw OCR/ASR or direct text extraction with provider provenance and source-region metadata.",
    },
    {
        "phase": "archive_phase3",
        "index": 3,
        "label": "Phase 3 — Structured Extraction",
        "summary": "Convert extracted text into temporal anchors, entity mentions, and semantic relationships.",
    },
    {
        "phase": "archive_phase4",
        "index": 4,
        "label": "Phase 4 — Visual Extraction",
        "summary": "Add visual embeddings, scene classification, and visual-estimate temporal anchors for sparse-text media.",
    },
    {
        "phase": "archive_phase5",
        "index": 5,
        "label": "Phase 5 — Cross-Reference Resolution",
        "summary": "Resolve entities and temporal anchors across the corpus and build semantic clusters incrementally.",
    },
)

_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_SEASONS = {
    "spring": ("03-01", "05-31"),
    "summer": ("06-01", "08-31"),
    "fall": ("09-01", "11-30"),
    "autumn": ("09-01", "11-30"),
    "winter": ("12-01", "02-28"),
}
_STOP_ENTITY_WORDS = {
    "the",
    "and",
    "for",
    "from",
    "side",
    "part",
    "page",
    "scan",
    "photo",
    "image",
    "untitled",
    "early",
    "meeting",
}
_PERSON_TITLES = {"mr", "mrs", "ms", "dr", "miss", "mister"}
_ERA_BUCKET_ORDER = (
    "pre-1900",
    "1900-1920",
    "1920-1940",
    "1940-1960",
    "1960-1980",
    "1980-present",
)
_OBJECT_TYPE_ORDER = (
    "photograph",
    "newspaper_page",
    "newspaper_article",
    "map_sheet",
    "document",
    "correspondence",
    "audio_recording",
    "unknown",
)
_COMPLEXITY_ORDER = ("single", "multi", "unknown")


def archive_backbone_version() -> str:
    return "archival_ingestion/v1"


def phase_label(phase: str) -> str:
    token = str(phase or "").replace("archive_", "").replace("_", " ").strip()
    return token.upper() or "PHASE"


def phase_goals() -> list[dict[str, Any]]:
    return [dict(item) for item in _PHASE_GOALS]


def _dedupe_assertions(assertions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in assertions:
        assertion_id = str(item.get("assertion_id") or "").strip()
        if not assertion_id:
            continue
        deduped[assertion_id] = dict(item)
    return list(deduped.values())


def _dedupe_assertion_edits(edits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in edits:
        edit_id = str(item.get("edit_id") or "").strip()
        if not edit_id:
            continue
        deduped[edit_id] = dict(item)
    return list(deduped.values())


def _title_assertion_for_object(obj: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(obj.get("metadata") or {})
    title_meta = dict(metadata.get("title_provenance") or {})
    files = list(obj.get("files") or [])
    first_file = files[0] if files else {}
    file_meta = dict(first_file.get("file") or {}) if isinstance(first_file, dict) else {}
    raw_title = str(
        title_meta.get("raw")
        or file_meta.get("basename")
        or obj.get("object_key")
        or obj.get("title")
        or ""
    )
    return {
        "assertion_id": f"assert-title-{str(obj.get('object_id') or '')}",
        "object_id": str(obj.get("object_id") or ""),
        "field": "title",
        "raw_extraction": raw_title,
        "current_value": str(obj.get("title") or raw_title),
        "current_confidence": 1.0,
        "extraction_model": str(obj.get("assembly_method") or "assembly"),
        "extraction_run_id": "",
        "source_file_id": str(first_file.get("file_id") or ""),
        "source_type": "assembly",
        "raw_region": {},
        "metadata": {"title_provenance": title_meta},
        "created_at": float(obj.get("created_at") or time.time()),
    }


def _phase0_assertions(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_title_assertion_for_object(obj) for obj in objects]


def _phase2_assertions(objects: list[dict[str, Any]], prior: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assertions = list(prior)
    for obj in objects:
        metadata = dict(obj.get("metadata") or {})
        for index, block in enumerate(metadata.get("text_blocks") or []):
            if not isinstance(block, dict):
                continue
            text = str(block.get("text") or "").strip()
            capability = str(block.get("capability") or "")
            if not text and capability == "ok":
                continue
            provider = str(block.get("provider") or "extractor")
            block_kind = str(block.get("block_kind") or "text_block")
            field = "transcript" if block_kind == "audio_transcript" else block_kind
            assertions.append(
                {
                    "assertion_id": f"assert-text-{uuid_from_text(str(obj.get('object_id') or '') + provider + str(index) + str(block.get('segment_id') or ''))}",
                    "object_id": str(obj.get("object_id") or ""),
                    "field": field,
                    "raw_extraction": text,
                    "current_value": text,
                    "current_confidence": float(block.get("confidence") or (0.84 if text else 0.0)),
                    "extraction_model": provider,
                    "extraction_run_id": "archive_phase2",
                    "source_file_id": str(block.get("source_file_id") or ""),
                    "source_type": provider,
                    "raw_region": dict(block.get("raw_region") or {}),
                    "metadata": {
                        "capability": capability,
                        "block_kind": block_kind,
                        "page_index": block.get("page_index"),
                        "segment_id": str(block.get("segment_id") or ""),
                        **dict(block.get("metadata") or {}),
                    },
                    "extraction_timestamp": time.time(),
                    "created_at": time.time(),
                }
            )
    return _dedupe_assertions(assertions)


def _phase3_assertions(
    anchors: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    prior: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    assertions = list(prior)
    for anchor in anchors:
        current_value = str(anchor.get("earliest") or "")
        latest = str(anchor.get("latest") or "")
        if latest and latest != current_value:
            current_value = f"{current_value} -> {latest}" if current_value else latest
        assertions.append(
            {
                "assertion_id": f"assert-anchor-{str(anchor.get('anchor_id') or '')}",
                "object_id": str(anchor.get("object_id") or ""),
                "field": f"temporal_{str(anchor.get('type') or 'anchor')}",
                "raw_extraction": str(anchor.get("raw_expression") or ""),
                "current_value": current_value,
                "current_confidence": float(anchor.get("confidence") or 0.0),
                "extraction_model": "archive_phase3",
                "extraction_run_id": "archive_phase3",
                "source_file_id": str((anchor.get("metadata") or {}).get("source_file_id") or ""),
                "source_type": str(anchor.get("source") or ""),
                "raw_region": dict((anchor.get("metadata") or {}).get("raw_region") or {}),
                "metadata": {"resolved": bool(anchor.get("resolved"))},
                "created_at": float(anchor.get("created_at") or time.time()),
            }
        )
    for mention in mentions:
        assertions.append(
            {
                "assertion_id": f"assert-mention-{str(mention.get('mention_id') or '')}",
                "object_id": str(mention.get("object_id") or ""),
                "field": "entity_mention",
                "raw_extraction": str(mention.get("mention_text") or mention.get("text_span") or ""),
                "current_value": str(mention.get("mention_text") or mention.get("text_span") or ""),
                "current_confidence": float(mention.get("mention_confidence") or 0.0),
                "extraction_model": "archive_phase3",
                "extraction_run_id": "archive_phase3",
                "source_file_id": str((mention.get("metadata") or {}).get("source_file_id") or ""),
                "source_type": "entity_extraction",
                "raw_region": dict((mention.get("metadata") or {}).get("raw_region") or {}),
                "metadata": {"entity_id": str(mention.get("entity_id") or "")},
                "created_at": float(mention.get("created_at") or time.time()),
            }
        )
    return _dedupe_assertions(assertions)


def _phase4_assertions(objects: list[dict[str, Any]], anchors: list[dict[str, Any]], prior: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assertions = list(prior)
    for obj in objects:
        visual = dict((obj.get("metadata") or {}).get("visual") or {})
        if not visual:
            continue
        assertions.append(
            {
                "assertion_id": f"assert-visual-{str(obj.get('object_id') or '')}",
                "object_id": str(obj.get("object_id") or ""),
                "field": "visual_scene",
                "raw_extraction": str(visual.get("scene_class") or ""),
                "current_value": str(visual.get("scene_class") or ""),
                "current_confidence": 0.42,
                "extraction_model": "archive_phase4",
                "extraction_run_id": "archive_phase4",
                "source_file_id": str(next((ref.get("file_id") for ref in (obj.get("files") or []) if isinstance(ref, dict)), "") or ""),
                "source_type": "visual_extraction",
                "raw_region": {},
                "metadata": {"mean_luma": visual.get("mean_luma")},
                "created_at": time.time(),
            }
        )
    for anchor in anchors:
        if str(anchor.get("type") or "") != "visual_estimate":
            continue
        assertions.append(
            {
                "assertion_id": f"assert-visual-anchor-{str(anchor.get('anchor_id') or '')}",
                "object_id": str(anchor.get("object_id") or ""),
                "field": "visual_temporal_estimate",
                "raw_extraction": str(anchor.get("raw_expression") or ""),
                "current_value": f"{str(anchor.get('earliest') or '')} -> {str(anchor.get('latest') or '')}",
                "current_confidence": float(anchor.get("confidence") or 0.0),
                "extraction_model": "archive_phase4",
                "extraction_run_id": "archive_phase4",
                "source_file_id": "",
                "source_type": "visual_estimation",
                "raw_region": {},
                "metadata": dict(anchor.get("metadata") or {}),
                "created_at": float(anchor.get("created_at") or time.time()),
            }
        )
    return _dedupe_assertions(assertions)


def _phase5_assertions(anchors: list[dict[str, Any]], prior: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assertions = list(prior)
    for anchor in anchors:
        if str(anchor.get("source") or "") not in {"manual", "cross_reference"}:
            continue
        assertions.append(
            {
                "assertion_id": f"assert-resolution-{str(anchor.get('anchor_id') or '')}",
                "object_id": str(anchor.get("object_id") or ""),
                "field": "temporal_resolution",
                "raw_extraction": str(anchor.get("raw_expression") or ""),
                "current_value": f"{str(anchor.get('earliest') or '')} -> {str(anchor.get('latest') or '')}",
                "current_confidence": float(anchor.get("confidence") or 0.0),
                "extraction_model": "archive_phase5",
                "extraction_run_id": "archive_phase5",
                "source_file_id": "",
                "source_type": str(anchor.get("source") or ""),
                "raw_region": dict((anchor.get("metadata") or {}).get("raw_region") or {}),
                "metadata": {"resolved": bool(anchor.get("resolved"))},
                "created_at": time.time(),
            }
        )
    return _dedupe_assertions(assertions)


def _emit_cell(
    callback: Optional[Callable[[dict[str, Any]], None]],
    *,
    index: int,
    phase: str,
    status: str,
    output: str,
    elapsed_ms: float = 0.0,
) -> None:
    if callback is None:
        return
    callback(
        {
            "cell_index": index,
            "cell_name": phase_label(phase),
            "cell_status": status,
            "output": str(output or ""),
            "elapsed_ms": float(elapsed_ms or 0.0),
        }
    )


def _slugify_title(text: str) -> str:
    raw = re.sub(r"[^A-Za-z0-9]+", " ", str(text or "")).strip()
    raw = re.sub(r"\s+", " ", raw)
    return raw[:160]


def _role_and_group_key(file_item: dict[str, Any]) -> tuple[str, str, int]:
    rel = str(file_item.get("relative_path") or "")
    stem = Path(rel).stem
    lowered = stem.lower().replace("-", " ").replace("_", " ")
    ordinal = 0
    role = "component"

    patterns: list[tuple[str, str]] = [
        (r"\bfront\b|\brecto\b", "front"),
        (r"\bback\b|\bverso\b", "back"),
        (r"\bside\s*a\b", "side_a"),
        (r"\bside\s*b\b", "side_b"),
        (r"\bpart\s*(\d+)\b|\bpt\s*(\d+)\b", "part"),
        (r"\bpage\s*(\d+)\b|\bp\s*(\d+)\b", "page"),
        (r"\bdetail\s*(\d+)\b", "detail"),
    ]
    cleaned = lowered
    for pattern, tag in patterns:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        role = tag
        nums = [group for group in match.groups() if group]
        if nums:
            try:
                ordinal = int(nums[0])
            except Exception:
                ordinal = 0
        cleaned = re.sub(pattern, " ", cleaned).strip()
        break

    cleaned = re.sub(r"\s+", " ", cleaned).strip() or lowered
    cleaned = cleaned.replace("  ", " ").strip()
    if role == "component":
        return role, cleaned, ordinal
    return role, cleaned, ordinal


def _stem_prefix_number(stem: str) -> tuple[str, Optional[int]]:
    match = re.match(r"^(.*?)(\d{1,5})$", stem)
    if not match:
        return stem, None
    return match.group(1), int(match.group(2))


def _image_complexity(path: Path) -> float:
    image = cv2.imread(str(path))
    if image is None:
        return 1.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges)) / float(max(1, edges.size))
    tonal_std = float(np.std(gray)) / 255.0
    return edge_density + tonal_std


def _year_to_bucket(year: int) -> str:
    if year < 1900:
        return "pre-1900"
    if year <= 1920:
        return "1900-1920"
    if year <= 1940:
        return "1920-1940"
    if year <= 1960:
        return "1940-1960"
    if year <= 1980:
        return "1960-1980"
    return "1980-present"


def _guess_object_type(group_files: list[dict[str, Any]]) -> str:
    suffixes = {str((item.get("file") or {}).get("extension") or item.get("extension") or "").lower() for item in group_files}
    title = " ".join(str((item.get("file") or {}).get("basename") or item.get("basename") or "") for item in group_files).lower()
    if any(s in {".wav", ".mp3", ".flac", ".m4a", ".aiff", ".aif", ".ogg", ".aup3"} for s in suffixes):
        return "audio_recording"
    if "map" in title or "plan" in title:
        return "map_sheet"
    if "letter" in title or "correspondence" in title:
        return "correspondence"
    if "newspaper" in title or "clipping" in title:
        return "newspaper_page"
    if any(s in {".pdf", ".txt", ".md", ".docx", ".rtf", ".doc"} for s in suffixes):
        return "document"
    roles = {str(item.get("role") or "") for item in group_files}
    if "page" in roles:
        return "document"
    if roles & {"front", "back"}:
        return "photograph"
    return "photograph" if any(s in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".psd"} for s in suffixes) else "unknown"


def _object_media_family(group_files: list[dict[str, Any]]) -> str:
    families = [str((item.get("file") or {}).get("media_family") or item.get("media_family") or "") for item in group_files]
    if "audio" in families:
        return "audio"
    if "document" in families:
        return "document"
    if "image" in families:
        return "image"
    return families[0] if families else "binary"


def _text_from_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _text_from_rtf(path: Path) -> str:
    raw = _text_from_txt(path)
    raw = re.sub(r"\\'[0-9a-fA-F]{2}", " ", raw)
    raw = re.sub(r"\\[A-Za-z]+\d* ?", " ", raw)
    raw = raw.replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", raw).strip()


def _text_from_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as handle:
            data = handle.read("word/document.xml").decode("utf-8", errors="replace")
    except Exception:
        return ""
    data = re.sub(r"<[^>]+>", " ", data)
    return re.sub(r"\s+", " ", data).strip()


def _phase2_provider_config(overrides: Optional[dict[str, Any]] = None) -> dict[str, str]:
    config = {
        "printed_ocr": "tesseract",
        "handwriting_ocr": "none",
        "audio_asr": "whisper_tiny",
    }
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            token = str(value or "").strip()
            if key in config and token:
                config[key] = token
    return config


def _phase2_block(
    *,
    provider: str,
    capability: str,
    block_kind: str,
    source_file_id: str,
    text: str = "",
    confidence: float = 0.0,
    page_index: Optional[int] = None,
    segment_id: str = "",
    raw_region: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload = {
        "block_id": f"block-{uuid_from_text(str(source_file_id) + str(provider) + str(block_kind) + str(page_index) + str(segment_id) + str(raw_region or {}) + str(text[:64]))}",
        "provider": str(provider or ""),
        "capability": str(capability or ""),
        "block_kind": str(block_kind or ""),
        "source_file_id": str(source_file_id or ""),
        "page_index": int(page_index) if page_index is not None else None,
        "segment_id": str(segment_id or ""),
        "text": str(text or ""),
        "confidence": float(confidence or 0.0),
        "raw_region": dict(raw_region or {}),
        "metadata": dict(metadata or {}),
    }
    return payload


def _text_from_pdf(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:
        return [], [], "capability_unavailable"
    try:
        doc = fitz.open(str(path))
    except Exception:
        return [], [], "failed"
    blocks: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    try:
        for page_index, page in enumerate(doc):
            text = str(page.get_text("text") or "").strip()
            segments.append(
                {
                    "segment_id": f"seg-page-{uuid_from_text(str(path) + str(page_index))}",
                    "segment_kind": "page",
                    "source_file_id": "",
                    "page_index": page_index,
                    "raw_region": {"kind": "page", "page_index": page_index},
                    "metadata": {},
                }
            )
            if text:
                blocks.append(
                    _phase2_block(
                        provider="pdf_text",
                        capability="ok",
                        block_kind="pdf_text",
                        source_file_id="",
                        page_index=page_index,
                        text=text,
                        confidence=0.98,
                        raw_region={"kind": "page", "page_index": page_index},
                    )
                )
    except Exception:
        doc.close()
        return [], [], "failed"
    doc.close()
    return blocks, segments, "ok"


def _transcribe_audio(path: Path, provider_name: str) -> tuple[list[dict[str, Any]], str]:
    provider_label = str(provider_name or "whisper_tiny").strip().lower()
    if provider_label in {"", "none"}:
        return [], "capability_unavailable"
    if provider_label not in {"whisper_tiny", "whisper"}:
        return [], "capability_unavailable"
    try:
        import whisper  # type: ignore[import-not-found]
    except Exception:
        return [], "capability_unavailable"
    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(str(path))
    except Exception:
        return [], "failed"
    output = result if isinstance(result, dict) else {}
    segments = output.get("segments") if isinstance(output.get("segments"), list) else []
    blocks: list[dict[str, Any]] = []
    file_id = str(path.name)
    if segments:
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text") or "").strip()
            start_sec = float(segment.get("start") or 0.0)
            end_sec = float(segment.get("end") or start_sec)
            confidence = 0.74
            avg_logprob = segment.get("avg_logprob")
            if isinstance(avg_logprob, (int, float)):
                confidence = max(0.0, min(1.0, 1.0 + float(avg_logprob) / 5.0))
            blocks.append(
                _phase2_block(
                    provider="audio_asr",
                    capability="ok",
                    block_kind="audio_transcript",
                    source_file_id=file_id,
                    segment_id=f"audio-{index}",
                    text=text,
                    confidence=confidence,
                    raw_region={"kind": "audio_segment", "start_sec": start_sec, "end_sec": end_sec},
                    metadata={"start_sec": start_sec, "end_sec": end_sec},
                )
            )
    else:
        text = str(output.get("text") or "").strip()
        blocks.append(
            _phase2_block(
                provider="audio_asr",
                capability="ok",
                block_kind="audio_transcript",
                source_file_id=file_id,
                segment_id="audio-0",
                text=text,
                confidence=0.66 if text else 0.0,
                raw_region={"kind": "audio_segment", "start_sec": 0.0, "end_sec": 0.0},
            )
        )
    return blocks, "ok"


def _ocr_blocks_from_image(
    image: Any,
    *,
    file_id: str,
    provider_name: str,
    block_kind: str,
    page_index: Optional[int] = None,
) -> tuple[list[dict[str, Any]], str]:
    provider_label = str(provider_name or "tesseract").strip().lower()
    if provider_label in {"", "none"}:
        return [], "capability_unavailable"
    if provider_label != "tesseract":
        return [], "capability_unavailable"
    try:
        import pytesseract  # type: ignore[import-not-found]
        from pytesseract import Output  # type: ignore[import-not-found]
    except Exception:
        return [], "capability_unavailable"
    try:
        if image is None:
            return [], "failed"
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = pytesseract.image_to_data(rgb, output_type=Output.DICT)
    except Exception:
        return [], "failed"
    line_groups: dict[tuple[int, int, int], dict[str, Any]] = {}
    total = int(len(data.get("text") or []))
    for index in range(total):
        token = str((data.get("text") or [""])[index] or "").strip()
        if not token:
            continue
        level_key = (
            int((data.get("block_num") or [0])[index] or 0),
            int((data.get("par_num") or [0])[index] or 0),
            int((data.get("line_num") or [0])[index] or 0),
        )
        conf_raw = str((data.get("conf") or ["0"])[index] or "0").strip()
        try:
            conf_value = float(conf_raw)
        except Exception:
            conf_value = 0.0
        left = int((data.get("left") or [0])[index] or 0)
        top = int((data.get("top") or [0])[index] or 0)
        width = int((data.get("width") or [0])[index] or 0)
        height = int((data.get("height") or [0])[index] or 0)
        entry = line_groups.setdefault(
            level_key,
            {
                "tokens": [],
                "conf": [],
                "left": left,
                "top": top,
                "right": left + width,
                "bottom": top + height,
            },
        )
        entry["tokens"].append(token)
        entry["conf"].append(conf_value)
        entry["left"] = min(int(entry["left"]), left)
        entry["top"] = min(int(entry["top"]), top)
        entry["right"] = max(int(entry["right"]), left + width)
        entry["bottom"] = max(int(entry["bottom"]), top + height)
    blocks: list[dict[str, Any]] = []
    for block_index, key in enumerate(sorted(line_groups)):
        entry = line_groups[key]
        text = " ".join(str(token) for token in entry["tokens"]).strip()
        width = max(0, int(entry["right"]) - int(entry["left"]))
        height = max(0, int(entry["bottom"]) - int(entry["top"]))
        confidence = 0.0
        if entry["conf"]:
            confidence = max(0.0, min(1.0, sum(max(0.0, value) for value in entry["conf"]) / max(1, len(entry["conf"])) / 100.0))
        blocks.append(
            _phase2_block(
                provider=provider_label,
                capability="ok",
                block_kind=block_kind,
                source_file_id=file_id,
                page_index=page_index,
                segment_id=f"ocr-{page_index if page_index is not None else 'img'}-{block_index}",
                text=text,
                confidence=confidence,
                raw_region={
                    "kind": "bbox",
                    "page_index": page_index,
                    "left": int(entry["left"]),
                    "top": int(entry["top"]),
                    "width": width,
                    "height": height,
                },
            )
        )
    if blocks:
        return blocks, "ok"
    return [
        _phase2_block(
            provider=provider_label,
            capability="ok",
            block_kind=block_kind,
            source_file_id=file_id,
            page_index=page_index,
            text="",
            confidence=0.0,
            raw_region={"kind": "page" if page_index is not None else "image", "page_index": page_index},
        )
    ], "ok"


def _extract_pdf_blocks(file_item: dict[str, Any], providers: dict[str, str]) -> list[dict[str, Any]]:
    path = Path(str(file_item.get("stored_path") or ""))
    file_id = str(file_item.get("file_id") or "")
    blocks, page_segments, capability = _text_from_pdf(path)
    for block in blocks:
        block["source_file_id"] = file_id
    if blocks or capability != "ok":
        if capability == "ok" or blocks:
            pass
        else:
            return [
                _phase2_block(
                    provider="pdf_text",
                    capability=capability,
                    block_kind="pdf_text",
                    source_file_id=file_id,
                    text="",
                    confidence=0.0,
                    raw_region={"kind": "file"},
                    metadata={"segmentation_segments": page_segments},
                )
            ]
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:
        fitz = None  # type: ignore[assignment]
    if blocks:
        for block in blocks:
            block.setdefault("metadata", {})
            block["metadata"]["segmentation_segments"] = page_segments
        return blocks
    if fitz is None:
        return [
            _phase2_block(
                provider=str(providers.get("printed_ocr") or "tesseract"),
                capability="capability_unavailable",
                block_kind="printed_ocr",
                source_file_id=file_id,
                text="",
                confidence=0.0,
                raw_region={"kind": "file"},
                metadata={"reason": "pdf_render_unavailable", "segmentation_segments": page_segments},
            )
        ]
    try:
        doc = fitz.open(str(path))
    except Exception:
        return [
            _phase2_block(
                provider=str(providers.get("printed_ocr") or "tesseract"),
                capability="failed",
                block_kind="printed_ocr",
                source_file_id=file_id,
                text="",
                confidence=0.0,
                raw_region={"kind": "file"},
                metadata={"segmentation_segments": page_segments},
            )
        ]
    ocr_blocks: list[dict[str, Any]] = []
    for page_index, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        buf = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        page_blocks, page_capability = _ocr_blocks_from_image(
            image,
            file_id=file_id,
            provider_name=str(providers.get("printed_ocr") or "tesseract"),
            block_kind="printed_ocr",
            page_index=page_index,
        )
        if page_blocks:
            for block in page_blocks:
                block.setdefault("metadata", {})
                block["metadata"]["page_index"] = page_index
            ocr_blocks.extend(page_blocks)
        elif page_capability != "ok":
            ocr_blocks.append(
                _phase2_block(
                    provider=str(providers.get("printed_ocr") or "tesseract"),
                    capability=page_capability,
                    block_kind="printed_ocr",
                    source_file_id=file_id,
                    page_index=page_index,
                    text="",
                    confidence=0.0,
                    raw_region={"kind": "page", "page_index": page_index},
                )
            )
    doc.close()
    for block in ocr_blocks:
        block.setdefault("metadata", {})
        block["metadata"]["segmentation_segments"] = page_segments
    return ocr_blocks


def _extract_image_blocks(file_item: dict[str, Any], providers: dict[str, str]) -> list[dict[str, Any]]:
    path = Path(str(file_item.get("stored_path") or ""))
    file_id = str(file_item.get("file_id") or "")
    image = cv2.imread(str(path))
    blocks, capability = _ocr_blocks_from_image(
        image,
        file_id=file_id,
        provider_name=str(providers.get("printed_ocr") or "tesseract"),
        block_kind="printed_ocr",
        page_index=None,
    )
    if blocks:
        return blocks
    return [
        _phase2_block(
            provider=str(providers.get("printed_ocr") or "tesseract"),
            capability=capability,
            block_kind="printed_ocr",
            source_file_id=file_id,
            text="",
            confidence=0.0,
            raw_region={"kind": "image"},
        )
    ]


def _extract_text_blocks(file_item: dict[str, Any], providers: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    provider_cfg = _phase2_provider_config(providers if isinstance(providers, dict) else None)
    path = Path(str(file_item.get("stored_path") or ""))
    file_id = str(file_item.get("file_id") or "")
    suffix = path.suffix.lower()
    blocks: list[dict[str, Any]] = []
    if suffix in {".txt", ".md"}:
        blocks.append(
            _phase2_block(
                provider="direct_text",
                capability="ok",
                block_kind="direct_text",
                source_file_id=file_id,
                text=_text_from_txt(path),
                confidence=1.0,
                raw_region={"kind": "file"},
            )
        )
    elif suffix == ".rtf":
        blocks.append(
            _phase2_block(
                provider="rtf_parser",
                capability="ok",
                block_kind="direct_text",
                source_file_id=file_id,
                text=_text_from_rtf(path),
                confidence=0.92,
                raw_region={"kind": "file"},
            )
        )
    elif suffix == ".docx":
        blocks.append(
            _phase2_block(
                provider="docx_parser",
                capability="ok",
                block_kind="direct_text",
                source_file_id=file_id,
                text=_text_from_docx(path),
                confidence=0.94,
                raw_region={"kind": "file"},
            )
        )
    elif suffix == ".pdf":
        blocks.extend(_extract_pdf_blocks(file_item, provider_cfg))
    elif suffix in PROCESSABLE_IMAGE_SUFFIXES:
        blocks.extend(_extract_image_blocks(file_item, provider_cfg))
    elif suffix in {".wav", ".mp3", ".flac", ".m4a", ".aiff", ".aif", ".ogg"}:
        audio_blocks, capability = _transcribe_audio(path, provider_cfg.get("audio_asr", "whisper_tiny"))
        if audio_blocks:
            for block in audio_blocks:
                block["source_file_id"] = file_id
            blocks.extend(audio_blocks)
        else:
            blocks.append(
                _phase2_block(
                    provider="audio_asr",
                    capability=capability,
                    block_kind="audio_transcript",
                    source_file_id=file_id,
                    text="",
                    confidence=0.0,
                    raw_region={"kind": "audio_segment", "start_sec": 0.0, "end_sec": 0.0},
                )
            )
    else:
        blocks.append(
            _phase2_block(
                provider="unsupported",
                capability="capability_unavailable",
                block_kind="direct_text",
                source_file_id=file_id,
                text="",
                confidence=0.0,
                raw_region={"kind": "file"},
            )
        )
    return blocks


def _month_date_anchor(month_name: str, day_str: str, year_str: str) -> tuple[str, str]:
    month = _MONTHS[month_name.lower()]
    day = max(1, min(31, int(day_str)))
    year = int(year_str)
    token = f"{year:04d}-{month:02d}-{day:02d}"
    return token, token


def _collect_temporal_anchors(text: str, object_id: str) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    clean = str(text or "")
    for match in re.finditer(
        r"\b(" + "|".join(_MONTHS.keys()) + r")\s+(\d{1,2}),\s*(18\d{2}|19\d{2}|20\d{2})\b",
        clean,
        flags=re.IGNORECASE,
    ):
        earliest, latest = _month_date_anchor(match.group(1), match.group(2), match.group(3))
        key = ("absolute", earliest, latest, match.group(0))
        if key in seen:
            continue
        seen.add(key)
        anchors.append(
            {
                "anchor_id": f"anchor-{uuid_from_text(object_id + match.group(0))}",
                "object_id": object_id,
                "type": "absolute",
                "earliest": earliest,
                "latest": latest,
                "confidence": 0.94,
                "source": "content",
                "is_publication_date": False,
                "raw_expression": match.group(0),
                "resolved": True,
                "resolution_requires": "",
                "metadata": {},
            }
        )
    for match in re.finditer(r"\b(spring|summer|fall|autumn|winter)\s+(18\d{2}|19\d{2}|20\d{2})\b", clean, flags=re.IGNORECASE):
        start_mmdd, end_mmdd = _SEASONS[match.group(1).lower()]
        year = int(match.group(2))
        earliest = f"{year:04d}-{start_mmdd}"
        latest = f"{year:04d}-{end_mmdd}" if match.group(1).lower() != "winter" else f"{year + 1:04d}-{end_mmdd}"
        key = ("partial", earliest, latest, match.group(0))
        if key in seen:
            continue
        seen.add(key)
        anchors.append(
            {
                "anchor_id": f"anchor-{uuid_from_text(object_id + match.group(0))}",
                "object_id": object_id,
                "type": "partial",
                "earliest": earliest,
                "latest": latest,
                "confidence": 0.72,
                "source": "content",
                "is_publication_date": False,
                "raw_expression": match.group(0),
                "resolved": True,
                "resolution_requires": "",
                "metadata": {},
            }
        )
    for match in re.finditer(r"\bat age\s+(\d{1,3})\b", clean, flags=re.IGNORECASE):
        age = int(match.group(1))
        key = ("age_based", "", "", match.group(0))
        if key in seen:
            continue
        seen.add(key)
        anchors.append(
            {
                "anchor_id": f"anchor-{uuid_from_text(object_id + match.group(0))}",
                "object_id": object_id,
                "type": "age_based",
                "earliest": "",
                "latest": "",
                "confidence": 0.48,
                "source": "content",
                "is_publication_date": False,
                "raw_expression": match.group(0),
                "resolved": False,
                "resolution_requires": "birth_date",
                "metadata": {"age": age},
            }
        )
    years = re.findall(r"\b(18\d{2}|19\d{2}|20\d{2})\b", clean)
    for year_text in years[:8]:
        year = int(year_text)
        earliest = f"{year:04d}-01-01"
        latest = f"{year:04d}-12-31"
        key = ("partial", earliest, latest, year_text)
        if key in seen:
            continue
        seen.add(key)
        anchors.append(
            {
                "anchor_id": f"anchor-{uuid_from_text(object_id + year_text + earliest)}",
                "object_id": object_id,
                "type": "partial",
                "earliest": earliest,
                "latest": latest,
                "confidence": 0.66,
                "source": "content",
                "is_publication_date": False,
                "raw_expression": year_text,
                "resolved": True,
                "resolution_requires": "",
                "metadata": {},
            }
        )
    return anchors


def uuid_from_text(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()[:12]


def _classify_entity_type(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("society", "club", "company", "dept", "department", "school")):
        return "organization"
    if any(token in lowered for token in ("street", "park", "valley", "alhambra", "san gabriel", "monterey park", "los angeles")):
        return "place"
    if any(token in lowered for token in ("meeting", "parade", "nursery", "day")):
        return "event"
    if any(token in lowered for token in ("building", "house", "hotel", "church")):
        return "structure"
    parts = [part for part in re.split(r"\s+", name) if part]
    if 1 < len(parts) <= 4:
        return "person"
    return "unknown"


def _extract_entity_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"\b[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,3}\b", text):
        token = match.group(0).strip()
        lowered = token.lower()
        words = lowered.split()
        if not token or any(word in _STOP_ENTITY_WORDS for word in words):
            continue
        candidates.append(token)
    return candidates


def _canonical_entity_key(name: str) -> str:
    lowered = re.sub(r"[^a-z0-9\s]", " ", name.lower())
    words = [word for word in lowered.split() if word and word not in _PERSON_TITLES]
    return " ".join(words)


def _similarity(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz  # type: ignore[import-not-found]
    except Exception:
        return SequenceMatcher(None, a, b).ratio()
    return float(fuzz.ratio(a, b)) / 100.0


def _assemble_phase0(store: ArchiveStore, dataset_version_id: str) -> list[dict[str, Any]]:
    files = store.list_files(dataset_version_id)
    usable = [item for item in files if str(item.get("ingest_status") or "") != "ignored_noise"]
    manual_overrides = store.list_assembly_overrides(dataset_version_id)

    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    singletons: list[dict[str, Any]] = []
    for file_item in usable:
        role, group_key, ordinal = _role_and_group_key(file_item)
        wrapped = {"file": file_item, "file_id": file_item["file_id"], "role": role, "ordinal": ordinal}
        if role == "component":
            singletons.append(wrapped)
        else:
            folder = str(Path(str(file_item.get("relative_path") or "")).parent.as_posix())
            by_key[(folder, group_key)].append(wrapped)

    objects: list[dict[str, Any]] = []
    consumed: set[str] = set()

    for (_folder, key), group_files in sorted(by_key.items(), key=lambda item: item[0][1]):
        if not group_files:
            continue
        group_files = sorted(group_files, key=lambda item: (int(item.get("ordinal") or 0), str((item.get("file") or {}).get("relative_path") or "")))
        object_id = f"obj-{uuid_from_text(dataset_version_id + key + ''.join(str(f['file_id']) for f in group_files))}"
        for wrapped in group_files:
            consumed.add(str(wrapped["file_id"]))
        objects.append(
            {
                "object_id": object_id,
                "object_key": key,
                "object_type": _guess_object_type(group_files),
                "title": _slugify_title(key or str((group_files[0].get("file") or {}).get("stem") or "")) or f"Object {len(objects) + 1}",
                "assembly_method": "filename_pattern",
                "assembly_confidence": 0.91,
                "status": "assembled",
                "earliest": "",
                "latest": "",
                "era_bucket": "",
                "media_family": _object_media_family(group_files),
                "content_complexity": "multi" if len(group_files) > 1 else "single",
                "unresolved_reason": "",
                "metadata": {
                    "source_paths": [str((wrapped.get("file") or {}).get("relative_path") or "") for wrapped in group_files],
                    "title_provenance": {"raw": key, "edited": False},
                },
                "files": [
                    {
                        "file_id": wrapped["file_id"],
                        "role": str(wrapped.get("role") or "component"),
                        "ordinal": int(wrapped.get("ordinal") or idx),
                        "confidence": 0.88,
                    }
                    for idx, wrapped in enumerate(group_files)
                ],
            }
        )

    loose_rasters: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in singletons:
        if str(item["file_id"]) in consumed:
            continue
        file_item = item["file"]
        path = Path(str(file_item.get("relative_path") or ""))
        prefix, number = _stem_prefix_number(path.stem)
        if number is not None and file_item.get("media_family") == "image":
            loose_rasters[(str(path.parent.as_posix()), prefix.lower())].append(item)

    sequential_pairs: set[str] = set()
    for (_folder, _prefix), items in loose_rasters.items():
        ordered = sorted(
            [item for item in items if _stem_prefix_number(Path(str((item.get("file") or {}).get("relative_path") or "")).stem)[1] is not None],
            key=lambda it: _stem_prefix_number(Path(str((it.get("file") or {}).get("relative_path") or "")).stem)[1] or 0,
        )
        idx = 0
        while idx + 1 < len(ordered):
            a = ordered[idx]
            b = ordered[idx + 1]
            a_num = _stem_prefix_number(Path(str((a.get("file") or {}).get("relative_path") or "")).stem)[1]
            b_num = _stem_prefix_number(Path(str((b.get("file") or {}).get("relative_path") or "")).stem)[1]
            if a_num is None or b_num is None or b_num != a_num + 1:
                idx += 1
                continue
            path_a = Path(str((a.get("file") or {}).get("stored_path") or ""))
            path_b = Path(str((b.get("file") or {}).get("stored_path") or ""))
            comp_a = _image_complexity(path_a)
            comp_b = _image_complexity(path_b)
            if comp_b <= comp_a * 0.78:
                key = Path(str((a.get("file") or {}).get("relative_path") or "")).stem
                object_id = f"obj-{uuid_from_text(dataset_version_id + key + str(a['file_id']) + str(b['file_id']))}"
                for wrapped in (a, b):
                    consumed.add(str(wrapped["file_id"]))
                    sequential_pairs.add(str(wrapped["file_id"]))
                objects.append(
                    {
                        "object_id": object_id,
                        "object_key": key.lower(),
                        "object_type": "photograph",
                        "title": _slugify_title(key),
                        "assembly_method": "sequential_pair",
                        "assembly_confidence": 0.74,
                        "status": "assembled",
                        "earliest": "",
                        "latest": "",
                        "era_bucket": "",
                        "media_family": "image",
                        "content_complexity": "single",
                        "unresolved_reason": "",
                        "metadata": {"sequential_pair": True, "title_provenance": {"raw": key, "edited": False}},
                        "files": [
                            {"file_id": a["file_id"], "role": "front", "ordinal": 1, "confidence": 0.74},
                            {"file_id": b["file_id"], "role": "back", "ordinal": 2, "confidence": 0.74},
                        ],
                    }
                )
                idx += 2
                continue
            idx += 1

    for item in usable:
        if str(item["file_id"]) in consumed:
            continue
        rel = str(item.get("relative_path") or "")
        group_files = [{"file": item, "file_id": item["file_id"], "role": "component", "ordinal": 1}]
        objects.append(
            {
                "object_id": f"obj-{uuid_from_text(dataset_version_id + rel)}",
                "object_key": Path(rel).stem.lower(),
                "object_type": _guess_object_type(group_files),
                "title": _slugify_title(Path(rel).stem),
                "assembly_method": "singleton",
                "assembly_confidence": 0.6 if bool(item.get("processable")) else 0.35,
                "status": "assembled" if bool(item.get("processable")) else "retained",
                "earliest": "",
                "latest": "",
                "era_bucket": "",
                "media_family": str(item.get("media_family") or classify_media_family(Path(rel))),
                "content_complexity": "single",
                "unresolved_reason": "",
                "metadata": {"source_paths": [rel], "title_provenance": {"raw": Path(rel).stem, "edited": False}},
                "files": [{"file_id": item["file_id"], "role": "component", "ordinal": 1, "confidence": 0.55}],
            }
        )

    if manual_overrides:
        objects = _apply_assembly_overrides(objects, manual_overrides)
    return sorted(objects, key=lambda item: str(item.get("title") or "").lower())


def _apply_assembly_overrides(objects: list[dict[str, Any]], overrides: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current = [deepcopy(item) for item in objects]
    for override in overrides:
        action = str(override.get("action") or "")
        payload = dict(override.get("payload") or {})
        if action == "merge_files":
            file_ids = {str(item) for item in payload.get("file_ids") or [] if str(item)}
            if len(file_ids) < 2:
                continue
            extracted: list[dict[str, Any]] = []
            remaining: list[dict[str, Any]] = []
            for obj in current:
                matches = [ref for ref in obj.get("files") or [] if str(ref.get("file_id") or "") in file_ids]
                if matches:
                    extracted.extend(matches)
                    keep = [ref for ref in obj.get("files") or [] if str(ref.get("file_id") or "") not in file_ids]
                    if keep:
                        clone = deepcopy(obj)
                        clone["files"] = keep
                        remaining.append(clone)
                else:
                    remaining.append(obj)
            if extracted:
                extracted = sorted(extracted, key=lambda item: (int(item.get("ordinal") or 0), str(item.get("file_id") or "")))
                remaining.append(
                    {
                        "object_id": f"obj-{uuid_from_text('merge' + ''.join(sorted(file_ids)))}",
                        "object_key": str(payload.get("group_key") or "manual-merge"),
                        "object_type": str(payload.get("object_type") or "unknown"),
                        "title": _slugify_title(str(payload.get("title") or "Manual Merge")),
                        "assembly_method": "manual",
                        "assembly_confidence": 1.0,
                        "status": "assembled",
                        "earliest": "",
                        "latest": "",
                        "era_bucket": "",
                        "media_family": str(payload.get("media_family") or ""),
                        "content_complexity": "multi",
                        "unresolved_reason": "",
                        "metadata": {
                            "override_id": override.get("override_id"),
                            "title_provenance": {"raw": str(payload.get("title") or "Manual Merge"), "edited": False},
                        },
                        "files": extracted,
                    }
                )
                current = remaining
        elif action == "split_object":
            target_file_ids = {str(item) for item in payload.get("file_ids") or [] if str(item)}
            if not target_file_ids:
                continue
            next_objects: list[dict[str, Any]] = []
            for obj in current:
                file_ids = {str(ref.get("file_id") or "") for ref in obj.get("files") or []}
                if file_ids != target_file_ids:
                    next_objects.append(obj)
                    continue
                for ref in obj.get("files") or []:
                    next_objects.append(
                        {
                            "object_id": f"obj-{uuid_from_text('split' + str(ref.get('file_id') or ''))}",
                            "object_key": f"{obj.get('object_key')}-{ref.get('file_id')}",
                            "object_type": obj.get("object_type") or "unknown",
                            "title": _slugify_title(str(obj.get("title") or "") + " " + str(ref.get("role") or "")),
                            "assembly_method": "manual",
                            "assembly_confidence": 1.0,
                            "status": "assembled",
                            "earliest": "",
                            "latest": "",
                            "era_bucket": "",
                            "media_family": obj.get("media_family") or "",
                            "content_complexity": "single",
                            "unresolved_reason": "",
                            "metadata": {
                                "override_id": override.get("override_id"),
                                "split_from": obj.get("object_id"),
                                "title_provenance": {"raw": str(obj.get("title") or ""), "edited": False},
                            },
                            "files": [dict(ref)],
                        }
                    )
            current = next_objects
        elif action == "set_roles":
            role_map = payload.get("file_roles") if isinstance(payload.get("file_roles"), list) else []
            mapping = {
                str(item.get("file_id") or ""): {
                    "role": str(item.get("role") or "component"),
                    "ordinal": int(item.get("ordinal") or 0),
                }
                for item in role_map
                if isinstance(item, dict)
            }
            if not mapping:
                continue
            for obj in current:
                for ref in obj.get("files") or []:
                    patch = mapping.get(str(ref.get("file_id") or ""))
                    if patch:
                        ref["role"] = patch["role"]
                        ref["ordinal"] = patch["ordinal"]
                obj["files"] = sorted(obj.get("files") or [], key=lambda item: (int(item.get("ordinal") or 0), str(item.get("file_id") or "")))
                obj["assembly_method"] = "manual"
                obj["assembly_confidence"] = 1.0
    return current


def _phase1_classify(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [deepcopy(item) for item in objects]
    for obj in out:
        title = str(obj.get("title") or "")
        meta = dict(obj.get("metadata") or {})
        file_paths = [str((ref.get("file") or {}).get("relative_path") or "") for ref in obj.get("files") or [] if isinstance(ref, dict)]
        text = " ".join([title] + file_paths)
        years = [int(year) for year in re.findall(r"\b(18\d{2}|19\d{2}|20\d{2})\b", text)]
        era_bucket = _year_to_bucket(years[0]) if years else ""
        roles = {str(ref.get("role") or "") for ref in obj.get("files") or []}
        if obj.get("object_type") == "unknown":
            obj["object_type"] = _guess_object_type(obj.get("files") or [])
        if not obj.get("content_complexity"):
            obj["content_complexity"] = "multi" if len(obj.get("files") or []) > 1 or "page" in roles else "single"
        routes = _phase1_routes_for_object(obj)
        obj["era_bucket"] = era_bucket
        meta["classification"] = {
            "era_bucket": era_bucket,
            "content_complexity": obj.get("content_complexity") or "",
            "object_type": obj.get("object_type") or "",
            "routes": routes,
        }
        obj["metadata"] = meta
    return out


def _phase1_routes_for_object(obj: dict[str, Any]) -> list[str]:
    object_type = str(obj.get("object_type") or "").strip().lower()
    media_family = str(obj.get("media_family") or "").strip().lower()
    complexity = str(obj.get("content_complexity") or "").strip().lower()
    routes: list[str] = []

    if object_type == "audio_recording" or media_family == "audio":
        routes.append("Audio Transcription")
    if object_type in {"document", "newspaper_page", "newspaper_article"}:
        routes.append("Printed OCR")
    if object_type == "correspondence":
        routes.append("Handwriting OCR")
    if object_type == "map_sheet":
        routes.extend(["Map Annotation OCR", "Visual Analysis"])
    if object_type == "photograph":
        routes.append("Visual Analysis")
        roles = {str(ref.get("role") or "").strip().lower() for ref in obj.get("files") or [] if isinstance(ref, dict)}
        if "back" in roles:
            routes.append("Handwriting OCR")
    if media_family == "document" and "Printed OCR" not in routes and "Handwriting OCR" not in routes:
        routes.append("Printed OCR")
    if media_family == "image" and "Visual Analysis" not in routes:
        routes.append("Visual Analysis")
    if complexity == "multi":
        routes.append("Layout Segmentation")
    deduped: list[str] = []
    seen: set[str] = set()
    for route in routes:
        label = str(route or "").strip()
        if not label or label in seen:
            continue
        seen.add(label)
        deduped.append(label)
    return deduped


def _classification_record(obj: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(obj.get("metadata") or {})
    classification = dict(metadata.get("classification") or {})
    object_type = str(classification.get("object_type") or obj.get("object_type") or "unknown").strip() or "unknown"
    era_bucket = str(classification.get("era_bucket") or obj.get("era_bucket") or "").strip()
    content_complexity = str(
        classification.get("content_complexity") or obj.get("content_complexity") or "unknown"
    ).strip() or "unknown"
    routes = classification.get("routes")
    if not isinstance(routes, list) or not routes:
        routes = _phase1_routes_for_object(obj)
    return {
        "object_type": object_type,
        "era_bucket": era_bucket,
        "content_complexity": content_complexity,
        "routes": [str(route) for route in routes if str(route or "").strip()],
    }


def _ordered_counts(counter: dict[str, int], preferred: tuple[str, ...], fallback_label: str = "unspecified") -> list[dict[str, Any]]:
    normalized: dict[str, int] = {}
    for key, count in counter.items():
        label = str(key or "").strip() or fallback_label
        normalized[label] = normalized.get(label, 0) + int(count or 0)
    ordered: list[dict[str, Any]] = []
    for key in preferred:
        if key in normalized:
            ordered.append({"value": key, "label": key, "count": int(normalized.pop(key))})
    for key in sorted(normalized):
        ordered.append({"value": key, "label": key, "count": int(normalized[key])})
    return ordered


def _summarize_classification_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    object_types: dict[str, int] = defaultdict(int)
    era_buckets: dict[str, int] = defaultdict(int)
    complexities: dict[str, int] = defaultdict(int)
    routes: dict[str, int] = defaultdict(int)
    for row in rows:
        if not isinstance(row, dict):
            continue
        classification = dict(row.get("classification") or {})
        object_types[str(classification.get("object_type") or row.get("object_type") or "unknown")] += 1
        era_buckets[str(classification.get("era_bucket") or row.get("era_bucket") or "")] += 1
        complexities[str(classification.get("content_complexity") or row.get("content_complexity") or "unknown")] += 1
        for route in classification.get("routes") or row.get("routing_labels") or []:
            routes[str(route or "").strip()] += 1
    return {
        "classified_count": len([row for row in rows if isinstance(row, dict)]),
        "object_types": _ordered_counts(object_types, _OBJECT_TYPE_ORDER, fallback_label="unknown"),
        "era_buckets": _ordered_counts(era_buckets, _ERA_BUCKET_ORDER, fallback_label="unspecified"),
        "content_complexities": _ordered_counts(complexities, _COMPLEXITY_ORDER, fallback_label="unknown"),
        "routes": _ordered_counts(routes, tuple(), fallback_label="unspecified"),
    }


def _phase2_handwriting_targets(obj: dict[str, Any]) -> list[dict[str, Any]]:
    object_type = str(obj.get("object_type") or "").strip().lower()
    targets: list[dict[str, Any]] = []
    for ref in obj.get("files") or []:
        if not isinstance(ref, dict):
            continue
        role = str(ref.get("role") or "").strip().lower()
        file_meta = ref.get("file") if isinstance(ref.get("file"), dict) else {}
        media_family = str((file_meta or {}).get("media_family") or obj.get("media_family") or "").strip().lower()
        if object_type == "correspondence" and media_family in {"image", "document"}:
            targets.append(ref)
            continue
        if role in {"back", "detail"} and media_family == "image":
            targets.append(ref)
    return targets


def _phase2_segmentation(obj: dict[str, Any], blocks: list[dict[str, Any]]) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    seen_file: set[str] = set()
    seen_page: set[tuple[str, int]] = set()
    seen_region: set[tuple[str, str]] = set()
    for ref in obj.get("files") or []:
        if not isinstance(ref, dict):
            continue
        file_id = str(ref.get("file_id") or "")
        if not file_id or file_id in seen_file:
            continue
        seen_file.add(file_id)
        file_meta = ref.get("file") if isinstance(ref.get("file"), dict) else {}
        segments.append(
            {
                "segment_id": f"seg-file-{file_id}",
                "segment_kind": "file",
                "source_file_id": file_id,
                "page_index": None,
                "raw_region": {"kind": "file"},
                "metadata": {
                    "role": str(ref.get("role") or ""),
                    "path": str((file_meta or {}).get("relative_path") or ""),
                },
            }
        )
    for block in blocks:
        if not isinstance(block, dict):
            continue
        file_id = str(block.get("source_file_id") or "")
        page_index = block.get("page_index")
        raw_region = dict(block.get("raw_region") or {})
        if file_id and isinstance(page_index, int):
            key = (file_id, page_index)
            if key not in seen_page:
                seen_page.add(key)
                segments.append(
                    {
                        "segment_id": f"seg-page-{file_id}-{page_index}",
                        "segment_kind": "page",
                        "source_file_id": file_id,
                        "page_index": page_index,
                        "raw_region": {"kind": "page", "page_index": page_index},
                        "metadata": {},
                    }
                )
        region_kind = str(raw_region.get("kind") or "")
        if file_id and region_kind and ({"left", "top", "width", "height"} & set(raw_region.keys()) or {"start_sec", "end_sec"} & set(raw_region.keys())):
            region_id = str(block.get("segment_id") or block.get("block_id") or "")
            key = (file_id, region_id)
            if key not in seen_region:
                seen_region.add(key)
                segments.append(
                    {
                        "segment_id": region_id or f"seg-region-{file_id}-{len(seen_region)}",
                        "segment_kind": "region",
                        "source_file_id": file_id,
                        "page_index": int(page_index) if isinstance(page_index, int) else None,
                        "raw_region": raw_region,
                        "metadata": {"block_kind": str(block.get("block_kind") or "")},
                    }
                )
    summary = {
        "file_segments": sum(1 for segment in segments if str(segment.get("segment_kind") or "") == "file"),
        "page_segments": sum(1 for segment in segments if str(segment.get("segment_kind") or "") == "page"),
        "region_segments": sum(1 for segment in segments if str(segment.get("segment_kind") or "") == "region"),
        "segment_count": len(segments),
    }
    return {"segments": segments, "summary": summary}


def _phase2_extraction_summary(obj: dict[str, Any], blocks: list[dict[str, Any]]) -> dict[str, Any]:
    provider_counts: dict[str, int] = defaultdict(int)
    capability_counts: dict[str, int] = defaultdict(int)
    kind_counts: dict[str, int] = defaultdict(int)
    extracted_text_blocks = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        provider_counts[str(block.get("provider") or "unknown")] += 1
        capability_counts[str(block.get("capability") or "unknown")] += 1
        kind_counts[str(block.get("block_kind") or "unknown")] += 1
        if str(block.get("text") or "").strip():
            extracted_text_blocks += 1
    has_ok = capability_counts.get("ok", 0) > 0
    has_failed = capability_counts.get("failed", 0) > 0
    has_unavailable = capability_counts.get("capability_unavailable", 0) > 0
    if has_ok:
        status = "extracted"
    elif has_failed:
        status = "failed"
    elif has_unavailable:
        status = "capability_unavailable"
    else:
        status = "extraction_ready"
    return {
        "status": status,
        "provider_counts": dict(provider_counts),
        "capability_counts": dict(capability_counts),
        "block_kind_counts": dict(kind_counts),
        "block_count": len(blocks),
        "text_block_count": extracted_text_blocks,
        "provider_summary": ", ".join(f"{key}:{provider_counts[key]}" for key in sorted(provider_counts)) or "none",
        "signal_kind": "audio" if kind_counts.get("audio_transcript", 0) > 0 else str(obj.get("media_family") or "document"),
    }


def _phase2_review_row(
    obj: dict[str, Any],
    *,
    assertion_total: int = 0,
    assertion_edited: int = 0,
) -> dict[str, Any]:
    metadata = dict(obj.get("metadata") or {})
    classification = _classification_record(obj)
    extraction_summary = dict(metadata.get("extraction_summary") or {})
    segmentation = dict(metadata.get("segmentation") or {})
    segmentation_summary = dict(segmentation.get("summary") or {})
    text_blocks = [dict(block) for block in (metadata.get("text_blocks") or []) if isinstance(block, dict)]
    files = [dict(ref) for ref in (obj.get("files") or []) if isinstance(ref, dict)]
    missing_file_count = sum(
        1
        for ref in files
        if not bool((ref.get("file") or {}).get("exists"))
    )
    return {
        "object_id": str(obj.get("object_id") or ""),
        "title": str(obj.get("title") or ""),
        "object_type": str(obj.get("object_type") or ""),
        "era_bucket": str(obj.get("era_bucket") or ""),
        "media_family": str(obj.get("media_family") or ""),
        "content_complexity": str(obj.get("content_complexity") or ""),
        "assembly_method": str(obj.get("assembly_method") or ""),
        "assembly_confidence": float(obj.get("assembly_confidence") or 0.0),
        "classification": classification,
        "routing_labels": list(classification.get("routes") or []),
        "route_summary": ", ".join(list(classification.get("routes") or [])[:3]),
        "status": str(obj.get("status") or ""),
        "file_count": len(files),
        "files": files,
        "missing_file_count": missing_file_count,
        "assertion_count": int(assertion_total),
        "edited_assertion_count": int(assertion_edited),
        "text_block_count": int(extraction_summary.get("text_block_count") or 0),
        "extraction_summary": extraction_summary,
        "segmentation_summary": segmentation_summary,
        "provider_summary": str(extraction_summary.get("provider_summary") or "none"),
        "signal_kind": str(extraction_summary.get("signal_kind") or obj.get("media_family") or "document"),
        "has_audio": bool((extraction_summary.get("block_kind_counts") or {}).get("audio_transcript")),
        "has_printed_ocr": bool((extraction_summary.get("block_kind_counts") or {}).get("printed_ocr")),
        "has_handwriting_ocr": bool((extraction_summary.get("block_kind_counts") or {}).get("handwriting_ocr")),
        "text_blocks": text_blocks,
        "metadata": metadata,
    }


def _phase2_extract_text(objects: list[dict[str, Any]], providers: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    out = [deepcopy(item) for item in objects]
    provider_cfg = _phase2_provider_config(providers if isinstance(providers, dict) else None)
    for obj in out:
        meta = dict(obj.get("metadata") or {})
        blocks: list[dict[str, Any]] = []
        for ref in obj.get("files") or []:
            file_meta = ref.get("file") if isinstance(ref.get("file"), dict) else {}
            if not isinstance(file_meta, dict) or not file_meta.get("processable"):
                continue
            for block in _extract_text_blocks(file_meta, provider_cfg):
                block_meta = dict(block.get("metadata") or {})
                block_meta.setdefault("file_role", str(ref.get("role") or "component"))
                block["metadata"] = block_meta
                blocks.append(block)
        for ref in _phase2_handwriting_targets(obj):
            file_id = str(ref.get("file_id") or "")
            if any(str(block.get("block_kind") or "") == "handwriting_ocr" and str(block.get("source_file_id") or "") == file_id for block in blocks):
                continue
            blocks.append(
                _phase2_block(
                    provider=str(provider_cfg.get("handwriting_ocr") or "none"),
                    capability="capability_unavailable",
                    block_kind="handwriting_ocr",
                    source_file_id=file_id,
                    text="",
                    confidence=0.0,
                    raw_region={"kind": "image"},
                    metadata={"file_role": str(ref.get("role") or "component")},
                )
            )
        title_text = str(obj.get("title") or "").strip()
        if title_text:
            blocks.insert(
                0,
                _phase2_block(
                    provider="title",
                    capability="ok",
                    block_kind="direct_text",
                    source_file_id="",
                    text=title_text,
                    confidence=1.0,
                    raw_region={"kind": "title"},
                ),
            )
        segmentation = _phase2_segmentation(obj, blocks)
        extraction_summary = _phase2_extraction_summary(obj, blocks)
        meta["text_blocks"] = blocks
        meta["text_content"] = "\n".join(str(block.get("text") or "") for block in blocks if str(block.get("text") or "").strip())
        meta["segmentation"] = segmentation
        meta["extraction_summary"] = extraction_summary
        meta["phase2_providers"] = provider_cfg
        obj["metadata"] = meta
    return out


def _phase3_structure(objects: list[dict[str, Any]]) -> dict[str, Any]:
    objects_out = [deepcopy(item) for item in objects]
    anchors: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    mentions: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    entity_by_key: dict[str, dict[str, Any]] = {}

    for obj in objects_out:
        meta = dict(obj.get("metadata") or {})
        raw_text = str(meta.get("text_content") or "")
        anchors.extend(_collect_temporal_anchors(raw_text, str(obj.get("object_id") or "")))
        candidates = _extract_entity_candidates(raw_text)
        fallback_title = str(obj.get("title") or "")
        if fallback_title:
            candidates.extend(_extract_entity_candidates(fallback_title.title()))
        object_entity_ids: list[str] = []
        for candidate in candidates:
            key = _canonical_entity_key(candidate)
            if not key:
                continue
            record = entity_by_key.get(key)
            if record is None:
                entity_id = f"entity-{uuid_from_text(key)}"
                record = {
                    "entity_id": entity_id,
                    "canonical_name": candidate,
                    "entity_type": _classify_entity_type(candidate),
                    "aliases": [candidate],
                    "confidence": 0.68,
                    "known_facts": {},
                }
                entity_by_key[key] = record
                entities.append(record)
            else:
                aliases = set(str(alias) for alias in record.get("aliases") or [])
                aliases.add(candidate)
                record["aliases"] = sorted(aliases)
            mention_id = f"mention-{uuid_from_text(str(obj.get('object_id') or '') + candidate + key)}"
            mentions.append(
                {
                    "mention_id": mention_id,
                    "entity_id": record["entity_id"],
                    "object_id": str(obj.get("object_id") or ""),
                    "text_span": candidate,
                    "mention_text": candidate,
                    "mention_confidence": 0.66,
                    "metadata": {},
                }
            )
            object_entity_ids.append(record["entity_id"])

        person_ids = [eid for eid in object_entity_ids if any(item["entity_id"] == eid and item["entity_type"] == "person" for item in entities)]
        place_ids = [eid for eid in object_entity_ids if any(item["entity_id"] == eid and item["entity_type"] == "place" for item in entities)]
        for anchor in [item for item in anchors if str(item.get("object_id") or "") == str(obj.get("object_id") or "") and str(item.get("type") or "") == "age_based"]:
            if person_ids:
                relationships.append(
                    {
                        "relationship_id": f"rel-{uuid_from_text(anchor['anchor_id'] + person_ids[0])}",
                        "source_entity_id": person_ids[0],
                        "target_entity_id": "",
                        "object_id": str(obj.get("object_id") or ""),
                        "relationship_type": "person_at_age",
                        "attributes": {"age": int((anchor.get("metadata") or {}).get("age") or 0)},
                        "confidence": 0.61,
                    }
                )
        if person_ids and place_ids:
            relationships.append(
                {
                    "relationship_id": f"rel-{uuid_from_text(str(obj.get('object_id') or '') + person_ids[0] + place_ids[0])}",
                    "source_entity_id": person_ids[0],
                    "target_entity_id": place_ids[0],
                    "object_id": str(obj.get("object_id") or ""),
                    "relationship_type": "person_at_place",
                    "attributes": {},
                    "confidence": 0.58,
                }
            )

        anchor_years = []
        for anchor in [item for item in anchors if str(item.get("object_id") or "") == str(obj.get("object_id") or "")]:
            earliest = str(anchor.get("earliest") or "")
            latest = str(anchor.get("latest") or "")
            if earliest:
                anchor_years.append(earliest)
            if latest:
                anchor_years.append(latest)
        if anchor_years:
            obj["earliest"] = min(anchor_years)
            obj["latest"] = max(anchor_years)
    return {
        "objects": objects_out,
        "anchors": anchors,
        "entities": entities,
        "mentions": mentions,
        "relationships": relationships,
        "clusters": [],
    }


def _phase4_visual(objects: list[dict[str, Any]], anchors: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    objects_out = [deepcopy(item) for item in objects]
    anchors_out = [deepcopy(item) for item in anchors]
    for obj in objects_out:
        if str(obj.get("media_family") or "") != "image":
            continue
        meta = dict(obj.get("metadata") or {})
        first_file = next((ref.get("file") for ref in obj.get("files") or [] if isinstance(ref.get("file"), dict)), None)
        if not isinstance(first_file, dict):
            continue
        image = cv2.imread(str(first_file.get("stored_path") or ""))
        if image is None:
            continue
        hist = cv2.calcHist([image], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(float)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_luma = float(np.mean(gray))
        scene_class = "portrait" if image.shape[0] > image.shape[1] * 1.15 else "landscape"
        if "building" in str(obj.get("title") or "").lower():
            scene_class = "building"
        meta["visual"] = {
            "scene_class": scene_class,
            "embedding": [round(float(value), 6) for value in hist[:24]],
            "mean_luma": round(mean_luma, 3),
        }
        obj["metadata"] = meta
        has_anchor = any(str(anchor.get("object_id") or "") == str(obj.get("object_id") or "") for anchor in anchors_out)
        if not has_anchor and str(obj.get("era_bucket") or ""):
            bucket = str(obj.get("era_bucket") or "")
            ranges = {
                "pre-1900": ("1850-01-01", "1899-12-31"),
                "1900-1920": ("1900-01-01", "1920-12-31"),
                "1920-1940": ("1920-01-01", "1940-12-31"),
                "1940-1960": ("1940-01-01", "1960-12-31"),
                "1960-1980": ("1960-01-01", "1980-12-31"),
                "1980-present": ("1980-01-01", "2026-12-31"),
            }
            earliest, latest = ranges.get(bucket, ("", ""))
            if earliest and latest:
                anchors_out.append(
                    {
                        "anchor_id": f"anchor-{uuid_from_text(str(obj.get('object_id') or '') + bucket + 'visual')}",
                        "object_id": str(obj.get("object_id") or ""),
                        "type": "visual_estimate",
                        "earliest": earliest,
                        "latest": latest,
                        "confidence": 0.22,
                        "source": "visual_estimation",
                        "is_publication_date": False,
                        "raw_expression": bucket,
                        "resolved": True,
                        "resolution_requires": "",
                        "metadata": {"scene_class": scene_class},
                    }
                )
                obj["earliest"] = obj.get("earliest") or earliest
                obj["latest"] = obj.get("latest") or latest
    return objects_out, anchors_out


def _phase5_resolve(
    objects: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    resolution_overrides: list[dict[str, Any]],
) -> dict[str, Any]:
    objects_out = [deepcopy(item) for item in objects]
    anchors_out = [deepcopy(item) for item in anchors]
    entities_out = [deepcopy(item) for item in entities]
    mentions_out = [deepcopy(item) for item in mentions]
    relationships_out = [deepcopy(item) for item in relationships]

    overrides_merge: list[dict[str, Any]] = []
    blocked_pairs: set[tuple[str, str]] = set()
    pinned_anchor_map: dict[str, dict[str, Any]] = {}
    for override in resolution_overrides:
        action = str(override.get("action") or "")
        payload = dict(override.get("payload") or {})
        if action == "merge_entities":
            overrides_merge.append({"target_id": str(override.get("target_id") or ""), "payload": payload})
        elif action == "reject_entity_merge":
            target = str(override.get("target_id") or "")
            for other in payload.get("other_entity_ids") or []:
                pair = tuple(sorted((target, str(other))))
                blocked_pairs.add(pair)
        elif action == "pin_date":
            pinned_anchor_map[str(override.get("target_id") or "")] = payload

    merged_entities: list[dict[str, Any]] = []
    used: set[str] = set()
    for entity in sorted(entities_out, key=lambda item: str(item.get("canonical_name") or "").lower()):
        entity_id = str(entity.get("entity_id") or "")
        if entity_id in used:
            continue
        merged = deepcopy(entity)
        merged_aliases = {str(alias) for alias in merged.get("aliases") or [] if str(alias)}
        key_a = _canonical_entity_key(str(merged.get("canonical_name") or ""))
        for other in entities_out:
            other_id = str(other.get("entity_id") or "")
            if other_id == entity_id or other_id in used:
                continue
            pair = tuple(sorted((entity_id, other_id)))
            if pair in blocked_pairs:
                continue
            key_b = _canonical_entity_key(str(other.get("canonical_name") or ""))
            # Exact-key duplicates are deterministically deduped here; fuzzy matches
            # are surfaced as entity_merge proposals (propose-then-confirm) rather than
            # being silently auto-merged. Confirmed merges arrive via resolution
            # overrides (manual edits) or memory-confirmed ops appended by run_phase.
            if key_a and key_b and key_a == key_b:
                used.add(other_id)
                merged_aliases.add(str(other.get("canonical_name") or ""))
                merged_aliases.update(str(alias) for alias in other.get("aliases") or [] if str(alias))
        for patch in overrides_merge:
            if str(patch.get("target_id") or "") != entity_id:
                continue
            for other_id in patch["payload"].get("other_entity_ids") or []:
                other = next((item for item in entities_out if str(item.get("entity_id") or "") == str(other_id)), None)
                if other is None:
                    continue
                used.add(str(other_id))
                merged_aliases.add(str(other.get("canonical_name") or ""))
                merged_aliases.update(str(alias) for alias in other.get("aliases") or [] if str(alias))
            if str(patch["payload"].get("canonical_name") or "").strip():
                merged["canonical_name"] = str(patch["payload"]["canonical_name"]).strip()
        merged["aliases"] = sorted(merged_aliases)
        merged_entities.append(merged)
        used.add(entity_id)
    entities_out = merged_entities

    entity_key_map: dict[str, str] = {}
    for entity in entities_out:
        for alias in [str(entity.get("canonical_name") or "")] + [str(alias) for alias in entity.get("aliases") or []]:
            key = _canonical_entity_key(alias)
            if key:
                entity_key_map[key] = str(entity.get("entity_id") or "")

    for mention in mentions_out:
        new_id = entity_key_map.get(_canonical_entity_key(str(mention.get("mention_text") or "")))
        if new_id:
            mention["entity_id"] = new_id

    birth_year_by_entity: dict[str, int] = {}
    for entity in entities_out:
        canon = str(entity.get("canonical_name") or "")
        match = re.search(r"\bborn\s+(18\d{2}|19\d{2}|20\d{2})\b", canon, flags=re.IGNORECASE)
        if match:
            birth_year_by_entity[str(entity.get("entity_id") or "")] = int(match.group(1))
    object_persons: dict[str, list[str]] = defaultdict(list)
    for mention in mentions_out:
        object_persons[str(mention.get("object_id") or "")].append(str(mention.get("entity_id") or ""))

    for anchor in anchors_out:
        pinned = pinned_anchor_map.get(str(anchor.get("anchor_id") or ""))
        if pinned:
            anchor["earliest"] = str(pinned.get("earliest") or anchor.get("earliest") or "")
            anchor["latest"] = str(pinned.get("latest") or anchor.get("latest") or "")
            anchor["resolved"] = True
            anchor["source"] = "manual"
            continue
        if str(anchor.get("type") or "") != "age_based":
            continue
        people = object_persons.get(str(anchor.get("object_id") or ""), [])
        if not people:
            continue
        birth_year = birth_year_by_entity.get(people[0])
        age = int((anchor.get("metadata") or {}).get("age") or 0)
        if birth_year and age > 0:
            year = birth_year + age
            anchor["earliest"] = f"{year:04d}-01-01"
            anchor["latest"] = f"{year:04d}-12-31"
            anchor["resolved"] = True
            anchor["source"] = "cross_reference"
            anchor["resolution_requires"] = ""

    cluster_map: dict[str, list[str]] = defaultdict(list)
    for mention in mentions_out:
        entity = next((item for item in entities_out if str(item.get("entity_id") or "") == str(mention.get("entity_id") or "")), None)
        if entity is None:
            continue
        label = str(entity.get("canonical_name") or "").strip()
        if label:
            cluster_map[label].append(str(mention.get("object_id") or ""))
    clusters = []
    for label, object_ids in cluster_map.items():
        uniq = sorted({obj_id for obj_id in object_ids if obj_id})
        if not uniq:
            continue
        related_anchors = [anchor for anchor in anchors_out if str(anchor.get("object_id") or "") in uniq and str(anchor.get("earliest") or "")]
        earliest = min((str(anchor.get("earliest") or "") for anchor in related_anchors), default="")
        latest = max((str(anchor.get("latest") or "") for anchor in related_anchors), default="")
        clusters.append(
            {
                "cluster_id": f"cluster-{uuid_from_text(label)}",
                "label": label,
                "object_ids": uniq,
                "earliest": earliest,
                "latest": latest,
                "dominant_entities": [label],
                "embedding_centroid": [],
                "metadata": {"object_count": len(uniq)},
            }
        )

    anchors_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor in anchors_out:
        anchors_by_object[str(anchor.get("object_id") or "")].append(anchor)
    for obj in objects_out:
        relevant = anchors_by_object.get(str(obj.get("object_id") or ""), [])
        dated = [anchor for anchor in relevant if str(anchor.get("earliest") or "")]
        if dated:
            obj["earliest"] = min(str(anchor.get("earliest") or "") for anchor in dated)
            obj["latest"] = max(str(anchor.get("latest") or "") for anchor in dated)
            obj["unresolved_reason"] = ""
        elif not str(obj.get("era_bucket") or ""):
            obj["unresolved_reason"] = "no_temporal_anchor"

    return {
        "objects": objects_out,
        "anchors": anchors_out,
        "entities": entities_out,
        "mentions": mentions_out,
        "relationships": relationships_out,
        "clusters": clusters,
    }


# --- Phase 5 propose-then-confirm -----------------------------------------

_PHASE5_MERGE_PROPOSAL_THRESHOLD = 0.82

_PHASE5_ERA_RANGES = {
    "pre-1900": ("1850-01-01", "1899-12-31"),
    "1900-1920": ("1900-01-01", "1920-12-31"),
    "1920-1940": ("1920-01-01", "1940-12-31"),
    "1940-1960": ("1940-01-01", "1960-12-31"),
    "1960-1980": ("1960-01-01", "1980-12-31"),
    "1980-present": ("1980-01-01", "2026-12-31"),
}


def _phase5_entity_index(
    entities: list[dict[str, Any]], mentions: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for entity in entities:
        eid = str(entity.get("entity_id") or "")
        if not eid:
            continue
        by_id[eid] = {
            "entity_id": eid,
            "canonical_name": str(entity.get("canonical_name") or ""),
            "entity_type": str(entity.get("entity_type") or "unknown"),
            "key": _canonical_entity_key(str(entity.get("canonical_name") or "")),
            "object_ids": [],
        }
    for mention in mentions:
        eid = str(mention.get("entity_id") or "")
        oid = str(mention.get("object_id") or "")
        if eid in by_id and oid and oid not in by_id[eid]["object_ids"]:
            by_id[eid]["object_ids"].append(oid)
    return by_id


def _generate_phase5_proposals(
    *,
    dataset_version_id: str,
    objects: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    decision_memory: list[dict[str, Any]],
    generator_run_id: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Emit explainable resolution proposals from raw (pre-merge) state.

    Returns (proposals, evidence, memory_ops). memory_ops are resolution-override
    shaped dicts for proposals a prior operator already confirmed (looked up by
    stable signature in cross-snapshot decision memory); they are applied during
    `_phase5_resolve` so confirmed merges/pins re-materialize on every rerun.
    """
    proposals: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    memory_ops: list[dict[str, Any]] = []
    mem_by_key = {
        (str(item.get("proposal_type")), str(item.get("signature"))): item
        for item in (decision_memory or [])
    }
    objects_by_id = {str(o.get("object_id") or ""): o for o in objects if isinstance(o, dict)}
    entity_names = {str(e.get("entity_id") or ""): str(e.get("canonical_name") or "") for e in entities}

    def _status_for(ptype: str, signature: str) -> tuple[str, str]:
        mem = mem_by_key.get((ptype, signature))
        if mem is None:
            return "proposed", ""
        decision = str(mem.get("decision") or "")
        if decision == "reject":
            return "auto_suppressed", "reject"
        if decision == "confirm":
            return "confirmed", "confirm"
        return "proposed", ""

    def _add(proposal: dict[str, Any], ev_list: list[dict[str, Any]]) -> None:
        proposals.append(proposal)
        for idx, ev in enumerate(ev_list):
            row = dict(ev)
            row["evidence_id"] = f"ev-{proposal['proposal_id']}-{idx}"
            row["proposal_id"] = proposal["proposal_id"]
            evidence.append(row)

    # --- entity_merge: fuzzy (non-exact) name matches ---
    indexed = sorted(_phase5_entity_index(entities, mentions).values(), key=lambda v: (v["key"], v["entity_id"]))
    for i in range(len(indexed)):
        for j in range(i + 1, len(indexed)):
            subject, related = indexed[i], indexed[j]
            if not subject["key"] or not related["key"] or subject["key"] == related["key"]:
                continue
            score = _similarity(subject["key"], related["key"])
            if score < _PHASE5_MERGE_PROPOSAL_THRESHOLD or score >= 1.0:
                continue
            signature = uuid_from_text("entity_merge|" + "|".join(sorted([subject["key"], related["key"]])))
            pid = f"prop-{signature}"
            canonical = (
                subject["canonical_name"]
                if len(subject["canonical_name"]) >= len(related["canonical_name"])
                else related["canonical_name"]
            )
            rep_obj = (subject["object_ids"] or related["object_ids"] or [""])[0]
            shared = sorted(set(subject["object_ids"]) & set(related["object_ids"]))
            status, decision = _status_for("entity_merge", signature)
            if decision == "confirm":
                memory_ops.append(
                    {
                        "action": "merge_entities",
                        "target_id": subject["entity_id"],
                        "payload": {"other_entity_ids": [related["entity_id"]], "canonical_name": canonical},
                    }
                )
            ev_list = [
                {
                    "evidence_type": "name_similarity",
                    "description": f"'{subject['canonical_name']}' ~ '{related['canonical_name']}' ({score:.2f})",
                    "weight": round(float(score), 4),
                    "supporting_refs": {
                        "entity_ids": [subject["entity_id"], related["entity_id"]],
                        "names": [subject["canonical_name"], related["canonical_name"]],
                    },
                }
            ]
            if shared:
                ev_list.append(
                    {
                        "evidence_type": "co_occurrence",
                        "description": f"co-mentioned on {len(shared)} object(s)",
                        "weight": round(min(1.0, 0.3 + 0.1 * len(shared)), 4),
                        "supporting_refs": {"object_ids": shared},
                    }
                )
            _add(
                {
                    "proposal_id": pid,
                    "proposal_type": "entity_merge",
                    "target_kind": "entity",
                    "target_id": subject["entity_id"],
                    "subject_id": subject["entity_id"],
                    "related_id": related["entity_id"],
                    "proposed_value": {"canonical_name": canonical, "merged_entity_ids": [subject["entity_id"], related["entity_id"]]},
                    "confidence": round(float(score), 4),
                    "signature": signature,
                    "status": status,
                    "review_bucket": "entity_merge",
                    "generator": "name_similarity",
                    "generator_run_id": generator_run_id,
                    "metadata": {"object_id": rep_obj, "subject_name": subject["canonical_name"], "related_name": related["canonical_name"]},
                },
                ev_list,
            )

    # --- anchor_resolution / temporal_propagation: unresolved anchors ---
    birth_year_by_entity: dict[str, int] = {}
    for entity in entities:
        match = re.search(r"\bborn\s+(18\d{2}|19\d{2}|20\d{2})\b", str(entity.get("canonical_name") or ""), flags=re.IGNORECASE)
        if match:
            birth_year_by_entity[str(entity.get("entity_id") or "")] = int(match.group(1))
    persons_by_object: dict[str, list[str]] = defaultdict(list)
    for mention in mentions:
        persons_by_object[str(mention.get("object_id") or "")].append(str(mention.get("entity_id") or ""))

    for anchor in anchors:
        if bool(anchor.get("resolved")):
            continue
        aid = str(anchor.get("anchor_id") or "")
        oid = str(anchor.get("object_id") or "")
        raw = str(anchor.get("raw_expression") or "")
        if not aid or not oid:
            continue
        atype = str(anchor.get("type") or "")
        proposed: Optional[tuple[str, str]] = None
        ptype = "anchor_resolution"
        ev_type = "temporal_constraint"
        ev_desc = f"era estimate for '{raw}'" if raw else "era estimate"
        if atype == "age_based":
            people = [pid_ for pid_ in persons_by_object.get(oid, []) if pid_ in birth_year_by_entity]
            age = int((anchor.get("metadata") or {}).get("age") or 0)
            if people and age > 0:
                year = birth_year_by_entity[people[0]] + age
                proposed = (f"{year:04d}-01-01", f"{year:04d}-12-31")
                ptype = "temporal_propagation"
                ev_type = "co_occurrence"
                ev_desc = f"age {age} + entity '{entity_names.get(people[0], people[0])}' born {birth_year_by_entity[people[0]]}"
        if proposed is None:
            era = str((objects_by_id.get(oid) or {}).get("era_bucket") or "")
            rng = _PHASE5_ERA_RANGES.get(era)
            if rng:
                proposed = rng
                ev_desc = f"object era bucket {era}"
        if proposed is None:
            continue
        signature = uuid_from_text(ptype + "|" + oid + "|" + raw)
        pid = f"prop-{signature}"
        status, decision = _status_for(ptype, signature)
        if decision == "confirm":
            memory_ops.append({"action": "pin_date", "target_id": aid, "payload": {"earliest": proposed[0], "latest": proposed[1]}})
        _add(
            {
                "proposal_id": pid,
                "proposal_type": ptype,
                "target_kind": "temporal_anchor",
                "target_id": aid,
                "subject_id": oid,
                "proposed_value": {"earliest": proposed[0], "latest": proposed[1]},
                "confidence": 0.45 if ptype == "temporal_propagation" else 0.3,
                "signature": signature,
                "status": status,
                "review_bucket": ptype,
                "generator": "cross_reference" if ptype == "temporal_propagation" else "era_estimate",
                "generator_run_id": generator_run_id,
                "metadata": {"object_id": oid, "raw_expression": raw, "anchor_type": atype},
            },
            [
                {
                    "evidence_type": ev_type,
                    "description": ev_desc,
                    "weight": 0.5,
                    "supporting_refs": {"anchor_ids": [aid], "object_ids": [oid]},
                }
            ],
        )

    # --- relationship: persons co-appearing on the same object ---
    for oid, _obj in objects_by_id.items():
        people = sorted({pid_ for pid_ in persons_by_object.get(oid, []) if pid_})
        if len(people) < 2:
            continue
        a_id, b_id = people[0], people[1]
        a_name = entity_names.get(a_id, a_id)
        b_name = entity_names.get(b_id, b_id)
        keys = sorted([_canonical_entity_key(a_name), _canonical_entity_key(b_name)])
        signature = uuid_from_text("relationship|" + oid + "|" + "|".join(keys))
        pid = f"prop-{signature}"
        status, _decision = _status_for("relationship", signature)
        _add(
            {
                "proposal_id": pid,
                "proposal_type": "relationship",
                "target_kind": "relationship",
                "target_id": "",
                "subject_id": a_id,
                "related_id": b_id,
                "proposed_value": {"relationship_type": "co_appears_with", "source": a_name, "target": b_name},
                "confidence": 0.35,
                "signature": signature,
                "status": status,
                "review_bucket": "relationship",
                "generator": "co_occurrence",
                "generator_run_id": generator_run_id,
                "metadata": {"object_id": oid, "source_name": a_name, "target_name": b_name},
            },
            [
                {
                    "evidence_type": "co_occurrence",
                    "description": f"'{a_name}' and '{b_name}' appear on the same object",
                    "weight": 0.5,
                    "supporting_refs": {"object_ids": [oid], "entity_ids": [a_id, b_id]},
                }
            ],
        )

    return proposals, evidence, memory_ops


def _phase5_cascade_for_confirm(proposal: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Confirming an entity_merge spawns a downstream cluster_membership proposal."""
    if str(proposal.get("proposal_type") or "") != "entity_merge":
        return [], []
    meta = dict(proposal.get("metadata") or {})
    label = str(meta.get("subject_name") or dict(proposal.get("proposed_value") or {}).get("canonical_name") or "")
    signature = uuid_from_text("cluster_membership|" + str(proposal.get("signature") or ""))
    pid = f"prop-casc-{signature}"
    cascade_proposal = {
        "proposal_id": pid,
        "proposal_type": "cluster_membership",
        "target_kind": "cluster",
        "target_id": "",
        "subject_id": str(proposal.get("subject_id") or ""),
        "related_id": str(proposal.get("related_id") or ""),
        "proposed_value": {"cluster_label": label},
        "confidence": round(float(proposal.get("confidence") or 0.0) * 0.9, 4),
        "signature": signature,
        "status": "proposed",
        "review_bucket": "cluster_membership",
        "generator": "cascade",
        "generator_run_id": "",
        "cascade_source_proposal_id": str(proposal.get("proposal_id") or ""),
        "metadata": {"object_id": str(meta.get("object_id") or ""), "cluster_label": label},
    }
    cascade_evidence = [
        {
            "evidence_id": f"ev-{pid}-0",
            "proposal_id": pid,
            "evidence_type": "shared_attribute",
            "description": f"merged entity '{label}' suggests a semantic cluster",
            "weight": 0.6,
            "supporting_refs": {"entity_ids": [str(proposal.get("subject_id") or ""), str(proposal.get("related_id") or "")]},
        }
    ]
    return [cascade_proposal], cascade_evidence


def _proposal_subject_summary(proposal: dict[str, Any]) -> dict[str, Any]:
    meta = dict(proposal.get("metadata") or {})
    pv = dict(proposal.get("proposed_value") or {})
    return {
        "type": str(proposal.get("proposal_type") or ""),
        "subject": str(meta.get("subject_name") or meta.get("source_name") or pv.get("canonical_name") or proposal.get("subject_id") or ""),
        "related": str(meta.get("related_name") or meta.get("target_name") or proposal.get("related_id") or ""),
    }


def _assertion_for_confirmed_proposal(proposal: dict[str, Any]) -> Optional[dict[str, Any]]:
    meta = dict(proposal.get("metadata") or {})
    object_id = str(meta.get("object_id") or "")
    if not object_id:
        return None  # an assertion must reference a real object in the snapshot
    ptype = str(proposal.get("proposal_type") or "")
    pv = dict(proposal.get("proposed_value") or {})
    if ptype == "entity_merge":
        field, value = "entity_resolution", f"merge -> {str(pv.get('canonical_name') or '')}"
    elif ptype in {"anchor_resolution", "temporal_propagation"}:
        field, value = "temporal_resolution", f"{str(pv.get('earliest') or '')} -> {str(pv.get('latest') or '')}"
    elif ptype == "relationship":
        field, value = "relationship", f"{str(pv.get('source') or '')} {str(pv.get('relationship_type') or 'related_to')} {str(pv.get('target') or '')}"
    elif ptype == "cluster_membership":
        field, value = "cluster_membership", str(pv.get("cluster_label") or "")
    else:
        field, value = "resolution", ""
    return {
        "assertion_id": f"assert-confirm-{str(proposal.get('proposal_id') or '')}",
        "object_id": object_id,
        "field": field,
        "raw_extraction": "",
        "current_value": value,
        "current_confidence": float(proposal.get("confidence") or 0.0),
        "extraction_model": "archive_phase5",
        "extraction_run_id": "archive_phase5_confirm",
        "source_type": "operator_confirmation",
        "metadata": {"proposal_id": str(proposal.get("proposal_id") or ""), "proposal_type": ptype, "signature": str(proposal.get("signature") or "")},
        "created_at": time.time(),
    }


def apply_proposal_decision(
    store: ArchiveStore,
    snapshot_id: str,
    proposal_id: str,
    decision: str,
    *,
    decided_by: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """Record an operator decision on a proposal (append-only, never mutates state).

    confirm  -> append-only assertion + immutable decision + decision memory(confirm)
                + any cascade proposals; the merge/pin materializes on the next rerun.
    reject   -> immutable decision + decision memory(reject); no assertion mutation.
    defer    -> immutable decision only; proposal stays in the queue.
    undo     -> immutable decision + clears decision memory; proposal returns to proposed.
    """
    decision = str(decision or "").strip()
    if decision not in {"confirm", "reject", "defer", "undo"}:
        raise ValueError(f"unsupported decision: {decision}")
    proposal = store.get_proposal(snapshot_id, proposal_id)
    snapshot = store.get_snapshot(snapshot_id)
    corpus_id = str(snapshot.get("corpus_id") or "")
    dataset_version_id = str(snapshot.get("dataset_version_id") or "")
    ptype = str(proposal.get("proposal_type") or "")
    signature = str(proposal.get("signature") or "")

    if decision == "confirm":
        assertion = _assertion_for_confirmed_proposal(proposal)
        resulting_assertion_id = ""
        if assertion:
            store.add_assertion(snapshot_id, assertion)
            resulting_assertion_id = str(assertion.get("assertion_id") or "")
        cascade_proposals, cascade_evidence = _phase5_cascade_for_confirm(proposal)
        if cascade_proposals:
            store.add_proposals(snapshot_id, cascade_proposals, cascade_evidence)
        cascade_ids = [str(item.get("proposal_id") or "") for item in cascade_proposals]
        store.record_proposal_decision(
            snapshot_id, proposal_id, "confirm",
            decided_by=decided_by, reason=reason,
            cascade_emitted=cascade_ids, resulting_assertion_id=resulting_assertion_id, new_status="confirmed",
        )
        store.upsert_decision_memory(
            corpus_id=corpus_id, dataset_version_id=dataset_version_id,
            proposal_type=ptype, signature=signature, decision="confirm",
            canonical_subject=_proposal_subject_summary(proposal),
            decided_by=decided_by, reason=reason, source_snapshot_id=snapshot_id,
        )
        return {"proposal_id": proposal_id, "status": "confirmed", "cascade_emitted": cascade_ids, "resulting_assertion_id": resulting_assertion_id}

    if decision == "reject":
        store.record_proposal_decision(snapshot_id, proposal_id, "reject", decided_by=decided_by, reason=reason, new_status="rejected")
        store.upsert_decision_memory(
            corpus_id=corpus_id, dataset_version_id=dataset_version_id,
            proposal_type=ptype, signature=signature, decision="reject",
            canonical_subject=_proposal_subject_summary(proposal),
            decided_by=decided_by, reason=reason, source_snapshot_id=snapshot_id,
        )
        return {"proposal_id": proposal_id, "status": "rejected", "cascade_emitted": [], "resulting_assertion_id": ""}

    if decision == "defer":
        store.record_proposal_decision(snapshot_id, proposal_id, "defer", decided_by=decided_by, reason=reason)
        return {"proposal_id": proposal_id, "status": str(proposal.get("status") or "proposed"), "cascade_emitted": [], "resulting_assertion_id": ""}

    # undo
    store.record_proposal_decision(snapshot_id, proposal_id, "undo", decided_by=decided_by, reason=reason, new_status="proposed")
    store.delete_decision_memory(dataset_version_id, ptype, signature)
    return {"proposal_id": proposal_id, "status": "proposed", "cascade_emitted": [], "resulting_assertion_id": ""}


def build_phase5_review_payload(store: ArchiveStore, snapshot_id: str, *, query: str = "") -> dict[str, Any]:
    snapshot = store.get_snapshot(snapshot_id)
    proposals = store.list_proposals(snapshot_id)
    evidence_by_proposal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in store.list_proposal_evidence(snapshot_id):
        evidence_by_proposal[str(ev.get("proposal_id") or "")].append(ev)
    entities = store.list_entities(snapshot_id)
    clusters = store.list_clusters(snapshot_id)
    anchors = store.list_anchors(snapshot_id)

    query_l = str(query or "").strip().lower()
    rows: list[dict[str, Any]] = []
    for proposal in proposals:
        row = dict(proposal)
        row["evidence"] = evidence_by_proposal.get(str(proposal.get("proposal_id") or ""), [])
        meta = dict(row.get("metadata") or {})
        blob = " ".join(
            [
                str(row.get("proposal_type") or ""),
                str(row.get("review_bucket") or ""),
                str(row.get("status") or ""),
                str(meta.get("subject_name") or ""),
                str(meta.get("related_name") or ""),
                str(meta.get("source_name") or ""),
                str(meta.get("target_name") or ""),
                " ".join(str(ev.get("description") or "") for ev in row["evidence"]),
            ]
        ).lower()
        if query_l and query_l not in blob:
            continue
        rows.append(row)

    open_rows = [row for row in rows if str(row.get("status") or "") == "proposed"]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in open_rows:
        buckets[str(row.get("review_bucket") or row.get("proposal_type") or "other")].append(row)
    type_counts: dict[str, int] = defaultdict(int)
    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        type_counts[str(row.get("proposal_type") or "")] += 1
        status_counts[str(row.get("status") or "")] += 1

    cluster_rows = [
        {
            "cluster_id": str(cluster.get("cluster_id") or ""),
            "label": str(cluster.get("label") or ""),
            "object_count": len(list(cluster.get("object_ids") or [])),
            "object_ids": list(cluster.get("object_ids") or []),
            "earliest": str(cluster.get("earliest") or ""),
            "latest": str(cluster.get("latest") or ""),
            "dominant_entities": list(cluster.get("dominant_entities") or []),
        }
        for cluster in clusters
    ]

    return {
        "snapshot_id": snapshot_id,
        "phase": str(snapshot.get("phase") or ""),
        "phase_goals": phase_goals(),
        "proposals": rows,
        "review_buckets": [
            {
                "bucket": bucket,
                "count": len(items),
                "proposals": sorted(items, key=lambda item: -float(item.get("confidence") or 0.0)),
            }
            for bucket, items in sorted(buckets.items(), key=lambda item: item[0])
        ],
        "clusters": sorted(cluster_rows, key=lambda item: (-int(item["object_count"]), str(item["label"]).lower())),
        "summary": {
            "proposal_count": len(rows),
            "open_count": len(open_rows),
            "confirmed_count": int(status_counts.get("confirmed", 0)),
            "rejected_count": int(status_counts.get("rejected", 0)),
            "auto_suppressed_count": int(status_counts.get("auto_suppressed", 0)),
            "type_counts": [{"type": key, "count": type_counts[key]} for key in sorted(type_counts)],
            "entity_count": len(entities),
            "cluster_count": len(clusters),
            "unresolved_anchor_count": sum(1 for anchor in anchors if not bool(anchor.get("resolved"))),
            "resolved_anchor_count": sum(1 for anchor in anchors if bool(anchor.get("resolved"))),
        },
    }


def run_phase(
    store: ArchiveStore,
    *,
    corpus_id: str,
    dataset_version_id: str,
    phase: str,
    parent_snapshot_id: str = "",
    provider_config: Optional[dict[str, Any]] = None,
    resolution_overrides: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    phase_name = str(phase or "").strip()
    if phase_name not in PHASE_SEQUENCE:
        raise ValueError(f"unsupported archive phase: {phase_name}")

    phase5_proposals: Optional[list[dict[str, Any]]] = None
    phase5_evidence: list[dict[str, Any]] = []

    if phase_name == "archive_phase0":
        objects = _assemble_phase0(store, dataset_version_id)
        state = {
            "objects": objects,
            "anchors": [],
            "entities": [],
            "mentions": [],
            "relationships": [],
            "clusters": [],
            "assertions": _phase0_assertions(objects),
            "assertion_edits": [],
        }
    else:
        if not parent_snapshot_id:
            latest = store.latest_snapshot(corpus_id, dataset_version_id)
            if latest is None:
                raise ValueError("parent snapshot is required for non-phase0 runs")
            parent_snapshot_id = str(latest.get("snapshot_id") or "")
        parent_state = store.load_snapshot_state(parent_snapshot_id)
        objects = parent_state.get("objects") or []
        anchors = parent_state.get("anchors") or []
        entities = parent_state.get("entities") or []
        mentions = parent_state.get("mentions") or []
        relationships = parent_state.get("relationships") or []
        assertions = [dict(item) for item in (parent_state.get("assertions") or []) if isinstance(item, dict)]
        assertion_edits = [dict(item) for item in (parent_state.get("assertion_edits") or []) if isinstance(item, dict)]
        if phase_name == "archive_phase1":
            objects = _phase1_classify(objects)
            state = {**parent_state, "objects": objects, "assertions": assertions, "assertion_edits": assertion_edits}
        elif phase_name == "archive_phase2":
            objects = _phase2_extract_text(objects, provider_config)
            state = {
                **parent_state,
                "objects": objects,
                "assertions": _phase2_assertions(objects, assertions),
                "assertion_edits": assertion_edits,
            }
        elif phase_name == "archive_phase3":
            state = _phase3_structure(objects)
            state["assertions"] = _phase3_assertions(
                list(state.get("anchors") or []),
                list(state.get("mentions") or []),
                assertions,
            )
            state["assertion_edits"] = assertion_edits
        elif phase_name == "archive_phase4":
            objects, anchors = _phase4_visual(objects, anchors)
            state = {
                "objects": objects,
                "anchors": anchors,
                "entities": entities,
                "mentions": mentions,
                "relationships": relationships,
                "clusters": parent_state.get("clusters") or [],
                "assertions": _phase4_assertions(objects, anchors, assertions),
                "assertion_edits": assertion_edits,
            }
        else:
            phase5_proposals, phase5_evidence, memory_ops = _generate_phase5_proposals(
                dataset_version_id=dataset_version_id,
                objects=objects,
                anchors=anchors,
                entities=entities,
                mentions=mentions,
                relationships=relationships,
                decision_memory=store.list_decision_memory(dataset_version_id),
            )
            state = _phase5_resolve(
                objects,
                anchors,
                entities,
                mentions,
                relationships,
                list(resolution_overrides or []) + memory_ops,
            )
            state["assertions"] = _phase5_assertions(list(state.get("anchors") or []), assertions)
            state["assertion_edits"] = assertion_edits

    snapshot = store.create_snapshot(
        corpus_id=corpus_id,
        dataset_version_id=dataset_version_id,
        phase=phase_name,
        label=phase_label(phase_name),
        parent_snapshot_id=parent_snapshot_id,
        status="complete",
        metrics={
            "object_count": len(state.get("objects") or []),
            "anchor_count": len(state.get("anchors") or []),
            "entity_count": len(state.get("entities") or []),
            "text_block_count": sum(
                len((dict(item.get("metadata") or {}).get("text_blocks") or []))
                for item in (state.get("objects") or [])
                if isinstance(item, dict)
            ),
        },
    )
    store.replace_snapshot_state(str(snapshot.get("snapshot_id") or ""), state)
    if phase_name == "archive_phase5" and phase5_proposals is not None:
        store.replace_proposals(str(snapshot.get("snapshot_id") or ""), phase5_proposals, phase5_evidence)
    return snapshot


def run_archive_job(
    store: ArchiveStore,
    *,
    corpus_id: str,
    dataset_version_id: str,
    phase: str,
    parent_snapshot_id: str = "",
    provider_config: Optional[dict[str, Any]] = None,
    job_id: str = "",
    cell_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    write_run_artifacts: bool = False,
    artifact_root: Optional[Path] = None,
) -> dict[str, Any]:
    phases = list(PHASE_SEQUENCE) if phase == "archive_pipeline" else [phase]
    if phase == "archive_reconcile":
        phases = ["archive_phase5"]
    latest_parent = str(parent_snapshot_id or "")
    phase_snapshots: dict[str, str] = {}
    last_snapshot: Optional[dict[str, Any]] = None
    for index, phase_name in enumerate(phases):
        _emit_cell(cell_callback, index=index, phase=phase_name, status="running", output=f"Running {phase_label(phase_name)}")
        run_id = store.begin_run(
            corpus_id=corpus_id,
            dataset_version_id=dataset_version_id,
            snapshot_id="",
            phase=phase_name,
            job_id=str(job_id or ""),
            backbone_version=archive_backbone_version(),
        )
        try:
            snapshot = run_phase(
                store,
                corpus_id=corpus_id,
                dataset_version_id=dataset_version_id,
                phase=phase_name,
                parent_snapshot_id=latest_parent,
                provider_config=provider_config,
                resolution_overrides=store.list_resolution_overrides(latest_parent) if latest_parent and phase_name == "archive_phase5" else [],
            )
            phase_snapshots[phase_name] = str(snapshot.get("snapshot_id") or "")
            last_snapshot = snapshot
            latest_parent = str(snapshot.get("snapshot_id") or "")
            metrics = dict(snapshot.get("metrics") or {})
            metrics["materialized_snapshot_id"] = latest_parent
            store.finish_run(run_id, status="complete", metrics=metrics)
            _emit_cell(
                cell_callback,
                index=index,
                phase=phase_name,
                status="done",
                output=f"{phase_label(phase_name)} complete ({metrics.get('object_count', 0)} objects)",
            )
        except Exception as exc:
            store.finish_run(run_id, status="error", metrics={}, error=str(exc))
            _emit_cell(cell_callback, index=index, phase=phase_name, status="error", output=str(exc))
            raise

    if last_snapshot is None:
        raise ValueError("archive job produced no snapshots")

    result = {
        "scenario": str(job_id or "archive"),
        "summary": f"{phase_label(phases[-1])} complete",
        "error": "",
        "artifact_policy": "path_only",
        "backbone_type": "archival_ingestion",
        "corpus_id": corpus_id,
        "dataset_version_id": dataset_version_id,
        "snapshot_id": str(last_snapshot.get("snapshot_id") or ""),
        "phase": phases[-1],
        "phase_snapshots": phase_snapshots,
        "result_path": store.get_dataset_version(dataset_version_id)["raw_root"],
        "metrics": dict(last_snapshot.get("metrics") or {}),
    }
    if write_run_artifacts and artifact_root is not None:
        artifact_root.mkdir(parents=True, exist_ok=True)
        metrics_path = artifact_root / "metrics.json"
        metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
        result["result_path"] = str(artifact_root)
    return result


def build_phase0_review_payload(
    store: ArchiveStore,
    corpus_id: str,
    dataset_version_id: str,
    *,
    query: str = "",
) -> dict[str, Any]:
    latest_phase0 = next(
        (
            item for item in store.list_snapshots(corpus_id, dataset_version_id)
            if str(item.get("phase") or "") == "archive_phase0"
        ),
        None,
    )
    if latest_phase0 is not None:
        state = store.load_snapshot_state(str(latest_phase0.get("snapshot_id") or ""))
        objects = list(state.get("objects") or [])
        assertions = list(state.get("assertions") or [])
        snapshot_id = str(latest_phase0.get("snapshot_id") or "")
    else:
        objects = _assemble_phase0(store, dataset_version_id)
        assertions = _phase0_assertions(objects)
        snapshot_id = ""

    store.refresh_file_states(dataset_version_id, force=True)
    file_lookup = {item["file_id"]: item for item in store.list_files(dataset_version_id)}
    query_l = str(query or "").strip().lower()
    assertion_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "edited": 0})
    for assertion in assertions:
        object_id = str(assertion.get("object_id") or "")
        assertion_counts[object_id]["total"] += 1
        if list(assertion.get("edits") or []):
            assertion_counts[object_id]["edited"] += 1

    methods: dict[str, list[dict[str, Any]]] = defaultdict(list)
    review_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    object_rows: list[dict[str, Any]] = []
    for obj in objects:
        title = str(obj.get("title") or "")
        object_id = str(obj.get("object_id") or "")
        files = []
        for ref in obj.get("files") or []:
            if not isinstance(ref, dict):
                continue
            enriched = dict(ref)
            file_meta = file_lookup.get(str(ref.get("file_id") or ""))
            if isinstance(file_meta, dict):
                enriched["file"] = dict(file_meta)
            files.append(enriched)
        missing_file_count = sum(1 for ref in files if isinstance(ref, dict) and not bool((ref.get("file") or {}).get("exists")))
        conf = float(obj.get("assembly_confidence") or 0.0)
        review_bucket = "high_confidence" if conf >= 0.9 and missing_file_count == 0 else "needs_review" if conf >= 0.55 else "retained"
        classification = _classification_record(obj)
        item = {
            "object_id": object_id,
            "object_key": str(obj.get("object_key") or ""),
            "title": title,
            "object_type": str(obj.get("object_type") or ""),
            "assembly_method": str(obj.get("assembly_method") or ""),
            "assembly_confidence": conf,
            "media_family": str(obj.get("media_family") or ""),
            "content_complexity": str(obj.get("content_complexity") or ""),
            "status": str(obj.get("status") or ""),
            "file_count": len(files),
            "missing_file_count": missing_file_count,
            "edited_assertion_count": int(assertion_counts[object_id]["edited"]),
            "assertion_count": int(assertion_counts[object_id]["total"]),
            "review_bucket": review_bucket,
            "classification": classification,
            "routing_labels": list(classification.get("routes") or []),
            "route_summary": ", ".join(list(classification.get("routes") or [])[:3]),
            "files": files,
            "metadata": dict(obj.get("metadata") or {}),
        }
        blob = " ".join(
            [
                title,
                item["object_type"],
                item["assembly_method"],
                item["media_family"],
                str(classification.get("era_bucket") or ""),
                str(classification.get("content_complexity") or ""),
                " ".join(item["routing_labels"]),
            ]
        ).lower()
        if query_l and query_l not in blob:
            continue
        methods[item["assembly_method"]].append(item)
        review_buckets[review_bucket].append(item)
        object_rows.append(item)

    versions = store.get_dataset_version(dataset_version_id)
    return {
        "corpus_id": corpus_id,
        "dataset_version_id": dataset_version_id,
        "snapshot_id": snapshot_id,
        "phase": "archive_phase0",
        "phase_goals": phase_goals(),
        "methods": [
            {"method": method, "count": len(rows), "objects": rows}
            for method, rows in sorted(methods.items(), key=lambda item: item[0])
        ],
        "review_buckets": [
            {"bucket": bucket, "count": len(rows), "objects": rows}
            for bucket, rows in sorted(review_buckets.items(), key=lambda item: item[0])
        ],
        "objects": sorted(object_rows, key=lambda item: (item["review_bucket"], item["title"].lower())),
        "classification_summary": _summarize_classification_rows(object_rows),
        "summary": {
            "object_count": len(object_rows),
            "file_count": int(versions.get("file_count") or 0),
            "processable_count": int(versions.get("processable_count") or 0),
            "missing_file_count": sum(int(item["missing_file_count"]) for item in object_rows),
            "needs_review_count": len(review_buckets.get("needs_review", [])),
            "high_confidence_count": len(review_buckets.get("high_confidence", [])),
            "retained_count": len(review_buckets.get("retained", [])),
            "override_count": len(store.list_assembly_overrides(dataset_version_id)),
        },
    }


def build_phase2_review_payload(store: ArchiveStore, snapshot_id: str, *, query: str = "") -> dict[str, Any]:
    state = store.load_snapshot_state(snapshot_id)
    snapshot = store.get_snapshot(snapshot_id)
    objects = list(state.get("objects") or [])
    assertions = list(state.get("assertions") or [])
    dataset_version_id = str(snapshot.get("dataset_version_id") or "")
    if dataset_version_id:
        store.refresh_file_states(dataset_version_id)
        objects = store.list_objects(snapshot_id)

    assertion_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "edited": 0})
    for assertion in assertions:
        if not isinstance(assertion, dict):
            continue
        object_id = str(assertion.get("object_id") or "")
        assertion_counts[object_id]["total"] += 1
        if list(assertion.get("edits") or []):
            assertion_counts[object_id]["edited"] += 1

    query_l = str(query or "").strip().lower()
    rows: list[dict[str, Any]] = []
    status_groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    provider_object_counts: dict[str, set[str]] = defaultdict(set)
    block_kind_object_counts: dict[str, set[str]] = defaultdict(set)
    segmentation_totals = {"file_segments": 0, "page_segments": 0, "region_segments": 0, "segment_count": 0}
    capability_unavailable_objects = 0
    failed_objects = 0

    for obj in objects:
        object_id = str(obj.get("object_id") or "")
        row = _phase2_review_row(
            obj,
            assertion_total=int(assertion_counts[object_id]["total"]),
            assertion_edited=int(assertion_counts[object_id]["edited"]),
        )
        extraction_summary = dict(row.get("extraction_summary") or {})
        classification = dict(row.get("classification") or {})
        status = str(extraction_summary.get("status") or "extraction_ready")
        provider_summary = str(extraction_summary.get("provider_summary") or "none")
        blob = " ".join(
            [
                str(row.get("title") or ""),
                str(row.get("object_type") or ""),
                str(row.get("media_family") or ""),
                str(classification.get("era_bucket") or ""),
                str(classification.get("content_complexity") or ""),
                " ".join(str(route) for route in (classification.get("routes") or [])),
                provider_summary,
                status,
            ]
        ).lower()
        if query_l and query_l not in blob:
            continue
        rows.append(row)
        status_groups[status][provider_summary].append(row)
        for provider_name, count in dict(extraction_summary.get("provider_counts") or {}).items():
            if int(count or 0) > 0:
                provider_object_counts[str(provider_name or "unknown")].add(object_id)
        for block_kind, count in dict(extraction_summary.get("block_kind_counts") or {}).items():
            if int(count or 0) > 0:
                block_kind_object_counts[str(block_kind or "unknown")].add(object_id)
        segmentation_summary = dict(row.get("segmentation_summary") or {})
        for key in segmentation_totals:
            segmentation_totals[key] += int(segmentation_summary.get(key) or 0)
        capability_counts = dict(extraction_summary.get("capability_counts") or {})
        if int(capability_counts.get("capability_unavailable") or 0) > 0:
            capability_unavailable_objects += 1
        if int(capability_counts.get("failed") or 0) > 0:
            failed_objects += 1

    grouped = []
    for status, provider_map in sorted(status_groups.items(), key=lambda item: item[0]):
        grouped.append(
            {
                "status": status,
                "count": sum(len(items) for items in provider_map.values()),
                "providers": [
                    {
                        "provider_summary": provider_summary,
                        "count": len(items),
                        "objects": sorted(items, key=lambda item: str(item.get("title") or "").lower()),
                    }
                    for provider_summary, items in sorted(provider_map.items(), key=lambda item: item[0])
                ],
            }
        )

    return {
        "snapshot_id": snapshot_id,
        "phase": str(snapshot.get("phase") or ""),
        "phase_goals": phase_goals(),
        "objects": sorted(rows, key=lambda item: (str((item.get("extraction_summary") or {}).get("status") or ""), str(item.get("title") or "").lower())),
        "groups": grouped,
        "classification_summary": _summarize_classification_rows(rows),
        "summary": {
            "object_count": len(rows),
            "extracted_count": sum(1 for row in rows if str((row.get("extraction_summary") or {}).get("status") or "") == "extracted"),
            "extraction_ready_count": sum(1 for row in rows if str((row.get("extraction_summary") or {}).get("status") or "") == "extraction_ready"),
            "capability_unavailable_count": capability_unavailable_objects,
            "failed_count": failed_objects,
            "printed_ocr_count": len(block_kind_object_counts.get("printed_ocr", set())),
            "handwriting_ocr_count": len(block_kind_object_counts.get("handwriting_ocr", set())),
            "audio_transcription_count": len(block_kind_object_counts.get("audio_transcript", set())),
            "text_block_count": sum(int((row.get("extraction_summary") or {}).get("text_block_count") or 0) for row in rows),
            "missing_file_count": sum(int(row.get("missing_file_count") or 0) for row in rows),
            "edited_assertion_count": sum(int(row.get("edited_assertion_count") or 0) for row in rows),
            "provider_counts": {provider: len(object_ids) for provider, object_ids in sorted(provider_object_counts.items())},
            "segmentation": segmentation_totals,
        },
    }


def _phase3_object_row(
    obj: dict[str, Any],
    *,
    anchors: list[dict[str, Any]],
    mentions: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    entities_by_id: dict[str, dict[str, Any]],
    assertion_total: int = 0,
    assertion_edited: int = 0,
) -> dict[str, Any]:
    object_id = str(obj.get("object_id") or "")
    files = [dict(ref) for ref in (obj.get("files") or []) if isinstance(ref, dict)]
    missing_file_count = sum(
        1
        for ref in files
        if not bool((ref.get("file") or {}).get("exists"))
    )
    object_anchors = [dict(item) for item in anchors if str(item.get("object_id") or "") == object_id]
    object_mentions = [dict(item) for item in mentions if str(item.get("object_id") or "") == object_id]
    object_relationships = [dict(item) for item in relationships if str(item.get("object_id") or "") == object_id]
    entity_names = []
    for mention in object_mentions:
        entity = entities_by_id.get(str(mention.get("entity_id") or ""))
        if entity is not None:
            entity_names.append(str(entity.get("canonical_name") or mention.get("mention_text") or ""))
    unresolved_anchors = [item for item in object_anchors if not bool(item.get("resolved"))]
    return {
        "object_id": object_id,
        "title": str(obj.get("title") or ""),
        "object_type": str(obj.get("object_type") or ""),
        "era_bucket": str(obj.get("era_bucket") or ""),
        "earliest": str(obj.get("earliest") or ""),
        "latest": str(obj.get("latest") or ""),
        "media_family": str(obj.get("media_family") or ""),
        "content_complexity": str(obj.get("content_complexity") or ""),
        "classification": _classification_record(obj),
        "anchor_count": len(object_anchors),
        "resolved_anchor_count": len(object_anchors) - len(unresolved_anchors),
        "unresolved_anchor_count": len(unresolved_anchors),
        "mention_count": len(object_mentions),
        "relationship_count": len(object_relationships),
        "entity_names": sorted({name for name in entity_names if name}),
        "anchor_types": sorted({str(item.get("type") or "unknown") for item in object_anchors}),
        "relationship_types": sorted({str(item.get("relationship_type") or "unknown") for item in object_relationships}),
        "missing_file_count": missing_file_count,
        "edited_assertion_count": int(assertion_edited),
        "assertion_count": int(assertion_total),
        "review_state": "needs_resolution" if unresolved_anchors else ("structured" if object_anchors or object_mentions or object_relationships else "no_structure"),
    }


def build_phase3_review_payload(store: ArchiveStore, snapshot_id: str, *, query: str = "") -> dict[str, Any]:
    state = store.load_snapshot_state(snapshot_id)
    snapshot = store.get_snapshot(snapshot_id)
    objects = list(state.get("objects") or [])
    anchors = [dict(item) for item in (state.get("anchors") or []) if isinstance(item, dict)]
    entities = [dict(item) for item in (state.get("entities") or []) if isinstance(item, dict)]
    mentions = [dict(item) for item in (state.get("mentions") or []) if isinstance(item, dict)]
    relationships = [dict(item) for item in (state.get("relationships") or []) if isinstance(item, dict)]
    assertions = [dict(item) for item in (state.get("assertions") or []) if isinstance(item, dict)]
    dataset_version_id = str(snapshot.get("dataset_version_id") or "")
    if dataset_version_id:
        store.refresh_file_states(dataset_version_id)
        objects = store.list_objects(snapshot_id)
        anchors = store.list_anchors(snapshot_id)
        entities = store.list_entities(snapshot_id)
        mentions = store.list_mentions(snapshot_id)
        relationships = store.list_relationships(snapshot_id)
        assertions = store.list_assertions(snapshot_id)

    entities_by_id = {str(item.get("entity_id") or ""): item for item in entities}
    objects_by_id = {str(item.get("object_id") or ""): item for item in objects if isinstance(item, dict)}
    assertion_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "edited": 0})
    for assertion in assertions:
        object_id = str(assertion.get("object_id") or "")
        assertion_counts[object_id]["total"] += 1
        if list(assertion.get("edits") or []):
            assertion_counts[object_id]["edited"] += 1

    query_l = str(query or "").strip().lower()
    rows: list[dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        object_id = str(obj.get("object_id") or "")
        row = _phase3_object_row(
            obj,
            anchors=anchors,
            mentions=mentions,
            relationships=relationships,
            entities_by_id=entities_by_id,
            assertion_total=int(assertion_counts[object_id]["total"]),
            assertion_edited=int(assertion_counts[object_id]["edited"]),
        )
        blob = " ".join(
            [
                str(row.get("title") or ""),
                str(row.get("object_type") or ""),
                str(row.get("era_bucket") or ""),
                str(row.get("review_state") or ""),
                " ".join(str(item) for item in row.get("entity_names") or []),
                " ".join(str(item) for item in row.get("anchor_types") or []),
                " ".join(str(item) for item in row.get("relationship_types") or []),
            ]
        ).lower()
        if query_l and query_l not in blob:
            continue
        rows.append(row)

    anchor_rows: list[dict[str, Any]] = []
    for anchor in anchors:
        object_id = str(anchor.get("object_id") or "")
        obj = objects_by_id.get(object_id, {})
        row = {
            **dict(anchor),
            "title": str(obj.get("title") or ""),
            "object_type": str(obj.get("object_type") or ""),
            "review_state": "resolved" if bool(anchor.get("resolved")) else "needs_resolution",
        }
        blob = " ".join(
            [
                str(row.get("title") or ""),
                str(row.get("type") or ""),
                str(row.get("raw_expression") or ""),
                str(row.get("earliest") or ""),
                str(row.get("latest") or ""),
                str(row.get("resolution_requires") or ""),
            ]
        ).lower()
        if query_l and query_l not in blob:
            continue
        anchor_rows.append(row)

    mention_rows: list[dict[str, Any]] = []
    for mention in mentions:
        entity = entities_by_id.get(str(mention.get("entity_id") or ""), {})
        obj = objects_by_id.get(str(mention.get("object_id") or ""), {})
        row = {
            **dict(mention),
            "title": str(obj.get("title") or ""),
            "object_type": str(obj.get("object_type") or ""),
            "canonical_name": str(entity.get("canonical_name") or mention.get("mention_text") or ""),
            "entity_type": str(entity.get("entity_type") or "unknown"),
        }
        blob = " ".join(
            [
                str(row.get("title") or ""),
                str(row.get("canonical_name") or ""),
                str(row.get("entity_type") or ""),
                str(row.get("mention_text") or ""),
            ]
        ).lower()
        if query_l and query_l not in blob:
            continue
        mention_rows.append(row)

    relationship_rows: list[dict[str, Any]] = []
    for rel in relationships:
        obj = objects_by_id.get(str(rel.get("object_id") or ""), {})
        source_entity = entities_by_id.get(str(rel.get("source_entity_id") or ""), {})
        target_entity = entities_by_id.get(str(rel.get("target_entity_id") or ""), {})
        row = {
            **dict(rel),
            "title": str(obj.get("title") or ""),
            "source_entity_name": str(source_entity.get("canonical_name") or rel.get("source_entity_id") or ""),
            "target_entity_name": str(target_entity.get("canonical_name") or rel.get("target_entity_id") or ""),
        }
        blob = " ".join(
            [
                str(row.get("title") or ""),
                str(row.get("relationship_type") or ""),
                str(row.get("source_entity_name") or ""),
                str(row.get("target_entity_name") or ""),
            ]
        ).lower()
        if query_l and query_l not in blob:
            continue
        relationship_rows.append(row)

    anchor_groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in anchor_rows:
        anchor_groups[str(row.get("review_state") or "needs_resolution")][str(row.get("type") or "unknown")].append(row)
    entity_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mention_rows:
        entity_groups[str(row.get("entity_type") or "unknown")].append(row)
    relationship_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in relationship_rows:
        relationship_groups[str(row.get("relationship_type") or "unknown")].append(row)

    return {
        "snapshot_id": snapshot_id,
        "phase": str(snapshot.get("phase") or ""),
        "phase_goals": phase_goals(),
        "objects": sorted(rows, key=lambda item: (str(item.get("review_state") or ""), str(item.get("title") or "").lower())),
        "anchors": sorted(anchor_rows, key=lambda item: (str(item.get("review_state") or ""), str(item.get("type") or ""), str(item.get("title") or "").lower())),
        "mentions": sorted(mention_rows, key=lambda item: (str(item.get("entity_type") or ""), str(item.get("canonical_name") or "").lower())),
        "relationships": sorted(relationship_rows, key=lambda item: (str(item.get("relationship_type") or ""), str(item.get("title") or "").lower())),
        "anchor_groups": [
            {
                "review_state": review_state,
                "count": sum(len(items) for items in type_map.values()),
                "types": [
                    {
                        "type": anchor_type,
                        "count": len(items),
                        "anchors": sorted(items, key=lambda item: str(item.get("title") or "").lower()),
                    }
                    for anchor_type, items in sorted(type_map.items(), key=lambda item: item[0])
                ],
            }
            for review_state, type_map in sorted(anchor_groups.items(), key=lambda item: item[0])
        ],
        "entity_groups": [
            {
                "entity_type": entity_type,
                "count": len(items),
                "mentions": sorted(items, key=lambda item: (str(item.get("canonical_name") or "").lower(), str(item.get("title") or "").lower())),
            }
            for entity_type, items in sorted(entity_groups.items(), key=lambda item: item[0])
        ],
        "relationship_groups": [
            {
                "relationship_type": rel_type,
                "count": len(items),
                "relationships": sorted(items, key=lambda item: str(item.get("title") or "").lower()),
            }
            for rel_type, items in sorted(relationship_groups.items(), key=lambda item: item[0])
        ],
        "classification_summary": _summarize_classification_rows(rows),
        "summary": {
            "object_count": len(rows),
            "anchor_count": len(anchor_rows),
            "resolved_anchor_count": sum(1 for item in anchor_rows if bool(item.get("resolved"))),
            "unresolved_anchor_count": sum(1 for item in anchor_rows if not bool(item.get("resolved"))),
            "entity_count": len(entities),
            "mention_count": len(mention_rows),
            "relationship_count": len(relationship_rows),
            "structured_object_count": sum(1 for item in rows if str(item.get("review_state") or "") != "no_structure"),
            "missing_file_count": sum(int(item.get("missing_file_count") or 0) for item in rows),
            "edited_assertion_count": sum(int(item.get("edited_assertion_count") or 0) for item in rows),
        },
    }


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def build_phase4_review_payload(store: ArchiveStore, snapshot_id: str, *, query: str = "") -> dict[str, Any]:
    """Surface the visual extraction the engine already computes (object metadata['visual']).

    Read-only: Phase 4 is enrichment/fallback, not object identity. Returns scene-class
    groups, visual-era cards, visual-estimate anchors, and color-histogram nearest neighbors.
    """
    snapshot = store.get_snapshot(snapshot_id)
    objects = store.list_objects(snapshot_id)
    anchors = store.list_anchors(snapshot_id)
    query_l = str(query or "").strip().lower()

    embeddings: dict[str, list[float]] = {}
    visual_rows: list[dict[str, Any]] = []
    image_object_count = 0
    for obj in objects:
        if str(obj.get("media_family") or "") == "image":
            image_object_count += 1
        visual = dict(dict(obj.get("metadata") or {}).get("visual") or {})
        if not visual:
            continue
        oid = str(obj.get("object_id") or "")
        missing = sum(
            1
            for ref in (obj.get("files") or [])
            if isinstance(ref, dict) and isinstance(ref.get("file"), dict) and not bool(ref["file"].get("exists", True))
        )
        emb = [float(x) for x in (visual.get("embedding") or []) if isinstance(x, (int, float))]
        if emb:
            embeddings[oid] = emb
        row = {
            "object_id": oid,
            "title": str(obj.get("title") or ""),
            "object_type": str(obj.get("object_type") or ""),
            "media_family": str(obj.get("media_family") or ""),
            "era_bucket": str(obj.get("era_bucket") or ""),
            "earliest": str(obj.get("earliest") or ""),
            "latest": str(obj.get("latest") or ""),
            "scene_class": str(visual.get("scene_class") or "unknown"),
            "mean_luma": round(float(visual.get("mean_luma") or 0.0), 3),
            "embedding_dims": len(emb),
            "missing_file_count": int(missing),
        }
        blob = " ".join([row["title"], row["object_type"], row["scene_class"], row["era_bucket"]]).lower()
        if query_l and query_l not in blob:
            continue
        visual_rows.append(row)

    title_by_id = {row["object_id"]: row["title"] for row in visual_rows}
    visible_ids = set(title_by_id.keys())

    scene_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    era_cards: dict[str, dict[str, Any]] = {}
    for row in visual_rows:
        scene_groups[row["scene_class"]].append(row)
        key = row["era_bucket"] or "undated"
        card = era_cards.setdefault(key, {"era_bucket": key, "object_count": 0, "scene_classes": defaultdict(int), "object_ids": []})
        card["object_count"] += 1
        card["scene_classes"][row["scene_class"]] += 1
        card["object_ids"].append(row["object_id"])

    visual_anchors: list[dict[str, Any]] = []
    for anchor in anchors:
        if str(anchor.get("type") or "") != "visual_estimate":
            continue
        oid = str(anchor.get("object_id") or "")
        if oid not in visible_ids:
            continue
        row = dict(anchor)
        row["title"] = title_by_id.get(oid, oid)
        visual_anchors.append(row)

    similarities: list[dict[str, Any]] = []
    embedded_ids = sorted(embeddings.keys())
    for oid in embedded_ids:
        scored = [
            (other, _cosine_similarity(embeddings[oid], embeddings[other]))
            for other in embedded_ids
            if other != oid
        ]
        scored = [item for item in scored if item[1] > 0.0]
        scored.sort(key=lambda item: -item[1])
        neighbors = [
            {"object_id": other, "title": title_by_id.get(other, other), "score": round(score, 4)}
            for other, score in scored[:5]
        ]
        if neighbors:
            similarities.append({"object_id": oid, "title": title_by_id.get(oid, oid), "neighbors": neighbors})

    return {
        "snapshot_id": snapshot_id,
        "phase": str(snapshot.get("phase") or ""),
        "phase_goals": phase_goals(),
        "objects": sorted(visual_rows, key=lambda item: (item["scene_class"], item["title"].lower())),
        "scene_groups": [
            {"scene_class": scene, "count": len(items), "objects": sorted(items, key=lambda item: item["title"].lower())}
            for scene, items in sorted(scene_groups.items(), key=lambda item: item[0])
        ],
        "era_cards": [
            {
                "era_bucket": card["era_bucket"],
                "object_count": card["object_count"],
                "scene_classes": [{"scene_class": k, "count": v} for k, v in sorted(card["scene_classes"].items())],
                "object_ids": card["object_ids"],
            }
            for card in sorted(era_cards.values(), key=lambda c: str(c["era_bucket"]))
        ],
        "visual_anchors": sorted(visual_anchors, key=lambda a: (str(a.get("earliest") or ""), str(a.get("object_id") or ""))),
        "similarities": similarities,
        "summary": {
            "visual_object_count": len(visual_rows),
            "image_object_count": image_object_count,
            "scene_class_count": len(scene_groups),
            "visual_anchor_count": len(visual_anchors),
            "embedded_object_count": len(embeddings),
        },
    }


def build_timeline_payload(store: ArchiveStore, snapshot_id: str, *, query: str = "", unresolved_only: bool = False) -> dict[str, Any]:
    state = store.load_snapshot_state(snapshot_id)
    objects = list(state.get("objects") or [])
    anchors = list(state.get("anchors") or [])
    mentions = list(state.get("mentions") or [])
    assertions = list(state.get("assertions") or [])
    snapshot = store.get_snapshot(snapshot_id)
    dataset_version_id = str(snapshot.get("dataset_version_id") or "")
    if dataset_version_id:
        store.refresh_file_states(dataset_version_id)
        objects = store.list_objects(snapshot_id)
    entities = {str(item.get("entity_id") or ""): item for item in (state.get("entities") or []) if isinstance(item, dict)}
    clusters = list(state.get("clusters") or [])
    query_l = str(query or "").strip().lower()
    anchors_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mention_names: dict[str, list[str]] = defaultdict(list)
    assertion_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "edited": 0})
    for anchor in anchors:
        anchors_by_object[str(anchor.get("object_id") or "")].append(anchor)
    for mention in mentions:
        entity = entities.get(str(mention.get("entity_id") or ""))
        if entity is not None:
            mention_names[str(mention.get("object_id") or "")].append(str(entity.get("canonical_name") or ""))
    for assertion in assertions:
        object_id = str(assertion.get("object_id") or "")
        assertion_counts[object_id]["total"] += 1
        if list(assertion.get("edits") or []):
            assertion_counts[object_id]["edited"] += 1

    items = []
    holding_pen = []
    for obj in objects:
        object_id = str(obj.get("object_id") or "")
        title = str(obj.get("title") or "")
        files = list(obj.get("files") or [])
        missing_file_count = sum(1 for ref in files if isinstance(ref, dict) and not bool((ref.get("file") or {}).get("exists")))
        classification = _classification_record(obj)
        item = {
            "object_id": object_id,
            "title": title,
            "object_type": str(obj.get("object_type") or ""),
            "era_bucket": str(obj.get("era_bucket") or ""),
            "earliest": str(obj.get("earliest") or ""),
            "latest": str(obj.get("latest") or ""),
            "media_family": str(obj.get("media_family") or ""),
            "content_complexity": str(obj.get("content_complexity") or ""),
            "unresolved": not bool(obj.get("earliest")) and not bool(obj.get("latest")),
            "entity_names": sorted(set(mention_names.get(object_id, []))),
            "cluster_labels": [
                str(cluster.get("label") or "")
                for cluster in clusters
                if object_id in {str(item) for item in (cluster.get("object_ids") or [])}
            ],
            "assembly_method": str(obj.get("assembly_method") or ""),
            "assembly_confidence": float(obj.get("assembly_confidence") or 0.0),
            "status": str(obj.get("status") or ""),
            "file_count": len(files),
            "missing_file_count": missing_file_count,
            "edited_assertion_count": int(assertion_counts[object_id]["edited"]),
            "assertion_count": int(assertion_counts[object_id]["total"]),
            "review_bucket": "high_confidence"
            if float(obj.get("assembly_confidence") or 0.0) >= 0.9 and missing_file_count == 0
            else ("needs_review" if float(obj.get("assembly_confidence") or 0.0) >= 0.55 else "retained"),
            "classification": classification,
            "routing_labels": list(classification.get("routes") or []),
            "route_summary": ", ".join(list(classification.get("routes") or [])[:3]),
        }
        blob = " ".join(
            [
                title,
                item["object_type"],
                item["era_bucket"],
                str(classification.get("content_complexity") or ""),
                " ".join(item["entity_names"]),
                " ".join(item["routing_labels"]),
            ]
        ).lower()
        if query_l and query_l not in blob:
            continue
        if unresolved_only and not item["unresolved"]:
            continue
        if item["unresolved"]:
            holding_pen.append(item)
        else:
            items.append(item)
    items.sort(key=lambda item: (str(item.get("earliest") or "9999-12-31"), str(item.get("title") or "").lower()))
    density: dict[str, int] = defaultdict(int)
    for item in items:
        year = str(item.get("earliest") or "")[:4]
        if year:
            density[year] += 1
    return {
        "snapshot_id": snapshot_id,
        "phase": str(snapshot.get("phase") or ""),
        "phase_goals": phase_goals(),
        "items": items,
        "holding_pen": holding_pen,
        "classification_summary": _summarize_classification_rows(items + holding_pen),
        "density": [{"bucket": year, "count": density[year]} for year in sorted(density)],
        "summary": {
            "object_count": len(objects),
            "timeline_count": len(items),
            "holding_pen_count": len(holding_pen),
            "anchor_count": len(anchors),
            "entity_count": len(state.get("entities") or []),
            "assertion_count": len(assertions),
            "edited_assertion_count": sum(int(item["edited_assertion_count"]) for item in items + holding_pen),
            "missing_file_count": sum(int(item["missing_file_count"]) for item in items + holding_pen),
            "classified_count": len(items) + len(holding_pen),
        },
    }
