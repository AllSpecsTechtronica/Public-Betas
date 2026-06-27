from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from base64 import b64encode
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from ..config import (
    AI_HTTP_TIMEOUT_SECONDS,
    ANTHROPIC_API_KEY,
    DEFAULT_MODEL_PATH,
    GALLERY_DB_PATH,
    INSIGHT_ANTHROPIC_MODEL,
    INSIGHT_OLLAMA_MODEL,
    INSIGHT_OLLAMA_URL,
    INSIGHT_OPENAI_MODEL,
    OPENAI_API_KEY,
    RECOGNITION_AUTO,
    RECOGNITION_MARGIN,
    RECOGNITION_THRESHOLD,
    RECOGNITION_TOP_K,
    RuntimeConfig,
    normalize_detection_mode,
    model_choice_name,
    normalize_fps,
    normalize_image_size,
    is_face_detection_model_path,
    resolve_model_path,
)
from ..export_apple import ensure_runtime_model
from .detector import YoloDetector, create_detector
from .action_runner import SingleFlightActionRunner
from .frame_source import FrameSource
from .gallery_db import GalleryDB, GalleryStats
from .perception import InsightPerceptionEngine
from .recognition_worker import RecognitionWorker
from .rt_pipeline import InsightRtPipeline
from .state_store import JsonStateStore
from .types import CapabilityReport, OperatingMode, SubsystemHealth
from .ui_adapter import SessionUiAdapter


class NullGalleryDB:
    def __init__(self) -> None:
        self.matrix = None
        self.matrix_labels: list[str] = []
        self.matrix_groups: list[str] = []
        self.matrix_sources: list[str] = []
        self.profile_matrix = None
        self.profile_labels: list[str] = []
        self.profile_groups: list[str] = []
        self.profile_sources: list[str] = []
        self.profile_sample_counts: list[int] = []

    @property
    def has_gallery(self) -> bool:
        return False

    def ensure_face_backend(self) -> bool:
        return False

    @property
    def face_backend_error(self) -> str:
        return "Gallery database unavailable"

    def ensure_similarity_backend(self) -> bool:
        return False

    @property
    def similarity_backend_error(self) -> str:
        return "Similarity search unavailable"

    def build_matrix(self) -> int:
        return 0

    def get_stats(self) -> GalleryStats:
        return GalleryStats(identity_count=0, image_count=0, group_names=[], last_rebuild=0.0)

    def list_identities(self, group_filter: str = "") -> list[Any]:
        return []

    def get_identity_images(self, identity_name: str) -> list[str]:
        return []

    def list_similarity_items(self) -> list[Any]:
        return []

    def get_similarity_item_path(self, item_id: int) -> str:
        return ""

    def find_similar_items(self, item_id: int, top_k: int = 12) -> list[dict[str, Any]]:
        return []

    def ingest_folder(self, folder: Path, identity_name: str, group_name: str = "", progress_cb=None) -> tuple[int, list[str]]:
        return 0, ["Gallery database unavailable"]

    def ingest_single(self, image_path: Path, identity_name: str, group_name: str = "") -> tuple[bool, str]:
        return False, "Gallery database unavailable"

    def ingest_similarity_image(self, image_path: Path, batch_label: str = "") -> tuple[bool, str]:
        return False, "Gallery database unavailable"

    def ingest_similarity_folder(self, folder: Path, progress_cb=None) -> tuple[int, list[str]]:
        return 0, ["Gallery database unavailable"]

    def ingest_bgr(self, bgr: Any, identity_name: str, group_name: str = "", source_label: str = "crop") -> tuple[bool, str]:
        return False, "Gallery database unavailable"

    def extract_face(self, bgr: Any):
        return None

    def delete_identity(self, identity_name: str) -> int:
        return 0

    def delete_similarity_item(self, item_id: int) -> int:
        return 0

    def rename_identity(self, old_name: str, new_name: str) -> None:
        return

    def close(self) -> None:
        return


class InsightSupervisor:
    """Supervises the runtime pipeline and degraded-mode recovery behavior."""

    def __init__(self, config: RuntimeConfig, ui: SessionUiAdapter, *, defer_boot: bool = False) -> None:
        self.config = config
        self.ui = ui
        self.closed = False
        self._booted = False
        self._lock = threading.RLock()
        self._snapshot_store = JsonStateStore(config.state_dir)
        self._snapshot = self._snapshot_store.load_snapshot()
        self._saved_settings = dict(self._snapshot.get("settings") or {})
        self._recovery_log = self._snapshot_store.load_recovery_log(limit=40)
        self._next_probe_ts: dict[str, float] = {}
        self._probe_backoff_index: dict[str, int] = {}
        self._last_snapshot_ts = 0.0
        self._recognition_error_streak = 0
        self._recovery_thread: Optional[threading.Thread] = None

        self.operating_mode: OperatingMode = "boot"
        self.capabilities = CapabilityReport()
        self.subsystem_health = {
            "state_store": SubsystemHealth("state_store", "starting", impact="Persistence unavailable"),
            "gallery_db": SubsystemHealth("gallery_db", "starting", impact="Recognition gallery unavailable"),
            "recognizer": SubsystemHealth("recognizer", "starting", impact="Identity recognition unavailable"),
            "detector": SubsystemHealth("detector", "starting", impact="Automated CV unavailable"),
            "frame_source": SubsystemHealth("frame_source", "starting", impact="No live video source"),
            "local_ai": SubsystemHealth("local_ai", "starting", impact="Local ROI AI unavailable"),
            "cloud_ai": SubsystemHealth("cloud_ai", "starting", impact="Cloud AI unavailable"),
        }

        self._apply_snapshot_defaults()

        self.gallery: GalleryDB | NullGalleryDB = NullGalleryDB()
        self.recognition_worker: Optional[RecognitionWorker] = None
        self.recognition_auto = RECOGNITION_AUTO
        self.detector: Optional[Any] = None
        self.frame_source: Optional[FrameSource] = None
        self.perception: Optional[InsightPerceptionEngine] = None
        self.pipeline: Optional[InsightRtPipeline] = None
        self._action_runner: Optional[SingleFlightActionRunner] = None

        if not defer_boot:
            self.boot()

    def boot(self) -> None:
        """Heavy initialisation: model loading, detector, pipeline, bootstrap.

        Safe to call once after __init__(defer_boot=True) when the UI is ready.
        """
        if self._booted:
            return
        self._booted = True

        self.config.model_path = self._prepare_runtime_model(self.config.model_path)
        self.detector = create_detector(self.config.model_path)
        self.frame_source = FrameSource(self.config, self._source_status_update)
        self.perception = InsightPerceptionEngine(
            config=self.config,
            broadcaster=self._broadcast,
            source_label_getter=self.frame_source.describe_source,
            detector=self.detector,
            recognition_worker=None,
            detector_state_callback=self._on_detector_state,
            detector_recovery_callback=self._emit_recovery_event,
            recognition_control_callback=self._on_recognition_control,
        )
        self.pipeline = InsightRtPipeline(
            frame_source=self.frame_source,
            perception=self.perception,
            ui=self.ui,
            source_label_getter=self.frame_source.describe_source,
        )
        self._action_runner = SingleFlightActionRunner(self.perception.set_status)

        self._boot_supervisor()
        self.pipeline.start()
        if hasattr(self.perception, "collect_broadcast_payloads"):
            self.ui.emit_many(self.perception.collect_broadcast_payloads(force=True))
        else:
            self.perception.publish_state(force=True)
        self._publish_system_state()

        self._recovery_thread = threading.Thread(
            target=self._recovery_loop,
            daemon=True,
            name="InsightSupervisor",
        )
        self._recovery_thread.start()

    def _apply_snapshot_defaults(self) -> None:
        if not getattr(self.config, "source_locked", False):
            source = str(self._snapshot.get("last_source", self.config.source) or self.config.source)
            if source in {"camera", "video"}:
                self.config.source = source
        detector_model = str(self._snapshot.get("detector_model", "") or "").strip()
        if detector_model:
            try:
                self.config.model_path = resolve_model_path(detector_model)
            except Exception:
                pass
        settings = self._snapshot.get("settings")
        if isinstance(settings, dict):
            for key in ("confidence", "iou"):
                if key in settings:
                    try:
                        setattr(self.config, key, float(settings[key]))
                    except (TypeError, ValueError):
                        pass
            if "detection_mode" in settings:
                self.config.detection_mode = normalize_detection_mode(settings["detection_mode"])
            if "image_size" in settings:
                try:
                    self.config.image_size = normalize_image_size(settings["image_size"])
                except (TypeError, ValueError):
                    pass
            if "max_det" in settings:
                try:
                    self.config.max_det = int(settings["max_det"])
                except (TypeError, ValueError):
                    pass
            for key in ("fps", "target_fps"):
                if key in settings:
                    try:
                        self.config.fps = normalize_fps(settings[key])
                    except (TypeError, ValueError):
                        pass

    def _boot_supervisor(self) -> None:
        self._set_mode("boot")
        self._verify_state_store()
        self._bootstrap_gallery()
        self._bootstrap_recognizer()
        self._bootstrap_detector()
        self._bootstrap_frame_source()
        self._bootstrap_local_ai()
        self._bootstrap_cloud_ai()
        self._restore_snapshot_state()
        self._recompute_operating_mode()
        self._save_snapshot()

    def _verify_state_store(self) -> None:
        try:
            self._snapshot_store.verify_writable()
            self._set_subsystem_state("state_store", "healthy", "State store writable")
        except Exception as exc:
            self._set_subsystem_state("state_store", "failed", str(exc))

    def _bootstrap_gallery(self) -> None:
        GALLERY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            ok, message = GalleryDB.verify_integrity(GALLERY_DB_PATH)
        except Exception as exc:
            ok, message = False, str(exc)
        if not ok:
            quarantine = GalleryDB.quarantine_database(GALLERY_DB_PATH, message)
            self._emit_recovery_event(
                "gallery_quarantine",
                f"Quarantined corrupt gallery DB to {quarantine.name if quarantine else 'unknown'}",
            )
            self._set_subsystem_state("gallery_db", "degraded", f"Recovered corrupt DB: {message}")
        try:
            self.gallery = GalleryDB(GALLERY_DB_PATH)
            self.capabilities.gallery_db = True
            self._set_subsystem_state("gallery_db", "healthy", "Gallery database ready")
        except Exception as exc:
            self.capabilities.gallery_db = False
            self._set_subsystem_state("gallery_db", "failed", str(exc))
            self.gallery = NullGalleryDB()

    def _bootstrap_recognizer(self) -> None:
        if not self.gallery.ensure_face_backend():
            self.capabilities.recognizer = False
            self.recognition_auto = False
            self._set_subsystem_state("recognizer", "degraded", getattr(self.gallery, "face_backend_error", "") or "Recognizer unavailable")
            self._schedule_probe("recognizer")
            return
        self.capabilities.recognizer = True
        self._set_subsystem_state("recognizer", "healthy", "Face recognizer ready")
        self._ensure_recognition_worker()

    def _bootstrap_detector(self) -> None:
        if self.detector.ensure_ready():
            self.capabilities.detector = True
            self._set_subsystem_state("detector", "healthy", "Detector ready")
            self.perception.attach_detector(self.detector)
            return
        self.capabilities.detector = False
        self._set_subsystem_state("detector", "failed", self.detector.last_error or "Detector unavailable")
        self._schedule_probe("detector")

    def _bootstrap_frame_source(self) -> None:
        requested = self.config.source
        target_order = [requested, "video" if requested == "camera" else "camera"]
        last_error = ""
        for target in target_order:
            try:
                prepared = self.frame_source.prepare_switch(target)
                self.frame_source.commit_prepared_switch(prepared)
                self.capabilities.frame_source = True
                self._set_subsystem_state("frame_source", "healthy", f"{target} ready")
                if target != requested:
                    self._emit_recovery_event("source_fallback", f"Booted on {target} after {requested} probe failed")
                return
            except Exception as exc:
                last_error = str(exc)
        self.capabilities.frame_source = False
        self._set_subsystem_state("frame_source", "failed", last_error or "No frame source available")
        self._schedule_probe("frame_source")

    def _bootstrap_local_ai(self) -> None:
        if self._probe_local_ai():
            self.capabilities.local_ai = True
            self._set_subsystem_state("local_ai", "healthy", "Local AI reachable")
            return
        self.capabilities.local_ai = False
        self._set_subsystem_state("local_ai", "degraded", "Local AI unavailable; rule-based fallback only")
        self._schedule_probe("local_ai")

    def _bootstrap_cloud_ai(self) -> None:
        if self.config.offline_only:
            self.capabilities.cloud_ai = False
            self._set_subsystem_state("cloud_ai", "disabled", "Offline-only mode")
            return
        if OPENAI_API_KEY or ANTHROPIC_API_KEY:
            self.capabilities.cloud_ai = True
            self._set_subsystem_state("cloud_ai", "healthy", "Cloud AI enabled")
        else:
            self.capabilities.cloud_ai = False
            self._set_subsystem_state("cloud_ai", "degraded", "No cloud AI credentials configured")

    def _restore_snapshot_state(self) -> None:
        if self._saved_settings:
            self._apply_settings_update(self._saved_settings)
            if "recog_auto" in self._saved_settings:
                enabled = bool(self._saved_settings.get("recog_auto", True))
                self.recognition_auto = enabled and self.capabilities.recognizer
                self.perception.recognition_worker = self.recognition_worker if self.recognition_auto else None
        roi = self._snapshot.get("roi")
        if isinstance(roi, dict):
            try:
                self.perception.set_roi(
                    float(roi["x1"]),
                    float(roi["y1"]),
                    float(roi["x2"]),
                    float(roi["y2"]),
                    shape=str(roi.get("shape", "rect")),
                )
            except (KeyError, TypeError, ValueError):
                pass

    def _source_status_update(self, message: str, level: str) -> None:
        self.perception.set_status(message, level)
        if level == "error":
            self._set_subsystem_state("frame_source", "degraded", message)
        elif "active" in message.lower():
            self.capabilities.frame_source = True
            self._set_subsystem_state("frame_source", "healthy", message)

    def _broadcast(self, payload: dict[str, Any]) -> None:
        if payload.get("type") == "recognition_result":
            self.perception.apply_recognition_result(payload)
            if payload.get("error"):
                self._recognition_error_streak += 1
                self._set_subsystem_state("recognizer", "degraded", str(payload.get("error")))
                if self._recognition_error_streak >= 3:
                    self._disable_recognition("Recognition disabled after repeated worker errors")
            else:
                self._recognition_error_streak = 0
                if self.capabilities.recognizer:
                    self._set_subsystem_state("recognizer", "healthy", "Recognition worker online")
        self.ui.emit_payload(payload)

    def _publish_system_state(self) -> None:
        metrics = self.perception.get_runtime_metrics() if self.perception is not None else {}
        payload = {
            "type": "system_health",
            "health": [asdict(item) for item in self.subsystem_health.values()],
            "metrics": metrics,
        }
        self.ui.emit_payload({"type": "operating_mode", "mode": self.operating_mode})
        self.ui.emit_payload({"type": "capability_report", "capabilities": asdict(self.capabilities)})
        self.ui.emit_payload(payload)

    def _set_subsystem_state(self, name: str, state: str, message: str = "") -> None:
        item = self.subsystem_health[name]
        item.state = state  # type: ignore[assignment]
        if state == "healthy":
            item.last_ok_ts = round(time.time(), 3)
            item.last_error = ""
            item.consecutive_failures = 0
        else:
            item.last_error = message
            if state in {"degraded", "failed"}:
                item.consecutive_failures += 1
        self._publish_system_state()

    def _set_mode(self, mode: OperatingMode) -> None:
        if self.operating_mode == mode:
            return
        self.operating_mode = mode
        self.ui.emit_payload({"type": "operating_mode", "mode": mode})
        self._save_snapshot()

    def _recompute_operating_mode(self) -> None:
        if self.capabilities.frame_source and self.capabilities.detector:
            if self.capabilities.recognizer and self.capabilities.local_ai:
                self._set_mode("full")
            else:
                self._set_mode("degraded_cv")
        elif self.capabilities.frame_source:
            self._set_mode("degraded_manual")
        else:
            self._set_mode("safe_idle")

    def _schedule_probe(self, name: str) -> None:
        idx = self._probe_backoff_index.get(name, 0)
        delays = self.config.boot_retry_backoff_sec
        delay = delays[min(idx, len(delays) - 1)]
        self._probe_backoff_index[name] = idx + 1
        self._next_probe_ts[name] = time.monotonic() + delay

    def _can_probe(self, name: str) -> bool:
        return time.monotonic() >= self._next_probe_ts.get(name, 0.0)

    def _clear_probe(self, name: str) -> None:
        self._next_probe_ts.pop(name, None)
        self._probe_backoff_index.pop(name, None)

    def _ensure_recognition_worker(self) -> None:
        if self.recognition_worker is not None:
            return
        self.recognition_worker = RecognitionWorker(
            embedder=None,
            gallery_db=self.gallery,
            broadcaster=self._broadcast,
            threshold=RECOGNITION_THRESHOLD,
            margin_threshold=RECOGNITION_MARGIN,
            top_k=RECOGNITION_TOP_K,
        )
        self.perception.recognition_worker = self.recognition_worker if self.recognition_auto else None

    def _disable_recognition(self, message: str) -> None:
        self.recognition_auto = False
        self.capabilities.recognizer = False
        self.perception.recognition_worker = None
        self._set_subsystem_state("recognizer", "disabled", message)
        self.perception.set_status(message, "warn")
        self._recompute_operating_mode()
        self._save_snapshot()

    def _on_recognition_control(self, enabled: bool, reason: str) -> None:
        if not enabled:
            self._disable_recognition(reason)

    def _on_detector_state(self, state: str, detail: str) -> None:
        self.capabilities.detector = state != "failed"
        self._set_subsystem_state("detector", state, detail)
        if state in {"degraded", "failed"}:
            self._schedule_probe("detector")
        else:
            self._clear_probe("detector")
        if state == "failed":
            self.capabilities.detector = False
        self._recompute_operating_mode()

    def _emit_recovery_event(self, action: str, detail: str) -> None:
        event = {
            "type": "recovery_event",
            "action": action,
            "detail": detail,
            "ts": round(time.time(), 3),
        }
        self._recovery_log.append(event)
        self._recovery_log = self._recovery_log[-40:]
        try:
            self._snapshot_store.append_recovery_event(event)
        except Exception:
            pass
        self.ui.emit_payload(event)

    @staticmethod
    def _post_json(url: str, payload: dict[str, Any], headers: Optional[dict[str, str]] = None) -> dict[str, Any]:
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=AI_HTTP_TIMEOUT_SECONDS) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            raise RuntimeError(f"HTTP {exc.code} from AI provider: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"AI provider connection failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("AI provider request timed out") from exc
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError("AI provider returned non-JSON response") from exc

    @staticmethod
    def _extract_openai_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") in {"output_text", "text"}:
                    text = str(block.get("text", "")).strip()
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()
        return ""

    @staticmethod
    def _extract_anthropic_text(content: Any) -> str:
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text", "")).strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()

    @staticmethod
    def _build_roi_ai_prompt(scan_results: list[dict[str, Any]], user_prompt: str) -> str:
        clean_prompt = (user_prompt or "").strip()
        if scan_results:
            summary_items = []
            for item in scan_results[:10]:
                label = str(item.get("label", "object"))
                confidence = int(round(float(item.get("confidence", 0.0)) * 100))
                area_pct = float(item.get("area_pct", 0.0))
                summary_items.append(f"- {label}: {confidence}% confidence, {area_pct:.1f}% area")
            scan_summary = "Detected objects in ROI:\n" + "\n".join(summary_items)
        else:
            scan_summary = "No objects were detected by the ROI scan."
        if not clean_prompt:
            clean_prompt = (
                "Describe what is happening in this ROI image, call out notable risks or anomalies, "
                "and suggest the most useful next check."
            )
        return f"{clean_prompt}\n\n{scan_summary}\n\nKeep the response concise and actionable."

    @staticmethod
    def _normalize_ai_provider(provider: Any) -> str:
        value = str(provider or "auto").strip().lower()
        aliases = {"claude": "anthropic", "gpt": "openai"}
        return aliases.get(value, value)

    def _resolve_ai_provider(self, requested: str) -> str:
        provider = self._normalize_ai_provider(requested)
        if provider not in {"auto", "ollama", "openai", "anthropic"}:
            raise RuntimeError(f"Unsupported AI provider: {provider}")
        if provider == "auto":
            return "ollama"
        if provider in {"openai", "anthropic"} and self.config.offline_only:
            raise RuntimeError("Cloud AI disabled in offline-only mode")
        if provider == "openai" and not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing")
        if provider == "anthropic" and not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is missing")
        return provider

    def _probe_local_ai(self) -> bool:
        try:
            parsed = urllib.parse.urlparse(INSIGHT_OLLAMA_URL)
            base_url = f"{parsed.scheme}://{parsed.netloc}/api/tags"
            request = urllib.request.Request(base_url, method="GET")
            with urllib.request.urlopen(request, timeout=1.5):
                return True
        except Exception:
            return False

    def _ask_ollama(self, image_b64: str, prompt: str, model: str = "") -> str:
        payload = {
            "model": model or INSIGHT_OLLAMA_MODEL,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
        data = self._post_json(INSIGHT_OLLAMA_URL, payload)
        text = str(data.get("response", "")).strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response")
        return text

    def _ask_openai(self, image_b64: str, prompt: str, model: str = "") -> str:
        payload = {
            "model": model or INSIGHT_OPENAI_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                }
            ],
            "temperature": 0.2,
        }
        data = self._post_json(
            "https://api.openai.com/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        )
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI returned no choices")
        message = choices[0].get("message", {})
        text = self._extract_openai_text(message.get("content"))
        if not text:
            raise RuntimeError("OpenAI returned an empty response")
        return text

    def _ask_anthropic(self, image_b64: str, prompt: str, model: str = "") -> str:
        payload = {
            "model": model or INSIGHT_ANTHROPIC_MODEL,
            "max_tokens": 600,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                    ],
                }
            ],
        }
        data = self._post_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01"},
        )
        text = self._extract_anthropic_text(data.get("content"))
        if not text:
            raise RuntimeError("Anthropic returned an empty response")
        return text

    def _build_local_roi_summary(self, capture: dict[str, Any], prompt: str) -> str:
        scan_results = capture.get("scan_results", []) or []
        if not scan_results:
            return "Local fallback: no detector-confirmed objects in the ROI. Review the capture manually and rescan if needed."
        labels = [str(item.get("label", "object")) for item in scan_results[:5]]
        strongest = max(scan_results, key=lambda item: float(item.get("confidence", 0.0)))
        confidence = int(round(float(strongest.get("confidence", 0.0)) * 100))
        lead = str(strongest.get("label", "object"))
        prompt_note = f" Prompt focus: {prompt.strip()}." if prompt.strip() else ""
        return (
            f"Local fallback:{prompt_note} ROI contains {len(scan_results)} detected object(s). "
            f"Most confident detection is {lead} at {confidence}%. "
            f"Top labels: {', '.join(labels)}. "
            "Suggested action: verify the lead object visually, then rescan or switch sources if confidence looks inconsistent."
        )

    def request_roi_ai_analysis(self, data: dict[str, Any]) -> None:
        capture = self.perception.get_last_roi_capture_context()
        if capture is None or not capture.get("image"):
            self.perception.set_status("Capture ROI first before Ask AI", "warn")
            return
        prompt = str(data.get("prompt", "") or "")
        requested_provider = self._normalize_ai_provider(data.get("provider", "auto"))
        requested_model = str(data.get("model", "") or "").strip()
        accepted = self._action_runner.submit(
            "roi_ai",
            "ROI AI analysis",
            lambda: self._run_roi_ai_analysis(capture, requested_provider, prompt, requested_model),
            "ROI AI already running",
        )
        if not accepted:
            return
        self._broadcast(
            {
                "type": "roi_ai_status",
                "stage": "started",
                "provider": requested_provider,
                "captured_at": capture.get("captured_at", 0),
            }
        )

    def _run_roi_ai_analysis(
        self, capture: dict[str, Any], requested_provider: str, prompt: str, model: str = ""
    ) -> None:
        captured_at = capture.get("captured_at", 0)
        try:
            provider = self._resolve_ai_provider(requested_provider)
            built_prompt = self._build_roi_ai_prompt(capture.get("scan_results", []), prompt)
            image_b64 = str(capture.get("image", ""))
            if provider == "openai":
                result = self._ask_openai(image_b64, built_prompt, model)
            elif provider == "anthropic":
                result = self._ask_anthropic(image_b64, built_prompt, model)
            else:
                result = self._ask_ollama(image_b64, built_prompt, model)
            self._set_subsystem_state("local_ai" if provider == "ollama" else "cloud_ai", "healthy", f"{provider} responded")
        except Exception as exc:
            if requested_provider in {"auto", "ollama"}:
                self.capabilities.local_ai = False
                self._set_subsystem_state("local_ai", "degraded", str(exc))
                self._schedule_probe("local_ai")
            result = self._build_local_roi_summary(capture, prompt)
            provider = "local-rule"
        self._broadcast(
            {
                "type": "roi_ai_result",
                "provider": provider,
                "text": result,
                "error": "",
                "captured_at": captured_at,
            }
        )
        self.perception.set_status(f"ROI AI complete ({provider})", "info")

    def _switch_target_label(self, target_source: str) -> str:
        if target_source == "camera":
            return f"camera {self.config.camera_index}"
        return f"video {self.config.video_path.name}"

    def _prepare_switch_worker(self, requested: Optional[str]) -> None:
        current_source = self.frame_source.current_source
        target_source = requested or ("video" if current_source == "camera" else "camera")
        target_label = self._switch_target_label(target_source)
        self._broadcast({"type": "source_switch", "stage": "starting", "target": target_source, "label": target_label})
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                prepared = self.frame_source.prepare_switch(requested)
                self._broadcast({"type": "source_switch", "stage": "prepared", "target": target_source, "label": target_label})
                if prepared != target_source:
                    self.confirm_prepared_switch(prepared)
                return
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.4)
        self._broadcast(
            {
                "type": "source_switch",
                "stage": "failed",
                "target": requested,
                "message": str(last_exc) if last_exc else "prepare failed",
            }
        )
        self._set_subsystem_state("frame_source", "degraded", str(last_exc) if last_exc else "prepare failed")
        self.perception.set_status(str(last_exc) if last_exc else "prepare failed", "error")

    def confirm_prepared_switch(self, requested: Optional[str] = None) -> None:
        self._broadcast({"type": "source_switch", "stage": "committing", "target": requested})
        try:
            active_source = self.frame_source.commit_prepared_switch(requested)
            self.perception.reset_scene()
            self.capabilities.frame_source = True
            self._set_subsystem_state("frame_source", "healthy", f"{active_source} active")
            self._broadcast(
                {
                    "type": "source_switch",
                    "stage": "ready",
                    "target": active_source,
                    "label": self._switch_target_label(active_source),
                }
            )
            self._recompute_operating_mode()
            self._save_snapshot()
        except Exception as exc:
            self._broadcast({"type": "source_switch", "stage": "failed", "target": requested, "message": str(exc)})
            self._set_subsystem_state("frame_source", "degraded", str(exc))
            self.perception.set_status(str(exc), "error")

    def cancel_prepared_switch(self) -> None:
        self.frame_source.cancel_prepared_switch()
        self._broadcast({"type": "source_switch", "stage": "cancelled"})

    def _settings_requires_background(self, settings: dict[str, Any]) -> bool:
        # Model/back-end changes can trigger heavy detector loads; keep them off the UI thread.
        return any(key in settings for key in ("detector_model", "detection_mode"))

    def _apply_settings_update(self, settings: dict[str, Any]) -> None:
        settings = dict(settings)
        self._saved_settings.update(settings)
        # Subsystems may not exist yet if boot() hasn't run (defer_boot mode).
        if self.perception is None or self.frame_source is None:
            self._save_snapshot()
            return
        detector_model = str(settings.get("detector_model", "") or "").strip()
        if detector_model:
            self._set_detector_model(detector_model)
            settings = {key: value for key, value in settings.items() if key != "detector_model"}
        if "detection_mode" in settings:
            settings["detection_mode"] = normalize_detection_mode(settings.get("detection_mode"))
        self.perception.update_settings(settings)
        self.frame_source.update_settings(settings)
        if self.recognition_worker is not None and "recog_threshold" in settings:
            try:
                self.recognition_worker.set_threshold(float(settings["recog_threshold"]))
            except (TypeError, ValueError):
                pass
        if self.recognition_worker is not None and "recog_top_k" in settings:
            try:
                self.recognition_worker._top_k = max(1, min(20, int(settings["recog_top_k"])))
            except (TypeError, ValueError):
                pass
        self._save_snapshot()

    def handle_client_message(self, data: dict[str, Any]) -> None:
        if self.perception is None:
            return
        message_type = data.get("type")
        if message_type == "client_ready":
            self.perception.publish_state(force=True)
            self._publish_system_state()
            for event in self._recovery_log[-10:]:
                self.ui.emit_payload(event)
            return
        if message_type == "select_track":
            try:
                self.perception.set_focus(int(data["track_id"]))
            except (KeyError, TypeError, ValueError):
                self.perception.set_status("Invalid track selection", "warn")
            return
        if message_type == "clear_focus":
            self.perception.clear_focus()
            return
        if message_type == "switch_source":
            requested = data.get("source")
            if requested not in (None, "camera", "video"):
                self.perception.set_status("Invalid source switch request", "warn")
                return
            self._action_runner.submit(
                "source_switch",
                "Source switch",
                lambda: self._prepare_switch_worker(requested),
                "Source switch already in progress",
            )
            return
        if message_type == "confirm_source_switch":
            requested = data.get("source")
            if requested not in (None, "camera", "video"):
                self.perception.set_status("Invalid source confirmation request", "warn")
                return
            self._action_runner.submit(
                "source_switch",
                "Source switch",
                lambda: self.confirm_prepared_switch(requested),
                "Source switch already in progress",
            )
            return
        if message_type == "cancel_source_switch":
            self.cancel_prepared_switch()
            return
        if message_type == "set_roi":
            try:
                self.perception.set_roi(
                    float(data["x1"]),
                    float(data["y1"]),
                    float(data["x2"]),
                    float(data["y2"]),
                    shape=data.get("shape", "rect"),
                )
                self._save_snapshot()
            except (KeyError, TypeError, ValueError):
                self.perception.set_status("Invalid ROI coordinates", "warn")
            return
        if message_type == "clear_roi":
            self.perception.clear_roi()
            self._save_snapshot()
            return
        if message_type == "capture_roi":
            self._action_runner.submit(
                "roi_capture",
                "ROI capture",
                self.perception.capture_roi_snapshot,
                "ROI capture already running",
            )
            return
        if message_type == "enroll_roi_burst":
            identity = str(data.get("identity", "")).strip()
            group = str(data.get("group", "")).strip() or "attendance"
            try:
                samples = max(3, min(20, int(data.get("samples", 10))))
            except (TypeError, ValueError):
                samples = 10
            try:
                duration_sec = max(1.0, min(8.0, float(data.get("duration_sec", 3.5))))
            except (TypeError, ValueError):
                duration_sec = 3.5
            if not identity:
                self.perception.set_status("enroll_roi_burst: identity required", "warn")
                return
            self._action_runner.submit(
                "gallery_write",
                "Gallery operation",
                lambda: self._run_enroll_roi_burst(identity, group, samples, duration_sec),
                "Gallery operation already running",
            )
            return
        if message_type == "ask_ai_roi":
            self.request_roi_ai_analysis(data)
            return
        if message_type == "clear_history":
            self.perception.clear_history()
            return
        if message_type == "delete_history_entry":
            try:
                self.perception.delete_history_entry(int(data["entry_id"]))
            except (KeyError, TypeError, ValueError):
                self.perception.set_status("Invalid history entry id", "warn")
            return
        if message_type == "update_settings":
            settings = data.get("settings")
            if isinstance(settings, dict):
                if self._settings_requires_background(settings):
                    self._action_runner.submit(
                        "settings_update",
                        "Settings update",
                        lambda: self._apply_settings_update(settings),
                        "Settings update already in progress",
                    )
                else:
                    self._apply_settings_update(settings)
            else:
                self.perception.set_status("Settings payload ignored", "warn")
            return
        if message_type == "set_ai_provider":
            return
        if message_type == "save_settings":
            self._save_snapshot()
            self.perception.set_status("Settings saved", "info")
            self._broadcast({"type": "settings_saved"})
            return
        if message_type == "ingest_gallery_folder":
            folder = data.get("folder", "")
            identity = str(data.get("identity", "")).strip()
            group = str(data.get("group", "")).strip()
            if not folder or not identity:
                self.perception.set_status("ingest_gallery_folder: folder and identity required", "warn")
                return
            self._action_runner.submit(
                "gallery_write",
                "Gallery operation",
                lambda: self._run_ingest_folder(folder, identity, group),
                "Gallery operation already running",
            )
            return
        if message_type == "ingest_gallery_media":
            mode = str(data.get("mode", "") or "").strip().lower()
            source_kind = str(data.get("source_kind", "") or "").strip().lower()
            path = str(data.get("path", "") or "").strip()
            identity = str(data.get("identity", "") or "").strip()
            group = str(data.get("group", "") or "").strip()
            if mode not in {"face", "similarity"}:
                self.perception.set_status("Choose how to use these photos first", "warn")
                return
            if source_kind not in {"image", "folder"}:
                self.perception.set_status("Choose an image or folder first", "warn")
                return
            if not path:
                self.perception.set_status("Select an image or folder to add", "warn")
                return
            if mode == "face" and not identity:
                self.perception.set_status("Enter a person name before adding photos", "warn")
                return
            self._action_runner.submit(
                "gallery_write",
                "Gallery operation",
                lambda: self._run_ingest_media(mode, source_kind, path, identity, group),
                "Gallery operation already running",
            )
            return
        if message_type == "rebuild_gallery_index":
            self._action_runner.submit(
                "gallery_write",
                "Gallery operation",
                self._run_rebuild_index,
                "Gallery operation already running",
            )
            return
        if message_type == "delete_gallery_identity":
            identity = str(data.get("identity", "")).strip()
            if not identity:
                self.perception.set_status("delete_gallery_identity: identity required", "warn")
                return
            self._action_runner.submit(
                "gallery_write",
                "Gallery operation",
                lambda: self._run_delete_identity(identity),
                "Gallery operation already running",
            )
            return
        if message_type == "rename_gallery_identity":
            old_name = str(data.get("old_name", "")).strip()
            new_name = str(data.get("new_name", "")).strip()
            if not old_name or not new_name:
                self.perception.set_status("rename_gallery_identity: old_name and new_name required", "warn")
                return
            self._action_runner.submit(
                "gallery_write",
                "Gallery operation",
                lambda: self._run_rename_identity(old_name, new_name),
                "Gallery operation already running",
            )
            return
        if message_type == "delete_similarity_item":
            try:
                item_id = int(data.get("item_id", 0))
            except (TypeError, ValueError):
                item_id = 0
            if item_id <= 0:
                self.perception.set_status("delete_similarity_item: item_id required", "warn")
                return
            self._action_runner.submit(
                "gallery_write",
                "Gallery operation",
                lambda: self._run_delete_similarity_item(item_id),
                "Gallery operation already running",
            )
            return
        if message_type == "find_similar_gallery_item":
            try:
                item_id = int(data.get("item_id", 0))
            except (TypeError, ValueError):
                item_id = 0
            if item_id <= 0:
                self.perception.set_status("find_similar_gallery_item: item_id required", "warn")
                return
            self._action_runner.submit(
                "similarity_search",
                "Similarity search",
                lambda: self._run_find_similar_item(item_id),
                "Similarity search already running",
            )
            return
        if message_type == "set_recognition_threshold":
            try:
                threshold = float(data["threshold"])
                if self.recognition_worker is not None:
                    self.recognition_worker.set_threshold(threshold)
                self.perception.set_status(f"Recognition threshold set to {round(threshold * 100)}%", "info")
            except (KeyError, TypeError, ValueError):
                self.perception.set_status("Invalid recognition threshold", "warn")
            return
        if message_type == "set_recognition_auto":
            enabled = bool(data.get("enabled", True))
            self.recognition_auto = enabled and self.capabilities.recognizer
            self._saved_settings["recog_auto"] = enabled
            self.perception.recognition_worker = self.recognition_worker if self.recognition_auto else None
            self.perception.set_status(f"Auto-recognition {'on' if self.recognition_auto else 'off'}", "info")
            self._save_snapshot()
            return
        if message_type == "recognize_entry":
            image_b64 = str(data.get("image_b64", ""))
            entry_id = int(data.get("entry_id", 0))
            track_id = int(data.get("track_id", 0))
            if not image_b64:
                self.perception.set_status("recognize_entry: image_b64 required", "warn")
                return
            if self.recognition_worker is None:
                self.perception.set_status("Recognition unavailable in current mode", "warn")
                return
            self._action_runner.submit(
                "manual_recognition",
                "Manual recognition",
                lambda: self._run_manual_recognize(entry_id, track_id, image_b64),
                "Manual recognition already running",
            )
            return
        if message_type == "enroll_scan_crop":
            crop_b64 = str(data.get("crop_b64", ""))
            identity = str(data.get("identity", "")).strip()
            group = str(data.get("group", "")).strip() or "scan"
            if not crop_b64 or not identity:
                self.perception.set_status("enroll_scan_crop: crop_b64 and identity required", "warn")
                return
            self._action_runner.submit(
                "gallery_write",
                "Gallery operation",
                lambda: self._run_enroll_crop(crop_b64, identity, group),
                "Gallery operation already running",
            )
            return
        if message_type == "get_gallery_state":
            self._broadcast_gallery_state()
            self._publish_system_state()
            return
        self.perception.set_status(f"Unknown message type: {message_type}", "warn")

    def _run_ingest_folder(self, folder: str, identity: str, group: str) -> None:
        folder_path = Path(folder).expanduser().resolve()
        if not folder_path.is_dir():
            self._broadcast({"type": "gallery_ingest_result", "mode": "face", "identity": identity, "added": 0, "errors": [f"Not a directory: {folder}"]})
            self.perception.set_status(f"Ingest failed: {folder} not found", "warn")
            return
        self.perception.set_status(f"Ingesting {identity} from {folder_path.name}...", "info")
        if not self.gallery.ensure_face_backend():
            err = self.gallery.face_backend_error or "Face model failed to load"
            self._broadcast({"type": "gallery_ingest_result", "mode": "face", "identity": identity, "added": 0, "errors": [err]})
            self.perception.set_status(f"Recognizer not ready: {err}", "error")
            self._set_subsystem_state("recognizer", "degraded", err)
            return

        def _progress(idx: int, total: int, name: str) -> None:
            self._broadcast({"type": "gallery_ingest_progress", "mode": "face", "identity": identity, "current": idx + 1, "total": total, "file": name})

        added, errors = self.gallery.ingest_folder(folder_path, identity, group, progress_cb=_progress)
        self.gallery.build_matrix()
        self._broadcast({"type": "gallery_ingest_result", "mode": "face", "identity": identity, "added": added, "errors": errors})
        self._broadcast_gallery_state()
        self.perception.set_status(
            f"Ingested {added} image(s) for '{identity}'" + (f" ({len(errors)} errors)" if errors else ""),
            "info",
        )
        self._save_snapshot()

    def _run_ingest_media(
        self,
        mode: str,
        source_kind: str,
        path: str,
        identity: str,
        group: str,
    ) -> None:
        if mode == "face":
            group_name = group or "attendance"
            if source_kind == "folder":
                self._run_ingest_folder(path, identity, group_name)
                return
            self._run_ingest_face_image(path, identity, group_name)
            return
        if source_kind == "folder":
            self._run_ingest_similarity_folder(path)
            return
        self._run_ingest_similarity_image(path)

    def _run_ingest_face_image(self, image_path: str, identity: str, group: str) -> None:
        image = Path(image_path).expanduser().resolve()
        if not image.is_file():
            self._broadcast(
                {
                    "type": "gallery_ingest_result",
                    "mode": "face",
                    "identity": identity,
                    "added": 0,
                    "errors": [f"Not a file: {image_path}"],
                }
            )
            self.perception.set_status(f"Add failed: {image_path} not found", "warn")
            return
        self.perception.set_status(f"Adding photos for {identity}...", "info")
        if not self.gallery.ensure_face_backend():
            err = self.gallery.face_backend_error or "Face model failed to load"
            self._broadcast({"type": "gallery_ingest_result", "mode": "face", "identity": identity, "added": 0, "errors": [err]})
            self.perception.set_status(f"Face recognition unavailable: {err}", "error")
            self._set_subsystem_state("recognizer", "degraded", err)
            return
        ok, err = self.gallery.ingest_single(image, identity, group)
        self.gallery.build_matrix()
        errors = [err] if err else []
        self._broadcast({"type": "gallery_ingest_result", "mode": "face", "identity": identity, "added": 1 if ok else 0, "errors": errors})
        self._broadcast_gallery_state()
        if ok:
            self.perception.set_status(f"Added photo for '{identity}'", "info")
            self._save_snapshot()
        else:
            self.perception.set_status(f"Add failed: {err}", "warn")

    def _run_ingest_similarity_image(self, image_path: str) -> None:
        image = Path(image_path).expanduser().resolve()
        if not image.is_file():
            self._broadcast(
                {
                    "type": "gallery_ingest_result",
                    "mode": "similarity",
                    "added": 0,
                    "errors": [f"Not a file: {image_path}"],
                }
            )
            self.perception.set_status(f"Add failed: {image_path} not found", "warn")
            return
        self.perception.set_status("Adding image to Similar Images...", "info")
        if not self.gallery.ensure_similarity_backend():
            err = self.gallery.similarity_backend_error or "Similarity model unavailable"
            self._broadcast({"type": "gallery_ingest_result", "mode": "similarity", "added": 0, "errors": [err]})
            self.perception.set_status(f"Similarity search unavailable: {err}", "warn")
            return
        ok, err = self.gallery.ingest_similarity_image(image)
        self.gallery.build_matrix()
        errors = [err] if err else []
        self._broadcast({"type": "gallery_ingest_result", "mode": "similarity", "added": 1 if ok else 0, "errors": errors})
        self._broadcast_gallery_state()
        if ok:
            self.perception.set_status("Added image to Similar Images", "info")
            self._save_snapshot()
        else:
            self.perception.set_status(f"Add failed: {err}", "warn")

    def _run_ingest_similarity_folder(self, folder: str) -> None:
        folder_path = Path(folder).expanduser().resolve()
        if not folder_path.is_dir():
            self._broadcast({"type": "gallery_ingest_result", "mode": "similarity", "added": 0, "errors": [f"Not a directory: {folder}"]})
            self.perception.set_status(f"Add failed: {folder} not found", "warn")
            return
        self.perception.set_status(f"Adding images from {folder_path.name}...", "info")
        if not self.gallery.ensure_similarity_backend():
            err = self.gallery.similarity_backend_error or "Similarity model unavailable"
            self._broadcast({"type": "gallery_ingest_result", "mode": "similarity", "added": 0, "errors": [err]})
            self.perception.set_status(f"Similarity search unavailable: {err}", "warn")
            return

        def _progress(idx: int, total: int, name: str) -> None:
            self._broadcast(
                {
                    "type": "gallery_ingest_progress",
                    "mode": "similarity",
                    "current": idx + 1,
                    "total": total,
                    "file": name,
                    "identity": "Similar Images",
                }
            )

        added, errors = self.gallery.ingest_similarity_folder(folder_path, progress_cb=_progress)
        self.gallery.build_matrix()
        self._broadcast({"type": "gallery_ingest_result", "mode": "similarity", "added": added, "errors": errors})
        self._broadcast_gallery_state()
        self.perception.set_status(
            f"Added {added} image(s) to Similar Images" + (f" ({len(errors)} errors)" if errors else ""),
            "info",
        )
        self._save_snapshot()

    def _run_rebuild_index(self) -> None:
        self.perception.set_status("Rebuilding gallery index...", "info")
        n = self.gallery.build_matrix()
        self._broadcast_gallery_state()
        self.perception.set_status(f"Gallery index rebuilt: {n} face samples", "info")

    def _run_delete_identity(self, identity: str) -> None:
        removed = self.gallery.delete_identity(identity)
        self.gallery.build_matrix()
        self._broadcast_gallery_state()
        self.perception.set_status(f"Deleted '{identity}' ({removed} record(s))", "info")
        self._save_snapshot()

    def _run_rename_identity(self, old_name: str, new_name: str) -> None:
        self.gallery.rename_identity(old_name, new_name)
        self.gallery.build_matrix()
        self._broadcast_gallery_state()
        self.perception.set_status(f"Renamed {old_name} to {new_name}", "info")
        self._save_snapshot()

    def _run_delete_similarity_item(self, item_id: int) -> None:
        removed = self.gallery.delete_similarity_item(item_id)
        self.gallery.build_matrix()
        self._broadcast_gallery_state()
        self.perception.set_status(
            "Deleted image from Similar Images" if removed else "Image was already removed",
            "info",
        )
        self._save_snapshot()

    def _run_find_similar_item(self, item_id: int) -> None:
        results = self.gallery.find_similar_items(item_id, top_k=12)
        source_path = self.gallery.get_similarity_item_path(item_id)
        self._broadcast(
            {
                "type": "similarity_search_result",
                "item_id": item_id,
                "source_path": source_path,
                "results": results,
            }
        )
        self.perception.set_status(
            "No similar images found" if not results else f"Found {len(results)} similar image(s)",
            "info",
        )

    def _run_manual_recognize(self, entry_id: int, track_id: int, image_b64: str) -> None:
        import base64 as _b64

        import cv2 as _cv2
        import numpy as _np

        try:
            raw = _b64.b64decode(image_b64)
            arr = _np.frombuffer(raw, dtype=_np.uint8)
            bgr = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Could not decode image")
        except Exception as exc:
            self._broadcast(
                {
                    "type": "recognition_result",
                    "entry_id": entry_id,
                    "track_id": track_id,
                    "identity": "unknown",
                    "confidence": 0.0,
                    "similarity": 0.0,
                    "threshold_met": False,
                    "top_matches": [],
                    "source": "manual",
                    "error": str(exc),
                }
            )
            return
        assert self.recognition_worker is not None
        self.recognition_worker.enqueue(entry_id, bgr, source="manual", track_id=track_id)

    def _run_enroll_crop(self, crop_b64: str, identity: str, group: str) -> None:
        import base64 as _b64

        import cv2 as _cv2
        import numpy as _np

        try:
            raw = _b64.b64decode(crop_b64)
            arr = _np.frombuffer(raw, dtype=_np.uint8)
            bgr = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Could not decode crop image")
        except Exception as exc:
            self.perception.set_status(f"Enroll failed: {exc}", "warn")
            return
        ok, err = self.gallery.ingest_bgr(bgr, identity, group, source_label="scan-crop")
        if ok:
            self.gallery.build_matrix()
            self._broadcast_gallery_state()
            self.perception.set_status(f"Enrolled '{identity}' from scan crop", "info")
            self._save_snapshot()
        else:
            self.perception.set_status(f"Enroll failed: {err}", "warn")

    def _run_enroll_roi_burst(
        self,
        identity: str,
        group: str,
        samples: int = 10,
        duration_sec: float = 3.5,
    ) -> None:
        if not self.gallery.ensure_face_backend():
            err = self.gallery.face_backend_error or "Face recognition backend unavailable"
            self.perception.set_status(f"ROI burst enroll failed: {err}", "error")
            self._set_subsystem_state("recognizer", "degraded", err)
            return

        self.perception.set_status(
            f"Capturing ROI burst for '{identity}' ({samples} samples / {duration_sec:.1f}s)...",
            "info",
        )

        candidates: list[dict[str, Any]] = []
        start = time.monotonic()
        deadline = start + duration_sec
        interval = max(0.12, duration_sec / max(1, samples))
        next_capture = start
        attempts = 0
        max_attempts = max(samples * 4, 20)

        while time.monotonic() < deadline and attempts < max_attempts:
            now = time.monotonic()
            if now < next_capture:
                time.sleep(min(0.05, next_capture - now))
                continue
            next_capture += interval
            attempts += 1

            crop, _shape = self.perception.get_current_roi_crop()
            if crop is None:
                continue
            sample = self.gallery.extract_face(crop)
            if sample is None:
                continue

            duplicate_idx: Optional[int] = None
            for idx, existing in enumerate(candidates):
                similarity = float(existing["feature"] @ sample.feature)
                if similarity >= 0.985:
                    duplicate_idx = idx
                    break

            if duplicate_idx is not None:
                if sample.quality > float(candidates[duplicate_idx]["quality"]):
                    candidates[duplicate_idx] = {
                        "crop": crop,
                        "feature": sample.feature.copy(),
                        "quality": float(sample.quality),
                    }
                continue

            candidates.append(
                {
                    "crop": crop,
                    "feature": sample.feature.copy(),
                    "quality": float(sample.quality),
                }
            )
            self.perception.set_status(
                f"ROI burst capture {min(len(candidates), samples)}/{samples} face sample(s) for '{identity}'",
                "info",
            )
            if len(candidates) >= samples:
                break

        selected = sorted(candidates, key=lambda item: float(item["quality"]), reverse=True)[:samples]
        added = 0
        errors: list[str] = []
        for idx, item in enumerate(selected, start=1):
            ok, err = self.gallery.ingest_bgr(
                item["crop"],
                identity,
                group,
                source_label=f"roi-burst-{idx}",
            )
            if ok:
                added += 1
            elif err:
                errors.append(err)

        if added > 0:
            self.gallery.build_matrix()
            self._broadcast_gallery_state()
            self._save_snapshot()

        self._broadcast(
            {
                "type": "roi_burst_enroll_result",
                "identity": identity,
                "group": group,
                "requested": samples,
                "captured": len(selected),
                "added": added,
                "errors": errors,
                "duration_sec": round(duration_sec, 2),
                "ts": round(time.time(), 3),
            }
        )

        if added > 0:
            self.perception.set_status(
                f"Enrolled {added} ROI burst face sample(s) for '{identity}'",
                "info",
            )
        else:
            detail = errors[0] if errors else "No usable faces were captured from the ROI"
            self.perception.set_status(f"ROI burst enroll failed: {detail}", "warn")

    def _broadcast_gallery_state(self) -> None:
        stats = self.gallery.get_stats()
        identities = self.gallery.list_identities()
        similarity_items = self.gallery.list_similarity_items()
        similarity_ready = self.gallery.ensure_similarity_backend()
        similarity_error = "" if similarity_ready else (self.gallery.similarity_backend_error or "Local similarity model unavailable")
        self._broadcast(
            {
                "type": "gallery_state",
                "identity_count": stats.identity_count,
                "image_count": stats.image_count,
                "similarity_item_count": stats.similarity_item_count,
                "group_names": stats.group_names,
                "last_rebuild": stats.last_rebuild,
                "people": [
                    {
                        "name": e.name,
                        "group_name": e.group_name,
                        "embedding_count": e.embedding_count,
                        "source_path": e.source_path,
                    }
                    for e in identities
                ],
                "identities": [
                    {
                        "name": e.name,
                        "group_name": e.group_name,
                        "embedding_count": e.embedding_count,
                        "source_path": e.source_path,
                    }
                    for e in identities
                ],
                "similarity_enabled": similarity_ready,
                "similarity_error": similarity_error,
                "similarity_items": [
                    {
                        "item_id": item.item_id,
                        "display_name": item.display_name,
                        "batch_label": item.batch_label,
                        "source_path": item.source_path,
                        "thumb_png_b64": b64encode(item.thumb_png).decode("ascii") if item.thumb_png else "",
                    }
                    for item in similarity_items
                ],
            }
        )

    def _recovery_loop(self) -> None:
        while not self.closed:
            now = time.monotonic()
            self._attempt_source_recovery()
            self._attempt_detector_recovery()
            self._attempt_recognizer_recovery()
            self._attempt_local_ai_recovery()
            if now - self._last_snapshot_ts >= self.config.snapshot_interval_sec:
                self._save_snapshot()
            time.sleep(max(0.5, self.config.health_poll_sec))

    def _attempt_source_recovery(self) -> None:
        if self.capabilities.frame_source and self.frame_source.failure_count == 0:
            return
        if not self._can_probe("frame_source"):
            return
        try:
            self.frame_source.reopen_current_source()
            self.capabilities.frame_source = True
            self._clear_probe("frame_source")
            self._set_subsystem_state("frame_source", "healthy", f"{self.frame_source.current_source} recovered")
            self._emit_recovery_event("source_recovered", f"{self.frame_source.current_source} recovered")
        except Exception:
            alternate = "video" if self.frame_source.current_source == "camera" else "camera"
            try:
                prepared = self.frame_source.prepare_switch(alternate)
                self.frame_source.commit_prepared_switch(prepared)
                self.capabilities.frame_source = True
                self._clear_probe("frame_source")
                self._set_subsystem_state("frame_source", "healthy", f"{alternate} recovered")
                self._emit_recovery_event("source_fallback", f"Recovered by switching to {alternate}")
            except Exception as exc:
                self.capabilities.frame_source = False
                self._set_subsystem_state("frame_source", "failed", str(exc))
                self._schedule_probe("frame_source")
        self._recompute_operating_mode()

    def _attempt_detector_recovery(self) -> None:
        detector_latched = bool(getattr(self.perception, "detector_latched", False))
        detector_errors = int(getattr(self.perception, "detector_consecutive_errors", 0))
        needs_recovery = (
            detector_latched
            or not self.capabilities.detector
            or detector_errors >= self.config.detector_error_threshold
        )
        if not needs_recovery:
            return
        if not self._can_probe("detector"):
            return
        if self.detector.reload():
            self.perception.attach_detector(self.detector)
            self.capabilities.detector = True
            self._clear_probe("detector")
            self._set_subsystem_state("detector", "healthy", "Detector recovered")
            self._emit_recovery_event("detector_recovered", "Detector recovered during supervisor probe")
        else:
            self.capabilities.detector = False
            self._set_subsystem_state("detector", "failed", self.detector.last_error or "Detector recovery failed")
            self._schedule_probe("detector")
        self._recompute_operating_mode()

    def _attempt_recognizer_recovery(self) -> None:
        if self.capabilities.recognizer:
            return
        if not self._can_probe("recognizer"):
            return
        if self.gallery.ensure_face_backend():
            self.capabilities.recognizer = True
            self.recognition_auto = RECOGNITION_AUTO
            self._ensure_recognition_worker()
            self.perception.recognition_worker = self.recognition_worker if self.recognition_auto else None
            self._clear_probe("recognizer")
            self._set_subsystem_state("recognizer", "healthy", "Face recognizer recovered")
            self._emit_recovery_event("recognizer_recovered", "Face recognizer recovered during supervisor probe")
        else:
            self._set_subsystem_state("recognizer", "degraded", self.gallery.face_backend_error or "Recognizer unavailable")
            self._schedule_probe("recognizer")
        self._recompute_operating_mode()

    def _attempt_local_ai_recovery(self) -> None:
        if self.capabilities.local_ai:
            return
        if not self._can_probe("local_ai"):
            return
        if self._probe_local_ai():
            self.capabilities.local_ai = True
            self._clear_probe("local_ai")
            self._set_subsystem_state("local_ai", "healthy", "Local AI recovered")
            self._emit_recovery_event("local_ai_recovered", "Local AI endpoint reachable again")
        else:
            self._set_subsystem_state("local_ai", "degraded", "Local AI still unavailable")
            self._schedule_probe("local_ai")
        self._recompute_operating_mode()

    def _save_snapshot(self) -> None:
        self._last_snapshot_ts = time.monotonic()
        if self.perception is None:
            snapshot = {"settings": dict(self._saved_settings)}
        else:
            snapshot = self.perception.get_runtime_snapshot()
        merged_settings = dict(self._saved_settings)
        merged_settings.update(snapshot.get("settings") or {})
        snapshot["settings"] = merged_settings
        snapshot.update(
            {
                "detector_model": model_choice_name(self.config.model_path),
                "last_source": self.frame_source.current_source if self.frame_source is not None else self.config.source,
                "mode": self.operating_mode,
                "capabilities": asdict(self.capabilities),
                "health": {name: asdict(item) for name, item in self.subsystem_health.items()},
                "last_error": {name: item.last_error for name, item in self.subsystem_health.items() if item.last_error},
            }
        )
        try:
            self._snapshot_store.save_snapshot(snapshot)
        except Exception:
            pass

    def get_saved_settings(self) -> dict[str, Any]:
        return dict(self._saved_settings)

    def _set_detector_model(self, model_name_or_path: str, *, force_reload: bool = False) -> bool:
        try:
            model_path = resolve_model_path(model_name_or_path)
            model_path = self._prepare_runtime_model(model_path, include_segmentation_pair=True)
        except Exception as exc:
            self.perception.set_status(f"Detector model ignored: {exc}", "warn")
            return False
        if model_path == self.config.model_path:
            if self.detector.is_ready and not force_reload:
                return True
            if not self.detector.reload():
                self.perception.set_status(
                    f"Detector model failed: {self.detector.last_error or model_path.name}",
                    "warn",
                )
                return False
            self.perception.attach_detector(self.detector)
            self.capabilities.detector = True
            self._set_subsystem_state("detector", "healthy", f"Detector ready ({model_path.name})")
            self.perception.set_status(f"Detector model reloaded ({model_choice_name(model_path)})", "info")
            self._recompute_operating_mode()
            self.perception.publish_state(force=True)
            self._publish_system_state()
            return True
        candidate = create_detector(model_path)
        if not candidate.ensure_ready():
            self.perception.set_status(
                f"Detector model failed: {candidate.last_error or model_path.name}",
                "warn",
            )
            return False
        self.detector = candidate
        self.config.model_path = model_path
        self.perception.attach_detector(candidate)
        self.capabilities.detector = True
        self._set_subsystem_state("detector", "healthy", f"Detector ready ({model_path.name})")
        self.perception.set_status(f"Detector model switched to {model_choice_name(model_path)}", "info")
        self._recompute_operating_mode()
        self.perception.publish_state(force=True)
        self._publish_system_state()
        return True

    def _prepare_runtime_model(self, model_path: Path, *, include_segmentation_pair: bool = False) -> Path:
        resolved = Path(model_path).expanduser().resolve()
        if is_face_detection_model_path(resolved):
            return resolved
        return ensure_runtime_model(resolved, image_size=self.config.image_size)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._set_mode("shutting_down")
        if self.recognition_worker is not None:
            self.recognition_worker.stop()
        if self.pipeline is not None:
            self.pipeline.stop()
        if self._action_runner is not None:
            self._action_runner.shutdown(timeout_sec=0.25)
        if self.gallery is not None:
            self.gallery.close()
        if self.frame_source is not None:
            self.frame_source.cleanup()
        self._save_snapshot()
