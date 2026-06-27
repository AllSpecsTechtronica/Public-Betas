from __future__ import annotations

import threading
import time
from queue import Empty, Queue
from typing import Any, Callable, Optional

import numpy as np

from .recognizer import cosine_search, decide_identity, prototype_search


class RecognitionJob:
    __slots__ = ("entry_id", "track_id", "crop", "label", "source", "queued_at")

    def __init__(
        self,
        entry_id: int,
        crop: np.ndarray,
        label: str = "",
        source: str = "auto",
        track_id: int = 0,
    ) -> None:
        self.entry_id = entry_id
        self.track_id = track_id
        self.crop = crop
        self.label = label
        self.source = source
        self.queued_at = time.monotonic()


class RecognitionWorker:
    """
    Single background thread that processes recognition jobs off the hot CV loop.
    Results are delivered via the broadcaster callback (same pattern as roi_ai_result).

    broadcaster receives payloads of type "recognition_result".
    """

    _QUEUE_MAXSIZE = 32
    _JOB_TIMEOUT_SEC = 10.0

    def __init__(
        self,
        embedder,
        gallery_db,
        broadcaster: Callable[[dict[str, Any]], None],
        threshold: float = 0.72,
        margin_threshold: float = 0.045,
        top_k: int = 5,
    ) -> None:
        self._embedder = embedder
        self._gallery = gallery_db
        self._broadcaster = broadcaster
        self._threshold = threshold
        self._margin_threshold = max(0.01, min(0.2, margin_threshold))
        self._top_k = top_k

        self._queue: Queue[Optional[RecognitionJob]] = Queue(maxsize=self._QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="InsightRecognition"
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(
        self,
        entry_id: int,
        crop: np.ndarray,
        label: str = "",
        source: str = "auto",
        track_id: int = 0,
    ) -> bool:
        """
        Add a recognition job. Returns False if queue is full (drop silently).
        Stale crops (>10s old) are never queued.
        """
        if not self._gallery.has_gallery:
            return False
        job = RecognitionJob(
            entry_id=entry_id,
            crop=crop.copy(),
            label=label,
            source=source,
            track_id=track_id,
        )
        try:
            self._queue.put_nowait(job)
            return True
        except Exception:
            return False

    def set_threshold(self, threshold: float) -> None:
        self._threshold = max(0.0, min(1.0, threshold))

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except Empty:
                continue
            if job is None:
                break
            age = time.monotonic() - job.queued_at
            if age > self._JOB_TIMEOUT_SEC:
                continue
            self._process(job)

    def _process(self, job: RecognitionJob) -> None:
        try:
            if job.label and job.label.lower() not in {"person", "face"}:
                self._broadcast_error(job, "Face recognition is only available for person or face tracks")
                return

            sample = self._gallery.extract_face(job.crop)
            if sample is None:
                self._broadcast_error(job, "No usable face detected")
                return
            vec = sample.feature

            matrix = self._gallery.matrix
            profile_matrix = self._gallery.profile_matrix
            if matrix is None or matrix.shape[0] == 0 or profile_matrix is None or profile_matrix.shape[0] == 0:
                self._broadcast_error(job, "Gallery empty")
                return

            labels = self._gallery.matrix_labels
            groups = self._gallery.matrix_groups
            sources = self._gallery.matrix_sources
            profile_labels = self._gallery.profile_labels
            profile_groups = self._gallery.profile_groups
            profile_sources = self._gallery.profile_sources
            profile_sample_counts = self._gallery.profile_sample_counts

            matches = cosine_search(
                query=vec,
                gallery_matrix=matrix,
                identity_labels=labels,
                group_labels=groups,
                source_paths=sources,
                top_k=self._top_k,
                threshold=self._threshold,
            )
            profile_matches = prototype_search(
                query=vec,
                profile_matrix=profile_matrix,
                identity_labels=profile_labels,
                group_labels=profile_groups,
                source_paths=profile_sources,
                sample_counts=profile_sample_counts,
                top_k=3,
            )
            identity, confidence, decision = decide_identity(
                raw_matches=matches,
                profile_matches=profile_matches,
                threshold=self._threshold,
                margin_threshold=self._margin_threshold,
            )

            self._broadcaster(
                {
                    "type": "recognition_result",
                    "entry_id": job.entry_id,
                    "track_id": job.track_id,
                    "label": job.label,
                    "identity": identity,
                    "confidence": round(confidence, 4),
                    "similarity": round(matches[0].similarity, 4) if matches else 0.0,
                    "threshold_met": bool(decision.get("accepted", False)),
                    "candidate_identity": profile_matches[0].identity if profile_matches else "unknown",
                    "vote_share": float(decision.get("vote_share", 0.0)),
                    "prototype_similarity": float(decision.get("prototype_similarity", 0.0)),
                    "margin": float(decision.get("margin", 0.0)),
                    "support_count": int(decision.get("support_count", 0)),
                    "decision_reason": str(decision.get("reason", "")),
                    "face_quality": round(float(sample.quality), 4),
                    "face_detection_score": round(float(sample.detection_score), 4),
                    "top_matches": [
                        {
                            "identity": m.identity,
                            "group_name": m.group_name,
                            "similarity": m.similarity,
                            "source_path": m.source_path,
                        }
                        for m in matches[:5]
                    ],
                    "source": job.source,
                    "ts": round(time.time(), 3),
                }
            )
        except Exception as exc:
            self._broadcast_error(job, str(exc))

    def _broadcast_error(self, job: RecognitionJob, error: str) -> None:
        self._broadcaster(
            {
                "type": "recognition_result",
                "entry_id": job.entry_id,
                "track_id": job.track_id,
                "label": job.label,
                "identity": "unknown",
                "confidence": 0.0,
                "similarity": 0.0,
                "threshold_met": False,
                "candidate_identity": "unknown",
                "vote_share": 0.0,
                "prototype_similarity": 0.0,
                "margin": 0.0,
                "support_count": 0,
                "decision_reason": "error",
                "face_quality": 0.0,
                "face_detection_score": 0.0,
                "top_matches": [],
                "source": job.source,
                "error": error,
                "ts": round(time.time(), 3),
            }
        )
