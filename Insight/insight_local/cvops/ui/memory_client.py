from __future__ import annotations

import threading
from typing import Any, Optional

from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal


class CvOpsMemoryClient(QObject):
    """In-process drop-in for CvOpsWsClient.

    The cvops backend runs in the same process as the Qt UI, so the live event
    stream does not need to round-trip through a loopback websocket. This client
    subscribes directly to CvOpsService as an in-process event sink and re-emits
    the same Qt signals CvOpsWsClient did, so the window/panels are unchanged.

    Events arrive on backend worker/dispatcher threads. A private signal is used
    with a queued connection to marshal each payload onto the Qt main thread
    before fanning out to the typed public signals (Qt widgets must only be
    touched on the main thread)."""

    connectedChanged = pyqtSignal(bool)
    jobStatus = pyqtSignal(dict)
    jobResult = pyqtSignal(str, dict)
    scenarioUpdated = pyqtSignal(dict)
    trainingProgress = pyqtSignal(dict)
    cellProgress = pyqtSignal(dict)
    socketError = pyqtSignal(str)
    hello = pyqtSignal(dict)
    rawEvent = pyqtSignal(dict)  # every non-hello message, for event pulse

    # Private: carries a raw payload from any backend thread onto the main thread.
    _delivered = pyqtSignal(dict)
    _flushTrainingLogsRequested = pyqtSignal()

    def __init__(self, service: Any = None, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._service = service
        self._connected = False
        self._subscribed = False
        self._last_seq = 0
        self._log_lock = threading.Lock()
        self._pending_training_logs: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._log_flush_scheduled = False
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setSingleShot(True)
        self._log_flush_timer.setInterval(80)
        self._log_flush_timer.timeout.connect(self._flush_training_logs)
        self._delivered.connect(self._dispatch, Qt.ConnectionType.QueuedConnection)
        self._flushTrainingLogsRequested.connect(
            self._schedule_training_log_flush,
            Qt.ConnectionType.QueuedConnection,
        )

    def set_service(self, service: Any) -> None:
        """Bind the backend service once it finishes starting in the background."""
        self._service = service

    def set_last_seq(self, seq: int) -> None:
        self._last_seq = max(self._last_seq, int(seq or 0))

    def last_seq(self) -> int:
        return int(self._last_seq)

    def start(self, *, replay_since: int = 0) -> None:
        # The service is bound asynchronously (server boots off the UI thread);
        # do nothing until it exists. The window calls start() again on ready.
        if self._service is None:
            return
        if not self._subscribed:
            try:
                self._service.add_event_sink(self._sink, replay_since=max(0, int(replay_since or 0)))
            except TypeError:
                self._service.add_event_sink(self._sink)
            self._subscribed = True
        if not self._connected:
            self._connected = True
            self.connectedChanged.emit(True)
        # Synthetic hello mirrors the websocket handshake so the window's
        # connected handler runs its initial resync and populates panels.
        self.hello.emit({"type": "hello", "service": "cvops"})

    def stop(self) -> None:
        if self._subscribed:
            self._service.remove_event_sink(self._sink)
            self._subscribed = False
        if self._connected:
            self._connected = False
            self.connectedChanged.emit(False)

    def reconnect_now(self) -> None:
        # No socket to reopen; re-emitting connected re-triggers the window resync.
        if self._service is None:
            return
        if not self._subscribed:
            try:
                self._service.add_event_sink(self._sink, replay_since=self._last_seq)
            except TypeError:
                self._service.add_event_sink(self._sink)
            self._subscribed = True
        self._connected = True
        self.connectedChanged.emit(True)

    def is_connected(self) -> bool:
        return self._service is not None

    def replay_since(self, seq: int) -> bool:
        if self._service is None:
            return False
        replay = getattr(self._service, "replay_events", None)
        if not callable(replay):
            return False
        try:
            return bool(replay(max(0, int(seq or 0)), self._sink))
        except Exception as exc:
            self.socketError.emit(str(exc))
            return False

    # Called from backend worker/dispatcher threads.
    def _sink(self, payload: dict[str, Any]) -> None:
        try:
            if (
                str(payload.get("type") or "") == "training_progress"
                and str(payload.get("event") or "") == "log"
            ):
                self._queue_training_log(payload)
                return
            # Shallow top-level copy: producers build a fresh dict per emit and do
            # not mutate it afterwards, so nested references are safe to share.
            self._delivered.emit(dict(payload))
        except Exception:
            pass

    def _queue_training_log(self, payload: dict[str, Any]) -> None:
        scenario = str(payload.get("scenario") or "")
        job_id = str(payload.get("job_id") or "")
        key = (scenario, job_id)
        request_flush = False
        with self._log_lock:
            self._pending_training_logs.setdefault(key, []).append(dict(payload))
            if not self._log_flush_scheduled:
                self._log_flush_scheduled = True
                request_flush = True
        if request_flush:
            self._flushTrainingLogsRequested.emit()

    def _schedule_training_log_flush(self) -> None:
        if not self._log_flush_timer.isActive():
            self._log_flush_timer.start()

    def _flush_training_logs(self) -> None:
        with self._log_lock:
            batches = self._pending_training_logs
            self._pending_training_logs = {}
            self._log_flush_scheduled = False
        for (_scenario, _job_id), events in batches.items():
            if not events:
                continue
            last = dict(events[-1])
            lines = [
                {
                    "line": str(event.get("line") or ""),
                    "stream": str(event.get("stream") or "stdout"),
                    "timestamp": event.get("timestamp"),
                }
                for event in events
            ]
            last["event"] = "log_batch"
            last["lines"] = lines
            last["line"] = str(lines[-1].get("line") or "") if lines else ""
            last["stream"] = str(lines[-1].get("stream") or "stdout") if lines else "stdout"
            last["batched_count"] = len(lines)
            self._dispatch(last)

    # Runs on the Qt main thread (queued connection).
    def _dispatch(self, payload: dict[str, Any]) -> None:
        seq = int(payload.get("seq") or 0)
        if seq > 0:
            if seq <= self._last_seq:
                return
            self._last_seq = seq
        mtype = str(payload.get("type", ""))
        if mtype == "hello":
            self.hello.emit(payload)
            return
        self.rawEvent.emit(payload)
        if mtype == "job_status":
            self.jobStatus.emit(payload)
            return
        if mtype == "job_result":
            job_id = str(payload.get("job_id", ""))
            result = payload.get("result")
            if isinstance(result, dict):
                self.jobResult.emit(job_id, result)
            return
        if mtype == "scenario_updated":
            self.scenarioUpdated.emit(payload)
            return
        if mtype == "training_progress":
            self.trainingProgress.emit(payload)
            return
        if mtype == "cell_progress":
            self.cellProgress.emit(payload)
            return
