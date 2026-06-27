from __future__ import annotations

import threading

from ..config import GALLERY_DB_PATH, RuntimeConfig
from ..export_apple import ensure_runtime_model
from . import rt_pipeline as rt_pipeline_mod
from . import supervisor as supervisor_mod
from .detector import YoloDetector, create_detector
from .frame_source import FrameSource
from .gallery_db import GalleryDB, GalleryStats
from .perception import InsightPerceptionEngine
from .recognition_worker import RecognitionWorker
from .state_store import JsonStateStore
from .supervisor import InsightSupervisor, NullGalleryDB
from .ui_adapter import SessionUiAdapter


class LocalInsightSession(InsightSupervisor):
    """Qt-facing wrapper around the split runtime supervisor."""

    def __init__(self, config: RuntimeConfig, *, defer_boot: bool = False) -> None:
        self._ui = SessionUiAdapter()
        self.hud_payload = self._ui.payload_ready
        self._sync_supervisor_dependencies()
        InsightSupervisor.__init__(self, config, self._ui, defer_boot=defer_boot)

    def _sync_supervisor_dependencies(self) -> None:
        supervisor_mod.FrameSource = FrameSource
        supervisor_mod.GalleryDB = GalleryDB
        supervisor_mod.RecognitionWorker = RecognitionWorker
        supervisor_mod.YoloDetector = YoloDetector
        supervisor_mod.create_detector = create_detector
        supervisor_mod.InsightPerceptionEngine = InsightPerceptionEngine
        supervisor_mod.JsonStateStore = JsonStateStore
        supervisor_mod.ensure_runtime_model = ensure_runtime_model
        supervisor_mod.GALLERY_DB_PATH = GALLERY_DB_PATH
        supervisor_mod.threading.Thread = threading.Thread
        rt_pipeline_mod.threading.Thread = threading.Thread
