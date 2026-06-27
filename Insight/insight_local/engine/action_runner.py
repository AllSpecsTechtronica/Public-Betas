from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable


class SingleFlightActionRunner:
    """Bounded executor that rejects duplicate in-flight actions by group."""

    def __init__(
        self,
        status_callback: Callable[[str, str], None],
        *,
        max_workers: int = 3,
        thread_name_prefix: str = "InsightAction",
    ) -> None:
        self._status_callback = status_callback
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._lock = threading.Lock()
        self._active_groups: set[str] = set()
        self._future_groups: dict[Future[None], str] = {}
        self._closed = False

    def submit(
        self,
        action_group: str,
        label: str,
        fn: Callable[[], None],
        busy_message: str,
    ) -> bool:
        with self._lock:
            if self._closed:
                return False
            if action_group in self._active_groups:
                self._status_callback(busy_message, "info")
                return False
            self._active_groups.add(action_group)
        try:
            future = self._executor.submit(self._run, label, fn)
        except Exception:
            with self._lock:
                self._active_groups.discard(action_group)
            raise
        with self._lock:
            self._future_groups[future] = action_group
        future.add_done_callback(self._finish_future)
        return True

    def is_active(self, action_group: str) -> bool:
        with self._lock:
            return action_group in self._active_groups

    def shutdown(self, timeout_sec: float = 0.25) -> None:
        with self._lock:
            self._closed = True
            futures = list(self._future_groups)
        for future in futures:
            future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        deadline = time.monotonic() + max(0.0, timeout_sec)
        while time.monotonic() < deadline:
            with self._lock:
                if not self._active_groups:
                    break
            time.sleep(0.01)

    def _run(self, label: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            self._status_callback(f"{label} failed: {exc}", "error")
            raise

    def _finish_future(self, future: Future[None]) -> None:
        with self._lock:
            action_group = self._future_groups.pop(future, None)
            if action_group is not None:
                self._active_groups.discard(action_group)
