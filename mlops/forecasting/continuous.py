from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .octopus import CompositeProjection, OctopusForecaster
from .spine import Sample, SignalSpine


ProjectionListener = Callable[[CompositeProjection], Awaitable[None] | None]


@dataclass
class SignalCadence:
    """Per-signal rate limit for ticks. Cheap arms run every sample; heavy
    arms (ARIMA, LSTM) rate-limited via min_interval."""
    signal_id: str
    min_interval: float = 0.0
    last_tick_ts: float = 0.0
    queued: bool = False


class ContinuousForecaster:
    """Long-running loop. On each new spine sample: dedup, respect cadence,
    tick octopus, persist and fan out projections to listeners (HUD, dashboard)."""

    def __init__(
        self,
        *,
        spine: SignalSpine,
        octopus: OctopusForecaster,
    ) -> None:
        self._spine = spine
        self._octopus = octopus
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None
        self._queue: asyncio.Queue[str] | None = None
        self._stop_event = threading.Event()
        self._listeners: list[ProjectionListener] = []
        self._cadence: dict[str, SignalCadence] = {}
        self._thread: threading.Thread | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._cadence_lock = threading.Lock()

    # ---- listener API ----------------------------------------------------

    def on_projection(self, listener: ProjectionListener) -> Callable[[], None]:
        self._listeners.append(listener)

        def _unsub() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return _unsub

    # ---- cadence ---------------------------------------------------------

    def set_cadence(self, signal_id: str, *, min_interval: float = 0.0) -> None:
        sid = str(signal_id or "").strip()
        if not sid:
            raise ValueError("signal_id required")
        with self._cadence_lock:
            cad = self._cadence.setdefault(sid, SignalCadence(signal_id=sid))
            cad.min_interval = float(max(0.0, min_interval))

    # ---- lifecycle -------------------------------------------------------

    def start_in_thread(self) -> None:
        """Start the loop in a dedicated thread — useful when there is no
        existing asyncio loop (e.g. called from Streamlit, FastAPI worker)."""
        if self._thread is not None and self._thread.is_alive():
            return
        ready = threading.Event()

        def _runner() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            ready.set()
            try:
                loop.run_until_complete(self._run())
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
                self._loop = None

        self._thread = threading.Thread(target=_runner, name="octopus-continuous", daemon=True)
        self._thread.start()
        ready.wait(timeout=2.0)

    async def start(self) -> None:
        """Start within the current asyncio loop."""
        self._loop = asyncio.get_running_loop()
        await self._run()

    def stop(self) -> None:
        self._stop_event.set()
        loop = self._loop
        queue = self._queue
        if loop is not None and queue is not None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, "__stop__")
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:
                pass
            self._unsubscribe = None

    # ---- internals -------------------------------------------------------

    async def _run(self) -> None:
        self._queue = asyncio.Queue()
        self._stop_event.clear()

        # Subscribe to spine samples. The callback fires on whatever thread the
        # producer called append() on, so we bounce into the loop.
        def _on_sample(sample: Sample) -> None:
            loop = self._loop
            q = self._queue
            if loop is None or q is None:
                return
            try:
                loop.call_soon_threadsafe(q.put_nowait, sample.signal_id)
            except RuntimeError:
                pass

        self._unsubscribe = self._spine.subscribe(_on_sample)

        try:
            while not self._stop_event.is_set():
                sid = await self._queue.get()
                if sid == "__stop__":
                    break
                await self._handle_signal(sid)
        finally:
            if self._unsubscribe is not None:
                self._unsubscribe()
                self._unsubscribe = None

    async def _handle_signal(self, sid: str) -> None:
        if not sid:
            return
        now = time.time()
        with self._cadence_lock:
            cad = self._cadence.setdefault(sid, SignalCadence(signal_id=sid))
            if cad.min_interval > 0 and (now - cad.last_tick_ts) < cad.min_interval:
                # Drop duplicate notifications during the cooldown window.
                # The next sample after the window will fire the tick.
                return
            cad.last_tick_ts = now

        try:
            composite = await self._octopus.tick(sid, ts_now=now)
        except Exception:
            return
        await self._fanout(composite)

    async def _fanout(self, composite: CompositeProjection) -> None:
        for listener in list(self._listeners):
            try:
                result = listener(composite)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass
