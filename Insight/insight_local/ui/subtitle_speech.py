from __future__ import annotations
from collections import deque
import threading
import time
from typing import Optional

import numpy as np
from PyQt6.QtCore import QByteArray, QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices
from PyQt6.QtWidgets import QFrame, QLabel, QWidget

try:
    import speech_recognition as _sr

    _HAS_SR = True
except ImportError:
    _HAS_SR = False

try:
    from faster_whisper import WhisperModel as _WhisperModel

    _HAS_FASTER_WHISPER = True
except ImportError:
    _HAS_FASTER_WHISPER = False


class AudioSpectrumWidget(QWidget):
    """FFT bar visualization behind subtitle text (web audioVizCanvas parity)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._levels: list[float] = [0.0] * 64
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_levels(self, levels: list[float]) -> None:
        self._levels = levels[:64] if levels else [0.0] * 64
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(198, 40, 40, 132))
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        n = len(self._levels)
        bar_w = w / max(1, n)
        for i, val in enumerate(self._levels):
            t = max(0.0, min(1.0, val))
            bar_h = t * h * 0.9
            alpha = int(255 * (0.15 + t * 0.35))
            painter.fillRect(
                int(i * bar_w),
                int(h - bar_h),
                max(1, int(bar_w) - 1),
                max(1, int(bar_h)),
                QColor(138, 20, 20, alpha),
            )
        painter.end()


class SubtitleBar(QFrame):
    """Bottom subtitle strip: spectrum + final / interim caption text."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame { background: rgba(198, 40, 40, 0.82); border: 1px solid rgba(138,20,20,0.40); }"
        )
        self._viz = AudioSpectrumWidget(self)
        self._text = QLabel(self)
        self._text.setWordWrap(True)
        self._text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._text.setStyleSheet("font-size: 13px; color: #140808; padding: 6px 16px; background: transparent;")
        self.hide()
        self._mode = "idle"
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._viz.setGeometry(0, 0, self.width(), self.height())
        self._text.setGeometry(0, 0, self.width(), self.height())

    def show_final(self, text: str, hold_ms: int = 4000) -> None:
        self._mode = "final"
        self._text.setText(text)
        self._text.setTextFormat(Qt.TextFormat.PlainText)
        self._text.setStyleSheet(
            "font-size: 13px; color: #140808; padding: 6px 16px; background: transparent;"
        )
        self.show()
        self.raise_()
        self._hide_timer.start(hold_ms)

    def show_interim(self, text: str, hold_ms: int = 8000) -> None:
        self._mode = "interim"
        self._text.setText(f'<span style="color: rgba(20,8,8,0.82);">{text}</span>')
        self._text.setTextFormat(Qt.TextFormat.RichText)
        self.show()
        self.raise_()
        self._hide_timer.start(hold_ms)

    def show_idle(self, text: str = "Listening...") -> None:
        if self._mode == "final" and self.isVisible():
            return
        self._mode = "idle"
        self._text.setText(text)
        self._text.setTextFormat(Qt.TextFormat.PlainText)
        self._text.setStyleSheet(
            "font-size: 12px; color: rgba(20,8,8,0.78); padding: 6px 16px; background: transparent;"
        )
        self._hide_timer.stop()
        self.show()
        self.raise_()

    def clear_interim(self) -> None:
        if self._mode != "interim":
            return
        self._mode = "idle"
        self._hide_timer.stop()
        self.hide()

    def push_levels(self, levels: list[float]) -> None:
        self._viz.set_levels(levels)


class MicSpeechController(QObject):
    """Captures mic via Qt Multimedia for spectrum; optional SpeechRecognition for captions."""

    spectrum = pyqtSignal(list)
    final_text = pyqtSignal(str)
    interim_text = pyqtSignal(str)
    status = pyqtSignal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._sample_rate = 16000
        self._stream_window_sec = 20.0
        self._silence_hold_sec = 0.55
        self._decode_interval_sec = 0.45
        self._min_decode_sec = 0.65
        self._speech_on_threshold = 0.03
        self._speech_off_threshold = 0.015
        self._sr_interim_interval_sec = 1.2
        self._audio: Optional[QAudioSource] = None
        self._device: Optional[object] = None
        self._io: Optional[object] = None
        self._buf = QByteArray()
        self._poll = QTimer(self)
        self._poll.timeout.connect(self._drain_audio)
        self._sr_stop = threading.Event()
        self._sr_thread: Optional[threading.Thread] = None
        self._audio_lock = threading.Lock()
        self._utterance_chunks: deque[np.ndarray] = deque()
        self._utterance_samples = 0
        self._speech_active = False
        self._speech_segment_done = False
        self._last_voice_ts = 0.0
        self._last_decode_ts = 0.0
        self._last_stream_text = ""
        self._last_interim_emit = 0.0
        self._last_final_emit = 0.0
        self._local_model: Optional[object] = None
        self._local_model_ready = False
        self._local_model_failed = False
        self._silent_poll_count = 0

    def start(self) -> None:
        if self._audio is not None or (self._sr_thread and self._sr_thread.is_alive()):
            self.stop()
        fmt = QAudioFormat()
        fmt.setSampleRate(16000)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        dev = QMediaDevices.defaultAudioInput()
        if dev is None or not dev.isFormatSupported(fmt):
            self.status.emit("Mic unavailable for spectrum (Qt Multimedia)", "warn")
            self._start_speech_only_fallback()
            return
        self._local_model_failed = False
        self._silent_poll_count = 0
        self._reset_stream_state()
        try:
            self._audio = QAudioSource(dev, fmt, self)
            self._io = self._audio.start()
        except Exception as exc:
            self.status.emit(f"Audio capture failed: {exc}", "warn")
            self._start_speech_only_fallback()
            return
        if self._io is None:
            self.status.emit("Mic failed to open — check system permissions", "warn")
            self._audio = None
            self._start_speech_only_fallback()
            return
        self._poll.start(50)
        if _HAS_FASTER_WHISPER:
            self._sr_stop.clear()
            self._sr_thread = threading.Thread(target=self._stream_loop, daemon=True, name="InsightSpeech")
            self._sr_thread.start()
            self.status.emit("Streaming subtitles active", "info")
        elif _HAS_SR:
            self._sr_stop.clear()
            self._sr_thread = threading.Thread(target=self._qt_speech_loop, daemon=True, name="InsightSpeech")
            self._sr_thread.start()
            self.status.emit("Subtitles active", "info")
        else:
            self.status.emit("Install SpeechRecognition + PyAudio for speech captions", "warn")

    def _start_speech_only_fallback(self) -> None:
        if _HAS_SR:
            self._sr_stop.clear()
            self._sr_thread = threading.Thread(target=self._speech_loop, daemon=True, name="InsightSpeech")
            self._sr_thread.start()
            self.status.emit("Subtitles active (speech only)", "info")

    def stop(self) -> None:
        self._poll.stop()
        self._sr_stop.set()
        if self._audio is not None:
            try:
                self._audio.stop()
            except Exception:
                pass
            self._audio = None
        self._io = None
        if self._sr_thread and self._sr_thread.is_alive():
            self._sr_thread.join(timeout=1.0)
        self._sr_thread = None
        self._reset_stream_state()

    def _drain_audio(self) -> None:
        if self._io is None or self._audio is None:
            return
        try:
            data = self._io.readAll()
        except Exception:
            return
        if data.isEmpty():
            return
        raw = data.data()
        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        elif not isinstance(raw, (bytes, bytearray)):
            raw = bytes(raw)
        if len(raw) < 2:
            return
        if len(raw) % 2:
            raw = raw[:-1]
        samples = np.frombuffer(raw, dtype=np.int16)
        if samples.size == 0:
            return
        n = min(64, max(8, len(samples) // 32))
        chunk = len(samples) // n
        levels: list[float] = []
        for i in range(n):
            seg = samples[i * chunk : (i + 1) * chunk]
            if seg.size == 0:
                levels.append(0.0)
            else:
                rms = float(np.sqrt(np.mean(seg.astype(np.float32) ** 2))) / 32768.0
                levels.append(min(1.0, rms * 8.0))
        self.spectrum.emit(levels)
        peak = max(levels) if levels else 0.0
        now = time.monotonic()
        if self._speech_active:
            active = peak >= self._speech_off_threshold
        else:
            active = peak >= self._speech_on_threshold

        # Warn once if mic signal has been silent for ~5 seconds (likely no permission)
        if peak > 0.01:
            self._silent_poll_count = -1  # has produced audio, never warn
        elif self._silent_poll_count >= 0:
            self._silent_poll_count += 1
            if self._silent_poll_count == 100:
                self.status.emit("No mic signal — check system mic permissions", "warn")

        prev_active = self._speech_active
        if _HAS_FASTER_WHISPER or _HAS_SR:
            self._update_stream_buffer(samples.copy(), active, now)
        else:
            if active and not self._speech_active:
                self._speech_active = True
            elif not active and self._speech_active:
                self._speech_active = False

        if active and not prev_active:
            if now - self._last_final_emit > 1.0 and now - self._last_interim_emit > 0.8:
                self._last_interim_emit = now
                self.interim_text.emit("Listening...")
        elif not active and prev_active:
            self.interim_text.emit("")

    def _update_stream_buffer(self, samples: np.ndarray, active: bool, now: float) -> None:
        max_samples = int(self._sample_rate * self._stream_window_sec)
        with self._audio_lock:
            if active and not self._speech_active:
                self._speech_active = True
                self._speech_segment_done = False
                self._last_decode_ts = 0.0
                self._last_stream_text = ""
                self._utterance_chunks.clear()
                self._utterance_samples = 0
            keep_chunk = active or self._speech_active or (
                self._last_voice_ts > 0.0 and (now - self._last_voice_ts) <= self._silence_hold_sec
            )
            if keep_chunk and samples.size > 0:
                self._utterance_chunks.append(samples)
                self._utterance_samples += int(samples.size)
                while self._utterance_samples > max_samples and self._utterance_chunks:
                    dropped = self._utterance_chunks.popleft()
                    self._utterance_samples -= int(dropped.size)
            if active:
                self._last_voice_ts = now
            elif self._speech_active and self._last_voice_ts > 0.0 and (now - self._last_voice_ts) > self._silence_hold_sec:
                self._speech_active = False
                self._speech_segment_done = True

    def _snapshot_utterance(self) -> tuple[Optional[np.ndarray], bool, bool, str]:
        with self._audio_lock:
            if not self._utterance_chunks or self._utterance_samples <= 0:
                return None, False, False, ""
            audio = np.concatenate(list(self._utterance_chunks)).astype(np.float32) / 32768.0
            return audio, self._speech_active, self._speech_segment_done, self._last_stream_text

    def _clear_utterance(self) -> None:
        with self._audio_lock:
            self._utterance_chunks.clear()
            self._utterance_samples = 0
            self._speech_segment_done = False
            self._speech_active = False
            self._last_voice_ts = 0.0
            self._last_decode_ts = 0.0
            self._last_stream_text = ""

    def _reset_stream_state(self) -> None:
        self._silent_poll_count = 0
        with self._audio_lock:
            self._utterance_chunks.clear()
            self._utterance_samples = 0
            self._speech_active = False
            self._speech_segment_done = False
            self._last_voice_ts = 0.0
            self._last_decode_ts = 0.0
            self._last_stream_text = ""

    def _stream_loop(self) -> None:
        try:
            model = self._get_local_model()
        except Exception as exc:
            self._local_model_failed = True
            if not self._sr_stop.is_set():
                self.status.emit(f"Streaming subtitles unavailable: {exc}", "warn")
            if _HAS_SR and not self._sr_stop.is_set():
                if self._audio is not None:
                    self._qt_speech_loop()
                else:
                    self._speech_loop()
            return
        min_samples = int(self._sample_rate * self._min_decode_sec)
        while not self._sr_stop.is_set():
            now = time.monotonic()
            should_decode = False
            finalize = False
            with self._audio_lock:
                enough_audio = self._utterance_samples >= min_samples
                if self._speech_active and enough_audio and (now - self._last_decode_ts) >= self._decode_interval_sec:
                    self._last_decode_ts = now
                    should_decode = True
                elif self._speech_segment_done and enough_audio:
                    should_decode = True
                    finalize = True
            if not should_decode:
                time.sleep(0.12)
                continue
            audio, _active, _done, last_text = self._snapshot_utterance()
            if audio is None or audio.size < min_samples:
                if finalize:
                    self._clear_utterance()
                time.sleep(0.08)
                continue
            try:
                text = self._transcribe_samples(model, audio)
            except Exception as exc:
                if not self._sr_stop.is_set():
                    self.status.emit(f"Recognition error: {exc}", "warn")
                time.sleep(0.2)
                continue
            text = text.strip()
            if finalize:
                final_text = text or last_text
                if final_text and not self._sr_stop.is_set():
                    self._last_final_emit = time.monotonic()
                    self.final_text.emit(final_text)
                self._clear_utterance()
                continue
            if text and text != last_text and not self._sr_stop.is_set():
                with self._audio_lock:
                    self._last_stream_text = text
                self._last_interim_emit = time.monotonic()
                self.interim_text.emit(text)

    def _qt_speech_loop(self) -> None:
        if not _HAS_SR:
            return
        recognizer = _sr.Recognizer()
        min_samples = int(self._sample_rate * max(self._min_decode_sec, 0.45))
        while not self._sr_stop.is_set():
            now = time.monotonic()
            should_decode = False
            finalize = False
            with self._audio_lock:
                enough_audio = self._utterance_samples >= min_samples
                if (
                    self._speech_active
                    and enough_audio
                    and (now - self._last_decode_ts) >= self._sr_interim_interval_sec
                ):
                    self._last_decode_ts = now
                    should_decode = True
                elif self._speech_segment_done and enough_audio:
                    should_decode = True
                    finalize = True
            if not should_decode:
                time.sleep(0.1)
                continue
            audio, _active, _done, last_text = self._snapshot_utterance()
            if audio is None or audio.size < min_samples:
                if finalize:
                    self._clear_utterance()
                continue
            audio_i16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16, copy=False)
            audio_data = _sr.AudioData(audio_i16.tobytes(), sample_rate=self._sample_rate, sample_width=2)
            try:
                text = self._transcribe_audio(recognizer, audio_data)
            except _sr.UnknownValueError:
                text = ""
            except Exception as exc:
                if not self._sr_stop.is_set():
                    self.status.emit(f"Recognition error: {exc}", "warn")
                if finalize:
                    self._clear_utterance()
                time.sleep(0.2)
                continue
            text = text.strip()
            if finalize:
                final_text = text or last_text
                if final_text and not self._sr_stop.is_set():
                    self._last_final_emit = time.monotonic()
                    self.final_text.emit(final_text)
                self._clear_utterance()
                continue
            if text and text != last_text and not self._sr_stop.is_set():
                with self._audio_lock:
                    self._last_stream_text = text
                self._last_interim_emit = time.monotonic()
                self.interim_text.emit(text)

    def _transcribe_samples(self, model, samples: np.ndarray) -> str:
        segments, _info = model.transcribe(
            samples,
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return " ".join(seg.text.strip() for seg in segments if seg.text).strip()

    def _speech_loop(self) -> None:
        if not _HAS_SR:
            return
        r = _sr.Recognizer()
        try:
            mic = _sr.Microphone()
        except Exception as exc:
            self.status.emit(f"Microphone open failed: {exc}", "warn")
            return
        try:
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.6)
        except Exception:
            pass
        while not self._sr_stop.is_set():
            try:
                with mic as source:
                    audio = r.listen(source, timeout=1, phrase_time_limit=12)
            except _sr.WaitTimeoutError:
                continue
            except Exception as exc:
                if not self._sr_stop.is_set():
                    self.status.emit(f"Listen error: {exc}", "warn")
                time.sleep(0.2)
                continue
            if self._sr_stop.is_set():
                break
            try:
                text = self._transcribe_audio(r, audio)
            except _sr.UnknownValueError:
                continue
            except Exception as exc:
                if not self._sr_stop.is_set():
                    self.status.emit(f"Recognition error: {exc}", "warn")
                continue
            if text and not self._sr_stop.is_set():
                self._last_final_emit = time.monotonic()
                self.final_text.emit(text.strip())

    def _transcribe_audio(self, recognizer: "_sr.Recognizer", audio: "_sr.AudioData") -> str:
        if _HAS_FASTER_WHISPER and not self._local_model_failed:
            try:
                model = self._get_local_model()
                raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                return self._transcribe_samples(model, samples)
            except Exception as exc:
                self._local_model_failed = True
                self._local_model = None
                if not self._sr_stop.is_set():
                    self.status.emit(f"Local subtitles unavailable, falling back: {exc}", "warn")
        return recognizer.recognize_google(audio)

    def _get_local_model(self):
        if self._local_model is not None:
            return self._local_model
        if not _HAS_FASTER_WHISPER:
            raise RuntimeError("faster_whisper not installed")
        self._local_model = _WhisperModel("tiny.en", device="cpu", compute_type="int8")
        if not self._local_model_ready and not self._sr_stop.is_set():
            self._local_model_ready = True
            self.status.emit("Local subtitles active", "info")
        return self._local_model
