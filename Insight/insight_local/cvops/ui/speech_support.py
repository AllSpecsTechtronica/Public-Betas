"""Native speech support for the Notes AI composer.

Two pieces, both built on the local/native system (no cloud):

* ``SpeechDictationController`` -- push-to-talk dictation. Records the
  microphone with ffmpeg (the same avfoundation/dshow/pulse path the Notes
  voice memo recorder uses) and transcribes the clip with the local
  ``notes_transcription`` engine (Vosk / Whisper) on a background thread.

* ``TtsPlaybackBar`` -- a native Qt transport for reading a model message
  aloud. Synthesizes speech with the platform voice (macOS ``say``, Linux
  ``espeak``/``espeak-ng``, Windows SAPI) to an audio file, then plays it
  through ``QMediaPlayer`` so the message is a seekable *playback* (elapsed
  | play/pause | slider | duration) rather than a re-streamed wall of text.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QProcess, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaDevices, QMediaPlayer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from ..notes_transcription import DEFAULT_TRANSCRIPTION_PROVIDER


def _new_temp_stem() -> Path:
    name = f"cvops_speech_{os.getpid()}_{int(time.time() * 1000)}"
    return Path(tempfile.gettempdir()) / name


# Sample rate the ffmpeg pitch math is expressed against. The native render is
# resampled to this first so ``asetrate`` produces a predictable shift.
_FX_SAMPLE_RATE = 44100


def list_system_voices() -> list[tuple[str, str]]:
    """Installed platform voices as ``(name, locale)``; ``[]`` if none.

    macOS reads ``say -v '?'``, Windows the SAPI ``GetInstalledVoices``, Linux
    ``espeak --voices``. Best-effort and fully guarded — a missing engine or a
    parse failure just yields an empty list (the designer then shows only the
    platform default).
    """
    try:
        if sys.platform == "darwin" and shutil.which("say"):
            proc = subprocess.run(
                ["say", "-v", "?"], check=True, capture_output=True, text=True
            )
            out: list[tuple[str, str]] = []
            for line in proc.stdout.splitlines():
                # "Daniel              en_GB    # Hello..." — name may contain
                # spaces, so split on the locale token before the '#'.
                head = line.split("#", 1)[0].rstrip()
                if not head:
                    continue
                parts = head.split()
                if len(parts) < 2:
                    continue
                locale = parts[-1]
                name = " ".join(parts[:-1]).strip()
                if name:
                    out.append((name, locale))
            return out

        espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        if espeak:
            proc = subprocess.run(
                [espeak, "--voices"], check=True, capture_output=True, text=True
            )
            out = []
            for line in proc.stdout.splitlines()[1:]:  # skip header row
                parts = line.split()
                if len(parts) >= 4:
                    out.append((parts[3], parts[1]))
            return out

        if sys.platform == "win32" and shutil.which("powershell"):
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "(New-Object System.Speech.Synthesis.SpeechSynthesizer)."
                "GetInstalledVoices() | ForEach-Object { "
                "$_.VoiceInfo.Name + '|' + $_.VoiceInfo.Culture.Name }"
            )
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=True, capture_output=True, text=True,
            )
            out = []
            for line in proc.stdout.splitlines():
                name, _, locale = line.partition("|")
                if name.strip():
                    out.append((name.strip(), locale.strip()))
            return out
    except Exception:
        return []
    return []


def build_ffmpeg_filter(profile: Optional[dict]) -> str:
    """Compose an ffmpeg ``-af`` chain from a voice profile (``""`` = no-op).

    Pure and deterministic so it can be unit tested. Pitch is shifted while
    preserving duration (``asetrate`` then a compensating ``atempo``); the rest
    are gentle, character-shaping stages kept subtle by the profile defaults.
    Effects are skipped individually when their value is the neutral one.
    """
    if not isinstance(profile, dict):
        return ""
    stages: list[str] = []

    def _num(key: str, default: float = 0.0) -> float:
        try:
            return float(profile.get(key, default))
        except (TypeError, ValueError):
            return default

    # Signal chain order: cleanup -> pitch -> tonal EQ -> de-ess -> richness ->
    # dynamics -> spatial. Each stage is skipped at its neutral value so a flat
    # profile yields no filter at all.

    low_cut = _num("low_cut_hz")
    if low_cut >= 1.0:
        # Clear sub-bass rumble and plosive thumps for a cleaner low end.
        stages.append(f"highpass=f={int(low_cut)}")

    pitch = _num("pitch_semitones")
    if abs(pitch) >= 0.01:
        factor = 2.0 ** (pitch / 12.0)
        shifted = int(round(_FX_SAMPLE_RATE * factor))
        # asetrate changes pitch+speed; aresample back + atempo restores duration.
        stages.append(f"aresample={_FX_SAMPLE_RATE}")
        stages.append(f"asetrate={shifted}")
        stages.append(f"aresample={_FX_SAMPLE_RATE}")
        stages.append(f"atempo={1.0 / factor:.5f}")

    if profile.get("comms_bandpass"):
        # Radio/helmet band-limit: drop sub-bass and air for a "comms" timbre.
        stages.append("highpass=f=450")
        stages.append("lowpass=f=3000")

    warmth = _num("warmth_db")
    if warmth >= 0.05:
        # Low shelf around 180 Hz adds body without muddiness.
        stages.append(f"bass=g={warmth:.2f}:f=180")

    presence = _num("presence_db")
    if abs(presence) >= 0.05:
        # Peaking EQ near 3 kHz: positive = more articulate, negative = softer.
        stages.append(f"equalizer=f=3000:width_type=o:width=1.2:g={presence:.2f}")

    high_cut = _num("high_cut_hz")
    if high_cut >= 1.0:
        stages.append(f"lowpass=f={int(high_cut)}")

    air = _num("air_db")
    if air >= 0.05:
        # High shelf ~10 kHz adds gentle breath/air (kept below any high cut).
        stages.append(f"treble=g={air:.2f}:f=10000")

    sibilance = _num("sibilance")
    if sibilance >= 0.01:
        # De-esser tames the harsh synthetic "s"/"sh" that flags a robotic voice.
        intensity = max(0.0, min(1.0, sibilance))
        stages.append(f"deesser=i={intensity:.2f}:m=0.5:f=0.5:s=o")

    depth = _num("depth")
    if depth >= 0.01:
        # Subtle chorus thickens a thin voice for a fuller, more human body.
        decay = max(0.0, min(0.6, depth * 0.6))
        stages.append(f"chorus=0.7:0.9:55:{decay:.2f}:0.25:2")

    smoothing = _num("smoothing")
    if smoothing >= 0.01:
        # Loudness leveling evens the delivery (a larger gauss window = smoother).
        gwin = int(round(3 + max(0.0, min(1.0, smoothing)) * 28)) | 1  # odd 3..31
        stages.append(f"dynaudnorm=f=200:g={gwin}")

    room = _num("room")
    if room >= 0.01:
        # Faint single-tap echo reads as a small room without obvious reverb.
        decay = max(0.0, min(0.9, room))
        stages.append(f"aecho=0.8:0.85:32:{decay:.2f}")

    return ",".join(stages)


def _apply_voice_effects(src: Path, profile: Optional[dict]) -> Path:
    """Run the profile's ffmpeg chain over ``src``; return processed wav or ``src``.

    Falls back to the unprocessed native render when there is no effect chain or
    ffmpeg is unavailable, so TTS always produces audio.
    """
    chain = build_ffmpeg_filter(profile)
    if not chain:
        return src
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return src
    out = src.with_name(src.stem + "_fx.wav")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(src), "-af", chain, str(out)],
            check=True,
            capture_output=True,
        )
    except Exception:
        return src
    if out.exists() and out.stat().st_size > 0:
        try:
            src.unlink(missing_ok=True)
        except Exception:
            pass
        return out
    return src


def synthesize_speech(text: str, profile: Optional[dict] = None) -> Optional[Path]:
    """Render ``text`` to an audio file using the platform's native voice.

    When ``profile`` is given, its ``base_voice``/``rate_wpm`` steer the native
    synthesizer and its effect chain is applied as an ffmpeg post-process. With
    no profile (or no usable effects/ffmpeg) the behavior is the original plain
    native render. Returns the rendered clip path, or ``None`` when no system
    voice is available. Text is passed via a temp file (not argv) so long
    messages and shell metacharacters are handled safely.
    """
    body = str(text or "").strip()
    if not body:
        return None

    base_voice = str((profile or {}).get("base_voice") or "").strip()
    rate_wpm = 0
    try:
        rate_wpm = int(round(float((profile or {}).get("rate_wpm") or 0)))
    except (TypeError, ValueError):
        rate_wpm = 0

    stem = _new_temp_stem()
    txt_path = stem.with_suffix(".txt")
    try:
        txt_path.write_text(body, encoding="utf-8")
    except Exception:
        return None

    rendered: Optional[Path] = None
    try:
        if sys.platform == "darwin" and shutil.which("say"):
            out = stem.with_suffix(".aiff")
            opts: list[str] = []
            if base_voice:
                opts += ["-v", base_voice]
            if rate_wpm > 0:
                opts += ["-r", str(rate_wpm)]
            cmd = ["say", *opts, "-f", str(txt_path), "-o", str(out)]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                # A profile may name a voice that isn't installed on this
                # machine; retry with the system default so TTS still works.
                if base_voice:
                    retry = ["say"]
                    if rate_wpm > 0:
                        retry += ["-r", str(rate_wpm)]
                    retry += ["-f", str(txt_path), "-o", str(out)]
                    subprocess.run(retry, check=True, capture_output=True)
                else:
                    raise
            rendered = out if out.exists() and out.stat().st_size > 0 else None

        if rendered is None:
            espeak = shutil.which("espeak-ng") or shutil.which("espeak")
            if espeak and sys.platform != "darwin":
                out = stem.with_suffix(".wav")
                cmd = [espeak, "-w", str(out), "-f", str(txt_path)]
                if base_voice:
                    cmd[1:1] = ["-v", base_voice]
                if rate_wpm > 0:
                    cmd[1:1] = ["-s", str(rate_wpm)]
                subprocess.run(cmd, check=True, capture_output=True)
                rendered = out if out.exists() and out.stat().st_size > 0 else None

        if rendered is None and sys.platform == "win32":
            out = stem.with_suffix(".wav")
            select = (
                f"$s.SelectVoice('{base_voice}'); " if base_voice else ""
            )
            rate_set = f"$s.Rate = {_sapi_rate(rate_wpm)}; " if rate_wpm > 0 else ""
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"{select}{rate_set}"
                f"$s.SetOutputToWaveFile('{out}'); "
                "$s.Speak([System.IO.File]::ReadAllText("
                f"'{txt_path}')); $s.Dispose();"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", script],
                check=True,
                capture_output=True,
            )
            rendered = out if out.exists() and out.stat().st_size > 0 else None
    except Exception:
        rendered = None
    finally:
        try:
            txt_path.unlink(missing_ok=True)
        except Exception:
            pass

    if rendered is None:
        return None
    return _apply_voice_effects(rendered, profile)


def _sapi_rate(wpm: int) -> int:
    """Map words-per-minute to the SAPI ``Rate`` scale (-10..10, ~200 wpm = 0)."""
    return max(-10, min(10, int(round((wpm - 200) / 15.0))))


def text_to_speech_available() -> bool:
    if sys.platform == "darwin":
        return bool(shutil.which("say"))
    if sys.platform == "win32":
        return bool(shutil.which("powershell"))
    return bool(shutil.which("espeak-ng") or shutil.which("espeak"))


def microphone_available() -> bool:
    try:
        return len(QMediaDevices.audioInputs()) > 0
    except Exception:
        return False


def _fmt_ms(ms: int) -> str:
    secs = max(0, int(ms)) // 1000
    return f"{secs // 60}:{secs % 60:02d}"


class _TranscribeWorker(QThread):
    """Run local ASR off the UI thread; emit (text, capability)."""

    done = pyqtSignal(str, str)

    def __init__(self, wav_path: str, provider: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._wav = wav_path
        self._provider = provider

    def run(self) -> None:  # type: ignore[override]
        try:
            from ..notes_transcription import transcribe_audio_note

            payload = transcribe_audio_note(self._wav, provider=self._provider)
            self.done.emit(
                str(payload.get("text") or ""),
                str(payload.get("capability") or ""),
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.done.emit("", f"error:{exc}")
        finally:
            try:
                Path(self._wav).unlink(missing_ok=True)
            except Exception:
                pass


class SpeechDictationController(QObject):
    """Microphone push-to-talk dictation backed by ffmpeg + local ASR.

    States emitted via ``stateChanged``: ``recording`` -> ``transcribing`` ->
    ``idle``. Transcribed text arrives on ``transcribed``.
    """

    transcribed = pyqtSignal(str)
    stateChanged = pyqtSignal(str)
    errorRaised = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QObject] = None,
        *,
        provider: str = DEFAULT_TRANSCRIPTION_PROVIDER,
    ) -> None:
        super().__init__(parent)
        self._provider = provider
        self._proc: Optional[QProcess] = None
        self._pcm = bytearray()
        self._recording = False
        self._token = 0
        self._worker: Optional[_TranscribeWorker] = None

    def is_recording(self) -> bool:
        return self._recording

    def is_busy(self) -> bool:
        return self._recording or (self._worker is not None and self._worker.isRunning())

    def toggle(self) -> None:
        if self._recording:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        if self._recording:
            return
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.errorRaised.emit("ffmpeg not found — install ffmpeg to enable dictation.")
            return
        if not microphone_available():
            self.errorRaised.emit("No microphone device available for dictation.")
            return

        self._pcm = bytearray()
        self._token += 1
        token = self._token
        proc = QProcess(self)
        proc.setProgram(ffmpeg)
        if sys.platform == "darwin":
            input_args = ["-f", "avfoundation", "-i", ":0"]
        elif sys.platform == "win32":
            input_args = ["-f", "dshow", "-i", "audio=default"]
        else:
            input_args = ["-f", "pulse", "-i", "default"]
        proc.setArguments(input_args + ["-ar", "16000", "-ac", "1", "-f", "s16le", "pipe:1"])
        proc.readyReadStandardOutput.connect(lambda _t=token: self._read_stdout(_t))
        proc.finished.connect(lambda _ec, _es, _t=token: self._on_finished(_t))
        self._proc = proc
        self._recording = True
        proc.start()
        self.stateChanged.emit("recording")

    def _read_stdout(self, token: int) -> None:
        if token != self._token or self._proc is None:
            return
        chunk = bytes(self._proc.readAllStandardOutput())
        if chunk:
            self._pcm.extend(chunk)

    def stop(self) -> None:
        if not self._recording:
            return
        self._recording = False
        proc = self._proc
        if proc is not None and proc.state() != QProcess.ProcessState.NotRunning:
            try:
                proc.write(b"q")
            except Exception:
                proc.terminate()
        self.stateChanged.emit("transcribing")

    def _on_finished(self, token: int) -> None:
        if token != self._token:
            return
        proc = self._proc
        self._proc = None
        if proc is not None:
            tail = bytes(proc.readAllStandardOutput())
            if tail:
                self._pcm.extend(tail)
            proc.deleteLater()

        frames = bytes(self._pcm)
        self._pcm.clear()
        if not frames:
            self.stateChanged.emit("idle")
            self.errorRaised.emit(
                "No audio captured — check microphone privacy permissions for the terminal/app."
            )
            return

        out = _new_temp_stem().with_suffix(".wav")
        try:
            with wave.open(str(out), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(frames)
        except Exception as exc:
            self.stateChanged.emit("idle")
            self.errorRaised.emit(f"Could not save dictation clip: {exc}")
            return

        worker = _TranscribeWorker(str(out), self._provider, self)
        worker.done.connect(self._on_transcribed)
        self._worker = worker
        worker.start()

    def _on_transcribed(self, text: str, capability: str) -> None:
        self._worker = None
        self.stateChanged.emit("idle")
        cap = str(capability or "").strip().lower()
        if cap.startswith("error:"):
            self.errorRaised.emit(f"Dictation failed: {cap[6:].strip()}")
            return
        if cap in {"model_unavailable", "capability_unavailable"}:
            self.errorRaised.emit(
                "No local speech-to-text model available. Install Vosk or Whisper to enable dictation."
            )
            return
        clean = str(text or "").strip()
        if not clean:
            self.errorRaised.emit("Dictation produced no text.")
            return
        self.transcribed.emit(clean)


class TtsPlaybackBar(QWidget):
    """Native transport bar: elapsed | play/pause | seek slider | duration.

    ``speak(text)`` synthesizes and starts playback; the bar shows itself and
    hides via the close control. Designed to sit above the composer so a model
    message is heard as seekable audio, not re-streamed text.
    """

    errorRaised = pyqtSignal(str)
    # Emitted whenever the bar is shown or hidden so an overlay host can resize
    # the composer dock to keep the text input fully visible alongside it.
    visibilityChanged = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("notesTtsPlaybackBar")
        self._scrubbing = False
        self._source_path: Optional[Path] = None
        # Active voice profile applied to every synthesis. None -> plain native
        # voice; set via set_voice_profile() from the AI-settings voice designer.
        self._voice_profile: Optional[dict] = None

        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_playback_state)
        self._player.errorOccurred.connect(self._on_player_error)

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(8)

        self._speaker = QLabel("\U0001F50A")  # speaker glyph
        row.addWidget(self._speaker)

        self._elapsed = QLabel("0:00")
        self._elapsed.setObjectName("notesTtsElapsed")
        self._elapsed.setMinimumWidth(34)
        self._elapsed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self._elapsed)

        self._play_btn = QPushButton("❚❚")  # pause glyph (starts playing)
        self._play_btn.setObjectName("notesTtsPlay")
        self._play_btn.setFixedWidth(34)
        self._play_btn.clicked.connect(self._toggle_play)
        row.addWidget(self._play_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setObjectName("notesTtsSeek")
        self._slider.setRange(0, 0)
        self._slider.sliderPressed.connect(self._on_scrub_start)
        self._slider.sliderReleased.connect(self._on_scrub_end)
        row.addWidget(self._slider, stretch=1)

        self._duration = QLabel("0:00")
        self._duration.setObjectName("notesTtsDuration")
        self._duration.setMinimumWidth(34)
        self._duration.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self._duration)

        self._close_btn = QPushButton("✕")
        self._close_btn.setObjectName("notesTtsClose")
        self._close_btn.setFixedWidth(26)
        self._close_btn.setToolTip("Stop and close playback")
        self._close_btn.clicked.connect(self.stop)
        row.addWidget(self._close_btn)

        self.setVisible(False)

    def set_voice_profile(self, profile: Optional[dict]) -> None:
        """Set the voice profile used for subsequent ``speak()`` calls."""
        self._voice_profile = dict(profile) if isinstance(profile, dict) else None

    def speak(self, text: str) -> None:
        if not text_to_speech_available():
            self.errorRaised.emit(
                "No system text-to-speech voice available on this platform."
            )
            return
        path = synthesize_speech(text, self._voice_profile)
        if path is None:
            self.errorRaised.emit("Text-to-speech synthesis failed.")
            return
        self._cleanup_source()
        self._source_path = path
        self._slider.setRange(0, 0)
        self._elapsed.setText("0:00")
        self._duration.setText("0:00")
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        was_visible = self.isVisible()
        self.setVisible(True)
        if not was_visible:
            self.visibilityChanged.emit()
        self._player.play()

    def stop(self) -> None:
        self._player.stop()
        was_visible = self.isVisible()
        self.setVisible(False)
        self._cleanup_source()
        if was_visible:
            self.visibilityChanged.emit()

    def _toggle_play(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            # Restart from the top once playback has reached the end.
            if (
                self._player.duration() > 0
                and self._player.position() >= self._player.duration()
            ):
                self._player.setPosition(0)
            self._player.play()

    def _on_scrub_start(self) -> None:
        self._scrubbing = True

    def _on_scrub_end(self) -> None:
        self._scrubbing = False
        self._player.setPosition(self._slider.value())

    def _on_position(self, pos: int) -> None:
        if not self._scrubbing:
            self._slider.setValue(int(pos))
        self._elapsed.setText(_fmt_ms(pos))

    def _on_duration(self, dur: int) -> None:
        self._slider.setRange(0, int(dur))
        self._duration.setText(_fmt_ms(dur))

    def _on_playback_state(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._play_btn.setText("❚❚")  # pause
        else:
            self._play_btn.setText("▶")  # play

    def _on_player_error(self, error: QMediaPlayer.Error, message: str) -> None:
        if error == QMediaPlayer.Error.NoError:
            return
        self.errorRaised.emit(f"Playback error: {message or error}")
        self.setVisible(False)

    def _cleanup_source(self) -> None:
        path = self._source_path
        self._source_path = None
        if path is None:
            return
        try:
            self._player.setSource(QUrl())
        except Exception:
            pass
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
