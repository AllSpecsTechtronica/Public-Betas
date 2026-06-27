"""Small WAV analysis and cleanup utilities for CV Ops audio datasets.

The implementation intentionally stays dependency-light so audio recognition can
run in the same local environments as the CV tooling. It supports uncompressed
PCM WAV input and writes cleaned 16-bit mono WAV output.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import time
import wave
from array import array
from pathlib import Path
from typing import Any


SUPPORTED_SAMPLE_WIDTHS = {1, 2, 4}
FEATURE_SCHEMA = [
    "duration_s",
    "sample_rate_norm",
    "channels",
    "sample_width",
    "rms",
    "zero_crossing_rate",
    "mean_abs",
    "std",
    "log_size",
]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = max(0.0, min(100.0, pct)) / 100.0 * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _read_pcm_wav(path: Path, *, max_seconds: float | None = None) -> tuple[list[float], dict[str, Any]]:
    """Read a PCM WAV as mono floats in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        channels = max(1, int(wf.getnchannels()))
        sample_width = int(wf.getsampwidth())
        frame_rate = max(1, int(wf.getframerate()))
        frames = int(wf.getnframes())
        if sample_width not in SUPPORTED_SAMPLE_WIDTHS:
            raise ValueError(f"unsupported WAV sample width: {sample_width} byte(s)")
        read_frames = frames
        if max_seconds is not None and max_seconds > 0:
            read_frames = min(frames, int(frame_rate * max_seconds))
        raw = wf.readframes(read_frames)

    if sample_width == 1:
        vals = [float(b - 128) / 128.0 for b in raw]
    elif sample_width == 2:
        arr = array("h")
        arr.frombytes(raw)
        if arr.itemsize != 2:
            arr.byteswap()
        vals = [float(v) / 32768.0 for v in arr]
    else:
        arr = array("i")
        arr.frombytes(raw)
        if arr.itemsize != 4:
            arr.byteswap()
        vals = [float(v) / 2147483648.0 for v in arr]

    mono: list[float] = []
    if channels == 1:
        mono = vals
    else:
        usable = len(vals) - (len(vals) % channels)
        for i in range(0, usable, channels):
            mono.append(sum(vals[i:i + channels]) / channels)

    meta = {
        "sample_rate": frame_rate,
        "channels": channels,
        "sample_width": sample_width,
        "frames": frames,
        "duration_s": float(frames / frame_rate) if frame_rate else 0.0,
        "decoded_samples": len(mono),
    }
    return mono, meta


def _frame_rms(samples: list[float], sample_rate: int, frame_ms: int = 20) -> list[float]:
    width = max(1, int(sample_rate * frame_ms / 1000.0))
    out: list[float] = []
    for start in range(0, len(samples), width):
        frame = samples[start:start + width]
        if not frame:
            continue
        out.append(math.sqrt(sum(v * v for v in frame) / len(frame)))
    return out


def analyze_wav(path: str | Path, *, max_seconds: float | None = None) -> dict[str, Any]:
    """Return audio quality and waveform metrics for a PCM WAV file."""
    wav_path = Path(path)
    samples, meta = _read_pcm_wav(wav_path, max_seconds=max_seconds)
    sample_rate = int(meta["sample_rate"])
    size = wav_path.stat().st_size if wav_path.exists() else 0
    if not samples:
        return {
            "path": str(wav_path),
            "format": "wav",
            **meta,
            "size_bytes": size,
            "rms": 0.0,
            "peak": 0.0,
            "mean_abs": 0.0,
            "std": 0.0,
            "dc_offset": 0.0,
            "zero_crossing_rate": 0.0,
            "clipping_ratio": 0.0,
            "silence_ratio": 1.0,
            "noise_floor_rms": 0.0,
            "noise_floor_dbfs": -120.0,
            "snr_db": 0.0,
        }

    abs_vals = [abs(v) for v in samples]
    peak = max(abs_vals)
    rms = math.sqrt(sum(v * v for v in samples) / len(samples))
    mean = sum(samples) / len(samples)
    zc = 0
    prev = samples[0]
    for value in samples[1:]:
        if (prev < 0 <= value) or (prev >= 0 > value):
            zc += 1
        prev = value
    frame_levels = _frame_rms(samples, sample_rate)
    noise_floor = _percentile(frame_levels, 20.0)
    silence_threshold = max(noise_floor * 1.5, peak * 0.01, 1e-5)
    silent = sum(1 for level in frame_levels if level <= silence_threshold)
    noise_db = 20.0 * math.log10(max(noise_floor, 1e-6))
    snr = 20.0 * math.log10(max(rms, 1e-6) / max(noise_floor, 1e-6))
    return {
        "path": str(wav_path),
        "format": "wav",
        **meta,
        "size_bytes": size,
        "rms": float(rms),
        "peak": float(peak),
        "mean_abs": float(sum(abs_vals) / len(abs_vals)),
        "std": float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
        "dc_offset": float(mean),
        "zero_crossing_rate": float(zc / max(1, len(samples) - 1)),
        "clipping_ratio": float(sum(1 for v in abs_vals if v >= 0.98) / len(abs_vals)),
        "silence_ratio": float(silent / len(frame_levels)) if frame_levels else 1.0,
        "noise_floor_rms": float(noise_floor),
        "noise_floor_dbfs": float(noise_db),
        "snr_db": float(snr),
    }


def extract_feature_vector(path: str | Path) -> list[float]:
    """Return the feature vector used by the lightweight audio recognizer."""
    wav_path = Path(path)
    try:
        metrics = analyze_wav(wav_path, max_seconds=30.0)
    except Exception:
        return [0.0] * len(FEATURE_SCHEMA)
    return [
        float(metrics.get("duration_s") or 0.0),
        float(metrics.get("sample_rate") or 0.0) / 48000.0,
        float(metrics.get("channels") or 0.0),
        float(metrics.get("sample_width") or 0.0),
        float(metrics.get("rms") or 0.0),
        float(metrics.get("zero_crossing_rate") or 0.0),
        float(metrics.get("mean_abs") or 0.0),
        float(metrics.get("std") or 0.0),
        math.log1p(float(metrics.get("size_bytes") or 0.0)),
    ]


def _write_wav16(path: Path, samples: list[float], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = array("h", [max(-32768, min(32767, int(round(v * 32767.0)))) for v in samples])
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def clean_wav(
    input_path: str | Path,
    output_path: str | Path,
    *,
    noise_reduce: bool = True,
    trim_silence: bool = True,
    normalize: bool = True,
    noise_reduction_strength: float = 0.65,
    target_peak: float = 0.90,
) -> dict[str, Any]:
    """Clean a PCM WAV and return before/after metrics plus the output path."""
    src = Path(input_path)
    dst = Path(output_path)
    before = analyze_wav(src)
    samples, meta = _read_pcm_wav(src)
    sample_rate = int(meta["sample_rate"])
    if not samples:
        _write_wav16(dst, [], sample_rate)
        after = analyze_wav(dst)
        return {"input_path": str(src), "output_path": str(dst), "before": before, "after": after}

    dc = sum(samples) / len(samples)
    cleaned = [v - dc for v in samples]

    strength = max(0.0, min(1.0, float(noise_reduction_strength)))
    frame_width = max(1, int(sample_rate * 0.02))
    frame_levels = _frame_rms(cleaned, sample_rate)
    noise_floor = _percentile(frame_levels, 20.0)
    if noise_reduce and noise_floor > 0:
        gate_threshold = max(noise_floor * (1.5 + strength * 3.0), 1e-5)
        attenuation = max(0.05, 1.0 - strength * 0.9)
        for frame_index, level in enumerate(frame_levels):
            if level >= gate_threshold:
                continue
            start = frame_index * frame_width
            end = min(len(cleaned), start + frame_width)
            scale = attenuation + (1.0 - attenuation) * min(1.0, level / gate_threshold)
            for i in range(start, end):
                cleaned[i] *= scale

    if trim_silence and cleaned:
        abs_peak = max(abs(v) for v in cleaned)
        trim_threshold = max(noise_floor * 1.8, abs_peak * 0.01, 1e-5)
        first = 0
        last = len(cleaned) - 1
        while first < len(cleaned) and abs(cleaned[first]) < trim_threshold:
            first += 1
        while last > first and abs(cleaned[last]) < trim_threshold:
            last -= 1
        pad = int(sample_rate * 0.05)
        first = max(0, first - pad)
        last = min(len(cleaned) - 1, last + pad)
        cleaned = cleaned[first:last + 1] if first < len(cleaned) else cleaned

    if normalize and cleaned:
        peak = max(abs(v) for v in cleaned)
        if peak > 0:
            gain = max(0.0, min(10.0, float(target_peak) / peak))
            cleaned = [max(-1.0, min(1.0, v * gain)) for v in cleaned]

    _write_wav16(dst, cleaned, sample_rate)
    after = analyze_wav(dst)
    return {
        "input_path": str(src),
        "output_path": str(dst),
        "created_at": time.time(),
        "settings": {
            "noise_reduce": bool(noise_reduce),
            "trim_silence": bool(trim_silence),
            "normalize": bool(normalize),
            "noise_reduction_strength": strength,
            "target_peak": float(target_peak),
        },
        "before": before,
        "after": after,
    }


def _safe_part(value: str, fallback: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value or "").strip())
    safe = safe.strip("._-")
    return safe or fallback


def extract_clip_to_wav(
    input_path: str | Path,
    output_path: str | Path,
    *,
    start_ms: int = 0,
    end_ms: int | None = None,
    clean: bool = False,
    noise_reduce: bool = True,
    trim_silence: bool = False,
    normalize: bool = True,
    noise_reduction_strength: float = 0.65,
) -> dict[str, Any]:
    """Extract an audio range from WAV or media into a training-ready mono WAV.

    PCM WAV sources are cut with the standard library. Other audio/video files
    are decoded with ffmpeg when available.
    """
    src = Path(input_path)
    dst = Path(output_path)
    if not src.is_file():
        raise FileNotFoundError(f"media file not found: {src}")
    start_s = max(0.0, int(start_ms or 0) / 1000.0)
    duration_s: float | None = None
    if end_ms is not None and int(end_ms) > int(start_ms or 0):
        duration_s = max(0.001, (int(end_ms) - int(start_ms or 0)) / 1000.0)

    tmp_out = dst.with_name(f"{dst.stem}.raw_extract{dst.suffix}")
    tmp_out.parent.mkdir(parents=True, exist_ok=True)
    if tmp_out.exists():
        tmp_out.unlink()

    if src.suffix.lower() == ".wav":
        samples, meta = _read_pcm_wav(src)
        sample_rate = int(meta["sample_rate"])
        start_i = min(len(samples), int(start_s * sample_rate))
        end_i = len(samples)
        if duration_s is not None:
            end_i = min(len(samples), start_i + int(duration_s * sample_rate))
        if end_i <= start_i:
            raise ValueError("selected audio range is empty")
        _write_wav16(tmp_out, samples[start_i:end_i], sample_rate)
    else:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to extract audio from non-WAV media")
        cmd = [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start_s:.3f}",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
        ]
        if duration_s is not None:
            cmd.extend(["-t", f"{duration_s:.3f}"])
        cmd.extend(["-acodec", "pcm_s16le", str(tmp_out)])
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "ffmpeg audio extraction failed").strip()
            raise RuntimeError(detail)

    if clean:
        result = clean_wav(
            tmp_out,
            dst,
            noise_reduce=noise_reduce,
            trim_silence=trim_silence,
            normalize=normalize,
            noise_reduction_strength=noise_reduction_strength,
        )
        try:
            tmp_out.unlink()
        except Exception:
            pass
    else:
        if dst.exists():
            dst.unlink()
        tmp_out.replace(dst)
        result = {
            "input_path": str(src),
            "output_path": str(dst),
            "before": analyze_wav(dst),
            "after": analyze_wav(dst),
            "settings": {"clean": False},
        }
    result["clip"] = {
        "source_path": str(src),
        "start_ms": int(start_ms or 0),
        "end_ms": int(end_ms) if end_ms is not None else None,
        "duration_ms": int((result.get("after") or {}).get("duration_s", 0.0) * 1000),
    }
    return result


def build_audio_dataset_clip_path(
    dataset_root: str | Path,
    *,
    split: str,
    label: str,
    source_path: str | Path,
    start_ms: int,
    end_ms: int | None = None,
) -> Path:
    """Build a stable destination path for a collected labeled clip."""
    root = Path(dataset_root)
    split_name = "val" if str(split or "").strip().lower() in {"val", "valid", "validation", "test"} else "train"
    label_name = _safe_part(label, "unlabeled")
    source = Path(source_path)
    stem = _safe_part(source.stem, "clip")
    if end_ms is None:
        suffix = f"{int(start_ms or 0)}ms"
    else:
        suffix = f"{int(start_ms or 0)}-{int(end_ms)}ms"
    return root / split_name / label_name / f"{stem}_{suffix}.wav"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze or clean PCM WAV files for CV Ops audio scenarios.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("path")
    clean = sub.add_parser("clean")
    clean.add_argument("input")
    clean.add_argument("output")
    clean.add_argument("--no-noise-reduce", action="store_true")
    clean.add_argument("--no-trim", action="store_true")
    clean.add_argument("--no-normalize", action="store_true")
    clean.add_argument("--strength", type=float, default=0.65)
    clip = sub.add_parser("clip")
    clip.add_argument("input")
    clip.add_argument("output")
    clip.add_argument("--start-ms", type=int, default=0)
    clip.add_argument("--end-ms", type=int, default=None)
    clip.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.cmd == "analyze":
        payload = analyze_wav(args.path)
    elif args.cmd == "clean":
        payload = clean_wav(
            args.input,
            args.output,
            noise_reduce=not args.no_noise_reduce,
            trim_silence=not args.no_trim,
            normalize=not args.no_normalize,
            noise_reduction_strength=args.strength,
        )
    else:
        payload = extract_clip_to_wav(
            args.input,
            args.output,
            start_ms=args.start_ms,
            end_ms=args.end_ms,
            clean=bool(args.clean),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
