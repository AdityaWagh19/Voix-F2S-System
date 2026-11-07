"""Video/audio I/O helpers for VoxCeleb2 corpus extraction.

The extraction pipeline needs evenly sampled frames with timestamps and a
16 kHz mono WAV for the speaker encoder.  Keeping this logic in one module
makes it reusable by the pilot, the full rebuild, and tests.

Performance notes
-----------------
VoxCeleb2 clips are short (typically 4-12 s) but reading every decoded frame
just to keep 8-16 of them wastes most of the video-decode budget.  The sampler
uses ``VideoCapture.grab`` for skipped frames and ``read`` only for the frames
we actually keep.  On clips with a reliable frame count this is several times
faster than a naive ``read`` loop.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


_FFMPEG_EXE: Optional[str] = None
_LAST_AUDIO_ERROR: Optional[str] = None


def last_audio_error() -> Optional[str]:
    """Return the reason for the most recent ``extract_audio`` failure.

    Extraction runs in a single prefetch thread, so a module-level value is
    sufficient and keeps the public ``bool`` return type unchanged.
    """
    return _LAST_AUDIO_ERROR


def get_ffmpeg_exe() -> str:
    """Return the bundled imageio-ffmpeg binary, or a system ffmpeg."""
    global _FFMPEG_EXE
    if _FFMPEG_EXE is not None:
        return _FFMPEG_EXE
    try:
        import imageio_ffmpeg

        _FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
        return _FFMPEG_EXE
    except ImportError:
        pass

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        _FFMPEG_EXE = ffmpeg
        return _FFMPEG_EXE
    raise RuntimeError(
        "ffmpeg not found. Install imageio-ffmpeg or add ffmpeg to PATH."
    )


def extract_audio(
    video_path: str | Path,
    wav_path: str | Path,
    *,
    sample_rate: int = 16000,
    max_seconds: Optional[float] = 12.0,
    timeout_sec: float = 120.0,
) -> bool:
    """Extract mono 16 kHz audio from a video into ``wav_path``.

    Returns True only when ffmpeg succeeds and the resulting WAV contains more
    than a bare header.  A timeout is treated as a failed clip rather than
    hanging the extraction worker.
    """
    global _LAST_AUDIO_ERROR
    _LAST_AUDIO_ERROR = None
    cmd = [
        get_ffmpeg_exe(),
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
    ]
    if max_seconds is not None:
        cmd.extend(["-t", f"{float(max_seconds):.3f}"])
    cmd.extend(
        [
            "-vn",
            "-ar",
            str(int(sample_rate)),
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
            str(wav_path),
        ]
    )
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=max(1.0, float(timeout_sec)),
        )
    except subprocess.TimeoutExpired:
        _LAST_AUDIO_ERROR = "timeout"
        return False
    except OSError as exc:
        _LAST_AUDIO_ERROR = f"oserror:{exc}"
        return False
    if result.returncode != 0:
        _LAST_AUDIO_ERROR = f"ffmpeg_rc:{result.returncode}"
        return False
    if not os.path.exists(wav_path):
        _LAST_AUDIO_ERROR = "missing_wav"
        return False
    if os.path.getsize(wav_path) <= 44:
        _LAST_AUDIO_ERROR = "empty_wav"
        return False
    return True


def _sample_indices(total_frames: int, max_frames: int) -> list[int]:
    """Evenly spaced frame indices, always including first and last."""
    limit = max(1, int(total_frames))
    count = max(1, min(int(max_frames), limit))
    if count == 1:
        return [0]
    picks = np.linspace(0, limit - 1, count)
    return sorted({int(round(float(p))) for p in picks})


def _open_capture(
    video_path: str | Path,
    *,
    open_timeout_ms: int = 15000,
    read_timeout_ms: int = 15000,
):
    """Open a video with FFmpeg timeouts, falling back to the default backend."""
    import cv2

    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        int(open_timeout_ms),
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        int(read_timeout_ms),
    ]
    try:
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG, params)
    except Exception:
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
    return cap


def load_sampled_frames(
    video_path: str | Path,
    *,
    max_frames: int = 16,
    max_seconds: float = 12.0,
    open_timeout_sec: float = 15.0,
    read_timeout_sec: float = 15.0,
) -> tuple[list[np.ndarray], list[float], dict]:
    """Read evenly sampled BGR frames and their timestamps.

    Frames are sampled across the first ``max_seconds`` of the clip.  When the
    container reports a reliable frame count, only the selected frames are
    decoded; otherwise the function falls back to sequential reading.

    Returns:
        frames: list of HxWx3 uint8 arrays.
        times: timestamp in seconds for each frame.
        meta: fps, total_frames, duration_sec, sampled_frames, sample_indices.
    """
    import cv2

    cap = _open_capture(
        video_path,
        open_timeout_ms=int(open_timeout_sec * 1000),
        read_timeout_ms=int(read_timeout_sec * 1000),
    )
    if not cap.isOpened():
        return [], [], {}

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames: list[np.ndarray] = []
    times: list[float] = []
    indices: list[int] = []

    if total_frames > 0:
        limit = min(total_frames, max(1, int(round(float(max_seconds) * fps))))
        indices = _sample_indices(limit, max_frames)
        idx = 0
        for target in indices:
            while idx < target:
                if not cap.grab():
                    break
                idx += 1
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            times.append(idx / fps)
            idx += 1
        duration_sec = limit / fps
    else:
        # Unknown frame count: read sequentially and keep every step-th frame.
        max_total = max(1, int(round(float(max_seconds) * fps)))
        step = max(1, int(round(max_total / max(1, int(max_frames)))))
        idx = 0
        while idx < max_total:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append(frame)
                times.append(idx / fps)
                indices.append(idx)
            idx += 1
        duration_sec = idx / fps

    cap.release()

    meta = {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": float(duration_sec),
        "sampled_frames": len(frames),
        "sample_indices": indices,
    }
    return frames, times, meta


def audio_duration_sec(wav_path: str | Path) -> float:
    """Return WAV duration in seconds, or 0.0 if unreadable."""
    try:
        import soundfile as sf

        info = sf.info(str(wav_path))
        return float(info.duration)
    except Exception:
        return 0.0
