"""Reproducible objective audit of generated speech audio.

This script measures what the demo actually writes to disk.  It deliberately
keeps perceptual claims out of the output: there is no MOS/CMOS here, only
signal measurements that can be compared before/after a change and across
conditions.

Measurements per clip:

* file hash, sample rate, duration, subtype;
* sample peak, true peak, RMS, integrated LUFS, crest factor, DC offset;
* frame-level noise floor, speech level, SNR proxy, silent-frame fraction;
* band energies relative to total speech energy (including 4-8 kHz and
  8-12 kHz, where broadband vocoder hiss is audible);
* spectral flatness in speech and noise-floor frames;
* median F0, F0 spread/range, voiced ratio, speech/pause statistics;
* clipping sample count and max absolute sample.

The same module is intended for before/after comparison, so output is JSON +
CSV + spectrogram PNGs and every row carries its input SHA-256.

Usage:
    python scripts/audit/audit_audio_quality.py \
        --audio-dir results/demo/research/audio \
        --output-dir results/audit/baseline_audio
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.naturalness import extract_naturalness_metrics  # noqa: E402
from voix.utils.audio import true_peak_dbfs  # noqa: E402

FRAME_SECONDS = 0.025
HOP_SECONDS = 0.010
N_FFT = 1024
BANDS_HZ: Sequence[tuple[str, float, float]] = (
    ("0_300", 0.0, 300.0),
    ("300_1000", 300.0, 1000.0),
    ("1000_2000", 1000.0, 2000.0),
    ("2000_4000", 2000.0, 4000.0),
    ("4000_6000", 4000.0, 6000.0),
    ("6000_8000", 6000.0, 8000.0),
    ("8000_10000", 8000.0, 10000.0),
    ("10000_12000", 10000.0, 12000.0),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _db(value: float, floor: float = 1e-12) -> float:
    return float(20.0 * math.log10(max(float(value), floor)))


def _frame_rms(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = max(1, int(round(FRAME_SECONDS * sample_rate)))
    hop = max(1, int(round(HOP_SECONDS * sample_rate)))
    if wav.size < frame:
        padded = np.pad(wav, (0, frame - wav.size))
        return np.array([float(np.sqrt(np.mean(padded**2)))])
    count = 1 + (wav.size - frame) // hop
    windows = np.lib.stride_tricks.as_strided(
        wav,
        shape=(count, frame),
        strides=(wav.strides[0] * hop, wav.strides[0]),
        writeable=False,
    )
    return np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1))


def _stft_power(wav: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hop = max(1, int(round(HOP_SECONDS * sample_rate)))
    window = np.hanning(N_FFT).astype(np.float64)
    if wav.size < N_FFT:
        wav = np.pad(wav, (0, N_FFT - wav.size))
    count = 1 + (wav.size - N_FFT) // hop
    frames = np.lib.stride_tricks.as_strided(
        wav,
        shape=(count, N_FFT),
        strides=(wav.strides[0] * hop, wav.strides[0]),
        writeable=False,
    )
    spectrum = np.fft.rfft(frames * window, axis=1)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(N_FFT, d=1.0 / sample_rate)
    return power, frequencies, np.arange(count) * hop


def _band_table(
    power: np.ndarray,
    frequencies: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Mean band power in dBFS, and share of total speech power in dB."""
    if not mask.any():
        return {}
    selected = power[mask]
    total = float(selected.sum(axis=1).mean()) + 1e-18
    result: Dict[str, float] = {}
    for name, low, high in BANDS_HZ:
        band = (frequencies >= low) & (frequencies < high)
        if not band.any():
            continue
        energy = float(selected[:, band].sum(axis=1).mean())
        result[f"band_{name}_dbfs"] = float(10.0 * math.log10(max(energy / max(1, int(band.sum())), 1e-18)))
        result[f"band_{name}_share_db"] = float(10.0 * math.log10(max(energy / total, 1e-18)))
    return result


def _spectral_flatness(power: np.ndarray, mask: np.ndarray) -> Optional[float]:
    if not mask.any():
        return None
    selected = np.maximum(power[mask], 1e-18)
    geometric = np.exp(np.mean(np.log(selected), axis=1))
    arithmetic = np.mean(selected, axis=1)
    flatness = geometric / np.maximum(arithmetic, 1e-18)
    return float(np.median(flatness))


def _long_term_spectrum(
    power: np.ndarray, frequencies: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(frequencies)
    mean_power = power[mask].mean(axis=0)
    return 10.0 * np.log10(np.maximum(mean_power, 1e-18))


def measure_clip(path: Path, display_name: Optional[str] = None) -> Dict[str, Any]:
    info = sf.info(str(path))
    wav, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float64).reshape(-1)
    if wav.size == 0:
        raise ValueError(f"empty audio: {path}")

    frame_rms = _frame_rms(wav, sample_rate)
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))
    noise_floor = float(np.percentile(frame_db, 10))
    speech_level = float(np.percentile(frame_db, 90))
    snr_proxy = float(speech_level - noise_floor)
    silent_threshold = speech_level - 30.0
    silent_mask = frame_db <= silent_threshold
    speech_mask = frame_db >= (speech_level - 20.0)

    power, frequencies, _ = _stft_power(wav, sample_rate)
    frame_count = min(power.shape[0], frame_db.size)
    power = power[:frame_count]
    speech_frames = np.zeros(frame_count, dtype=bool)
    speech_frames[: min(speech_mask.size, frame_count)] = speech_mask[:frame_count]
    silent_frames = np.zeros(frame_count, dtype=bool)
    silent_frames[: min(silent_mask.size, frame_count)] = silent_mask[:frame_count]

    band_speech = _band_table(power, frequencies, speech_frames)
    band_noise = _band_table(power, frequencies, silent_frames)

    metrics: Dict[str, Any] = {
        "file": display_name or path.name,
        "sha256": _sha256(path),
        "sample_rate_hz": int(sample_rate),
        "frames": int(info.frames),
        "duration_s": float(wav.size / sample_rate),
        "subtype": info.subtype,
        "channels": int(info.channels),
        "sample_peak_dbfs": _db(np.max(np.abs(wav))),
        "true_peak_dbtp": float(true_peak_dbfs(wav)),
        "rms_dbfs": _db(np.sqrt(np.mean(wav**2))),
        "dc_offset": float(np.mean(wav)),
        "crest_factor_db": _db(np.max(np.abs(wav))) - _db(np.sqrt(np.mean(wav**2))),
        "clipped_samples": int(np.sum(np.abs(wav) >= 0.999)),
        "noise_floor_dbfs": noise_floor,
        "speech_level_dbfs": speech_level,
        "snr_proxy_db": snr_proxy,
        "silent_frame_ratio": float(silent_mask.mean()) if silent_mask.size else 0.0,
        "spectral_flatness_speech": _spectral_flatness(power, speech_frames),
        "spectral_flatness_noise": _spectral_flatness(power, silent_frames),
        "frame_count": int(frame_count),
    }

    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sample_rate)
        loudness = float(meter.integrated_loudness(wav))
        metrics["lufs"] = loudness if math.isfinite(loudness) else None
    except Exception as exc:  # pragma: no cover - environment dependent
        metrics["lufs"] = None
        metrics["lufs_error"] = f"{type(exc).__name__}: {exc}"

    metrics.update(
        {
            (f"speech_{key}" if not key.startswith("band_") else f"speech_{key}"): value
            for key, value in band_speech.items()
        }
    )
    metrics.update(
        {
            (f"noise_{key}" if not key.startswith("band_") else f"noise_{key}"): value
            for key, value in band_noise.items()
        }
    )

    naturalness = extract_naturalness_metrics(wav, sample_rate=sample_rate)
    metrics.update({f"nat_{key}": value for key, value in naturalness.items()})
    metrics["long_term_spectrum_dbfs"] = _long_term_spectrum(
        power, frequencies, speech_frames
    ).tolist()
    metrics["frequencies_hz"] = frequencies.tolist()
    return metrics


def _plot_spectrogram(
    path: Path,
    output_path: Path,
    *,
    title: Optional[str] = None,
) -> None:
    import librosa
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wav, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    durations = wav.size / sample_rate
    n_fft = 1024
    hop = 128
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop)
    magnitude = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    image = librosa.display.specshow(
        magnitude,
        sr=sample_rate,
        hop_length=hop,
        x_axis="time",
        y_axis="linear",
        ax=axes[0],
        cmap="magma",
    )
    axes[0].set_ylim(0, 12000)
    axes[0].set_title(title or path.name)
    axes[0].set_ylabel("Hz")
    fig.colorbar(image, ax=axes[0], format="%+2.0f dB")

    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    mean_power = np.abs(stft).mean(axis=1) ** 2
    axes[1].semilogx(
        frequencies[1:],
        10.0 * np.log10(np.maximum(mean_power[1:], 1e-18)),
        linewidth=1.0,
    )
    axes[1].set_xlim(50, sample_rate / 2)
    axes[1].set_ylim(-90, 0)
    axes[1].set_xlabel("Hz (log)")
    axes[1].set_ylabel("Mean power (dB)")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].axvspan(4000, 8000, color="tab:orange", alpha=0.12, label="4-8 kHz")
    axes[1].axvspan(8000, 12000, color="tab:red", alpha=0.10, label="8-12 kHz")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    del durations


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    excluded = {"long_term_spectrum_dbfs", "frequencies_hz"}
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in excluded and key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _environment() -> Dict[str, Any]:
    import librosa
    import pyloudnorm
    import scipy
    import torch

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "librosa": librosa.__version__,
        "soundfile": sf.__version__,
        "pyloudnorm": getattr(pyloudnorm, "__version__", "unknown"),
    }


def _git_revision() -> Dict[str, Any]:
    import subprocess

    result: Dict[str, Any] = {}
    for key, command in (
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    ):
        try:
            result[key] = subprocess.check_output(
                command, cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            result[key] = None
    try:
        result["dirty_entries"] = int(
            subprocess.check_output(
                ["git", "status", "--short"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .splitlines()
            .__len__()
        )
    except Exception:
        result["dirty_entries"] = None
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=ROOT / "results" / "demo" / "research" / "audio",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "audit" / "baseline_audio",
    )
    parser.add_argument(
        "--label",
        default="baseline",
        help="Label stored in the report, e.g. baseline or post_fix.",
    )
    parser.add_argument(
        "--no-spectrograms",
        action="store_true",
        help="Skip spectrogram rendering (metrics only).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    audio_dir = args.audio_dir
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"audio directory not found: {audio_dir}")
    paths = sorted(audio_dir.rglob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"no WAV files in {audio_dir}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: List[Dict[str, Any]] = []
    spectrogram_dir = output_dir / "spectrograms"
    for path in paths:
        relative = path.relative_to(audio_dir).as_posix()
        print(f"[measure] {relative}", flush=True)
        row = measure_clip(path, display_name=relative)
        rows.append(row)
        if not args.no_spectrograms:
            safe_name = relative.replace("/", "__").replace("\\", "__")
            _plot_spectrogram(
                path,
                spectrogram_dir / f"{Path(safe_name).stem}.png",
                title=f"{relative} | {row['duration_s']:.2f}s | "
                f"noise floor {row['noise_floor_dbfs']:.1f} dBFS",
            )

    _write_csv(rows, output_dir / "audio_metrics.csv")
    report = {
        "schema_version": "voix.audio_audit.v1",
        "label": args.label,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "audio_dir": str(audio_dir),
        "n_files": len(rows),
        "environment": _environment(),
        "git": _git_revision(),
        "measurement_notes": [
            "noise_floor_dbfs is the 10th percentile of 25 ms frame RMS; "
            "snr_proxy_db is p90 - p10.",
            "silent frames are frames at least 30 dB below the 90th percentile "
            "frame level; their spectra are used for noise-band measurements.",
            "band_*_share_db is band power relative to total power, so a "
            "broadband hiss raises high-band shares even when overall level is "
            "normalized.",
            "spectral_flatness is the median frame geometric/arithmetic mean "
            "ratio; 1.0 is white noise.",
            "These are objective proxies, not perceptual quality or MOS.",
        ],
        "clips": rows,
    }
    (output_dir / "audio_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    # Long-term spectra are stored separately: they are useful for plots and
    # before/after comparisons but unwieldy in the main CSV.
    spectra = {
        "schema_version": "voix.audio_spectra.v1",
        "frequencies_hz": rows[0]["frequencies_hz"],
        "clips": {
            row["file"]: {
                "sha256": row["sha256"],
                "spectrum_dbfs": row["long_term_spectrum_dbfs"],
            }
            for row in rows
        },
    }
    (output_dir / "long_term_spectra.json").write_text(
        json.dumps(spectra, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[done] {len(rows)} clips in {time.time() - started:.1f}s -> {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
