"""Controlled audio-quality diagnostics for the VOIX demo.

The demo raises two recurring questions:

* Where does the residual noise come from?
* Why does the output sound like a polished audiobook narrator?

This script answers both with measurements instead of assertions.  It holds the
face, speaker embedding, text, and seed fixed, then varies one factor at a time:

* ``diffusion_steps``: 5 (the demo default), 10, and 20.
* ``emotion``: neutral versus happy (happy enables the pitch/time-stretch DSP).
* ``dsp``: the neutral style rendered with and without the librosa pitch shift
  and time stretch that the emotion path applies.
* ``save_normalization``: the raw backend waveform versus both the legacy
  peak-normalized writer and the fixed true-peak-limited writer.

The last point matters: the backend normalizes to -23 LUFS, but the legacy demo
writer then peak-normalized every asset to 0.90.  That second step overrode LUFS
and raised the noise floor of quiet conditions.  Both the legacy and fixed
writers are measured here so the historical result stays comparable.

Usage:
    python scripts/audit/diagnose_audio_quality.py --device cuda
    python scripts/audit/diagnose_audio_quality.py --steps 5 10 20 --save-audio results/audio_quality
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.utils.audio import limit_true_peak  # noqa: E402

SCHEMA = "voix.audio_quality_diagnostics.v1"
DEFAULT_TEXT = (
    "The human voice reflects both physical anatomy and lived culture, "
    "carrying the unique cadence of personal identity."
)


def _as_mono(wav: Any) -> np.ndarray:
    array = np.asarray(wav, dtype=np.float32)
    if array.ndim > 1:
        array = array.squeeze()
    if array.ndim > 1:
        array = array.mean(axis=0)
    return np.ascontiguousarray(array)


def frame_rms_db(y: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    """Frame-wise RMS in dBFS, floored at -100 dB to keep logs finite."""
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if y.size < frame:
        y = np.pad(y, (0, frame - y.size))
    n_frames = 1 + (y.size - frame) // hop
    frames = np.stack([y[i * hop : i * hop + frame] for i in range(n_frames)])
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-5))


def peak_normalize(y: np.ndarray, target: float = 0.90) -> np.ndarray:
    """Reproduce ``voix.inference.io.save_audio_sample`` peak normalization."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 1e-4:
        return y
    return (y / peak * target).astype(np.float32)


def _lufs(y: np.ndarray, sr: int) -> Optional[float]:
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        value = float(meter.integrated_loudness(y.astype(np.float64)))
        return value if np.isfinite(value) else None
    except Exception:
        return None


def measure_audio(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Measure level, noise, spectral, and pitch characteristics.

    ``noise_floor_dbfs`` is the 10th percentile of frame RMS and
    ``speech_level_dbfs`` is the 90th percentile; their difference is the
    speech-to-noise proxy used in the audit.  This is a diagnostic proxy, not a
    calibrated noise measurement.
    """
    import librosa

    y = _as_mono(y)
    if y.size == 0:
        return {
            "status": "unavailable",
            "reason": "empty waveform",
        }

    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    frames = frame_rms_db(y, sr)
    noise_floor = float(np.percentile(frames, 10))
    speech_level = float(np.percentile(frames, 90))
    voiced = frames > max(-60.0, noise_floor + 6.0)

    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    flatness = float(np.mean(spectral_flatness)) if spectral_flatness.size else None

    stft = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    power = stft**2
    total = float(np.sum(power))
    hf = float(np.sum(power[freqs >= 8000.0])) if total > 0 else 0.0
    hf_ratio = hf / total if total > 0 else None

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = float(np.mean(centroid)) if centroid.size else None

    f0 = librosa.yin(
        y,
        fmin=50.0,
        fmax=600.0,
        sr=sr,
        frame_length=2048,
        hop_length=256,
    )
    f0_values = f0[np.isfinite(f0)]
    f0_values = f0_values[(f0_values >= 50.0) & (f0_values <= 600.0)]
    f0_mean = float(np.median(f0_values)) if f0_values.size else None
    lufs_value = _lufs(y, sr)

    return {
        "status": "computed",
        "duration_seconds": round(float(y.size) / sr, 3),
        "peak": round(peak, 5),
        "clipping_fraction": round(float(np.mean(np.abs(y) >= 0.999)), 6),
        "rms_dbfs": round(20.0 * np.log10(max(rms, 1e-10)), 2),
        "crest_factor_db": round(20.0 * np.log10(max(peak, 1e-10) / max(rms, 1e-10)), 2),
        "lufs": None if lufs_value is None else round(float(lufs_value), 2),
        "noise_floor_dbfs": round(noise_floor, 2),
        "speech_level_dbfs": round(speech_level, 2),
        "snr_proxy_db": round(speech_level - noise_floor, 2),
        "silence_fraction_below_minus_60db": round(float(np.mean(frames < -60.0)), 4),
        "voiced_frame_fraction": round(float(np.mean(voiced)), 4),
        "spectral_flatness_mean": None if flatness is None else round(flatness, 6),
        "spectral_centroid_mean_hz": None if centroid_mean is None else round(centroid_mean, 1),
        "hf_energy_ratio_above_8khz": None if hf_ratio is None else round(hf_ratio, 6),
        "f0_median_hz": None if f0_mean is None else round(f0_mean, 2),
    }


def _delta(raw: Dict[str, Any], other: Dict[str, Any], key: str) -> Optional[float]:
    a = raw.get(key)
    b = other.get(key)
    if a is None or b is None:
        return None
    return round(float(b) - float(a), 2)


def _pair(
    raw: Dict[str, Any],
    normalized: Dict[str, Any],
    current: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Summarize the effect of the legacy and current demo writers."""
    paired: Dict[str, Any] = {
        "raw": raw,
        "save_peak_normalized": normalized,
        "delta_save_normalization": {
            key: _delta(raw, normalized, key)
            for key in (
                "noise_floor_dbfs",
                "speech_level_dbfs",
                "snr_proxy_db",
                "lufs",
                "crest_factor_db",
            )
        },
    }
    if current is not None:
        paired["save_current"] = current
        paired["delta_save_current"] = {
            key: _delta(raw, current, key)
            for key in (
                "noise_floor_dbfs",
                "speech_level_dbfs",
                "snr_proxy_db",
                "lufs",
                "crest_factor_db",
            )
        }
    return paired


def _save(path: Path, y: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)


def run_diagnostics(
    *,
    face: str,
    text: str,
    device: str,
    seed: int,
    steps: Iterable[int],
    emotion: str,
    emotion_intensity: float,
    save_audio: Optional[Path] = None,
) -> Dict[str, Any]:
    import torch

    from voix.inference.io import resolve_face_input
    from voix.inference.pipeline import VoixPipeline

    steps = [int(step) for step in steps]
    if not steps:
        raise ValueError("at least one diffusion step count is required")

    # Match the demo generation path: the showcase passes watermark=False, so
    # AudioSeal must not alter the audio being measured.
    pipeline = VoixPipeline(device=device, enable_watermarking=False)
    e_f_raw, gender, age, components, _ = resolve_face_input(pipeline, face)
    e_geo_np = components.get("e_geo")
    e_geo = (
        torch.from_numpy(np.asarray(e_geo_np, dtype=np.float32)).float().unsqueeze(0).to(pipeline.device)
        if e_geo_np is not None
        else None
    )

    # The demo's primary candidate is the fixed identity for every comparison.
    pipeline._inf_cfg.update({"alpha": 0.18, "beta": 0.22, "diffusion_steps": int(list(steps)[0])})
    primary = pipeline.generate(
        face_input=e_f_raw,
        e_geo=e_geo,
        gender=gender,
        age=age,
        text=text,
        num_voices=1,
        temperature=0.85,
        seed=seed,
        selection="diverse",
        watermark=False,
    )[0]
    speaker_embedding = primary["speaker_embedding"]
    neutral_style = primary["style_embedding"].unsqueeze(0)

    results: List[Dict[str, Any]] = []
    sr = int(pipeline.tts.sample_rate)

    def record(
        label: str,
        waveform: Any,
        *,
        diffusion_steps: int,
        emotion_name: Optional[str],
        dsp: str,
    ) -> None:
        raw = _as_mono(waveform.detach().cpu().numpy() if hasattr(waveform, "detach") else waveform)
        legacy_normalized = peak_normalize(raw)
        current_saved = limit_true_peak(raw)
        item = {
            "label": label,
            "diffusion_steps": diffusion_steps,
            "emotion": emotion_name,
            "dsp": dsp,
            "measurement": _pair(
                measure_audio(raw, sr),
                measure_audio(legacy_normalized, sr),
                measure_audio(current_saved, sr),
            ),
        }
        results.append(item)
        if save_audio is not None:
            _save(save_audio / f"{label}.wav", raw, sr)
            _save(save_audio / f"{label}__save_normalized.wav", legacy_normalized, sr)
            _save(save_audio / f"{label}__save_current.wav", current_saved, sr)

    for step_count in steps:
        step_count = int(step_count)
        pipeline._inf_cfg["diffusion_steps"] = step_count

        neutral = pipeline.generate(
            face_input=e_f_raw,
            e_geo=e_geo,
            gender=gender,
            age=age,
            text=text,
            num_voices=1,
            emotion=None,
            speaker_embedding=speaker_embedding,
            watermark=False,
        )[0]
        record(
            f"neutral_steps{step_count}",
            neutral["waveform"],
            diffusion_steps=step_count,
            emotion_name=None,
            dsp="none",
        )

        happy = pipeline.generate(
            face_input=e_f_raw,
            e_geo=e_geo,
            gender=gender,
            age=age,
            text=text,
            num_voices=1,
            emotion=emotion,
            emotion_intensity=emotion_intensity,
            speaker_embedding=speaker_embedding,
            watermark=False,
        )[0]
        record(
            f"{emotion}_steps{step_count}",
            happy["waveform"],
            diffusion_steps=step_count,
            emotion_name=emotion,
            dsp="pitch+time_stretch+energy",
        )

    # Isolate the librosa DSP from the style-space emotion delta.  The neutral
    # style is rendered twice with the same diffusion budget; only the pitch and
    # duration multipliers change.
    neutral_beta = 0.22
    for label, f0_scale, duration_scale in (
        ("neutral_dsp_control", 1.0, 1.0),
        ("neutral_dsp_pitch116_time092", 1.16, 0.92),
    ):
        waveform = pipeline.tts.synthesize(
            text=text,
            style=neutral_style,
            gender=gender,
            age=age,
            alpha=0.18,
            beta=neutral_beta,
            duration_scale=duration_scale,
            diffusion_steps=int(list(steps)[0]),
            f0_scale=f0_scale,
        )
        record(
            label,
            waveform,
            diffusion_steps=int(list(steps)[0]),
            emotion_name=None,
            dsp="none" if f0_scale == 1.0 and duration_scale == 1.0 else "pitch+time_stretch",
        )

    return {
        "schema_version": SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "face": str(face),
        "text": text,
        "device": device,
        "seed": seed,
        "diffusion_steps_tested": [int(s) for s in steps],
        "emotion": emotion,
        "emotion_intensity": emotion_intensity,
        "primary_style_reference": "candidate 1 from the diverse pool (seed fixed)",
        "notes": [
            "noise_floor_dbfs is the 10th percentile of 25 ms frame RMS; snr_proxy_db is p90 - p10.",
            "These are diagnostic proxies, not calibrated noise or perceptual quality measurements.",
            "save_peak_normalized reproduces the legacy save_audio_sample(), which peak-normalized to 0.90 after the backend's LUFS normalization.",
            "save_current reproduces the fixed save_audio_sample(): no peak normalization, only a -1 dBTP true-peak safety limiter.",
            "neutral_dsp_* isolates librosa pitch/time stretch from the style-space emotion delta.",
            "Watermarking is disabled to match the demo's watermark=False generation path.",
        ],
        "conditions": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure noise and polish in VOIX demo audio")
    parser.add_argument("--face", default=str(ROOT / "demo" / "subject_face.png"))
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--emotion", default="happy")
    parser.add_argument("--emotion-intensity", type=float, default=0.9)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "audio_quality_diagnostics.json",
    )
    parser.add_argument(
        "--save-audio",
        type=Path,
        default=None,
        help="Optional directory for raw, legacy peak-normalized, and current WAVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_diagnostics(
        face=args.face,
        text=args.text,
        device=args.device,
        seed=args.seed,
        steps=args.steps,
        emotion=args.emotion,
        emotion_intensity=args.emotion_intensity,
        save_audio=args.save_audio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
