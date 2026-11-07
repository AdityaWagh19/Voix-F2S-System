"""Reusable inference I/O helpers.

These functions used to live in the legacy ``voix.inference.cli`` demo runner.
They are kept here so the supported CLI, diagnostics, and tests share one
waveform-writing and face-resolution path without importing a retired CLI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import soundfile as sf
import torch

from voix.utils.audio import limit_true_peak


def compute_audio_metrics(
    waveform: np.ndarray,
    sample_rate: int = 24000,
) -> Dict[str, Any]:
    """Calculate basic acoustic characteristics for a waveform."""
    wav = np.asarray(waveform).flatten().astype(np.float32)
    duration = float(len(wav) / sample_rate)
    peak = float(np.max(np.abs(wav))) if len(wav) > 0 else 0.0
    rms = float(np.sqrt(np.mean(wav ** 2))) if len(wav) > 0 else 0.0

    mean_f0 = 0.0
    f0_std = 0.0
    voiced_pct = 0.0

    if len(wav) > sample_rate // 10:
        try:
            import librosa

            f0, voiced, _ = librosa.pyin(
                wav,
                fmin=60,
                fmax=500,
                sr=sample_rate,
                frame_length=2048,
            )
            if voiced is not None and np.any(voiced):
                valid_f0 = f0[voiced]
                mean_f0 = float(np.nanmean(valid_f0))
                f0_std = float(np.nanstd(valid_f0))
                voiced_pct = float(np.mean(voiced)) * 100.0
        except Exception:
            try:
                corr = np.correlate(wav, wav, mode="full")[len(wav) - 1 :]
                min_lag = int(sample_rate / 400)
                max_lag = int(sample_rate / 70)
                if max_lag < len(corr):
                    peak_lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
                    mean_f0 = float(sample_rate / peak_lag) if peak_lag > 0 else 0.0
                    voiced_pct = 85.0
            except Exception:
                mean_f0 = 0.0

    return {
        "duration_seconds": round(duration, 3),
        "peak_amplitude": round(peak, 4),
        "rms_energy": round(rms, 4),
        "f0_mean_hz": round(mean_f0, 1),
        "f0_std_hz": round(f0_std, 1),
        "voiced_percentage": round(voiced_pct, 1),
    }


def save_audio_sample(
    waveform: Union[torch.Tensor, np.ndarray],
    output_path: Path,
    sample_rate: int = 24000,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Write a waveform with a true-peak limiter, without re-normalizing.

    The neural backend has already targeted an integrated loudness. Applying
    peak normalization here would defeat that target and raise the noise floor
    of quiet conditions. The limiter leaves audio below -1 dBTP untouched.
    """
    if isinstance(waveform, torch.Tensor):
        wav_np = waveform.detach().cpu().numpy().squeeze()
    else:
        wav_np = np.asarray(waveform).squeeze()

    wav_np = limit_true_peak(wav_np)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), wav_np, sample_rate)
    return wav_np, compute_audio_metrics(wav_np, sample_rate)


def resolve_face_input(
    pipeline: Any,
    face_input_str: str,
) -> Tuple[torch.Tensor, int, float, Dict[str, Any], Dict[str, Any]]:
    """Resolve a face image path or demographic preset ID for inference."""
    face_path = Path(face_input_str)

    if face_path.is_file():
        e_f, gender, age, e_geo = pipeline.extract_face_features(str(face_path))
        e_f_np = e_f.detach().cpu().numpy().squeeze()
        e_id = e_f_np[:512] if len(e_f_np) >= 512 else np.zeros(512, dtype=np.float32)
        e_geo_np = (
            e_geo.cpu().numpy()
            if hasattr(e_geo, "cpu")
            else np.asarray(e_geo, dtype=np.float32)
        )
        e_demo = e_f_np[544:] if len(e_f_np) >= 560 else np.zeros(16, dtype=np.float32)
        components = {
            "e_id": e_id,
            "e_geo": e_geo_np,
            "morphology": e_geo_np,
            "e_demo": e_demo,
            "gender": gender,
            "age": age,
        }
        import cv2

        img = cv2.imread(str(face_path))
        height, width = (img.shape[0], img.shape[1]) if img is not None else (0, 0)
        subject_info = {
            "input_type": "image_upload",
            "image_filename": face_path.name,
            "image_resolution": f"{width}x{height}",
        }
        return e_f, gender, age, components, subject_info

    try:
        preset = pipeline.get_demographic_preset(face_input_str)
    except KeyError as exc:
        raise FileNotFoundError(
            f"Input '{face_input_str}' is neither an existing file nor a recognized preset."
        ) from exc

    gender = preset.gender
    age = preset.age
    e_f = preset.embedding.to(pipeline.device)
    e_f_np = e_f.detach().cpu().numpy().squeeze()
    e_id = e_f_np[:512] if len(e_f_np) >= 512 else np.zeros(512, dtype=np.float32)
    e_geo = (
        np.asarray(preset.geometry, dtype=np.float32)
        if preset.geometry is not None
        else np.zeros(32, dtype=np.float32)
    )
    e_demo = e_f_np[544:] if len(e_f_np) >= 560 else np.zeros(16, dtype=np.float32)
    components = {
        "e_id": e_id,
        "e_geo": e_geo,
        "morphology": e_geo,
        "e_demo": e_demo,
        "gender": gender,
        "age": age,
    }
    subject_info = {
        "input_type": "demographic_preset",
        "preset_id": preset.preset_id,
        "display_name": preset.display_name,
        "description": preset.description,
        "image_filename": f"{preset.preset_id}.preset",
        "image_resolution": "synthetic_vector_560d",
    }
    return e_f, gender, age, components, subject_info
