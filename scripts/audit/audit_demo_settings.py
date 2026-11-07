"""Test demo-level extensions before locking the final research demo.

The condition-response audit showed that, on calibrated face-derived voices:

* happy and angry residuals raise F0, but angry shortens duration more;
* the sad residual lengthens speech but does not lower F0 (the E1 real-voice
  effect is -11.6 Hz);
* accent residuals at scale 0.3/0.5 change F0 and duration only slightly.

This script tests two explicit demo-level extensions:

1. sad residual plus a post-synthesis ``f0_scale`` chosen to reproduce the E1
   sad F0 direction (ratio 0.93/0.90), and larger sad residual scales;
2. accent scales above the A3-selected 0.3/0.5, to see whether the residual
   produces a larger acoustic change without collapsing identity.

Every clip is compared with the voice's neutral from the condition-response
audit.  The output is exploratory evidence for choosing demo settings; it is
not a perceptual validation and it does not modify the serving path.

Usage:
    python scripts/audit/audit_demo_settings.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_audio_quality import measure_clip  # noqa: E402
from audit_condition_response import (  # noqa: E402
    _ecapa_embedding,
    _sha256,
    _synthesize,
)

DEFAULT_FACE_REPORT = (
    ROOT / "results" / "audit" / "face_voice_diversity" / "face_voice_diversity.json"
)
DEFAULT_CALIBRATION = (
    ROOT / "results" / "audit" / "style_calibration" / "calibration.json"
)
DEFAULT_CONDITION_RESPONSE = (
    ROOT / "results" / "audit" / "condition_response" / "condition_response.json"
)
DEFAULT_EMOTION = ROOT / "checkpoints" / "emotion_residuals" / "class_mean.pt"
DEFAULT_ACCENT = ROOT / "checkpoints" / "accent_residuals" / "class_mean.pt"
DEFAULT_OUT = ROOT / "results" / "audit" / "demo_settings"
ECAPA_DIR = ROOT / "checkpoints" / "speechbrain" / "spkrec-ecapa"
DEFAULT_FACES = (
    "C2_12_black_female_Lupita_Nyongo",
    "C1_02_young_male_Timothee_Chalamet",
    "C2_09_east_asian_male_Jackie_Chan",
)
DEFAULT_ACCENTS = ("us", "england", "australian")
DEFAULT_TEXT = (
    "The human voice carries identity, emotion, and the rhythm of lived "
    "experience."
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", nargs="*", default=list(DEFAULT_FACES))
    parser.add_argument("--accents", nargs="*", default=list(DEFAULT_ACCENTS))
    parser.add_argument("--face-report", type=Path, default=DEFAULT_FACE_REPORT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--condition-response", type=Path, default=DEFAULT_CONDITION_RESPONSE
    )
    parser.add_argument("--emotion-artifact", type=Path, default=DEFAULT_EMOTION)
    parser.add_argument("--accent-artifact", type=Path, default=DEFAULT_ACCENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--beta", type=float, default=0.22)
    parser.add_argument("--calibration-alpha", type=float, default=1.0)
    parser.add_argument(
        "--sad-scales", nargs="*", type=float, default=[0.56, 0.84, 1.12]
    )
    parser.add_argument(
        "--sad-f0-scales", nargs="*", type=float, default=[0.93, 0.90]
    )
    parser.add_argument(
        "--accent-scales", nargs="*", type=float, default=[1.0, 1.5]
    )
    parser.add_argument("--accent-timbre-weight", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _add_row(
    *,
    rows: List[Dict[str, Any]],
    audio_dir: Path,
    name: str,
    face_label: str,
    gender: int,
    age: float,
    track: str,
    condition: str,
    style: torch.Tensor,
    residual: Optional[Dict[str, Any]],
    f0_scale: float,
    neutral_embedding: np.ndarray,
    neutral_metrics: Dict[str, Any],
    ecapa,
    sample_rate: int,
    elapsed: float,
) -> Dict[str, Any]:
    wav_path = audio_dir / f"{name}.wav"
    wav, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
    embedding = _ecapa_embedding(ecapa, np.asarray(wav).reshape(-1), sample_rate)
    metrics = measure_clip(wav_path, display_name=f"{name}.wav")
    row = {
        "name": name,
        "face_label": face_label,
        "gender": int(gender),
        "age": float(age),
        "track": track,
        "condition": condition,
        "style_norm": float(style.norm().item()),
        "style_sha256": hashlib.sha256(
            np.asarray(style.detach().cpu()).round(8).tobytes()
        ).hexdigest(),
        "residual": residual,
        "f0_scale": float(f0_scale),
        "wav_path": str(wav_path),
        "audio_sha256": _sha256(wav_path),
        "audio_duration_s": float(metrics["duration_s"]),
        "f0_delta_hz": (
            None
            if neutral_metrics.get("nat_f0_median_hz") is None
            or metrics.get("nat_f0_median_hz") is None
            else float(
                metrics["nat_f0_median_hz"]
                - neutral_metrics["nat_f0_median_hz"]
            )
        ),
        "duration_delta_s": float(
            metrics["duration_s"] - neutral_metrics["duration_s"]
        ),
        "noise_floor_dbfs": float(metrics["noise_floor_dbfs"]),
        "noise_delta_db": float(
            metrics["noise_floor_dbfs"] - neutral_metrics["noise_floor_dbfs"]
        ),
        "snr_proxy_db": float(metrics["snr_proxy_db"]),
        "speech_band_4000_6000_share_db": float(
            metrics["speech_band_4000_6000_share_db"]
        ),
        "speech_band_6000_8000_share_db": float(
            metrics["speech_band_6000_8000_share_db"]
        ),
        "speech_band_8000_10000_share_db": float(
            metrics["speech_band_8000_10000_share_db"]
        ),
        "speech_band_10000_12000_share_db": float(
            metrics["speech_band_10000_12000_share_db"]
        ),
        "nat_f0_median_hz": metrics.get("nat_f0_median_hz"),
        "nat_f0_std_st": metrics.get("nat_f0_std_st"),
        "nat_f0_range_st": metrics.get("nat_f0_range_st"),
        "nat_pause_ratio": metrics.get("nat_pause_ratio"),
        "lufs": metrics.get("lufs"),
        "true_peak_dbtp": metrics.get("true_peak_dbtp"),
        "ecapa_to_neutral": float(np.dot(neutral_embedding, embedding)),
        "ecapa_vector": embedding.tolist(),
        "elapsed_s": float(elapsed),
    }
    rows.append(row)
    print(
        f"[row] {name} | noise {row['noise_floor_dbfs']:.1f} | "
        f"dF0 {row['f0_delta_hz']:+6.1f} | dDur {row['duration_delta_s']:+5.2f} | "
        f"ECAPA {row['ecapa_to_neutral']:.3f}",
        flush=True,
    )
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    face_report = json.loads(args.face_report.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    condition_response = json.loads(
        args.condition_response.read_text(encoding="utf-8")
    )
    entries = {entry["label"]: entry for entry in face_report["entries"]}
    missing = [label for label in args.faces if label not in entries]
    if missing:
        raise KeyError(f"faces missing from report: {missing}")

    from voix.models.accent_residual import apply_accent_residual
    from voix.models.emotion_residual import apply_style_residual, residual_from_artifact
    from voix.models.style_calibration import apply_translation

    emotion_payload = torch.load(
        args.emotion_artifact, map_location="cpu", weights_only=False
    )
    accent_payload = torch.load(
        args.accent_artifact, map_location="cpu", weights_only=False
    )

    neutral_rows = {
        row["face_label"]: row
        for row in condition_response["conditions"]
        if row["track"] == "baseline"
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    from speechbrain.inference.speaker import EncoderClassifier
    from voix.models.tts_backend import StyleTTS2Backend

    print("[setup] loading StyleTTS2", flush=True)
    tts = StyleTTS2Backend(device=args.device, use_neural=True, allow_fallback=False)
    tts.load()
    if tts._model is None or tts._use_fallback:
        raise RuntimeError("StyleTTS2 failed to load")
    sample_rate = int(tts.sample_rate)

    print("[setup] loading ECAPA", flush=True)
    ecapa = EncoderClassifier.from_hparams(
        source=str(ECAPA_DIR),
        savedir=str(ECAPA_DIR),
        run_opts={"device": args.device},
    )

    rows: List[Dict[str, Any]] = []
    for face_label in args.faces:
        entry = entries[face_label]
        neutral = neutral_rows[face_label]
        neutral_embedding = np.asarray(neutral["ecapa_vector"], dtype=np.float64)
        neutral_metrics = {
            "duration_s": neutral["audio_duration_s"],
            "noise_floor_dbfs": neutral["noise_floor_dbfs"],
            "nat_f0_median_hz": neutral["nat_f0_median_hz"],
        }
        gender = int(entry["gender"])
        age = float(entry["age"])
        gender_key = "female" if gender == 0 else "male"
        base = torch.as_tensor(entry["style_vector"], dtype=torch.float32).reshape(1, 256)
        base = apply_translation(
            base,
            calibration,
            gender_key,
            alpha=float(args.calibration_alpha),
        )
        voice = face_label.split("_")[0] + "_" + face_label.split("_")[1]
        print(
            f"[voice] {face_label} calibrated style norm {base.norm().item():.2f}",
            flush=True,
        )

        sad_residual = residual_from_artifact(
            emotion_payload, "sad", variant="prosody_only"
        )
        for scale in args.sad_scales:
            style = apply_style_residual(
                base, sad_residual, intensity=1.0, scale=float(scale)
            )
            f0_values = (1.0, *args.sad_f0_scales) if float(scale) == 0.56 else (1.0,)
            for f0_scale in f0_values:
                name = f"{voice}__sad_s{scale:g}_f0{f0_scale:g}"
                print(f"[synth] {name}", flush=True)
                started = time.time()
                wav = _synthesize(
                    tts,
                    text=args.text,
                    style=style,
                    gender=gender,
                    age=age,
                    seed=args.seed,
                    steps=args.diffusion_steps,
                    alpha=args.alpha,
                    beta=args.beta,
                    f0_scale=float(f0_scale),
                )
                sf.write(
                    str(audio_dir / f"{name}.wav"),
                    wav.detach().cpu().numpy().reshape(-1),
                    sample_rate,
                    subtype="PCM_16",
                )
                _add_row(
                    rows=rows,
                    audio_dir=audio_dir,
                    name=name,
                    face_label=face_label,
                    gender=gender,
                    age=age,
                    track="sad_extension",
                    condition="sad",
                    style=style,
                    residual={
                        "artifact": "emotion_class_mean",
                        "variant": "prosody_only",
                        "scale": float(scale),
                        "norm": float(sad_residual.norm().item()),
                        "post_f0_scale": float(f0_scale),
                    },
                    f0_scale=float(f0_scale),
                    neutral_embedding=neutral_embedding,
                    neutral_metrics=neutral_metrics,
                    ecapa=ecapa,
                    sample_rate=sample_rate,
                    elapsed=time.time() - started,
                )

        for accent in args.accents:
            residual = torch.as_tensor(accent_payload["residuals"][accent]).float().reshape(-1)
            for scale in args.accent_scales:
                style = apply_accent_residual(
                    base,
                    residual,
                    scale=float(scale),
                    timbre_weight=float(args.accent_timbre_weight),
                )
                name = f"{voice}__accent_{accent}_s{scale:g}"
                print(f"[synth] {name}", flush=True)
                started = time.time()
                wav = _synthesize(
                    tts,
                    text=args.text,
                    style=style,
                    gender=gender,
                    age=age,
                    seed=args.seed,
                    steps=args.diffusion_steps,
                    alpha=args.alpha,
                    beta=args.beta,
                )
                sf.write(
                    str(audio_dir / f"{name}.wav"),
                    wav.detach().cpu().numpy().reshape(-1),
                    sample_rate,
                    subtype="PCM_16",
                )
                _add_row(
                    rows=rows,
                    audio_dir=audio_dir,
                    name=name,
                    face_label=face_label,
                    gender=gender,
                    age=age,
                    track="accent_extension",
                    condition=accent,
                    style=style,
                    residual={
                        "artifact": "accent_class_mean",
                        "scale": float(scale),
                        "timbre_weight": float(args.accent_timbre_weight),
                        "norm": float(residual.norm().item()),
                    },
                    f0_scale=1.0,
                    neutral_embedding=neutral_embedding,
                    neutral_metrics=neutral_metrics,
                    ecapa=ecapa,
                    sample_rate=sample_rate,
                    elapsed=time.time() - started,
                )

    summary: Dict[str, Any] = {}
    for face_label in args.faces:
        neutral = neutral_rows[face_label]
        voice = face_label.split("_")[0] + "_" + face_label.split("_")[1]
        summary[face_label] = {
            "voice": voice,
            "neutral": {
                "noise_floor_dbfs": neutral["noise_floor_dbfs"],
                "f0_median_hz": neutral["nat_f0_median_hz"],
                "duration_s": neutral["audio_duration_s"],
            },
            "sad": [
                {
                    "name": row["name"],
                    "scale": row["residual"]["scale"],
                    "post_f0_scale": row["residual"]["post_f0_scale"],
                    "f0_delta_hz": row["f0_delta_hz"],
                    "duration_delta_s": row["duration_delta_s"],
                    "noise_delta_db": row["noise_delta_db"],
                    "ecapa_to_neutral": row["ecapa_to_neutral"],
                }
                for row in rows
                if row["face_label"] == face_label and row["track"] == "sad_extension"
            ],
            "accents": [
                {
                    "name": row["name"],
                    "accent": row["condition"],
                    "scale": row["residual"]["scale"],
                    "f0_delta_hz": row["f0_delta_hz"],
                    "duration_delta_s": row["duration_delta_s"],
                    "f0_range_delta_st": (
                        None
                        if row.get("nat_f0_range_st") is None
                        or neutral.get("nat_f0_range_st") is None
                        else float(
                            row["nat_f0_range_st"] - neutral["nat_f0_range_st"]
                        )
                    ),
                    "noise_delta_db": row["noise_delta_db"],
                    "ecapa_to_neutral": row["ecapa_to_neutral"],
                }
                for row in rows
                if row["face_label"] == face_label
                and row["track"] == "accent_extension"
            ],
        }

    report = {
        "schema_version": "voix.demo_settings_audit.v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "text": args.text,
        "seed": int(args.seed),
        "diffusion_steps": int(args.diffusion_steps),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "calibration_alpha": float(args.calibration_alpha),
        "faces": args.faces,
        "sad_scales": [float(x) for x in args.sad_scales],
        "sad_f0_scales": [float(x) for x in args.sad_f0_scales],
        "accent_scales": [float(x) for x in args.accent_scales],
        "face_report": str(args.face_report),
        "face_report_sha256": _sha256(args.face_report),
        "calibration": str(args.calibration),
        "calibration_sha256": _sha256(args.calibration),
        "condition_response": str(args.condition_response),
        "condition_response_sha256": _sha256(args.condition_response),
        "emotion_artifact": str(args.emotion_artifact),
        "emotion_artifact_sha256": _sha256(args.emotion_artifact),
        "accent_artifact": str(args.accent_artifact),
        "accent_artifact_sha256": _sha256(args.accent_artifact),
        "conditions": rows,
        "by_face": summary,
        "notes": [
            "Neutral clips and embeddings are reused from the condition-response audit.",
            "Sad post-synthesis f0_scale is a demo-level compensation for the E1 F0 direction; it is not part of the class-mean residual.",
            "Accent scales above 0.5 are exploratory extensions beyond the A3-selected 0.3.",
            "No human MOS/CMOS or emotion/accent identification study was run.",
        ],
    }
    output = args.output_dir / "demo_settings.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
