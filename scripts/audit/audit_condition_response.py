"""Measure emotion/accent response after style-manifold calibration.

The calibration audit showed that translating the face-style mean onto the
gender-matched real-style mean removes most of the vocoder hiss.  This script
asks the next question: do the demo's emotion and accent residuals still work
on the calibrated style, and are the three selected emotions distinguishable?

For each selected voice it synthesizes:

* neutral;
* happy/sad/angry at two residual scales (0.28 = validated E1 scale, 0.56 =
  doubled exploratory scale);
* US/England/Australian at two A3 scales (0.3 = selected scale, 0.5 =
  exploratory extension).

All clips use the demo's exact synthesis settings, text, and seed.  The report
contains noise, high-frequency energy, F0, duration, and ECAPA similarity to
the voice's neutral.  These are objective proxies, not human ratings.

Usage:
    python scripts/audit/audit_condition_response.py
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
from audit_synthesis_matrix import _set_seed  # noqa: E402

DEFAULT_FACE_REPORT = (
    ROOT / "results" / "audit" / "face_voice_diversity" / "face_voice_diversity.json"
)
DEFAULT_BANK = ROOT / "results" / "audit" / "style_calibration" / "real_style_bank.json"
DEFAULT_EMOTION = ROOT / "checkpoints" / "emotion_residuals" / "class_mean.pt"
DEFAULT_ACCENT = ROOT / "checkpoints" / "accent_residuals" / "class_mean.pt"
DEFAULT_OUT = ROOT / "results" / "audit" / "condition_response"
ECAPA_DIR = ROOT / "checkpoints" / "speechbrain" / "spkrec-ecapa"
DEFAULT_FACES = (
    "C2_12_black_female_Lupita_Nyongo",
    "C1_02_young_male_Timothee_Chalamet",
    "C2_09_east_asian_male_Jackie_Chan",
)
DEFAULT_TEXT = (
    "The human voice carries identity, emotion, and the rhythm of lived "
    "experience."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ecapa_embedding(model, wav: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != 16000:
        import librosa

        wav = librosa.resample(
            wav.astype(np.float32), orig_sr=sample_rate, target_sr=16000
        )
    tensor = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_batch(tensor).squeeze().detach().cpu().numpy()
    vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _synthesize(
    tts,
    *,
    text: str,
    style: torch.Tensor,
    gender: int,
    age: float,
    seed: int,
    steps: int,
    alpha: float,
    beta: float,
    f0_scale: float = 1.0,
):
    _set_seed(seed)
    with torch.no_grad():
        return tts.synthesize(
            text=text,
            style=style,
            gender=int(gender),
            age=float(age),
            alpha=float(alpha),
            beta=float(beta),
            duration_scale=1.0,
            diffusion_steps=int(steps),
            f0_scale=float(f0_scale),
            prosody_bias_strength=0.0,
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", nargs="*", default=list(DEFAULT_FACES))
    parser.add_argument("--face-report", type=Path, default=DEFAULT_FACE_REPORT)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--emotion-artifact", type=Path, default=DEFAULT_EMOTION)
    parser.add_argument("--accent-artifact", type=Path, default=DEFAULT_ACCENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--beta", type=float, default=0.22)
    parser.add_argument("--calibration-alpha", type=float, default=1.0)
    parser.add_argument("--emotion-scales", nargs="*", type=float, default=[0.28, 0.56])
    parser.add_argument("--accent-scales", nargs="*", type=float, default=[0.3, 0.5])
    parser.add_argument("--emotions", nargs="*", default=["happy", "sad", "angry"])
    parser.add_argument(
        "--accents", nargs="*", default=["us", "england", "australian"]
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    face_report = json.loads(args.face_report.read_text(encoding="utf-8"))
    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    entries = {entry["label"]: entry for entry in face_report["entries"]}
    missing = [label for label in args.faces if label not in entries]
    if missing:
        raise KeyError(f"faces missing from report: {missing}")

    by_gender: Dict[str, List[np.ndarray]] = {"female": [], "male": []}
    for entry in face_report["entries"]:
        gender = "female" if entry["gender"] == 0 else "male"
        by_gender[gender].append(np.asarray(entry["style_vector"], dtype=np.float64))
    face_mean = {
        gender: np.mean(np.asarray(vectors), axis=0)
        for gender, vectors in by_gender.items()
    }
    real_mean = {
        gender: np.asarray(bank["by_gender"][gender]["mean_vector"], dtype=np.float64)
        for gender in ("female", "male")
    }

    emotion_payload = torch.load(args.emotion_artifact, map_location="cpu", weights_only=False)
    accent_payload = torch.load(args.accent_artifact, map_location="cpu", weights_only=False)
    from voix.models.emotion_residual import apply_style_residual, residual_from_artifact
    from voix.models.accent_residual import apply_accent_residual

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    from voix.models.tts_backend import StyleTTS2Backend
    from speechbrain.inference.speaker import EncoderClassifier

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
    neutral_embeddings: Dict[str, np.ndarray] = {}

    def add_row(
        *,
        name: str,
        face_label: str,
        gender: int,
        age: float,
        track: str,
        condition: str,
        scale: Optional[float],
        style: torch.Tensor,
        elapsed: float,
    ) -> None:
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
            "scale": scale,
            "style_norm": float(style.norm().item()),
            "style_sha256": hashlib.sha256(
                np.asarray(style.detach().cpu()).round(8).tobytes()
            ).hexdigest(),
            "wav_path": str(wav_path),
            "audio_sha256": _sha256(wav_path),
            "audio_duration_s": float(metrics["duration_s"]),
            "noise_floor_dbfs": float(metrics["noise_floor_dbfs"]),
            "snr_proxy_db": float(metrics["snr_proxy_db"]),
            "speech_band_4000_6000_share_db": float(metrics["speech_band_4000_6000_share_db"]),
            "speech_band_6000_8000_share_db": float(metrics["speech_band_6000_8000_share_db"]),
            "speech_band_8000_10000_share_db": float(metrics["speech_band_8000_10000_share_db"]),
            "speech_band_10000_12000_share_db": float(metrics["speech_band_10000_12000_share_db"]),
            "lufs": metrics.get("lufs"),
            "true_peak_dbtp": metrics.get("true_peak_dbtp"),
            "nat_f0_median_hz": metrics.get("nat_f0_median_hz"),
            "nat_f0_std_st": metrics.get("nat_f0_std_st"),
            "nat_f0_range_st": metrics.get("nat_f0_range_st"),
            "nat_pause_ratio": metrics.get("nat_pause_ratio"),
            "nat_speech_duration_s": metrics.get("nat_speech_duration_s"),
            "elapsed_s": float(elapsed),
            "ecapa_vector": embedding.tolist(),
        }
        rows.append(row)
        print(
            f"[row] {name} | noise {row['noise_floor_dbfs']:.1f} | "
            f"F0 {row['nat_f0_median_hz']:.1f} | dur {row['audio_duration_s']:.2f}",
            flush=True,
        )

    for face_label in args.faces:
        entry = entries[face_label]
        gender = int(entry["gender"])
        age = float(entry["age"])
        gender_key = "female" if gender == 0 else "male"
        delta = real_mean[gender_key] - face_mean[gender_key]
        base = np.asarray(entry["style_vector"], dtype=np.float64) + float(
            args.calibration_alpha
        ) * delta
        base_style = torch.from_numpy(base).float().unsqueeze(0)
        voice = face_label.split("_")[0] + "_" + face_label.split("_")[1]
        print(f"[voice] {face_label} calibrated style norm {base_style.norm().item():.2f}", flush=True)

        started = time.time()
        wav = _synthesize(
            tts,
            text=args.text,
            style=base_style,
            gender=gender,
            age=age,
            seed=args.seed,
            steps=args.diffusion_steps,
            alpha=args.alpha,
            beta=args.beta,
        )
        neutral_path = audio_dir / f"{voice}__neutral.wav"
        sf.write(str(neutral_path), wav.detach().cpu().numpy().reshape(-1), sample_rate, subtype="PCM_16")
        add_row(
            name=f"{voice}__neutral",
            face_label=face_label,
            gender=gender,
            age=age,
            track="baseline",
            condition="neutral",
            scale=None,
            style=base_style,
            elapsed=time.time() - started,
        )
        neutral_embeddings[face_label] = np.asarray(rows[-1]["ecapa_vector"])

        for emotion in args.emotions:
            residual = residual_from_artifact(emotion_payload, emotion, variant="prosody_only")
            for scale in args.emotion_scales:
                style = apply_style_residual(base_style, residual, intensity=1.0, scale=float(scale))
                name = f"{voice}__emotion_{emotion}_s{scale:g}"
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
                destination = audio_dir / f"{name}.wav"
                sf.write(str(destination), wav.detach().cpu().numpy().reshape(-1), sample_rate, subtype="PCM_16")
                add_row(
                    name=name,
                    face_label=face_label,
                    gender=gender,
                    age=age,
                    track="emotion",
                    condition=emotion,
                    scale=float(scale),
                    style=style,
                    elapsed=time.time() - started,
                )

        for accent in args.accents:
            residual = torch.as_tensor(accent_payload["residuals"][accent]).float().reshape(-1)
            for scale in args.accent_scales:
                style = apply_accent_residual(base_style, residual, scale=float(scale), timbre_weight=0.25)
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
                destination = audio_dir / f"{name}.wav"
                sf.write(str(destination), wav.detach().cpu().numpy().reshape(-1), sample_rate, subtype="PCM_16")
                add_row(
                    name=name,
                    face_label=face_label,
                    gender=gender,
                    age=age,
                    track="accent",
                    condition=accent,
                    scale=float(scale),
                    style=style,
                    elapsed=time.time() - started,
                )

    by_face: Dict[str, Dict[str, Any]] = {}
    for face_label in args.faces:
        neutral = next(row for row in rows if row["face_label"] == face_label and row["track"] == "baseline")
        neutral_vec = np.asarray(neutral["ecapa_vector"])
        deltas: Dict[str, Any] = {}
        for row in rows:
            if row["face_label"] != face_label or row["track"] == "baseline":
                continue
            key = f"{row['track']}:{row['condition']}:s{row['scale']:g}"
            vec = np.asarray(row["ecapa_vector"])
            deltas[key] = {
                "f0_delta_hz": None
                if neutral["nat_f0_median_hz"] is None or row["nat_f0_median_hz"] is None
                else float(row["nat_f0_median_hz"] - neutral["nat_f0_median_hz"]),
                "duration_delta_s": float(row["audio_duration_s"] - neutral["audio_duration_s"]),
                "noise_delta_db": float(row["noise_floor_dbfs"] - neutral["noise_floor_dbfs"]),
                "ecapa_to_neutral": float(np.dot(neutral_vec, vec)),
            }
        by_face[face_label] = {"neutral": neutral, "conditions": deltas}

    report = {
        "schema_version": "voix.condition_response.v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "text": args.text,
        "seed": int(args.seed),
        "diffusion_steps": int(args.diffusion_steps),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "prosody_bias_strength": 0.0,
        "calibration_alpha": float(args.calibration_alpha),
        "faces": args.faces,
        "emotion_scales": [float(x) for x in args.emotion_scales],
        "accent_scales": [float(x) for x in args.accent_scales],
        "face_report": str(args.face_report),
        "face_report_sha256": _sha256(args.face_report),
        "bank": str(args.bank),
        "bank_sha256": _sha256(args.bank),
        "emotion_artifact": str(args.emotion_artifact),
        "emotion_artifact_sha256": _sha256(args.emotion_artifact),
        "accent_artifact": str(args.accent_artifact),
        "accent_artifact_sha256": _sha256(args.accent_artifact),
        "conditions": rows,
        "by_face": by_face,
        "notes": [
            "neutral is the calibrated face style with no emotion/accent residual.",
            "0.28 is the E1-validated emotion scale; 0.56 is a doubled exploratory scale.",
            "0.3 is the A3-selected accent scale; 0.5 is an exploratory extension.",
            "ECAPA similarity is to the same voice's neutral; real AMI between-speaker "
            "same-gender median is about 0.115 (see ecapa_reference.json).",
            "No human MOS/CMOS or emotion/accent identification study was run.",
        ],
    }
    output = args.output_dir / "condition_response.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print("\nresponse summary")
    for face_label, block in by_face.items():
        neutral = block["neutral"]
        print(
            f"\n{face_label} | neutral noise {neutral['noise_floor_dbfs']:.1f} "
            f"F0 {neutral['nat_f0_median_hz']:.1f} dur {neutral['audio_duration_s']:.2f}"
        )
        for key, value in block["conditions"].items():
            print(
                f"  {key:>34s} | dF0 {value['f0_delta_hz']:+6.1f} | "
                f"ddur {value['duration_delta_s']:+6.2f} | "
                f"dnoise {value['noise_delta_db']:+5.1f} | "
                f"ECAPA {value['ecapa_to_neutral']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
