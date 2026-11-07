"""Test style-manifold calibration on synthesized audio.

The face-derived StyleAdapter output sits off the real StyleTTS2 style manifold
(mean cosine ~0.65 against the gender-matched real mean).  A simple translation

    s' = s + alpha * (mean_real_gender - mean_face_gender)

preserves pairwise face differences exactly while moving the whole face-style
cloud toward real speech.  This script synthesizes the same text and seed for
several faces and alphas, then measures noise, high-frequency energy, F0, and
ECAPA speaker similarity.

This is an exploratory demo calibration, not a serving change.

Usage:
    python scripts/audit/audit_style_calibration.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_audio_quality import measure_clip  # noqa: E402
from audit_synthesis_matrix import (  # noqa: E402
    DEFAULT_TEXT,
    _normalized,
    _raw_inference,
    _save,
    _set_seed,
)

DEFAULT_FACE_REPORT = (
    ROOT / "results" / "audit" / "face_voice_diversity" / "face_voice_diversity.json"
)
DEFAULT_BANK = (
    ROOT / "results" / "audit" / "style_calibration" / "real_style_bank.json"
)
DEFAULT_OUT = ROOT / "results" / "audit" / "style_calibration"
ECAPA_DIR = ROOT / "checkpoints" / "speechbrain" / "spkrec-ecapa"
DEFAULT_FACES = (
    "C2_12_black_female_Lupita_Nyongo",
    "C1_02_young_male_Timothee_Chalamet",
    "C1_06_senior_male_Morgan_Freeman",
    "C2_09_east_asian_male_Jackie_Chan",
    "C2_14_caucasian_female_Scarlett_Johansson",
)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    )


def _ecapa_embedding(model, wav: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != 16000:
        import librosa

        wav = librosa.resample(
            wav.astype(np.float32), orig_sr=sample_rate, target_sr=16000
        )
    tensor = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_batch(tensor).squeeze().detach().cpu().numpy()
    embedding = np.asarray(embedding, dtype=np.float64).reshape(-1)
    return embedding / max(float(np.linalg.norm(embedding)), 1e-12)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", nargs="*", default=list(DEFAULT_FACES))
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.0, 0.5, 1.0, 1.25])
    parser.add_argument("--face-report", type=Path, default=DEFAULT_FACE_REPORT)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--alpha-ab", type=float, default=0.18)
    parser.add_argument("--beta-ab", type=float, default=0.22)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    face_report = json.loads(args.face_report.read_text(encoding="utf-8"))
    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    faces_by_label = {entry["label"]: entry for entry in face_report["entries"]}
    missing = [label for label in args.faces if label not in faces_by_label]
    if missing:
        raise KeyError(f"faces not in report: {missing}")

    face_vectors = {
        label: np.asarray(entry["style_vector"], dtype=np.float64)
        for label, entry in faces_by_label.items()
    }
    by_gender: Dict[str, List[np.ndarray]] = {"female": [], "male": []}
    for label, entry in faces_by_label.items():
        by_gender["female" if entry["gender"] == 0 else "male"].append(
            face_vectors[label]
        )
    face_mean = {
        gender: np.mean(np.asarray(vectors), axis=0)
        for gender, vectors in by_gender.items()
    }
    real_mean = {
        gender: np.asarray(bank["by_gender"][gender]["mean_vector"], dtype=np.float64)
        for gender in ("female", "male")
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.output_dir / "audio"
    raw_dir = args.output_dir / "raw"
    audio_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

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
    for label in args.faces:
        entry = faces_by_label[label]
        gender = "female" if entry["gender"] == 0 else "male"
        base = face_vectors[label]
        delta = real_mean[gender] - face_mean[gender]
        for alpha in args.alphas:
            style = torch.from_numpy(base + float(alpha) * delta).float()
            style = style.unsqueeze(0)
            name = f"{label}__shift{alpha:g}"
            print(
                f"[synth] {name} | style norm {style.norm().item():.2f} | "
                f"cos to real mean {_cos(style.numpy().reshape(-1), real_mean[gender]):.3f}",
                flush=True,
            )
            started = time.time()
            raw = _raw_inference(
                tts,
                text=args.text,
                style=style,
                alpha=args.alpha_ab,
                beta=args.beta_ab,
                diffusion_steps=args.diffusion_steps,
                seed=args.seed,
            )
            raw_path = raw_dir / f"{name}.wav"
            raw_sha = _save(raw_path, raw, sample_rate)
            normalized = _normalized(raw, sample_rate)
            wav_path = audio_dir / f"{name}.wav"
            audio_sha = _save(wav_path, normalized, sample_rate)
            embedding = _ecapa_embedding(ecapa, normalized, sample_rate)
            rows.append(
                {
                    "name": name,
                    "face_label": label,
                    "gender": gender,
                    "age": float(entry["age"]),
                    "alpha": float(alpha),
                    "style_norm": float(style.norm().item()),
                    "style_cos_to_real_mean": _cos(
                        style.numpy().reshape(-1), real_mean[gender]
                    ),
                    "style_sha256": hashlib.sha256(
                        np.asarray(style.numpy()).round(8).tobytes()
                    ).hexdigest(),
                    "raw_path": str(raw_path),
                    "raw_sha256": raw_sha,
                    "wav_path": str(wav_path),
                    "audio_sha256": audio_sha,
                    "raw_metrics": measure_clip(raw_path, display_name=f"{name}.raw.wav"),
                    "metrics": measure_clip(wav_path, display_name=f"{name}.wav"),
                    "ecapa_vector": embedding.tolist(),
                    "elapsed_s": float(time.time() - started),
                }
            )
            print(
                f"[synth] {name} done | noise "
                f"{rows[-1]['metrics']['noise_floor_dbfs']:.1f} dBFS | "
                f"4-8k {rows[-1]['metrics']['speech_band_4000_6000_share_db']:.1f}/"
                f"{rows[-1]['metrics']['speech_band_6000_8000_share_db']:.1f}",
                flush=True,
            )

    # Pairwise ECAPA similarity across the same face and across faces.
    labels = [row["name"] for row in rows]
    vectors = np.asarray([row["ecapa_vector"] for row in rows])
    cosine = vectors @ vectors.T
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            pairs.append(
                {
                    "a": labels[i],
                    "b": labels[j],
                    "same_face": rows[i]["face_label"] == rows[j]["face_label"],
                    "cosine": float(cosine[i, j]),
                }
            )

    report = {
        "schema_version": "voix.style_calibration_audit.v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "text": args.text,
        "seed": int(args.seed),
        "diffusion_steps": int(args.diffusion_steps),
        "alpha_ab": float(args.alpha_ab),
        "beta_ab": float(args.beta_ab),
        "faces": args.faces,
        "alphas": [float(a) for a in args.alphas],
        "face_report": str(args.face_report),
        "face_report_sha256": hashlib.sha256(args.face_report.read_bytes()).hexdigest(),
        "bank": str(args.bank),
        "bank_sha256": hashlib.sha256(args.bank.read_bytes()).hexdigest(),
        "face_mean_by_gender": {k: v.tolist() for k, v in face_mean.items()},
        "real_mean_by_gender": {k: v.tolist() for k, v in real_mean.items()},
        "conditions": rows,
        "ecapa_pairs": pairs,
        "notes": [
            "alpha=0 is the uncalibrated face-derived style.",
            "The translation preserves pairwise face-style differences exactly; "
            "it only moves the cloud's mean.",
            "No human MOS/CMOS was collected; these are objective proxies.",
        ],
    }
    (args.output_dir / "style_calibration_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / 'style_calibration_audit.json'}")
    print("\nnoise / high-frequency summary")
    for row in rows:
        m = row["metrics"]
        print(
            f"{row['name']:>60s} | noise {m['noise_floor_dbfs']:>6.1f} | "
            f"4-8k {m['speech_band_4000_6000_share_db']:>6.1f}/"
            f"{m['speech_band_6000_8000_share_db']:>6.1f} | "
            f"8-12k {m['speech_band_8000_10000_share_db']:>6.1f}/"
            f"{m['speech_band_10000_12000_share_db']:>6.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
