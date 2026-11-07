"""Audit whether different faces map to genuinely different voices.

The demo claims one face-derived identity per card, so the face -> voice path
must not collapse to a small set of default voices.  This script measures the
canonical path end to end:

    face image -> fused face vector -> CFVN -> CVAE prior mean -> StyleAdapter
    -> StyleTTS2 style vector -> synthesized waveform -> ECAPA embedding

It reports pairwise cosine similarity at every stage.  A high style cosine or
a high ECAPA cosine between different faces is direct evidence of voice
convergence; low values are evidence of separation.  This is an objective
speaker-similarity audit, not a human identity judgment.

Usage:
    python scripts/audit/audit_face_voice_diversity.py
    python scripts/audit/audit_face_voice_diversity.py --skip-synthesis
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_audio_quality import measure_clip  # noqa: E402

DEFAULT_TEXT = (
    "The human voice carries identity, emotion, and the rhythm of lived "
    "experience."
)
DEFAULT_OUT = ROOT / "results" / "audit" / "face_voice_diversity"
ECAPA_DIR = ROOT / "checkpoints" / "speechbrain" / "spkrec-ecapa"


def _cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def _pair_summary(matrix: np.ndarray, labels: Sequence[str]) -> Dict[str, Any]:
    count = matrix.shape[0]
    pairs: List[Dict[str, Any]] = []
    for i in range(count):
        for j in range(i + 1, count):
            pairs.append(
                {
                    "a": labels[i],
                    "b": labels[j],
                    "cosine": float(matrix[i, j]),
                }
            )
    values = [pair["cosine"] for pair in pairs]
    nearest: Dict[str, Dict[str, Any]] = {}
    for i, label in enumerate(labels):
        others = [
            (float(matrix[i, j]), labels[j])
            for j in range(count)
            if j != i
        ]
        if others:
            best = max(others)
            nearest[label] = {"nearest": best[1], "cosine": best[0]}
    return {
        "n_pairs": len(pairs),
        "min": float(np.min(values)) if values else None,
        "p10": float(np.percentile(values, 10)) if values else None,
        "median": float(np.median(values)) if values else None,
        "p90": float(np.percentile(values, 90)) if values else None,
        "max": float(np.max(values)) if values else None,
        "mean": float(np.mean(values)) if values else None,
        "pairs": pairs,
        "nearest_neighbor": nearest,
        "n_pairs_above_0_90": int(sum(value > 0.90 for value in values)),
        "n_pairs_above_0_80": int(sum(value > 0.80 for value in values)),
    }


def _farthest_point_selection(
    vectors: np.ndarray, labels: Sequence[str], k: int
) -> List[str]:
    """Greedy farthest-point sampling in normalized cosine space."""
    normalized = vectors / np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12
    )
    similarity = normalized @ normalized.T
    # Start at the point with the lowest mean similarity to all others: the
    # most unusual face in the pool, not an arbitrary first row.
    start = int(np.argmin(similarity.mean(axis=1)))
    selected = [start]
    while len(selected) < min(k, len(labels)):
        remaining = [index for index in range(len(labels)) if index not in selected]
        # Pick the point whose maximum similarity to the selected set is lowest.
        scores = [
            (max(float(similarity[index, chosen]) for chosen in selected), index)
            for index in remaining
        ]
        selected.append(min(scores)[1])
    return [labels[index] for index in selected]


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
    parser.add_argument(
        "--faces",
        nargs="*",
        type=Path,
        default=None,
        help="Face images; defaults to data/test_subjects/C1_*, C2_*, C3_*.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--beta", type=float, default=0.22)
    parser.add_argument("--skip-synthesis", action="store_true")
    parser.add_argument("--select-k", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.faces is None:
        faces = sorted(
            path
            for pattern in ("C1_*.jpg", "C2_*.jpg", "C3_*.jpg")
            for path in (ROOT / "data" / "test_subjects").glob(pattern)
        )
    else:
        faces = list(args.faces)
    if not faces:
        raise FileNotFoundError("no face images found")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from voix.inference.pipeline import VoixPipeline

    print(f"[setup] loading pipeline for {len(faces)} faces", flush=True)
    pipeline = VoixPipeline(
        device=args.device,
        enable_watermarking=False,
        tts_use_neural=not args.skip_synthesis,
        tts_allow_fallback=False,
    )
    tts = pipeline.tts
    if not args.skip_synthesis:
        tts.load()
        if tts._model is None or getattr(tts, "_use_fallback", True):
            raise RuntimeError("neural StyleTTS2 failed to load")

    ecapa = None
    if not args.skip_synthesis:
        from speechbrain.inference.speaker import EncoderClassifier

        print("[setup] loading ECAPA", flush=True)
        ecapa = EncoderClassifier.from_hparams(
            source=str(ECAPA_DIR),
            savedir=str(ECAPA_DIR),
            run_opts={"device": args.device},
        )

    entries: List[Dict[str, Any]] = []
    style_vectors: List[np.ndarray] = []
    es_vectors: List[np.ndarray] = []
    efv_vectors: List[np.ndarray] = []
    ecapa_vectors: List[np.ndarray] = []
    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for face in faces:
        label = face.stem
        print(f"[face] {label}", flush=True)
        started = time.time()
        e_f_raw, gender, age, _e_geo = pipeline.extract_face_features(str(face))
        e_f = pipeline.normalize_face(e_f_raw.to(pipeline.device))
        with torch.no_grad():
            e_f_voice = pipeline.cfvn(e_f, normalize=True)
            mu_prior, logvar = pipeline.cvae.prior_net(e_f_voice)
            e_s = F.normalize(pipeline.cvae.decoder(mu_prior), dim=-1)
            style = pipeline.adapter(e_s)
            if style.dim() == 3 and style.shape[1] == 1:
                style = style.squeeze(1)

        entry: Dict[str, Any] = {
            "label": label,
            "face": str(face),
            "face_sha256": hashlib.sha256(face.read_bytes()).hexdigest(),
            "gender": int(gender),
            "age": float(age),
            "style_norm": float(style.norm().item()),
            "e_f_voice_norm": float(e_f_voice.norm().item()),
            "e_s_norm": float(e_s.norm().item()),
            "prior_mu_norm": float(mu_prior.norm().item()),
            "prior_logvar_mean": float(logvar.mean().item()),
            "style_sha256": hashlib.sha256(
                np.asarray(style.detach().cpu()).round(8).tobytes()
            ).hexdigest(),
            "style_vector": style.detach().cpu().numpy().reshape(-1).tolist(),
            "e_s_vector": e_s.detach().cpu().numpy().reshape(-1).tolist(),
            "e_f_voice_vector": e_f_voice.detach().cpu().numpy().reshape(-1).tolist(),
        }
        style_vectors.append(entry["style_vector"])
        es_vectors.append(entry["e_s_vector"])
        efv_vectors.append(entry["e_f_voice_vector"])

        if not args.skip_synthesis:
            torch.manual_seed(int(args.seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(args.seed))
            np.random.seed(int(args.seed))
            with torch.no_grad():
                wav = tts.synthesize(
                    text=args.text,
                    style=style.detach(),
                    gender=int(gender),
                    age=float(age),
                    alpha=float(args.alpha),
                    beta=float(args.beta),
                    duration_scale=1.0,
                    diffusion_steps=int(args.diffusion_steps),
                    f0_scale=1.0,
                    prosody_bias_strength=0.0,
                )
            audio = wav.detach().cpu().numpy().reshape(-1).astype(np.float32)
            wav_path = audio_dir / f"{label}.wav"
            sf.write(str(wav_path), audio, int(tts.sample_rate), subtype="PCM_16")
            entry["wav_path"] = str(wav_path)
            entry["audio_sha256"] = hashlib.sha256(wav_path.read_bytes()).hexdigest()
            entry["audio_metrics"] = measure_clip(wav_path, display_name=f"{label}.wav")
            entry["ecapa_vector"] = _ecapa_embedding(
                ecapa, audio, int(tts.sample_rate)
            ).tolist()
            ecapa_vectors.append(entry["ecapa_vector"])
        entry["elapsed_s"] = float(time.time() - started)
        entries.append(entry)
        print(
            f"[face] {label} done in {entry['elapsed_s']:.1f}s "
            f"(gender={gender}, age={age:.1f}, style_norm={entry['style_norm']:.2f})",
            flush=True,
        )

    labels = [entry["label"] for entry in entries]
    matrices: Dict[str, Any] = {}
    for name, vectors in (
        ("style", style_vectors),
        ("e_s", es_vectors),
        ("e_f_voice", efv_vectors),
        ("ecapa", ecapa_vectors),
    ):
        if not vectors:
            continue
        matrix = _cosine_matrix(np.asarray(vectors, dtype=np.float64))
        matrices[name] = {
            "labels": labels,
            "matrix": matrix.tolist(),
            "summary": _pair_summary(matrix, labels),
        }

    selection: Dict[str, Any] = {}
    if style_vectors and args.select_k >= 2:
        style_array = np.asarray(style_vectors, dtype=np.float64)
        style_pick = _farthest_point_selection(style_array, labels, args.select_k)
        selection["farthest_style"] = style_pick
        if es_vectors:
            es_pick = _farthest_point_selection(
                np.asarray(es_vectors, dtype=np.float64), labels, args.select_k
            )
            selection["farthest_e_s"] = es_pick
        if ecapa_vectors:
            ecapa_pick = _farthest_point_selection(
                np.asarray(ecapa_vectors, dtype=np.float64), labels, args.select_k
            )
            selection["farthest_ecapa"] = ecapa_pick

    report = {
        "schema_version": "voix.face_voice_diversity.v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "text": args.text,
        "seed": int(args.seed),
        "diffusion_steps": int(args.diffusion_steps),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "prosody_bias_strength": 0.0,
        "n_faces": len(entries),
        "synthesis": not args.skip_synthesis,
        "entries": entries,
        "similarity": matrices,
        "selection": selection,
        "interpretation": [
            "Cosine similarity is computed after L2 normalization at each stage.",
            "Style cosine compares 256-D StyleTTS2 conditioning vectors.",
            "ECAPA cosine compares synthesized audio embeddings; >0.90 is "
            "near-duplicate voice territory for this model, not proof of "
            "perceptual identity.",
            "No human identity or naturalness judgment was collected.",
        ],
    }
    (args.output_dir / "face_voice_diversity.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output_dir / 'face_voice_diversity.json'}", flush=True)
    for name, block in matrices.items():
        summary = block["summary"]
        print(
            f"[similarity] {name}: n={summary['n_pairs']} "
            f"min={summary['min']:.3f} median={summary['median']:.3f} "
            f"max={summary['max']:.3f} >0.90={summary['n_pairs_above_0_90']}",
            flush=True,
        )
    for name, pick in selection.items():
        print(f"[selection] {name}: {pick}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
