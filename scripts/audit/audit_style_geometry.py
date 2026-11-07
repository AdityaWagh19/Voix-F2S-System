"""Measure the geometry of face-derived vs real-speech StyleTTS2 styles.

The synthesis matrix showed that a real reference style is ~20 dB cleaner than
the face-derived style under identical inference settings.  This script checks
whether the adapter output is out of distribution in a simple, measurable way:
norm, cosine to real styles, nearest-real distance, and covariance scale.

Usage:
    python scripts/audit/audit_style_geometry.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_FACE = ROOT / "demo" / "subject_face.png"
DEFAULT_REF_DIR = ROOT / "results" / "audit" / "reference_audio"
DEFAULT_OUT = ROOT / "results" / "audit" / "style_geometry"


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


def _style_stats(styles: np.ndarray) -> Dict[str, Any]:
    norms = np.linalg.norm(styles, axis=1)
    unit = styles / np.maximum(norms[:, None], 1e-12)
    cosine = unit @ unit.T
    values = cosine[np.triu_indices_from(cosine, k=1)]
    centered = styles - styles.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    return {
        "n": int(styles.shape[0]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "pairwise_cosine_mean": float(values.mean()),
        "pairwise_cosine_min": float(values.min()),
        "pairwise_cosine_max": float(values.max()),
        "mean_vector_norm": float(np.linalg.norm(styles.mean(axis=0))),
        "centered_rms": float(np.sqrt(np.mean(centered**2))),
        "singular_values": singular.tolist(),
        "effective_rank_90pct": int(
            np.searchsorted(
                np.cumsum(singular**2) / max(float(np.sum(singular**2)), 1e-12),
                0.90,
            )
            + 1
        ),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face", type=Path, default=DEFAULT_FACE)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REF_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT / "style_geometry.json")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from voix.inference.pipeline import VoixPipeline

    faces = sorted(
        path
        for pattern in ("C1_*.jpg", "C2_*.jpg", "C3_*.jpg")
        for path in (ROOT / "data" / "test_subjects").glob(pattern)
    )
    references = sorted(args.reference_dir.glob("*.wav"))
    if not references:
        raise FileNotFoundError(f"no reference WAVs in {args.reference_dir}")

    print("[setup] loading pipeline", flush=True)
    pipeline = VoixPipeline(
        device=args.device,
        enable_watermarking=False,
        tts_use_neural=True,
        tts_allow_fallback=False,
    )
    # Face styles first, then free the face stack before vocoder/style encoder.
    face_entries: List[Dict[str, Any]] = []
    face_prior: List[np.ndarray] = []
    face_posterior: List[np.ndarray] = []
    for face in faces:
        print(f"[face] {face.stem}", flush=True)
        e_f_raw, gender, age, _e_geo = pipeline.extract_face_features(str(face))
        e_f = pipeline.normalize_face(e_f_raw.to(pipeline.device))
        with torch.no_grad():
            e_f_voice = pipeline.cfvn(e_f, normalize=True)
            mu, _ = pipeline.cvae.prior_net(e_f_voice)
            e_s = F.normalize(pipeline.cvae.decoder(mu), dim=-1)
            style = pipeline.adapter(e_s)
            if style.dim() == 3 and style.shape[1] == 1:
                style = style.squeeze(1)
            torch.manual_seed(42)
            e_s_post = F.normalize(
                pipeline.cvae.sample_voices(e_f_voice, num_samples=1, temperature=0.7),
                dim=-1,
            )
            style_post = pipeline.adapter(e_s_post)
            if style_post.dim() == 3 and style_post.shape[1] == 1:
                style_post = style_post.squeeze(1)
        prior = style.detach().cpu().numpy().reshape(-1).astype(np.float64)
        posterior = style_post.detach().cpu().numpy().reshape(-1).astype(np.float64)
        face_prior.append(prior)
        face_posterior.append(posterior)
        face_entries.append(
            {
                "face": str(face),
                "gender": int(gender),
                "age": float(age),
                "prior_norm": float(np.linalg.norm(prior)),
                "posterior_norm": float(np.linalg.norm(posterior)),
                "prior_posterior_cosine": _cos(prior, posterior),
            }
        )

    pipeline._arcface = None
    pipeline._facemesh = None
    pipeline._demog = None
    pipeline._fusion = None
    for module_name in ("cfvn", "cvae", "adapter"):
        module = getattr(pipeline, module_name, None)
        if module is not None:
            module.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()

    print("[setup] loading StyleTTS2 for real reference styles", flush=True)
    pipeline.tts.load()
    real_entries: List[Dict[str, Any]] = []
    real_styles: List[np.ndarray] = []
    for path in references:
        print(f"[reference] {path.name}", flush=True)
        style = pipeline.tts.compute_style(str(path))
        vector = style.detach().cpu().numpy().reshape(-1).astype(np.float64)
        real_styles.append(vector)
        real_entries.append(
            {
                "file": path.name,
                "norm": float(np.linalg.norm(vector)),
            }
        )

    face_prior_array = np.asarray(face_prior)
    face_post_array = np.asarray(face_posterior)
    real_array = np.asarray(real_styles)
    real_mean = real_array.mean(axis=0)
    real_unit = real_array / np.linalg.norm(real_array, axis=1, keepdims=True)
    prior_unit = face_prior_array / np.linalg.norm(face_prior_array, axis=1, keepdims=True)
    post_unit = face_post_array / np.linalg.norm(face_post_array, axis=1, keepdims=True)

    for index, entry in enumerate(face_entries):
        prior_cos = real_unit @ prior_unit[index]
        post_cos = real_unit @ post_unit[index]
        entry["prior_nearest_real_cosine"] = float(prior_cos.max())
        entry["prior_nearest_real_file"] = real_entries[int(prior_cos.argmax())]["file"]
        entry["prior_mean_real_cosine"] = float(prior_cos.mean())
        entry["posterior_nearest_real_cosine"] = float(post_cos.max())
        entry["posterior_mean_real_cosine"] = float(post_cos.mean())
        entry["prior_to_real_mean_cosine"] = _cos(face_prior_array[index], real_mean)
        entry["posterior_to_real_mean_cosine"] = _cos(face_post_array[index], real_mean)
        entry["prior_norm_ratio_to_real_mean"] = float(
            np.linalg.norm(face_prior_array[index]) / max(np.linalg.norm(real_mean), 1e-12)
        )

    report = {
        "schema_version": "voix.style_geometry.v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "faces": face_entries,
        "references": real_entries,
        "face_prior_stats": _style_stats(face_prior_array),
        "face_posterior_stats": _style_stats(face_post_array),
        "real_reference_stats": _style_stats(real_array),
        "real_mean_vector": real_mean.tolist(),
        "face_prior_mean_vector": face_prior_array.mean(axis=0).tolist(),
        "notes": [
            "Real reference styles are computed by StyleTTS2's frozen style encoder "
            "on AMI headset speech.",
            "The face-derived style is the canonical adapter output after the "
            "craniofacial residual.",
            "This is a geometry/diagnostic report, not a perceptual quality claim.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("face_prior_stats", "face_posterior_stats", "real_reference_stats")}, indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
