"""Generate the E0 fixed-delta CREMA-D paired evaluation audio.

This is a research-only, identity-conditioned generator.  It never loads the
CFVN, CVAE, face extractors, or the canonical face-to-voice style adapter.  For
each actor it takes one real neutral CREMA-D clip, extracts the frozen
StyleTTS2 style/predictor vector from that clip, and applies the existing fixed
emotion delta plus the existing post-synthesis scalar modifiers.

The canonical face-to-voice adapter expects a face embedding, not an ECAPA
embedding.  Feeding it ECAPA audio embeddings is out of distribution and was
measured to collapse speaker identity (neutral-to-real ECAPA 0.067), so this
experiment uses StyleTTS2's own frozen style encoder instead.  The face-to-voice
mapping remains untouched.

The output index is one row per generated WAV.  Neutral audio is generated once
per ``(actor, sentence, seed)`` and shared by every emotion row.

Examples::

    # Smoke: one actor, two sentences, one seed
    python scripts/demo/generate_crema_e0.py --max-pairs 2 --seeds 0

    # Full E0 on the validation grid, one seed
    python scripts/demo/generate_crema_e0.py --seeds 0

    # Final comparison, three seeds
    python scripts/demo/generate_crema_e0.py --seeds 0 1 2

Fear and disgust have no fixed-delta artifact in this checkout.  They are
reported as ``unavailable`` rather than silently rendered as neutral.
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
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.crema_paired import (  # noqa: E402
    CREMA_EMOTIONS,
    NON_NEUTRAL_EMOTIONS,
    validate_paired_grid,
)

DEFAULT_MANIFEST = ROOT / "data" / "processed" / "crema_d_paired_eval_val.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "emotion" / "e0" / "audio"
DEFAULT_INDEX = ROOT / "results" / "emotion" / "e0" / "generation_index.csv"
DEFAULT_REPORT = ROOT / "results" / "emotion" / "e0" / "generation_report.json"
DEFAULT_RESIDUAL_ARTIFACT = (
    ROOT / "checkpoints" / "emotion_residuals" / "class_mean.pt"
)
DEFAULT_ADAPTER_ARTIFACT = (
    ROOT / "checkpoints" / "emotion_adapter" / "e2_adapter_seed0.pt"
)
GENERATION_SCHEMA = "voix.crema_generation.v3"

INDEX_COLUMNS = (
    "audio_id",
    "condition",
    "emotion",
    "intensity",
    "actor_id",
    "speaker_id",
    "sentence",
    "text",
    "seed",
    "audio_path",
    "neutral_audio_id",
    "neutral_audio_path",
    "identity_clip_id",
    "identity_path",
    "neutral_clip_id",
    "neutral_path",
    "emotion_clip_id",
    "emotion_path",
    "has_real_emotion_clip",
    "diffusion_steps",
    "alpha",
    "beta",
    "duration_scale",
    "f0_scale",
    "energy_scale",
    "scalar_modifier_source",
    "style_delta_l2",
    "style_prosody_delta_l2",
    "residual_variant",
    "residual_raw_norm",
    "residual_applied_norm",
    "residual_target_norm",
    "residual_scale",
    "residual_artifact_sha256",
    "identity_style_sha256",
    "identity_style_norm",
    "generation_seconds",
    "audio_duration_sec",
    "sample_rate",
    "backend",
    "gender",
    "age",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: "Any") -> str:
    array = tensor.detach().cpu().numpy().astype(np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve(path: str) -> Path:
    candidate = Path(str(path))
    return candidate if candidate.is_absolute() else ROOT / candidate


def _gender_code(sex: str) -> int:
    value = str(sex).strip().lower()
    if value.startswith("f"):
        return 0
    if value.startswith("m"):
        return 1
    return 1


def _load_tts(device: str):
    from voix.models.tts_backend import StyleTTS2Backend

    tts = StyleTTS2Backend(
        device=device,
        use_neural=True,
        allow_fallback=False,
    )
    tts.load()
    if tts.backend_name != "neural":
        raise RuntimeError(
            "StyleTTS2 did not load the neural backend; refusing to generate "
            f"research audio from the fallback ({tts.last_error})"
        )
    return tts


def _synthesize(
    tts,
    *,
    text: str,
    style,
    gender: int,
    age: float,
    alpha: float,
    beta: float,
    duration_scale: float,
    f0_scale: float,
    diffusion_steps: int,
    seed: int,
):
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))
    with torch.no_grad():
        waveform = tts.synthesize(
            text=text,
            style=style,
            gender=int(gender),
            age=float(age),
            alpha=float(alpha),
            beta=float(beta),
            duration_scale=float(duration_scale),
            diffusion_steps=int(diffusion_steps),
            f0_scale=float(f0_scale),
        )
    return waveform


def _scalar_modifiers(modulator, emotion: str, mode: str = "emotion"):
    """Return post-synthesis modifiers and record where they came from.

    Happy/sad/angry have an emotion profile.  Fear/disgust have no profile in
    this checkout, so they use the neutral profile rather than silently
    borrowing another emotion's timing or pitch.  The source is written to the
    generation index so the fallback is visible in every report.

    ``mode="neutral"`` is the controlled condition: every emotion uses the
    neutral profile, so the style residual is the only emotion variable.  It
    exists because the emotion profiles change pitch, duration, and energy
    after synthesis and can pass a direction gate without any style change.
    """
    if mode == "neutral":
        mods = modulator.get_acoustic_modifiers(emotion="neutral", intensity=1.0)
        return dict(mods), "neutral_control"
    try:
        mods = modulator.get_acoustic_modifiers(emotion=emotion, intensity=1.0)
        return dict(mods), "emotion_profile"
    except ValueError:
        mods = modulator.get_acoustic_modifiers(emotion="neutral", intensity=1.0)
        return dict(mods), "neutral_fallback"


def _apply_energy(waveform, energy_scale: float):
    import torch

    from voix.utils.audio import limit_true_peak

    if abs(float(energy_scale) - 1.0) <= 1e-4:
        return waveform
    wav_np = waveform.detach().cpu().numpy().squeeze() * float(energy_scale)
    return torch.from_numpy(limit_true_peak(wav_np)).unsqueeze(0).to(waveform.device)


def _save_waveform(path: Path, waveform, sample_rate: int) -> float:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    wav_np = waveform.detach().cpu().numpy().squeeze().astype(np.float32)
    sf.write(str(path), wav_np, int(sample_rate), subtype="PCM_16")
    return float(len(wav_np) / float(sample_rate))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate E0 fixed-delta CREMA-D audio")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--condition",
        choices=("fixed_delta", "class_mean", "learned_residual"),
        default="fixed_delta",
        help=(
            "Style-delta source: fixed deltas, the E1 class mean, or the E2 "
            "learned adapter"
        ),
    )
    parser.add_argument(
        "--residual-artifact",
        type=Path,
        default=DEFAULT_RESIDUAL_ARTIFACT,
    )
    parser.add_argument(
        "--adapter-artifact",
        type=Path,
        default=DEFAULT_ADAPTER_ARTIFACT,
        help="E2 learned adapter artifact (also used for the label-shuffled control)",
    )
    parser.add_argument(
        "--residual-variant",
        choices=("full", "prosody_only"),
        default="prosody_only",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=None,
        help="Override the artifact scale (default: artifact metadata or 0.28)",
    )
    parser.add_argument(
        "--match-fixed-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rescale each class-mean residual to the fixed-delta norm (default: on)",
    )
    parser.add_argument(
        "--scalar-modifiers",
        choices=("emotion", "neutral"),
        default="emotion",
        help=(
            "Post-synthesis scalar modifiers: 'emotion' uses each emotion's "
            "profile; 'neutral' holds the neutral profile for every emotion so "
            "the style residual is the only variable"
        ),
    )
    parser.add_argument("--condition-label", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--beta", type=float, default=0.22)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument(
        "--emotions",
        default="happy,sad,angry",
        help="Comma-separated non-neutral emotions. Fear/disgust have no fixed delta.",
    )
    parser.add_argument("--actors", default=None, help="Optional comma-separated actor ids")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--check-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify that every manifest audio path exists (default: on)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    import torch

    from voix.models.emotion_adapter import (
        apply_learned_residual,
        load_adapter,
        predict_residual,
    )
    from voix.models.emotion_control import (
        EmotionModulator,
        apply_emotion_style_delta,
        get_emotion_style_delta,
    )
    from voix.models.emotion_residual import (
        apply_style_residual,
        load_artifact,
        residual_from_artifact,
    )

    if not args.manifest.is_file():
        raise FileNotFoundError(f"paired evaluation manifest not found: {args.manifest}")

    frame = pd.read_csv(args.manifest)
    errors = validate_paired_grid(frame, check_audio=args.check_audio)
    if errors:
        raise ValueError("invalid paired evaluation manifest: " + "; ".join(errors))

    requested = [item.strip().lower() for item in args.emotions.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(NON_NEUTRAL_EMOTIONS))
    if unknown:
        raise ValueError(f"unsupported non-neutral emotions: {unknown}")

    condition_name = args.condition_label or (
        "fixed_delta"
        if args.condition == "fixed_delta"
        else (
            f"class_mean_{args.residual_variant}"
            if args.condition == "class_mean"
            else "learned_residual"
        )
    )

    residual_payload = None
    adapter_model = None
    adapter_payload = None
    residual_artifact_sha = ""
    residual_scale = 0.28
    fixed_norms: Dict[str, float] = {}
    if args.condition == "class_mean":
        if not args.residual_artifact.is_file():
            raise FileNotFoundError(
                f"class-mean residual artifact not found: {args.residual_artifact}"
            )
        residual_payload = load_artifact(args.residual_artifact)
        residual_artifact_sha = _sha256(args.residual_artifact)
        metadata = residual_payload.get("metadata", {})
        residual_scale = (
            float(args.residual_scale)
            if args.residual_scale is not None
            else float(metadata.get("scale", 0.28))
        )
        available = [
            emotion
            for emotion in requested
            if emotion in residual_payload["emotions"]
        ]
        unavailable = {
            emotion: "not present in the class-mean residual artifact"
            for emotion in requested
            if emotion not in residual_payload["emotions"]
        }
    elif args.condition == "learned_residual":
        if not args.adapter_artifact.is_file():
            raise FileNotFoundError(
                f"learned adapter artifact not found: {args.adapter_artifact}"
            )
        adapter_model, adapter_payload = load_adapter(args.adapter_artifact)
        residual_artifact_sha = _sha256(args.adapter_artifact)
        # The adapter output is already bounded by max_norm; do not apply an
        # extra fixed-delta scale on top of it.
        residual_scale = 1.0
        available = [
            emotion
            for emotion in requested
            if emotion in adapter_payload["emotions"]
        ]
        unavailable = {
            emotion: "not present in the learned adapter artifact"
            for emotion in requested
            if emotion not in adapter_payload["emotions"]
        }
    else:
        # A missing delta silently leaves the style unchanged in the runtime
        # helper.  Check explicitly so a missing artifact is never rendered as
        # neutral.
        available = []
        unavailable = {}
        for emotion in requested:
            delta = get_emotion_style_delta(emotion, device="cpu")
            if delta is None:
                unavailable[emotion] = (
                    f"no fixed-delta artifact at checkpoints/emotion_deltas/{emotion}.pt"
                )
            else:
                available.append(emotion)
    if not available:
        raise RuntimeError(
            f"none of the requested emotions is available for {condition_name}: "
            + "; ".join(f"{key}: {value}" for key, value in unavailable.items())
        )

    # Match the class-mean displacement to the fixed-delta norm so E1 compares
    # direction, not a different expression strength.  Fear/disgust have no
    # fixed artifact, so they use the mean norm across the available deltas.
    if args.condition == "class_mean" and args.match_fixed_norm:
        for emotion in NON_NEUTRAL_EMOTIONS:
            delta = get_emotion_style_delta(emotion, device="cpu")
            if delta is not None:
                fixed_norms[emotion] = float(delta.norm().item())
        if not fixed_norms:
            raise RuntimeError("cannot match fixed-delta norms: no fixed deltas found")
    mean_fixed_norm = (
        float(np.mean(list(fixed_norms.values()))) if fixed_norms else 1.0
    )

    actors = None
    if args.actors:
        actors = {item.strip() for item in args.actors.split(",") if item.strip()}
        frame = frame[frame["actor_id"].astype(str).isin(actors)].copy()
        if frame.empty:
            raise ValueError(f"no rows for actors {sorted(actors)}")

    # One generation unit per (actor, sentence); neutral is generated once.
    units = (
        frame[["actor_id", "speaker_id", "sentence", "text", "identity_clip_id", "identity_path", "neutral_clip_id", "neutral_path", "age", "sex"]]
        .drop_duplicates(subset=["actor_id", "sentence"])
        .sort_values(["actor_id", "sentence"])
        .reset_index(drop=True)
    )
    if args.max_pairs is not None:
        units = units.head(int(args.max_pairs)).copy()
    if units.empty:
        raise ValueError("no generation units selected")

    selected_pairs = set(zip(units["actor_id"].astype(str), units["sentence"].astype(str)))
    frame = frame[
        [
            (str(actor), str(sentence)) in selected_pairs
            for actor, sentence in zip(frame["actor_id"], frame["sentence"])
        ]
    ].copy()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")

    print("Loading StyleTTS2 ...", flush=True)
    tts = _load_tts(device)
    sample_rate = int(tts.sample_rate)

    # The identity base is one frozen StyleTTS2 style/predictor vector per
    # actor, extracted from the actor's real neutral identity clip.  This is
    # per actor, not per sentence.
    identity_styles: Dict[str, Dict[str, Any]] = {}
    style_bases: Dict[str, Any] = {}
    neutral_zero_max = 0.0
    neutral_zero_failures: List[str] = []

    for actor in sorted(units["actor_id"].astype(str).unique()):
        row = units[units["actor_id"].astype(str) == actor].iloc[0]
        identity_path = _resolve(str(row["identity_path"]))
        with torch.no_grad():
            style_base = tts.compute_style(str(identity_path))
        if tuple(style_base.shape) != (1, 256):
            raise RuntimeError(f"unexpected style shape for actor {actor}: {tuple(style_base.shape)}")
        identity_styles[actor] = {
            "clip_id": str(row["identity_clip_id"]),
            "path": str(row["identity_path"]),
            "sha256": _sha256_tensor(style_base),
            "norm": float(style_base.norm().item()),
            "source": "styletts2_style_encoder",
        }
        style_bases[actor] = style_base

        # Neutral must be exactly the canonical style, and intensity 0 must be
        # a no-op for every available emotion.
        if args.condition == "class_mean":
            neutral_style = style_base
            for emotion in available:
                residual = residual_from_artifact(
                    residual_payload,
                    emotion,
                    variant=args.residual_variant,
                )
                zero_style = apply_style_residual(
                    style_base,
                    residual,
                    intensity=0.0,
                    scale=residual_scale,
                )
                neutral_zero_max = max(
                    neutral_zero_max,
                    float((zero_style - style_base).abs().max().item()),
                )
        elif args.condition == "learned_residual":
            neutral_style = style_base
            for emotion in available:
                zero_style = predict_residual(
                    adapter_model,
                    adapter_payload,
                    style_base,
                    emotion,
                    intensity=0.0,
                    device=str(style_base.device),
                )
                neutral_zero_max = max(
                    neutral_zero_max,
                    float(zero_style.abs().max().item()),
                )
        else:
            neutral_style = apply_emotion_style_delta(style_base, "neutral", 1.0)
            neutral_zero_max = max(
                neutral_zero_max,
                float((neutral_style - style_base).abs().max().item()),
            )
            for emotion in available:
                zero_style = apply_emotion_style_delta(style_base, emotion, 0.0)
                neutral_zero_max = max(
                    neutral_zero_max,
                    float((zero_style - style_base).abs().max().item()),
                )
        print(f"  actor {actor}: identity {row['identity_clip_id']} ready", flush=True)

    if neutral_zero_max > 1e-7:
        neutral_zero_failures.append(
            f"neutral-zero max abs error {neutral_zero_max:.3e} exceeds 1e-7"
        )

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    started = time.time()

    for _, unit in units.iterrows():
        actor = str(unit["actor_id"])
        sentence = str(unit["sentence"])
        text = str(unit["text"])
        seed_values = [int(seed) for seed in args.seeds]
        style_base = style_bases[actor]
        gender = _gender_code(unit["sex"])
        age = float(unit["age"])
        pair_id = f"{actor}_{sentence}"

        for seed in seed_values:
            neutral_id = f"{pair_id}_neutral_seed{seed}"
            neutral_path = (
                args.output_dir / "neutral" / "neutral" / f"{pair_id}_seed{seed}.wav"
            )
            neutral_row = {
                "audio_id": neutral_id,
                "condition": "neutral",
                "emotion": "neutral",
                "intensity": 0.0,
                "actor_id": actor,
                "speaker_id": str(unit["speaker_id"]),
                "sentence": sentence,
                "text": text,
                "seed": seed,
                "audio_path": _relative(neutral_path),
                "neutral_audio_id": neutral_id,
                "neutral_audio_path": _relative(neutral_path),
                "identity_clip_id": str(unit["identity_clip_id"]),
                "identity_path": str(unit["identity_path"]),
                "neutral_clip_id": str(unit["neutral_clip_id"]),
                "neutral_path": str(unit["neutral_path"]),
                "emotion_clip_id": str(unit["neutral_clip_id"]),
                "emotion_path": str(unit["neutral_path"]),
                "has_real_emotion_clip": True,
                "diffusion_steps": int(args.diffusion_steps),
                "alpha": float(args.alpha),
                "beta": float(args.beta),
                "duration_scale": float(args.duration_scale),
                "f0_scale": 1.0,
                "energy_scale": 1.0,
                "scalar_modifier_source": (
                    "neutral_control"
                    if args.scalar_modifiers == "neutral"
                    else "neutral_profile"
                ),
                "style_delta_l2": 0.0,
                "style_prosody_delta_l2": 0.0,
                "residual_variant": "",
                "residual_raw_norm": 0.0,
                "residual_applied_norm": 0.0,
                "residual_target_norm": None,
                "residual_scale": None,
                "residual_artifact_sha256": "",
                "identity_style_sha256": identity_styles[actor]["sha256"],
                "identity_style_norm": identity_styles[actor]["norm"],
                "generation_seconds": 0.0,
                "audio_duration_sec": None,
                "sample_rate": sample_rate,
                "backend": tts.backend_name,
                "gender": gender,
                "age": age,
            }

            if args.skip_existing and neutral_path.exists():
                import soundfile as sf

                neutral_row["audio_duration_sec"] = float(sf.info(str(neutral_path)).duration)
            else:
                try:
                    t0 = time.time()
                    waveform = _synthesize(
                        tts,
                        text=text,
                        style=style_base,
                        gender=gender,
                        age=age,
                        alpha=args.alpha,
                        beta=args.beta,
                        duration_scale=args.duration_scale,
                        f0_scale=1.0,
                        diffusion_steps=args.diffusion_steps,
                        seed=seed,
                    )
                    neutral_row["audio_duration_sec"] = _save_waveform(
                        neutral_path, waveform, sample_rate
                    )
                    neutral_row["generation_seconds"] = round(time.time() - t0, 3)
                except Exception as exc:  # pragma: no cover - runtime guard
                    failures.append(
                        {
                            "audio_id": neutral_id,
                            "pair_id": pair_id,
                            "emotion": "neutral",
                            "seed": seed,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
            rows.append(neutral_row)

            for emotion in available:
                audio_id = f"{pair_id}_{emotion}_seed{seed}"
                audio_path = (
                    args.output_dir / condition_name / emotion / f"{pair_id}_seed{seed}.wav"
                )
                mods, scalar_source = _scalar_modifiers(
                    EmotionModulator,
                    emotion,
                    mode=args.scalar_modifiers,
                )
                residual_raw_norm = 0.0
                residual_target_norm = None
                if args.condition == "class_mean":
                    residual = residual_from_artifact(
                        residual_payload,
                        emotion,
                        variant=args.residual_variant,
                    )
                    residual_raw_norm = float(residual.norm().item())
                    if args.match_fixed_norm:
                        target_norm = float(
                            fixed_norms.get(emotion, mean_fixed_norm)
                        )
                        if residual_raw_norm > 0:
                            residual = residual * (
                                target_norm / residual_raw_norm
                            )
                        residual_target_norm = target_norm
                    modified_style = apply_style_residual(
                        style_base,
                        residual,
                        intensity=1.0,
                        scale=residual_scale,
                    )
                elif args.condition == "learned_residual":
                    residual = predict_residual(
                        adapter_model,
                        adapter_payload,
                        style_base,
                        emotion,
                        intensity=1.0,
                        device=str(style_base.device),
                    )
                    residual_raw_norm = float(residual.norm().item())
                    modified_style = apply_learned_residual(style_base, residual)
                else:
                    modified_style = apply_emotion_style_delta(
                        style_base,
                        emotion_id=emotion,
                        intensity=1.0,
                    )
                delta = (modified_style - style_base).squeeze(0)
                row = {
                    "audio_id": audio_id,
                    "condition": condition_name,
                    "emotion": emotion,
                    "intensity": 1.0,
                    "actor_id": actor,
                    "speaker_id": str(unit["speaker_id"]),
                    "sentence": sentence,
                    "text": text,
                    "seed": seed,
                    "audio_path": _relative(audio_path),
                    "neutral_audio_id": neutral_id,
                    "neutral_audio_path": _relative(neutral_path),
                    "identity_clip_id": str(unit["identity_clip_id"]),
                    "identity_path": str(unit["identity_path"]),
                    "neutral_clip_id": str(unit["neutral_clip_id"]),
                    "neutral_path": str(unit["neutral_path"]),
                    "emotion_clip_id": "",
                    "emotion_path": "",
                    "has_real_emotion_clip": False,
                    "diffusion_steps": int(args.diffusion_steps),
                    "alpha": float(args.alpha),
                    "beta": float(
                        np.clip(args.beta * float(mods["beta_mult"]), 0.10, 0.90)
                    ),
                    "duration_scale": float(
                        args.duration_scale * float(mods["duration_scale"])
                    ),
                    "f0_scale": float(mods["pitch_factor"]),
                    "energy_scale": float(mods["energy_scale"]),
                    "scalar_modifier_source": scalar_source,
                    "style_delta_l2": float(delta.norm().item()),
                    "style_prosody_delta_l2": float(delta[128:256].norm().item()),
                    "residual_variant": (
                        args.residual_variant
                        if args.condition == "class_mean"
                        else ("learned" if args.condition == "learned_residual" else "")
                    ),
                    "residual_raw_norm": residual_raw_norm,
                    "residual_applied_norm": float(delta.norm().item()),
                    "residual_target_norm": residual_target_norm,
                    "residual_scale": (
                        residual_scale
                        if args.condition in {"class_mean", "learned_residual"}
                        else None
                    ),
                    "residual_artifact_sha256": residual_artifact_sha,
                    "identity_style_sha256": identity_styles[actor]["sha256"],
                    "identity_style_norm": identity_styles[actor]["norm"],
                    "generation_seconds": 0.0,
                    "audio_duration_sec": None,
                    "sample_rate": sample_rate,
                    "backend": tts.backend_name,
                    "gender": gender,
                    "age": age,
                }
                # Attach the real target clip for this emotion, if it exists.
                target = frame[
                    (frame["actor_id"].astype(str) == actor)
                    & (frame["sentence"].astype(str) == sentence)
                    & (frame["emotion"].astype(str) == emotion)
                ]
                if not target.empty:
                    target_row = target.iloc[0]
                    row["emotion_clip_id"] = str(target_row["emotion_clip_id"])
                    row["emotion_path"] = str(target_row["emotion_path"])
                    row["has_real_emotion_clip"] = bool(target_row["has_real_emotion_clip"])

                if args.skip_existing and audio_path.exists():
                    import soundfile as sf

                    row["audio_duration_sec"] = float(sf.info(str(audio_path)).duration)
                else:
                    try:
                        t0 = time.time()
                        waveform = _synthesize(
                            tts,
                            text=text,
                            style=modified_style,
                            gender=gender,
                            age=age,
                            alpha=args.alpha,
                            beta=float(row["beta"]),
                            duration_scale=float(row["duration_scale"]),
                            f0_scale=float(row["f0_scale"]),
                            diffusion_steps=args.diffusion_steps,
                            seed=seed,
                        )
                        waveform = _apply_energy(waveform, float(row["energy_scale"]))
                        row["audio_duration_sec"] = _save_waveform(
                            audio_path, waveform, sample_rate
                        )
                        row["generation_seconds"] = round(time.time() - t0, 3)
                    except Exception as exc:  # pragma: no cover - runtime guard
                        failures.append(
                            {
                                "audio_id": audio_id,
                                "pair_id": pair_id,
                                "emotion": emotion,
                                "seed": seed,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        continue
                rows.append(row)
            print(
                f"  {pair_id} seed {seed}: neutral + {len(available)} emotion(s)",
                flush=True,
            )

    index = pd.DataFrame(rows, columns=list(INDEX_COLUMNS))
    args.index.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.index, index=False)

    counts: Dict[str, Any] = {}
    for (condition, emotion), subset in index.groupby(["condition", "emotion"]):
        counts[f"{condition}:{emotion}"] = {
            "rows": int(len(subset)),
            "actors": int(subset["actor_id"].nunique()),
            "pairs": int(
                (subset["actor_id"].astype(str) + "_" + subset["sentence"].astype(str)).nunique()
            ),
            "audio_seconds": float(subset["audio_duration_sec"].fillna(0.0).sum()),
            "mean_style_delta_l2": float(
                subset["style_delta_l2"].replace(0.0, np.nan).mean()
            )
            if subset["style_delta_l2"].replace(0.0, np.nan).notna().any()
            else None,
            "mean_generation_seconds": float(
                subset["generation_seconds"].replace(0.0, np.nan).mean()
            )
            if subset["generation_seconds"].replace(0.0, np.nan).notna().any()
            else None,
        }

    report = {
        "schema_version": GENERATION_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "condition": condition_name,
        "scope": (
            "Identity-conditioned CREMA-D evaluation. No face, CFVN, CVAE, or "
            "canonical face-to-voice style adapter is used. The identity base is "
            "the frozen StyleTTS2 style/predictor encoder applied to one real "
            "neutral clip per actor. The style residual is "
            f"{condition_name}."
        ),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "index": str(args.index),
        "index_sha256": _sha256(args.index),
        "identity_style_source": "styletts2_style_encoder",
        "residual_artifact": (
            str(args.residual_artifact)
            if args.condition == "class_mean"
            else (str(args.adapter_artifact) if args.condition == "learned_residual" else None)
        ),
        "residual_artifact_sha256": residual_artifact_sha or None,
        "residual_variant": (
            args.residual_variant
            if args.condition == "class_mean"
            else ("learned" if args.condition == "learned_residual" else None)
        ),
        "residual_scale": (
            residual_scale
            if args.condition in {"class_mean", "learned_residual"}
            else None
        ),
        "match_fixed_norm": bool(
            args.condition == "class_mean" and args.match_fixed_norm
        ),
        "adapter": (
            {
                "artifact": str(args.adapter_artifact),
                "artifact_sha256": residual_artifact_sha,
                "schema_version": adapter_payload.get("schema_version"),
                "emotions": adapter_payload.get("emotions"),
                "max_norm": (adapter_payload.get("config") or {}).get("max_norm"),
                "style_dim": adapter_payload.get("style_dim"),
                "metadata": adapter_payload.get("metadata", {}),
            }
            if args.condition == "learned_residual"
            else None
        ),
        "fixed_delta_norms": fixed_norms,
        "mean_fixed_delta_norm": mean_fixed_norm if fixed_norms else None,
        "device": device,
        "seeds": [int(seed) for seed in args.seeds],
        "diffusion_steps": int(args.diffusion_steps),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "duration_scale": float(args.duration_scale),
        "scalar_modifiers": str(args.scalar_modifiers),
        "sample_rate": sample_rate,
        "backend": tts.backend_name,
        "emotions_requested": requested,
        "emotions_generated": available,
        "emotions_unavailable": unavailable,
        "identity_styles": identity_styles,
        "counts": counts,
        "neutral_zero": {
            "max_abs_error": neutral_zero_max,
            "tolerance": 1e-7,
            "passed": not neutral_zero_failures,
            "failures": neutral_zero_failures,
        },
        "failures": failures,
        "elapsed_seconds": round(time.time() - started, 2),
        "notes": [
            "Neutral is generated once per (actor, sentence, seed) and shared by emotion rows.",
            (
                "The class-mean condition applies a speaker-balanced mean style residual "
                "estimated from real CREMA-D train actors."
                if args.condition == "class_mean"
                else (
                    "The learned-residual condition applies the E2 adapter prediction "
                    "directly; the adapter output is hard-bounded at max_norm. This is "
                    "a research artifact, not a serving checkpoint."
                    if args.condition == "learned_residual"
                    else "The fixed-delta condition uses the current EmotionModulator style delta and scalar modifiers."
                )
            ),
            (
                "Class-mean residuals are rescaled to the fixed-delta norm for matched strength."
                if args.condition == "class_mean" and args.match_fixed_norm
                else (
                    "The learned residual is not rescaled; the adapter's max_norm is the bound."
                    if args.condition == "learned_residual"
                    else "The identity base is the frozen StyleTTS2 style/predictor encoder, not the face-to-voice adapter."
                )
            ),
            (
                "Fear and disgust are unavailable for fixed deltas because this checkout has no artifact for them."
                if args.condition == "fixed_delta"
                else (
                    "Fear and disgust are available in the learned adapter artifact."
                    if args.condition == "learned_residual"
                    else "Fear and disgust are available in the class-mean artifact."
                )
            ),
            (
                "Fear/disgust use the neutral post-synthesis scalar profile; "
                "the class-mean style residual carries the emotion. The index "
                "records scalar_modifier_source for every row."
                if args.condition == "class_mean"
                else (
                    "Fear/disgust use the neutral post-synthesis scalar profile; the "
                    "learned style residual carries the emotion. The index records "
                    "scalar_modifier_source for every row."
                    if args.condition == "learned_residual"
                    else "Emotion rows use the current EmotionModulator scalar profile."
                )
            ),
            "No human MOS is claimed; perceptual quality is unmeasured.",
            (
                "Controlled scalar condition: every emotion uses the neutral "
                "post-synthesis profile, so pitch, duration, and energy do not "
                "carry the emotion label."
                if args.scalar_modifiers == "neutral"
                else "Emotion rows use each emotion's post-synthesis scalar profile."
            ),
        ],
        "mos": {
            "status": "unavailable",
            "reason": "no human ratings were collected for this generation run",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Rows: {len(index)}  Failures: {len(failures)}")
    print(f"Neutral-zero max abs error: {neutral_zero_max:.3e}")
    for key, values in counts.items():
        print(
            f"  {key}: rows={values['rows']} pairs={values['pairs']} "
            f"audio_seconds={values['audio_seconds']:.1f}"
        )
    print(f"Wrote {args.index}")
    print(f"Wrote {args.report}")
    return 1 if failures or neutral_zero_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
