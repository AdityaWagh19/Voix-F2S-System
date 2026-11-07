"""Generate the A3 accent style-residual comparison set.

A3 applies the class-contrast GLOBE style residual in StyleTTS2 style
space instead of changing phonemes.  This bypasses the failed A2 G2P path: the
canonical gruut phonemizer is used for every clip, and the only experimental
variable is the style residual.

The residual is built from training speakers only by
``scripts/training/build_accent_class_mean.py``.  The timbre half can be down-weighted
(``--timbre-weight``) so identity is less affected; the prosody half carries the
accent signal.  ``regex`` is the no-residual baseline and ``style_s<scale>`` is
the residual condition.  A validation sweep over scales is expected before a
single scale is used for the final comparison or demo.
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
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.models.accent_g2p import (  # noqa: E402
    ACCENT_IDS,
    ACCENT_PROMPT_SET,
    TRAINABLE_ACCENTS,
)
from voix.models.accent_residual import (  # noqa: E402
    apply_accent_residual as apply_residual,
)

DEFAULT_MANIFEST = ROOT / "data" / "processed" / "globe_accent_manifest.csv"
DEFAULT_AUDIO_ROOT = ROOT / "data" / "processed" / "globe_audio"
DEFAULT_ARTIFACT = ROOT / "checkpoints" / "accent_residuals" / "class_mean.pt"
DEFAULT_OUTPUT = ROOT / "results" / "accent" / "a3" / "audio"
DEFAULT_INDEX = ROOT / "results" / "accent" / "a3" / "generation_index.csv"
DEFAULT_REPORT = ROOT / "results" / "accent" / "a3" / "generation_report.json"

INDEX_COLUMNS = [
    "clip_id",
    "accent_class",
    "condition",
    "prompt_id",
    "text",
    "phonemes",
    "wav_path",
    "duration_sec",
    "seed",
    "diffusion_steps",
    "style_sha256",
    "residual_scale",
    "timbre_weight",
    "residual_norm",
    "identity_clip_id",
    "identity_path",
    "trainable_class",
    "track",
    "level",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate A3 accent residuals")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--residual-artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--identity-clip", default=None)
    parser.add_argument("--accents", nargs="+", default=list(TRAINABLE_ACCENTS))
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[0.2],
        help="Residual scales to generate; each becomes condition style_s<scale>.",
    )
    parser.add_argument("--timbre-weight", type=float, default=0.25)
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also generate the no-residual regex baseline.",
    )
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--beta", type=float, default=0.22)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def _audio_path(row: pd.Series, root: Path) -> Path:
    return (
        root
        / str(row["speaker_split"])
        / str(row["accent_class"])
        / f"{row['clip_id']}.wav"
    )


def _select_identity(frame: pd.DataFrame, root: Path, clip_id: Optional[str]) -> pd.Series:
    if clip_id:
        matches = frame[frame["clip_id"].astype(str) == str(clip_id)]
        if matches.empty:
            raise KeyError(f"identity clip {clip_id!r} is not in the manifest")
        return matches.iloc[0]
    candidates = frame[
        (frame["speaker_split"].astype(str) == "val")
        & (frame["accent_class"].astype(str) == "us")
    ].sort_values("clip_id")
    for _, row in candidates.iterrows():
        if _audio_path(row, root).is_file():
            return row
    raise RuntimeError("no val US identity clip with audio was found")


def _load_tts(device: str):
    from voix.models.tts_backend import StyleTTS2Backend

    tts = StyleTTS2Backend(device=device, use_neural=True, allow_fallback=False)
    tts.load()
    if getattr(tts, "_use_fallback", True):
        raise RuntimeError("StyleTTS2 fell back to the non-neural backend")
    return tts


def _synthesize(
    tts,
    *,
    text: str,
    style,
    seed: int,
    diffusion_steps: int,
    alpha: float,
    beta: float,
):
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))
    kwargs = dict(
        text=text,
        style=style,
        gender=1,
        age=30.0,
        alpha=float(alpha),
        beta=float(beta),
        duration_scale=1.0,
        diffusion_steps=int(diffusion_steps),
        f0_scale=1.0,
        prosody_bias_strength=0.0,
    )
    with torch.no_grad():
        return tts.synthesize(**kwargs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if not args.residual_artifact.is_file():
        raise FileNotFoundError(f"residual artifact not found: {args.residual_artifact}")

    accents = list(dict.fromkeys(str(item).lower() for item in args.accents))
    unknown = [accent for accent in accents if accent not in ACCENT_IDS]
    if unknown:
        raise ValueError(f"unknown accents: {unknown}")
    scales = [float(scale) for scale in args.scales]
    if not scales:
        raise ValueError("at least one residual scale is required")

    frame = pd.read_csv(args.manifest)
    identity = _select_identity(frame, args.audio_root, args.identity_clip)
    identity_path = _audio_path(identity, args.audio_root)
    if not identity_path.is_file():
        raise FileNotFoundError(f"identity audio not found: {identity_path}")

    artifact = torch.load(args.residual_artifact, map_location="cpu")
    residuals = {
        str(key): value.float().reshape(-1)
        for key, value in artifact["residuals"].items()
    }
    missing = [accent for accent in accents if accent not in residuals]
    if missing:
        raise KeyError(f"residual artifact has no class for {missing}")

    print("Loading StyleTTS2 ...", flush=True)
    tts = _load_tts(args.device)
    sample_rate = int(tts.sample_rate)
    with torch.no_grad():
        base_style = tts.compute_style(str(identity_path))
    base_style_sha = _sha256_text(
        str(base_style.detach().cpu().numpy().round(8).tolist())
    )

    canonical = getattr(tts._model, "phoneme_converter", None)
    if canonical is None:
        raise RuntimeError("the neural backend has no phoneme converter")

    prompts = list(ACCENT_PROMPT_SET)
    if args.max_prompts > 0:
        prompts = prompts[: int(args.max_prompts)]
    conditions: List[tuple[str, Optional[float]]] = []
    if args.include_baseline:
        conditions.append(("regex", None))
    for scale in scales:
        conditions.append((f"style_s{scale:g}", scale))

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    started = time.time()
    total = len(accents) * len(prompts) * len(conditions) * len(args.seeds)
    done = 0
    for accent in accents:
        for prompt_id, text in enumerate(prompts):
            try:
                phonemes = canonical.phonemize(text)
            except Exception as exc:  # pragma: no cover - runtime guard
                failures.append(
                    {
                        "accent": accent,
                        "prompt_id": str(prompt_id),
                        "error": f"phonemize {type(exc).__name__}: {exc}",
                    }
                )
                continue
            for condition, scale in conditions:
                style = base_style
                residual_norm = 0.0
                if scale is not None:
                    style = apply_residual(
                        base_style,
                        residuals[accent],
                        scale=float(scale),
                        timbre_weight=float(args.timbre_weight),
                    )
                    residual_norm = float(
                        (style - base_style).norm().item()
                    )
                style_sha = _sha256_text(
                    str(style.detach().cpu().numpy().round(8).tolist())
                )
                for seed in args.seeds:
                    clip_id = f"a3_{accent}_{prompt_id:02d}_{condition}_s{int(seed)}"
                    destination = args.output_dir / accent / condition / f"{clip_id}.wav"
                    if args.skip_existing and destination.is_file():
                        info = sf.info(str(destination))
                        rows.append(
                            {
                                "clip_id": clip_id,
                                "accent_class": accent,
                                "condition": condition,
                                "prompt_id": prompt_id,
                                "text": text,
                                "phonemes": phonemes,
                                "wav_path": str(destination),
                                "duration_sec": float(info.duration),
                                "seed": int(seed),
                                "diffusion_steps": int(args.diffusion_steps),
                                "style_sha256": style_sha,
                                "residual_scale": float(scale) if scale is not None else 0.0,
                                "timbre_weight": float(args.timbre_weight),
                                "residual_norm": residual_norm,
                                "identity_clip_id": str(identity["clip_id"]),
                                "identity_path": str(identity_path),
                                "trainable_class": accent in TRAINABLE_ACCENTS,
                                "track": "accent",
                                "level": "exploratory",
                            }
                        )
                        done += 1
                        continue
                    try:
                        waveform = _synthesize(
                            tts,
                            text=text,
                            style=style,
                            seed=int(seed),
                            diffusion_steps=int(args.diffusion_steps),
                            alpha=float(args.alpha),
                            beta=float(args.beta),
                        )
                        audio = waveform.detach().cpu().numpy().reshape(-1)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(str(destination), audio, sample_rate, subtype="PCM_16")
                    except Exception as exc:  # pragma: no cover - runtime guard
                        failures.append(
                            {
                                "accent": accent,
                                "prompt_id": str(prompt_id),
                                "condition": condition,
                                "seed": str(seed),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        done += 1
                        continue
                    rows.append(
                        {
                            "clip_id": clip_id,
                            "accent_class": accent,
                            "condition": condition,
                            "prompt_id": prompt_id,
                            "text": text,
                            "phonemes": phonemes,
                            "wav_path": str(destination),
                            "duration_sec": float(len(audio) / sample_rate),
                            "seed": int(seed),
                            "diffusion_steps": int(args.diffusion_steps),
                            "style_sha256": style_sha,
                            "residual_scale": float(scale) if scale is not None else 0.0,
                            "timbre_weight": float(args.timbre_weight),
                            "residual_norm": residual_norm,
                            "identity_clip_id": str(identity["clip_id"]),
                            "identity_path": str(identity_path),
                            "trainable_class": accent in TRAINABLE_ACCENTS,
                            "track": "accent",
                            "level": "exploratory",
                        }
                    )
                    done += 1
                elapsed = time.time() - started
                print(
                    f"  {done}/{total} ({done / max(elapsed, 1e-6):.2f}/s, "
                    f"{elapsed:.0f}s, failures={len(failures)})",
                    flush=True,
                )

    index = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(args.index, index=False)
    report = {
        "schema_version": "voix.accent_style_residual_generation.v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "track": "accent",
        "experiment": "A3",
        "level": "exploratory",
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "residual_artifact": str(args.residual_artifact),
        "residual_artifact_sha256": _sha256(args.residual_artifact),
        "index": str(args.index),
        "index_sha256": _sha256(args.index),
        "output_dir": str(args.output_dir),
        "accents": accents,
        "trainable_accents": [accent for accent in accents if accent in TRAINABLE_ACCENTS],
        "eval_only_accents": [accent for accent in accents if accent not in TRAINABLE_ACCENTS],
        "conditions": [condition for condition, _ in conditions],
        "scales": scales,
        "timbre_weight": float(args.timbre_weight),
        "prompts": prompts,
        "prompt_count": len(prompts),
        "seeds": list(args.seeds),
        "diffusion_steps": int(args.diffusion_steps),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "prosody_bias_strength": 0.0,
        "identity": {
            "clip_id": str(identity["clip_id"]),
            "path": str(identity_path),
            "speaker_id": str(identity["client_id"]),
            "accent_class": str(identity["accent_class"]),
            "base_style_sha256": base_style_sha,
            "base_style_norm": float(base_style.norm().item()),
        },
        "rows": int(len(index)),
        "expected_rows": int(total),
        "failures": failures,
        "residual_norms": {
            accent: float(residuals[accent].norm().item()) for accent in accents
        },
        "notes": [
            "The canonical gruut phonemizer is used for every clip; no G2P is applied.",
            "The identity style vector is frozen across every clip.",
            "The residual is a class contrast built from GLOBE train speakers only.",
            "Timbre half is down-weighted; the prosody half carries the accent signal.",
            "Gender/age prosody bias is disabled so the residual is the only accent variable.",
        ],
        "limitations": [
            "Six prompts are enough for a diagnostic, not a confirmatory study.",
            "The classifier proxy was trained on real speech and may not respond to generated accent.",
            "No human accent-identification or MOS study was run.",
        ],
        "mos": {"status": "unavailable", "reason": "no human ratings collected"},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.index} ({len(index)} rows)")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
