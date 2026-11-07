"""Generate the A2 accent G2P comparison set.

For every accent (five trainable classes plus evaluation-only Irish) and every
prompt in the shared phonetically balanced prompt set, synthesize two versions:

    regex: StyleTTS2's canonical gruut phonemizer
    g2p:   the IPA-level accent phonemizer in ``voix.models.accent_g2p``

The identity style vector is frozen and identical for every clip, so the only
experimental variable is the phoneme string.  The biometric gender/age residual
is disabled to avoid a second, uncontrolled source of variation.

This is a research generation script.  It does not load the CFVN, CVAE, or the
canonical face-to-voice adapter.
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
    phonemize_accent,
    use_accent_phonemizer,
)

DEFAULT_MANIFEST = ROOT / "data" / "processed" / "globe_accent_manifest.csv"
DEFAULT_AUDIO_ROOT = ROOT / "data" / "processed" / "globe_audio"
DEFAULT_OUTPUT = ROOT / "results" / "accent" / "a2" / "audio"
DEFAULT_INDEX = ROOT / "results" / "accent" / "a2" / "generation_index.csv"
DEFAULT_REPORT = ROOT / "results" / "accent" / "a2" / "generation_report.json"

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
    parser = argparse.ArgumentParser(description="Generate A2 accent G2P clips")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--accents",
        nargs="+",
        default=list(ACCENT_IDS),
        help="Accent classes to synthesize (default: five trainable + Irish)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["regex", "g2p"],
        choices=["regex", "g2p"],
    )
    parser.add_argument("--identity-clip", default=None, help="GLOBE clip_id for the style donor")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--disable-rules",
        nargs="+",
        default=[],
        help="Rule ids to skip in the G2P path (ablation only)",
    )
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--beta", type=float, default=0.22)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def _audio_path(row: pd.Series, root: Path) -> Path:
    return root / str(row["speaker_split"]) / str(row["accent_class"]) / f"{row['clip_id']}.wav"


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
    accent: str,
    condition: str,
    seed: int,
    diffusion_steps: int,
    alpha: float,
    beta: float,
    disabled_rules: Sequence[str] = (),
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
        if condition == "g2p":
            with use_accent_phonemizer(
                tts, accent, disabled_rules=disabled_rules
            ):
                waveform = tts.synthesize(**kwargs)
        else:
            waveform = tts.synthesize(**kwargs)
    return waveform


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    accents = list(dict.fromkeys(str(item).lower() for item in args.accents))
    unknown = [accent for accent in accents if accent not in ACCENT_IDS]
    if unknown:
        raise ValueError(f"unknown accents: {unknown}")
    frame = pd.read_csv(args.manifest)
    identity = _select_identity(frame, args.audio_root, args.identity_clip)
    identity_path = _audio_path(identity, args.audio_root)
    if not identity_path.is_file():
        raise FileNotFoundError(f"identity audio not found: {identity_path}")

    print("Loading StyleTTS2 ...", flush=True)
    tts = _load_tts(args.device)
    sample_rate = int(tts.sample_rate)
    with torch.no_grad():
        style = tts.compute_style(str(identity_path))
    style_sha = _sha256_text(str(style.detach().cpu().numpy().round(8).tolist()))

    canonical = getattr(tts._model, "phoneme_converter", None)
    if canonical is None:
        raise RuntimeError("the neural backend has no phoneme converter")

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    phoneme_changes: List[Dict[str, Any]] = []
    started = time.time()
    total = len(accents) * len(ACCENT_PROMPT_SET) * len(args.conditions) * len(args.seeds)
    done = 0
    for accent in accents:
        for prompt_id, text in enumerate(ACCENT_PROMPT_SET):
            for condition in args.conditions:
                try:
                    if condition == "g2p":
                        phonemes = phonemize_accent(
                            text, accent, disabled_rules=args.disable_rules
                        )
                    else:
                        phonemes = canonical.phonemize(text)
                except Exception as exc:  # pragma: no cover - runtime guard
                    failures.append(
                        {
                            "accent": accent,
                            "prompt_id": str(prompt_id),
                            "condition": condition,
                            "error": f"phonemize {type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                if condition == "g2p":
                    try:
                        base = canonical.phonemize(text)
                    except Exception:
                        base = ""
                    phoneme_changes.append(
                        {
                            "accent": accent,
                            "prompt_id": prompt_id,
                            "base_phonemes": base,
                            "g2p_phonemes": phonemes,
                            "changed": base.strip() != phonemes.strip(),
                        }
                    )
                for seed in args.seeds:
                    clip_id = f"a2_{accent}_{prompt_id:02d}_{condition}_s{int(seed)}"
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
                            accent=accent,
                            condition=condition,
                            seed=int(seed),
                            diffusion_steps=int(args.diffusion_steps),
                            alpha=float(args.alpha),
                            beta=float(args.beta),
                            disabled_rules=args.disable_rules,
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
        "schema_version": "voix.accent_g2p_generation.v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "track": "accent",
        "experiment": "A2",
        "level": "exploratory",
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "index": str(args.index),
        "index_sha256": _sha256(args.index),
        "output_dir": str(args.output_dir),
        "accents": accents,
        "trainable_accents": [accent for accent in accents if accent in TRAINABLE_ACCENTS],
        "eval_only_accents": [accent for accent in accents if accent not in TRAINABLE_ACCENTS],
        "conditions": list(args.conditions),
        "prompts": list(ACCENT_PROMPT_SET),
        "prompt_count": len(ACCENT_PROMPT_SET),
        "seeds": list(args.seeds),
        "diffusion_steps": int(args.diffusion_steps),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "disabled_rules": list(args.disable_rules),
        "prosody_bias_strength": 0.0,
        "identity": {
            "clip_id": str(identity["clip_id"]),
            "path": str(identity_path),
            "speaker_id": str(identity["client_id"]),
            "accent_class": str(identity["accent_class"]),
            "style_sha256": style_sha,
            "norm": float(style.norm().item()),
            "note": "One frozen style vector for every clip; accent is carried only by phonemes.",
        },
        "rows": int(len(index)),
        "expected_rows": int(total),
        "failures": failures,
        "phoneme_changes": phoneme_changes,
        "changed_prompts_by_accent": {
            accent: int(
                sum(
                    1
                    for item in phoneme_changes
                    if item["accent"] == accent and item["changed"]
                )
            )
            for accent in accents
        },
        "notes": [
            "regex is the canonical gruut phonemizer; g2p is the IPA accent rule path.",
            "The identity style vector is frozen across every clip.",
            "Gender/age prosody bias is disabled so phonemes are the only accent variable.",
            "Irish is evaluation-only and is never a training class.",
        ],
        "limitations": [
            "The accent rules are hand-written, not learned G2P.",
            "Six prompts are enough for a diagnostic, not a confirmatory study.",
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
