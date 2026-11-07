"""Controlled synthesis matrix for the VOIX demo audio audit.

The demo's hiss/brightness and "over-polished" delivery could come from several
places.  This script holds the face, text, and seed fixed and varies one factor
at a time:

* diffusion steps (5, 10, 20, 30);
* alpha/beta style blending (the demo's 0.18/0.22 vs StyleTTS2 defaults);
* the N1 LoRA adapter (off/on);
* the style source (face-derived prior mean, face-derived posterior sample,
  face-derived style without the craniofacial residual, and a real AMI
  reference style);
* output stage (raw model output vs LUFS-normalized/limited output).

Every waveform is measured with the same objective audit used for the shipped
demo, so the comparison is reproducible and does not rely on listening claims.

Usage:
    python scripts/audit/audit_synthesis_matrix.py
    python scripts/audit/audit_synthesis_matrix.py --conditions base lora steps20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

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

DEFAULT_FACE = ROOT / "demo" / "subject_face.png"
DEFAULT_TEXT = (
    "The human voice carries identity, emotion, and the rhythm of lived "
    "experience."
)
DEFAULT_REFERENCE = (
    ROOT
    / "results"
    / "audit"
    / "reference_audio"
    / "ES2003a_MEE009_0053.wav"
)
DEFAULT_LORA = ROOT / "results" / "naturalness" / "n1" / "ami_lora_pilot.pt"
DEFAULT_OUT = ROOT / "results" / "audit" / "synthesis_matrix"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))


def _face_style(
    pipeline,
    face: Path,
    *,
    sample_latent: bool = False,
    temperature: float = 0.7,
    seed: int = 42,
):
    e_f_raw, gender, age, _e_geo = pipeline.extract_face_features(str(face))
    e_f = pipeline.normalize_face(e_f_raw.to(pipeline.device))
    with torch.no_grad():
        e_f_voice = pipeline.cfvn(e_f, normalize=True)
        if sample_latent:
            _set_seed(seed)
            e_s = pipeline.cvae.sample_voices(
                e_f_voice, num_samples=1, temperature=temperature
            )
            e_s = F.normalize(e_s, dim=-1)
        else:
            mu_prior, _ = pipeline.cvae.prior_net(e_f_voice)
            e_s = F.normalize(pipeline.cvae.decoder(mu_prior), dim=-1)
        style = pipeline.adapter(e_s)
        if style.dim() == 3 and style.shape[1] == 1:
            style = style.squeeze(1)
    return style.detach(), int(gender), float(age), e_f_voice.detach()


def _raw_inference(
    tts,
    *,
    text: str,
    style: torch.Tensor,
    alpha: float,
    beta: float,
    diffusion_steps: int,
    seed: int,
) -> np.ndarray:
    _set_seed(seed)
    ref_s = style.to(tts.device).float()
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)
    with torch.no_grad():
        output = tts._model.inference(
            text=text,
            ref_s=ref_s,
            alpha=float(alpha),
            beta=float(beta),
            diffusion_steps=int(diffusion_steps),
        )
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    return np.asarray(output, dtype=np.float32).reshape(-1)


def _normalized(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    from voix.models.tts_backend import _lufs_normalize

    tensor = torch.from_numpy(wav).float().unsqueeze(0)
    with torch.no_grad():
        return _lufs_normalize(tensor, sample_rate).squeeze(0).numpy()


def _save(path: Path, wav: np.ndarray, sample_rate: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(wav, dtype=np.float32), int(sample_rate), subtype="FLOAT")
    return _sha256(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face", type=Path, default=DEFAULT_FACE)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Subset of condition names (default: all).",
    )
    return parser.parse_args(argv)


def build_conditions() -> List[Dict[str, Any]]:
    """Return the experimental cells.  Names are stable for before/after diffs."""
    return [
        {
            "name": "base",
            "description": "current demo neutral path: 10 steps, alpha .18, beta .22",
            "steps": 10,
            "alpha": 0.18,
            "beta": 0.22,
            "style": "face_prior",
            "lora": False,
        },
        {
            "name": "steps5",
            "description": "diffusion steps 5",
            "steps": 5,
            "alpha": 0.18,
            "beta": 0.22,
            "style": "face_prior",
            "lora": False,
        },
        {
            "name": "steps20",
            "description": "diffusion steps 20",
            "steps": 20,
            "alpha": 0.18,
            "beta": 0.22,
            "style": "face_prior",
            "lora": False,
        },
        {
            "name": "steps30",
            "description": "diffusion steps 30",
            "steps": 30,
            "alpha": 0.18,
            "beta": 0.22,
            "style": "face_prior",
            "lora": False,
        },
        {
            "name": "default_ab",
            "description": "StyleTTS2 defaults: alpha .30, beta .70",
            "steps": 10,
            "alpha": 0.30,
            "beta": 0.70,
            "style": "face_prior",
            "lora": False,
        },
        {
            "name": "high_ab",
            "description": "alpha .50, beta .50",
            "steps": 10,
            "alpha": 0.50,
            "beta": 0.50,
            "style": "face_prior",
            "lora": False,
        },
        {
            "name": "beta70",
            "description": "alpha .18, beta .70",
            "steps": 10,
            "alpha": 0.18,
            "beta": 0.70,
            "style": "face_prior",
            "lora": False,
        },
        {
            "name": "lora",
            "description": "N1 LoRA on, current demo settings",
            "steps": 10,
            "alpha": 0.18,
            "beta": 0.22,
            "style": "face_prior",
            "lora": True,
        },
        {
            "name": "posterior",
            "description": "CVAE posterior sample instead of prior mean",
            "steps": 10,
            "alpha": 0.18,
            "beta": 0.22,
            "style": "face_posterior",
            "lora": False,
        },
        {
            "name": "real_ref",
            "description": "real AMI reference style, current alpha/beta",
            "steps": 10,
            "alpha": 0.18,
            "beta": 0.22,
            "style": "real_reference",
            "lora": False,
        },
        {
            "name": "real_ref_default_ab",
            "description": "real AMI reference style, StyleTTS2 defaults",
            "steps": 10,
            "alpha": 0.30,
            "beta": 0.70,
            "style": "real_reference",
            "lora": False,
        },
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    raw_dir = output_dir / "raw"
    audio_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    from voix.inference.pipeline import VoixPipeline
    from voix.training.lora import inject_lora, load_lora, set_lora_enabled

    print("[setup] loading pipeline", flush=True)
    pipeline = VoixPipeline(
        device=args.device,
        enable_watermarking=False,
        tts_use_neural=True,
        tts_allow_fallback=False,
    )
    tts = pipeline.tts

    print("[setup] extracting face styles", flush=True)
    face_prior, gender, age, _ = _face_style(pipeline, args.face)
    face_posterior, _, _, _ = _face_style(
        pipeline, args.face, sample_latent=True, temperature=0.7, seed=args.seed
    )

    if not args.reference.is_file():
        raise FileNotFoundError(f"reference audio not found: {args.reference}")

    # The visual extractors and face-to-style modules are no longer needed.
    # Release them before StyleTTS2 inference: the 4 GB budget GPU otherwise
    # OOMs with both the face stack and the vocoder resident.
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

    print("[setup] loading StyleTTS2", flush=True)
    tts.load()
    if tts._model is None or getattr(tts, "_use_fallback", True):
        raise RuntimeError("neural StyleTTS2 failed to load")
    sample_rate = int(tts.sample_rate)
    model = tts._model.model

    print(f"[setup] computing real reference style: {args.reference.name}", flush=True)
    real_ref = pipeline.tts.compute_style(str(args.reference)).to(tts.device).float()
    if real_ref.shape[-1] != 256:
        raise RuntimeError(f"expected 256-D reference style, got {tuple(real_ref.shape)}")

    styles = {
        "face_prior": face_prior,
        "face_posterior": face_posterior,
        "real_reference": real_ref,
    }

    lora_loaded = False
    if args.lora.is_file():
        print(f"[setup] injecting N1 LoRA: {args.lora.name}", flush=True)
        payload = torch.load(args.lora, map_location="cpu", weights_only=False)
        config = payload.get("config", {})
        inject_lora(
            model.diffusion.unet,
            rank=int(config.get("rank", 8)),
            alpha=float(config.get("alpha", 16.0)),
            dropout=float(config.get("dropout", 0.0)),
        )
        load_lora(model.diffusion.unet, args.lora)
        set_lora_enabled(model.diffusion.unet, False)
        lora_loaded = True
    else:
        print(f"[warn] LoRA not found, skipping LoRA cells: {args.lora}", flush=True)

    wanted = set(args.conditions) if args.conditions else None
    rows: List[Dict[str, Any]] = []
    for condition in build_conditions():
        if wanted is not None and condition["name"] not in wanted:
            continue
        if condition["name"] == "lora" and not lora_loaded:
            continue
        if condition["name"] == "lora":
            set_lora_enabled(model.diffusion.unet, True)
        elif lora_loaded:
            set_lora_enabled(model.diffusion.unet, False)

        style = styles[condition["style"]]
        name = condition["name"]
        print(
            f"[synth] {name}: steps={condition['steps']} "
            f"alpha={condition['alpha']} beta={condition['beta']} "
            f"style={condition['style']} lora={condition['lora']}",
            flush=True,
        )
        started = time.time()
        raw = _raw_inference(
            tts,
            text=args.text,
            style=style,
            alpha=float(condition["alpha"]),
            beta=float(condition["beta"]),
            diffusion_steps=int(condition["steps"]),
            seed=args.seed,
        )
        raw_path = raw_dir / f"{name}.wav"
        raw_sha = _save(raw_path, raw, sample_rate)
        normalized = _normalized(raw, sample_rate)
        normalized_path = audio_dir / f"{name}.wav"
        normalized_sha = _save(normalized_path, normalized, sample_rate)

        row = {
            **condition,
            "style_sha256": hashlib.sha256(
                np.asarray(style.detach().cpu()).round(8).tobytes()
            ).hexdigest(),
            "raw_path": str(raw_path),
            "raw_sha256": raw_sha,
            "wav_path": str(normalized_path),
            "audio_sha256": normalized_sha,
            "raw_metrics": measure_clip(raw_path, display_name=f"{name}.raw.wav"),
            "metrics": measure_clip(
                normalized_path, display_name=f"{name}.wav"
            ),
            "elapsed_s": float(time.time() - started),
        }
        rows.append(row)
        print(
            f"[synth] {name} done in {row['elapsed_s']:.1f}s | "
            f"raw noise {row['raw_metrics']['noise_floor_dbfs']:.1f} dBFS -> "
            f"normalized noise {row['metrics']['noise_floor_dbfs']:.1f} dBFS",
            flush=True,
        )

    if lora_loaded:
        set_lora_enabled(model.diffusion.unet, False)

    report = {
        "schema_version": "voix.synthesis_matrix.v1",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "face": {
            "path": str(args.face),
            "sha256": _sha256(args.face),
            "gender": int(gender),
            "age": float(age),
        },
        "text": args.text,
        "seed": int(args.seed),
        "sample_rate": sample_rate,
        "reference": {
            "path": str(args.reference),
            "sha256": _sha256(args.reference),
            "style_sha256": hashlib.sha256(
                np.asarray(real_ref.detach().cpu()).round(8).tobytes()
            ).hexdigest(),
        },
        "lora": {
            "path": str(args.lora),
            "sha256": _sha256(args.lora) if args.lora.is_file() else None,
        },
        "conditions": rows,
        "notes": [
            "Raw files are the model's direct output before LUFS normalization "
            "and the true-peak limiter.",
            "Normalized files use the current backend path (-23 LUFS, -1 dBTP).",
            "The real reference style is a real AMI speaker, not a face-derived "
            "identity; it is a diagnostic control, not a demo condition.",
            "These are objective measurements, not MOS/CMOS.",
        ],
    }
    (output_dir / "synthesis_matrix.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'synthesis_matrix.json'}", flush=True)
    for row in rows:
        raw_m = row["raw_metrics"]
        norm_m = row["metrics"]
        print(
            f"{row['name']:>20s} | lufs {norm_m.get('lufs'):>7.2f} | "
            f"noise raw {raw_m['noise_floor_dbfs']:>6.1f} -> norm "
            f"{norm_m['noise_floor_dbfs']:>6.1f} | "
            f"4-8k {norm_m['speech_band_4000_6000_share_db']:>6.1f}/"
            f"{norm_m['speech_band_6000_8000_share_db']:>6.1f} | "
            f"8-12k {norm_m['speech_band_8000_10000_share_db']:>6.1f}/"
            f"{norm_m['speech_band_10000_12000_share_db']:>6.1f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
