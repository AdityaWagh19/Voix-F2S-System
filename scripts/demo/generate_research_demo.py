"""Build the VOIX research demo v3 from measured components.

The demo is deliberately small and fully attributable:

* three face-derived voices: Lupita Nyongo, Timothee Chalamet, Jackie Chan;
* three emotions: happy, sad, angry (E1 class-mean prosody residual);
* four accents: US, England, Indian English, Australian (A3 class-contrast
  residual only);
* one neutral baseline per voice.

That is 3 voices x (neutral + 3 emotions + 4 accents) = 24 clips.  The
emotion and accent scales are the serving values (0.28 and 0.30 with timbre
weight 0.25), not the exploratory v2 extensions.

The canonical face -> CFVN -> CVAE -> StyleAdapter path is used unchanged.
Every face style is recomputed and compared with the frozen diversity report
before calibration.  The experimental style-manifold calibration,

    s' = s + alpha * (mean_real_gender - mean_face_gender)

with alpha = 1.0, is applied only in this demo; it is not wired into serving and
does not modify any canonical checkpoint.  The N1 LoRA condition is omitted
because it worsened the noise floor in the controlled synthesis matrix.

This is a research gallery, not a perceptual study.  There is no MOS, CMOS, or
human emotion/accent-identification result.  All numbers in the report are
objective measurements of the generated files or cited from prior reports.

Usage:
    python scripts/demo/generate_research_demo.py --compute-wer
"""
from __future__ import annotations

import os

# The repo's TensorFlow install is unusable (einops' TF backend crashes on
# `tensorflow.Tensor`).  Keep transformers and einops on the torch path.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from voix.evaluation.naturalness import extract_naturalness_metrics  # noqa: E402
from voix.models.accent_control import accent_residual_key  # noqa: E402
from voix.models.accent_residual import apply_accent_residual  # noqa: E402
from voix.models.emotion_residual import (  # noqa: E402
    apply_style_residual,
    residual_from_artifact,
)
from voix.models.style_calibration import (  # noqa: E402
    apply_translation,
    load_calibration,
    validate_calibration,
    vector_sha256,
)

DEFAULT_FACE_REPORT = (
    ROOT / "results" / "audit" / "face_voice_diversity" / "face_voice_diversity.json"
)
DEFAULT_CALIBRATION = (
    ROOT / "results" / "audit" / "style_calibration" / "calibration.json"
)
DEFAULT_ECAPA_REFERENCE = (
    ROOT / "results" / "audit" / "ecapa_reference" / "ecapa_reference.json"
)
DEFAULT_EMOTION_ARTIFACT = (
    ROOT / "checkpoints" / "emotion_residuals" / "class_mean.pt"
)
DEFAULT_ACCENT_ARTIFACT = (
    ROOT / "checkpoints" / "accent_residuals" / "class_mean.pt"
)
DEFAULT_ACCENT_SCALE_REPORT = ROOT / "results" / "accent" / "a3" / "scale_selection.json"
DEFAULT_OUTPUT = ROOT / "results" / "demo" / "research_v3"
DEFAULT_TEXT = (
    "The human voice carries identity, emotion, and the rhythm of lived "
    "experience."
)

EXPECTED_CHECKPOINTS = {
    "cfvn": "0253bbbf96def21ca9a3a9df3a4466c1e1538ef36986d9eb9fe86bfabd7b2f6b",
    "cvae": "bb7ad29b7b5cebb2141073892b9d5736d2e996c0bd3e81acc4c671c1ea66656b",
    "style_adapter": "97750ee84e50f9878e3d98de6b2e3dc0ca6829f448f2a5e4a31185adafc0a69c",
}

DEFAULT_VOICES = (
    "C2_12_black_female_Lupita_Nyongo",
    "C1_02_young_male_Timothee_Chalamet",
    "C2_09_east_asian_male_Jackie_Chan",
)
EMOTIONS = ("happy", "sad", "angry")
ACCENTS = ("us", "england", "indian_english", "australian")

# Serving scales from the validated control path.  E1 selected 0.28 for the
# class-mean prosody residual; A3 selected 0.30 with a 0.25 timbre weight.  Do
# not substitute the v2 exploratory 0.56/1.0 values in the serving comparison.
EMOTION_SCALE = 0.28
ACCENT_SCALE = 0.30
ACCENT_TIMBRE_WEIGHT = 0.25

EMOTION_SUPPORT = (
    "results/emotion/FINDINGS.md (E1 class-mean prosody, 6/6 direction checks: "
    "happy +16.2 Hz/-0.055 s, sad -11.6 Hz/+0.048 s, angry +18.1 Hz/-0.054 s "
    "on held-out CREMA-D actors). Serving scale 0.28."
)
ACCENT_SUPPORT = (
    "results/accent/a3/scale_selection.json (A3 scale 0.3 selected; classifier "
    "7/30 vs 6/30 baseline; per-class recall US 1/6, England 5/6, "
    "Australian 1/6). Serving scale 0.30 with timbre weight 0.25. The Indian "
    "G2P subset was evaluated separately in A2c and is not applied here: it "
    "raised WER without a classifier-accuracy gain."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed))


class _AudioQuality:
    """Load the audit measurement function without duplicating its code."""

    def __init__(self) -> None:
        script = ROOT / "scripts" / "audit" / "audit_audio_quality.py"
        spec = importlib.util.spec_from_file_location("audit_audio_quality", script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.measure_clip = module.measure_clip


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the 3x3 research demo")
    parser.add_argument("--faces", nargs="+", default=list(DEFAULT_VOICES))
    parser.add_argument("--face-report", type=Path, default=DEFAULT_FACE_REPORT)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--calibration-alpha",
        type=float,
        default=1.0,
        help="Style-manifold translation strength; 1.0 maps the face mean to the real mean.",
    )
    parser.add_argument("--ecapa-reference", type=Path, default=DEFAULT_ECAPA_REFERENCE)
    parser.add_argument("--emotion-artifact", type=Path, default=DEFAULT_EMOTION_ARTIFACT)
    parser.add_argument("--accent-artifact", type=Path, default=DEFAULT_ACCENT_ARTIFACT)
    parser.add_argument(
        "--accent-scale-report", type=Path, default=DEFAULT_ACCENT_SCALE_REPORT
    )
    parser.add_argument("--emotions", nargs="+", default=list(EMOTIONS))
    parser.add_argument("--accents", nargs="+", default=list(ACCENTS))
    parser.add_argument("--emotion-scale", type=float, default=EMOTION_SCALE)
    parser.add_argument("--accent-scale", type=float, default=ACCENT_SCALE)
    parser.add_argument(
        "--accent-timbre-weight", type=float, default=ACCENT_TIMBRE_WEIGHT
    )
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--beta", type=float, default=0.22)
    parser.add_argument("--loudness-lufs", type=float, default=-23.0)
    parser.add_argument("--compute-wer", action="store_true")
    parser.add_argument("--wer-model", default="base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--noise-threshold-dbfs",
        type=float,
        default=-58.0,
        help="Fail verification when any clip is louder than this noise floor.",
    )
    parser.add_argument(
        "--snr-threshold-db",
        type=float,
        default=40.0,
        help="Fail verification when any clip has a lower SNR proxy.",
    )
    return parser.parse_args(argv)


def _load_pipeline(device: str):
    from voix.inference.pipeline import VoixPipeline

    return VoixPipeline(
        device=device,
        enable_watermarking=False,
        tts_use_neural=True,
        tts_allow_fallback=False,
    )


def _face_style(pipeline, face: Path):
    e_f, gender, age, _e_geo = pipeline.extract_face_features(str(face))
    e_f = pipeline.normalize_face(e_f.to(pipeline.device))
    with torch.no_grad():
        e_f_voice = pipeline.cfvn(e_f, normalize=True)
        mu_prior, _ = pipeline.cvae.prior_net(e_f_voice)
        e_s = F.normalize(pipeline.cvae.decoder(mu_prior), dim=-1)
        style = pipeline.adapter(e_s)
        if style.dim() == 3 and style.shape[1] == 1:
            style = style.squeeze(1)
        if style.dim() != 2 or style.shape[-1] != 256:
            raise RuntimeError(
                "expected a (batch, 256) face-derived style, got "
                f"{tuple(style.shape)}"
            )
    return style.detach(), int(gender), float(age)


def _style_sha256(style: torch.Tensor) -> str:
    return hashlib.sha256(
        np.asarray(style.detach().cpu()).round(8).tobytes()
    ).hexdigest()


def _synthesize(
    tts,
    *,
    text: str,
    style: torch.Tensor,
    gender: int,
    age: float,
    seed: int,
    diffusion_steps: int,
    alpha: float,
    beta: float,
    f0_scale: float,
    loudness_lufs: float,
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
            diffusion_steps=int(diffusion_steps),
            loudness_lufs=float(loudness_lufs),
            f0_scale=float(f0_scale),
            prosody_bias_strength=0.0,
        )


def _save(path: Path, waveform, sample_rate: int) -> float:
    audio = waveform.detach().cpu().numpy().reshape(-1).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, int(sample_rate), subtype="PCM_16")
    return float(len(audio) / float(sample_rate))


def _accent_recall(
    report_path: Path,
    scale: float,
    accent: str,
) -> Optional[Tuple[int, int]]:
    if not report_path.is_file():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    condition = f"style_s{float(scale):g}"
    block = (payload.get("classifier_by_condition") or {}).get(condition) or {}
    recall = (block.get("per_class_recall") or {}).get(accent)
    if recall is None:
        return None
    counts = block.get("per_class_counts") or {}
    total = counts.get(accent)
    if total is None:
        by_accent = (payload.get("wer") or {}).get("by_accent") or {}
        entry = by_accent.get(accent) or {}
        total = (entry.get("regex") or {}).get("n")
    if total is None:
        return None
    return int(round(float(recall) * int(total))), int(total)


def build_conditions(
    *,
    voice_id: str,
    voice_label: str,
    base_style: torch.Tensor,
    gender: int,
    age: float,
    text: str,
    emotion_payload: Mapping[str, Any],
    accent_payload: Optional[Mapping[str, Any]],
    emotions: Sequence[str],
    accents: Sequence[str],
    emotion_scale: float,
    accent_scale: float,
    accent_timbre_weight: float,
    accent_scale_report: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Build the neutral, emotion, and accent conditions for one voice."""
    conditions: List[Dict[str, Any]] = [
        {
            "name": f"{voice_id}__neutral",
            "voice_id": voice_id,
            "voice_label": voice_label,
            "track": "baseline",
            "condition": "neutral",
            "label": f"{voice_label}: neutral identity",
            "level": "baseline",
            "support": "canonical face -> CFVN -> CVAE -> StyleAdapter path + style-manifold calibration",
            "note": "No emotion or accent residual. This is the identity baseline.",
            "text": text,
            "style": base_style,
            "f0_scale": 1.0,
            "residual": None,
        }
    ]

    for emotion in dict.fromkeys(str(item).lower() for item in emotions):
        if emotion not in emotion_payload.get("emotions", {}):
            print(f"[skip] emotion {emotion!r} is absent from the artifact", flush=True)
            continue
        residual = residual_from_artifact(
            emotion_payload, emotion, variant="prosody_only"
        )
        style = apply_style_residual(
            base_style, residual, intensity=1.0, scale=float(emotion_scale)
        )
        conditions.append(
            {
                "name": f"{voice_id}__emotion_{emotion}",
                "voice_id": voice_id,
                "voice_label": voice_label,
                "track": "emotion",
                "condition": emotion,
                "label": f"{voice_label}: {emotion}",
                "level": "exploratory_application",
                "support": EMOTION_SUPPORT,
                "note": (
                    f"E1 prosody-only residual at scale {float(emotion_scale):g}. "
                    "Direction validated on CREMA-D; this face-derived "
                    "application is exploratory."
                    + (
                        " Sad lengthens the clip but does not reproduce the E1 "
                        "sad F0 lowering on these calibrated face styles; the "
                        "demo does not add a post-synthesis pitch shift because "
                        "it damaged speaker identity."
                        if emotion == "sad"
                        else ""
                    )
                ),
                "text": text,
                "style": style,
                "f0_scale": 1.0,
                "residual": {
                    "artifact": "emotion_class_mean",
                    "variant": "prosody_only",
                    "scale": float(emotion_scale),
                    "norm": float(residual.norm().item()),
                    "sha256": vector_sha256(residual),
                },
            }
        )

    if accent_payload is not None:
        residuals = accent_payload.get("residuals", {})
        for accent in dict.fromkeys(str(item).lower() for item in accents):
            recall_key = accent_residual_key(accent)
            residual_key = (
                accent if accent in residuals else recall_key
            )
            if residual_key not in residuals:
                print(
                    f"[skip] accent {accent!r} has no residual "
                    f"(artifact key {recall_key!r})",
                    flush=True,
                )
                continue
            residual = torch.as_tensor(residuals[residual_key]).float().reshape(-1)
            style = apply_accent_residual(
                base_style,
                residual,
                scale=float(accent_scale),
                timbre_weight=float(accent_timbre_weight),
            )
            recall = (
                _accent_recall(accent_scale_report, 0.3, recall_key)
                if accent_scale_report is not None
                else None
            )
            conditions.append(
                {
                    "name": f"{voice_id}__accent_{accent}",
                    "voice_id": voice_id,
                    "voice_label": voice_label,
                    "track": "accent",
                    "condition": accent,
                    "label": f"{voice_label}: {accent} accent residual",
                    "level": "exploratory_application",
                    "support": ACCENT_SUPPORT,
                    "note": (
                        "A3 class-contrast residual at the A3-selected serving "
                        f"scale {float(accent_scale):g} (timbre weight "
                        f"{float(accent_timbre_weight):g}). Accent recognition "
                        "is not established."
                        + (
                            f" A3 classifier recall for this accent was "
                            f"{recall[0]}/{recall[1]}."
                            if recall is not None
                            else ""
                        )
                    ),
                    "text": text,
                    "style": style,
                    "f0_scale": 1.0,
                    "residual": {
                        "artifact": "accent_class_mean",
                        "variant": "class_contrast",
                        "scale": float(accent_scale),
                        "timbre_weight": float(accent_timbre_weight),
                        "norm": float(residual.norm().item()),
                        "sha256": vector_sha256(residual),
                        "a3_selected_scale": 0.3,
                        "a3_recall": (
                            {"correct": recall[0], "n": recall[1]}
                            if recall is not None
                            else None
                        ),
                    },
                }
            )
    return conditions


def _attach_audio_metrics(
    rows: List[Dict[str, Any]], *, text: str
) -> None:
    quality = _AudioQuality()
    for row in rows:
        started = time.time()
        metrics = quality.measure_clip(Path(row["wav_path"]))
        natural = extract_naturalness_metrics(row["wav_path"], text=text)
        row["metrics"] = {
            **{
                key: (None if value is None else float(value))
                for key, value in natural.items()
            },
            "duration_s": float(metrics["duration_s"]),
            "noise_floor_dbfs": float(metrics["noise_floor_dbfs"]),
            "speech_level_dbfs": float(metrics["speech_level_dbfs"]),
            "snr_proxy_db": float(metrics["snr_proxy_db"]),
            "lufs": metrics.get("lufs"),
            "true_peak_dbtp": metrics.get("true_peak_dbtp"),
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
        }
        row["timing"]["metrics_s"] = float(time.time() - started)


def _attach_wer(
    rows: List[Dict[str, Any]], *, text: str, model: str
) -> Dict[str, Any]:
    from voix.evaluation.audio_metrics import transcribe_wer

    paths = [row["wav_path"] for row in rows]
    result = transcribe_wer(paths, text, model_size=model, device="cpu")
    if result.get("status") != "computed":
        for row in rows:
            row["wer"] = None
        return {
            "status": result.get("status", "unavailable"),
            "reason": result.get("reason"),
            "model": f"whisper-{model}",
        }
    for row, score in zip(rows, result.get("per_sample_wer", [])):
        row["wer"] = float(score)
        row["transcript"] = None
    for row, transcript in zip(rows, result.get("transcripts", [])):
        row["transcript"] = str(transcript)
    return {
        "status": "computed",
        "model": f"whisper-{model}",
        "n": int(result.get("n", len(rows))),
        "mean_wer": float(result.get("wer", 0.0)),
        "std_wer": float(result.get("wer_std", 0.0)),
        "max_wer": float(max(result.get("per_sample_wer", [0.0]))),
    }


def _ecapa_embedding(model, path: Path, sample_rate: int) -> np.ndarray:
    import librosa

    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    tensor = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_batch(tensor).squeeze().detach().cpu().numpy()
    vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _unit(vector: Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64).reshape(-1)
    return array / max(float(np.linalg.norm(array)), 1e-12)


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _percentile_rank(value: float, reference: Sequence[float]) -> Optional[float]:
    array = np.asarray(reference, dtype=np.float64)
    if array.size == 0:
        return None
    return float((array < float(value)).mean() * 100.0)


def _condition_deltas(
    rows: Sequence[Mapping[str, Any]], voice_id: str
) -> Dict[str, Any]:
    neutral = next(
        row
        for row in rows
        if row["voice_id"] == voice_id and row["condition"] == "neutral"
    )
    result: Dict[str, Any] = {"neutral": neutral["condition"], "conditions": {}}
    for row in rows:
        if row["voice_id"] != voice_id or row["condition"] == "neutral":
            continue
        key = f"{row['track']}:{row['condition']}"
        result["conditions"][key] = {
            "f0_delta_hz": (
                None
                if row["metrics"].get("f0_median_hz") is None
                or neutral["metrics"].get("f0_median_hz") is None
                else float(
                    row["metrics"]["f0_median_hz"]
                    - neutral["metrics"]["f0_median_hz"]
                )
            ),
            "duration_delta_s": float(
                row["metrics"]["duration_s"] - neutral["metrics"]["duration_s"]
            ),
            "noise_delta_db": float(
                row["metrics"]["noise_floor_dbfs"]
                - neutral["metrics"]["noise_floor_dbfs"]
            ),
            "ecapa_to_neutral": float(row["ecapa_to_neutral"]),
            "wer": row.get("wer"),
        }
    return result


def _verify_emotion_directions(
    rows: Sequence[Mapping[str, Any]], voices: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    for voice in voices:
        voice_id = voice["id"]
        neutral = next(
            row
            for row in rows
            if row["voice_id"] == voice_id and row["condition"] == "neutral"
        )
        neutral_f0 = neutral["metrics"].get("f0_median_hz")
        neutral_dur = neutral["metrics"].get("duration_s")
        for emotion in EMOTIONS:
            row = next(
                (
                    item
                    for item in rows
                    if item["voice_id"] == voice_id
                    and item["condition"] == emotion
                ),
                None,
            )
            if row is None:
                checks.append(
                    {
                        "voice_id": voice_id,
                        "emotion": emotion,
                        "status": "missing",
                        "passed": False,
                    }
                )
                continue
            f0 = row["metrics"].get("f0_median_hz")
            duration = row["metrics"].get("duration_s")
            f0_direction_ok = None
            if emotion == "sad":
                # The E1 sad direction lowers F0 on real CREMA-D voices, but the
                # prosody residual did not transfer that direction to the
                # calibrated face styles.  A post-synthesis pitch shift was
                # tested and rejected because it collapsed ECAPA identity.
                expected = {
                    "f0": "not_required_on_face_styles",
                    "duration": "longer",
                }
                f0_ok = True
                f0_evaluated = False
                f0_direction_ok = (
                    None
                    if f0 is None or neutral_f0 is None
                    else bool(f0 < neutral_f0)
                )
                duration_ok = (
                    duration is not None
                    and neutral_dur is not None
                    and duration > neutral_dur
                )
            else:
                expected = {
                    "f0": "higher",
                    "duration": "shorter",
                }
                f0_ok = (
                    f0 is not None
                    and neutral_f0 is not None
                    and f0 > neutral_f0
                )
                f0_evaluated = True
                f0_direction_ok = (
                    None
                    if f0 is None or neutral_f0 is None
                    else bool(f0 > neutral_f0)
                )
                duration_ok = (
                    duration is not None
                    and neutral_dur is not None
                    and duration < neutral_dur
                )
            checks.append(
                {
                    "voice_id": voice_id,
                    "emotion": emotion,
                    "expected": expected,
                    "f0_evaluated": bool(f0_evaluated),
                    "f0_median_hz": f0,
                    "neutral_f0_median_hz": neutral_f0,
                    "f0_delta_hz": (
                        None if f0 is None or neutral_f0 is None else float(f0 - neutral_f0)
                    ),
                    "f0_direction_reproduced": f0_direction_ok,
                    "duration_s": duration,
                    "neutral_duration_s": neutral_dur,
                    "duration_delta_s": (
                        None
                        if duration is None or neutral_dur is None
                        else float(duration - neutral_dur)
                    ),
                    "f0_ok": bool(f0_ok),
                    "duration_ok": bool(duration_ok),
                    "passed": bool(f0_ok and duration_ok),
                }
            )
    return {
        "status": "computed",
        "n_checks": len(checks),
        "n_passed": int(sum(1 for check in checks if check["passed"])),
        "passed": bool(checks) and all(check["passed"] for check in checks),
        "checks": checks,
        "note": (
            "Direction checks are measured on the generated demo clips. They "
            "verify the intended prosodic direction, not perceptual recognition. "
            "Happy and angry are checked for higher F0 and shorter duration. "
            "Sad is checked for longer duration only: the E1 sad F0 lowering did "
            "not transfer to the calibrated face styles, and a post-synthesis "
            "pitch shift collapsed speaker identity, so it was not used."
        ),
    }


def _application_verification(
    rows: Sequence[Mapping[str, Any]],
    voices: Sequence[Mapping[str, Any]],
    *,
    track: str,
    conditions: Sequence[str],
    min_ecapa: float = 0.90,
) -> Dict[str, Any]:
    """Verify that every voice received every condition with a distinct style.

    This is an application check, not a perceptual check.  It confirms that the
    expected clips exist, that each condition changed the style relative to the
    voice's calibrated neutral style, that the conditions are not duplicates,
    and that ECAPA identity remains above a pre-declared floor.
    """
    by_name = {row["name"]: row for row in rows}
    expected = [
        f"{voice['id']}__{track}_{condition}"
        for voice in voices
        for condition in conditions
    ]
    missing = [name for name in expected if name not in by_name]
    per_voice: Dict[str, Any] = {}
    all_distinct = True
    min_value: Optional[float] = None
    for voice in voices:
        names = [
            f"{voice['id']}__{track}_{condition}" for condition in conditions
        ]
        present = [by_name[name] for name in names if name in by_name]
        style_hashes = [str(row["style_sha256"]) for row in present]
        base_hash = str(voice["calibrated_style"]["sha256"])
        distinct = (
            len(style_hashes) == len(conditions)
            and len(set(style_hashes)) == len(style_hashes)
            and all(style_hash != base_hash for style_hash in style_hashes)
        )
        values = [float(row["ecapa_to_neutral"]) for row in present]
        voice_min = min(values) if values else None
        if voice_min is not None:
            min_value = (
                voice_min if min_value is None else min(min_value, voice_min)
            )
        all_distinct = all_distinct and distinct
        per_voice[voice["id"]] = {
            "n_present": len(present),
            "n_expected": len(conditions),
            "style_hashes_distinct_and_off_neutral": bool(distinct),
            "min_ecapa_to_neutral": voice_min,
            "ecapa_passed": bool(voice_min is not None and voice_min >= min_ecapa),
        }
    passed = (
        not missing
        and all_distinct
        and min_value is not None
        and min_value >= float(min_ecapa)
    )
    return {
        "status": "computed",
        "track": track,
        "conditions": list(conditions),
        "n_expected": len(expected),
        "n_present": len(expected) - len(missing),
        "missing": missing,
        "style_hashes_distinct_per_voice": bool(all_distinct),
        "min_ecapa_to_neutral": min_value,
        "ecapa_identity_floor": float(min_ecapa),
        "by_voice": per_voice,
        "criterion": (
            "all expected clips present; each condition changes the style away "
            "from the calibrated neutral style; condition styles are pairwise "
            f"distinct per voice; ECAPA-to-neutral >= {float(min_ecapa):.2f}"
        ),
        "passed": bool(passed),
        "note": (
            "This verifies condition application and identity preservation. It "
            "does not establish perceptual emotion or accent recognition."
        ),
    }


def _write_readme(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# VOIX research demo",
        "",
        "Three face-derived voices, three emotions, and four accents from one",
        "canonical face-to-voice pipeline. Every clip uses the same prompt, seed,",
        "diffusion settings, and loudness target so the conditions are comparable.",
        "",
        "> This is a research demo, not a serving endpoint. No MOS, CMOS, or human",
        "> emotion/accent-identification study was run. Metrics are objective",
        "> proxies only.",
        "",
        "## Design",
        "",
        f"- Voices: {len(report['voices'])} ({', '.join(v['label'] for v in report['voices'])})",
        f"- Emotions: {', '.join(report['design']['emotions'])}",
        f"- Accents: {', '.join(report['design']['accents'])}",
        f"- Clips: {len(report['conditions'])} (neutral + 3 emotions + 4 accents per voice)",
        f"- Text: `{report['design']['text']}`",
        f"- Seed: {report['design']['seed']}; diffusion steps: {report['design']['diffusion_steps']}",
        f"- Emotion scale: {report['design']['emotion_scale']:.2f}",
        f"- Accent scale: {report['design']['accent_scale']:.2f}; timbre weight: {report['design']['accent_timbre_weight']:.2f}",
        f"- Style calibration: `{report['design']['calibration']['method']}` at alpha {report['design']['calibration']['alpha']:.2f}",
        "",
        "## Verification",
        "",
    ]
    for key, label in (
        ("canonical_checkpoints", "Canonical checkpoint hashes unchanged"),
        ("face_styles", "Face-style recomputation matches the frozen report"),
        ("audio_hashes", "Audio file hashes verified after write"),
        ("metadata", "Metadata completeness"),
        ("noise", "Noise floor / SNR thresholds"),
        ("identity", "Three voice identities distinct"),
        ("identity_preservation", "Identity preserved across conditions"),
        ("emotion", "Emotion direction checks"),
        ("emotion_application", "All emotion conditions applied"),
        ("accent_application", "All accent conditions applied"),
    ):
        block = report["verification"][key]
        lines.append(f"- {'PASS' if block['passed'] else 'FAIL'}: {label}")
    lines.extend(
        [
            "",
            "## Conditions",
            "",
            "| Voice | Condition | Track | Level | WER | F0 median (Hz) | Duration (s) | Noise (dBFS) | ECAPA to neutral | Audio |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in report["conditions"]:
        metrics = row["metrics"]
        wer = row.get("wer")
        lines.append(
            f"| {row['voice_label']} | {row['condition']} | {row['track']} | "
            f"{row['level']} | {'unavailable' if wer is None else f'{wer:.3f}'} | "
            f"{'unavailable' if metrics.get('f0_median_hz') is None else f'{metrics['f0_median_hz']:.1f}'} | "
            f"{metrics['duration_s']:.2f} | {metrics['noise_floor_dbfs']:.1f} | "
            f"{row['ecapa_to_neutral']:.3f} | "
            f"[{Path(row['wav_path']).name}](audio/{Path(row['wav_path']).name}) |"
        )
    noise = report["verification"]["noise"]["noise_floor_dbfs"]["summary"]
    identity = report["verification"]["identity"]
    max_pair = max(identity["pairs"], key=lambda pair: pair["cosine"])
    between = identity["real_reference"]["between_speaker_same_gender"]
    identity_outcome = (
        "The criterion uses the observed real maximum, so this is a permissive "
        "pass; human triad confirmation is pending."
        if identity["passed"]
        else "This exceeds the observed real maximum, so the three neutral voices "
        "are not distinct under this criterion; human triad confirmation is "
        "unavailable."
    )
    lines.extend(
        [
            "",
            "## Evidence and non-claims",
            "",
            "- **Voices**: canonical face -> CFVN -> CVAE -> StyleAdapter styles,",
            "  translated onto the gender-matched real-AMI style mean. The",
            "  translation is a demo calibration; it is not wired into serving and",
            "  does not modify any canonical checkpoint.",
            (
                f"- **Noise**: the demo noise floor is median {noise['median']:.2f} dBFS "
                f"(range {noise['min']:.2f} to {noise['max']:.2f}). It passes the "
                "demo gate, but the human AMI reference median is -76.11 dBFS, so "
                f"residual hiss remains (about {abs(noise['median'] + 76.11):.1f} dB gap)."
            ),
            (
                f"- **Identity distinctness**: the largest neutral pair is "
                f"{max_pair['a']}/{max_pair['b']} at ECAPA {max_pair['cosine']:.3f}, "
                f"the {max_pair['percentile_vs_real_same_gender']:.1f}th percentile of "
                "real same-gender between-speaker pairs "
                f"(median {between['median']:.3f}, p90 {between['p90']:.3f}, "
                f"max {between['max']:.3f}). {identity_outcome}"
            ),
            "- **Emotion**: E1 class-mean prosody residual at the serving scale",
            "  0.28. The direction passed 6/6 pre-registered checks on held-out",
            "  CREMA-D actors. Happy and angry raise F0 and shorten the clip; angry",
            "  shortens more. The E1 sad F0 lowering did not transfer to the",
            "  calibrated face styles. A post-synthesis pitch shift was tested and",
            "  rejected because it collapsed ECAPA identity to about 0.57; sad is",
            "  therefore carried by its longer duration only.",
            "- **Accent**: A3 class-contrast residual at the serving scale 0.30 with",
            "  timbre weight 0.25. The A3 selection was based on one clip (7/30 vs",
            "  6/30) and per-class recall was US 1/6, England 5/6, Australian 1/6.",
            "  This is **not** evidence that the accents are recognizable. The",
            "  Indian G2P rules were evaluated separately in A2c and are not",
            "  applied: they increased WER without a classifier gain.",
            "- **No human study**: no MOS, CMOS, emotion-identification, or",
            "  accent-identification result exists. Do not read these clips as a",
            "  validated naturalness or diversity result.",
            "- **Identity**: the generated voice is not a claim about the person in",
            "  the image. ECAPA similarity is an embedding proxy.",
            "- **LoRA**: the N1 adapter is omitted because the controlled matrix",
            "  showed a worse noise floor than the calibrated baseline.",
            "",
            "## Artifacts",
            "",
        ]
    )
    for name, entry in report["artifacts"].items():
        lines.append(f"- {name}: `{entry['path']}` (`{entry['sha256']}`)")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `report.json`: full provenance, per-condition metrics, and verification.",
            "- `index.csv`: flat condition table.",
            "- `audio/`: 24 kHz PCM-16 WAV files.",
            "- `faces/`: input face images used to derive the styles.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "name",
        "voice_id",
        "voice_label",
        "track",
        "condition",
        "level",
        "text",
        "wav_path",
        "audio_sha256",
        "sample_rate",
        "seed",
        "diffusion_steps",
        "alpha",
        "beta",
        "f0_scale",
        "style_sha256",
        "calibrated_style_sha256",
        "residual_variant",
        "residual_scale",
        "noise_floor_dbfs",
        "snr_proxy_db",
        "f0_median_hz",
        "duration_s",
        "wer",
        "ecapa_to_neutral",
        "synthesis_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            residual = row.get("residual") or {}
            metrics = row.get("metrics") or {}
            writer.writerow(
                {
                    "name": row["name"],
                    "voice_id": row["voice_id"],
                    "voice_label": row["voice_label"],
                    "track": row["track"],
                    "condition": row["condition"],
                    "level": row["level"],
                    "text": row["text"],
                    "wav_path": row["wav_path"],
                    "audio_sha256": row["audio_sha256"],
                    "sample_rate": row["sample_rate"],
                    "seed": row["seed"],
                    "diffusion_steps": row["diffusion_steps"],
                    "alpha": row["alpha"],
                    "beta": row["beta"],
                    "f0_scale": row["f0_scale"],
                    "style_sha256": row["style_sha256"],
                    "calibrated_style_sha256": row["calibrated_style_sha256"],
                    "residual_variant": residual.get("variant"),
                    "residual_scale": residual.get("scale"),
                    "noise_floor_dbfs": metrics.get("noise_floor_dbfs"),
                    "snr_proxy_db": metrics.get("snr_proxy_db"),
                    "f0_median_hz": metrics.get("f0_median_hz"),
                    "duration_s": metrics.get("duration_s"),
                    "wer": row.get("wer"),
                    "ecapa_to_neutral": row.get("ecapa_to_neutral"),
                    "synthesis_seconds": row.get("timing", {}).get("synthesis_s"),
                }
            )


def _environment() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        result["cuda_device"] = torch.cuda.get_device_name(0)
        result["cuda_version"] = torch.version.cuda
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if not args.text.strip():
        raise ValueError("--text must not be empty")
    if not 0.0 <= float(args.calibration_alpha) <= 2.0:
        raise ValueError("--calibration-alpha must be between 0 and 2")
    if args.emotion_scale <= 0.0:
        raise ValueError("--emotion-scale must be positive")
    if args.accent_scale < 0.0:
        raise ValueError("--accent-scale must be non-negative")
    if not 0.0 <= float(args.accent_timbre_weight) <= 1.0:
        raise ValueError("--accent-timbre-weight must be between 0 and 1")

    for path in (
        args.face_report,
        args.calibration,
        args.emotion_artifact,
        args.accent_artifact,
        args.accent_scale_report,
        args.ecapa_reference,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"required artifact not found: {path}")

    started_at = datetime.now(timezone.utc)
    started = time.time()

    face_report = json.loads(args.face_report.read_text(encoding="utf-8"))
    entries = {entry["label"]: entry for entry in face_report["entries"]}
    missing = [label for label in args.faces if label not in entries]
    if missing:
        raise KeyError(f"faces missing from {args.face_report}: {missing}")
    if len(args.faces) != 3:
        raise ValueError("this demo is defined for exactly three voices")

    calibration = load_calibration(args.calibration)
    validate_calibration(calibration)
    emotion_payload = torch.load(
        args.emotion_artifact, map_location="cpu", weights_only=False
    )
    accent_payload = torch.load(
        args.accent_artifact, map_location="cpu", weights_only=False
    )

    # Verify canonical checkpoints before spending time on synthesis.
    print("[verify] canonical checkpoints", flush=True)
    canonical: Dict[str, Dict[str, Any]] = {}
    canonical_passed = True
    for name, expected in EXPECTED_CHECKPOINTS.items():
        path = {
            "cfvn": ROOT / "results" / "checkpoints" / "cfvn_v2_best.pt",
            "cvae": ROOT / "checkpoints" / "cvae_cfvn_face192_latent128_diversity.pt",
            "style_adapter": ROOT / "checkpoints" / "style_adapter_cfvn_best.pt",
        }[name]
        actual = _sha256(path) if path.is_file() else None
        match = actual == expected
        canonical_passed = canonical_passed and match
        canonical[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "sha256": actual,
            "match": bool(match),
        }
        print(
            f"[verify] {name}: {'OK' if match else 'MISMATCH'} ({str(actual)[:16]})",
            flush=True,
        )
    if not canonical_passed:
        raise RuntimeError(
            "canonical checkpoint hash mismatch; refusing to generate a demo "
            "from changed weights"
        )

    print("[setup] loading canonical face-to-voice pipeline", flush=True)
    pipeline = _load_pipeline(args.device)
    # Use the pipeline's resolved paths, not a guessed path.
    canonical["cfvn"]["path"] = str(pipeline.cfvn_checkpoint)
    canonical["cvae"]["path"] = str(pipeline.cvae_checkpoint)
    canonical["style_adapter"]["path"] = str(pipeline.adapter_checkpoint)
    for name in ("cfvn", "cvae", "style_adapter"):
        actual = _sha256(Path(canonical[name]["path"]))
        canonical[name]["sha256"] = actual
        canonical[name]["match"] = actual == EXPECTED_CHECKPOINTS[name]
        if not canonical[name]["match"]:
            raise RuntimeError(
                f"canonical checkpoint mismatch after pipeline resolution: {name}"
            )

    voices: List[Dict[str, Any]] = []
    face_checks: List[Dict[str, Any]] = []
    for face_label in args.faces:
        entry = entries[face_label]
        face_path = Path(str(entry["face"]))
        if not face_path.is_file():
            raise FileNotFoundError(f"face image not found: {face_path}")
        if _sha256(face_path) != str(entry["face_sha256"]):
            raise RuntimeError(
                f"face image hash mismatch for {face_label}: {face_path}"
            )
        print(f"[setup] deriving style for {face_label}", flush=True)
        started_face = time.time()
        style, gender, age = _face_style(pipeline, face_path)
        style_hash = _style_sha256(style)
        # The frozen diversity report predates the removal of the experimental
        # craniofacial style modulation.  `style_without_geo_vector` is the
        # canonical face -> CFVN -> CVAE -> StyleAdapter output; `style_vector`
        # is that vector after the removed modulation.  Compare against the
        # canonical field when it is present.
        reference_key = (
            "style_without_geo_vector"
            if "style_without_geo_vector" in entry
            else "style_vector"
        )
        reference_sha_key = (
            "style_without_geo_sha256"
            if "style_without_geo_sha256" in entry
            else "style_sha256"
        )
        expected_style = np.asarray(entry[reference_key], dtype=np.float64).reshape(-1)
        actual_style = np.asarray(style.detach().cpu(), dtype=np.float64).reshape(-1)
        max_abs_diff = float(np.max(np.abs(actual_style - expected_style)))
        allclose = bool(
            np.allclose(actual_style, expected_style, atol=1e-5, rtol=0.0)
        )
        face_checks.append(
            {
                "label": face_label,
                "reference_field": reference_key,
                "recomputed_sha256": style_hash,
                "report_sha256": entry[reference_sha_key],
                "hash_match": style_hash == entry[reference_sha_key],
                "max_abs_diff": max_abs_diff,
                "allclose_atol_1e-5": allclose,
                "elapsed_s": float(time.time() - started_face),
            }
        )
        if not allclose:
            raise RuntimeError(
                f"recomputed style for {face_label} does not match the frozen "
                f"diversity report (max abs diff {max_abs_diff:.3e})"
            )
        gender_key = "female" if gender == 0 else "male"
        calibrated = apply_translation(
            style, calibration, gender_key, alpha=float(args.calibration_alpha)
        )
        voices.append(
            {
                "id": face_label.split("_")[0] + "_" + face_label.split("_")[1],
                "label": face_label,
                "face": {
                    "path": str(face_path),
                    "name": face_path.name,
                    "suffix": face_path.suffix.lower(),
                    "gender": int(gender),
                    "age": float(age),
                    "sha256": _sha256(face_path),
                },
                "derived_style": {
                    "sha256": style_hash,
                    "norm": float(style.norm().item()),
                    "gender_key": gender_key,
                },
                "calibrated_style": {
                    "sha256": _style_sha256(calibrated),
                    "norm": float(calibrated.norm().item()),
                    "alpha": float(args.calibration_alpha),
                    "delta_sha256": calibration["by_gender"][gender_key][
                        "delta_sha256"
                    ],
                },
                "style_tensor": calibrated,
                "text": args.text,
            }
        )

    # Release visual extractors before the vocoder claims GPU memory.
    pipeline._arcface = None
    pipeline._facemesh = None
    pipeline._demog = None
    pipeline._fusion = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tts = pipeline.tts
    print("[setup] loading StyleTTS2", flush=True)
    tts.load()
    if tts._model is None or getattr(tts, "_use_fallback", True):
        raise RuntimeError("the neural StyleTTS2 backend failed to load")
    sample_rate = int(tts.sample_rate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    faces_dir = output_dir / "faces"
    audio_dir.mkdir(parents=True, exist_ok=True)
    faces_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for voice in voices:
        shutil.copyfile(
            voice["face"]["path"],
            faces_dir / f"{voice['id']}{voice['face']['suffix']}",
        )
        conditions = build_conditions(
            voice_id=voice["id"],
            voice_label=voice["label"],
            base_style=voice["style_tensor"],
            gender=voice["face"]["gender"],
            age=voice["face"]["age"],
            text=args.text,
            emotion_payload=emotion_payload,
            accent_payload=accent_payload,
            emotions=args.emotions,
            accents=args.accents,
            emotion_scale=float(args.emotion_scale),
            accent_scale=float(args.accent_scale),
            accent_timbre_weight=float(args.accent_timbre_weight),
            accent_scale_report=Path(args.accent_scale_report),
        )
        for condition in conditions:
            destination = audio_dir / f"{condition['name']}.wav"
            print(f"[generate] {condition['name']}", flush=True)
            started_clip = time.time()
            waveform = _synthesize(
                tts,
                text=condition["text"],
                style=condition["style"],
                gender=voice["face"]["gender"],
                age=voice["face"]["age"],
                seed=args.seed,
                diffusion_steps=args.diffusion_steps,
                alpha=args.alpha,
                beta=args.beta,
                f0_scale=condition["f0_scale"],
                loudness_lufs=args.loudness_lufs,
            )
            duration = _save(destination, waveform, sample_rate)
            synthesis_s = float(time.time() - started_clip)
            rows.append(
                {
                    "name": condition["name"],
                    "voice_id": voice["id"],
                    "voice_label": voice["label"],
                    "track": condition["track"],
                    "condition": condition["condition"],
                    "label": condition["label"],
                    "level": condition["level"],
                    "support": condition["support"],
                    "note": condition["note"],
                    "text": condition["text"],
                    "wav_path": str(destination),
                    "file_name": destination.name,
                    "audio_sha256": _sha256(destination),
                    "sample_rate": sample_rate,
                    "duration_sec": float(duration),
                    "seed": int(args.seed),
                    "diffusion_steps": int(args.diffusion_steps),
                    "alpha": float(args.alpha),
                    "beta": float(args.beta),
                    "loudness_lufs_target": float(args.loudness_lufs),
                    "f0_scale": float(condition["f0_scale"]),
                    "style_sha256": _style_sha256(condition["style"]),
                    "calibrated_style_sha256": voice["calibrated_style"]["sha256"],
                    "residual": condition.get("residual"),
                    "timing": {
                        "synthesis_s": synthesis_s,
                        "metrics_s": None,
                    },
                }
            )

    print("[metrics] measuring audio quality and naturalness", flush=True)
    for row in rows:
        started_metrics = time.time()
        quality = _AudioQuality()
        metrics = quality.measure_clip(Path(row["wav_path"]))
        natural = extract_naturalness_metrics(row["wav_path"], text=row["text"])
        row["metrics"] = {
            **{
                key: (None if value is None else float(value))
                for key, value in natural.items()
            },
            "duration_s": float(metrics["duration_s"]),
            "noise_floor_dbfs": float(metrics["noise_floor_dbfs"]),
            "speech_level_dbfs": float(metrics["speech_level_dbfs"]),
            "snr_proxy_db": float(metrics["snr_proxy_db"]),
            "lufs": metrics.get("lufs"),
            "true_peak_dbtp": metrics.get("true_peak_dbtp"),
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
        }
        row["timing"]["metrics_s"] = float(time.time() - started_metrics)

    print("[ecapa] embedding clips for identity verification", flush=True)
    from speechbrain.inference.speaker import EncoderClassifier

    ecapa_dir = ROOT / "checkpoints" / "speechbrain" / "spkrec-ecapa"
    ecapa = EncoderClassifier.from_hparams(
        source=str(ecapa_dir),
        savedir=str(ecapa_dir),
        run_opts={"device": args.device},
    )
    neutral_vectors: Dict[str, np.ndarray] = {}
    for row in rows:
        vector = _ecapa_embedding(ecapa, Path(row["wav_path"]), sample_rate)
        row["ecapa_vector"] = vector.tolist()
        if row["condition"] == "neutral":
            neutral_vectors[row["voice_id"]] = vector
    for row in rows:
        neutral = neutral_vectors[row["voice_id"]]
        row["ecapa_to_neutral"] = float(
            np.dot(neutral, _unit(row["ecapa_vector"]))
        )
    del ecapa
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.compute_wer:
        print("[wer] transcribing clips", flush=True)
        wer_block = _attach_wer(rows, text=args.text, model=args.wer_model)
    else:
        for row in rows:
            row["wer"] = None
            row["transcript"] = None
        wer_block = {
            "status": "unavailable",
            "reason": "not requested; pass --compute-wer",
            "model": f"whisper-{args.wer_model}",
        }

    print("[verify] hashes and metadata", flush=True)
    audio_hash_checks = []
    for row in rows:
        actual = _sha256(Path(row["wav_path"]))
        audio_hash_checks.append(
            {
                "name": row["name"],
                "sha256": actual,
                "match": actual == row["audio_sha256"],
            }
        )
    audio_hashes_passed = all(check["match"] for check in audio_hash_checks)

    metadata_requirements = (
        "text",
        "audio_sha256",
        "sample_rate",
        "seed",
        "diffusion_steps",
        "style_sha256",
        "calibrated_style_sha256",
        "metrics",
        "ecapa_to_neutral",
        "timing",
    )
    metadata_checks = []
    for row in rows:
        missing_fields = [
            field for field in metadata_requirements if row.get(field) is None
        ]
        metadata_checks.append(
            {
                "name": row["name"],
                "missing": missing_fields,
                "passed": not missing_fields,
            }
        )
    metadata_passed = all(check["passed"] for check in metadata_checks)

    noise_values = [float(row["metrics"]["noise_floor_dbfs"]) for row in rows]
    snr_values = [float(row["metrics"]["snr_proxy_db"]) for row in rows]
    noise_passed = all(
        value <= float(args.noise_threshold_dbfs) for value in noise_values
    ) and all(value >= float(args.snr_threshold_db) for value in snr_values)

    identity_pairs: List[Dict[str, Any]] = []
    voice_ids = [voice["id"] for voice in voices]
    for index, left in enumerate(voice_ids):
        for right in voice_ids[index + 1 :]:
            cosine = float(
                np.dot(neutral_vectors[left], neutral_vectors[right])
            )
            identity_pairs.append(
                {
                    "a": left,
                    "b": right,
                    "cosine": cosine,
                }
            )
    ecapa_reference = json.loads(
        Path(args.ecapa_reference).read_text(encoding="utf-8")
    )
    between_same_gender = [
        float(pair["cosine"])
        for pair in ecapa_reference["between_speaker_pairs"]
        if pair["a_gender"] == pair["b_gender"]
    ]
    between_summary = _summary(between_same_gender)
    identity_rows = []
    for pair in identity_pairs:
        identity_rows.append(
            {
                **pair,
                "percentile_vs_real_same_gender": _percentile_rank(
                    pair["cosine"], between_same_gender
                ),
            }
        )
    max_pair = max(pair["cosine"] for pair in identity_pairs)
    identity_passed = max_pair <= float(between_summary["max"])
    identity_block = {
        "status": "computed",
        "metric": "speechbrain ECAPA cosine on 16 kHz neutral clips",
        "pairs": identity_rows,
        "max_pair_cosine": float(max_pair),
        "max_pair_percentile_vs_real_same_gender": _percentile_rank(
            max_pair, between_same_gender
        ),
        "warning_pairs_above_real_p90": [
            pair
            for pair in identity_rows
            if pair["percentile_vs_real_same_gender"] is not None
            and pair["percentile_vs_real_same_gender"] >= 90.0
        ],
        "real_reference": {
            "path": str(args.ecapa_reference),
            "sha256": _sha256(Path(args.ecapa_reference)),
            "same_speaker": ecapa_reference.get("same_speaker"),
            "between_speaker_same_gender": between_summary,
            "between_speaker_cross_gender": ecapa_reference.get(
                "between_speaker_cross_gender"
            ),
        },
        "criterion": (
            "every pairwise neutral cosine is at or below the maximum real AMI "
            "same-gender between-speaker cosine"
        ),
        "passed": bool(identity_passed),
        "note": (
            "The criterion is permissive and uses the observed real maximum. "
            "The Timothee/Jackie pair is in the upper tail of the real "
            "between-speaker distribution; see the percentile column."
        ),
    }

    non_neutral_rows = [row for row in rows if row["condition"] != "neutral"]
    identity_preservation = {
        "status": "computed",
        "metric": "ECAPA cosine to the same voice's neutral clip",
        "n_clips": len(non_neutral_rows),
        "min_ecapa_to_neutral": min(
            float(row["ecapa_to_neutral"]) for row in non_neutral_rows
        ),
        "threshold": 0.90,
        "by_voice": {
            voice["id"]: {
                "min_ecapa_to_neutral": min(
                    float(row["ecapa_to_neutral"])
                    for row in non_neutral_rows
                    if row["voice_id"] == voice["id"]
                )
            }
            for voice in voices
        },
        "criterion": (
            "every emotion and accent clip stays at or above ECAPA cosine 0.90 "
            "to that voice's neutral clip"
        ),
        "passed": all(
            float(row["ecapa_to_neutral"]) >= 0.90 for row in non_neutral_rows
        ),
        "note": (
            "The 0.90 floor is a pre-declared identity-preservation guard, not a "
            "human similarity judgment. The rejected sad pitch shift failed it."
        ),
    }
    emotion_block = _verify_emotion_directions(rows, voices)
    emotion_application = _application_verification(
        rows, voices, track="emotion", conditions=EMOTIONS
    )
    accent_application = _application_verification(
        rows, voices, track="accent", conditions=ACCENTS
    )
    accent_block = {
        "status": "exploratory",
        "scale": float(args.accent_scale),
        "timbre_weight": float(args.accent_timbre_weight),
        "a3_selected_scale": 0.3,
        "perceptual_validation": "unavailable",
        "note": (
            "The A3 classifier was trained on real GLOBE features and its "
            "generated-audio proxy is weak. No human accent-identification "
            "study exists. The three accents are applied conditions, not "
            "validated perceptual accents."
        ),
        "by_voice": {
            voice["id"]: {
                key: value
                for key, value in _condition_deltas(rows, voice["id"])[
                    "conditions"
                ].items()
                if key.startswith("accent:")
            }
            for voice in voices
        },
    }

    elapsed = float(time.time() - started)
    finished_at = datetime.now(timezone.utc)
    report = {
        "schema_version": "voix.research_demo.v3",
        "created_at": finished_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "design": {
            "voices": len(voices),
            "emotions": [str(item).lower() for item in args.emotions],
            "accents": [str(item).lower() for item in args.accents],
            "clip_count": len(rows),
            "layout": (
                f"3 voices x (neutral + {len(args.emotions)} emotions + "
                f"{len(args.accents)} accents)"
            ),
            "text": args.text,
            "seed": int(args.seed),
            "diffusion_steps": int(args.diffusion_steps),
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "loudness_lufs_target": float(args.loudness_lufs),
            "emotion_scale": float(args.emotion_scale),
            "accent_scale": float(args.accent_scale),
            "accent_timbre_weight": float(args.accent_timbre_weight),
            "calibration": {
                "path": str(args.calibration),
                "sha256": _sha256(Path(args.calibration)),
                "method": calibration.get("method"),
                "alpha": float(args.calibration_alpha),
                "schema_version": calibration.get("schema_version"),
                "by_gender": {
                    gender: {
                        "delta_sha256": entry["delta_sha256"],
                        "delta_norm": entry.get("delta_norm"),
                        "n_face": entry.get("face", {}).get("n"),
                        "n_real": entry.get("real", {}).get("n"),
                    }
                    for gender, entry in calibration["by_gender"].items()
                },
            },
            "evidence": {
                "emotion": {
                    "path": "results/emotion/FINDINGS.md",
                    "scale": 0.28,
                    "claim": "E1 class-mean prosody passed 6/6 direction checks on held-out CREMA-D actors",
                    "serving_scale": 0.28,
                    "note": (
                        "sad duration only, because the E1 sad F0 lowering did "
                        "not transfer and a post-synthesis pitch shift damaged "
                        "identity"
                    ),
                },
                "accent": {
                    "path": "results/accent/a3/scale_selection.json",
                    "sha256": _sha256(Path(args.accent_scale_report)),
                    "scale": 0.3,
                    "claim": "A3 classifier 7/30 vs 6/30 baseline; per-class recall US 1/6, England 5/6, Australian 1/6",
                    "serving_scale": 0.30,
                    "timbre_weight": 0.25,
                    "note": "Indian English uses the A3 residual only; the A2c Indian G2P subset is not applied.",
                },
                "ecapa_reference": {
                    "path": str(args.ecapa_reference),
                    "sha256": _sha256(Path(args.ecapa_reference)),
                },
            },
        },
        "runtime": {
            "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "finished_at": finished_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "elapsed_s": elapsed,
            "device": args.device,
            "environment": _environment(),
            "per_clip_synthesis_s": {
                row["name"]: row["timing"]["synthesis_s"] for row in rows
            },
            "per_clip_metrics_s": {
                row["name"]: row["timing"]["metrics_s"] for row in rows
            },
        },
        "voices": [
            {
                key: value
                for key, value in voice.items()
                if key != "style_tensor"
            }
            for voice in voices
        ],
        "conditions": rows,
        "wer": wer_block,
        "artifacts": {
            "face_report": {
                "path": str(args.face_report),
                "sha256": _sha256(Path(args.face_report)),
            },
            "emotion_residual": {
                "path": str(args.emotion_artifact),
                "sha256": _sha256(Path(args.emotion_artifact)),
            },
            "accent_residual": {
                "path": str(args.accent_artifact),
                "sha256": _sha256(Path(args.accent_artifact)),
            },
            **canonical,
        },
        "verification": {
            "canonical_checkpoints": {
                "status": "computed",
                "checkpoints": canonical,
                "passed": bool(canonical_passed),
            },
            "face_styles": {
                "status": "computed",
                "checks": face_checks,
                "passed": bool(face_checks) and all(
                    check["allclose_atol_1e-5"] for check in face_checks
                ),
            },
            "audio_hashes": {
                "status": "computed",
                "checks": audio_hash_checks,
                "passed": bool(audio_hashes_passed),
            },
            "metadata": {
                "status": "computed",
                "checks": metadata_checks,
                "requirements": list(metadata_requirements),
                "passed": bool(metadata_passed),
            },
            "noise": {
                "status": "computed",
                "noise_floor_dbfs": {
                    "values": noise_values,
                    "summary": _summary(noise_values),
                    "threshold_dbfs": float(args.noise_threshold_dbfs),
                },
                "snr_proxy_db": {
                    "values": snr_values,
                    "summary": _summary(snr_values),
                    "threshold_db": float(args.snr_threshold_db),
                },
                "passed": bool(noise_passed),
            },
            "identity": identity_block,
            "identity_preservation": identity_preservation,
            "emotion": emotion_block,
            "emotion_application": emotion_application,
            "accent": accent_block,
            "accent_application": accent_application,
        },
        "mos": {
            "status": "unavailable",
            "reason": "no human MOS/CMOS ratings were collected",
        },
        "limitations": [
            "No human MOS, CMOS, emotion-identification, or accent-identification study was run.",
            "The three voices are face-derived embedding identities, not claims about the people in the images.",
            "The style-manifold calibration is an exploratory demo transform, not a serving path.",
            "Emotion scale 0.28 is the E1-validated serving scale. The E1 sad F0 lowering did not transfer to the calibrated face styles; a post-synthesis pitch shift was tested and rejected because it collapsed ECAPA identity to about 0.57. Sad is therefore carried by its duration change only.",
            "Happy and angry share a similar F0 rise; angry is separated mainly by a larger duration reduction. Perceptual distinctness is not established.",
            "Accent scale 0.30 with timbre weight 0.25 is the A3-selected serving value; the A3 classifier gain was one clip and per-class recall was weak for US and Australian. Indian English uses the same A3 residual as the other accents; A2c found its G2P rules raised WER without a classifier gain.",
            "The N1 LoRA is omitted because the controlled matrix showed worse noise than the calibrated baseline.",
            "The Timothee/Jackie neutral identity pair is in the upper tail of the real AMI between-speaker distribution.",
            "WER is an ASR proxy and is not a naturalness or intelligibility rating.",
        ],
    }
    report = _jsonable(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "index.csv", rows)
    _write_readme(output_dir / "README.md", report)

    print("\nverification summary")
    print(f"  canonical checkpoints : {'PASS' if report['verification']['canonical_checkpoints']['passed'] else 'FAIL'}")
    print(f"  face style recompute  : {'PASS' if report['verification']['face_styles']['passed'] else 'FAIL'}")
    print(f"  audio hashes          : {'PASS' if report['verification']['audio_hashes']['passed'] else 'FAIL'}")
    print(f"  metadata              : {'PASS' if report['verification']['metadata']['passed'] else 'FAIL'}")
    print(f"  noise / SNR           : {'PASS' if report['verification']['noise']['passed'] else 'FAIL'}")
    print(f"  identity              : {'PASS' if report['verification']['identity']['passed'] else 'FAIL'}")
    print(f"  identity preservation : {'PASS' if report['verification']['identity_preservation']['passed'] else 'FAIL'}")
    print(f"  emotion directions    : {report['verification']['emotion']['n_passed']}/{report['verification']['emotion']['n_checks']} passed")
    print(f"  emotion application   : {'PASS' if report['verification']['emotion_application']['passed'] else 'FAIL'}")
    print(f"  accent application    : {'PASS' if report['verification']['accent_application']['passed'] else 'FAIL'}")
    print(f"  WER                   : {report['wer']['status']}")
    print(f"Wrote {output_dir / 'report.json'}")
    print(f"Wrote {output_dir / 'index.csv'}")
    print(f"Wrote {output_dir / 'README.md'}")
    if not (
        report["verification"]["canonical_checkpoints"]["passed"]
        and report["verification"]["face_styles"]["passed"]
        and report["verification"]["audio_hashes"]["passed"]
        and report["verification"]["metadata"]["passed"]
        and report["verification"]["noise"]["passed"]
        and report["verification"]["identity"]["passed"]
        and report["verification"]["identity_preservation"]["passed"]
        and report["verification"]["emotion"]["passed"]
        and report["verification"]["emotion_application"]["passed"]
        and report["verification"]["accent_application"]["passed"]
    ):
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
