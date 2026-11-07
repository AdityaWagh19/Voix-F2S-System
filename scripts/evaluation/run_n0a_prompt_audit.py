"""N0a: prompt/delivery audit on the canonical face-to-voice pipeline.

This is an inference-time audit, not training.  One real VoxCeleb2 face is held
fixed; the text register is varied across ten formal/conversational prompt
pairs; three seeds per cell provide a resampling noise floor.  The audit asks
whether conversational phrasing changes objective delivery proxies more than
the pipeline's own seed noise.

It does not use AMI, does not train a LoRA, does not touch the canonical
checkpoints, and does not claim perceptual naturalness.  MOS is unavailable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.identity_guardrail import verify_canonical_hashes
from voix.evaluation.naturalness import (
    CLIP_METRICS,
    PROMPT_EFFECT_METRICS,
    PROMPT_REGISTERS,
    extract_naturalness_metrics,
    load_prompt_pairs,
    paired_prompt_effects,
    summarize_prompt_effects,
)
from voix.evaluation.research_report import (
    build_report,
    computed_metric,
    sha256_file,
    unavailable_metric,
    write_report,
)
from voix.evaluation.stats import bootstrap_mean_ci
from voix.utils.paths import project_path

DEFAULT_FEATURES = project_path(
    "data", "processed", "voxceleb2_v2", "pilot100_features.h5"
)
DEFAULT_PROMPTS = project_path("configs", "research", "n0a_prompt_pairs.json")
DEFAULT_OUTPUT_DIR = project_path("results", "naturalness", "n0a", "audio")
DEFAULT_INDEX = project_path("results", "naturalness", "n0a", "clips.csv")
DEFAULT_REPORT = project_path("results", "naturalness", "n0a", "report.json")

METRIC_UNITS = {
    "f0_std_st": "semitones",
    "f0_range_st": "semitones (p95-p5)",
    "words_per_second": "words/s of speech span",
    "energy_std_db": "dB",
    "pause_ratio": "ratio",
    "spectral_tilt_db_per_khz": "dB/kHz",
}

CSV_COLUMNS = [
    "audio_id",
    "prompt_pair_id",
    "topic",
    "register",
    "seed",
    "text",
    "audio_path",
    "status",
    "error",
    *CLIP_METRICS,
    "style_embedding_sha256",
    "speaker_embedding_sha256",
    "backend",
    "sample_rate",
    "diffusion_steps",
    "temperature",
    "watermark",
    "face_speaker_id",
    "face_clip_id",
]


def _decode(values: Sequence[Any]) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values
        ],
        dtype=object,
    )


def _sha256_array(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    return hashlib.sha256(array.tobytes()).hexdigest()


def select_face_clip(
    features_path: Path,
    *,
    split: str = "val",
    speaker_id: Optional[str] = None,
    clip_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Deterministically select one real face clip from the VoxCeleb2 store.

    Speaker selection uses the best overall face-quality score in the split;
    clip selection uses the same score within the chosen speaker.  Ties break
    by speaker/clip id so the audit is reproducible.
    """
    import h5py

    with h5py.File(features_path, "r") as store:
        required = {
            "clip_ids",
            "speaker_ids",
            "split",
            "face_features",
            "face_gender",
            "face_age",
            "face_quality",
        }
        missing = required - set(store.keys())
        if missing:
            raise ValueError(
                f"{features_path} is missing datasets: {sorted(missing)}"
            )
        clip_ids = _decode(store["clip_ids"][:])
        speaker_ids = _decode(store["speaker_ids"][:])
        splits = _decode(store["split"][:])
        features = np.asarray(store["face_features"][:], dtype=np.float32)
        genders = np.asarray(store["face_gender"][:], dtype=np.int64).reshape(-1)
        ages = np.asarray(store["face_age"][:], dtype=np.float32).reshape(-1)
        quality = np.asarray(store["face_quality"][:], dtype=np.float32)

    if features.ndim != 2 or features.shape[1] != 560:
        raise ValueError(
            f"face_features must be (N, 560), got {tuple(features.shape)}"
        )
    if quality.ndim != 2 or quality.shape[1] < 11:
        raise ValueError(
            f"face_quality must have at least 11 columns, got {tuple(quality.shape)}"
        )

    mask = splits == split
    if speaker_id is not None:
        mask &= speaker_ids == str(speaker_id)
    if clip_id is not None:
        mask &= clip_ids == str(clip_id)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(
            f"no {split} clip matched speaker={speaker_id!r} clip={clip_id!r}"
        )

    if speaker_id is None:
        best_by_speaker: Dict[str, Tuple[float, int]] = {}
        for index in indices:
            score = float(quality[index, 10])
            speaker = str(speaker_ids[index])
            if speaker not in best_by_speaker or score > best_by_speaker[speaker][0]:
                best_by_speaker[speaker] = (score, int(index))
        chosen_speaker = sorted(
            best_by_speaker.items(), key=lambda item: (-item[1][0], item[0])
        )[0][0]
        indices = indices[speaker_ids[indices] == chosen_speaker]

    chosen = sorted(
        (int(index) for index in indices),
        key=lambda index: (-float(quality[index, 10]), str(clip_ids[index])),
    )[0]
    feature = np.asarray(features[chosen], dtype=np.float32)
    if not np.all(np.isfinite(feature)):
        raise ValueError(f"selected face feature is non-finite: {clip_ids[chosen]}")
    age = float(ages[chosen])
    return {
        "features_path": str(features_path),
        "features_sha256": sha256_file(features_path),
        "split": str(splits[chosen]),
        "speaker_id": str(speaker_ids[chosen]),
        "clip_id": str(clip_ids[chosen]),
        "face_features": feature,
        "face_features_sha256": _sha256_array(feature),
        "face_gender": int(genders[chosen]),
        "face_age": age if age > 0.0 else 30.0,
        "face_age_raw": age,
        "quality_vector": [float(value) for value in quality[chosen]],
        "quality_score": float(quality[chosen, 10]),
        "av_sync": float(quality[chosen, 3]),
        "face_area": float(quality[chosen, 6]),
    }


def _sidecar_path(audio_path: Path) -> Path:
    return audio_path.with_suffix(".json")


def _write_sidecar(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_sidecar(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _save_waveform(path: Path, waveform: Any, sample_rate: int) -> None:
    import soundfile as sf

    if hasattr(waveform, "detach"):
        array = waveform.detach().cpu().numpy()
    else:
        array = np.asarray(waveform)
    array = np.asarray(array, dtype=np.float32).squeeze()
    if array.ndim != 1:
        raise ValueError(f"waveform must be mono, got {array.shape}")
    if array.size == 0:
        raise ValueError("refusing to write an empty waveform")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), array, int(sample_rate))


def _generate_one(
    pipeline: Any,
    *,
    face_features: np.ndarray,
    text: str,
    seed: int,
    gender: int,
    age: float,
    temperature: float,
) -> Dict[str, Any]:
    results = pipeline.generate(
        face_input=face_features,
        text=text,
        num_voices=1,
        temperature=float(temperature),
        watermark=False,
        gender=int(gender),
        age=float(age),
        seed=int(seed),
        selection=None,
    )
    if len(results) != 1:
        raise RuntimeError(
            f"expected exactly one generated candidate, got {len(results)}"
        )
    return results[0]


def _ecapa_prompt_consistency(
    rows: Sequence[Mapping[str, Any]],
    *,
    device: str,
    n_resamples: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    """ECAPA cosine between the two registers at the same seed.

    This is a generated-domain consistency check, not a real-speaker
    verification score and not an identity-guardrail replacement.
    """
    try:
        import torch
        import torch.nn.functional as F

        from voix.data.speaker_extractor import ECAPAExtractor
    except Exception as exc:  # pragma: no cover - environment-dependent
        return {
            "status": "unavailable",
            "reason": f"ECAPA dependencies unavailable: {type(exc).__name__}: {exc}",
        }

    by_key: Dict[Tuple[str, int], Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (str(row["prompt_pair_id"]), int(row["seed"]))
        by_key.setdefault(key, {})[str(row["register"])] = row

    pairs: Dict[str, List[float]] = {}
    embeddings: Dict[str, Any] = {}
    extractor = ECAPAExtractor(device=device)
    try:
        for key, registers in sorted(by_key.items()):
            if "formal" not in registers or "conversational" not in registers:
                continue
            formal = registers["formal"]
            conversational = registers["conversational"]
            for row in (formal, conversational):
                audio_id = str(row["audio_id"])
                if audio_id not in embeddings:
                    embeddings[audio_id] = extractor.extract_from_path(
                        str(row["audio_path"])
                    ).reshape(-1)
            cosine = float(
                F.cosine_similarity(
                    embeddings[str(formal["audio_id"])],
                    embeddings[str(conversational["audio_id"])],
                    dim=0,
                ).item()
            )
            pairs.setdefault(key[0], []).append(cosine)
    finally:
        del extractor

    per_pair = {
        pair_id: float(np.mean(values)) for pair_id, values in pairs.items()
    }
    if len(per_pair) < 2:
        return {
            "status": "unavailable",
            "reason": "fewer than two prompt pairs with both registers",
        }
    interval = bootstrap_mean_ci(
        list(per_pair.values()), n_resamples=n_resamples, seed=bootstrap_seed
    )
    all_values = [value for values in pairs.values() for value in values]
    return {
        "status": "computed",
        "method": "ECAPA cosine between formal and conversational audio at the same seed",
        "mean_cosine": interval.estimate,
        "ci_low": interval.ci_low,
        "ci_high": interval.ci_high,
        "min_cosine": float(np.min(all_values)),
        "n_pair_seed_observations": int(len(all_values)),
        "n_prompt_pairs": int(len(per_pair)),
        "per_pair": per_pair,
        "note": (
            "Generated-domain consistency only; not a real-speaker "
            "verification score and not the CFVN identity guardrail."
        ),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="N0a prompt/delivery audit on the canonical face-to-voice pipeline"
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--split", default="val")
    parser.add_argument("--speaker-id", default=None)
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--beta", type=float, default=0.22)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--run-id", default="n0a_prompt_audit")
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-identity-check",
        action="store_true",
        help="Skip the ECAPA generated-domain prompt-consistency check",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested but CUDA is unavailable")
    if args.diffusion_steps < 1:
        raise ValueError("--diffusion-steps must be positive")

    seeds = [int(value.strip()) for value in str(args.seeds).split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")

    pairs = load_prompt_pairs(args.prompts)
    if args.max_pairs is not None:
        pairs = pairs[: int(args.max_pairs)]
    if not pairs:
        raise ValueError("no prompt pairs selected")

    face = select_face_clip(
        args.features,
        split=args.split,
        speaker_id=args.speaker_id,
        clip_id=args.clip_id,
    )
    print(
        f"Face: {face['speaker_id']} / {face['clip_id']} "
        f"({face['split']}, gender={face['face_gender']}, "
        f"age={face['face_age']:.1f}, quality={face['quality_score']:.3f})",
        flush=True,
    )

    canonical = verify_canonical_hashes()
    if not canonical.get("passed"):
        raise RuntimeError(
            "canonical checkpoint hashes do not match results/canonical_hashes.json; "
            "refusing to run N0a on modified canonical artifacts"
        )

    from voix.inference.pipeline import VoixPipeline

    pipeline = VoixPipeline(
        device=args.device,
        enable_watermarking=False,
        tts_use_neural=True,
        tts_allow_fallback=False,
    )
    # Inference-only audit override.  This does not modify serving defaults or
    # any checkpoint; it pins the N0a step count and style controls.
    pipeline._inf_cfg = {
        **pipeline._inf_cfg,
        "diffusion_steps": int(args.diffusion_steps),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "duration_scale": 1.0,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for pair in pairs:
        for register in PROMPT_REGISTERS:
            text = str(pair[register])
            for seed in seeds:
                audio_id = f"{pair['pair_id']}_{register}_seed{seed}"
                audio_path = args.output_dir / f"{audio_id}.wav"
                sidecar_path = _sidecar_path(audio_path)
                sidecar = None if args.overwrite else _load_sidecar(sidecar_path)
                if (
                    sidecar is not None
                    and audio_path.is_file()
                    and "sample_rate" in sidecar
                ):
                    print(f"  reuse {audio_id}", flush=True)
                else:
                    print(f"  synth {audio_id}", flush=True)
                    try:
                        result = _generate_one(
                            pipeline,
                            face_features=face["face_features"],
                            text=text,
                            seed=seed,
                            gender=face["face_gender"],
                            age=face["face_age"],
                            temperature=args.temperature,
                        )
                        _save_waveform(
                            audio_path,
                            result["waveform"],
                            int(result["sample_rate"]),
                        )
                        sidecar = {
                            "audio_id": audio_id,
                            "prompt_pair_id": pair["pair_id"],
                            "register": register,
                            "seed": seed,
                            "text": text,
                            "sample_rate": int(result["sample_rate"]),
                            "backend": str(result.get("backend", "")),
                            "watermark": bool(result.get("is_watermarked", False)),
                            "style_embedding_sha256": _sha256_array(
                                result["style_embedding"]
                            ),
                            "speaker_embedding_sha256": _sha256_array(
                                result["speaker_embedding"]
                            ),
                            "diffusion_steps": int(args.diffusion_steps),
                            "temperature": float(args.temperature),
                        }
                        _write_sidecar(sidecar_path, sidecar)
                    except Exception as exc:
                        failure = {
                            "audio_id": audio_id,
                            "prompt_pair_id": pair["pair_id"],
                            "register": register,
                            "seed": seed,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        failures.append(failure)
                        rows.append(
                            {
                                "audio_id": audio_id,
                                "prompt_pair_id": pair["pair_id"],
                                "topic": pair["topic"],
                                "register": register,
                                "seed": seed,
                                "text": text,
                                "audio_path": str(audio_path),
                                "status": "failed",
                                "error": failure["error"],
                                **{name: None for name in CLIP_METRICS},
                                "style_embedding_sha256": None,
                                "speaker_embedding_sha256": None,
                                "backend": None,
                                "sample_rate": None,
                                "diffusion_steps": int(args.diffusion_steps),
                                "temperature": float(args.temperature),
                                "watermark": False,
                                "face_speaker_id": face["speaker_id"],
                                "face_clip_id": face["clip_id"],
                            }
                        )
                        continue

                try:
                    metrics = extract_naturalness_metrics(
                        audio_path,
                        text=text,
                        sample_rate=int(sidecar["sample_rate"]),
                    )
                except Exception as exc:
                    failure = {
                        "audio_id": audio_id,
                        "prompt_pair_id": pair["pair_id"],
                        "register": register,
                        "seed": seed,
                        "error": f"metrics {type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    metrics = {name: None for name in CLIP_METRICS}

                rows.append(
                    {
                        "audio_id": audio_id,
                        "prompt_pair_id": pair["pair_id"],
                        "topic": pair["topic"],
                        "register": register,
                        "seed": seed,
                        "text": text,
                        "audio_path": str(audio_path),
                        "status": "ok" if all(
                            value is not None for value in metrics.values()
                        ) else "partial",
                        "error": "",
                        **metrics,
                        "style_embedding_sha256": sidecar.get(
                            "style_embedding_sha256"
                        ),
                        "speaker_embedding_sha256": sidecar.get(
                            "speaker_embedding_sha256"
                        ),
                        "backend": sidecar.get("backend"),
                        "sample_rate": sidecar.get("sample_rate"),
                        "diffusion_steps": sidecar.get(
                            "diffusion_steps", int(args.diffusion_steps)
                        ),
                        "temperature": sidecar.get(
                            "temperature", float(args.temperature)
                        ),
                        "watermark": bool(sidecar.get("watermark", False)),
                        "face_speaker_id": face["speaker_id"],
                        "face_clip_id": face["clip_id"],
                    }
                )

    args.index.parent.mkdir(parents=True, exist_ok=True)
    with args.index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: "" if row.get(column) is None else row.get(column)
                    for column in CSV_COLUMNS
                }
            )
    index_sha256 = sha256_file(args.index)
    print(f"Wrote {args.index} ({len(rows)} rows, {len(failures)} failures)", flush=True)

    usable = [row for row in rows if row.get("status") in {"ok", "partial"}]
    if not usable:
        status = "failed"
    elif failures or any(row.get("status") == "partial" for row in rows):
        status = "partial"
    else:
        status = "completed"

    effects = paired_prompt_effects(
        usable,
        metrics=PROMPT_EFFECT_METRICS,
        n_resamples=int(args.n_resamples),
        seed=int(args.bootstrap_seed),
    )
    summary = summarize_prompt_effects(effects)

    style_pairs: Dict[Tuple[str, int], Dict[str, str]] = {}
    for row in usable:
        if row.get("style_embedding_sha256") is None:
            continue
        key = (str(row["prompt_pair_id"]), int(row["seed"]))
        style_pairs.setdefault(key, {})[str(row["register"])] = str(
            row["style_embedding_sha256"]
        )
    complete_style_cells = [
        key for key, hashes in style_pairs.items() if len(hashes) == 2
    ]
    style_mismatches = [
        {"prompt_pair_id": pair_id, "seed": seed}
        for (pair_id, seed), hashes in sorted(style_pairs.items())
        if len(hashes) == 2 and hashes.get("formal") != hashes.get("conversational")
    ]
    style_consistency = bool(complete_style_cells) and not style_mismatches

    identity: Dict[str, Any]
    if args.no_identity_check:
        identity = {
            "status": "unavailable",
            "reason": "identity check disabled by --no-identity-check",
        }
    else:
        print("Extracting ECAPA prompt-consistency embeddings ...", flush=True)
        identity = _ecapa_prompt_consistency(
            usable,
            device=args.device,
            n_resamples=int(args.n_resamples),
            bootstrap_seed=int(args.bootstrap_seed),
        )

    report_metrics: Dict[str, Any] = {}
    for metric in PROMPT_EFFECT_METRICS:
        entry = effects["metrics"][metric]
        if entry.get("status") != "computed":
            report_metrics[metric] = unavailable_metric(
                str(entry.get("reason", "metric unavailable"))
            )
            continue
        formal_values = [
            float(row[metric])
            for row in usable
            if row.get("register") == "formal" and row.get(metric) is not None
        ]
        conversational_values = [
            float(row[metric])
            for row in usable
            if row.get("register") == "conversational"
            and row.get(metric) is not None
        ]
        report_metrics[metric] = computed_metric(
            {
                "mean_delta": entry["mean_delta"],
                "ci_low": entry["ci_low"],
                "ci_high": entry["ci_high"],
                "n_pairs": entry["n_pairs"],
                "direction": entry["direction"],
            },
            unit=METRIC_UNITS.get(metric, ""),
            formal_mean=float(np.mean(formal_values)) if formal_values else None,
            conversational_mean=(
                float(np.mean(conversational_values))
                if conversational_values
                else None
            ),
            seed_noise_sd=entry["seed_noise_sd"],
            seed_difference_noise_sd=entry["seed_difference_noise_sd"],
            effect_to_noise=entry["effect_to_noise"],
            cohens_dz=entry["cohens_dz"],
            sign_consistency=entry["sign_consistency"],
            detected=entry["detected"],
            robustly_detected=entry["robustly_detected"],
        )

    report_metrics["style_embedding_consistency"] = computed_metric(
        {
            "consistent": bool(style_consistency),
            "n_pair_seed_cells": len(complete_style_cells),
            "n_mismatches": len(style_mismatches),
        },
        note=(
            "The canonical adapter/style path is text-independent; a mismatch "
            "would mean the audit did not hold identity conditioning fixed."
        ),
    )
    if identity.get("status") == "computed":
        report_metrics["ecapa_prompt_consistency"] = computed_metric(
            identity["mean_cosine"],
            ci_low=identity["ci_low"],
            ci_high=identity["ci_high"],
            min_cosine=identity["min_cosine"],
            n_prompt_pairs=identity["n_prompt_pairs"],
            n_pair_seed_observations=identity["n_pair_seed_observations"],
            note=identity["note"],
        )
    else:
        report_metrics["ecapa_prompt_consistency"] = unavailable_metric(
            str(identity.get("reason", "not measured"))
        )

    core_detected = bool(
        "f0_std_st" in summary["metrics_detected"]
        and "words_per_second" in summary["metrics_detected"]
    )
    any_detected = bool(summary["n_metrics_detected"] > 0)
    if core_detected:
        decision = (
            "Prompt register shifts the core prosody pair (F0 variability and "
            "speech rate). N0b should test whether a conversational prosody "
            "prototype adds anything beyond prompt phrasing."
        )
    elif any_detected:
        decision = (
            "Prompt register shifts at least one delivery proxy, but not both "
            "F0 variability and speech rate. Do not assume prompt wording alone "
            "fixes audiobook-like delivery."
        )
    else:
        decision = (
            "No delivery proxy has a paired CI excluding zero across these ten "
            "prompt pairs. Prompt wording alone is not a demonstrated lever in "
            "this setup."
        )

    gates = {
        "n0a_any_prompt_sensitivity": any_detected,
        "n0a_core_prompt_sensitivity": core_detected,
        "n0a_robust_prompt_sensitivity": bool(
            summary["n_metrics_robustly_detected"] > 0
        ),
        "n0a_style_conditioning_consistent": style_consistency,
        "n0a_decision": decision,
    }

    limitations = [
        "N0a is exploratory; no human MOS/CMOS was collected. Perceptual naturalness remains unmeasured.",
        "One real VoxCeleb2 face is used; prompt effects may not generalize to other faces, voices, or accents.",
        "Ten prompt pairs are used. Lexical and phonetic content differs between registers, so the contrast is register phrasing as realized in these pairs, not a word-level causal effect.",
        "Seed noise includes CVAE identity resampling. The effect-to-noise ratio is therefore conservative and descriptive, not a formal power calculation.",
        "Diffusion sampling noise is not exactly controlled across registers; three seeds per cell average it only partially.",
        "AMI is not used. N0b/N2 conversational adaptation remains data-blocked and untrained.",
        "The ECAPA prompt-consistency check is generated-domain only; it is not a real-speaker verification score and does not replace the CFVN identity guardrail.",
        "Watermarking is disabled so the measurement is not contaminated by an audio watermark.",
        "The pilot100 face-feature store is a 382-clip pilot, not the full VoxCeleb2 corpus.",
        "No checkpoint, adapter, CVAE, or CFVN artifact is modified by this run.",
    ]

    artifacts = {
        "clips_csv": str(args.index),
        "clips_csv_sha256": index_sha256,
        "audio_dir": str(args.output_dir),
        "prompt_config": str(args.prompts),
        "prompt_config_sha256": sha256_file(args.prompts),
        "face_features": str(args.features),
        "face_features_sha256": face["features_sha256"],
    }

    report = build_report(
        track="naturalness",
        level="exploratory",
        run_id=str(args.run_id),
        metrics=report_metrics,
        gates=gates,
        status=status,
        limitations=limitations,
        seeds=seeds,
        config_path=args.prompts,
        artifacts=artifacts,
        extra={
            "design": {
                "prompt_pairs": len(pairs),
                "registers": list(PROMPT_REGISTERS),
                "generation_seeds": seeds,
                "diffusion_steps": int(args.diffusion_steps),
                "temperature": float(args.temperature),
                "alpha": float(args.alpha),
                "beta": float(args.beta),
                "watermark": False,
                "device": str(args.device),
                "level_note": "exploratory inference-time audit; no training",
            },
            "face": {
                key: value
                for key, value in face.items()
                if key != "face_features"
            },
            "prompt_pairs": pairs,
            "paired_effects": effects,
            "summary": summary,
            "identity": identity,
            "style_consistency": {
                "consistent": style_consistency,
                "n_pair_seed_cells": len(complete_style_cells),
                "mismatches": style_mismatches,
            },
            "canonical_hashes": canonical,
            "pipeline_provenance": pipeline.provenance,
            "failures": failures,
        },
    )
    write_report(args.report, report)
    print(f"Wrote {args.report}", flush=True)
    print(
        f"Detected: {summary['metrics_detected']} | "
        f"robust: {summary['metrics_robustly_detected']} | "
        f"status={status}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
