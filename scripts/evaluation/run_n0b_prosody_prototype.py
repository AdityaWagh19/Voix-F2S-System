"""N0b: inference-time AMI prosody prototype.

The frozen StyleTTS2 style space is split into a timbre half (dims 0:128) and a
prosody half (dims 128:256).  We build two usable prototypes from real AMI
training clips:

    mean:             clip-weighted mean style[128:] across training clips
    speaker_balanced: mean of each training speaker's mean style[128:], with
                      equal weight per speaker

A third construction, ``speaker_centered_diagnostic``, subtracts each speaker's
mean and then averages.  That is mathematically zero (each speaker's centered
rows sum to zero), so it is reported only as a diagnostic and is not used as a
condition.  The earlier draft called this the speaker-centered prototype; that
name and condition were retired after the zero-norm check.

For a fixed real face and a fixed canonical style vector, the timbre half is
kept unchanged and the prosody half is blended:

    prosody = (1 - alpha) * base_prosody + alpha * prototype

with alpha in {0, 0.25, 0.5}.  Generation uses held-out AMI transcripts from
speaker-disjoint validation and test meetings.  Clips shorter than 2.5 s are
deprioritised so F0, pause, and energy metrics are stable; alpha=0 is the
paired baseline.

This is an inference-time experiment.  It does not train or modify any
checkpoint, does not touch the canonical artifacts, and does not claim MOS.
Confidence intervals are speaker-cluster bootstraps, not row-level intervals.

Audio generation and acoustic measurement are separate phases: the TTS model
is released before librosa/pyin runs, so the 8 GB host does not hold synthesis
and metric-extraction memory at the same time.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.identity_guardrail import verify_canonical_hashes  # noqa: E402
from voix.evaluation.naturalness import (  # noqa: E402
    extract_naturalness_metrics,
)
from voix.evaluation.research_report import (  # noqa: E402
    build_report,
    computed_metric,
    sha256_file,
    unavailable_metric,
    write_report,
)
from voix.utils.paths import project_path  # noqa: E402

DEFAULT_CACHE = project_path("data", "processed", "ami_style_cache.npz")
DEFAULT_MANIFEST = project_path("data", "processed", "ami_naturalness", "manifest.csv")
DEFAULT_OUTPUT_DIR = project_path("results", "naturalness", "n0b", "audio")
DEFAULT_INDEX = project_path("results", "naturalness", "n0b", "clips.csv")
DEFAULT_REPORT = project_path("results", "naturalness", "n0b", "report.json")
DEFAULT_FEATURES = project_path(
    "data", "processed", "voxceleb2_v2", "pilot100_features.h5"
)
ALPHAS = (0.0, 0.25, 0.5)
VARIANTS = ("mean", "speaker_balanced")
EVAL_METRICS = (
    "f0_median_hz",
    "f0_range_st",
    "words_per_second",
    "energy_std_db",
    "pause_ratio",
    "spectral_tilt_db_per_khz",
)
# Median F0 is speaker-dependent and the face donor is not an AMI speaker.
# Keep it as a diagnostic, but do not let a gender/register mismatch satisfy
# the naturalness gate.
GATE_METRICS = tuple(metric for metric in EVAL_METRICS if metric != "f0_median_hz")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the N0b prosody prototype")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-clips-per-split", type=int, default=8)
    parser.add_argument(
        "--min-clip-duration",
        type=float,
        default=2.5,
        help=(
            "Prefer held-out clips at least this long so F0, pause, and energy "
            "metrics are stable; shorter clips are used only as a fallback."
        ),
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--alpha", type=float, default=0.18)
    parser.add_argument("--beta", type=float, default=0.22)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-wer", action="store_true")
    parser.add_argument("--wer-model", default="base")
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-index",
        action="store_true",
        help=(
            "Reuse an existing clips CSV with all generated-audio metrics "
            "instead of regenerating audio and recomputing proxies. The index "
            "is validated against the current clip/variant/alpha/seed grid."
        ),
    )
    return parser.parse_args(argv)


def _sha256_array(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity for ECAPA vectors, which are not unit-normalized."""
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _load_cache(path: Path) -> Dict[str, np.ndarray]:
    cache = np.load(path, allow_pickle=True)
    return {key: cache[key] for key in cache.files}


def _build_prototypes(cache: Mapping[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
    styles = np.asarray(cache["style"], dtype=np.float32)
    splits = cache["split"].astype(str)
    speakers = cache["speaker_id"].astype(str)
    train_mask = splits == "train"
    if train_mask.sum() < 2:
        raise RuntimeError("need at least two AMI training styles")
    train_styles = styles[train_mask]
    train_speakers = speakers[train_mask]

    mean_prosody = train_styles[:, 128:].mean(axis=0)

    speaker_means: List[np.ndarray] = []
    centered_rows: List[np.ndarray] = []
    for speaker in sorted(set(train_speakers)):
        group = train_styles[train_speakers == speaker]
        speaker_mean = group.mean(axis=0, keepdims=True)
        speaker_means.append(group[:, 128:].mean(axis=0))
        centered_rows.append((group - speaker_mean)[:, 128:])
    balanced_prosody = np.mean(speaker_means, axis=0)
    centered = np.concatenate(centered_rows, axis=0)
    centered_prosody = centered.mean(axis=0)

    return {
        "mean": {
            "prosody": mean_prosody.astype(np.float32),
            "n_clips": int(train_mask.sum()),
            "n_speakers": int(len(set(train_speakers))),
            "speaker_balanced": False,
        },
        "speaker_balanced": {
            "prosody": balanced_prosody.astype(np.float32),
            "n_clips": int(train_mask.sum()),
            "n_speakers": int(len(set(train_speakers))),
            "speaker_balanced": True,
        },
        "speaker_centered_diagnostic": {
            "prosody": centered_prosody.astype(np.float32),
            "n_clips": int(train_mask.sum()),
            "n_speakers": int(len(set(train_speakers))),
            "speaker_balanced": False,
            "note": (
                "Subtracting each speaker's mean and then averaging is "
                "mathematically zero; reported only to document why this "
                "construction is not a usable prototype."
            ),
        },
    }


def _real_metrics_by_clip(
    cache: Mapping[str, np.ndarray],
    *,
    metrics: Sequence[str] = EVAL_METRICS,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Map each cached clip to its real AMI delivery metrics.

    The cache stores parallel arrays; keep the lookup explicit so a missing or
    non-finite value becomes ``None`` rather than silently pairing the wrong
    clip with the wrong metric.
    """
    clip_ids = [str(item) for item in cache["clip_id"]]
    arrays = {
        metric: np.asarray(cache[metric], dtype=np.float64)
        for metric in metrics
        if metric in cache
    }
    result: Dict[str, Dict[str, Optional[float]]] = {}
    for index, clip_id in enumerate(clip_ids):
        result[clip_id] = {
            metric: (
                float(arrays[metric][index])
                if metric in arrays
                and index < len(arrays[metric])
                and np.isfinite(arrays[metric][index])
                else None
            )
            for metric in metrics
        }
    return result


def _gate_improved_metrics(
    toward_ami: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
    variant: str,
) -> List[str]:
    """Distinct gate metrics that improved, excluding median-F0 diagnostics."""
    return sorted(
        {
            metric
            for block in toward_ami.get(variant, {}).values()
            for metric, result in block.items()
            if metric in GATE_METRICS and result.get("improved")
        }
    )


def _paired_difference_rows(
    subset: pd.DataFrame,
    *,
    alpha: float,
    metric: str,
    target: Optional[float] = None,
) -> pd.DataFrame:
    """Build one paired difference per clip/seed for a metric.

    ``target`` switches from the raw alpha-minus-base difference to the
    absolute-error difference ``|alpha - target| - |base - target|`` used by
    the move-toward-AMI test.  The speaker id comes from the row, never from
    the metric Series, so a scalar is not indexed as if it were a DataFrame.
    """
    rows: List[Dict[str, Any]] = []
    for _, group in subset.groupby(["ami_clip_id", "seed"]):
        base_rows = group[np.isclose(group["alpha"], 0.0)]
        changed_rows = group[np.isclose(group["alpha"], alpha)]
        if not len(base_rows) or not len(changed_rows):
            continue
        base_value = float(base_rows.iloc[0][metric])
        alpha_value = float(changed_rows.iloc[0][metric])
        if not (np.isfinite(base_value) and np.isfinite(alpha_value)):
            continue
        difference = alpha_value - base_value
        if target is not None:
            difference = abs(alpha_value - target) - abs(base_value - target)
        rows.append(
            {
                "speaker_id": str(base_rows.iloc[0]["speaker_id"]),
                "difference": difference,
            }
        )
    return pd.DataFrame(rows)


def _cluster_paired_interval(
    pairs: pd.DataFrame,
    *,
    n_resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> Optional[Dict[str, Any]]:
    """Speaker-cluster bootstrap for a paired mean difference.

    Each speaker contributes one mean difference; resampling speakers rather
    than clip/seed rows avoids treating repeated generations of the same AMI
    clip as independent observations.
    """
    if pairs.empty or "speaker_id" not in pairs or "difference" not in pairs:
        return None
    per_speaker = pairs.groupby("speaker_id")["difference"].mean()
    if per_speaker.size < 2:
        return None
    values = per_speaker.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(
        means,
        [(1.0 - confidence) * 50.0, (1.0 + confidence) * 50.0],
    )
    return {
        "mean_change": float(values.mean()),
        "mean_pair_change": float(pairs["difference"].mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_pairs": int(len(pairs)),
        "n_speakers": int(per_speaker.size),
        "n_resamples": int(n_resamples),
        "confidence": float(confidence),
    }


def _select_eval_clips(
    manifest: pd.DataFrame,
    max_per_split: int,
    *,
    available_clip_ids: Optional[set[str]] = None,
    min_duration: float = 0.0,
) -> pd.DataFrame:
    """Select held-out clips with enough audio for stable acoustic metrics.

    Selection uses only clip duration and the existence of a real style-cache
    row; it never looks at generated audio or metric outcomes.  The first
    clips above ``min_duration`` are used, with the longest remaining clips as
    a fallback so every speaker still contributes when possible.
    """
    rows = []
    for split in ("val", "test"):
        subset = manifest[manifest["split"].astype(str) == split].copy()
        if available_clip_ids is not None:
            subset = subset[
                subset["clip_id"].astype(str).isin(available_clip_ids)
            ].copy()
        if subset.empty:
            continue
        per_speaker = max(1, int(max_per_split) // max(1, subset["speaker_id"].nunique()))
        for _, group in subset.groupby("speaker_id", sort=True):
            group = group.sort_values("clip_id")
            preferred = group[
                group["duration_sec"].astype(float) >= float(min_duration)
            ].head(per_speaker)
            if len(preferred) < per_speaker:
                remaining = group[
                    ~group["clip_id"].astype(str).isin(
                        preferred["clip_id"].astype(str)
                    )
                ]
                extra = remaining.sort_values(
                    ["duration_sec", "clip_id"], ascending=[False, True]
                ).head(per_speaker - len(preferred))
                preferred = pd.concat([preferred, extra])
            rows.append(preferred)
    if not rows:
        raise RuntimeError("no AMI validation/test clips available")
    return pd.concat(rows, ignore_index=True).sort_values("clip_id").reset_index(drop=True)


def _load_reusable_index(
    path: Path,
    eval_clips: pd.DataFrame,
    *,
    seeds: Sequence[int],
    expected_rows: int,
) -> pd.DataFrame:
    """Load and validate a previously written clips CSV.

    Reuse is only allowed when the file covers exactly the current held-out
    clip / variant / alpha / seed grid and every referenced WAV exists.  This
    keeps a rerun from silently mixing a stale index with a new selection.
    """
    if not path.is_file():
        raise FileNotFoundError(f"reusable index not found: {path}")
    frame = pd.read_csv(path)
    required = {
        "ami_clip_id",
        "variant",
        "alpha",
        "seed",
        "speaker_id",
        "wav_path",
        *EVAL_METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"reusable index is missing columns: {missing}")
    if len(frame) != int(expected_rows):
        raise ValueError(
            f"reusable index has {len(frame)} rows, expected {expected_rows}"
        )
    expected = {
        (str(clip_id), variant, float(alpha), int(seed))
        for clip_id in eval_clips["clip_id"].astype(str)
        for variant in VARIANTS
        for alpha in ALPHAS
        for seed in seeds
    }
    actual = {
        (
            str(row.ami_clip_id),
            str(row.variant),
            float(row.alpha),
            int(row.seed),
        )
        for row in frame.itertuples()
    }
    if actual != expected:
        raise ValueError(
            "reusable index does not match the current clip/variant/alpha/seed grid"
        )
    missing_audio = [
        str(path) for path in frame["wav_path"].astype(str) if not Path(path).is_file()
    ]
    if missing_audio:
        raise FileNotFoundError(
            f"reusable index references missing audio: {missing_audio[:5]}"
        )
    return frame


def _load_tts_pipeline(device: str):
    from voix.inference.pipeline import VoixPipeline

    pipeline = VoixPipeline(
        device=device,
        enable_watermarking=False,
        tts_use_neural=True,
        tts_allow_fallback=False,
    )
    return pipeline


def _release_tts_model() -> None:
    """Drop the process-wide StyleTTS2 model and collect before ECAPA/metrics.

    ``StyleTTS2Backend`` caches the loaded model in a class attribute, so
    deleting the pipeline alone leaves the synthesis model resident.  On the
    8 GB host that is enough to make the later librosa/ECAPA stages fail.
    """
    import gc

    try:
        from voix.models.tts_backend import StyleTTS2Backend

        StyleTTS2Backend._SHARED_MODEL = None
    except Exception:  # pragma: no cover - environment-dependent
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - environment-dependent
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    seeds = [int(item) for item in str(args.seeds).split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    canonical = verify_canonical_hashes()
    if not canonical.get("passed"):
        raise RuntimeError("canonical checkpoint hashes do not match; refusing N0b")

    cache = _load_cache(args.cache)
    prototypes = _build_prototypes(cache)
    manifest = pd.read_csv(args.manifest)
    cache_clip_ids = {str(item) for item in cache["clip_id"]}
    eval_clips = _select_eval_clips(
        manifest,
        args.max_clips_per_split,
        available_clip_ids=cache_clip_ids,
        min_duration=float(args.min_clip_duration),
    )
    print(
        f"Evaluating on {len(eval_clips)} held-out AMI clip(s) from "
        f"{eval_clips['speaker_id'].nunique()} speaker(s); "
        f"median duration {eval_clips['duration_sec'].median():.2f}s",
        flush=True,
    )

    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    from run_n0a_prompt_audit import select_face_clip

    face = select_face_clip(
        args.features,
        split="val",
        speaker_id=None,
        clip_id=None,
    )
    print(
        f"Face: {face['speaker_id']} / {face['clip_id']} "
        f"(gender={face['face_gender']}, age={face['face_age']:.1f})",
        flush=True,
    )

    pipeline = _load_tts_pipeline(args.device)
    pipeline_provenance = dict(pipeline.provenance)
    pipeline._inf_cfg = {
        **pipeline._inf_cfg,
        "diffusion_steps": int(args.diffusion_steps),
        "alpha": float(args.alpha),
        "beta": float(args.beta),
        "duration_scale": 1.0,
    }

    # One fixed canonical style vector.  It is sampled once from the canonical
    # face-to-voice path and then held fixed across every alpha/transcript/seed,
    # so the only experimental variable is the prosody blend.
    import torch

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    base_results = pipeline.generate(
        face_input=face["face_features"],
        text="This is a fixed style reference for the prosody prototype.",
        num_voices=1,
        temperature=float(args.temperature),
        watermark=False,
        gender=int(face["face_gender"]),
        age=float(face["face_age"]),
        seed=0,
        selection=None,
    )
    if len(base_results) != 1:
        raise RuntimeError("expected exactly one base style candidate")
    base_style = base_results[0]["style_embedding"].detach().cpu().float().reshape(-1)
    if base_style.numel() != 256:
        raise RuntimeError(f"expected 256-D style, got {base_style.numel()}")
    print(f"Base style norm: {base_style.norm().item():.4f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    started = time.time()
    total = len(eval_clips) * len(VARIANTS) * len(ALPHAS) * len(seeds)
    done = 0
    reuse = bool(args.reuse_index)
    frame: Optional[pd.DataFrame] = (
        _load_reusable_index(
            args.index, eval_clips, seeds=seeds, expected_rows=total
        )
        if reuse
        else None
    )
    if reuse and frame is not None:
        done = len(frame)
        print(f"Reusing {done} validated clip rows from {args.index}", flush=True)

    # Phase 1: generate any missing audio and release the TTS model before
    # metric extraction.  On this 8 GB host, holding StyleTTS2 while librosa
    # pyin runs over hundreds of clips caused the process to be killed.
    if not reuse:
        missing = 0
        for _, clip in eval_clips.iterrows():
            text = str(clip["text"])
            for variant in VARIANTS:
                prototype = torch.from_numpy(prototypes[variant]["prosody"]).float()
                for alpha in ALPHAS:
                    blended = base_style.clone()
                    if alpha > 0.0:
                        blended[128:] = (
                            (1.0 - float(alpha)) * base_style[128:]
                            + float(alpha) * prototype
                        )
                    for seed in seeds:
                        audio_id = (
                            f"n0b_{clip['clip_id']}_{variant}_a{alpha:.2f}_s{int(seed)}"
                        )
                        audio_path = args.output_dir / f"{audio_id}.wav"
                        if audio_path.is_file() and not args.overwrite:
                            continue
                        try:
                            torch.manual_seed(int(seed))
                            if torch.cuda.is_available():
                                torch.cuda.manual_seed_all(int(seed))
                            np.random.seed(int(seed))
                            with torch.no_grad():
                                waveform = pipeline.tts.synthesize(
                                    text=text,
                                    style=blended,
                                    gender=int(face["face_gender"]),
                                    age=float(face["face_age"]),
                                    alpha=float(args.alpha),
                                    beta=float(args.beta),
                                    duration_scale=1.0,
                                    diffusion_steps=int(args.diffusion_steps),
                                    f0_scale=1.0,
                                )
                            audio = waveform.detach().cpu().numpy().reshape(-1)
                            sf.write(
                                str(audio_path), audio, 24000, subtype="PCM_16"
                            )
                        except Exception as exc:  # pragma: no cover - runtime guard
                            failures.append(
                                {
                                    "audio_id": audio_id,
                                    "clip_id": str(clip["clip_id"]),
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            continue
                        missing += 1
                        if missing % 3 == 0:
                            elapsed = time.time() - started
                            print(
                                f"  generated {missing} new clip(s) "
                                f"({elapsed:.0f}s, failures={len(failures)})",
                                flush=True,
                            )

        _release_tts_model()
        del pipeline
        pipeline = None
        print("Released TTS pipeline before metric extraction", flush=True)

        # Phase 2: measure the complete audio grid without the TTS model
        # resident.  The WAVs are the source of truth; a missing or corrupt
        # file is recorded as a failure rather than silently dropped.
        for _, clip in eval_clips.iterrows():
            text = str(clip["text"])
            for variant in VARIANTS:
                prototype = torch.from_numpy(prototypes[variant]["prosody"]).float()
                for alpha in ALPHAS:
                    blended = base_style.clone()
                    if alpha > 0.0:
                        blended[128:] = (
                            (1.0 - float(alpha)) * base_style[128:]
                            + float(alpha) * prototype
                        )
                    for seed in seeds:
                        audio_id = (
                            f"n0b_{clip['clip_id']}_{variant}_a{alpha:.2f}_s{int(seed)}"
                        )
                        audio_path = args.output_dir / f"{audio_id}.wav"
                        if not audio_path.is_file():
                            failures.append(
                                {
                                    "audio_id": audio_id,
                                    "clip_id": str(clip["clip_id"]),
                                    "error": "missing generated audio",
                                }
                            )
                            done += 1
                            continue
                        try:
                            info = sf.info(str(audio_path))
                            duration = float(info.duration)
                            metrics = extract_naturalness_metrics(
                                str(audio_path), text=text, sample_rate=24000
                            )
                        except Exception as exc:  # pragma: no cover - runtime guard
                            failures.append(
                                {
                                    "audio_id": audio_id,
                                    "clip_id": str(clip["clip_id"]),
                                    "error": f"metrics {type(exc).__name__}: {exc}",
                                }
                            )
                            if len(failures) <= 8:
                                print(
                                    f"  metric failure {audio_id}: "
                                    f"{type(exc).__name__}: {exc}",
                                    flush=True,
                                )
                            done += 1
                            continue
                        rows.append(
                            {
                                "audio_id": audio_id,
                                "ami_clip_id": str(clip["clip_id"]),
                                "meeting_id": str(clip["meeting_id"]),
                                "speaker_id": str(clip["speaker_id"]),
                                "split": str(clip["split"]),
                                "text": text,
                                "variant": variant,
                                "alpha": float(alpha),
                                "seed": int(seed),
                                "wav_path": str(audio_path),
                                "duration_s": duration,
                                "base_style_sha256": _sha256_array(base_style),
                                "blended_style_sha256": _sha256_array(blended),
                                "prototype_norm": float(prototype.norm().item()),
                                "prototype_sha256": _sha256_array(prototype),
                                **metrics,
                            }
                        )
                        done += 1
                    elapsed = time.time() - started
                    print(
                        f"  measured {done}/{total} "
                        f"({done / max(elapsed, 1e-6):.2f}/s, "
                        f"{elapsed:.0f}s, failures={len(failures)})",
                        flush=True,
                    )

    if frame is None:
        frame = pd.DataFrame(rows)
        args.index.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.index, index=False)

    # Paired comparisons against alpha=0 for the same variant/transcript/seed.
    comparisons: Dict[str, Any] = {}
    for variant in VARIANTS:
        subset = frame[frame["variant"] == variant]
        if subset.empty:
            continue
        variant_block: Dict[str, Any] = {}
        for alpha in ALPHAS:
            if alpha == 0.0:
                continue
            alpha_block: Dict[str, Any] = {}
            for metric in EVAL_METRICS:
                interval = _cluster_paired_interval(
                    _paired_difference_rows(
                        subset, alpha=alpha, metric=metric
                    ),
                    n_resamples=args.n_resamples,
                    seed=args.seed,
                )
                alpha_block[metric] = (
                    interval
                    if interval is not None
                    else {
                        "status": "unavailable",
                        "reason": "fewer than two speakers with finite pairs",
                    }
                )
            variant_block[f"alpha_{alpha:.2f}"] = alpha_block
        comparisons[variant] = variant_block

    # Move-toward-AMI test: absolute error to the matched real AMI clip.
    # The real metrics live in the style cache, not in the corpus manifest.
    real_metrics = _real_metrics_by_clip(cache)
    toward_ami: Dict[str, Any] = {}
    for variant in VARIANTS:
        subset = frame[frame["variant"] == variant]
        if subset.empty:
            continue
        variant_block: Dict[str, Any] = {}
        for alpha in ALPHAS:
            if alpha == 0.0:
                continue
            alpha_block: Dict[str, Any] = {}
            for metric in EVAL_METRICS:
                pair_frames: List[pd.DataFrame] = []
                for clip_id, group in subset.groupby("ami_clip_id"):
                    target = real_metrics.get(str(clip_id), {}).get(metric)
                    if target is None:
                        continue
                    pair_frames.append(
                        _paired_difference_rows(
                            group,
                            alpha=alpha,
                            metric=metric,
                            target=target,
                        )
                    )
                interval = _cluster_paired_interval(
                    pd.concat(pair_frames, ignore_index=True)
                    if pair_frames
                    else pd.DataFrame(),
                    n_resamples=args.n_resamples,
                    seed=args.seed,
                )
                if interval is not None:
                    alpha_block[metric] = {
                        "mean_abs_error_change": interval["mean_change"],
                        "mean_pair_abs_error_change": interval["mean_pair_change"],
                        "ci_low": interval["ci_low"],
                        "ci_high": interval["ci_high"],
                        "n_pairs": interval["n_pairs"],
                        "n_speakers": interval["n_speakers"],
                        "improved": bool(interval["ci_high"] < 0.0),
                    }
                else:
                    alpha_block[metric] = {
                        "status": "unavailable",
                        "reason": "fewer than two speakers with finite pairs",
                    }
            variant_block[f"alpha_{alpha:.2f}"] = alpha_block
        toward_ami[variant] = variant_block

    # Generated-domain identity consistency: alpha condition vs alpha 0.
    # Free the TTS pipeline first so ECAPA fits on the 4 GB card.
    _release_tts_model()
    try:
        del pipeline
    except UnboundLocalError:  # pragma: no cover - already released above
        pass
    identity: Dict[str, Any] = {}
    try:
        from voix.data.speaker_extractor import ECAPAExtractor

        extractor = ECAPAExtractor(device=args.device)
        embeddings = {
            str(row["audio_id"]): extractor.extract_from_path(str(row["wav_path"]))
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            for _, row in frame.iterrows()
        }
        for variant in VARIANTS:
            subset = frame[frame["variant"] == variant]
            variant_block: Dict[str, Any] = {}
            for alpha in ALPHAS:
                if alpha == 0.0:
                    continue
                values: List[float] = []
                for (clip_id, seed), group in subset.groupby(["ami_clip_id", "seed"]):
                    base = group[np.isclose(group["alpha"], 0.0)]
                    changed = group[np.isclose(group["alpha"], alpha)]
                    if len(base) and len(changed):
                        left = embeddings[str(base.iloc[0]["audio_id"])]
                        right = embeddings[str(changed.iloc[0]["audio_id"])]
                        values.append(_cosine(left, right))
                variant_block[f"alpha_{alpha:.2f}"] = {
                    "n_pairs": len(values),
                    "mean_cosine_to_alpha0": float(np.mean(values)) if values else None,
                    "min_cosine_to_alpha0": float(np.min(values)) if values else None,
                }
            identity[variant] = variant_block
    except Exception as exc:  # pragma: no cover - environment-dependent
        identity = {
            "status": "unavailable",
            "reason": f"ECAPA unavailable: {type(exc).__name__}: {exc}",
        }

    # Free ECAPA before Whisper so the 4 GB card has room for the ASR model.
    try:
        import torch as _torch

        del extractor
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:
        pass

    # WER is optional and measured per variant/alpha.
    wer: Dict[str, Any] = {
        "status": "unavailable",
        "reason": "WER was not requested; pass --compute-wer",
    }
    if args.compute_wer:
        from voix.evaluation.audio_metrics import transcribe_wer

        wer_blocks: Dict[str, Any] = {}
        for (variant, alpha), group in frame.groupby(["variant", "alpha"], sort=True):
            group = group.sort_values(["ami_clip_id", "seed"])
            outcome = transcribe_wer(
                group["wav_path"].astype(str).tolist(),
                group["text"].astype(str).tolist(),
                model_size=args.wer_model,
                device=args.device,
            )
            wer_blocks[f"{variant}:alpha_{alpha:.2f}"] = outcome
        wer = {"status": "computed", "conditions": wer_blocks}

    # Gate: at least two dimensions move toward held-out AMI with CI excluding
    # zero, no WER increase above +0.05, and generated-domain identity does not
    # drop by more than 0.05 relative to alpha 0.
    gate_by_variant: Dict[str, Any] = {}
    for variant in VARIANTS:
        # Count distinct prosody dimensions, not alpha-by-metric pairs.  One
        # metric improving at two blend strengths is still one dimension.
        improved_metrics = _gate_improved_metrics(toward_ami, variant)
        improvements = [
            f"{alpha_key}:{metric}"
            for alpha_key, block in toward_ami.get(variant, {}).items()
            for metric, result in block.items()
            if metric in GATE_METRICS and result.get("improved")
        ]
        identity_ok = False
        identity_measured = isinstance(identity, dict) and "status" not in identity
        if identity_measured:
            variant_identity = identity.get(variant, {})
            identity_ok = any(
                block.get("n_pairs", 0) > 0
                for block in variant_identity.values()
            )
            if identity_ok:
                for alpha_key, block in variant_identity.items():
                    value = block.get("mean_cosine_to_alpha0")
                    if value is None or value < 0.95:
                        identity_ok = False
        wer_ok = False
        if wer.get("status") == "computed":
            conditions = wer.get("conditions", {})
            base_key = f"{variant}:alpha_0.00"
            base_wer = conditions.get(base_key, {}).get("wer")
            if base_wer is not None:
                wer_ok = True
                for key, outcome in conditions.items():
                    if not key.startswith(variant + ":") or key == base_key:
                        continue
                    value = outcome.get("wer")
                    if value is None or float(value) > float(base_wer) + 0.05:
                        wer_ok = False
        gate_by_variant[variant] = {
            "status": "computed",
            "toward_ami_improvements": improvements,
            "improved_metrics": improved_metrics,
            "n_improved_metrics": len(improved_metrics),
            "identity_measured": bool(identity_measured),
            "identity_ok": identity_ok,
            "wer_ok": wer_ok,
            "passed": bool(
                len(improved_metrics) >= 2 and identity_ok and wer_ok
            ),
        }

    metrics: Dict[str, Any] = {
        "prototypes": computed_metric(
            {
                name: {
                    "norm": float(np.linalg.norm(block["prosody"])),
                    "n_clips": block["n_clips"],
                    "n_speakers": block["n_speakers"],
                    "speaker_balanced": block.get("speaker_balanced", False),
                    "note": block.get("note"),
                    "sha256": _sha256_array(block["prosody"]),
                }
                for name, block in prototypes.items()
            }
        ),
        "paired_change_vs_alpha0": computed_metric(comparisons),
        "toward_ami_absolute_error": computed_metric(toward_ami),
        "identity_generated_consistency": (
            computed_metric(identity)
            if isinstance(identity, dict) and "status" not in identity
            else unavailable_metric(identity.get("reason", "identity unavailable"))
        ),
        "wer": (
            computed_metric(wer)
            if wer.get("status") == "computed"
            else unavailable_metric(wer.get("reason", "WER unavailable"))
        ),
        "clip_count": computed_metric(int(len(frame))),
    }

    report = build_report(
        track="naturalness",
        level="exploratory",
        run_id="n0b_prosody_prototype",
        status="completed" if not failures else "partial",
        metrics=metrics,
        gates={"n0b": gate_by_variant},
        limitations=[
            "N0b is an inference-time prototype, not a trained naturalness model.",
            "AMI audio is 16 kHz headset speech; generated audio is 24 kHz.",
            "The face donor and AMI speakers are different people, so absolute F0 comparisons are confounded; median F0 is reported as a diagnostic but excluded from the gate.",
            "Confidence intervals use a speaker-cluster bootstrap over the held-out AMI speakers; repeated clip/seed generations are not treated as independent.",
            "No human MOS/CMOS was collected; perceptual naturalness remains unmeasured.",
            "Irish and accent/emotion artifacts are not used here.",
            "The speaker_balanced prototype is a robustness check, not independent evidence: on this pilot it is nearly collinear with the clip-weighted mean.",
            "The retired speaker_centered construction is mathematically zero and is retained only as a diagnostic, not as a condition.",
            "Generated pause ratios are near zero; a small absolute move toward AMI can pass the gate without closing the distribution gap.",
        ],
        mos={
            "status": "unavailable",
            "reason": "no human ratings were collected for this run",
        },
        seeds=seeds,
        manifest_path=args.manifest,
        per_class={
            "by_variant": {
                variant: {
                    "alpha0_mean": {
                        metric: float(
                            frame[
                                (frame["variant"] == variant)
                                & np.isclose(frame["alpha"], 0.0)
                            ][metric].mean()
                        )
                        for metric in EVAL_METRICS
                    }
                }
                for variant in VARIANTS
            }
        },
        artifacts={
            "clips_csv": str(args.index),
            "clips_csv_sha256": sha256_file(args.index),
            "audio_dir": str(args.output_dir),
            "style_cache": str(args.cache),
            "style_cache_sha256": sha256_file(args.cache),
        },
        extra={
            "experiment": "N0b",
            "face": {
                "speaker_id": str(face["speaker_id"]),
                "clip_id": str(face["clip_id"]),
                "gender": int(face["face_gender"]),
                "age": float(face["face_age"]),
            },
            "base_style_sha256": _sha256_array(base_style),
            "base_style_norm": float(base_style.norm().item()),
            "alphas": list(ALPHAS),
            "variants": list(VARIANTS),
            "failures": failures,
            "canonical_hashes": canonical,
            "pipeline_provenance": pipeline_provenance,
        },
    )
    write_report(args.report, report)
    print(f"Wrote {args.index} ({len(frame)} rows)")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
