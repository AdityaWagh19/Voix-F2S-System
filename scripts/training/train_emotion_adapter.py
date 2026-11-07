"""Train the E2 learned emotion residual on real CREMA-D style targets.

The adapter is a small research model, not a serving checkpoint.  It is trained
on paired ``style(emotion) - style(neutral)`` targets from the 73 CREMA-D
training actors, with speaker-disjoint early stopping inside that split.  The 9
validation actors are never used for fitting or early stopping; the 9 test
actors remain untouched until E3.

A label-shuffled control is trained with the same architecture and schedule.
It permutes emotion labels and residual targets together, so any apparent
emotion structure in the control comes from capacity or leakage rather than the
class labels.

Examples::

    python scripts/training/train_emotion_adapter.py
    python scripts/training/train_emotion_adapter.py --device cuda --seed 0
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

DEFAULT_MANIFEST = ROOT / "data" / "processed" / "crema_d_manifest.csv"
DEFAULT_CACHE = ROOT / "data" / "processed" / "crema_d_style_cache_train.npz"
DEFAULT_ARTIFACT = (
    ROOT / "checkpoints" / "emotion_adapter" / "e2_adapter_seed0.pt"
)
DEFAULT_CONTROL_ARTIFACT = (
    ROOT / "checkpoints" / "emotion_adapter" / "e2_adapter_label_shuffled_seed0.pt"
)
DEFAULT_REPORT = ROOT / "results" / "emotion" / "e2" / "train_report.json"
DEFAULT_CLASS_MEAN = ROOT / "checkpoints" / "emotion_residuals" / "class_mean.pt"

REPORT_SCHEMA = "voix.emotion_adapter_training.v1"
STYLE_DIM = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _load_cache(path: Path) -> Dict[str, np.ndarray]:
    cache = np.load(path, allow_pickle=True)
    required = {"paths", "styles"}
    missing = required - set(cache.files)
    if missing:
        raise ValueError(f"style cache is missing arrays: {sorted(missing)}")
    paths = np.asarray(cache["paths"]).astype(str)
    styles = np.asarray(cache["styles"], dtype=np.float32)
    if styles.ndim != 2 or styles.shape != (paths.size, STYLE_DIM):
        raise ValueError(
            f"style cache must be (N, {STYLE_DIM}), got {styles.shape}"
        )
    if not np.all(np.isfinite(styles)):
        raise ValueError("style cache contains non-finite values")
    return {"paths": paths, "styles": styles}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the E2 learned emotion residual adapter"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--control-artifact", type=Path, default=DEFAULT_CONTROL_ARTIFACT
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--class-mean", type=Path, default=DEFAULT_CLASS_MEAN)
    parser.add_argument("--split", default="train")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-norm", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--timbre-weight", type=float, default=0.25)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--run-id", default="e2_adapter_seed0")
    return parser.parse_args(argv)


def _neutral_zero_check(
    model: Any,
    payload: Dict[str, Any],
    sample_style: Any,
    *,
    emotions: Sequence[str],
) -> Dict[str, Any]:
    import torch

    from voix.models.emotion_adapter import (
        apply_learned_residual,
        predict_residual,
    )

    style = torch.as_tensor(sample_style, dtype=torch.float32).reshape(-1)
    if style.numel() != STYLE_DIM:
        raise ValueError(f"sample style must be {STYLE_DIM}-D")
    neutral = predict_residual(model, payload, style, "neutral", intensity=1.0)
    zero_intensity = {
        emotion: predict_residual(model, payload, style, emotion, intensity=0.0)
        for emotion in emotions
    }
    max_abs_error = float(neutral.abs().max().item())
    for delta in zero_intensity.values():
        max_abs_error = max(max_abs_error, float(delta.abs().max().item()))
    applied = apply_learned_residual(style, torch.zeros_like(style))
    identity_preserved = applied is style
    return {
        "max_abs_error": max_abs_error,
        "tolerance": 1e-7,
        "passed": bool(max_abs_error <= 1e-7 and identity_preserved),
        "neutral_zero_exact": bool(neutral.abs().sum().item() == 0.0),
        "zero_intensity_exact": bool(
            all(delta.abs().sum().item() == 0.0 for delta in zero_intensity.values())
        ),
        "apply_zero_returns_same_object": bool(identity_preserved),
    }


def _training_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "best_epoch": int(result["best_epoch"]),
        "best_val_loss": float(result["best_val_loss"]),
        "val_metrics": result["val_metrics"],
        "n_fit": int(result["n_fit"]),
        "n_val": int(result["n_val"]),
        "fit_speakers": result["fit_speakers"],
        "val_speakers": result["val_speakers"],
        "n_parameters": int(result["n_parameters"]),
        "config": result["config"],
        "history_tail": result["history"][-5:],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    import torch

    if args.threads > 0:
        torch.set_num_threads(int(args.threads))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda was requested but CUDA is unavailable")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"CREMA-D manifest not found: {args.manifest}")
    if not args.cache.is_file():
        raise FileNotFoundError(f"style cache not found: {args.cache}")

    from voix.models.emotion_adapter import (
        EMOTIONS,
        NON_NEUTRAL_EMOTIONS,
        build_examples,
        save_adapter,
        shuffle_labels,
        train_adapter,
    )
    from voix.models.emotion_residual import build_pairs, select_one_per_group

    frame = pd.read_csv(args.manifest)
    required = {
        "actor_id",
        "sentence",
        "intended_emotion",
        "intensity",
        "clip_id",
        "path",
        "split",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")

    train = frame[frame["split"].astype(str) == args.split].copy()
    if train.empty:
        raise ValueError(f"no rows for split {args.split!r}")
    excluded = frame[frame["split"].astype(str) != args.split]
    overlap = set(train["actor_id"].astype(str)) & set(
        excluded["actor_id"].astype(str)
    )
    if overlap:
        raise ValueError(f"split leaks actors into other splits: {sorted(overlap)}")

    clips = select_one_per_group(train)
    pairs = build_pairs(clips, emotions=NON_NEUTRAL_EMOTIONS)
    train_actors = sorted(clips["actor_id"].astype(str).unique())
    if len(train_actors) < 2:
        raise ValueError("adapter training requires at least two training speakers")

    missing_audio = [
        str(path)
        for path in list(pairs["neutral_path"]) + list(pairs["emotion_path"])
        if not _resolve(str(path)).is_file()
    ]
    if missing_audio:
        raise FileNotFoundError(f"missing paired audio: {missing_audio[:5]}")

    cache = _load_cache(args.cache)
    style_by_path = {
        str(path): style for path, style in zip(cache["paths"], cache["styles"])
    }
    pair_paths = sorted(
        {
            str(path)
            for path in list(pairs["neutral_path"]) + list(pairs["emotion_path"])
        }
    )
    missing_styles = [path for path in pair_paths if path not in style_by_path]
    if missing_styles:
        raise KeyError(
            "style cache does not cover all paired clips; "
            f"missing {len(missing_styles)} (first: {missing_styles[:3]})"
        )

    examples = build_examples(pairs, style_by_path)
    if len(examples["targets"]) != len(pairs):
        raise RuntimeError("example count does not match pair count")
    if len(set(examples["speakers"].tolist())) != len(train_actors):
        raise RuntimeError("example speaker set does not match training actors")

    common_kwargs = {
        "hidden_dim": int(args.hidden_dim),
        "max_norm": float(args.max_norm),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "timbre_weight": float(args.timbre_weight),
        "mse_weight": float(args.mse_weight),
        "cosine_weight": float(args.cosine_weight),
        "patience": int(args.patience),
        "val_fraction": float(args.val_fraction),
        "device": str(args.device),
    }

    started = time.time()
    print(
        f"Training main adapter: {len(examples['targets'])} examples, "
        f"{len(train_actors)} actors, device={args.device}",
        flush=True,
    )
    main_result = train_adapter(examples, seed=int(args.seed), **common_kwargs)
    print(
        f"Main: best epoch {main_result['best_epoch']}, "
        f"val loss {main_result['best_val_loss']:.4f}, "
        f"val cosine {main_result['val_metrics']['cosine_mean']:.4f}",
        flush=True,
    )

    sample_style = torch.from_numpy(
        np.asarray(examples["neutral_styles"][0], dtype=np.float32)
    )
    main_payload = {
        "style_mean": torch.from_numpy(main_result["style_mean"]),
        "style_std": torch.from_numpy(main_result["style_std"]),
    }
    main_neutral_zero = _neutral_zero_check(
        main_result["model"],
        main_payload,
        sample_style,
        emotions=EMOTIONS,
    )
    if not main_neutral_zero["passed"]:
        raise RuntimeError(f"main neutral-zero check failed: {main_neutral_zero}")

    main_metadata = {
        "level": "exploratory",
        "role": "research_adapter_not_serving",
        "run_id": str(args.run_id),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "style_cache": str(args.cache),
        "style_cache_sha256": _sha256(args.cache),
        "train_actors": train_actors,
        "n_train_actors": len(train_actors),
        "excluded_splits": sorted(excluded["split"].astype(str).unique()),
        "excluded_actors": sorted(excluded["actor_id"].astype(str).unique()),
        "pairs": int(len(pairs)),
        "pairs_per_emotion": {
            str(emotion): int(count)
            for emotion, count in pairs["emotion"].value_counts().sort_index().items()
        },
        "best_epoch": int(main_result["best_epoch"]),
        "best_val_loss": float(main_result["best_val_loss"]),
        "val_metrics": main_result["val_metrics"],
        "neutral_zero": main_neutral_zero,
        "class_mean_baseline": str(args.class_mean),
        "class_mean_baseline_sha256": (
            _sha256(args.class_mean) if args.class_mean.is_file() else None
        ),
    }
    save_adapter(
        args.artifact,
        main_result["model"],
        style_mean=main_result["style_mean"],
        style_std=main_result["style_std"],
        metadata=main_metadata,
    )

    control_seed = (
        int(args.control_seed)
        if args.control_seed is not None
        else int(args.seed) + 1
    )
    control_examples = shuffle_labels(examples, seed=control_seed)
    print(
        f"Training label-shuffled control: seed={control_seed}",
        flush=True,
    )
    control_result = train_adapter(
        control_examples, seed=int(args.seed), **common_kwargs
    )
    print(
        f"Control: best epoch {control_result['best_epoch']}, "
        f"val loss {control_result['best_val_loss']:.4f}, "
        f"val cosine {control_result['val_metrics']['cosine_mean']:.4f}",
        flush=True,
    )
    control_payload = {
        "style_mean": torch.from_numpy(control_result["style_mean"]),
        "style_std": torch.from_numpy(control_result["style_std"]),
    }
    control_neutral_zero = _neutral_zero_check(
        control_result["model"],
        control_payload,
        sample_style,
        emotions=EMOTIONS,
    )
    if not control_neutral_zero["passed"]:
        raise RuntimeError(
            f"control neutral-zero check failed: {control_neutral_zero}"
        )

    control_metadata = dict(main_metadata)
    control_metadata.update(
        {
            "role": "label_shuffled_control_not_serving",
            "control_seed": control_seed,
            "control_type": "label_shuffle",
            "label_shuffled": True,
            "best_epoch": int(control_result["best_epoch"]),
            "best_val_loss": float(control_result["best_val_loss"]),
            "val_metrics": control_result["val_metrics"],
            "neutral_zero": control_neutral_zero,
        }
    )
    save_adapter(
        args.control_artifact,
        control_result["model"],
        style_mean=control_result["style_mean"],
        style_std=control_result["style_std"],
        metadata=control_metadata,
    )

    report = {
        "schema_version": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "level": "exploratory",
        "run_id": str(args.run_id),
        "split": str(args.split),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "style_cache": str(args.cache),
        "style_cache_sha256": _sha256(args.cache),
        "train_actors": train_actors,
        "n_train_actors": len(train_actors),
        "excluded_splits": sorted(excluded["split"].astype(str).unique()),
        "excluded_actors": sorted(excluded["actor_id"].astype(str).unique()),
        "pairs": int(len(pairs)),
        "pairs_per_emotion": {
            str(emotion): int(count)
            for emotion, count in pairs["emotion"].value_counts().sort_index().items()
        },
        "examples": int(len(examples["targets"])),
        "main": {
            "artifact": str(args.artifact),
            "artifact_sha256": _sha256(args.artifact),
            **_training_summary(main_result),
            "neutral_zero": main_neutral_zero,
        },
        "label_shuffled_control": {
            "artifact": str(args.control_artifact),
            "artifact_sha256": _sha256(args.control_artifact),
            "control_seed": control_seed,
            "control_type": "label_shuffle",
            **_training_summary(control_result),
            "neutral_zero": control_neutral_zero,
        },
        "class_mean_baseline": {
            "artifact": str(args.class_mean),
            "artifact_sha256": (
                _sha256(args.class_mean) if args.class_mean.is_file() else None
            ),
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "notes": [
            "Targets are style(emotion) - style(neutral) for the same actor and sentence.",
            "Speaker-disjoint early stopping uses only the 73 training actors; validation and test actors are excluded.",
            "Neutral and zero intensity are exact no-ops; the adapter output is hard-bounded at max_norm.",
            "The label-shuffled control permutes emotion labels while keeping each neutral style's real target residual fixed.",
            "This is a research artifact, not a serving checkpoint.",
        ],
        "limitations": [
            "CREMA-D is acted speech and every emotion class is below the 2 h confirmatory gate.",
            "The adapter is trained on one seed; seed variance is not estimated here.",
            "The style targets come from the frozen StyleTTS2 encoder, not from the canonical face-to-voice path.",
            "No generated audio was evaluated in this training step; E2 evaluation is separate.",
            "No human MOS, CMOS, or emotion-identification study was run.",
        ],
        "mos": {
            "status": "unavailable",
            "reason": "no human ratings were collected for this run",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {args.artifact}", flush=True)
    print(f"Wrote {args.control_artifact}", flush=True)
    print(f"Wrote {args.report}", flush=True)
    print(
        f"Main val cosine {main_result['val_metrics']['cosine_mean']:.4f}; "
        f"control val cosine {control_result['val_metrics']['cosine_mean']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
