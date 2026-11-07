"""Train the A4 learned accent residual and a label-shuffled control.

Inputs are the GLOBE ECAPA feature cache and the frozen StyleTTS2 style cache.
Targets are speaker-centered style vectors.  Training and early stopping use
speaker-disjoint GLOBE splits; the test split is never touched here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.models.accent_adapter import (  # noqa: E402
    ACCENTS,
    AccentResidualAdapter,
    accent_one_hot,
    residual_metrics,
    save_accent_adapter,
    weighted_style_mse,
)

DEFAULT_FEATURES = ROOT / "data" / "processed" / "globe_accent_features.npz"
DEFAULT_STYLES = ROOT / "data" / "processed" / "globe_accent_style_cache.npz"
DEFAULT_MAIN = ROOT / "checkpoints" / "accent_adapter" / "a4_adapter_seed0.pt"
DEFAULT_CONTROL = (
    ROOT / "checkpoints" / "accent_adapter" / "a4_adapter_label_shuffled_seed0.pt"
)
DEFAULT_REPORT = ROOT / "results" / "accent" / "a4" / "train_report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the A4 accent adapter")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--styles", type=Path, default=DEFAULT_STYLES)
    parser.add_argument("--main-artifact", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--control-artifact", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--classes", nargs="+", default=list(ACCENTS))
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-norm", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--timbre-weight", type=float, default=0.25)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def _load_styles(path: Path) -> pd.DataFrame:
    cache = np.load(path, allow_pickle=True)
    return pd.DataFrame(
        {
            "clip_id": cache["clip_id"].astype(str),
            "speaker_id": cache["speaker_id"].astype(str),
            "accent_class": cache["accent_class"].astype(str),
            "split": cache["split"].astype(str),
            "style": [np.asarray(row, dtype=np.float32).reshape(-1) for row in cache["style"]],
        }
    )


def _load_features(path: Path) -> pd.DataFrame:
    cache = np.load(path, allow_pickle=True)
    features = np.asarray(cache["features"], dtype=np.float32)
    return pd.DataFrame(
        {
            "clip_id": cache["clip_id"].astype(str),
            "speaker_id": cache["speaker_id"].astype(str),
            "accent_class": cache["label"].astype(str),
            "split": cache["split"].astype(str),
            "ecapa": [row[:192] for row in features],
        }
    )


def _center_styles(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for speaker, group in frame.groupby("speaker_id"):
        styles = np.stack(group["style"].to_list())
        mean = styles.mean(axis=0, keepdims=True)
        for (_, row), target in zip(group.iterrows(), styles - mean):
            rows.append(
                {
                    **{key: row[key] for key in ("clip_id", "speaker_id", "accent_class", "split", "ecapa")},
                    "target": target.astype(np.float32),
                }
            )
    return pd.DataFrame(rows)


def _tensors(frame: pd.DataFrame, classes: Sequence[str]) -> Dict[str, torch.Tensor]:
    ecapa = torch.tensor(np.stack(frame["ecapa"].to_list()), dtype=torch.float32)
    targets = torch.tensor(np.stack(frame["target"].to_list()), dtype=torch.float32)
    onehot = torch.stack(
        [accent_one_hot(label, accents=classes) for label in frame["accent_class"]]
    )
    intensity = torch.ones((len(frame), 1), dtype=torch.float32)
    return {"ecapa": ecapa, "onehot": onehot, "intensity": intensity, "target": targets}


def _train(
    train: Dict[str, torch.Tensor],
    val: Dict[str, torch.Tensor],
    *,
    args: argparse.Namespace,
    seed: int,
    accents: Sequence[str],
) -> Tuple[AccentResidualAdapter, Dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AccentResidualAdapter(
        hidden_dim=args.hidden_dim,
        n_accents=len(accents),
        max_norm=args.max_norm,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 0
    history: List[Dict[str, float]] = []
    stale = 0
    batch_size = max(1, int(args.batch_size))
    n = train["ecapa"].shape[0]

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = torch.randperm(n)
        total = 0.0
        for start in range(0, n, batch_size):
            indices = order[start : start + batch_size]
            prediction = model(
                train["ecapa"][indices].to(args.device),
                train["onehot"][indices].to(args.device),
                train["intensity"][indices].to(args.device),
            )
            target = train["target"][indices].to(args.device)
            loss = args.mse_weight * weighted_style_mse(
                prediction, target, timbre_weight=args.timbre_weight
            ) + args.cosine_weight * (
                1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(indices)
        train_loss = total / max(n, 1)

        model.eval()
        with torch.no_grad():
            prediction = model(
                val["ecapa"].to(args.device),
                val["onehot"].to(args.device),
                val["intensity"].to(args.device),
            )
            target = val["target"].to(args.device)
            val_loss = float(
                (
                    args.mse_weight
                    * weighted_style_mse(
                        prediction, target, timbre_weight=args.timbre_weight
                    )
                    + args.cosine_weight
                    * (1.0 - F.cosine_similarity(prediction, target, dim=-1).mean())
                ).item()
            )
            metrics = residual_metrics(
                prediction, target, timbre_weight=args.timbre_weight
            )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
        )
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "val_metrics": residual_metrics(
            model(
                val["ecapa"].to(args.device),
                val["onehot"].to(args.device),
                val["intensity"].to(args.device),
            ),
            val["target"].to(args.device),
            timbre_weight=args.timbre_weight,
        ),
        "history": history,
        "n_parameters": model.n_parameters,
    }


def _class_mean_baseline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    classes: Sequence[str],
) -> Dict[str, Any]:
    means: Dict[str, np.ndarray] = {}
    for accent in classes:
        subset = train[train["accent_class"] == accent]
        if subset.empty:
            continue
        per_speaker = []
        for _, group in subset.groupby("speaker_id"):
            per_speaker.append(np.stack(group["target"].to_list()).mean(axis=0))
        means[accent] = np.stack(per_speaker).mean(axis=0)
    predictions = []
    targets = []
    for _, row in val.iterrows():
        mean = means.get(str(row["accent_class"]))
        if mean is None:
            continue
        predictions.append(mean)
        targets.append(row["target"])
    if not predictions:
        return {"status": "unavailable"}
    prediction = torch.tensor(np.stack(predictions), dtype=torch.float32)
    target = torch.tensor(np.stack(targets), dtype=torch.float32)
    return {
        "status": "computed",
        "cosine_mean": float(
            F.cosine_similarity(prediction, target, dim=-1).mean().item()
        ),
        "weighted_mse": float(weighted_style_mse(prediction, target).item()),
        "prediction_norm_mean": float(prediction.norm(dim=-1).mean().item()),
        "target_norm_mean": float(target.norm(dim=-1).mean().item()),
        "n": int(len(predictions)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    classes = list(dict.fromkeys(str(item).lower() for item in args.classes))
    if "irish" in classes:
        raise ValueError("Irish is evaluation-only and cannot train an accent adapter")
    styles = _load_styles(args.styles)
    features = _load_features(args.features)
    frame = styles.merge(
        features,
        on="clip_id",
        suffixes=("_style", "_feature"),
        how="inner",
    )
    frame = frame[
        frame["accent_class_style"].isin(classes)
        & (frame["speaker_id_style"] == frame["speaker_id_feature"])
        & (frame["split_style"] == frame["split_feature"])
    ].copy()
    frame = frame.rename(
        columns={
            "speaker_id_style": "speaker_id",
            "accent_class_style": "accent_class",
            "split_style": "split",
        }
    )
    frame = frame[["clip_id", "speaker_id", "accent_class", "split", "style", "ecapa"]]
    if frame.empty:
        raise RuntimeError("no aligned style/feature rows")
    frame = _center_styles(frame)
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    val = frame[frame["split"] == "val"].reset_index(drop=True)
    if train.empty or val.empty:
        raise RuntimeError("train and val rows are required")

    train_tensors = _tensors(train, classes)
    val_tensors = _tensors(val, classes)
    main_model, main_result = _train(
        train_tensors, val_tensors, args=args, seed=args.seed, accents=classes
    )
    main_sha = save_accent_adapter(
        args.main_artifact,
        main_model,
        accents=classes,
        metadata={
            "track": "accent",
            "level": "exploratory",
            "role": "a4_research_adapter_not_serving",
            "style_cache": str(args.styles),
            "style_cache_sha256": _sha256(args.styles),
            "features": str(args.features),
            "features_sha256": _sha256(args.features),
            "split": "train",
            "val_split": "val",
            "classes": classes,
            "seed": args.seed,
        },
    )

    control_train = dict(train_tensors)
    shuffled = train["accent_class"].to_numpy().copy()
    control_seed = args.seed + 1
    rng = np.random.default_rng(control_seed)
    rng.shuffle(shuffled)
    control_train["onehot"] = torch.stack(
        [accent_one_hot(label, accents=classes) for label in shuffled]
    )
    control_model, control_result = _train(
        control_train, val_tensors, args=args, seed=args.seed, accents=classes
    )
    control_sha = save_accent_adapter(
        args.control_artifact,
        control_model,
        accents=classes,
        metadata={
            "track": "accent",
            "level": "exploratory",
            "role": "a4_label_shuffled_control",
            "classes": classes,
            "seed": args.seed,
            "control_seed": control_seed,
        },
    )

    baseline = _class_mean_baseline(train, val, classes)
    report = {
        "schema_version": "voix.accent_adapter_training.v1",
        "track": "accent",
        "level": "exploratory",
        "features": str(args.features),
        "features_sha256": _sha256(args.features),
        "styles": str(args.styles),
        "styles_sha256": _sha256(args.styles),
        "classes": classes,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "train_speakers": int(train["speaker_id"].nunique()),
        "val_speakers": int(val["speaker_id"].nunique()),
        "main": {
            "artifact": str(args.main_artifact),
            "artifact_sha256": main_sha,
            "best_epoch": main_result["best_epoch"],
            "best_val_loss": main_result["best_val_loss"],
            "val_metrics": main_result["val_metrics"],
            "n_parameters": main_result["n_parameters"],
        },
        "label_shuffled_control": {
            "artifact": str(args.control_artifact),
            "artifact_sha256": control_sha,
            "best_epoch": control_result["best_epoch"],
            "best_val_loss": control_result["best_val_loss"],
            "val_metrics": control_result["val_metrics"],
        },
        "class_mean_baseline": baseline,
        "config": vars(args),
        "limitations": [
            "GLOBE has no within-speaker cross-accent pairs; targets are speaker-centered.",
            "Speaker centering does not remove all identity leakage.",
            "No generated audio was evaluated in this training step.",
            "No human MOS/CMOS or accent-identification study was run.",
        ],
        "mos": {"status": "unavailable", "reason": "no human ratings collected"},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        f"Main val cosine {main_result['val_metrics']['cosine_mean']:.4f}; "
        f"control {control_result['val_metrics']['cosine_mean']:.4f}; "
        f"class-mean {baseline.get('cosine_mean', float('nan')):.4f}"
    )
    print(f"Wrote {args.main_artifact}")
    print(f"Wrote {args.control_artifact}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
