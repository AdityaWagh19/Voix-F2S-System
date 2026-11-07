"""Evaluate a FaceFeatureV2 corpus without training the generative stack.

The script is intentionally diagnostic: it reports retrieval, embedding health,
gender probes, audio-visual selection quality, and extraction failures.  It is
used as the quality gate before spending GPU time on CFVN/CVAE/adapter training.

Examples:
    python scripts/evaluation/evaluate_v2_corpus.py \
        --features data/processed/voxceleb2_v2/features_v2.h5 \
        --index data/processed/voxceleb2_v2/index.csv \
        --output data/processed/voxceleb2_v2/shard1_health.json

    python scripts/evaluation/evaluate_v2_corpus.py --skip-gender-probe --skip-linear-probe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.retrieval import (
    constant_control,
    embedding_health,
    shuffled_control,
    speaker_retrieval,
)
from voix.training.v2_utils import query_gallery_indices


QUALITY_COLUMNS = [
    "n_faces",
    "n_tracks",
    "track_len",
    "av_sync",
    "mouth_var",
    "mouth_mean",
    "face_area",
    "frontal",
    "centrality",
    "det_score",
    "score",
    "frames_used",
]


def _decode(values: Iterable) -> np.ndarray:
    return np.asarray(
        [v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v) for v in values],
        dtype=object,
    )


def _l2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _query_gallery(speakers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Use the shared splitter: singleton speakers are excluded, never self-matched."""
    return query_gallery_indices(speakers)


def _retrieval(
    features: np.ndarray,
    speakers: np.ndarray,
    *,
    name: str,
    controls: bool = False,
) -> Dict[str, float]:
    query_idx, gallery_idx = _query_gallery(speakers)
    query = torch.from_numpy(_l2(features[query_idx]))
    gallery = torch.from_numpy(_l2(features[gallery_idx]))
    query_speakers = speakers[query_idx]
    gallery_speakers = speakers[gallery_idx]
    result = speaker_retrieval(
        query, query_speakers, gallery, gallery_speakers
    )
    output = {f"{name}_{key}": float(value) for key, value in result.items()}
    if controls:
        shuffled = shuffled_control(
            query, query_speakers, gallery, gallery_speakers
        )
        constant = constant_control(gallery, gallery_speakers)
        output[f"{name}_shuffled_recall_at_1"] = float(
            shuffled["recall_at_1"]
        )
        output[f"{name}_constant_recall_at_1"] = float(
            constant["recall_at_1"]
        )
        output[f"{name}_negative_control_chance"] = float(
            result["chance_at_1"]
        )
    return output


def _within_between(features: np.ndarray, speakers: np.ndarray) -> Dict[str, float]:
    """O(N) within/between cosine means for L2-normalized vectors."""
    normed = _l2(features).astype(np.float64)
    total_sum = normed.sum(axis=0)
    total_pairs = float(len(normed) * (len(normed) - 1))
    within_sum = 0.0
    within_pairs = 0.0
    for speaker in np.unique(speakers):
        group = normed[speakers == speaker]
        if len(group) < 2:
            continue
        group_sum = group.sum(axis=0)
        within_sum += float(group_sum @ group_sum - len(group))
        within_pairs += float(len(group) * (len(group) - 1))
    between_pairs = total_pairs - within_pairs
    between_sum = float(total_sum @ total_sum - len(normed)) - within_sum
    return {
        "within": within_sum / within_pairs if within_pairs else float("nan"),
        "between": between_sum / between_pairs if between_pairs else float("nan"),
        "within_pairs": within_pairs,
        "between_pairs": between_pairs,
    }


def _fit_ridge(
    face: np.ndarray,
    voice: np.ndarray,
    *,
    ridge: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(np.asarray(face, dtype=np.float32))
    y = torch.from_numpy(np.asarray(voice, dtype=np.float32))
    x_aug = torch.cat([x, torch.ones(len(x), 1)], dim=1)
    gram = x_aug.T @ x_aug
    reg = torch.eye(gram.shape[0]) * float(ridge)
    reg[-1, -1] = 0.0
    weights = torch.linalg.solve(gram + reg, x_aug.T @ y)
    return weights[:-1], weights[-1]


def _apply_ridge(face: np.ndarray, weight: torch.Tensor, bias: torch.Tensor) -> np.ndarray:
    x = torch.from_numpy(np.asarray(face, dtype=np.float32))
    return torch.nn.functional.normalize(x @ weight + bias, dim=-1).numpy()


def _linear_probe(
    face_train: np.ndarray,
    voice_train: np.ndarray,
    face_eval: np.ndarray,
    voice_eval: np.ndarray,
    speakers_eval: np.ndarray,
    *,
    name: str,
) -> Dict[str, float]:
    weight, bias = _fit_ridge(face_train, voice_train)
    predicted = _apply_ridge(face_eval, weight, bias)
    result = speaker_retrieval(
        torch.from_numpy(predicted),
        speakers_eval,
        torch.from_numpy(_l2(voice_eval)),
        speakers_eval,
    )
    return {f"{name}_{key}": float(value) for key, value in result.items()}


def _gender_probe(
    features: np.ndarray,
    genders: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    *,
    name: str,
) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train_mask = train_mask & np.isin(genders, ["f", "m"])
    eval_mask = eval_mask & np.isin(genders, ["f", "m"])
    if train_mask.sum() < 20 or eval_mask.sum() < 10:
        return {f"{name}_gender_acc": float("nan"), f"{name}_gender_n": 0.0}
    y_train = (genders[train_mask] == "m").astype(np.int32)
    y_eval = (genders[eval_mask] == "m").astype(np.int32)
    if len(np.unique(y_train)) < 2:
        return {f"{name}_gender_acc": float("nan"), f"{name}_gender_n": 0.0}
    scaler = StandardScaler().fit(features[train_mask])
    classifier = LogisticRegression(max_iter=1000, C=1.0)
    classifier.fit(scaler.transform(features[train_mask]), y_train)
    predictions = classifier.predict(scaler.transform(features[eval_mask]))
    return {
        f"{name}_gender_acc": float((predictions == y_eval).mean()),
        f"{name}_gender_n": float(eval_mask.sum()),
        f"{name}_gender_chance": float(max(y_eval.mean(), 1.0 - y_eval.mean())),
    }


def _percentiles(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        "mean": float(values.mean()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def evaluate(
    features_path: str | Path,
    index_path: Optional[str | Path] = None,
    *,
    scaler_path: Optional[str | Path] = None,
    skip_gender_probe: bool = False,
    skip_linear_probe: bool = False,
) -> dict:
    features_path = Path(features_path)
    with h5py.File(features_path, "r") as store:
        required = {"clip_ids", "speaker_ids", "split", "gender", "face_features", "arcface"}
        missing = required.difference(store.keys())
        if missing:
            raise ValueError(f"{features_path} is not a FaceFeatureV2 corpus; missing {sorted(missing)}")
        clip_ids = _decode(store["clip_ids"][:])
        speakers = _decode(store["speaker_ids"][:])
        splits = _decode(store["split"][:])
        genders = _decode(store["gender"][:])
        arcface = np.asarray(store["arcface"][:], dtype=np.float32)
        fused = np.asarray(store["face_features"][:], dtype=np.float32)
        voice = np.asarray(store["speaker_features"][:], dtype=np.float32)
        quality = (
            np.asarray(store["face_quality"][:], dtype=np.float32)
            if "face_quality" in store
            else None
        )
        face_gender = (
            np.asarray(store["face_gender"][:], dtype=np.int16)
            if "face_gender" in store
            else None
        )
        face_meta_raw = (
            store["face_meta"][:] if "face_meta" in store else None
        )

    n = len(clip_ids)
    if not (len(speakers) == len(splits) == len(genders) == len(arcface) == len(fused) == len(voice) == n):
        raise ValueError("FaceFeatureV2 arrays have inconsistent lengths")
    if len(set(clip_ids.tolist())) != n:
        raise ValueError("FaceFeatureV2 contains duplicate clip_ids")

    if index_path is not None and Path(index_path).exists():
        manifest = pd.read_csv(index_path).drop_duplicates("clip_id").set_index("clip_id")
        unknown = [cid for cid in clip_ids if cid not in manifest.index]
        if unknown:
            raise ValueError(f"{len(unknown)} extracted clips are not in the manifest")
    else:
        manifest = None

    train_mask = splits == "train"
    fused_z = fused
    scaler_source = "none"
    scaler_used = None
    if scaler_path is not None and Path(scaler_path).exists():
        payload = torch.load(scaler_path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema_version") == "FaceFeatureV2"
            and tuple(payload["face_mean"].shape) == (fused.shape[1],)
        ):
            mean = payload["face_mean"].numpy()
            std = np.maximum(payload["face_std"].numpy(), 1e-6)
            fused_z = ((fused - mean) / std).astype(np.float32)
            scaler_source = "artifact"
            scaler_used = str(scaler_path)
    if scaler_source == "none" and train_mask.any():
        mean = fused[train_mask].mean(axis=0)
        std = np.maximum(fused[train_mask].std(axis=0), 1e-6)
        fused_z = ((fused - mean) / std).astype(np.float32)
        scaler_source = "train_stats"

    per_speaker = pd.Series(speakers).value_counts()
    coverage = {
        "speakers": int(len(per_speaker)),
        "singleton_speakers": int((per_speaker == 1).sum()),
        "speakers_with_fewer_than_2_clips": int((per_speaker < 2).sum()),
        "clips_per_speaker_min": int(per_speaker.min()) if len(per_speaker) else 0,
        "clips_per_speaker_median": float(per_speaker.median()) if len(per_speaker) else 0.0,
        "clips_per_speaker_max": int(per_speaker.max()) if len(per_speaker) else 0,
    }
    if manifest is not None:
        manifest_speakers = set(manifest["speaker_id"].astype(str))
        coverage["manifest_speakers"] = int(len(manifest_speakers))
        coverage["manifest_speakers_missing"] = int(
            len(manifest_speakers.difference(set(speakers.tolist())))
        )

    report: dict = {
        "features": str(features_path),
        "index": str(index_path) if index_path else None,
        "clips": int(n),
        "speakers": int(len(set(speakers.tolist()))),
        "coverage": coverage,
        "scaler": scaler_used,
        "scaler_source": scaler_source,
        "splits": {},
        "representations": {},
        "quality": {},
    }
    for split in sorted(set(splits.tolist())):
        mask = splits == split
        report["splits"][split] = {
            "clips": int(mask.sum()),
            "speakers": int(len(set(speakers[mask].tolist()))),
            "gender": {str(k): int(v) for k, v in pd.Series(genders[mask]).value_counts().items()},
        }

    for name, values in (
        ("arcface", arcface),
        ("fused", fused),
        ("fused_z", fused_z),
        ("voice", voice),
    ):
        report["representations"][name] = {
            "retrieval_all": _retrieval(values, speakers, name=name),
            "health_all": _within_between(values, speakers),
        }

    # Held-out retrieval: evaluate val and test separately when present.
    eval_splits = [split for split in ("val", "test") if (splits == split).any()]
    for split in eval_splits:
        mask = splits == split
        for name, values in (
            ("arcface", arcface),
            ("fused", fused),
            ("fused_z", fused_z),
            ("voice", voice),
        ):
            report["representations"][name][f"retrieval_{split}"] = _retrieval(
                values[mask], speakers[mask], name=name, controls=True
            )

    # Face -> voice linear probe, fit on train and evaluated on held-out splits.
    if not skip_linear_probe and train_mask.any():
        for split in eval_splits:
            eval_mask = splits == split
            for name, values in (
                ("arcface", arcface),
                ("fused", fused),
                ("fused_z", fused_z),
            ):
                key = f"{name}_linear_face_to_voice_{split}"
                report["representations"][name][key] = _linear_probe(
                    _l2(values[train_mask]),
                    _l2(voice[train_mask]),
                    _l2(values[eval_mask]),
                    voice[eval_mask],
                    speakers[eval_mask],
                    name=name,
                )

    if not skip_gender_probe and train_mask.any():
        for split in eval_splits:
            eval_mask = splits == split
            for name, values in (
                ("arcface", arcface),
                ("fused", fused),
                ("fused_z", fused_z),
                ("voice", voice),
            ):
                report["representations"][name][f"gender_probe_{split}"] = _gender_probe(
                    _l2(values), genders, train_mask, eval_mask, name=name
                )

    if face_gender is not None:
        known = np.isin(genders, ["f", "m"]) & (face_gender >= 0)
        mapped = np.where(face_gender[known] == 1, "m", "f")
        report["face_gender_attribute_acc"] = float((mapped == genders[known]).mean())
        report["face_gender_attribute_n"] = int(known.sum())

    if quality is not None:
        if quality.shape[1] < len(QUALITY_COLUMNS):
            raise ValueError("face_quality has fewer columns than expected")
        for index, name in enumerate(QUALITY_COLUMNS):
            report["quality"][name] = _percentiles(quality[:, index])
        av = quality[:, 3]
        pairs = quality[:, 2]
        report["quality"]["av_sync_positive_fraction"] = float((av > 0.3).mean())
        report["quality"]["track_len_ge2_fraction"] = float((pairs >= 2).mean())

    if face_meta_raw is not None:
        av_pairs = []
        geometry_ok = []
        track_lens = []
        n_tracks = []
        for raw in face_meta_raw:
            try:
                meta = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
            except Exception:
                continue
            av_pairs.append(int(meta.get("av_pairs", 0)))
            geometry_ok.append(bool(meta.get("geometry_ok", False)))
            track_lens.append(int(meta.get("track_len", 0)))
            n_tracks.append(int(meta.get("n_tracks", 0)))
        report["selection"] = {
            "av_pairs": _percentiles(np.asarray(av_pairs)),
            "av_pairs_ge3_fraction": float(np.mean(np.asarray(av_pairs) >= 3)) if av_pairs else float("nan"),
            "geometry_ok_fraction": float(np.mean(geometry_ok)) if geometry_ok else float("nan"),
            "track_len": _percentiles(np.asarray(track_lens)),
            "n_tracks": _percentiles(np.asarray(n_tracks)),
        }

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a FaceFeatureV2 corpus")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument(
        "--scaler",
        type=Path,
        default=ROOT / "data" / "processed" / "voxceleb2_v2" / "face_scaler_v2.pt",
        help="Canonical 560-D scaler; falls back to train-split statistics if absent",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--skip-gender-probe", action="store_true")
    parser.add_argument("--skip-linear-probe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        args.features,
        args.index,
        scaler_path=args.scaler,
        skip_gender_probe=args.skip_gender_probe,
        skip_linear_probe=args.skip_linear_probe,
    )
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    print(text)


if __name__ == "__main__":
    main()
