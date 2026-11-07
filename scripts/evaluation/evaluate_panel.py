"""Run an independent external evaluator panel on real and generated audio.

The panel is deliberately outside the synthesis path.  It never feeds a
recognition result back into the accent, emotion, or face-to-voice controls.
It exists to answer one question: do frozen, third-party recognizers see the
intended categories in the generated clips, and are those recognizers valid at
all on this audio domain?

Evaluators
----------
* CommonAccent ECAPA (MIT) for the four supported accents.
* SUPERB wav2vec2-base emotion recognizer (Apache-2.0) for the four supported
  emotions.
* emotion2vec+ large (FunASR model license, ``other``) for the same emotions.

The panel uses the same standardized-centroid domain-shift statistic as the
CREMA-D evaluator.  A recognizer is only allowed to support a conclusion when
its real-speech accuracy is above chance, its predictions have not collapsed,
and the generated feature centroid is inside the real-speech spread
(``ratio <= 3.0``).  Otherwise the recognizer is reported as invalid for the
generated domain and no category claim is made.

This script does not measure MOS, CMOS, emotion identification by humans, or
accent identification by humans.  It does not produce a serving adapter.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# The installed TensorFlow is incomplete on this machine.  Transformers probes
# TensorFlow while padding audio and crashes on `tf.Tensor`.  The panel is
# PyTorch-only, so disable the TensorFlow backend before Transformers imports.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voix.evaluation.accent import classifier_metrics, majority_baseline
from voix.evaluation.crema_e0 import feature_domain_shift

SCHEMA_VERSION = "voix.evaluation.panel.v1"

ACCENT_LABELS: Tuple[str, ...] = ("us", "england", "indian_english", "australian")
EMOTION_LABELS: Tuple[str, ...] = ("neutral", "happy", "sad", "angry")

DEFAULT_GLOBE_MANIFEST = ROOT / "data" / "processed" / "globe_accent_manifest.csv"
DEFAULT_GLOBE_AUDIO_ROOT = ROOT / "data" / "processed" / "globe_audio"
DEFAULT_CREMA_MANIFEST = ROOT / "data" / "processed" / "crema_d_manifest.csv"
DEFAULT_DEMO = ROOT / "results" / "demo" / "research_v2"
DEFAULT_OUTPUT = ROOT / "results" / "evaluation" / "panel_v2"
DEFAULT_MODEL_CACHE = ROOT / "checkpoints" / "models"

COMMONACCENT_MODEL_ID = "Jzuluaga/accent-id-commonaccent_ecapa"
COMMONACCENT_REVISION = "14bebf44b7e7a34204d0acc2c897935945fb5c51"
SUPERB_MODEL_ID = "superb/wav2vec2-base-superb-er"
SUPERB_REVISION = "441a7599c3b22107314dcbd9166621c5c83f2cc5"
EMOTION2VEC_MODEL_ID = "emotion2vec/emotion2vec_plus_large"
EMOTION2VEC_REVISION = "6c303ba987b86b93193de93e34bb2b077a6bedc4"
EMOTION2VEC_FUNASR_RUNTIME = DEFAULT_MODEL_CACHE / "funasr_runtime"

# Static task map so the input guard can require only the generated tracks the
# selected evaluators actually consume.
EVALUATOR_TASKS: Dict[str, str] = {
    "commonaccent": "accent",
    "superb_er": "emotion",
    "emotion2vec": "emotion",
}

EMOTION2VEC_LABELS: Tuple[str, ...] = (
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
)

DOMAIN_SHIFT_THRESHOLD = 3.0
MIN_REAL_ACCURACY_MARGIN = 0.10
MIN_REAL_PER_CLASS = 10
MIN_REAL_ROWS = 80
MAX_MAJORITY_PREDICTION_FRACTION = 0.90

# GLOBE uses self-reported accent names that are slightly different from the
# CommonAccent label vocabulary.
GLOBE_TO_VOIX: Dict[str, str] = {
    "us": "us",
    "england": "england",
    "australian": "australian",
    "india_south_asia": "indian_english",
}

ACCENT_LABEL_ALIASES: Dict[str, str] = {
    "us": "us",
    "american": "us",
    "general_american": "us",
    "en_us": "us",
    "england": "england",
    "british": "england",
    "british_rp": "england",
    "en_gb": "england",
    "australia": "australian",
    "australian": "australian",
    "indian": "indian_english",
    "india": "indian_english",
    "indian_english": "indian_english",
    "india_south_asia": "indian_english",
}

EMOTION_LABEL_ALIASES: Dict[str, str] = {
    "neutral": "neutral",
    "neu": "neutral",
    "happy": "happy",
    "hap": "happy",
    "sad": "sad",
    "sadness": "sad",
    "angry": "angry",
    "ang": "angry",
    "anger": "angry",
}


def normalize_accent_label(value: Any) -> Optional[str]:
    """Map a recognizer/GLOBE accent label to one of the four VOIX accents."""
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return ACCENT_LABEL_ALIASES.get(key)


def normalize_emotion_label(value: Any) -> Optional[str]:
    """Map a recognizer/CREMA-D emotion label to one of the four VOIX emotions."""
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return EMOTION_LABEL_ALIASES.get(key)


def canonical_emotion2vec_label(value: Any) -> Optional[str]:
    """Map a bilingual emotion2vec label (``angry/生气``) to a VOIX label."""
    text = str(value).strip().lower()
    for part in re.split(r"[/|]", text):
        key = part.strip()
        if key in EMOTION2VEC_LABELS:
            return key
    return normalize_emotion_label(text)


def _resolve_path(value: Any) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> Dict[str, Any]:
    """Describe an artifact without hashing multi-hundred-megabyte weights."""
    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    record: Dict[str, Any] = {
        "path": str(path),
        "status": "present",
        "size_bytes": int(path.stat().st_size),
    }
    if path.stat().st_size <= 128 * 1024 * 1024:
        record["sha256"] = _sha256(path)
    else:
        record["sha256"] = None
        record["sha256_note"] = "skipped for files larger than 128 MiB"
    return record


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def _format_ci(entry: Mapping[str, Any]) -> str:
    if entry.get("status") != "computed":
        return "unavailable"
    accuracy = float(entry.get("accuracy", 0.0))
    interval = entry.get("speaker_cluster_ci") or {}
    if interval.get("status") == "computed":
        return (
            f"{accuracy:.3f} "
            f"[{float(interval['accuracy_low']):.3f}, "
            f"{float(interval['accuracy_high']):.3f}]"
        )
    return f"{accuracy:.3f}"


# ---------------------------------------------------------------------------
# Dataset manifests
# ---------------------------------------------------------------------------


def _sample_per_class(
    frame: pd.DataFrame,
    *,
    max_per_class: int,
    seed: int,
    label_column: str = "label",
) -> pd.DataFrame:
    if max_per_class <= 0 or frame.empty:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    chunks: List[pd.DataFrame] = []
    for label in sorted(frame[label_column].dropna().unique()):
        subset = frame[frame[label_column] == label]
        if len(subset) > max_per_class:
            indices = rng.choice(len(subset), size=int(max_per_class), replace=False)
            subset = subset.iloc[np.sort(indices)]
        chunks.append(subset)
    if not chunks:
        return frame.iloc[0:0].reset_index(drop=True)
    return pd.concat(chunks, ignore_index=True)


def build_globe_rows(
    manifest: Path,
    audio_root: Path,
    *,
    speaker_split: str,
    max_per_class: int = 0,
    seed: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select speaker-disjoint GLOBE rows for one split.

    The audio folders are keyed by ``speaker_split`` (train/test), not by the
    manifest's ``split`` column.  A row with ``split == train`` and
    ``speaker_split == test`` is a held-out-speaker clip and lives under
    ``globe_audio/test/...``.
    """
    frame = pd.read_csv(manifest)
    required = {"clip_id", "accent_class", "client_id", "speaker_split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"GLOBE manifest is missing columns: {missing}")

    source = frame[frame["speaker_split"].astype(str) == str(speaker_split)].copy()
    source = source[source["accent_class"].astype(str).isin(GLOBE_TO_VOIX)].copy()
    source["label"] = source["accent_class"].astype(str).map(GLOBE_TO_VOIX)
    source["speaker_id"] = source["client_id"].astype(str)
    source["clip_id"] = source["clip_id"].astype(str)
    source["wav_path"] = [
        _resolve_path(
            audio_root
            / str(speaker_split)
            / str(accent_class)
            / f"{clip_id}.wav"
        )
        for accent_class, clip_id in zip(source["accent_class"], source["clip_id"])
    ]

    missing_by_class: Dict[str, int] = {}
    present_mask: List[bool] = []
    for row in source.itertuples(index=False):
        exists = Path(row.wav_path).is_file()
        present_mask.append(exists)
        if not exists:
            missing_by_class[str(row.label)] = missing_by_class.get(str(row.label), 0) + 1
    source = source.loc[present_mask].copy()
    selected = _sample_per_class(source, max_per_class=max_per_class, seed=seed)
    selected = selected.sort_values(["label", "clip_id"]).reset_index(drop=True)
    diagnostics = {
        "dataset": "GLOBE",
        "manifest": str(manifest),
        "audio_root": str(audio_root),
        "speaker_split": str(speaker_split),
        "source_rows": int(len(frame[frame["speaker_split"].astype(str) == str(speaker_split)])),
        "target_class_rows": int(len(source) + sum(missing_by_class.values())),
        "missing_audio": int(sum(missing_by_class.values())),
        "missing_by_class": missing_by_class,
        "n_selected": int(len(selected)),
        "n_speakers": int(selected["speaker_id"].nunique()) if not selected.empty else 0,
        "class_counts": selected["label"].value_counts().sort_index().to_dict(),
        "sampling_seed": int(seed),
        "max_per_class": int(max_per_class),
    }
    return selected, diagnostics


def build_crema_rows(
    manifest: Path,
    *,
    split: str,
    max_per_class: int = 0,
    seed: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select CREMA-D rows for the four supported emotions."""
    frame = pd.read_csv(manifest)
    required = {"clip_id", "path", "actor_id", "intended_emotion", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CREMA-D manifest is missing columns: {missing}")

    source = frame[frame["split"].astype(str) == str(split)].copy()
    source = source[source["intended_emotion"].astype(str).isin(EMOTION_LABELS)].copy()
    source["label"] = source["intended_emotion"].astype(str)
    source["speaker_id"] = source["actor_id"].astype(str)
    source["clip_id"] = source["clip_id"].astype(str)
    source["wav_path"] = [_resolve_path(value) for value in source["path"]]

    missing_by_class: Dict[str, int] = {}
    present_mask: List[bool] = []
    for row in source.itertuples(index=False):
        exists = Path(row.wav_path).is_file()
        present_mask.append(exists)
        if not exists:
            missing_by_class[str(row.label)] = missing_by_class.get(str(row.label), 0) + 1
    source = source.loc[present_mask].copy()
    selected = _sample_per_class(source, max_per_class=max_per_class, seed=seed)
    selected = selected.sort_values(["label", "clip_id"]).reset_index(drop=True)
    diagnostics = {
        "dataset": "CREMA-D",
        "manifest": str(manifest),
        "split": str(split),
        "source_rows": int(len(frame[frame["split"].astype(str) == str(split)])),
        "target_class_rows": int(len(source) + sum(missing_by_class.values())),
        "missing_audio": int(sum(missing_by_class.values())),
        "missing_by_class": missing_by_class,
        "n_selected": int(len(selected)),
        "n_speakers": int(selected["speaker_id"].nunique()) if not selected.empty else 0,
        "class_counts": selected["label"].value_counts().sort_index().to_dict(),
        "sampling_seed": int(seed),
        "max_per_class": int(max_per_class),
        "excluded_classes": ["fear", "disgust"],
    }
    return selected, diagnostics


def build_generated_rows(demo_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a research-demo index and attach normalized target labels."""
    index_path = demo_dir / "index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"demo index not found: {index_path}")
    frame = pd.read_csv(index_path)
    required = {"name", "voice_id", "track", "condition", "wav_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"demo index is missing columns: {missing}")

    frame = frame.copy()
    frame["clip_id"] = frame["name"].astype(str)
    frame["speaker_id"] = frame["voice_id"].astype(str)
    frame["track"] = frame["track"].astype(str).str.lower()
    frame["condition"] = frame["condition"].astype(str).str.lower()
    frame["wav_path"] = [_resolve_path(value) for value in frame["wav_path"]]

    missing_paths = [str(path) for path in frame["wav_path"] if not path.is_file()]
    frame = frame[frame["wav_path"].map(Path.is_file)].copy()
    diagnostics = {
        "dataset": "generated_demo",
        "demo_dir": str(demo_dir),
        "index": str(index_path),
        "n_index_rows": int(len(frame) + len(missing_paths)),
        "n_selected": int(len(frame)),
        "missing_audio": len(missing_paths),
        "missing_paths": missing_paths[:10],
        "voices": sorted(frame["speaker_id"].unique().tolist()),
        "tracks": frame["track"].value_counts().sort_index().to_dict(),
    }
    return frame, diagnostics


def generated_task_rows(
    frame: pd.DataFrame,
    task: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select the generated rows that have an intended label for ``task``."""
    if task == "accent":
        source = frame[frame["track"] == "accent"].copy()
        source["label"] = source["condition"].map(normalize_accent_label)
        expected = ACCENT_LABELS
    elif task == "emotion":
        source = frame[frame["track"].isin(["baseline", "emotion"])].copy()
        source["label"] = source["condition"].map(normalize_emotion_label)
        expected = EMOTION_LABELS
    else:
        raise ValueError(f"unknown task: {task}")

    selected = source[source["label"].notna()].copy()
    selected = selected.sort_values(["speaker_id", "label"]).reset_index(drop=True)
    present = [label for label in expected if label in set(selected["label"])]
    missing = [label for label in expected if label not in set(selected["label"])]
    diagnostics = {
        "task": task,
        "n": int(len(selected)),
        "n_voices": int(selected["speaker_id"].nunique()) if not selected.empty else 0,
        "voices": sorted(selected["speaker_id"].unique().tolist()),
        "labels": present,
        "missing_labels": missing,
        "class_counts": selected["label"].value_counts().sort_index().to_dict(),
    }
    return selected, diagnostics


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def _load_audio_16k(path: Path):
    import librosa
    import torch

    wav, _sr = librosa.load(str(path), sr=16000, mono=True)
    if wav.size == 0:
        raise ValueError(f"empty audio: {path}")
    return torch.from_numpy(np.asarray(wav, dtype=np.float32))


def _pad_waveforms(waveforms: Sequence[Any]):
    import torch

    if not waveforms:
        raise ValueError("no waveforms")
    max_len = max(int(wav.numel()) for wav in waveforms)
    padded = torch.zeros((len(waveforms), max_len), dtype=torch.float32)
    lengths = torch.empty(len(waveforms), dtype=torch.float32)
    for index, wav in enumerate(waveforms):
        length = int(wav.numel())
        padded[index, :length] = wav
        lengths[index] = float(length) / float(max_len)
    return padded, lengths


# ---------------------------------------------------------------------------
# Evaluator adapters
# ---------------------------------------------------------------------------


@dataclass
class EvaluatorOutputs:
    paths: List[str]
    features: np.ndarray
    probs: np.ndarray
    model_labels: List[str]
    predicted_raw: List[str]
    predicted: List[Optional[str]]
    feature_kind: str = "embedding"

    def subset(self, indices: Sequence[int]) -> "EvaluatorOutputs":
        index_array = np.asarray(list(indices), dtype=int)
        if index_array.size == 0:
            return EvaluatorOutputs(
                paths=[],
                features=np.asarray(self.features)[0:0],
                probs=np.asarray(self.probs)[0:0],
                model_labels=list(self.model_labels),
                predicted_raw=[],
                predicted=[],
                feature_kind=self.feature_kind,
            )
        return EvaluatorOutputs(
            paths=[self.paths[int(index)] for index in index_array],
            features=np.asarray(self.features)[index_array],
            probs=np.asarray(self.probs)[index_array],
            model_labels=list(self.model_labels),
            predicted_raw=[self.predicted_raw[int(index)] for index in index_array],
            predicted=[self.predicted[int(index)] for index in index_array],
            feature_kind=self.feature_kind,
        )


class BaseEvaluator:
    name = "base"
    task = "accent"
    model_id = ""

    def fingerprint(self) -> str:
        raise NotImplementedError

    def predict_files(
        self,
        paths: Sequence[Path],
        *,
        batch_size: int = 8,
    ) -> EvaluatorOutputs:
        raise NotImplementedError


class CommonAccentEvaluator(BaseEvaluator):
    name = "commonaccent"
    task = "accent"
    model_id = COMMONACCENT_MODEL_ID

    def __init__(self, *, device: str, cache_dir: Path):
        from speechbrain.inference import EncoderClassifier
        from speechbrain.utils.fetching import FetchConfig, LocalStrategy

        self.device = device
        self.cache_dir = Path(cache_dir)
        # Windows does not grant symlink privileges by default, so use a
        # physical copy.  Pinning the revision keeps the panel reproducible.
        self.model = EncoderClassifier.from_hparams(
            source=COMMONACCENT_MODEL_ID,
            savedir=str(self.cache_dir),
            local_strategy=LocalStrategy.COPY,
            fetch_config=FetchConfig(revision=COMMONACCENT_REVISION),
            run_opts={"device": device},
        )
        encoder = self.model.hparams.label_encoder
        ind2lab = getattr(encoder, "ind2lab", None)
        if ind2lab:
            self.model_labels = [str(ind2lab[index]) for index in sorted(ind2lab)]
        else:
            decoded = encoder.decode_torch(
                __import__("torch").arange(len(encoder))
            )
            self.model_labels = [str(label) for label in decoded]
        self._label_to_index = {
            label: index for index, label in enumerate(self.model_labels)
        }
        self._artifacts = [
            _artifact_record(self.cache_dir / "embedding_model.ckpt"),
            _artifact_record(self.cache_dir / "classifier.ckpt"),
            _artifact_record(self.cache_dir / "accent_encoder.txt"),
        ]

    def fingerprint(self) -> str:
        return (
            f"{self.model_id}|{COMMONACCENT_REVISION}|{self.cache_dir}|"
            + "|".join(self.model_labels)
        )

    def predict_files(
        self,
        paths: Sequence[Path],
        *,
        batch_size: int = 4,
    ) -> EvaluatorOutputs:
        import torch

        features: List[np.ndarray] = []
        probs: List[np.ndarray] = []
        predicted_raw: List[str] = []
        for start in range(0, len(paths), max(1, int(batch_size))):
            chunk = [Path(path) for path in paths[start : start + max(1, int(batch_size))]]
            waveforms = [_load_audio_16k(path) for path in chunk]
            padded, lengths = _pad_waveforms(waveforms)
            with torch.no_grad():
                embeddings = self.model.encode_batch(padded, lengths)
                logits = self.model.mods.classifier(embeddings)
                if logits.dim() == 3:
                    logits = logits.squeeze(1)
                chunk_probs = logits.softmax(dim=-1)
            features.append(
                embeddings.detach().cpu().numpy().reshape(len(chunk), -1).astype(np.float32)
            )
            probs.append(chunk_probs.detach().cpu().numpy().astype(np.float32))
            for row in chunk_probs.detach().cpu().numpy():
                predicted_raw.append(
                    self.model_labels[int(np.argmax(row))]
                )
        return EvaluatorOutputs(
            paths=[str(Path(path)) for path in paths],
            features=np.concatenate(features, axis=0),
            probs=np.concatenate(probs, axis=0),
            model_labels=list(self.model_labels),
            predicted_raw=predicted_raw,
            predicted=[normalize_accent_label(label) for label in predicted_raw],
            feature_kind="embedding",
        )

    def artifacts(self) -> List[Dict[str, Any]]:
        return list(self._artifacts)


class SuperbEmotionEvaluator(BaseEvaluator):
    name = "superb_er"
    task = "emotion"
    model_id = SUPERB_MODEL_ID

    def __init__(self, *, device: str, cache_dir: Path):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.device = device
        self.cache_dir = Path(cache_dir)
        local_model = (
            str(self.cache_dir)
            if (self.cache_dir / "config.json").is_file()
            else SUPERB_MODEL_ID
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            local_model,
            revision=None if local_model == str(self.cache_dir) else SUPERB_REVISION,
            cache_dir=str(self.cache_dir),
        )
        self.model = (
            AutoModelForAudioClassification.from_pretrained(
                local_model,
                revision=None if local_model == str(self.cache_dir) else SUPERB_REVISION,
                cache_dir=str(self.cache_dir),
            )
            .to(device)
            .eval()
        )
        self.positional_conv_repaired = self._repair_positional_convolution()
        id2label = self.model.config.id2label
        self.model_labels = [
            str(id2label[index]) for index in sorted(id2label)
        ]
        self._artifacts = [
            _artifact_record(self.cache_dir / "config.json"),
            _artifact_record(self.cache_dir / "preprocessor_config.json"),
            _artifact_record(self.cache_dir / "pytorch_model.bin"),
        ]

    def _repair_positional_convolution(self) -> bool:
        """Load old-style wav2vec2 weight-norm tensors into the new module.

        The SUPERB checkpoint predates the ``torch.nn.utils.parametrizations``
        weight-norm migration.  Transformers 4.40 renames the positional
        convolution parameters, leaves them randomly initialized, and only
        warns.  That would silently corrupt every prediction, so copy the
        original ``weight_g``/``weight_v`` tensors into ``original0``/
        ``original1`` after loading.
        """
        import torch

        weight_path = self.cache_dir / "pytorch_model.bin"
        if not weight_path.is_file():
            return False
        try:
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older torch
            state = torch.load(weight_path, map_location="cpu")
        prefix = "wav2vec2.encoder.pos_conv_embed.conv."
        weight_g = state.get(prefix + "weight_g")
        weight_v = state.get(prefix + "weight_v")
        if weight_g is None or weight_v is None:
            return False
        conv = self.model.wav2vec2.encoder.pos_conv_embed.conv
        with torch.no_grad():
            conv.parametrizations.weight.original0.copy_(weight_g)
            conv.parametrizations.weight.original1.copy_(weight_v)
        return True

    def fingerprint(self) -> str:
        return (
            f"{self.model_id}|{SUPERB_REVISION}|"
            f"pos_conv_repaired={self.positional_conv_repaired}|"
            + "|".join(self.model_labels)
        )

    def predict_files(
        self,
        paths: Sequence[Path],
        *,
        batch_size: int = 8,
    ) -> EvaluatorOutputs:
        import torch

        features: List[np.ndarray] = []
        probs: List[np.ndarray] = []
        predicted_raw: List[str] = []
        for start in range(0, len(paths), max(1, int(batch_size))):
            chunk = [Path(path) for path in paths[start : start + max(1, int(batch_size))]]
            waveforms = [
                _load_audio_16k(path).numpy() for path in chunk
            ]
            inputs = self.feature_extractor(
                waveforms,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                chunk_probs = outputs.logits.softmax(dim=-1)
                hidden = outputs.hidden_states[-1]
                attention = inputs.get("attention_mask")
                if attention is not None:
                    try:
                        feat_lengths = (
                            self.model.wav2vec2._get_feat_extract_output_lengths(
                                attention.sum(dim=-1)
                            ).to(hidden.device)
                        )
                    except Exception:
                        feat_lengths = torch.full(
                            (hidden.shape[0],),
                            hidden.shape[1],
                            dtype=torch.long,
                            device=hidden.device,
                        )
                    frame_mask = (
                        torch.arange(hidden.shape[1], device=hidden.device)[None, :]
                        < feat_lengths[:, None]
                    )
                    pooled = (
                        hidden * frame_mask.unsqueeze(-1)
                    ).sum(dim=1) / frame_mask.sum(dim=1, keepdim=True).clamp(min=1)
                else:
                    pooled = hidden.mean(dim=1)
            features.append(pooled.detach().cpu().numpy().astype(np.float32))
            probs.append(chunk_probs.detach().cpu().numpy().astype(np.float32))
            for row in chunk_probs.detach().cpu().numpy():
                predicted_raw.append(self.model_labels[int(np.argmax(row))])
        return EvaluatorOutputs(
            paths=[str(Path(path)) for path in paths],
            features=np.concatenate(features, axis=0),
            probs=np.concatenate(probs, axis=0),
            model_labels=list(self.model_labels),
            predicted_raw=predicted_raw,
            predicted=[normalize_emotion_label(label) for label in predicted_raw],
            feature_kind="embedding",
        )

    def artifacts(self) -> List[Dict[str, Any]]:
        return list(self._artifacts)


class Emotion2VecEvaluator(BaseEvaluator):
    name = "emotion2vec"
    task = "emotion"
    model_id = EMOTION2VEC_MODEL_ID

    def __init__(self, *, device: str, cache_dir: Path):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not (self.cache_dir / "model.pt").is_file():
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=EMOTION2VEC_MODEL_ID,
                revision=EMOTION2VEC_REVISION,
                local_dir=str(self.cache_dir),
            )
        # funasr is installed in an isolated target so it cannot upgrade the
        # project's NumPy/PyTorch environment.  Add it only for this adapter.
        runtime = Path(EMOTION2VEC_FUNASR_RUNTIME)
        if runtime.is_dir() and str(runtime) not in sys.path:
            sys.path.insert(0, str(runtime))
        from funasr import AutoModel

        self.model = AutoModel(
            model=str(self.cache_dir),
            device=device,
            disable_update=True,
        )
        self.backend = "funasr"
        self.model_labels = list(EMOTION2VEC_LABELS)
        self._artifacts = [
            _artifact_record(self.cache_dir / "config.yaml"),
            _artifact_record(self.cache_dir / "configuration.json"),
            _artifact_record(self.cache_dir / "model.pt"),
        ]

    def fingerprint(self) -> str:
        return (
            f"{self.model_id}|{EMOTION2VEC_REVISION}|{self.backend}|"
            + "|".join(self.model_labels)
        )

    def _infer_one(self, path: Path) -> Mapping[str, Any]:
        if self.backend != "funasr":  # pragma: no cover - defensive
            raise RuntimeError("emotion2vec backend is not initialized")
        result = self.model.generate(
            input=str(path),
            granularity="utterance",
            extract_embedding=True,
        )
        if isinstance(result, list):
            if not result:
                raise ValueError("emotion2vec returned an empty list")
            item = result[0]
        else:
            item = result
        if not isinstance(item, Mapping):
            raise ValueError(f"unexpected emotion2vec output: {type(item)!r}")
        return item

    def predict_files(
        self,
        paths: Sequence[Path],
        *,
        batch_size: int = 1,
    ) -> EvaluatorOutputs:
        features: List[np.ndarray] = []
        probs: List[np.ndarray] = []
        predicted_raw: List[str] = []
        feature_kind = "embedding"
        for path in paths:
            item = self._infer_one(Path(path))
            labels = item.get("labels", item.get("label"))
            scores = item.get("scores", item.get("score"))
            if labels is None or scores is None:
                raise ValueError(
                    f"emotion2vec output has no labels/scores for {path}: "
                    f"{sorted(item)}"
                )
            if isinstance(labels, str):
                labels = [labels]
            labels = [str(label) for label in labels]
            scores = np.asarray(scores, dtype=np.float64).reshape(-1)
            if scores.size != len(labels):
                raise ValueError(
                    "emotion2vec labels and scores have different lengths for "
                    f"{path}"
                )
            if scores.size == 0:
                raise ValueError(f"emotion2vec returned no scores for {path}")
            if scores.sum() <= 0.0 or not np.isfinite(scores).all():
                scores = np.zeros_like(scores)
            else:
                scores = scores / scores.sum()
            full = np.zeros(len(self.model_labels), dtype=np.float32)
            for label, score in zip(labels, scores):
                key = canonical_emotion2vec_label(label)
                if key in self.model_labels:
                    full[self.model_labels.index(key)] = float(score)
            if full.sum() > 0.0:
                full = full / full.sum()
            probs.append(full)
            raw = labels[int(np.argmax(scores))]
            predicted_raw.append(raw)

            embedding = None
            for key in ("feats", "embeddings", "embedding", "feature"):
                if key in item and item[key] is not None:
                    embedding = np.asarray(item[key], dtype=np.float32)
                    break
            if embedding is not None and embedding.size > 0:
                if embedding.ndim == 2:
                    embedding = embedding.mean(axis=0)
                features.append(embedding.reshape(-1))
                feature_kind = "embedding"
            else:
                features.append(full.astype(np.float32))
                feature_kind = "probabilities"

        feature_matrix = np.vstack(features).astype(np.float32)
        return EvaluatorOutputs(
            paths=[str(Path(path)) for path in paths],
            features=feature_matrix,
            probs=np.vstack(probs).astype(np.float32),
            model_labels=list(self.model_labels),
            predicted_raw=predicted_raw,
            predicted=[canonical_emotion2vec_label(label) for label in predicted_raw],
            feature_kind=feature_kind,
        )

    def artifacts(self) -> List[Dict[str, Any]]:
        return list(self._artifacts)


def build_evaluator(
    name: str,
    *,
    device: str,
    model_cache: Path,
) -> BaseEvaluator:
    key = str(name).strip().lower()
    if key == "commonaccent":
        return CommonAccentEvaluator(
            device=device,
            cache_dir=model_cache / "accent-id-commonaccent_ecapa",
        )
    if key in {"superb_er", "superb"}:
        return SuperbEmotionEvaluator(
            device=device,
            cache_dir=model_cache / "superb_wav2vec2-base-superb-er",
        )
    if key in {"emotion2vec", "emotion2vec_plus_large"}:
        return Emotion2VecEvaluator(
            device=device,
            cache_dir=model_cache / "emotion2vec_plus_large",
        )
    raise ValueError(f"unknown evaluator: {name}")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def _cache_file(cache_dir: Path, evaluator: str, split: str) -> Path:
    safe_name = str(evaluator).replace("/", "_").replace("\\", "_")
    safe_split = str(split).replace("/", "_").replace("\\", "_")
    return cache_dir / f"{safe_name}__{safe_split}.npz"


def _load_cached_outputs(
    path: Path,
    *,
    expected_paths: Sequence[str],
    fingerprint: str,
) -> Optional[EvaluatorOutputs]:
    if not path.is_file():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        cached_paths = [str(value) for value in data["paths"].tolist()]
        if cached_paths != [str(value) for value in expected_paths]:
            return None
        if str(data["fingerprint"].item()) != str(fingerprint):
            return None
        predicted = [
            value if value else None for value in data["predicted"].tolist()
        ]
        return EvaluatorOutputs(
            paths=cached_paths,
            features=np.asarray(data["features"], dtype=np.float32),
            probs=np.asarray(data["probs"], dtype=np.float32),
            model_labels=[str(value) for value in data["model_labels"].tolist()],
            predicted_raw=[str(value) for value in data["predicted_raw"].tolist()],
            predicted=predicted,
            feature_kind=str(data["feature_kind"].item()),
        )
    except Exception:
        return None


def _save_cached_outputs(
    path: Path,
    outputs: EvaluatorOutputs,
    *,
    fingerprint: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        paths=np.asarray(outputs.paths, dtype=str),
        features=np.asarray(outputs.features, dtype=np.float32),
        probs=np.asarray(outputs.probs, dtype=np.float32),
        model_labels=np.asarray(outputs.model_labels, dtype=str),
        predicted_raw=np.asarray(outputs.predicted_raw, dtype=str),
        predicted=np.asarray(
            [value if value is not None else "" for value in outputs.predicted],
            dtype=str,
        ),
        feature_kind=np.asarray(outputs.feature_kind, dtype=str),
        fingerprint=np.asarray(fingerprint, dtype=str),
    )


def get_outputs(
    evaluator: BaseEvaluator,
    *,
    split_name: str,
    paths: Sequence[Path],
    cache_dir: Path,
    batch_size: int,
    force: bool = False,
) -> EvaluatorOutputs:
    expected_paths = [str(Path(path)) for path in paths]
    cache_path = _cache_file(cache_dir, evaluator.name, split_name)
    fingerprint = evaluator.fingerprint()
    if not force:
        cached = _load_cached_outputs(
            cache_path,
            expected_paths=expected_paths,
            fingerprint=fingerprint,
        )
        if cached is not None:
            print(
                f"[{evaluator.name}] {split_name}: loaded {len(cached.paths)} "
                f"cached rows",
                flush=True,
            )
            return cached
    print(
        f"[{evaluator.name}] {split_name}: measuring {len(paths)} files ...",
        flush=True,
    )
    started = time.time()
    outputs = evaluator.predict_files(paths, batch_size=batch_size)
    if outputs.paths != expected_paths:
        raise RuntimeError(
            f"evaluator returned paths in an unexpected order for {split_name}"
        )
    _save_cached_outputs(cache_path, outputs, fingerprint=fingerprint)
    print(
        f"[{evaluator.name}] {split_name}: done in {time.time() - started:.1f}s",
        flush=True,
    )
    return outputs


# ---------------------------------------------------------------------------
# Metrics and gates
# ---------------------------------------------------------------------------


def _prediction_metrics(
    rows: pd.DataFrame,
    outputs: EvaluatorOutputs,
    *,
    labels: Sequence[str],
    n_resamples: int,
    seed: int,
) -> Dict[str, Any]:
    if len(rows) != len(outputs.predicted):
        raise ValueError("rows and evaluator outputs have different lengths")
    label_list = [str(label) for label in labels]
    y_true = [str(value) for value in rows["label"].tolist()]
    y_pred = [
        value if value is not None else "__out_of_scope__"
        for value in outputs.predicted
    ]
    groups = [str(value) for value in rows["speaker_id"].tolist()]
    metrics = classifier_metrics(
        y_true,
        y_pred,
        groups,
        labels=label_list,
        n_resamples=int(n_resamples),
        confidence=0.95,
        seed=int(seed),
    )
    if metrics.get("status") != "computed":
        return metrics
    counts = Counter(y_pred)
    normalized_counts = Counter(
        value for value in outputs.predicted if value is not None
    )
    total = max(1, len(y_pred))
    metrics["predicted_counts"] = dict(sorted(counts.items()))
    metrics["n_distinct_predictions"] = len(normalized_counts)
    metrics["majority_prediction_fraction"] = (
        max(counts.values()) / float(total) if counts else None
    )
    metrics["out_of_scope_predictions"] = int(
        sum(value is None for value in outputs.predicted)
    )
    metrics["out_of_scope_rate"] = (
        metrics["out_of_scope_predictions"] / float(total)
    )
    return metrics


def _task_labels_for_rows(rows: pd.DataFrame, expected: Sequence[str]) -> List[str]:
    present = set(str(value) for value in rows["label"].tolist())
    return [str(label) for label in expected if label in present]


def panel_gate(
    *,
    real_metrics: Mapping[str, Any],
    domain_shift: Mapping[str, Any],
    labels: Sequence[str],
    n_resamples: int,
) -> Dict[str, Any]:
    reasons: List[str] = []
    chance = 1.0 / max(1, len(labels))
    minimum_accuracy = chance + MIN_REAL_ACCURACY_MARGIN
    if real_metrics.get("status") != "computed":
        reasons.append("real-speech metrics are unavailable")
    else:
        accuracy = float(real_metrics.get("accuracy", 0.0))
        if int(real_metrics.get("n", 0)) < MIN_REAL_ROWS:
            reasons.append(
                f"only {int(real_metrics.get('n', 0))} real evaluation rows; "
                f"at least {MIN_REAL_ROWS} are required"
            )
        if accuracy < minimum_accuracy:
            reasons.append(
                f"real-speech accuracy {accuracy:.3f} is below the "
                f"chance-plus-margin floor {minimum_accuracy:.3f}"
            )
        supports = [
            int(entry.get("support", 0))
            for entry in (real_metrics.get("per_class") or {}).values()
        ]
        if supports and min(supports) < MIN_REAL_PER_CLASS:
            reasons.append(
                f"the smallest real per-class support is {min(supports)}; "
                f"at least {MIN_REAL_PER_CLASS} is required"
            )
        if int(real_metrics.get("n_distinct_predictions", 0)) < 2:
            reasons.append("real-speech predictions collapsed to one class")
        majority = real_metrics.get("majority_prediction_fraction")
        if majority is not None and float(majority) > MAX_MAJORITY_PREDICTION_FRACTION:
            reasons.append(
                "one predicted class accounts for "
                f"{float(majority):.1%} of real-speech predictions"
            )
    if domain_shift.get("status") != "computed":
        reasons.append("domain-shift calibration is unavailable")
    elif not domain_shift.get("valid"):
        reasons.append(
            "generated features are outside the real-speech domain "
            f"(ratio {float(domain_shift.get('ratio', float('nan'))):.2f} "
            f"> {float(domain_shift.get('threshold', DOMAIN_SHIFT_THRESHOLD)):.2f})"
        )
    return {
        "valid": not reasons,
        "reasons": reasons,
        "chance_accuracy": chance,
        "minimum_real_accuracy": minimum_accuracy,
        "domain_shift_threshold": DOMAIN_SHIFT_THRESHOLD,
        "n_resamples": int(n_resamples),
    }


def _domain_shift_block(
    train_outputs: EvaluatorOutputs,
    real_outputs: EvaluatorOutputs,
    generated_outputs: EvaluatorOutputs,
    *,
    threshold: float,
) -> Dict[str, Any]:
    result = feature_domain_shift(
        train_outputs.features,
        real_outputs.features,
        generated_outputs.features,
        threshold=float(threshold),
    )
    result["train_rows"] = int(len(train_outputs.paths))
    result["real_rows"] = int(len(real_outputs.paths))
    result["generated_rows"] = int(len(generated_outputs.paths))
    result["feature_kind"] = generated_outputs.feature_kind
    result["feature_dim"] = int(
        np.asarray(generated_outputs.features).shape[1]
        if np.asarray(generated_outputs.features).ndim == 2
        else 0
    )
    return result


def _confidence_series(outputs: EvaluatorOutputs) -> List[Optional[float]]:
    probs = np.asarray(outputs.probs)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return [None] * len(outputs.paths)
    return [float(value) for value in probs.max(axis=1)]


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def _environment_block(device: str) -> Dict[str, Any]:
    import platform

    import numpy as np
    import sklearn
    import torch
    import transformers

    block: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        import speechbrain

        block["speechbrain"] = speechbrain.__version__
    except Exception:
        block["speechbrain"] = None
    try:
        import huggingface_hub

        block["huggingface_hub"] = huggingface_hub.__version__
    except Exception:
        block["huggingface_hub"] = None
    try:
        import funasr

        block["funasr"] = getattr(funasr, "__version__", "installed")
    except Exception:
        block["funasr"] = None
    try:
        import modelscope

        block["modelscope"] = modelscope.__version__
    except Exception:
        block["modelscope"] = None
    if torch.cuda.is_available():
        block["gpu"] = torch.cuda.get_device_name(0)
    return block


def _predictions_frame(
    *,
    evaluator: str,
    task: str,
    split: str,
    rows: pd.DataFrame,
    outputs: EvaluatorOutputs,
) -> pd.DataFrame:
    confidences = _confidence_series(outputs)
    return pd.DataFrame(
        {
            "evaluator": evaluator,
            "task": task,
            "split": split,
            "clip_id": rows["clip_id"].astype(str).tolist(),
            "speaker_id": rows["speaker_id"].astype(str).tolist(),
            "true_label": rows["label"].astype(str).tolist(),
            "predicted_label": [
                value if value is not None else "out_of_scope"
                for value in outputs.predicted
            ],
            "predicted_raw": outputs.predicted_raw,
            "confidence": confidences,
            "wav_path": [str(path) for path in rows["wav_path"].tolist()],
        }
    )


def _evaluator_summary(
    *,
    name: str,
    task: str,
    labels: Sequence[str],
    real_metrics: Mapping[str, Any],
    generated_metrics: Mapping[str, Any],
    domain_shift: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> Dict[str, Any]:
    valid = bool(gate.get("valid"))
    summary: Dict[str, Any] = {
        "evaluator": name,
        "task": task,
        "labels": list(labels),
        "panel_valid": valid,
        "gate": dict(gate),
        "real": {
            "accuracy": real_metrics.get("accuracy"),
            "macro_f1": real_metrics.get("macro_f1"),
            "n": real_metrics.get("n"),
            "n_speakers": real_metrics.get("n_speakers"),
            "speaker_cluster_ci": real_metrics.get("speaker_cluster_ci"),
            "per_class": real_metrics.get("per_class"),
            "confusion_matrix": real_metrics.get("confusion_matrix"),
            "predicted_counts": real_metrics.get("predicted_counts"),
            "out_of_scope_predictions": real_metrics.get(
                "out_of_scope_predictions"
            ),
        },
        "domain_shift": dict(domain_shift),
        "generated": {
            "accuracy": generated_metrics.get("accuracy"),
            "macro_f1": generated_metrics.get("macro_f1"),
            "n": generated_metrics.get("n"),
            "n_speakers": generated_metrics.get("n_speakers"),
            "per_class": generated_metrics.get("per_class"),
            "confusion_matrix": generated_metrics.get("confusion_matrix"),
            "predicted_counts": generated_metrics.get("predicted_counts"),
            "out_of_scope_predictions": generated_metrics.get(
                "out_of_scope_predictions"
            ),
            "majority_prediction_fraction": generated_metrics.get(
                "majority_prediction_fraction"
            ),
        },
    }
    if valid:
        summary["interpretation"] = (
            "The evaluator passed its real-speech and domain-shift gates. "
            "Generated accuracy is a measurable external-classifier result, "
            "not a human perceptual result."
        )
    else:
        summary["interpretation"] = (
            "The evaluator did not pass its real-speech/domain-shift gates. "
            "Generated predictions are reported for audit only and do not "
            "support an accent or emotion claim."
        )
    return summary


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


def write_readme(path: Path, report: Mapping[str, Any]) -> None:
    lines: List[str] = [
        "# External evaluator panel",
        "",
        "This directory contains an **evaluation-only** run of frozen external "
        "recognizers on real holdout audio and generated VOIX demo audio.",
        "The recognizers are not controllers and no result was fed back into "
        "synthesis.",
        "",
        "> No MOS, CMOS, human emotion-identification, or human "
        "accent-identification study was run. All numbers below are model "
        "outputs on this specific audio.",
        "",
        "## Run",
        "",
        f"- Created: {report['created_at']}",
        f"- Demo: `{report['inputs']['generated']['demo_dir']}`",
        f"- Generated clips: {report['inputs']['generated']['n_selected']}",
        f"- Device: `{report['environment']['device']}`",
        f"- Domain-shift threshold: {report['config']['domain_shift_threshold']:.2f}",
        "",
        "```text",
        report["command"],
        "```",
        "",
        "## Data",
        "",
    ]
    dataset_rows = []
    for key in ("accent_train", "accent_test", "emotion_train", "emotion_test"):
        entry = report["inputs"][key]
        dataset_rows.append(
            [
                key,
                entry.get("n_selected"),
                entry.get("n_speakers"),
                ", ".join(
                    f"{label}={count}"
                    for label, count in (entry.get("class_counts") or {}).items()
                ),
            ]
        )
    lines.extend(_markdown_table(["Split", "Clips", "Speakers", "Classes"], dataset_rows))
    lines.extend(["", "## Panel results", ""])
    summary_rows = []
    for name, entry in report["evaluators"].items():
        if entry.get("status") != "computed":
            summary_rows.append(
                [name, entry.get("task"), "unavailable", "-", "-", "-", "-"]
            )
            continue
        real = entry["summary"]["real"]
        generated = entry["summary"]["generated"]
        gate = entry["summary"]["gate"]
        summary_rows.append(
            [
                name,
                entry.get("task"),
                _format_ci(
                    {
                        "status": "computed",
                        "accuracy": real.get("accuracy"),
                        "speaker_cluster_ci": real.get("speaker_cluster_ci"),
                    }
                )
                if real.get("accuracy") is not None
                else "unavailable",
                "-"
                if real.get("macro_f1") is None
                else f"{float(real['macro_f1']):.3f}",
                "-"
                if entry["summary"]["domain_shift"].get("ratio") is None
                else f"{float(entry['summary']['domain_shift']['ratio']):.2f}",
                "yes" if gate.get("valid") else "no",
                "-"
                if generated.get("accuracy") is None
                else f"{float(generated['accuracy']):.3f}",
                generated.get("n"),
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "Evaluator",
                "Task",
                "Real accuracy [95% CI]",
                "Real macro-F1",
                "Shift ratio",
                "Gate",
                "Generated accuracy",
                "Generated n",
            ],
            summary_rows,
        )
    )
    lines.extend(["", "## Gates and interpretation", ""])
    for name, entry in report["evaluators"].items():
        lines.append(f"### {name}")
        lines.append("")
        if entry.get("status") != "computed":
            lines.append(f"Status: **unavailable**. {entry.get('reason', '')}")
            lines.append("")
            continue
        summary = entry["summary"]
        lines.append(
            f"Panel valid: **{'yes' if summary['panel_valid'] else 'no'}**."
        )
        if summary["gate"].get("reasons"):
            lines.append("")
            lines.append("Reasons:")
            for reason in summary["gate"]["reasons"]:
                lines.append(f"- {reason}")
        lines.append("")
        lines.append(summary["interpretation"])
        lines.append("")
        per_class_rows = []
        real_per_class = summary["real"].get("per_class") or {}
        generated_per_class = summary["generated"].get("per_class") or {}
        for label in summary["labels"]:
            real_entry = real_per_class.get(label, {})
            generated_entry = generated_per_class.get(label, {})
            per_class_rows.append(
                [
                    label,
                    real_entry.get("support", 0),
                    f"{float(real_entry.get('recall', 0.0)):.3f}",
                    generated_entry.get("support", 0),
                    f"{float(generated_entry.get('recall', 0.0)):.3f}",
                ]
            )
        lines.extend(
            _markdown_table(
                [
                    "Label",
                    "Real support",
                    "Real recall",
                    "Generated support",
                    "Generated recall",
                ],
                per_class_rows,
            )
        )
        lines.append("")
    lines.extend(
        [
            "## Limitations",
            "",
            "- CommonAccent is a 16-class model. Its accuracy on the four VOIX "
            "classes is not the published 16-class accuracy and is reported "
            "only for this subset.",
            "- GLOBE labels are self-reported; the evaluator uses the "
            "speaker-disjoint test split.",
            "- CREMA-D is acted speech. Fear and disgust were excluded because "
            "VOIX has no validated direction for them.",
            "- The generated set has three voices and a small number of target "
            "clips. Generated accuracy is a sanity check, not a "
            "speaker-generalization estimate.",
            "- A failed domain-shift gate means the recognizer cannot measure "
            "the generated category. It does not prove the generated control "
            "is wrong.",
            "- emotion2vec+ uses the FunASR model license (`other`), not an "
            "OSI-approved license. Review redistribution terms before "
            "shipping model files.",
            "- The panel does not replace a listener study. No perceptual "
            "claim is made.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, default=DEFAULT_DEMO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--globe-manifest", type=Path, default=DEFAULT_GLOBE_MANIFEST)
    parser.add_argument(
        "--globe-audio-root", type=Path, default=DEFAULT_GLOBE_AUDIO_ROOT
    )
    parser.add_argument("--crema-manifest", type=Path, default=DEFAULT_CREMA_MANIFEST)
    parser.add_argument(
        "--evaluators",
        nargs="+",
        default=["commonaccent", "superb_er", "emotion2vec"],
        help="Any of: commonaccent, superb_er, emotion2vec, all",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--train-per-class", type=int, default=80)
    parser.add_argument(
        "--max-test-per-class",
        type=int,
        default=0,
        help="0 keeps all held-out test clips; positive values subsample.",
    )
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--domain-shift-threshold", type=float, default=DOMAIN_SHIFT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifests and print split sizes without loading models.",
    )
    return parser.parse_args(argv)


def _resolve_device(requested: str) -> str:
    import torch

    value = str(requested).strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return value


def _normalize_evaluator_names(values: Sequence[str]) -> List[str]:
    names: List[str] = []
    for value in values:
        key = str(value).strip().lower()
        if key == "all":
            for candidate in ("commonaccent", "superb_er", "emotion2vec"):
                if candidate not in names:
                    names.append(candidate)
            continue
        if key == "superb":
            key = "superb_er"
        if key in {"commonaccent", "superb_er", "emotion2vec"} and key not in names:
            names.append(key)
    return names


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    started = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"

    print("[panel] loading manifests", flush=True)
    accent_train, accent_train_info = build_globe_rows(
        args.globe_manifest,
        args.globe_audio_root,
        speaker_split="train",
        max_per_class=int(args.train_per_class),
        seed=int(args.seed),
    )
    accent_test, accent_test_info = build_globe_rows(
        args.globe_manifest,
        args.globe_audio_root,
        speaker_split="test",
        max_per_class=int(args.max_test_per_class),
        seed=int(args.seed) + 1,
    )
    emotion_train, emotion_train_info = build_crema_rows(
        args.crema_manifest,
        split="train",
        max_per_class=int(args.train_per_class),
        seed=int(args.seed),
    )
    emotion_test, emotion_test_info = build_crema_rows(
        args.crema_manifest,
        split="test",
        max_per_class=int(args.max_test_per_class),
        seed=int(args.seed) + 1,
    )
    generated, generated_info = build_generated_rows(args.demo)
    generated_accent, generated_accent_info = generated_task_rows(generated, "accent")
    generated_emotion, generated_emotion_info = generated_task_rows(
        generated, "emotion"
    )

    inputs = {
        "accent_train": accent_train_info,
        "accent_test": accent_test_info,
        "emotion_train": emotion_train_info,
        "emotion_test": emotion_test_info,
        "generated": generated_info,
        "generated_accent": generated_accent_info,
        "generated_emotion": generated_emotion_info,
    }
    evaluator_names = _normalize_evaluator_names(args.evaluators)
    if not evaluator_names:
        raise ValueError("no evaluators selected")

    if args.dry_run:
        print(json.dumps(_jsonable(inputs), indent=2))
        print(f"[panel] evaluators: {evaluator_names}")
        print(f"[panel] output: {output_dir}")
        return 0

    for split_name, split in (
        ("accent_train", accent_train),
        ("accent_test", accent_test),
        ("emotion_train", emotion_train),
        ("emotion_test", emotion_test),
    ):
        if split.empty:
            raise ValueError(f"{split_name} has no usable clips")
    required_generated = {
        f"generated_{EVALUATOR_TASKS[name]}"
        for name in evaluator_names
        if name in EVALUATOR_TASKS
    }
    generated_tracks = {
        "generated_accent": generated_accent,
        "generated_emotion": generated_emotion,
    }
    for track_name in sorted(required_generated):
        if generated_tracks[track_name].empty:
            raise ValueError(f"{track_name} has no usable clips")

    device = _resolve_device(args.device)
    print(f"[panel] device={device} evaluators={evaluator_names}", flush=True)
    prediction_frames: List[pd.DataFrame] = []
    evaluator_reports: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}

    for name in evaluator_names:
        print(f"[panel] starting {name}", flush=True)
        try:
            evaluator = build_evaluator(
                name,
                device=device,
                model_cache=Path(args.model_cache_dir).resolve(),
            )
        except Exception as exc:
            evaluator_reports[name] = {
                "status": "unavailable",
                "task": "accent" if name == "commonaccent" else "emotion",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            print(f"[panel] {name} unavailable: {exc}", flush=True)
            continue

        task = evaluator.task
        try:
            if task == "accent":
                labels = _task_labels_for_rows(accent_test, ACCENT_LABELS)
                generated_rows = generated_accent
                generated_labels = _task_labels_for_rows(
                    generated_accent, ACCENT_LABELS
                )
                train_rows = accent_train
                real_rows = accent_test
            else:
                labels = _task_labels_for_rows(emotion_test, EMOTION_LABELS)
                generated_rows = generated_emotion
                generated_labels = _task_labels_for_rows(
                    generated_emotion, EMOTION_LABELS
                )
                train_rows = emotion_train
                real_rows = emotion_test

            if generated_rows.empty:
                evaluator_reports[name] = {
                    "status": "unavailable",
                    "task": task,
                    "reason": "the demo index has no generated rows for this task",
                }
                print(
                    f"[panel] {name} skipped: no generated {task} rows",
                    flush=True,
                )
                continue

            train_outputs = get_outputs(
                evaluator,
                split_name=f"{task}_train",
                paths=train_rows["wav_path"].astype(str).tolist(),
                cache_dir=cache_dir,
                batch_size=int(args.batch_size),
                force=bool(args.force_recompute),
            )
            real_outputs = get_outputs(
                evaluator,
                split_name=f"{task}_test",
                paths=real_rows["wav_path"].astype(str).tolist(),
                cache_dir=cache_dir,
                batch_size=int(args.batch_size),
                force=bool(args.force_recompute),
            )
            generated_outputs_all = get_outputs(
                evaluator,
                split_name="generated",
                paths=generated["wav_path"].astype(str).tolist(),
                cache_dir=cache_dir,
                batch_size=int(args.batch_size),
                force=bool(args.force_recompute),
            )

            generated_index = {
                str(path): index
                for index, path in enumerate(generated_outputs_all.paths)
            }
            target_indices = [
                generated_index[str(path)]
                for path in generated_rows["wav_path"].astype(str).tolist()
            ]
            generated_outputs = generated_outputs_all.subset(target_indices)

            real_metrics = _prediction_metrics(
                real_rows,
                real_outputs,
                labels=labels,
                n_resamples=int(args.n_resamples),
                seed=int(args.seed),
            )
            generated_metrics = _prediction_metrics(
                generated_rows,
                generated_outputs,
                labels=generated_labels,
                n_resamples=int(args.n_resamples),
                seed=int(args.seed) + 2,
            )
            domain_shift = _domain_shift_block(
                train_outputs,
                real_outputs,
                generated_outputs,
                threshold=float(args.domain_shift_threshold),
            )
            gate = panel_gate(
                real_metrics=real_metrics,
                domain_shift=domain_shift,
                labels=labels,
                n_resamples=int(args.n_resamples),
            )
            baseline = majority_baseline(
                train_rows["label"].astype(str).tolist(),
                real_rows["label"].astype(str).tolist(),
            )
            real_metrics["majority_baseline"] = {
                "label": baseline["label"],
                "accuracy": baseline["accuracy"],
                "train_support": baseline["train_support"],
            }

            summary_entry = _evaluator_summary(
                name=name,
                task=task,
                labels=labels,
                real_metrics=real_metrics,
                generated_metrics=generated_metrics,
                domain_shift=domain_shift,
                gate=gate,
            )
            evaluator_reports[name] = {
                "status": "computed",
                "task": task,
                "model": {
                    "id": evaluator.model_id,
                    "role": "evaluation_only",
                    "controls_synthesis": False,
                    "feature_kind": generated_outputs.feature_kind,
                    "feature_dim": int(generated_outputs.features.shape[1])
                    if generated_outputs.features.ndim == 2
                    else 0,
                    "artifacts": evaluator.artifacts()
                    if hasattr(evaluator, "artifacts")
                    else [],
                },
                "input_counts": {
                    "train": int(len(train_rows)),
                    "real": int(len(real_rows)),
                    "generated": int(len(generated_rows)),
                },
                "real": real_metrics,
                "generated": generated_metrics,
                "domain_shift": domain_shift,
                "summary": summary_entry,
            }
            prediction_frames.append(
                _predictions_frame(
                    evaluator=name,
                    task=task,
                    split=f"{task}_test",
                    rows=real_rows,
                    outputs=real_outputs,
                )
            )
            prediction_frames.append(
                _predictions_frame(
                    evaluator=name,
                    task=task,
                    split=f"{task}_generated",
                    rows=generated_rows,
                    outputs=generated_outputs,
                )
            )
            summary[name] = summary_entry
        finally:
            try:
                del evaluator
            except Exception:
                pass
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": stamp,
        "role": "independent_external_evaluation",
        "controls_synthesis": False,
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "config": {
            "evaluators": evaluator_names,
            "seed": int(args.seed),
            "train_per_class": int(args.train_per_class),
            "max_test_per_class": int(args.max_test_per_class),
            "n_resamples": int(args.n_resamples),
            "domain_shift_threshold": float(args.domain_shift_threshold),
            "batch_size": int(args.batch_size),
            "device_requested": str(args.device),
            "device_resolved": device,
            "model_cache_dir": str(Path(args.model_cache_dir).resolve()),
        },
        "environment": _environment_block(device),
        "inputs": inputs,
        "datasets": {
            "globe": _artifact_record(Path(args.globe_manifest)),
            "crema_d": _artifact_record(Path(args.crema_manifest)),
            "demo_index": _artifact_record(Path(args.demo) / "index.csv"),
            "demo_report": _artifact_record(Path(args.demo) / "report.json"),
        },
        "evaluators": evaluator_reports,
        "summary": summary,
        "limitations": [
            "External recognizers are proxies, not listeners.",
            "CommonAccent is a 16-class model evaluated on a four-class subset.",
            "GLOBE accents are self-reported and the test split is speaker-disjoint.",
            "CREMA-D is acted speech; fear and disgust are excluded.",
            "The generated set has three voices; generated accuracy is not a "
            "speaker-generalization estimate.",
            "A failed domain-shift gate invalidates the generated prediction "
            "comparison; it does not prove the synthesis control is wrong.",
            "emotion2vec+ uses the FunASR model license (other); review "
            "redistribution terms.",
            "No MOS, CMOS, or human identification study was run.",
        ],
    }

    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
        predictions.to_csv(output_dir / "predictions.csv", index=False)
        report["predictions_csv"] = str(output_dir / "predictions.csv")
    else:
        report["predictions_csv"] = None

    _write_json(output_dir / "report.json", report)
    write_readme(output_dir / "README.md", report)
    print(
        f"[panel] wrote {output_dir / 'report.json'} in "
        f"{time.time() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
