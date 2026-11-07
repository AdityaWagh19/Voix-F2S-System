"""Tests for the external evaluator panel's pure helpers and gates."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "evaluation" / "evaluate_panel.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("evaluate_panel", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


panel = _load_module()


def _touch_audio(root: Path, split: str, label: str, clip_id: str) -> Path:
    path = root / split / label / f"{clip_id}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not real audio")
    return path


def test_accent_label_normalization_covers_globe_and_commonaccent():
    assert panel.normalize_accent_label("us") == "us"
    assert panel.normalize_accent_label("Australia") == "australian"
    assert panel.normalize_accent_label("indian") == "indian_english"
    assert panel.normalize_accent_label("india_south_asia") == "indian_english"
    assert panel.normalize_accent_label("canada") is None
    assert panel.normalize_accent_label(None) is None


def test_emotion_label_normalization_covers_superb_and_crema():
    assert panel.normalize_emotion_label("neu") == "neutral"
    assert panel.normalize_emotion_label("hap") == "happy"
    assert panel.normalize_emotion_label("ang") == "angry"
    assert panel.normalize_emotion_label("sad") == "sad"
    assert panel.normalize_emotion_label("fear") is None
    assert panel.normalize_emotion_label("disgust") is None


def test_emotion2vec_bilingual_labels_are_canonicalized():
    assert panel.canonical_emotion2vec_label("生气/angry") == "angry"
    assert panel.canonical_emotion2vec_label("中立/neutral") == "neutral"
    assert panel.canonical_emotion2vec_label("开心/happy") == "happy"
    assert panel.canonical_emotion2vec_label("难过/sad") == "sad"
    assert panel.canonical_emotion2vec_label("<unk>") is None
    assert panel.canonical_emotion2vec_label("其他/other") == "other"


def test_build_globe_rows_uses_speaker_split_audio_folder(tmp_path):
    audio_root = tmp_path / "globe_audio"
    _touch_audio(audio_root, "test", "us", "us_1")
    _touch_audio(audio_root, "train", "us", "us_2")
    _touch_audio(audio_root, "test", "australian", "au_1")
    manifest = tmp_path / "globe.csv"
    pd.DataFrame(
        [
            {
                "clip_id": "us_1",
                "accent_class": "us",
                "client_id": "speaker_a",
                "speaker_split": "test",
            },
            {
                "clip_id": "us_2",
                "accent_class": "us",
                "client_id": "speaker_b",
                "speaker_split": "train",
            },
            {
                "clip_id": "au_1",
                "accent_class": "australian",
                "client_id": "speaker_c",
                "speaker_split": "test",
            },
            {
                "clip_id": "ca_1",
                "accent_class": "canadian",
                "client_id": "speaker_d",
                "speaker_split": "test",
            },
            {
                "clip_id": "missing",
                "accent_class": "england",
                "client_id": "speaker_e",
                "speaker_split": "test",
            },
        ]
    ).to_csv(manifest, index=False)
    rows, info = panel.build_globe_rows(
        manifest,
        audio_root,
        speaker_split="test",
        max_per_class=0,
        seed=0,
    )
    assert rows["clip_id"].tolist() == ["au_1", "us_1"]
    assert rows["label"].tolist() == ["australian", "us"]
    assert info["missing_by_class"] == {"england": 1}
    assert info["n_selected"] == 2


def test_build_crema_rows_excludes_unvalidated_emotions(tmp_path):
    paths = []
    for clip_id, emotion in (
        ("a", "angry"),
        ("b", "happy"),
        ("c", "neutral"),
        ("d", "sad"),
        ("e", "fear"),
        ("f", "disgust"),
    ):
        path = tmp_path / f"{clip_id}.wav"
        path.write_bytes(b"x")
        paths.append((clip_id, str(path), emotion))
    manifest = tmp_path / "crema.csv"
    pd.DataFrame(
        [
            {
                "clip_id": clip_id,
                "path": path,
                "actor_id": "actor_1",
                "intended_emotion": emotion,
                "split": "test",
            }
            for clip_id, path, emotion in paths
        ]
    ).to_csv(manifest, index=False)
    rows, info = panel.build_crema_rows(
        manifest, split="test", max_per_class=0, seed=0
    )
    assert set(rows["label"]) == {"angry", "happy", "neutral", "sad"}
    assert info["excluded_classes"] == ["fear", "disgust"]
    assert info["n_selected"] == 4


def test_generated_task_rows_split_tracks_and_report_missing_labels(tmp_path):
    demo = tmp_path / "demo"
    demo.mkdir()
    rows = []
    for voice in ("v1", "v2", "v3"):
        for track, condition in (
            ("baseline", "neutral"),
            ("emotion", "happy"),
            ("emotion", "sad"),
            ("emotion", "angry"),
            ("accent", "us"),
            ("accent", "england"),
            ("accent", "australian"),
        ):
            wav = demo / f"{voice}__{track}_{condition}.wav"
            wav.write_bytes(b"x")
            rows.append(
                {
                    "name": f"{voice}__{track}_{condition}",
                    "voice_id": voice,
                    "track": track,
                    "condition": condition,
                    "wav_path": str(wav),
                }
            )
    pd.DataFrame(rows).to_csv(demo / "index.csv", index=False)

    generated, info = panel.build_generated_rows(demo)
    accent_rows, accent_info = panel.generated_task_rows(generated, "accent")
    emotion_rows, emotion_info = panel.generated_task_rows(generated, "emotion")

    assert info["n_selected"] == 21
    assert len(accent_rows) == 9
    assert accent_info["missing_labels"] == ["indian_english"]
    assert len(emotion_rows) == 12
    assert emotion_info["missing_labels"] == []
    assert set(emotion_rows["label"]) == set(panel.EMOTION_LABELS)


def _outputs(predicted, features=None):
    count = len(predicted)
    return panel.EvaluatorOutputs(
        paths=[f"clip_{index}.wav" for index in range(count)],
        features=np.ones((count, 4), dtype=np.float32)
        if features is None
        else np.asarray(features, dtype=np.float32),
        probs=np.full((count, 4), 0.25, dtype=np.float32),
        model_labels=["us", "england", "indian_english", "australian"],
        predicted_raw=[value or "other" for value in predicted],
        predicted=predicted,
    )


def test_prediction_metrics_counts_out_of_scope_as_errors():
    rows = pd.DataFrame(
        {
            "label": ["us", "england", "australian", "indian_english"],
            "speaker_id": ["s1", "s2", "s3", "s4"],
        }
    )
    outputs = _outputs(["us", None, "australian", None])
    outputs.predicted_raw = ["us", "canada", "australian", "other"]
    metrics = panel._prediction_metrics(
        rows,
        outputs,
        labels=panel.ACCENT_LABELS,
        n_resamples=50,
        seed=0,
    )
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["out_of_scope_predictions"] == 2
    assert metrics["predicted_counts"]["__out_of_scope__"] == 2
    assert metrics["n_distinct_predictions"] == 2


def _valid_gate_inputs():
    real = {
        "status": "computed",
        "n": 200,
        "accuracy": 0.60,
        "n_distinct_predictions": 4,
        "majority_prediction_fraction": 0.35,
        "per_class": {
            label: {"support": 50}
            for label in panel.ACCENT_LABELS
        },
    }
    shift = {
        "status": "computed",
        "valid": True,
        "ratio": 1.5,
        "threshold": 3.0,
    }
    return real, shift


def test_panel_gate_accepts_valid_evaluator():
    real, shift = _valid_gate_inputs()
    gate = panel.panel_gate(
        real_metrics=real,
        domain_shift=shift,
        labels=panel.ACCENT_LABELS,
        n_resamples=100,
    )
    assert gate["valid"] is True
    assert gate["reasons"] == []
    assert gate["chance_accuracy"] == pytest.approx(0.25)


def test_panel_gate_rejects_domain_shift_collapse_and_low_accuracy():
    real, shift = _valid_gate_inputs()
    shift = dict(shift, valid=False, ratio=8.0)
    real = dict(real, accuracy=0.20, n_distinct_predictions=1)
    gate = panel.panel_gate(
        real_metrics=real,
        domain_shift=shift,
        labels=panel.ACCENT_LABELS,
        n_resamples=100,
    )
    assert gate["valid"] is False
    joined = " ".join(gate["reasons"])
    assert "domain" in joined
    assert "chance" in joined
    assert "collapsed" in joined


def test_domain_shift_block_reports_feature_metadata():
    rng = np.random.default_rng(0)
    train = panel.EvaluatorOutputs(
        paths=["a", "b", "c"],
        features=rng.normal(size=(3, 8)).astype(np.float32),
        probs=np.zeros((3, 2), dtype=np.float32),
        model_labels=["a", "b"],
        predicted_raw=["a", "b", "a"],
        predicted=["us", "england", "us"],
    )
    real = panel.EvaluatorOutputs(
        paths=["d", "e"],
        features=rng.normal(size=(2, 8)).astype(np.float32),
        probs=np.zeros((2, 2), dtype=np.float32),
        model_labels=["a", "b"],
        predicted_raw=["a", "b"],
        predicted=["us", "england"],
    )
    generated = panel.EvaluatorOutputs(
        paths=["f"],
        features=rng.normal(size=(1, 8)).astype(np.float32),
        probs=np.zeros((1, 2), dtype=np.float32),
        model_labels=["a", "b"],
        predicted_raw=["a"],
        predicted=["us"],
    )
    block = panel._domain_shift_block(
        train, real, generated, threshold=3.0
    )
    assert block["status"] == "computed"
    assert block["train_rows"] == 3
    assert block["real_rows"] == 2
    assert block["generated_rows"] == 1
    assert block["feature_dim"] == 8


def test_parse_args_defaults_are_evaluation_only():
    args = panel.parse_args([])
    assert args.demo.name == "research_v2"
    assert args.output_dir.name == "panel_v2"
    assert args.evaluators == ["commonaccent", "superb_er", "emotion2vec"]
    assert args.n_resamples == 10_000
    assert args.domain_shift_threshold == pytest.approx(3.0)
