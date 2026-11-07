"""Unit tests for the N0b selection and real-metric helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "evaluation" / "run_n0b_prosody_prototype.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_n0b_prosody_prototype", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_metrics_keep_clip_alignment_and_report_missing_values():
    module = _load_module()
    cache = {
        "clip_id": np.asarray(["a", "b"], dtype=str),
        "f0_median_hz": np.asarray([120.0, np.nan], dtype=np.float32),
        "words_per_second": np.asarray([4.0, 5.0], dtype=np.float32),
    }
    metrics = module._real_metrics_by_clip(
        cache, metrics=("f0_median_hz", "words_per_second", "pause_ratio")
    )
    assert metrics["a"]["f0_median_hz"] == 120.0
    assert metrics["a"]["words_per_second"] == 4.0
    assert metrics["a"]["pause_ratio"] is None
    assert metrics["b"]["f0_median_hz"] is None
    assert metrics["b"]["words_per_second"] == 5.0


def test_cosine_normalizes_unnormalized_ecapa_vectors():
    module = _load_module()
    left = np.asarray([3.0, 4.0, 0.0], dtype=np.float32)
    right = np.asarray([6.0, 8.0, 0.0], dtype=np.float32)
    assert np.isclose(module._cosine(left, right), 1.0)
    assert np.isclose(module._cosine(left, -right), -1.0)
    assert module._cosine(left, np.zeros(3, dtype=np.float32)) == 0.0


def test_speaker_balanced_prototype_is_mean_of_speaker_means_and_diagnostic_is_zero():
    module = _load_module()
    style = np.zeros((5, 256), dtype=np.float32)
    # Unequal clip counts: s1 prosody rows [1, 3, 5], s2 rows [9, 9].
    style[0, 128] = 1.0
    style[1, 128] = 3.0
    style[2, 128] = 5.0
    style[3, 128] = 9.0
    style[4, 128] = 9.0
    cache = {
        "style": style,
        "split": np.asarray(["train"] * 5),
        "speaker_id": np.asarray(["s1", "s1", "s1", "s2", "s2"]),
    }
    prototypes = module._build_prototypes(cache)
    balanced = prototypes["speaker_balanced"]["prosody"]
    # Speaker means are 3 and 9, so the speaker-balanced prototype is 6.0,
    # while the clip-weighted mean is 5.4.
    assert np.isclose(balanced[0], 6.0)
    assert np.isclose(prototypes["mean"]["prosody"][0], 5.4)
    assert np.linalg.norm(balanced) > 0.0
    diagnostic = prototypes["speaker_centered_diagnostic"]["prosody"]
    assert np.linalg.norm(diagnostic) < 1e-6
    assert "speaker_centered_diagnostic" not in module.VARIANTS


def test_reusable_index_requires_the_exact_grid_and_existing_audio(tmp_path):
    module = _load_module()
    eval_clips = pd.DataFrame({"clip_id": ["c1", "c2"]})
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"")
    rows = []
    for clip_id in ("c1", "c2"):
        for variant in module.VARIANTS:
            for alpha in module.ALPHAS:
                for seed in (0,):
                    rows.append(
                        {
                            "ami_clip_id": clip_id,
                            "variant": variant,
                            "alpha": alpha,
                            "seed": seed,
                            "speaker_id": "s1",
                            "wav_path": str(wav),
                            **{metric: 1.0 for metric in module.EVAL_METRICS},
                        }
                    )
    index = tmp_path / "clips.csv"
    pd.DataFrame(rows).to_csv(index, index=False)
    expected = 2 * len(module.VARIANTS) * len(module.ALPHAS)
    frame = module._load_reusable_index(
        index, eval_clips, seeds=[0], expected_rows=expected
    )
    assert len(frame) == expected

    broken = pd.DataFrame(rows[:-1])
    broken.to_csv(index, index=False)
    with pytest.raises(ValueError):
        module._load_reusable_index(
            index, eval_clips, seeds=[0], expected_rows=expected
        )


def test_paired_rows_read_speaker_from_the_row_not_the_metric_series():
    module = _load_module()
    frame = pd.DataFrame(
        {
            "ami_clip_id": ["c1", "c1", "c1", "c2", "c2", "c2"],
            "seed": [0, 0, 0, 0, 0, 0],
            "speaker_id": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "variant": ["mean"] * 6,
            "alpha": [0.0, 0.25, 0.5, 0.0, 0.25, 0.5],
            "f0_range_st": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
        }
    )
    pairs = module._paired_difference_rows(
        frame, alpha=0.25, metric="f0_range_st"
    )
    assert len(pairs) == 2
    assert pairs["difference"].tolist() == [1.0, 2.0]
    assert pairs["speaker_id"].tolist() == ["s1", "s2"]

    toward = module._paired_difference_rows(
        frame, alpha=0.25, metric="f0_range_st", target=2.0
    )
    # |2-2|-|1-2| = -1 and |6-2|-|4-2| = 2
    assert toward["difference"].tolist() == [-1.0, 2.0]


def test_gate_counts_distinct_metrics_and_excludes_median_f0():
    module = _load_module()
    toward_ami = {
        "mean": {
            "alpha_0.25": {
                "words_per_second": {"improved": True},
                "f0_range_st": {"improved": True},
                "f0_median_hz": {"improved": True},
            },
            "alpha_0.50": {
                "words_per_second": {"improved": True},
                "energy_std_db": {"improved": False},
            },
        }
    }
    improved = module._gate_improved_metrics(toward_ami, "mean")
    assert improved == ["f0_range_st", "words_per_second"]
    assert module._gate_improved_metrics(toward_ami, "missing") == []


def test_cluster_interval_resamples_speakers_not_rows():
    module = _load_module()
    pairs = pd.DataFrame(
        {
            "speaker_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "difference": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        }
    )
    interval = module._cluster_paired_interval(
        pairs, n_resamples=200, seed=0
    )
    assert interval is not None
    assert interval["n_pairs"] == 6
    assert interval["n_speakers"] == 3
    assert interval["mean_change"] == 2.0
    assert interval["ci_low"] <= interval["mean_change"] <= interval["ci_high"]

    too_few = module._cluster_paired_interval(
        pairs[pairs["speaker_id"] == "s1"], n_resamples=10, seed=0
    )
    assert too_few is None


def test_eval_selection_applies_duration_floor_and_cache_availability():
    module = _load_module()
    manifest = pd.DataFrame(
        {
            "clip_id": ["v1", "v2", "v3", "v4"],
            "speaker_id": ["s1", "s1", "s1", "s1"],
            "split": ["val", "val", "val", "val"],
            "duration_sec": [1.0, 3.0, 5.0, 7.0],
            "text": ["one", "two", "three", "four"],
        }
    )
    selected = module._select_eval_clips(
        manifest,
        2,
        available_clip_ids={"v1", "v2", "v3"},
        min_duration=2.5,
    )
    assert list(selected["clip_id"]) == ["v2", "v3"]


def test_eval_selection_falls_back_to_longest_when_no_clip_meets_floor():
    module = _load_module()
    manifest = pd.DataFrame(
        {
            "clip_id": ["a", "b"],
            "speaker_id": ["s1", "s1"],
            "split": ["val", "val"],
            "duration_sec": [1.0, 1.5],
            "text": ["one", "two"],
        }
    )
    selected = module._select_eval_clips(
        manifest,
        1,
        available_clip_ids={"a", "b"},
        min_duration=2.5,
    )
    assert list(selected["clip_id"]) == ["b"]
