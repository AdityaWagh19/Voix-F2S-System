"""Tests for the Phase 1 face-encoder ceiling audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "audit" / "compare_face_encoder_ceiling.py"

spec = importlib.util.spec_from_file_location("compare_face_encoder_ceiling", SCRIPT)
assert spec and spec.loader
ceiling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ceiling)


class TestSpeakerAggregation:
    def test_speaker_centroids_are_stable_and_averaged(self):
        values = np.asarray([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]], dtype=np.float32)

        centroids, names = ceiling._speaker_centroids(
            values, np.asarray(["b", "b", "a"])
        )

        assert names == ["a", "b"]
        np.testing.assert_allclose(centroids, [[0.0, 2.0], [2.0, 0.0]])

    def test_normalize_centroids_only_touches_embedding_blocks(self):
        values = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)

        normalized = ceiling._normalize_centroids(values, "cfvn_es")
        untouched = ceiling._normalize_centroids(values, "fused")

        np.testing.assert_allclose(
            np.linalg.norm(normalized, axis=1), [1.0, 1.0], rtol=1e-6
        )
        np.testing.assert_allclose(untouched, values)


class TestSpeakerTargets:
    def test_returns_one_row_per_speaker_in_requested_order(self):
        targets = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        result = ceiling._speaker_targets(
            targets, np.asarray(["b", "b", "a"]), ["a", "b"]
        )

        torch.testing.assert_close(result, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))

    def test_rejects_within_speaker_target_variation(self):
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        with pytest.raises(ValueError, match="varies within speaker"):
            ceiling._speaker_targets(targets, np.asarray(["a", "a"]), ["a"])

    def test_rejects_speaker_order_mismatch(self):
        targets = torch.tensor([[1.0, 0.0]])

        with pytest.raises(ValueError, match="speaker order mismatch"):
            ceiling._speaker_targets(targets, np.asarray(["a"]), ["b"])


class TestResidualMetrics:
    def test_perfect_prediction_has_unit_residual_and_target_cosine(self):
        center = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([[2.0, 2.0, 3.0], [1.0, 4.0, 3.0]])

        metrics = ceiling._residual_metrics(target.clone(), target, center)

        assert metrics["residual_cosine"] == pytest.approx(1.0)
        assert metrics["target_cosine"] == pytest.approx(1.0)
        assert metrics["residual_norm_ratio"] == pytest.approx(1.0)
        assert metrics["n"] == 2


class TestPairTail:
    def test_same_speaker_pairs_are_masked(self):
        center = torch.zeros(2)
        predictions = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        )
        codes = torch.tensor([0, 0, 1])

        stats = ceiling._centered_pair_tail(predictions, center, codes)

        assert stats["n_pairs"] == 4
        assert stats["mean"] == pytest.approx(0.0)
        assert stats["max"] == pytest.approx(0.0)
        assert stats["frac_gt_0.5"] == pytest.approx(0.0)

    def test_tail_fractions_count_directed_pairs(self):
        center = torch.zeros(3)
        predictions = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        codes = torch.tensor([0, 1, 2])

        stats = ceiling._centered_pair_tail(predictions, center, codes)

        # Three different-speaker pairs, counted in both directions.
        assert stats["n_pairs"] == 6
        assert stats["max"] == pytest.approx(1.0)
        assert stats["frac_gt_0.8"] == pytest.approx(1.0 / 3.0)


class TestRidge:
    def test_recovers_a_linear_map(self):
        torch.manual_seed(0)
        x = torch.randn(120, 4)
        weights = torch.randn(4, 3)
        offset = torch.tensor([1.0, -2.0, 0.5])
        y = x @ weights + offset

        model = ceiling._fit_standardized_ridge(
            x[:80], y[:80], x[80:], y[80:], y[:80].mean(0), [0.01, 0.1, 1.0]
        )
        prediction = ceiling._apply_ridge(x[80:], model, y[:80].mean(0))

        assert model["best"]["val"]["residual_cosine"] > 0.999
        assert torch.allclose(prediction, y[80:], atol=5e-3)

    def test_selection_is_the_highest_validation_residual_cosine(self):
        torch.manual_seed(1)
        x = torch.randn(90, 3)
        y = x @ torch.randn(3, 2)

        model = ceiling._fit_standardized_ridge(
            x[:60], y[:60], x[60:], y[60:], y[:60].mean(0), [0.1, 1.0, 100.0]
        )
        best = max(
            row["val"]["residual_cosine"] for row in model["rows"]
        )

        assert model["best"]["val"]["residual_cosine"] == pytest.approx(best)
        assert model["best"]["lambda"] in {0.1, 1.0, 100.0}


def _split(residual: float, p90: float, frac_gt_0_8: float, target: float) -> dict:
    return {
        "residual_cosine": residual,
        "target_cosine": target,
        "pair_tail": {"p90": p90, "frac_gt_0.8": frac_gt_0_8},
    }


def _representation(val: dict, test: dict) -> dict:
    return {"splits": {"val": val, "test": test}}


def _demo(watch: dict, gate_max: float) -> dict:
    baseline = {
        key: {"variants": {"cfvn_es": {"centered_cosine": 0.60}}}
        for key in watch
    }
    candidate = {
        key: {"variants": {"candidate": {"centered_cosine": value}}}
        for key, value in watch.items()
    }
    return {
        "baseline_watch_pairs": baseline,
        "candidate_watch_pairs": candidate,
        "candidate_gate_summary": {"candidate": {"centered_max": gate_max}},
    }


class TestDecisionMatrix:
    def _baseline(self) -> dict:
        return _representation(
            _split(0.40, 0.80, 0.10, 0.90),
            _split(0.40, 0.80, 0.10, 0.90),
        )

    def _candidate(self, gain: float = 0.06) -> dict:
        return _representation(
            _split(0.40 + gain, 0.80 - gain, 0.05, 0.895),
            _split(0.40 + gain, 0.80 - gain, 0.05, 0.895),
        )

    def test_go_requires_linear_and_demo_gates(self):
        decision = ceiling._decision(
            self._baseline(),
            "candidate",
            self._candidate(),
            _demo({"C1_03/C1_05": 0.35, "C1_04/C2_15": 0.40}, 0.50),
        )

        assert decision["status"] == "go"
        assert decision["linear_pass"] is True
        assert decision["demo_pass"] is True

    def test_conditional_allows_one_retrain_when_demo_max_misses(self):
        decision = ceiling._decision(
            self._baseline(),
            "candidate",
            self._candidate(),
            _demo({"C1_03/C1_05": 0.35, "C1_04/C2_15": 0.40}, 0.65),
        )

        assert decision["status"] == "conditional"
        assert decision["linear_pass"] is True
        assert decision["demo_pass"] is False

    def test_failed_linear_gate_is_no_go(self):
        decision = ceiling._decision(
            self._baseline(),
            "candidate",
            self._candidate(gain=0.01),
            _demo({"C1_03/C1_05": 0.35, "C1_04/C2_15": 0.40}, 0.50),
        )

        assert decision["status"] == "no_go"
        assert decision["linear_pass"] is False

    def test_watch_pair_regression_is_no_go(self):
        decision = ceiling._decision(
            self._baseline(),
            "candidate",
            self._candidate(),
            _demo({"C1_03/C1_05": 0.35, "C1_04/C2_15": 0.75}, 0.65),
        )

        assert decision["status"] == "no_go"

    def test_watch_improvement_below_threshold_is_no_go(self):
        decision = ceiling._decision(
            self._baseline(),
            "candidate",
            self._candidate(),
            _demo({"C1_03/C1_05": 0.35, "C1_04/C2_15": 0.52}, 0.65),
        )

        assert decision["status"] == "no_go"


class TestIdentityOverlap:
    def test_flags_a_nearest_train_speaker(self):
        rows = {"demo": {"arcface": [1.0, 0.0]}}
        train = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        result = ceiling._identity_overlap(
            rows, train, np.asarray(["train_a", "train_b"])
        )

        assert result["demo"]["top1"]["speaker"] == "train_a"
        assert result["demo"]["top1"]["cosine"] == pytest.approx(1.0)
        assert result["demo"]["possible_overlap"] is True
        assert result["demo"]["n_train_faces_ge_0.6"] == 1
