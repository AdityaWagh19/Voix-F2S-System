"""Tests for expression-control evaluation metrics.

Everything external (prosody, speaker similarity, classifiers) is injected, so
these tests never download models or require real audio content. They do
require files to exist because the module deliberately refuses to score
missing audio.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from voix.evaluation import expression_metrics as em


def _touch(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF")
    return path


def _sample(audio: str, control: str = "emotion", label: str = "happy", **extra):
    return {"audio": audio, "control": control, "label": label, **extra}


def _prosody_map(mapping):
    def fn(path):
        return dict(mapping[Path(path).name])

    return fn


def _manifest(*samples):
    return {
        "schema_version": em.EXPRESSION_MANIFEST_SCHEMA,
        "reference_text": "The quick brown fox.",
        "samples": list(samples),
    }


def _passing_report():
    return {
        "speaker_preservation": {"status": "computed", "mean": 0.90},
        "emotion_classifier": {"status": "computed", "accuracy": 0.60},
        "accent_classifier": {"status": "computed", "accuracy": 0.55},
        "intensity_monotonicity": {
            "status": "computed",
            "conditions": {
                "emotion:happy": {
                    "metrics": {
                        "f0_mean_hz": {
                            "rho": 0.90,
                            "expected_direction": 1,
                        }
                    }
                }
            },
        },
    }


class TestManifestValidation:
    def test_valid_manifest_has_no_errors(self):
        payload = _manifest(_sample("happy.wav"))
        assert em.validate_expression_manifest(payload, require_schema=True) == []

    def test_wrong_schema_is_flagged_when_required(self):
        payload = _manifest(_sample("happy.wav"))
        payload["schema_version"] = "something.else"
        errors = em.validate_expression_manifest(payload, require_schema=True)
        assert any("schema_version" in error for error in errors)

    def test_missing_schema_is_allowed_when_not_required(self):
        payload = _manifest(_sample("happy.wav"))
        payload.pop("schema_version")
        assert em.validate_expression_manifest(payload) == []

    @pytest.mark.parametrize("samples", [None, [], "not-a-list"])
    def test_samples_must_be_non_empty_list(self, samples):
        errors = em.validate_expression_manifest({"samples": samples})
        assert any("samples must be a non-empty list" in error for error in errors)

    def test_invalid_sample_fields_are_all_reported(self):
        payload = {
            "samples": [
                "not-an-object",
                {"control": "emotion", "label": "happy"},
                {"audio": "a.wav", "control": "language", "label": ""},
                {"audio": "b.wav", "control": "accent", "label": "irish", "intensity": "loud"},
                {"audio": "c.wav", "control": "accent", "label": "irish", "intensity": 1.5},
            ]
        }
        errors = em.validate_expression_manifest(payload)
        joined = "\n".join(errors)
        assert "samples[0] must be an object" in joined
        assert "samples[1].audio is required" in joined
        assert "samples[2].control must be one of" in joined
        assert "samples[2].label is required" in joined
        assert "samples[3].intensity must be numeric" in joined
        assert "samples[4].intensity must be in [0, 1]" in joined

    def test_load_manifest_round_trip(self, tmp_path):
        path = tmp_path / "manifest.json"
        payload = _manifest(_sample("happy.wav", intensity=0.8))
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = em.load_expression_manifest(path)
        assert loaded["samples"][0]["intensity"] == 0.8

    def test_load_manifest_rejects_invalid_payload(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"samples": []}), encoding="utf-8")
        with pytest.raises(ValueError, match="invalid expression manifest"):
            em.load_expression_manifest(path)


class TestAggregateConditions:
    def test_aggregates_per_condition(self, tmp_path):
        _touch(tmp_path, "happy_a.wav")
        _touch(tmp_path, "happy_b.wav")
        _touch(tmp_path, "sad.wav")
        prosody = _prosody_map(
            {
                "happy_a.wav": {"f0_mean_hz": 200.0, "duration_seconds": 1.0, "energy_mean": 0.2, "f0_voiced_ratio": 0.8},
                "happy_b.wav": {"f0_mean_hz": 220.0, "duration_seconds": 1.2, "energy_mean": 0.4, "f0_voiced_ratio": 0.6},
                "sad.wav": {"f0_mean_hz": 120.0, "duration_seconds": 2.0, "energy_mean": 0.1, "f0_voiced_ratio": 0.7},
            }
        )
        samples = [
            _sample("happy_a.wav", label="happy"),
            _sample("happy_b.wav", label="happy"),
            _sample("sad.wav", label="sad"),
        ]

        result = em.aggregate_conditions(samples, root=tmp_path, prosody_fn=prosody)

        assert result["status"] == "computed"
        assert result["n_samples"] == 3
        happy = result["conditions"]["emotion:happy"]
        assert happy["n"] == 2
        assert happy["f0_mean_hz_mean"] == pytest.approx(210.0)
        assert happy["duration_seconds_mean"] == pytest.approx(1.1)
        assert happy["energy_mean_mean"] == pytest.approx(0.3)
        assert happy["voiced_ratio_mean"] == pytest.approx(0.7)
        assert result["conditions"]["emotion:sad"]["n"] == 1

    def test_missing_audio_is_unavailable_not_zero(self, tmp_path):
        _touch(tmp_path, "present.wav")
        prosody = _prosody_map(
            {"present.wav": {"f0_mean_hz": 100.0, "duration_seconds": 1.0}}
        )
        samples = [
            _sample("present.wav"),
            _sample("missing.wav"),
        ]

        result = em.aggregate_conditions(samples, root=tmp_path, prosody_fn=prosody)

        assert result["status"] == "computed"
        assert result["n_samples"] == 2
        assert result["conditions"]["emotion:happy"]["n"] == 1
        missing = result["per_sample"][1]["prosody"]
        assert missing["status"] == "unavailable"
        assert "missing audio" in missing["reason"]

    def test_all_audio_missing_is_unavailable(self, tmp_path):
        result = em.aggregate_conditions(
            [_sample("nope.wav")], root=tmp_path, prosody_fn=lambda _: {}
        )
        assert result["status"] == "unavailable"
        assert result["conditions"] == {}
        assert result["reason"] == "no readable audio sample"

    def test_prosody_backend_failure_is_contained(self, tmp_path):
        _touch(tmp_path, "bad.wav")

        def boom(_):
            raise RuntimeError("backend exploded")

        result = em.aggregate_conditions(
            [_sample("bad.wav")], root=tmp_path, prosody_fn=boom
        )
        assert result["status"] == "unavailable"
        assert "RuntimeError" in result["per_sample"][0]["prosody"]["reason"]


class TestIntensityMonotonicity:
    def test_happy_sweep_matches_expected_directions(self, tmp_path):
        prosody = {}
        for name, intensity in (("a.wav", 0.2), ("b.wav", 0.5), ("c.wav", 0.9)):
            _touch(tmp_path, name)
            prosody[name] = {
                "f0_mean_hz": 100.0 + 50.0 * intensity,
                "duration_seconds": 1.5 - 0.5 * intensity,
            }
        samples = [
            _sample(name, intensity=intensity)
            for name, intensity in (("a.wav", 0.2), ("b.wav", 0.5), ("c.wav", 0.9))
        ]

        result = em.intensity_monotonicity(
            samples, root=tmp_path, prosody_fn=_prosody_map(prosody)
        )

        assert result["status"] == "computed"
        condition = result["conditions"]["emotion:happy"]
        assert condition["status"] == "computed"
        assert condition["metrics"]["f0_mean_hz"]["rho"] == pytest.approx(1.0)
        assert condition["metrics"]["f0_mean_hz"]["direction_matches"] is True
        assert condition["metrics"]["duration_seconds"]["rho"] == pytest.approx(-1.0)
        assert condition["metrics"]["duration_seconds"]["direction_matches"] is True

    def test_too_few_points_are_insufficient(self, tmp_path):
        for name in ("a.wav", "b.wav"):
            _touch(tmp_path, name)
        prosody = {
            "a.wav": {"f0_mean_hz": 100.0},
            "b.wav": {"f0_mean_hz": 120.0},
        }
        result = em.intensity_monotonicity(
            [_sample("a.wav", intensity=0.2), _sample("b.wav", intensity=0.9)],
            root=tmp_path,
            prosody_fn=_prosody_map(prosody),
        )
        condition = result["conditions"]["emotion:happy"]
        assert condition["status"] == "insufficient_data"
        assert condition["distinct_intensities"] == 2

    def test_single_distinct_intensity_is_insufficient(self, tmp_path):
        prosody = {}
        samples = []
        for name in ("a.wav", "b.wav", "c.wav"):
            _touch(tmp_path, name)
            prosody[name] = {"f0_mean_hz": 100.0}
            samples.append(_sample(name, intensity=0.5))
        result = em.intensity_monotonicity(
            samples, root=tmp_path, prosody_fn=_prosody_map(prosody)
        )
        condition = result["conditions"]["emotion:happy"]
        assert condition["status"] == "insufficient_data"
        assert condition["distinct_intensities"] == 1

    def test_unknown_label_has_no_expected_direction(self, tmp_path):
        prosody = {}
        samples = []
        for name, intensity in (("a.wav", 0.2), ("b.wav", 0.5), ("c.wav", 0.9)):
            _touch(tmp_path, name)
            prosody[name] = {"f0_mean_hz": 100.0 + intensity}
            samples.append(_sample(name, label="neutral", intensity=intensity))
        result = em.intensity_monotonicity(
            samples, root=tmp_path, prosody_fn=_prosody_map(prosody)
        )
        metric = result["conditions"]["emotion:neutral"]["metrics"]["f0_mean_hz"]
        assert metric["expected_direction"] is None
        assert metric["direction_matches"] is None

    def test_accent_has_no_expected_directions(self, tmp_path):
        prosody = {}
        samples = []
        for name, intensity in (("a.wav", 0.2), ("b.wav", 0.5), ("c.wav", 0.9)):
            _touch(tmp_path, name)
            prosody[name] = {"f0_mean_hz": 100.0 + intensity}
            samples.append(
                _sample(name, control="accent", label="irish", intensity=intensity)
            )
        result = em.intensity_monotonicity(
            samples, root=tmp_path, prosody_fn=_prosody_map(prosody)
        )
        metric = result["conditions"]["accent:irish"]["metrics"]["f0_mean_hz"]
        assert metric["expected_direction"] is None

    def test_samples_without_intensity_are_ignored(self, tmp_path):
        _touch(tmp_path, "a.wav")
        result = em.intensity_monotonicity(
            [_sample("a.wav")],
            root=tmp_path,
            prosody_fn=lambda _: {"f0_mean_hz": 100.0},
        )
        assert result["status"] == "unavailable"


class TestSpeakerPreservation:
    def test_computes_mean_and_min(self, tmp_path):
        _touch(tmp_path, "neutral.wav")
        _touch(tmp_path, "happy.wav")
        _touch(tmp_path, "sad.wav")

        def similarity(audio, reference, device="cpu"):
            assert device == "cpu"
            return 0.9 if Path(audio).name == "happy.wav" else 0.7

        result = em.speaker_preservation(
            [
                _sample("happy.wav", neutral_reference="neutral.wav"),
                _sample("sad.wav", label="sad", neutral_reference="neutral.wav"),
            ],
            root=tmp_path,
            similarity_fn=similarity,
        )

        assert result["status"] == "computed"
        assert result["n"] == 2
        assert result["mean"] == pytest.approx(0.8)
        assert result["min"] == pytest.approx(0.7)

    def test_sample_without_reference_is_skipped(self, tmp_path):
        _touch(tmp_path, "happy.wav")
        result = em.speaker_preservation(
            [_sample("happy.wav")],
            root=tmp_path,
            similarity_fn=lambda *args, **kwargs: 0.9,
        )
        assert result["status"] == "unavailable"
        assert result["per_sample"] == []

    def test_missing_files_are_reported(self, tmp_path):
        _touch(tmp_path, "happy.wav")
        result = em.speaker_preservation(
            [_sample("happy.wav", neutral_reference="missing.wav")],
            root=tmp_path,
            similarity_fn=lambda *args, **kwargs: 0.9,
        )
        assert result["status"] == "unavailable"
        entry = result["per_sample"][0]
        assert entry["status"] == "unavailable"
        assert entry["reason"] == "missing audio or reference"

    def test_two_argument_similarity_double_is_supported(self, tmp_path):
        _touch(tmp_path, "happy.wav")
        _touch(tmp_path, "neutral.wav")
        result = em.speaker_preservation(
            [_sample("happy.wav", neutral_reference="neutral.wav")],
            root=tmp_path,
            similarity_fn=lambda a, b: 0.75,
        )
        assert result["status"] == "computed"
        assert result["mean"] == pytest.approx(0.75)

    def test_none_similarity_is_unavailable_not_zero(self, tmp_path):
        _touch(tmp_path, "happy.wav")
        _touch(tmp_path, "neutral.wav")
        result = em.speaker_preservation(
            [_sample("happy.wav", neutral_reference="neutral.wav")],
            root=tmp_path,
            similarity_fn=lambda *args, **kwargs: None,
        )
        assert result["status"] == "unavailable"
        entry = result["per_sample"][0]
        assert entry["status"] == "unavailable"
        assert entry["similarity"] is None
        assert entry["reason"] == "similarity unavailable"

    def test_backend_exception_is_contained(self, tmp_path):
        _touch(tmp_path, "happy.wav")
        _touch(tmp_path, "neutral.wav")

        def boom(*args, **kwargs):
            raise RuntimeError("model missing")

        result = em.speaker_preservation(
            [_sample("happy.wav", neutral_reference="neutral.wav")],
            root=tmp_path,
            similarity_fn=boom,
        )
        assert result["status"] == "unavailable"
        assert "RuntimeError" in result["per_sample"][0]["reason"]


class TestClassifierAccuracy:
    def test_no_classifier_is_unavailable(self, tmp_path):
        result = em.classifier_accuracy(
            [_sample("happy.wav")], root=tmp_path, classifier=None
        )
        assert result["status"] == "unavailable"
        assert result["accuracy"] is None
        assert result["reason"] == "no classifier supplied"

    def test_no_samples_for_requested_control_is_unavailable(self, tmp_path):
        result = em.classifier_accuracy(
            [_sample("happy.wav", control="emotion")],
            root=tmp_path,
            classifier=lambda _: "happy",
            control="accent",
        )
        assert result["status"] == "unavailable"
        assert "control='accent'" in result["reason"]

    def test_accuracy_confusion_and_per_class(self, tmp_path):
        for name in ("a.wav", "b.wav", "c.wav"):
            _touch(tmp_path, name)
        predictions = {"a.wav": "happy", "b.wav": "happy", "c.wav": "HAPPY"}

        result = em.classifier_accuracy(
            [
                _sample("a.wav", label="happy"),
                _sample("b.wav", label="sad"),
                _sample("c.wav", label="happy"),
            ],
            root=tmp_path,
            classifier=lambda path: predictions[Path(path).name],
            control="emotion",
        )

        assert result["status"] == "computed"
        assert result["n"] == 3
        assert result["accuracy"] == pytest.approx(2 / 3)
        assert result["confusion"]["sad"]["happy"] == 1
        assert result["confusion"]["happy"]["happy"] == 2
        assert result["per_class"]["happy"]["accuracy"] == pytest.approx(1.0)
        assert result["per_class"]["sad"]["accuracy"] == pytest.approx(0.0)

    def test_control_filter_ignores_other_controls(self, tmp_path):
        _touch(tmp_path, "happy.wav")
        _touch(tmp_path, "irish.wav")

        result = em.classifier_accuracy(
            [
                _sample("happy.wav", control="emotion", label="happy"),
                _sample("irish.wav", control="accent", label="irish"),
            ],
            root=tmp_path,
            classifier=lambda _: "happy",
            control="emotion",
        )

        assert result["n"] == 1
        assert result["accuracy"] == pytest.approx(1.0)

    def test_missing_audio_is_excluded_from_accuracy(self, tmp_path):
        _touch(tmp_path, "a.wav")
        result = em.classifier_accuracy(
            [
                _sample("a.wav", label="happy"),
                _sample("missing.wav", label="sad"),
            ],
            root=tmp_path,
            classifier=lambda _: "happy",
        )
        assert result["status"] == "computed"
        assert result["n"] == 1
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["per_sample"][1]["status"] == "unavailable"

    def test_all_audio_missing_is_unavailable(self, tmp_path):
        result = em.classifier_accuracy(
            [_sample("missing.wav")],
            root=tmp_path,
            classifier=lambda _: "happy",
        )
        assert result["status"] == "unavailable"
        assert result["accuracy"] is None
        assert result["reason"] == "classifier produced no predictions"

    def test_classifier_exception_is_contained(self, tmp_path):
        _touch(tmp_path, "a.wav")

        def boom(_):
            raise ValueError("bad audio")

        result = em.classifier_accuracy(
            [_sample("a.wav")], root=tmp_path, classifier=boom
        )
        assert result["status"] == "unavailable"
        assert "ValueError" in result["per_sample"][0]["reason"]


class TestEvaluateExpressionManifest:
    def test_report_contains_explicit_unavailable_sections(self, tmp_path):
        _touch(tmp_path, "happy.wav")
        _touch(tmp_path, "neutral.wav")
        manifest = _manifest(
            _sample(
                "happy.wav",
                intensity=0.8,
                neutral_reference="neutral.wav",
            )
        )

        report = em.evaluate_expression_manifest(
            manifest,
            root=tmp_path,
            prosody_fn=lambda _: {"f0_mean_hz": 180.0, "duration_seconds": 1.0},
            similarity_fn=lambda *args, **kwargs: 0.85,
        )

        assert report["schema_version"] == em.EXPRESSION_MANIFEST_SCHEMA
        assert report["n_samples"] == 1
        assert report["prosody"]["status"] == "computed"
        assert report["speaker_preservation"]["mean"] == pytest.approx(0.85)
        assert report["emotion_classifier"]["status"] == "unavailable"
        assert report["accent_classifier"]["status"] == "unavailable"
        assert report["mos"]["status"] == "unavailable"
        assert report["mos"]["reason"] == "awaiting real human ratings"

    def test_invalid_manifest_raises(self, tmp_path):
        with pytest.raises(ValueError, match="invalid expression manifest"):
            em.evaluate_expression_manifest({"samples": []}, root=tmp_path)


class TestExpressionGates:
    def test_all_measured_gates_pass(self):
        result = em.check_expression_gates(_passing_report())
        assert result["status"] == "pass"
        assert result["gates"]["speaker_preservation"]["passed"] is True
        assert result["gates"]["intensity_monotonicity"]["passed"] is True

    def test_failing_gate_fails(self):
        report = _passing_report()
        report["speaker_preservation"]["mean"] = 0.5
        result = em.check_expression_gates(report)
        assert result["status"] == "fail"
        assert result["gates"]["speaker_preservation"]["passed"] is False

    def test_unavailable_metrics_are_not_counted_as_passed(self):
        report = _passing_report()
        report["accent_classifier"] = {"status": "unavailable", "reason": "no classifier"}
        result = em.check_expression_gates(report)
        assert result["status"] == "unavailable"
        assert result["gates"]["accent_classifier"]["passed"] is None

    def test_empty_report_is_unavailable(self):
        result = em.check_expression_gates({})
        assert result["status"] == "unavailable"
        assert result["gates"]

    def test_monotonicity_direction_mismatch_fails(self):
        report = _passing_report()
        report["intensity_monotonicity"]["conditions"]["emotion:happy"]["metrics"][
            "f0_mean_hz"
        ] = {"rho": -0.9, "expected_direction": 1}
        result = em.check_expression_gates(report)
        gate = result["gates"]["intensity_monotonicity"]
        assert gate["passed"] is False
        assert gate["checks"][0]["direction_matches"] is False

    def test_monotonicity_magnitude_below_threshold_fails(self):
        report = _passing_report()
        report["intensity_monotonicity"]["conditions"]["emotion:happy"]["metrics"][
            "f0_mean_hz"
        ] = {"rho": 0.5, "expected_direction": 1}
        result = em.check_expression_gates(report)
        gate = result["gates"]["intensity_monotonicity"]
        assert gate["passed"] is False
        assert gate["checks"][0]["magnitude_ok"] is False

    def test_wer_gate_passes_and_fails(self):
        passing = em.check_expression_gates(_passing_report(), wer=0.15, max_wer=0.20)
        assert passing["gates"]["wer"]["passed"] is True

        failing = em.check_expression_gates(_passing_report(), wer=0.35, max_wer=0.20)
        assert failing["status"] == "fail"
        assert failing["gates"]["wer"]["passed"] is False

    def test_wer_threshold_without_value_is_unavailable(self):
        result = em.check_expression_gates(_passing_report(), max_wer=0.20)
        assert result["status"] == "unavailable"
        assert result["gates"]["wer"]["status"] == "unavailable"
        assert result["gates"]["wer"]["passed"] is None
