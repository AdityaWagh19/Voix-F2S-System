"""Tests for the research-demo v3 condition builder and verification helpers.

The demo is a 3 x (neutral + 3 emotions + 4 accents) gallery.  These tests use
synthetic residuals so they exercise the composition rules without loading any
checkpoint or synthesizing audio.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "demo" / "generate_research_demo.py"
STYLE_DIM = 256
PROSODY_START = 128


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_research_demo", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prosody_residual(value: float = 1.0) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(PROSODY_START),
            torch.full((STYLE_DIM - PROSODY_START,), float(value)),
        ]
    )


def _emotion_payload(emotions=("happy", "sad", "angry")):
    return {
        "emotions": {
            emotion: {
                "delta": _prosody_residual(1.0),
                "delta_prosody_only": _prosody_residual(1.0),
            }
            for emotion in emotions
        }
    }


def _accent_payload(accents=("us", "england", "indian_english", "australian")):
    return {
        "residuals": {
            accent: torch.ones(STYLE_DIM) for accent in accents
        }
    }


def _conditions(
    *,
    emotions=("happy", "sad", "angry"),
    accents=("us", "england", "indian_english", "australian"),
    emotion_payload=None,
    accent_payload=None,
    emotion_scale: float = 0.28,
    accent_scale: float = 0.30,
    accent_timbre_weight: float = 0.25,
    accent_scale_report=None,
):
    module = _load_module()
    return module.build_conditions(
        voice_id="C2_12",
        voice_label="C2_12_black_female_Lupita_Nyongo",
        base_style=torch.zeros(1, STYLE_DIM),
        gender=0,
        age=30.0,
        text="A fixed research prompt.",
        emotion_payload=_emotion_payload() if emotion_payload is None else emotion_payload,
        accent_payload=_accent_payload() if accent_payload is None else accent_payload,
        emotions=list(emotions),
        accents=list(accents),
        emotion_scale=float(emotion_scale),
        accent_scale=float(accent_scale),
        accent_timbre_weight=float(accent_timbre_weight),
        accent_scale_report=accent_scale_report,
    )


def test_demo_design_is_three_voices_three_emotions_four_accents():
    module = _load_module()
    args = module.parse_args([])
    assert len(module.DEFAULT_VOICES) == 3
    assert len(module.EMOTIONS) == 3
    assert len(module.ACCENTS) == 4
    assert tuple(args.emotions) == module.EMOTIONS
    assert tuple(args.accents) == module.ACCENTS
    assert args.emotion_scale == 0.28
    assert args.accent_scale == 0.30
    assert args.accent_timbre_weight == 0.25
    assert module.DEFAULT_OUTPUT.name == "research_v3"
    assert not hasattr(args, "include_lora")
    assert not hasattr(args, "lora")
    assert not hasattr(module, "_write_html")


def test_build_conditions_maps_indian_english_to_the_globe_artifact_key():
    conditions = _conditions(
        accent_payload=_accent_payload(
            ("us", "england", "india_south_asia", "australian")
        )
    )
    accent_conditions = {
        condition["condition"]: condition
        for condition in conditions
        if condition["track"] == "accent"
    }
    assert set(accent_conditions) == {
        "us",
        "england",
        "indian_english",
        "australian",
    }
    assert accent_conditions["indian_english"]["residual"]["artifact"] == (
        "accent_class_mean"
    )


def test_build_conditions_returns_neutral_emotions_and_accents_in_order():
    conditions = _conditions()
    assert [condition["name"] for condition in conditions] == [
        "C2_12__neutral",
        "C2_12__emotion_happy",
        "C2_12__emotion_sad",
        "C2_12__emotion_angry",
        "C2_12__accent_us",
        "C2_12__accent_england",
        "C2_12__accent_indian_english",
        "C2_12__accent_australian",
    ]
    assert [condition["track"] for condition in conditions] == [
        "baseline",
        "emotion",
        "emotion",
        "emotion",
        "accent",
        "accent",
        "accent",
        "accent",
    ]
    assert conditions[0]["level"] == "baseline"
    assert all(
        condition["level"] == "exploratory_application"
        for condition in conditions[1:]
    )
    assert all(condition["text"] == "A fixed research prompt." for condition in conditions)


def test_emotion_condition_applies_only_the_prosody_half():
    conditions = _conditions(emotion_scale=0.28)
    happy = conditions[1]
    assert torch.allclose(happy["style"][0, :PROSODY_START], torch.zeros(PROSODY_START))
    assert torch.allclose(
        happy["style"][0, PROSODY_START:],
        torch.full((STYLE_DIM - PROSODY_START,), 0.28),
    )
    assert happy["residual"]["variant"] == "prosody_only"
    assert happy["residual"]["scale"] == 0.28
    assert happy["f0_scale"] == 1.0


def test_sad_condition_keeps_identity_without_a_pitch_shift():
    conditions = _conditions()
    sad = conditions[2]
    assert sad["condition"] == "sad"
    # The E1 sad F0 lowering did not transfer to the calibrated face styles.
    # A post-synthesis pitch shift was rejected because it collapsed ECAPA.
    assert sad["f0_scale"] == 1.0
    assert "pitch shift" in sad["note"]
    assert sad["residual"]["variant"] == "prosody_only"
    assert "compensation" not in sad["residual"]
    assert torch.allclose(
        sad["style"][0, :PROSODY_START], torch.zeros(PROSODY_START)
    )


def test_accent_condition_down_weights_timbre_and_keeps_prosody():
    conditions = _conditions(accent_scale=0.30, accent_timbre_weight=0.25)
    us = conditions[4]
    assert torch.allclose(
        us["style"][0, :PROSODY_START],
        torch.full((PROSODY_START,), 0.30 * 0.25),
    )
    assert torch.allclose(
        us["style"][0, PROSODY_START:],
        torch.full((STYLE_DIM - PROSODY_START,), 0.30),
    )
    assert us["f0_scale"] == 1.0
    assert us["residual"]["timbre_weight"] == 0.25
    assert us["residual"]["scale"] == 0.30
    assert us["residual"]["a3_selected_scale"] == 0.3


def test_missing_emotion_and_accent_are_skipped():
    conditions = _conditions(
        emotions=("happy", "fear"),
        accents=("us", "indian_english"),
        emotion_payload=_emotion_payload(("happy",)),
        accent_payload=_accent_payload(("us",)),
    )
    assert [condition["name"] for condition in conditions] == [
        "C2_12__neutral",
        "C2_12__emotion_happy",
        "C2_12__accent_us",
    ]


def test_duplicate_conditions_are_removed_while_preserving_order():
    conditions = _conditions(emotions=("happy", "happy"), accents=("us", "us"))
    assert [condition["name"] for condition in conditions] == [
        "C2_12__neutral",
        "C2_12__emotion_happy",
        "C2_12__accent_us",
    ]


def test_accent_recall_prefers_classifier_counts_then_wer_counts(tmp_path):
    module = _load_module()
    report = tmp_path / "selection.json"
    report.write_text(
        """{
          "classifier_by_condition": {
            "style_s0.3": {
              "per_class_recall": {"england": 0.5, "indian_english": 0.0},
              "per_class_counts": {"england": 6, "indian_english": 6}
            }
          },
          "wer": {"by_accent": {"australian": {"regex": {"n": 6}}}}
        }""",
        encoding="utf-8",
    )
    assert module._accent_recall(report, 0.3, "england") == (3, 6)
    assert module._accent_recall(report, 0.3, "indian_english") == (0, 6)
    # No classifier recall entry -> None, never a fabricated zero.
    assert module._accent_recall(report, 0.3, "australian") is None
    assert module._accent_recall(report, 0.2, "england") is None
    assert module._accent_recall(tmp_path / "missing.json", 0.3, "england") is None


def test_accent_recall_falls_back_to_wer_counts(tmp_path):
    module = _load_module()
    report = tmp_path / "selection.json"
    report.write_text(
        """{
          "classifier_by_condition": {
            "style_s0.3": {"per_class_recall": {"england": 0.8333333333333334}}
          },
          "wer": {"by_accent": {"england": {"regex": {"n": 6}}}}
        }""",
        encoding="utf-8",
    )
    assert module._accent_recall(report, 0.3, "england") == (5, 6)


def _voice_row(condition: str, *, f0: float, duration: float):
    return {
        "voice_id": "C2_12",
        "condition": condition,
        "metrics": {
            "f0_median_hz": f0,
            "duration_s": duration,
            "noise_floor_dbfs": -62.0,
        },
        "ecapa_to_neutral": 0.97,
        "wer": None,
    }


def test_emotion_direction_checks_pass_when_all_three_move_as_expected():
    module = _load_module()
    rows = [
        _voice_row("neutral", f0=180.0, duration=7.5),
        _voice_row("happy", f0=196.0, duration=7.2),
        _voice_row("sad", f0=171.0, duration=7.7),
        _voice_row("angry", f0=195.0, duration=7.0),
    ]
    block = module._verify_emotion_directions(rows, [{"id": "C2_12"}])
    assert block["n_checks"] == 3
    assert block["n_passed"] == 3
    assert block["passed"] is True
    happy = next(check for check in block["checks"] if check["emotion"] == "happy")
    assert happy["f0_evaluated"] is True
    assert happy["f0_direction_reproduced"] is True


def test_sad_direction_passes_on_duration_even_when_f0_is_flat():
    module = _load_module()
    rows = [
        _voice_row("neutral", f0=180.0, duration=7.5),
        _voice_row("happy", f0=196.0, duration=7.2),
        _voice_row("sad", f0=180.8, duration=7.7),
        _voice_row("angry", f0=195.0, duration=7.0),
    ]
    block = module._verify_emotion_directions(rows, [{"id": "C2_12"}])
    assert block["n_passed"] == 3
    assert block["passed"] is True
    sad = next(check for check in block["checks"] if check["emotion"] == "sad")
    assert sad["f0_evaluated"] is False
    assert sad["f0_direction_reproduced"] is False
    assert sad["duration_ok"] is True
    assert sad["passed"] is True


def test_sad_direction_fails_when_duration_does_not_lengthen():
    module = _load_module()
    rows = [
        _voice_row("neutral", f0=180.0, duration=7.5),
        _voice_row("happy", f0=196.0, duration=7.2),
        _voice_row("sad", f0=181.0, duration=7.4),
        _voice_row("angry", f0=195.0, duration=7.0),
    ]
    block = module._verify_emotion_directions(rows, [{"id": "C2_12"}])
    assert block["n_passed"] == 2
    assert block["passed"] is False
    sad = next(check for check in block["checks"] if check["emotion"] == "sad")
    assert sad["duration_ok"] is False
    assert sad["passed"] is False


def test_happy_direction_fails_when_f0_does_not_rise():
    module = _load_module()
    rows = [
        _voice_row("neutral", f0=180.0, duration=7.5),
        _voice_row("happy", f0=179.0, duration=7.2),
        _voice_row("sad", f0=180.8, duration=7.7),
        _voice_row("angry", f0=195.0, duration=7.0),
    ]
    block = module._verify_emotion_directions(rows, [{"id": "C2_12"}])
    assert block["n_passed"] == 2
    happy = next(check for check in block["checks"] if check["emotion"] == "happy")
    assert happy["f0_evaluated"] is True
    assert happy["f0_ok"] is False
    assert happy["passed"] is False


def test_condition_deltas_are_relative_to_that_voice_neutral():
    module = _load_module()
    rows = [
        _voice_row("neutral", f0=180.0, duration=7.5),
        _voice_row("happy", f0=196.0, duration=7.2),
    ]
    rows[1]["track"] = "emotion"
    block = module._condition_deltas(rows, "C2_12")
    happy = block["conditions"]["emotion:happy"]
    assert happy["f0_delta_hz"] == pytest.approx(16.0)
    assert happy["duration_delta_s"] == pytest.approx(-0.3)
    assert happy["ecapa_to_neutral"] == 0.97


def _application_row(voice_id: str, track: str, condition: str, style_hash: str, ecapa: float):
    return {
        "name": f"{voice_id}__{track}_{condition}",
        "voice_id": voice_id,
        "condition": condition,
        "style_sha256": style_hash,
        "ecapa_to_neutral": ecapa,
    }


def test_application_verification_requires_all_conditions_and_identity_floor():
    module = _load_module()
    voice = {
        "id": "C2_12",
        "calibrated_style": {"sha256": "neutral-style"},
    }
    rows = [
        _application_row("C2_12", "accent", "us", "us-style", 0.97),
        _application_row("C2_12", "accent", "england", "england-style", 0.96),
        _application_row(
            "C2_12", "accent", "indian_english", "indian-style", 0.96
        ),
        _application_row("C2_12", "accent", "australian", "australian-style", 0.95),
    ]
    block = module._application_verification(
        rows,
        [voice],
        track="accent",
        conditions=("us", "england", "indian_english", "australian"),
    )
    assert block["n_present"] == 4
    assert block["missing"] == []
    assert block["style_hashes_distinct_per_voice"] is True
    assert block["min_ecapa_to_neutral"] == 0.95
    assert block["passed"] is True


def test_application_verification_fails_on_missing_or_identity_loss():
    module = _load_module()
    voice = {
        "id": "C2_12",
        "calibrated_style": {"sha256": "neutral-style"},
    }
    rows = [
        _application_row("C2_12", "accent", "us", "same-style", 0.91),
        _application_row("C2_12", "accent", "england", "same-style", 0.89),
    ]
    block = module._application_verification(
        rows,
        [voice],
        track="accent",
        conditions=("us", "england", "indian_english", "australian"),
    )
    assert block["missing"] == [
        "C2_12__accent_indian_english",
        "C2_12__accent_australian",
    ]
    assert block["style_hashes_distinct_per_voice"] is False
    assert block["min_ecapa_to_neutral"] == 0.89
    assert block["passed"] is False


class _StubCVAE:
    def prior_net(self, e_f_voice):
        return torch.zeros(e_f_voice.shape[0], 128), None

    def decoder(self, mu_prior):
        return torch.ones(mu_prior.shape[0], 192)


class _StubPipeline:
    device = "cpu"

    def __init__(self):
        self.cvae = _StubCVAE()

    def extract_face_features(self, path):
        return torch.zeros(1, 512), 0, 30.0, None

    def normalize_face(self, e_f):
        return e_f

    def cfvn(self, e_f, normalize=True):
        return torch.ones(1, 192)

    def adapter(self, e_s):
        assert e_s.dim() == 2
        return torch.ones(e_s.shape[0], 256)


def test_face_style_is_two_dimensional():
    module = _load_module()
    style, gender, age = module._face_style(_StubPipeline(), Path("face.png"))
    assert style.shape == (1, 256)
    assert gender == 0
    assert age == 30.0
