"""Tests for the C1 reference-prosody selector.

These tests use a synthetic manifest and style cache, so they do not load
StyleTTS2 or read CREMA-D audio.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "demo" / "generate_reference_prosody_c1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "generate_reference_prosody_c1", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_chunk(tmp_path: Path) -> tuple[Path, Path, dict[int, np.ndarray]]:
    """Create a manifest and cache with known per-actor prosody rows."""
    rows = []
    styles = {}
    for index, (actor, sex, emotion, split, intensity) in enumerate(
        [
            ("F1", "Female", "happy", "train", "low"),
            ("F1", "Female", "happy", "train", "high"),
            ("F2", "Female", "happy", "train", "medium"),
            ("F3", "Female", "happy", "test", "high"),
            ("F4", "Female", "sad", "train", "high"),
            ("F5", "Female", "angry", "train", "high"),
            ("M1", "Male", "happy", "train", "high"),
        ]
    ):
        clip_id = f"{actor}_IEO_{emotion.upper()}_{index:02d}"
        rows.append(
            {
                "clip_id": clip_id,
                "actor_id": actor,
                "intended_emotion": emotion,
                "intensity": intensity,
                "split": split,
                "sex": sex,
            }
        )
        style = np.zeros(256, dtype=np.float32)
        style[128 + index] = float(index + 1)
        styles[clip_id] = style

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    cache = tmp_path / "cache.npz"
    np.savez_compressed(
        cache,
        paths=np.asarray([f"fake/{clip_id}.wav" for clip_id in styles], dtype=str),
        styles=np.stack([styles[clip_id] for clip_id in styles]),
    )
    return manifest, cache, styles


def test_reference_styles_use_train_only_gender_matched_actors(tmp_path: Path):
    module = _load_module()
    manifest, cache, _styles = _write_chunk(tmp_path)

    references = module._reference_styles(
        manifest_path=manifest,
        cache_path=cache,
        sex="Female",
        emotions=("happy", "sad", "angry"),
        per_class=2,
        seed=42,
    )

    happy = references["happy"]
    assert happy["n_references"] == 2
    assert set(happy["actor_ids"]) == {"F1", "F2"}
    assert all(actor.startswith("F") for actor in happy["actor_ids"])
    assert references["sad"]["actor_ids"] == ["F4"]
    assert references["angry"]["actor_ids"] == ["F5"]
    # The test-split actor F3 and the male actor M1 never enter a reference.
    all_actors = {
        actor
        for entry in references.values()
        for actor in entry["actor_ids"]
    }
    assert "F3" not in all_actors
    assert "M1" not in all_actors


def test_reference_styles_pick_one_clip_per_actor_and_are_deterministic(
    tmp_path: Path,
):
    module = _load_module()
    manifest, cache, styles = _write_chunk(tmp_path)

    first = module._reference_styles(
        manifest_path=manifest,
        cache_path=cache,
        sex="Female",
        emotions=("happy",),
        per_class=1,
        seed=7,
    )
    second = module._reference_styles(
        manifest_path=manifest,
        cache_path=cache,
        sex="Female",
        emotions=("happy",),
        per_class=1,
        seed=7,
    )

    assert first["happy"]["actor_ids"] == second["happy"]["actor_ids"]
    assert first["happy"]["n_references"] == 1
    actor = first["happy"]["actor_ids"][0]
    clip_ids = first["happy"]["clip_ids"]
    assert len(clip_ids) == 1
    if actor == "F1":
        # F1 has low and high variants; the higher-intensity clip wins.
        assert clip_ids[0].endswith("F1_IEO_HAPPY_01")

    # The mean is computed over the prosody half only.
    selected = styles[clip_ids[0]]
    expected = selected[128:]
    actual = np.asarray(first["happy"]["prosody_mean"], dtype=np.float32)
    assert expected.shape == (128,)
    assert actual.shape == (128,)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert selected[:128].sum() == 0.0  # the synthetic timbre half is empty


def test_reference_styles_reject_missing_emotion(tmp_path: Path):
    module = _load_module()
    manifest, cache, _styles = _write_chunk(tmp_path)

    try:
        module._reference_styles(
            manifest_path=manifest,
            cache_path=cache,
            sex="Male",
            emotions=("sad",),
            per_class=1,
            seed=0,
        )
    except ValueError as exc:
        assert "sad" in str(exc)
    else:  # pragma: no cover - the selector must fail loudly
        raise AssertionError("missing class did not raise ValueError")
