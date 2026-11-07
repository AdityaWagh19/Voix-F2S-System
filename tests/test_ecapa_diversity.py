"""End-to-end check that distinct faces map to distinct speaker embeddings.

This used to run model construction at import time and print diagnostics.  It is
now an artifact-gated test: a missing V2 checkpoint skips, and a collapsed
CFVN fails instead of printing numbers nobody asserts on.
"""
from __future__ import annotations

import numpy as np
import pytest

from voix.inference.pipeline import VoixPipeline
from voix.utils.paths import project_path

FACES = [
    "C1_01_young_female_Zendaya.jpg",
    "C1_02_young_male_Timothee_Chalamet.jpg",
    "C2_11_black_male_Barack_Obama.jpg",
]

TEXT = (
    "The human voice reflects both physical anatomy and lived culture, "
    "carrying the unique cadence of personal identity."
)


@pytest.mark.neural
def test_distinct_faces_produce_distinct_speaker_embeddings(v2_artifacts):
    pipeline = VoixPipeline(
        device="cpu",
        enable_watermarking=False,
        tts_use_neural=False,
    )

    embeddings = {}
    for face_name in FACES:
        face_path = project_path("data", "test_subjects", face_name)
        if not face_path.exists():
            pytest.skip(f"face fixture missing: {face_path}")
        results = pipeline.generate(
            face_input=str(face_path),
            text=TEXT,
            num_voices=1,
            temperature=1.0,
            watermark=False,
        )
        embeddings[face_name] = results[0]["speaker_embedding"].numpy().flatten()

    names = list(embeddings)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = embeddings[names[i]]
            b = embeddings[names[j]]
            cosine = float(
                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            )
            assert np.isfinite(cosine)
            assert cosine < 0.99, (
                f"{names[i]} and {names[j]} map to nearly identical "
                f"embeddings (cosine={cosine:.4f})"
            )
