"""Shared pytest fixtures.

Learned-artifact tests are skipped, not silently downgraded to a fallback,
when the retrained V2 stack is unavailable.  This is the Phase 4 contract:
a missing checkpoint must be visible as a skip, never as a green test that
exercised random weights or a harmonic stand-in.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from voix.utils.paths import project_path


def v2_artifact_paths() -> dict[str, Path]:
    return {
        "face scaler": project_path(
            "data", "processed", "voxceleb2_v2", "face_scaler_v2.pt"
        ),
        "CFVN": project_path("results", "checkpoints", "cfvn_v2_best.pt"),
        "CVAE": project_path(
            "checkpoints", "cvae_cfvn_face192_latent128_diversity.pt"
        ),
        "adapter": project_path("checkpoints", "style_adapter_cfvn_best.pt"),
    }


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "neural: requires trained V2 checkpoints or a neural backend"
    )


@pytest.fixture(scope="session")
def v2_artifacts() -> dict[str, Path]:
    """Skip when the retrained FaceFeatureV2 stack is incomplete."""
    paths = v2_artifact_paths()
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        pytest.skip(
            "V2 artifacts unavailable:\n  - " + "\n  - ".join(missing)
        )
    return paths
