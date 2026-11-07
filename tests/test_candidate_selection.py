"""Tests for deterministic diverse candidate selection."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from voix.inference.candidate_selection import select_diverse_embeddings


class TestSelectDiverseEmbeddings:
    def test_selects_central_mode_then_diverse_alternatives(self):
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        )

        selected = select_diverse_embeddings(embeddings, k=3, reference=embeddings[0])

        assert selected.shape == (3, 2)
        assert torch.allclose(selected[0], embeddings[0])
        # The two alternatives should be the antipodal and orthogonal modes,
        # not the near-duplicate at index 1.
        assert torch.allclose(selected[1], embeddings[3])
        assert torch.allclose(selected[2], embeddings[2])
        pairwise = F.normalize(selected, dim=-1) @ F.normalize(selected, dim=-1).T
        off_diagonal = pairwise[~torch.eye(3, dtype=torch.bool)]
        assert off_diagonal.max().item() <= 0.0

    def test_k_equals_candidate_count_returns_everything(self):
        embeddings = torch.randn(4, 8)
        selected = select_diverse_embeddings(embeddings, k=4)
        assert selected.shape == (4, 8)
        assert {tuple(row.tolist()) for row in selected} == {
            tuple(row.tolist()) for row in embeddings
        }

    def test_single_candidate_is_the_central_mode(self):
        embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        selected = select_diverse_embeddings(embeddings, k=1, reference=embeddings[0])
        assert selected.shape == (1, 2)
        assert torch.allclose(selected[0], embeddings[0])

    def test_deterministic(self):
        embeddings = torch.randn(10, 16)
        first = select_diverse_embeddings(embeddings, k=3)
        second = select_diverse_embeddings(embeddings, k=3)
        assert torch.equal(first, second)

    def test_default_reference_uses_pool_mean(self):
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        )
        selected = select_diverse_embeddings(embeddings, k=1)
        # The normalized pool mean points into the first quadrant, so the
        # primary is one of the two positive-x candidates.
        assert selected[0].tolist() in (embeddings[0].tolist(), embeddings[1].tolist())

    def test_rejects_wrong_reference_shape(self):
        embeddings = torch.randn(3, 4)
        with pytest.raises(ValueError, match="reference must have shape"):
            select_diverse_embeddings(embeddings, k=1, reference=torch.randn(5))

    def test_rejects_invalid_k(self):
        embeddings = torch.randn(3, 4)
        with pytest.raises(ValueError, match="k must be >= 1"):
            select_diverse_embeddings(embeddings, k=0)
        with pytest.raises(ValueError, match="exceeds"):
            select_diverse_embeddings(embeddings, k=4)

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            select_diverse_embeddings(torch.randn(4), k=1)
