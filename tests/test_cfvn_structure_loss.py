"""Tests for the CFVN raw-face structure-preservation regularizer."""
from __future__ import annotations

import torch

from voix.training.trainer_cfvn import structure_preservation_loss


def test_identical_structure_has_zero_loss():
    reference = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    loss = structure_preservation_loss(reference, reference, temperature=0.1)
    assert float(loss) < 1e-5


def test_broken_structure_has_positive_loss():
    reference = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    broken = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    assert float(structure_preservation_loss(broken, reference)) > 0.1


def test_single_row_is_safe():
    row = torch.ones(1, 4)
    loss = structure_preservation_loss(row, row)
    assert float(loss) == 0.0
