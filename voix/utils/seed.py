"""Global deterministic seed control for reproducibility across all experiments."""

import os
import random


def set_global_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility across runs.

    Applies seeds to: Python random, NumPy, PyTorch CPU/GPU, and CUDA
    determinism flags. Must be called at the start of every training script.

    Args:
        seed: Integer seed value. Default 42.
    """
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_generator(seed: int = 42) -> "torch.Generator":
    """Return a seeded torch.Generator for DataLoader reproducibility."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g