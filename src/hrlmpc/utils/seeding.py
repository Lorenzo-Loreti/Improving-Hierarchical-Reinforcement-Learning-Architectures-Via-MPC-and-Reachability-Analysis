"""Seeding of every random number generator used by an experiment (code rule C3)."""

from __future__ import annotations

import importlib.util
import random

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> np.random.Generator:
    """Seed Python, NumPy and, if installed, PyTorch.

    Components should draw from the returned generator, or from generators
    derived from it, rather than from NumPy's legacy global state; that state
    is seeded too, for libraries that still use it.

    With ``deterministic_torch`` PyTorch is restricted to deterministic
    algorithms, so a CPU run is repeatable bit for bit. On a GPU some
    operations then also need the environment variable
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8``, set before PyTorch is imported.

    Args:
        seed: Non-negative seed.
        deterministic_torch: Whether to enforce deterministic PyTorch algorithms.

    Returns:
        A NumPy generator seeded with ``seed``.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch.manual_seed(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)
