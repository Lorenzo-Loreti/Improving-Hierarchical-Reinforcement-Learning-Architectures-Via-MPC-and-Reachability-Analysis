"""Tests of the seeding utility (code rule C3)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from hrlmpc.utils.seeding import seed_everything


def _draws(seed: int) -> tuple[float, float, float]:
    generator = seed_everything(seed, deterministic_torch=False)
    return random.random(), float(np.random.random()), float(generator.random())


def test_same_seed_gives_the_same_streams() -> None:
    assert _draws(3) == _draws(3)


def test_different_seeds_give_different_streams() -> None:
    first, second = _draws(3), _draws(4)
    assert all(a != b for a, b in zip(first, second))


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)


def test_torch_is_seeded_and_made_deterministic() -> None:
    torch = pytest.importorskip("torch")
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        seed_everything(5)
        first = torch.rand(3)
        seed_everything(5)
        second = torch.rand(3)
        assert torch.equal(first, second)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(was_deterministic)
