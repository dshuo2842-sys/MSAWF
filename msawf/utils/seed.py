"""Deterministic random-seed setup without implicit device selection."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SeedState:
    seed: int
    deterministic: bool
    numpy_seeded: bool
    cuda_seeded: bool


def seed_everything(seed: int, *, deterministic: bool = True) -> SeedState:
    """Seed Python, available NumPy, and PyTorch RNGs explicitly."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)

    numpy_seeded = False
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_seeded = True
    except (ImportError, OSError):
        # NumPy is optional for this foundation and is not installed implicitly.
        numpy_seeded = False

    torch.manual_seed(seed)
    cuda_seeded = torch.cuda.is_available()
    if cuda_seeded:
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

    return SeedState(
        seed=seed,
        deterministic=deterministic,
        numpy_seeded=numpy_seeded,
        cuda_seeded=cuda_seeded,
    )


def make_generator(seed: int) -> torch.Generator:
    """Create an independently seeded CPU generator for data transformations."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator
