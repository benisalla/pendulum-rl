"""Small shared helpers: seeding and CPU thread control."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so that a run is repeatable on the same machine."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_threads(n: int) -> None:
    """Limit PyTorch (and BLAS, for child processes) to ``n`` CPU threads.

    Small RL networks are faster on one thread than on eight: the matrices are tiny and
    thread synchronisation costs more than it saves.
    """
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    torch.set_num_threads(n)


def to_tensor(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)
