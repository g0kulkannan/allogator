"""Candidate-residue selection shared by prediction and benchmarking."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def candidate_mask(sequence_length: int, active_residues: Iterable[int]) -> np.ndarray:
    """Return a mask excluding active-site and sequence-adjacent positions."""
    mask = np.ones(sequence_length, dtype=bool)
    for active_position in active_residues:
        for offset in (-1, 0, 1):
            position = int(active_position) + offset
            if 1 <= position <= sequence_length:
                mask[position - 1] = False
    return mask
