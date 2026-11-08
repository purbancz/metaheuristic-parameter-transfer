"""Swarm diversity metrics shared across algorithm families."""

import numpy as np


def normalized_solution_candidates_diversity(positions, lower_bound, upper_bound) -> float:
    """Mean Euclidean distance of positions to their centroid, normalized by
    the domain diagonal ||upper_bound - lower_bound||.

    The normalization makes convergence thresholds scale- and
    dimension-independent: a uniformly random swarm scores ~0.29 regardless
    of bounds and dimensionality, and 0 means fully collapsed.
    """
    positions = np.asarray(positions, dtype=float)
    centroid = positions.mean(axis=0)
    diversity = float(np.mean(np.linalg.norm(positions - centroid, axis=1)))
    diagonal = float(np.linalg.norm(
        np.asarray(upper_bound, dtype=float) - np.asarray(lower_bound, dtype=float)
    ))
    if diagonal == 0.0:
        return 0.0
    return diversity / diagonal
