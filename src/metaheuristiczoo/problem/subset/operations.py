"""Neighbourhoods and reproducible random subsets, for the baselines (baselines.py).

These are operations of SUBSET space. The optimisers never use them: they move a continuous state and the
decoder turns it into a subset (see the package docstring). Subsets are sorted tuples of candidate indices,
as decode.py returns them.

Random subsets
--------------
For a cardinality RANGE, `random_subset_in_range` draws k uniformly from the range and then a subset
uniformly of that size. This is NOT uniform over all admissible subsets (that would weight size k by
C(n, k)), and it is not what a uniform state in [0, 1]^n decodes to either; which distribution a "random
search" baseline uses changes its results, so the choice is explicit here. On the study's instances the
range is a single cardinality p, where all three coincide: uniform over the C(n, p) subsets of size p.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

from metaheuristiczoo.problem.subset.decode import Subset


# -- validation -------------------------------------------------------------------------------------

def as_subset(members: Sequence[int]) -> Subset:
    """A sorted tuple of distinct non-negative ints, or ValueError."""
    subset = tuple(int(i) for i in members)
    if list(subset) != sorted(set(subset)) or (subset and subset[0] < 0):
        raise ValueError(f"a subset is a strictly increasing tuple of non-negative indices, got {members!r}")
    return subset


# -- membership -------------------------------------------------------------------------------------

def add(subset: Subset, element: int) -> Subset:
    """subset ∪ {element}; ValueError if element is already a member."""
    if element in subset:
        raise ValueError(f"{element} is already in {subset}")
    return tuple(sorted(subset + (int(element),)))


def remove(subset: Subset, element: int) -> Subset:
    """subset \\ {element}; ValueError if element is not a member."""
    if element not in subset:
        raise ValueError(f"{element} is not in {subset}")
    return tuple(i for i in subset if i != element)


def swap(subset: Subset, out_element: int, in_element: int) -> Subset:
    """(subset \\ {out}) ∪ {in}: the cardinality-preserving elementary move."""
    return add(remove(subset, out_element), in_element)


# -- neighbourhoods ---------------------------------------------------------------------------------

def swap_neighbours(subset: Subset, n_candidates: int) -> Iterator[Subset]:
    """All |S| (n - |S|) one-swap neighbours, in (out, in) lexicographic order."""
    members = set(subset)
    outside = [i for i in range(int(n_candidates)) if i not in members]
    for out_element in subset:
        for in_element in outside:
            yield swap(subset, out_element, in_element)


def add_neighbours(subset: Subset, n_candidates: int) -> Iterator[Subset]:
    members = set(subset)
    for element in range(int(n_candidates)):
        if element not in members:
            yield add(subset, element)


def remove_neighbours(subset: Subset) -> Iterator[Subset]:
    for element in subset:
        yield remove(subset, element)


# -- distances --------------------------------------------------------------------------------------


# -- counting and random generation -----------------------------------------------------------------


def _generator(rng) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def random_subset(rng, n_candidates: int, k: int) -> Subset:
    """Uniform over the C(n, k) subsets of size k. `rng`: a numpy Generator or a seed."""
    n, k = int(n_candidates), int(k)
    if not 0 <= k <= n:
        raise ValueError(f"k must lie in [0, {n}], got {k}")
    if k == 0:
        return ()
    return tuple(sorted(int(i) for i in _generator(rng).choice(n, size=k, replace=False)))


def random_subset_in_range(rng, n_candidates: int, min_cardinality: int, max_cardinality: int) -> Subset:
    """Cardinality uniform on [min, max], then a subset uniform of that size (see module docstring)."""
    rng = _generator(rng)
    lo, hi = int(min_cardinality), int(max_cardinality)
    if not 0 <= lo <= hi <= int(n_candidates):
        raise ValueError(f"need 0 <= min <= max <= n, got {lo}, {hi}, {n_candidates}")
    k = int(rng.integers(lo, hi + 1))
    return random_subset(rng, n_candidates, k)
