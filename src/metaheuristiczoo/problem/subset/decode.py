"""Deterministic decoders from a continuous state to a subset of candidates.

A decoder is a pure function `decode(x) -> S` from a state x in [0, 1]^n (one coordinate per
candidate) to a subset S of {0, ..., n-1}, returned as a SORTED TUPLE of candidate indices - so
that two decodings of equal states compare equal, a subset can key a cache, and subsets have a
total order. A decoder holds no random state and calls no random number generator: a subset
problem must consume exactly the random numbers any other FloatProblem consumes, or seeded
parity with the rest of the repository is lost.

Ranking (shared by both decoders)
---------------------------------
Candidates are ranked by (-x_i, i): descending value, and on equal values the LOWER INDEX ranks
first. The rank is computed with `np.lexsort`, so it does not depend on the sort kind numpy picks
for `argsort`. Plain `np.argsort(x)[-K:]` is deterministic for a given input but not index-ordered:
on x = [.5, .5, .5, .5, .1] with K = 2 it selects {1, 3} and on [.9, .2, .9, .4, .9, .1, .9] with
K = 3 it selects {2, 4, 6}; the rule here selects {0, 1} and {0, 2, 4}.

Tie rules (a control, not a default)
------------------------------------
The default tie rule above ranks equal values by index. That is harmless while ties are rare, but a swarm that clips
at the box piles up coordinates at exactly 0.0 or 1.0 (single_objective_pso: "clip" keeps the velocity of a particle
that hit the bound), and then the index decides the subset for many particles. `tie_break` makes the rule an
experimental factor; every option is a pure function of the state, so determinism, the subset cache and seeded runs
survive:

    "index"     the default: lower index first;
    "shuffled"  ties ranked by a fixed random permutation of the candidates (`tie_seed`): no relation to the order of
                the data, but every tied particle still decodes the same way, so duplicate decodings remain;
    "hashed"    ties ranked by keys drawn from a generator seeded by a hash of the state itself (and `tie_seed`): two
                states that tie in different ways decode differently, identical states identically. This is
                randomised tie-breaking without a random-number stream: the decoder still consumes none.

Only "index" is the rule the stored results of the rest of the repository assume; the others change `repr(decoder)`, and
with it the problem name, so their runs file under their own identity.

Decoders
--------
`TopKDecoder(n, k)`: S = the k best-ranked candidates. Cardinality is exactly k for every state,
including constant states (then S = {0, ..., k-1} by the tie rule). The set of states that decode
to a given S is the cone {x : min_{i in S} x_i > max_{j not in S} x_j} with the tie rule closing
its boundary - a full-dimensional region, so a positive fraction of state moves leave S unchanged.

`ThresholdDecoder(n, threshold, min_cardinality, max_cardinality)`: S = {i : x_i >= threshold},
then repaired onto [min_cardinality, max_cardinality] with the SAME ranking: too few members ->
the best-ranked non-members are added, too many -> the worst-ranked members are dropped. Repair
rather than rejection keeps every state feasible, which the optimisers require (they have no
notion of an infeasible state). With min_cardinality = 0 the empty subset is a legal decoding
and the objective must define its value; with max_cardinality = n the full set is legal. The
region of states decoding to a given S is a box intersected with the repair rule, so cardinality
varies across the swarm and Jaccard rather than symmetric-difference size is the comparable
distance (operations.py, Stage F).

Invalid states
--------------
A state of the wrong length raises ValueError. A state containing NaN raises ValueError: NaN has
no position in an order, so no deterministic decoding exists, and since the optimisers clip to
the box (single_objective_pso.py:136-140) a NaN state means a broken caller, not a legitimate
point. +-inf are ordered normally and need no special case.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

Subset = Tuple[int, ...]
"""A decoded subset: strictly increasing candidate indices."""


def _as_state(state: Sequence[float], n_candidates: int) -> np.ndarray:
    x = np.asarray(state, dtype=float)
    if x.ndim != 1 or x.size != n_candidates:
        raise ValueError(f"state must have {n_candidates} coordinates, got shape {x.shape}")
    if np.isnan(x).any():
        raise ValueError("state contains NaN: no deterministic decoding exists")
    return x


def rank_candidates(state: Sequence[float]) -> np.ndarray:
    """Candidate indices from best to worst: descending value, lower index first on ties.

    `np.lexsort` sorts by the LAST key first, so (index, -x) ranks by -x and breaks ties by index.
    """
    x = np.asarray(state, dtype=float)
    return np.lexsort((np.arange(x.size), -x))


TIE_RULES = ("index", "shuffled", "hashed")


class TieRule:
    """How candidates with equal values are ranked: a pure function of the state (see the module docstring)."""

    def __init__(self, n_candidates: int, mode: str = "index", seed: int = 0) -> None:
        if mode not in TIE_RULES:
            raise ValueError(f"tie_break must be one of {TIE_RULES}, got {mode!r}")
        self.n_candidates, self.mode, self.seed = int(n_candidates), mode, int(seed)
        self._fixed = (np.random.default_rng([self.seed, self.n_candidates]).random(self.n_candidates)
                       if mode == "shuffled" else None)

    def rank(self, x: np.ndarray) -> np.ndarray:
        """Candidate indices from best to worst; equal values ordered by the rule."""
        if self.mode == "index":
            return rank_candidates(x)
        return np.lexsort((self._key(x), -x))

    def _key(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "shuffled":
            return self._fixed
        digest = hashlib.blake2b(np.ascontiguousarray(x, dtype=np.float64).tobytes(), digest_size=8,
                                 key=self.seed.to_bytes(8, "little", signed=False)).digest()
        return np.random.default_rng(int.from_bytes(digest, "little")).random(x.size)

    def suffix(self) -> str:
        """Appended to a decoder's repr; empty for the default so existing names and identities are unchanged."""
        if self.mode == "index":
            return ""
        return f", tie_break={self.mode!r}" + (f", tie_seed={self.seed}" if self.mode in ("shuffled", "hashed") else "")


@runtime_checkable
class Decoder(Protocol):
    """What a subset problem needs from a decoder. Implementations hold no random state."""

    n_candidates: int

    def decode(self, state: Sequence[float]) -> Subset:
        """The subset a state stands for, as a sorted tuple of candidate indices."""
        ...

    def cardinality_bounds(self) -> Tuple[int, int]:
        """(smallest, largest) cardinality any state can decode to."""
        ...


class TopKDecoder:
    """The k best-ranked candidates; cardinality exactly k."""

    def __init__(self, n_candidates: int, k: int, tie_break: str = "index", tie_seed: int = 0) -> None:
        n_candidates, k = int(n_candidates), int(k)
        if n_candidates < 1:
            raise ValueError(f"n_candidates must be positive, got {n_candidates}")
        if not 1 <= k <= n_candidates:
            raise ValueError(f"k must lie in [1, {n_candidates}], got {k}")
        self.n_candidates = n_candidates
        self.k = k
        self.tie = TieRule(n_candidates, tie_break, tie_seed)

    def decode(self, state: Sequence[float]) -> Subset:
        x = _as_state(state, self.n_candidates)
        chosen = self.tie.rank(x)[:self.k]
        return tuple(int(i) for i in np.sort(chosen))

    def cardinality_bounds(self) -> Tuple[int, int]:
        return self.k, self.k

    def __repr__(self) -> str:
        return f"TopKDecoder(n_candidates={self.n_candidates}, k={self.k}{self.tie.suffix()})"


class ThresholdDecoder:
    """Candidates at or above a threshold, repaired onto a cardinality range by rank."""

    def __init__(self, n_candidates: int, threshold: float = 0.5, min_cardinality: int = 1,
                 max_cardinality: int | None = None, tie_break: str = "index", tie_seed: int = 0) -> None:
        n_candidates = int(n_candidates)
        if n_candidates < 1:
            raise ValueError(f"n_candidates must be positive, got {n_candidates}")
        threshold = float(threshold)
        if np.isnan(threshold):
            raise ValueError("threshold must not be NaN")
        min_cardinality = int(min_cardinality)
        max_cardinality = n_candidates if max_cardinality is None else int(max_cardinality)
        if not 0 <= min_cardinality <= max_cardinality <= n_candidates:
            raise ValueError(f"need 0 <= min_cardinality <= max_cardinality <= {n_candidates}, "
                             f"got {min_cardinality}, {max_cardinality}")
        self.n_candidates = n_candidates
        self.threshold = threshold
        self.min_cardinality = min_cardinality
        self.max_cardinality = max_cardinality
        self.tie = TieRule(n_candidates, tie_break, tie_seed)

    def decode(self, state: Sequence[float]) -> Subset:
        x = _as_state(state, self.n_candidates)
        selected = x >= self.threshold
        count = int(selected.sum())
        if count < self.min_cardinality or count > self.max_cardinality:
            # Repair by rank: members are exactly the first `count` ranked candidates only when
            # the threshold splits the ranking, which it does (x >= t is monotone in x, and ties
            # at the threshold are all selected). Truncating or extending the ranked prefix is
            # therefore "drop the worst members" / "add the best non-members" in one operation.
            target = min(max(count, self.min_cardinality), self.max_cardinality)
            selected = np.zeros(self.n_candidates, dtype=bool)
            selected[self.tie.rank(x)[:target]] = True
        return tuple(int(i) for i in np.flatnonzero(selected))

    def cardinality_bounds(self) -> Tuple[int, int]:
        return self.min_cardinality, self.max_cardinality

    def __repr__(self) -> str:
        return (f"ThresholdDecoder(n_candidates={self.n_candidates}, threshold={self.threshold!r}, "
                f"min_cardinality={self.min_cardinality}, max_cardinality={self.max_cardinality}{self.tie.suffix()})")
