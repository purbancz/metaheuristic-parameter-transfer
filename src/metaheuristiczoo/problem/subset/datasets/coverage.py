"""A subset problem whose objective depends on the candidates jointly: the weight of the union of features.

    problem = build_instance("SJC324", p=2, radius=800, root=root)       # problem/subset/transfer/mclp.py

A candidate "reaches" a feature when its stored value is at least the cut (`threshold`, or the `quantile`
of the stored values); the objective of a subset is the total weight of the features reached by at least
one member. On the study's instances a candidate is a site, a feature is a demand node, a site reaches the
nodes within the coverage radius, and every weight is 1, so the objective is the number of covered nodes
(unweighted maximum coverage, `direction="maximize"`: monotone submodular, greedy is within 1 - 1/e of the
optimum). The problem is deterministic, so the base class's subset cache is valid.

Constraints: an optional linear cost cap on a candidate-table column, scored as a finite penalty
(not used by the study's instances). Anything else (required or forbidden candidates, exclusive
groups) is not implemented; a group column of the candidate table is there for a subclass to use.

Nothing about what a feature is enters here; `feature_weights` is a plain non-negative vector (for
example a measured per-feature quantity the researcher decides to use).
"""

from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import numpy as np

from metaheuristiczoo.problem.subset.datasets import reference_values
from metaheuristiczoo.problem.subset.datasets.common import CandidateMatrix
from metaheuristiczoo.problem.subset.decode import Decoder, Subset, ThresholdDecoder, TopKDecoder
from metaheuristiczoo.problem.subset.subset_problem import SubsetProblem

DIRECTIONS = ("minimize", "maximize")


class CoverageSubsetProblem(SubsetProblem):
    """Select candidates to maximise (or minimise) the weight of the features the selection reaches."""

    def __init__(self, matrix: CandidateMatrix, *, threshold: Optional[float] = None, quantile: Optional[float] = None,
                 direction: str = "maximize", k: Optional[int] = None, min_cardinality: Optional[int] = None,
                 max_cardinality: Optional[int] = None, decoder_threshold: float = 0.5, decoder: Optional[Decoder] = None,
                 feature_weights: Optional[Sequence[float]] = None, cost_column: Optional[str] = None,
                 cost_cap: Optional[float] = None, penalty_weight: float = 1.0, label: Optional[str] = None,
                 **kwargs) -> None:
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
        if (cost_column is None) != (cost_cap is None):
            raise ValueError("cost_column and cost_cap go together")
        if penalty_weight < 0:
            raise ValueError("penalty_weight must be non-negative")
        reach, cut = matrix.binary(threshold, quantile)
        n, m = reach.shape
        weights = np.ones(m) if feature_weights is None else np.asarray(feature_weights, dtype=float)
        if weights.shape != (m,) or not np.all(np.isfinite(weights)) or (weights < 0).any():
            raise ValueError(f"feature_weights must be {m} finite non-negative numbers")
        costs = None
        if cost_column is not None:
            if cost_column not in matrix.candidate_table.columns:
                raise ValueError(f"column {cost_column!r} not in the candidate table; columns are "
                                 f"{list(matrix.candidate_table.columns)}")
            costs = matrix.candidate_table[cost_column].to_numpy(dtype=float)
            if not np.all(np.isfinite(costs)):
                raise ValueError(f"column {cost_column!r} has missing values; fill or restrict the candidates first")

        if decoder is None:
            if k is not None:
                if min_cardinality is not None or max_cardinality is not None:
                    raise ValueError("give either k or a cardinality range, not both")
                decoder = TopKDecoder(n, k)
            else:
                decoder = ThresholdDecoder(n, decoder_threshold, 1 if min_cardinality is None else min_cardinality,
                                           max_cardinality)
        elif k is not None or min_cardinality is not None or max_cardinality is not None:
            raise ValueError("give either a decoder or k / a cardinality range, not both")

        super().__init__(matrix.candidates, decoder, **kwargs)
        self.key, self.cut, self.direction = matrix.key, cut, direction
        self.sign = -1.0 if direction == "maximize" else 1.0
        self.indptr, self.indices = reach.indptr.copy(), reach.indices.copy()
        self.n_features = m
        self.weights = weights
        self.unit_weights = feature_weights is None
        self.cost_column, self.costs = cost_column, costs
        self.cost_cap = None if cost_cap is None else float(cost_cap)
        self.penalty_weight = float(penalty_weight)
        self.label = label
        self.instance_id = self._content_hash(matrix.key, matrix.candidates, matrix.features)
        aliases = matrix.notes.get("reference_key_aliases", ())
        self.reference_instance_ids = (self.instance_id,) + tuple(
            self._content_hash(str(alias), matrix.candidates, matrix.features) for alias in aliases)
        self._sidecar_reference_values = dict(matrix.notes.get("_reference_values", {}))

    def _content_hash(self, key: str, candidates, features) -> str:
        digest = hashlib.sha256()
        digest.update(repr((key, tuple(candidates), tuple(features))).encode("utf-8"))
        for array in (self.indptr, self.indices, self.weights):
            digest.update(np.ascontiguousarray(array).tobytes())
        if self.costs is not None:
            digest.update(b"cost")
            digest.update(np.ascontiguousarray(self.costs, dtype=np.float64).tobytes())
        return digest.hexdigest()

    # -- the objective ----------------------------------------------------------------------------

    def reached(self, subset: Subset) -> np.ndarray:
        """Sorted indices of the features reached by at least one member of the subset."""
        if not subset:
            return np.empty(0, dtype=self.indices.dtype)
        return np.unique(np.concatenate([self.indices[self.indptr[i]:self.indptr[i + 1]] for i in subset]))

    def aggregate(self, subset: Subset) -> float:
        """The weight of the union, in the researcher's own sign."""
        hit = self.reached(subset)
        return float(hit.size) if self.unit_weights else float(self.weights[hit].sum())

    def cost(self, subset: Subset) -> Optional[float]:
        if self.costs is None:
            return None
        return float(self.costs[list(subset)].sum()) if subset else 0.0

    def penalty(self, subset: Subset) -> float:
        if self.costs is None:
            return 0.0
        return self.penalty_weight * max(0.0, self.cost(subset) - self.cost_cap)

    def evaluate_subset(self, subset: Subset) -> float:
        return self.sign * self.aggregate(subset) + self.penalty(subset)

    def evaluate_subset_components(self, subset: Subset):
        components = {"aggregate": self.aggregate(subset)}
        if self.costs is not None:
            components.update(cost=self.cost(subset), penalty=self.penalty(subset))
        return components


    def reference(self) -> Optional[dict]:
        """The stored reference entry for this instance, direction and k (see reference_values), if there is one.

        Only a plain instance is looked up: exact cardinality, unit weights, no cost cap - the entries were computed
        for that and mean nothing for a variant."""
        if self.costs is not None or not self.unit_weights or not isinstance(self.decoder, TopKDecoder):
            return None
        return reference_values.find_in(self._sidecar_reference_values, self.reference_instance_ids,
                                        self.direction, self.decoder.k)

    def known_optimum(self) -> Optional[float]:
        """The optimum in objective (minimised) sign when it was enumerated or proven; else None.

        A best-known value that is not proven optimal is NOT returned here: the result writer stores this as the
        error scale, and a reference that may be beaten would turn a good result into a negative error."""
        entry = self.reference()
        return None if entry is None or entry["status"] not in reference_values.EXACT else self.sign * entry["value"]

    def best_known(self) -> Optional[float]:
        """The stored best value in objective sign whatever its status (optimum or not); None if none is stored."""
        entry = self.reference()
        return None if entry is None else self.sign * entry["value"]

    # -- reporting --------------------------------------------------------------------------------

    def describe(self) -> dict:
        lo, hi = self.decoder.cardinality_bounds()
        return {"label": self.label, "dataset": self.key, "n_candidates": self.number_of_variables(),
                "n_features": self.n_features, "cut": self.cut, "reach_density": round(
                    len(self.indices) / (self.number_of_variables() * self.n_features), 4),
                "objective": "union", "direction": self.direction, "cardinality": (lo, hi),
                "decoder": repr(self.decoder), "weighted": not self.unit_weights, "cost_column": self.cost_column,
                "cost_cap": self.cost_cap, "instance_id": self.instance_id, "known_optimum": self.known_optimum(),
                "reference_status": (self.reference() or {}).get("status"), "deterministic": self.deterministic}

    def name(self) -> str:
        if self._name is not None:
            return self._name
        label = f"{self.label}:" if self.label else f"{self.key}:"
        return f"CoverageSubsetProblem({label}cut={self.cut:g},{self.direction},{self.decoder!r})"
