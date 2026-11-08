"""The objective interface of a subset problem: a FloatProblem whose objective scores decode(x).

    class MyProblem(SubsetProblem):
        def __init__(self, table):
            super().__init__(candidates=table.ids, decoder=TopKDecoder(len(table.ids), k=5))
            self.values = table.values
        def evaluate_subset(self, subset):            # subset: sorted tuple of candidate indices
            return -float(sum(self.values[i] for i in subset))

`evaluate(solution)` decodes `solution.variables`, scores the subset, writes `objectives[0]` and
returns the solution - the contract every optimiser, observer and result writer of the repository
relies on. Nothing else about the optimisers changes: the box is [0, 1]^n, `create_solution` is
jMetalPy's uniform draw, `number_of_variables()` is the candidate count.

Evaluation accounting (why the cache counts hits)
-------------------------------------------------
jMetalPy's `algorithm.evaluations` is an arithmetic counter stepped by `swarm_size` per
generation, not a tally of `problem.evaluate` calls, and the repository's budget tests assert
`algorithm.evaluations == problem.evaluation_count == max_evaluations`
(tests/algorithm/basic/test_pso_core.py). A cache that skipped `evaluate` calls would silently
give a cached problem a larger effective budget than an uncached one and break comparability
with every other problem in the zoo. So:

* `evaluation_count`   increments on EVERY `evaluate` call, cache hit or miss - the budget;
* `subset_evaluations` counts calls to `evaluate_subset`, i.e. objective computations actually
                       performed - a cost / speed-up diagnostic, never a budget;
* `cache_hits`         = evaluation_count - subset_evaluations when the cache is on;
* `distinct_subsets`   subsets decoded at least once since the last reset.

A cached and an uncached instance therefore return the same objective sequence under the same
evaluation budget; only `subset_evaluations` differs (tested).

Determinism and the cache
-------------------------
The cache is keyed on the decoded tuple and is only used when `deterministic=True` (default).
A stochastic objective (fresh noise per evaluation) must
declare `deterministic=False`: the cache is then never consulted and every evaluation calls
`evaluate_subset`, while `distinct_subsets` is still tracked. `set_seed(seed)` - called by the
runner before every run when present - clears the cache and every counter, so a run never sees
values computed under a previous run's seed; subclasses with their own noise stream override it
and call `super().set_seed(seed)`.

Memory: the cache (or the visited set) holds one entry per distinct subset - at most one per
evaluation. For a campaign-scale budget this is bounded by the budget; pass `cache=False,
track_subsets=False` to hold nothing.

Constraints and finite objectives
---------------------------------
The experiment runner refuses non-finite objectives (experiment/running/runner.py:28-41 raises
UndefinedObjective). An infeasible subset must therefore be scored by a FINITE penalty inside
`evaluate_subset`; `evaluate` raises ValueError on a non-finite or non-numeric return so that the
mistake surfaces in a unit test rather than in a campaign worker. jMetalPy's constraint slots
(`number_of_constraints`) stay at 0 because the runner, writer and statistics ignore
`solution.constraints`. Cardinality constraints are the decoder's business, not the objective's.

Several values (extension point)
--------------------------------
The whole experiment stack is single-objective (`objectives[0]` in core/observer.py,
experiment/results/writer.py, experiment/statistics). `evaluate_subset` therefore returns one
float. An application whose score is a combination of components may additionally implement
`evaluate_subset_components(subset) -> Mapping[str, float]`; when it does, the components of the
most recent objective computation are kept as `last_components` and reported by
`subset_snapshot()`. Nothing is collapsed on the application's behalf: the scalar returned by
`evaluate_subset` is the only thing the optimiser sees, and multi-objective search is not faked.

Picklability
------------
The campaign runner deep-copies and pickles the problem into worker processes
(experiment/running/runner.py:259-260). The objective is a METHOD of a subclass for that reason;
a callable passed to the constructor would work in tests and fail in the pool.
"""

from __future__ import annotations

import math
import numbers
from abc import ABC, abstractmethod
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

from metaheuristiczoo.problem.subset.decode import Decoder, Subset


class SubsetProblem(FloatProblem, ABC):
    """A single-objective FloatProblem on [0, 1]^n scoring the subset a state decodes to."""

    def __init__(self, candidates: Sequence[Hashable], decoder: Decoder, *, cache: bool = True,
                 deterministic: bool = True, track_subsets: bool = True, name: Optional[str] = None) -> None:
        super().__init__()
        candidates = tuple(candidates)
        if not candidates:
            raise ValueError("a subset problem needs at least one candidate")
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidate identifiers must be unique")
        if not isinstance(decoder, Decoder):
            raise TypeError("decoder must provide n_candidates, decode(state) and cardinality_bounds()")
        if decoder.n_candidates != len(candidates):
            raise ValueError(f"decoder is for {decoder.n_candidates} candidates, {len(candidates)} given")
        self.candidates: Tuple[Hashable, ...] = candidates
        self.decoder = decoder
        self.deterministic = bool(deterministic)
        self.cache_enabled = bool(cache) and self.deterministic
        self.track_subsets = bool(track_subsets)
        self._name = name

        n = len(candidates)
        self.lower_bound = [0.0] * n
        self.upper_bound = [1.0] * n
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(subset)"]

        self.seed: Optional[int] = None
        self.reset_accounting()

    # -- the application's part ----------------------------------------------------------------

    @abstractmethod
    def evaluate_subset(self, subset: Subset) -> float:
        """The objective value (minimised) of a subset given as sorted candidate indices.

        Must return a finite real number; score an infeasible subset with a finite penalty."""

    def evaluate_subset_components(self, subset: Subset) -> Optional[Mapping[str, float]]:
        """Optional: the named components the scalar objective is built from. Default: none."""
        return None

    # -- jMetalPy protocol ---------------------------------------------------------------------

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution.objectives[0] = self._counted(self.decoder.decode(solution.variables))
        return solution

    def name(self) -> str:
        if self._name is not None:
            return self._name
        return f"{type(self).__name__}(n={self.number_of_variables()}, {self.decoder!r})"

    # -- scoring ---------------------------------------------------------------------------------

    def _counted(self, subset: Subset) -> float:
        """One budgeted evaluation of a subset: counted whether or not the cache answers it."""
        self.evaluation_count += 1
        self.cardinality_total += len(subset)
        self.last_subset = subset
        if self.cache_enabled and subset in self._cache:
            self.cache_hits += 1
            return self._cache[subset]
        value = self._score(subset)
        if self.cache_enabled:
            self._cache[subset] = value
        elif self.track_subsets:
            self._seen.add(subset)
        return value

    def _score(self, subset: Subset) -> float:
        self.subset_evaluations += 1
        raw = self.evaluate_subset(subset)
        if isinstance(raw, bool) or not isinstance(raw, numbers.Real):
            raise ValueError(f"evaluate_subset must return a real number, got {raw!r}")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"evaluate_subset returned {value!r} for subset {subset}; the experiment "
                             "runner refuses non-finite objectives - score infeasibility with a finite penalty")
        components = self.evaluate_subset_components(subset)
        if components is not None:
            self.last_components = {str(k): float(v) for k, v in components.items()}
        return value

    # -- helpers for applications, baselines and diagnostics -------------------------------------

    def decode(self, state: Sequence[float]) -> Subset:
        """The subset a state stands for (no evaluation, no accounting)."""
        return self.decoder.decode(state)


    def identifiers(self, subset: Subset) -> Tuple[Hashable, ...]:
        """Candidate identifiers of a subset of indices."""
        return tuple(self.candidates[i] for i in subset)

    def score_subset(self, subset: Subset) -> float:
        """Score a subset directly, with the same cache, counting and validation as `evaluate`.

        For baselines and exact methods that search subset space rather than the box; counted in
        `evaluation_count` so their budgets are comparable with the optimisers'."""
        subset = tuple(int(i) for i in subset)
        if list(subset) != sorted(set(subset)):
            raise ValueError(f"a subset is a strictly increasing tuple of candidate indices, got {subset}")
        if subset and (subset[0] < 0 or subset[-1] >= self.number_of_variables()):
            raise ValueError(f"subset {subset} has indices outside [0, {self.number_of_variables()})")
        return self._counted(subset)

    # -- accounting ------------------------------------------------------------------------------

    def reset_accounting(self) -> None:
        """Clear the cache and every counter (a new run)."""
        self.evaluation_count = 0
        self.subset_evaluations = 0
        self.cache_hits = 0
        self.cardinality_total = 0
        self.last_subset: Optional[Subset] = None
        self.last_components: Optional[Dict[str, float]] = None
        self._cache: Dict[Subset, float] = {}
        self._seen: set = set()

    def set_seed(self, seed: int) -> None:
        """Called by the experiment runner before each run: a new run starts with an empty cache."""
        self.seed = int(seed)
        self.reset_accounting()

    @property
    def distinct_subsets(self) -> int:
        return len(self._cache) if self.cache_enabled else len(self._seen)

    @property
    def duplicate_decodes(self) -> int:
        """Evaluations whose subset had been decoded before (representation-induced neutrality)."""
        if self.cache_enabled or self.track_subsets:
            return self.evaluation_count - self.distinct_subsets
        return 0

    def subset_snapshot(self) -> Dict[str, Any]:
        """Diagnostics of the representation: evaluations, objective computations, neutrality, size."""
        n_eval = self.evaluation_count
        return {
            "evaluation_count": n_eval,
            "subset_evaluations": self.subset_evaluations,
            "cache_hits": self.cache_hits,
            "distinct_subsets": self.distinct_subsets if (self.cache_enabled or self.track_subsets) else None,
            "duplicate_decodes": self.duplicate_decodes if (self.cache_enabled or self.track_subsets) else None,
            "duplicate_rate": (self.duplicate_decodes / n_eval) if n_eval and (self.cache_enabled or self.track_subsets) else None,
            "mean_cardinality": (self.cardinality_total / n_eval) if n_eval else None,
            "cardinality_bounds": self.decoder.cardinality_bounds(),
            "last_subset": self.last_subset,
            "last_components": dict(self.last_components) if self.last_components else None,
        }
