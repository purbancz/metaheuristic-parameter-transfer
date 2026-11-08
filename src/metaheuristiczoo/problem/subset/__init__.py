"""Subset (combinatorial) decision spaces behind continuous optimisers.

The optimisers move a continuous state x in a box and are not changed; the solution is made discrete:

    x in [0, 1]^n,  one coordinate per candidate          (the state the optimiser moves)
    S = decode(x)   a subset of {0, ..., n-1}              (the solution the objective scores)
    f(x) = evaluate_subset(S)

`decode` is deterministic and RNG-free, so a subset problem consumes exactly the random numbers any other
FloatProblem consumes and the optimisers run on it unchanged.

Modules
-------
- decode.py         : the `Decoder` protocol and the deterministic decoders (`TopKDecoder`, `ThresholdDecoder`)
                      with explicit tie, cardinality and invalid-state rules.
- subset_problem.py : `SubsetProblem(FloatProblem)`, the objective interface (`evaluate_subset`), with a cache that
                      never reduces the evaluation budget, run-seed reset and finite-objective validation.
- operations.py     : subset-space operations for the baselines - add / remove / swap, neighbourhoods,
                      reproducible random subsets. Not used by the optimisers.
- baselines.py      : random sampling, greedy, 1-swap local search and restarted local search, all through
                      `score_subset`, so their budgets compare with the optimisers'.
- datasets/         : pinned data files and their verification, and `CoverageSubsetProblem`, the number of demand
                      points a subset of sites covers.
- transfer/         : the ten maximum-coverage instances of the study, their networks and certified optima.
"""

from metaheuristiczoo.problem.subset.decode import (TIE_RULES, Decoder, Subset, ThresholdDecoder, TieRule,
                                                     TopKDecoder, rank_candidates)
from metaheuristiczoo.problem.subset.subset_problem import SubsetProblem
from metaheuristiczoo.problem.subset.baselines import BaselineResult

__all__ = ["BaselineResult", "Decoder", "Subset", "SubsetProblem", "TIE_RULES", "ThresholdDecoder", "TieRule",
           "TopKDecoder", "rank_candidates"]
