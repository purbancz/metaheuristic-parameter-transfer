"""SHADE: success-history based adaptive differential evolution without population reduction.

Reference
---------
Tanabe, R., & Fukunaga, A. (2013). Success-history based parameter adaptation for
differential evolution. In *2013 IEEE Congress on Evolutionary Computation (CEC)*,
pp. 71-78. IEEE. https://doi.org/10.1109/CEC.2013.6557555

What SHADE is, relative to the L-SHADE implementation it subclasses
--------------------------------------------------------------------
SHADE is the direct predecessor of L-SHADE (`src/metaheuristiczoo/algorithm/sota/lshade.py`). Both use
current-to-pbest/1/bin with an external archive and a historical memory (M_CR, M_F) of
H entries sampled per individual. The published differences, implemented here as
overrides of the L-SHADE extension points:

* **No population size reduction.** N stays constant (Algorithm 1 of the paper); this
  is the ablation isolating what LPSR contributes to L-SHADE.
* **Per-individual greediness** p_i = rand[p_min, 0.2] with p_min = 2/N (Eq. 20), instead
  of a fixed p.
* **Memory update** (Eqs. 17-19): M_CR by the weighted *arithmetic* mean of S_CR (Eq. 13,
  after Peng et al.), M_F by the weighted Lehmer mean of S_F; weights are the fitness
  improvements. There is no terminal value for M_CR (that was introduced in SHADE 1.1 /
  L-SHADE).
* **Bound handling** by the JADE midpoint rule, Eq. (21): v = (x_min + x_parent)/2 or
  (x_max + x_parent)/2. (The repository's L-SHADE clips instead.)
* **Archive size** |A| = N (Sec. IV-B), memory size H = N by default (the paper's
  experiments use N = H = 100).

Kept from the parent: CR ~ randn(M_CR, 0.1) clipped to [0, 1], F ~ randc(M_F, 0.1)
regenerated while F <= 0 and truncated to 1, r1 from P and r2 from P u A, and the
parent's tie rule (a trial replaces its parent, and enters S_CR / S_F, only when it is
strictly better; the paper's Eq. (6) keeps the trial on ties as well, an event of measure
zero on continuous objectives).
"""

import random
from typing import List

import numpy as np
from jmetal.config import store
from jmetal.core.problem import FloatProblem
from jmetal.util.evaluator import Evaluator
from jmetal.util.termination_criterion import TerminationCriterion

from metaheuristiczoo.algorithm.sota.lshade import LSHADE


class SHADE(LSHADE):
    def __init__(
            self,
            problem: FloatProblem,
            termination_criterion: TerminationCriterion,
            evaluator: Evaluator = store.default_evaluator,
            population_size: int = 100,
            memory_size: int = None,
            archive_size_rate: float = 1.0,
            p_max: float = 0.2,
            population_factor: float = None,
    ):
        # `population_factor` gives the dimension-relative form N = round(factor * D) used by the
        # tuner (a tuned absolute N would not transfer from D = 100 to D = 1000).
        if population_factor is not None:
            population_size = max(4, int(round(population_factor * problem.number_of_variables())))
        super().__init__(
            problem=problem,
            termination_criterion=termination_criterion,
            evaluator=evaluator,
            initial_population_size=population_size,
            memory_size=population_size if memory_size is None else memory_size,
            p_best_rate=p_max,
            archive_size_rate=archive_size_rate,
        )
        self.p_max = p_max
        self.p_min = 2.0 / population_size

    def _p_best_rate(self) -> float:
        """Eq. (20): p_i = rand[p_min, p_max], p_min = 2/N (at least two candidates)."""
        return random.uniform(self.p_min, self.p_max)

    def _repair(self, trial_vars: List[float], current_vars: List[float]) -> List[float]:
        """Eq. (21): a violating component is moved halfway between the bound and the parent."""
        lower, upper = self.problem.lower_bound, self.problem.upper_bound
        return [
            (lower[j] + current_vars[j]) / 2 if v < lower[j] else
            (upper[j] + current_vars[j]) / 2 if v > upper[j] else v
            for j, v in enumerate(trial_vars)
        ]

    def _update_memory(self, successful_f: List[float], successful_cr: List[float],
                       fitness_improvements: List[float]) -> None:
        """Eqs. (17)-(19): weighted arithmetic mean for M_CR, weighted Lehmer mean for M_F."""
        weights = np.array(fitness_improvements) / sum(fitness_improvements)
        sf = np.array(successful_f)
        scr = np.array(successful_cr)
        self.memory_f[self.memory_pos] = float(np.sum(weights * sf ** 2) / np.sum(weights * sf))
        self.memory_cr[self.memory_pos] = float(np.sum(weights * scr))
        self.memory_pos = (self.memory_pos + 1) % self.memory_size

    def _next_population_size(self, current_evaluations: int) -> int:
        """SHADE keeps N constant."""
        return self.population_size

    def get_name(self) -> str:
        return "SHADE"
