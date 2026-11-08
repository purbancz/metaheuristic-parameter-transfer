"""jSO: the CEC-2017 variant of iL-SHADE with weighted current-to-pBest-w/1 mutation.

Reference
---------
Brest, J., Maučec, M. S., & Bošković, B. (2017). Single objective real-parameter
optimization: Algorithm jSO. In *2017 IEEE Congress on Evolutionary Computation (CEC)*,
pp. 1311-1318. IEEE. https://doi.org/10.1109/CEC.2017.7969456

Implemented rules (Algorithm 1 and Eqs. (1), (3), (4) of the paper), as overrides of the
L-SHADE extension points in `src/metaheuristiczoo/algorithm/sota/lshade.py`:

* **Mutation** current-to-pBest-w/1, Eq. (3): v = x_i + F_w (x_pbest - x_i) + F (x_r1 - x_r2)
  with F_w = 0.7 F while nfes < 0.2 max_nfes, 0.8 F while nfes < 0.4 max_nfes, 1.2 F
  afterwards (Eq. 4).
* **Memory.** H = 5; M_F initialised to 0.3, M_CR to 0.8 (lines 3-4). The last memory
  slot is read as M_F = M_CR = 0.9 (lines 10-13; the slot is never used with another
  value). Update: the weighted Lehmer mean of the successful values is averaged with the
  previous memory content (iL-SHADE feature 2, Sec. II-B); the L-SHADE terminal value for
  M_CR (max S_CR = 0, or already terminal) is kept.
* **CR** (lines 14-23): 0 if M_CR is terminal, else randn(M_CR, 0.1) clipped to [0, 1];
  then CR = max(CR, 0.7) while nfes < 0.25 max_nfes and CR = max(CR, 0.6) while
  nfes < 0.5 max_nfes. Algorithm 1 applies the floors after the terminal-value rule, so an
  early terminal memory still yields CR = 0.7 or 0.6; this is implemented literally.
* **F** (lines 24-27): randc(M_F, 0.1), regenerated while F <= 0, truncated to 1, and capped
  at 0.7 while nfes < 0.6 max_nfes.
* **Greediness** Eq. (1): p = p_min + (p_max - p_min) nfes / max_nfes with p_max = 0.25 and
  p_min = p_max / 2, i.e. p grows from 0.125 to 0.25 over the run. (Sec. IV of the paper
  says "linearly decreases" in prose; Eq. (1) and Algorithm 1 line 42 give the increasing
  schedule, which is what is implemented here.)
* **Population.** N_init = round(25 ln(D) sqrt(D)), linear reduction to N_min = 4 (LPSR, as in
  L-SHADE).
* **Archive.** The paper lists the parameters that differ from L-SHADE / iL-SHADE and the
  archive rate is not among them, so the L-SHADE value r_arc = 2.6 is used; this is an
  inference from the text, not a value printed in the paper (constructor argument
  `archive_size_rate`).
* **Bounds.** The paper says a "repeat mechanism" is applied to out-of-bound variables
  without defining it; the JADE / L-SHADE midpoint rule (bound + parent)/2 is used, as in
  the L-SHADE code base jSO derives from.

The generation's evaluation count at the start of reproduction stands for nfes in the
schedules, so all individuals of one generation share the same F_w factor, CR floor, F cap
and p (the paper's Algorithm 1 tests generation counters g / G_MAX in the same way).
"""

import math
from typing import List, Tuple

import numpy as np
from jmetal.config import store
from jmetal.core.problem import FloatProblem
from jmetal.util.evaluator import Evaluator
from jmetal.util.termination_criterion import StoppingByEvaluations, TerminationCriterion

from metaheuristiczoo.algorithm.sota.lshade import LSHADE


class JSO(LSHADE):
    def __init__(
            self,
            problem: FloatProblem,
            termination_criterion: TerminationCriterion,
            evaluator: Evaluator = store.default_evaluator,
            initial_population_size: int = None,
            memory_size: int = 5,
            p_max: float = 0.25,
            archive_size_rate: float = 2.6,
            initial_memory_f: float = 0.3,
            initial_memory_cr: float = 0.8,
            fixed_memory_value: float = 0.9,
            population_coefficient: float = 25.0,
    ):
        if not isinstance(termination_criterion, StoppingByEvaluations):
            raise ValueError("jSO's schedules (Eqs. 1 and 4, CR floors, F cap) need an evaluation budget: "
                             "pass a StoppingByEvaluations termination criterion.")
        dimension = problem.number_of_variables()
        if initial_population_size is None:
            # N_init = coefficient * ln(D) * sqrt(D); the paper's coefficient is 25 (tunable, dimension-relative)
            initial_population_size = int(round(population_coefficient * math.log(dimension) * math.sqrt(dimension)))
        super().__init__(
            problem=problem,
            termination_criterion=termination_criterion,
            evaluator=evaluator,
            initial_population_size=max(4, initial_population_size),
            memory_size=memory_size,
            p_best_rate=p_max,
            archive_size_rate=archive_size_rate,
        )
        self.p_max = p_max
        self.p_min = p_max / 2
        self.fixed_memory_value = fixed_memory_value
        self.memory_f = [initial_memory_f] * memory_size
        self.memory_cr = [initial_memory_cr] * memory_size

    # -- schedule helpers -------------------------------------------------------------

    def _budget_fraction(self) -> float:
        """nfes / max_nfes at the start of the current generation."""
        return self.evaluations / self.termination_criterion.max_evaluations

    # -- L-SHADE extension points -------------------------------------------------------

    def _memory_values(self, index: int) -> Tuple[float, float]:
        """Lines 10-13: the last memory slot always reads (0.9, 0.9)."""
        if index == self.memory_size - 1:
            return self.fixed_memory_value, self.fixed_memory_value
        return self.memory_cr[index], self.memory_f[index]

    def _sample_cr(self, mu_cr) -> float:
        """Lines 14-23: terminal value / normal sample, then the early-run floors."""
        cr = 0.0 if mu_cr is None else float(np.clip(np.random.normal(mu_cr, 0.1), 0, 1))
        fraction = self._budget_fraction()
        if fraction < 0.25:
            cr = max(cr, 0.7)
        elif fraction < 0.5:
            cr = max(cr, 0.6)
        return cr

    def _sample_f(self, mu_f: float) -> float:
        """Lines 24-27: Cauchy sample as in L-SHADE, capped at 0.7 during the first 60 %."""
        f = super()._sample_f(mu_f)
        if self._budget_fraction() < 0.6 and f > 0.7:
            f = 0.7
        return f

    def _p_best_rate(self) -> float:
        """Eq. (1): p rises linearly from p_max/2 to p_max over the budget."""
        return self.p_min + (self.p_max - self.p_min) * self._budget_fraction()

    def _p_best_weight(self, f: float) -> float:
        """Eq. (4): F_w = 0.7 F, 0.8 F or 1.2 F depending on the consumed budget."""
        fraction = self._budget_fraction()
        if fraction < 0.2:
            return 0.7 * f
        if fraction < 0.4:
            return 0.8 * f
        return 1.2 * f

    def _repair(self, trial_vars: List[float], current_vars: List[float]) -> List[float]:
        """Midpoint rule of JADE / L-SHADE: (bound + parent)/2 for a violating component."""
        lower, upper = self.problem.lower_bound, self.problem.upper_bound
        return [
            (lower[j] + current_vars[j]) / 2 if v < lower[j] else
            (upper[j] + current_vars[j]) / 2 if v > upper[j] else v
            for j, v in enumerate(trial_vars)
        ]

    def _update_memory(self, successful_f: List[float], successful_cr: List[float],
                       fitness_improvements: List[float]) -> None:
        """iL-SHADE update: new entry = (weighted Lehmer mean + old entry) / 2, with the
        L-SHADE terminal value for M_CR."""
        weights = np.array(fitness_improvements) / sum(fitness_improvements)
        sf = np.array(successful_f)
        lehmer_f = float(np.sum(weights * sf ** 2) / np.sum(weights * sf))
        self.memory_f[self.memory_pos] = 0.5 * (lehmer_f + self.memory_f[self.memory_pos])

        scr = np.array(successful_cr)
        if self.memory_cr[self.memory_pos] is None or scr.max() == 0:
            self.memory_cr[self.memory_pos] = None
        else:
            lehmer_cr = float(np.sum(weights * scr ** 2) / np.sum(weights * scr))
            self.memory_cr[self.memory_pos] = 0.5 * (lehmer_cr + self.memory_cr[self.memory_pos])

        self.memory_pos = (self.memory_pos + 1) % self.memory_size

    def get_name(self) -> str:
        return "jSO"
