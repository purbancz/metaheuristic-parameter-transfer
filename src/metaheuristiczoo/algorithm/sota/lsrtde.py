"""L-SRTDE: success-rate-based adaptive differential evolution (CEC-2024 winner).

Reference
---------
Stanovov, V., & Semenkin, E. (2024). Success rate-based adaptive differential evolution
L-SRTDE for CEC 2024 competition. In *2024 IEEE Congress on Evolutionary Computation
(CEC)*, pp. 1-8. IEEE. https://doi.org/10.1109/CEC60901.2024.10611907

The implementation follows the author's released C++ source (repository
VladimirStanovov/L-SRTDE_CEC-2024, file L-SRTDE.cpp, GPL-3.0), function
Optimizer::MainCycle, which is the code the competition results were produced with. The
generic names below map to the source as: `top` = Popul (the pool that is re-sorted and
truncated to the best N every generation), `front` = PopulFront (a cyclic buffer of size N
that receives every successful trial), `success_rate` = SuccessRate.

Algorithm (one generation, current population size N)
-----------------------------------------------------
1. mean_F = 0.4 + 0.25 tanh(5 SR); `p` set size ps = max(2, int(0.7 N exp(-7 SR))), where SR
   is the previous generation's success rate (initially 0.5).
2. Ranks: `top` sorted ascending by fitness (indices I), `front` sorted ascending (indices
   I2); rank weights w_k = exp(-3 k / N) over the sorted `front`.
3. For each of N trials: the target t is a *uniformly random* member of `front` (not a
   sweep); memory slot h uniform in [0, H); pbest = I[U(0, ps)] from `top`, redrawn while its
   index equals t; r1 = I2[rank-weighted draw] from `front`, redrawn while r1 == pbest;
   r2 = I[U(0, N)] from `top`, redrawn while r2 in {pbest, r1}. (The index comparisons across
   the two pools are as in the source.) F ~ N(mean_F, 0.02) redrawn outside [0, 1];
   Cr ~ N(M_Cr[h], 0.05) clipped to [0, 1]. Binomial crossover with one forced component:
   u_j = x_t,j + F (top_pbest,j - x_t,j) + F (front_r1,j - top_r2,j); a component outside the
   box is redrawn uniformly in [low_j, high_j]. The realised crossover rate (fraction of
   mutated components) is what enters the memory.
4. Acceptance: if f(u) <= f(x_t): u is appended to `top` (position N + successes) and
   written into `front` at the cyclic position PF (which is then advanced), and
   (realised Cr, |f(x_t) - f(u)|) recorded. As in the source, the fitness difference is read
   after the `front` write, so it is 0 when PF == t.
5. End of generation: SR = successes / N; N' = int(((4 - N_init) / max_nfes) nfes + N_init);
   `front` drops its worst N - N' members; M_Cr[k] = (M_Cr[k] + mean_WL(S_Cr))/2 when there
   were successes (weights = fitness differences; mean_WL returns 1 when the weighted sum
   of Cr is ~0), then k advances cyclically; `top` = the best N' of its first N' + successes
   entries (again literally as in the source: the truncation bound uses the *new* N').

Parameters (source defaults): N_init = 20 D, N_min = 4, H = 5, M_Cr initialised to 1.0,
sigma_F = 0.02, sigma_Cr = 0.05, SR_0 = 0.5. Bounds are handled per coordinate with the
problem's box (the source hard-codes [-100, 100]). Evaluations are counted exactly against
the budget; a generation is cut short when the budget runs out.
"""

import math
import random
import time
from typing import List, TypeVar

import numpy as np
from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.evaluator import Evaluator
from jmetal.util.termination_criterion import StoppingByEvaluations, TerminationCriterion

S = TypeVar("S", bound=FloatSolution)
R = TypeVar("R")


class LSRTDE(Algorithm[S, R]):
    def __init__(
            self,
            problem: FloatProblem,
            termination_criterion: TerminationCriterion,
            evaluator: Evaluator = store.default_evaluator,
            population_factor: float = 20,
            initial_population_size: int = None,
            min_population_size: int = 4,
            memory_size: int = 5,
            sigma_f: float = 0.02,
            sigma_cr: float = 0.05,
            initial_memory_cr: float = 1.0,
            initial_success_rate: float = 0.5,
    ):
        super().__init__()
        if not isinstance(termination_criterion, StoppingByEvaluations):
            raise ValueError("L-SRTDE's linear population size reduction needs an evaluation budget: "
                             "pass a StoppingByEvaluations termination criterion.")
        self.problem = problem
        self.termination_criterion = termination_criterion
        self.evaluator = evaluator
        self.observable.register(self.termination_criterion)

        self.dimension = problem.number_of_variables()
        self.lower = np.array(problem.lower_bound, dtype=float)
        self.upper = np.array(problem.upper_bound, dtype=float)
        self.max_evaluations = termination_criterion.max_evaluations

        self.initial_population_size = (int(round(population_factor * self.dimension))
                                        if initial_population_size is None else initial_population_size)
        self.initial_population_size = max(min_population_size, self.initial_population_size)
        self.population_size = self.initial_population_size
        self.min_population_size = min_population_size
        self.memory_size = memory_size
        self.sigma_f = sigma_f
        self.sigma_cr = sigma_cr
        self.memory_cr = [initial_memory_cr] * memory_size
        self.memory_pos = 0
        self.success_rate = initial_success_rate

        # `top`: capacity 2 N_init (N members plus at most N successful trials per generation)
        self.top = np.empty((2 * self.initial_population_size, self.dimension))
        self.top_fit = np.full(2 * self.initial_population_size, np.inf)
        self.top_count = 0  # NIndsCurrent
        self.front = np.empty((self.initial_population_size, self.dimension))
        self.front_fit = np.full(self.initial_population_size, np.inf)
        self.front_write_pos = 0  # PFIndex

        self.best_solution = None
        self.generations = 0
        self.evaluations_in_current_step = 0

    # -- jMetal lifecycle ------------------------------------------------------------------

    def create_initial_solutions(self) -> List[S]:
        return [self.problem.create_solution() for _ in range(self.initial_population_size)]

    def evaluate(self, solution_list: List[S]) -> List[S]:
        return self.evaluator.evaluate(solution_list, self.problem)

    def init_progress(self) -> None:
        n = self.initial_population_size
        for i, solution in enumerate(self.solutions):
            self.top[i] = solution.variables
            self.top_fit[i] = solution.objectives[0]
            self._record_best(self.top[i], self.top_fit[i])
        self.top_count = n
        order = np.argsort(self.top_fit[:n], kind="stable")
        self.front[:n] = self.top[order]
        self.front_fit[:n] = self.top_fit[order]
        self.evaluations = n
        self.observable.notify_all(**self.observable_data())

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met or self.evaluations >= self.max_evaluations

    def step(self) -> None:
        n = self.population_size
        mean_f = 0.4 + 0.25 * math.tanh(5.0 * self.success_rate)
        p_size = max(2, int(n * 0.7 * math.exp(-7.0 * self.success_rate)))

        top_rank = np.argsort(self.top_fit[:n], kind="stable")          # Indices
        front_rank = np.argsort(self.front_fit[:n], kind="stable")      # Indices2
        rank_weights = np.exp(-3.0 * np.arange(n) / n)
        cum_weights = np.cumsum(rank_weights).tolist()
        rank_positions = list(range(n))

        success_cr: List[float] = []
        success_delta: List[float] = []
        evaluated = 0
        for _ in range(n):
            if self.evaluations + evaluated >= self.max_evaluations:
                break
            target = random.randrange(n)
            memory_index = random.randrange(self.memory_size)
            while True:
                p_best = top_rank[random.randrange(p_size)]
                if p_best != target:
                    break
            while True:
                r1 = front_rank[random.choices(rank_positions, cum_weights=cum_weights)[0]]
                if r1 != p_best:
                    break
            while True:
                r2 = top_rank[random.randrange(n)]
                if r2 != p_best and r2 != r1:
                    break
            while True:
                f = random.gauss(mean_f, self.sigma_f)
                if 0.0 <= f <= 1.0:
                    break
            cr = min(max(random.gauss(self.memory_cr[memory_index], self.sigma_cr), 0.0), 1.0)

            x_target = self.front[target]
            mutant = x_target + f * (self.top[p_best] - x_target) + f * (self.front[r1] - self.top[r2])
            forced = random.randrange(self.dimension)
            mask = np.random.random(self.dimension) < cr
            mask[forced] = True
            trial = np.where(mask, mutant, x_target)
            outside = mask & ((trial < self.lower) | (trial > self.upper))
            if outside.any():
                trial[outside] = self.lower[outside] + np.random.random(int(outside.sum())) * (
                        self.upper[outside] - self.lower[outside])
            realised_cr = float(mask.sum()) / self.dimension

            trial_fit = self._evaluate_vector(trial)
            evaluated += 1
            if trial_fit <= self.front_fit[target]:
                successes = len(success_cr)
                self.top[self.top_count + successes] = trial
                self.top_fit[self.top_count + successes] = trial_fit
                self.front[self.front_write_pos] = trial
                self.front_fit[self.front_write_pos] = trial_fit
                success_cr.append(realised_cr)
                success_delta.append(abs(float(self.front_fit[target]) - trial_fit))  # after the write, as in the source
                self.front_write_pos = (self.front_write_pos + 1) % n

        self.evaluations_in_current_step = evaluated
        successes = len(success_cr)
        self.success_rate = successes / n

        nfes = self.evaluations + evaluated
        new_n = int((self.min_population_size - self.initial_population_size) / self.max_evaluations * nfes
                    + self.initial_population_size)
        new_n = max(self.min_population_size, min(new_n, n))
        self._remove_worst_from_front(n, new_n)
        self.population_size = new_n

        if successes:
            self.memory_cr[self.memory_pos] = 0.5 * (self._weighted_lehmer(success_cr, success_delta)
                                                     + self.memory_cr[self.memory_pos])
            self.memory_pos = (self.memory_pos + 1) % self.memory_size

        self.top_count = new_n + successes
        self.generations += 1
        if self.top_count > new_n:
            order = np.argsort(self.top_fit[:self.top_count], kind="stable")[:new_n]
            self.top[:new_n] = self.top[order]
            self.top_fit[:new_n] = self.top_fit[order]
            self.top_count = new_n
        # In the source the cyclic write position is not reset when N shrinks; a position
        # beyond the new size is folded back so the buffer stays in range.
        self.front_write_pos %= new_n

    def update_progress(self) -> None:
        self.evaluations += self.evaluations_in_current_step
        self.observable.notify_all(**self.observable_data())

    # -- helpers ---------------------------------------------------------------------------

    def _evaluate_vector(self, variables: np.ndarray) -> float:
        solution = FloatSolution(self.problem.lower_bound, self.problem.upper_bound,
                                 self.problem.number_of_objectives(), self.problem.number_of_constraints())
        solution.variables = variables.tolist()
        self.evaluate([solution])
        value = solution.objectives[0]
        self._record_best(variables, value, solution)
        return value

    def _record_best(self, variables: np.ndarray, value: float, solution: FloatSolution = None) -> None:
        if self.best_solution is None or value < self.best_solution.objectives[0]:
            if solution is None:
                solution = FloatSolution(self.problem.lower_bound, self.problem.upper_bound,
                                         self.problem.number_of_objectives(), self.problem.number_of_constraints())
                solution.variables = list(variables)
                solution.objectives[0] = value
            self.best_solution = solution

    def _remove_worst_from_front(self, n: int, new_n: int) -> None:
        """RemoveWorst: drop the n - new_n worst members of `front`, preserving order."""
        if new_n >= n:
            return
        keep = np.argsort(self.front_fit[:n], kind="stable")[:new_n]
        keep.sort()
        self.front[:new_n] = self.front[keep]
        self.front_fit[:new_n] = self.front_fit[keep]

    @staticmethod
    def _weighted_lehmer(values: List[float], weights: List[float]) -> float:
        """MeanWL of the source: weighted Lehmer mean; 1.0 when the weighted sum is ~0
        (which includes the all-zero-weights case that the C code falls through to)."""
        total = float(sum(weights))
        if total <= 0.0:
            return 1.0
        w = np.array(weights) / total
        v = np.array(values)
        weighted_sum = float(np.sum(w * v))
        if abs(weighted_sum) > 1e-8:
            return float(np.sum(w * v * v) / weighted_sum)
        return 1.0

    # -- reporting -------------------------------------------------------------------------

    def result(self) -> R:
        return self.best_solution

    def get_name(self) -> str:
        return "L-SRTDE"

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "POPULATION_SIZE": self.population_size,
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }
