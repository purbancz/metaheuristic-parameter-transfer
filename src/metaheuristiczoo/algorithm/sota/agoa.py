"""Adversarial Game Optimization Algorithm (AGOA).

Reference
---------
C. Li, Q. Zhang, J. Li, S. Tao, D. Oliva: "Adversarial game optimization: A
game-theoretic metaheuristic for efficient complex optimization and
engineering applications", Information Sciences 735 (2026) 123022.
https://doi.org/10.1016/j.ins.2025.123022. Reference MATLAB code:
https://github.com/tsingke/AGOA (AGOA.m, Q. Zhang, SDNU CILab).

Algorithm (Sec. 2, Algorithm 1 and AGOA.m)
------------------------------------------
Every generation the population is sorted and split into elite (top 20 %),
ordinary (middle 60 %) and worst (bottom 20 %) roles, Eq. (2). For every
individual x_i a chaos value and the adaptive step are drawn

    cf   = chi * rand + (1 - chi) * sin(pi * rand)          (Eq. 5, chaos)
    F_a  = F * cf

and a mutant is built from one elite x1, two ordinary x2, x3 and one worst
individual x4 (Eq. 3):

    x_m = E_r * x1 + F_a * ((x2 - x3) - D_c * x4).

If f(x_m) < f(gbest) the mutant is the trial. Otherwise a chaotic binomial
crossover with rate C_o * cf copies mutant components into a copy of x_i
(Eq. 4, at least one component forced). The trial replaces x_i if it is
better (greedy), and the global best is updated. A stagnation counter counts
consecutive rejected trials and resets on any acceptance (AGOA.m).

After the individual loop the elite factor E_r and the counterattack factor
D_c are adapted from an experience library (a FIFO of the last 100
generations' fitness vectors): up to 10 stored generations are sampled,
Delta f = f(gbest) - min(stored fitness) and Eq. (5) is applied,

    Delta f > 0: E_r += 0.1 Delta f, D_c -= 0.1 Delta f
    else:        E_r -= 0.1 Delta f, D_c += 0.1 Delta f,

both clipped to [0, 1]. Finally the diversity Div = mean over dimensions of
the population standard deviation (Eq. 6) is computed; if Div < D_threshold
or the stagnation counter exceeds T_stag, the worst-role individuals are
re-initialised uniformly at random (game-breaking restart).

Defaults (Sec. 3.3, Table 2): E_r = D_c = C_o = 0.1, F = 0.7, chi = 0.1,
D_threshold = 0.1, T_stag = 100, N = 50.

Faithfulness notes
------------------
* Diversity is computed on box-normalised variables, as stated in Sec. 2.2
  ("computed on normalized variables ... about 35 % of the expected initial
  diversity", the standard deviation of U(0, 1) being 0.289). The released
  MATLAB code measures it on raw variables; `normalized_diversity=False`
  reproduces that behaviour.
* Positions are clipped to the box, the handling the paper reports for its
  constrained problems (Sec. 5); AGOA.m applies no bound handling at all.
* The experience library stores fitness vectors only; the MATLAB code also
  stores positions but never reads them.
* Evaluation accounting is exact: an individual costs one evaluation when
  its mutant beats the global best and two otherwise; restarts cost one per
  re-initialised individual. A generation stops early when fewer than two
  evaluations remain, so the budget is never exceeded.
* Because E_r multiplies the elite position directly (E_r = 0.1 initially),
  the mutant is contracted towards the origin until E_r grows through Eq.
  (5). This origin bias is part of the published method and is kept.
"""

import math
import random
import time
from collections import deque
from copy import deepcopy
from typing import List, TypeVar

import numpy as np
from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import StoppingByEvaluations, TerminationCriterion

S = TypeVar("S", bound=FloatSolution)
R = TypeVar("R", bound=FloatSolution)


class AGOA(Algorithm[S, R]):
    def __init__(self,
                 problem: FloatProblem,
                 population_size: int = 50,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 elite_reinforcement: float = 0.1,
                 defensive_counterattack: float = 0.1,
                 cooperation_probability: float = 0.1,
                 scaling_factor: float = 0.7,
                 chaos_factor: float = 0.1,
                 diversity_threshold: float = 0.1,
                 stagnation_limit: int = 100,
                 elite_fraction: float = 0.2,
                 ordinary_fraction: float = 0.6,
                 replay_size: int = 100,
                 replay_samples: int = 10,
                 adaptation_rate: float = 0.1,
                 normalized_diversity: bool = True):
        super().__init__()
        if population_size < 5:
            raise ValueError("AGOA needs at least five individuals for three non-empty roles")
        if not 0.0 < elite_fraction < 1.0 or not 0.0 < ordinary_fraction < 1.0 \
                or elite_fraction + ordinary_fraction >= 1.0:
            raise ValueError("elite_fraction and ordinary_fraction must be positive and sum below one")
        self.problem = problem
        self.population_size = population_size
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.elite_reinforcement = float(elite_reinforcement)
        self.defensive_counterattack = float(defensive_counterattack)
        self.cooperation_probability = float(cooperation_probability)
        self.scaling_factor = float(scaling_factor)
        self.chaos_factor = float(chaos_factor)
        self.diversity_threshold = float(diversity_threshold)
        self.stagnation_limit = int(stagnation_limit)
        self.elite_fraction = float(elite_fraction)
        self.ordinary_fraction = float(ordinary_fraction)
        self.replay_samples = int(replay_samples)
        self.adaptation_rate = float(adaptation_rate)
        self.normalized_diversity = normalized_diversity

        self.lower = np.asarray(problem.lower_bound, dtype=float)
        self.upper = np.asarray(problem.upper_bound, dtype=float)
        self.dimension = problem.number_of_variables()

        self.positions: np.ndarray = None
        self.fitness: np.ndarray = None
        self.best_solution: S = None
        self.stagnation_count = 0
        self.replay_buffer = deque(maxlen=int(replay_size))
        self.restarts = 0
        self.evaluations = 0

    # ----------------------------------------------------------------- helpers

    def _evaluate_vector(self, vector: np.ndarray) -> S:
        solution = self.problem.create_solution()
        solution.variables = vector.tolist()
        self.problem.evaluate(solution)
        self.evaluations += 1
        return solution

    def _remaining(self) -> int:
        if isinstance(self.termination_criterion, StoppingByEvaluations):
            return self.termination_criterion.max_evaluations - self.evaluations
        return 2 ** 31

    def _clip(self, vector: np.ndarray) -> np.ndarray:
        return np.clip(vector, self.lower, self.upper)

    def _update_best(self, solution: S) -> None:
        if self.best_solution is None or solution.objectives[0] < self.best_solution.objectives[0]:
            self.best_solution = deepcopy(solution)

    def roles(self):
        """Elite / ordinary / worst index arrays from the current fitness ranking (Eq. 2)."""
        order = np.argsort(self.fitness, kind="stable")
        elite_count = math.ceil(self.population_size * self.elite_fraction)
        ordinary_count = math.ceil(self.population_size * self.ordinary_fraction)
        ordinary_count = min(ordinary_count, self.population_size - elite_count - 1)
        elite = order[:elite_count]
        ordinary = order[elite_count:elite_count + ordinary_count]
        worst = order[elite_count + ordinary_count:]
        return elite, ordinary, worst

    def chaos_value(self) -> float:
        return self.chaos_factor * random.random() + (1.0 - self.chaos_factor) * math.sin(random.random() * math.pi)

    def diversity(self) -> float:
        """Eq. (6): mean over dimensions of the population standard deviation."""
        x = self.positions
        if self.normalized_diversity:
            span = np.where(self.upper > self.lower, self.upper - self.lower, 1.0)
            x = (x - self.lower) / span
        return float(np.mean(np.std(x, axis=0, ddof=1)))

    # --------------------------------------------------------------- lifecycle

    def create_initial_solutions(self) -> List[S]:
        return [self.problem.create_solution() for _ in range(self.population_size)]

    def evaluate(self, solution_list: List[S]) -> List[S]:
        for solution in solution_list:
            self.problem.evaluate(solution)
        return solution_list

    def init_progress(self) -> None:
        self.evaluations = self.population_size
        self.positions = np.array([s.variables for s in self.solutions], dtype=float)
        self.fitness = np.array([s.objectives[0] for s in self.solutions], dtype=float)
        for solution in self.solutions:
            self._update_best(solution)
        self.observable.notify_all(**self.observable_data())

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met or self._remaining() < 2

    def step(self) -> None:
        elite, ordinary, worst = self.roles()

        for i in range(self.population_size):
            if self._remaining() < 2:
                break
            cf = self.chaos_value()
            f_adaptive = self.scaling_factor * cf
            x1 = self.positions[random.choice(elite)]
            x2 = self.positions[random.choice(ordinary)]
            x3 = self.positions[random.choice(ordinary)]
            x4 = self.positions[random.choice(worst)]

            mutant = self._clip(self.elite_reinforcement * x1
                                + f_adaptive * ((x2 - x3) - self.defensive_counterattack * x4))    # Eq. (3)
            mutant_solution = self._evaluate_vector(mutant)

            if mutant_solution.objectives[0] < self.best_solution.objectives[0]:
                trial_solution = mutant_solution
            else:                                                                            # Eq. (4)
                mask = np.random.random(self.dimension) < self.cooperation_probability * cf
                if not mask.any():
                    mask[random.randrange(self.dimension)] = True
                trial = self.positions[i].copy()
                trial[mask] = mutant[mask]
                trial_solution = self._evaluate_vector(trial)

            if trial_solution.objectives[0] < self.fitness[i]:
                self.positions[i] = np.asarray(trial_solution.variables, dtype=float)
                self.fitness[i] = trial_solution.objectives[0]
                self.solutions[i] = trial_solution
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
            self._update_best(trial_solution)

        self.adapt_parameters()
        self.replay_buffer.append(self.fitness.copy())

        if self.diversity() < self.diversity_threshold or self.stagnation_count > self.stagnation_limit:
            self.restart(worst)

    def adapt_parameters(self) -> None:
        """Eq. (5) with f_reply = best fitness of a randomly sampled stored generation (AGOA.m)."""
        if not self.replay_buffer:
            return
        for _ in range(min(self.replay_samples, len(self.replay_buffer))):
            stored = self.replay_buffer[random.randrange(len(self.replay_buffer))]
            delta = self.best_solution.objectives[0] - float(np.min(stored))
            if delta > 0.0:
                self.elite_reinforcement += self.adaptation_rate * delta
                self.defensive_counterattack -= self.adaptation_rate * delta
            else:
                self.elite_reinforcement -= self.adaptation_rate * delta
                self.defensive_counterattack += self.adaptation_rate * delta
        self.elite_reinforcement = min(max(self.elite_reinforcement, 0.0), 1.0)
        self.defensive_counterattack = min(max(self.defensive_counterattack, 0.0), 1.0)

    def restart(self, worst: np.ndarray) -> None:
        """Game-breaking: re-initialise the worst-role individuals uniformly at random."""
        for index in worst:
            if self._remaining() < 1:
                break
            fresh = np.random.uniform(self.lower, self.upper)
            solution = self._evaluate_vector(fresh)
            self.positions[index] = fresh
            self.fitness[index] = solution.objectives[0]
            self.solutions[index] = solution
            self._update_best(solution)
        self.stagnation_count = 0
        self.restarts += 1

    def update_progress(self) -> None:
        self.observable.notify_all(**self.observable_data())

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "ELITE_REINFORCEMENT": self.elite_reinforcement,
            "DEFENSIVE_COUNTERATTACK": self.defensive_counterattack,
            "RESTARTS": self.restarts,
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def result(self) -> R:
        return self.best_solution

    def get_name(self) -> str:
        return "AGOA"
