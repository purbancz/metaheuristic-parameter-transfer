"""JADE: adaptive differential evolution with optional external archive.

Reference
---------
J. Zhang, A. C. Sanderson: "JADE: Adaptive Differential Evolution With
Optional External Archive", IEEE Transactions on Evolutionary Computation
13(5), 945-958 (2009). https://doi.org/10.1109/TEVC.2009.2014613

Algorithm (Sec. III of the paper)
---------------------------------
For every target x_i: CR_i ~ N(mu_CR, 0.1) truncated to [0, 1]; F_i ~
Cauchy(mu_F, 0.1), regenerated while F_i <= 0 and truncated to 1. Mutation
DE/current-to-pbest/1 with archive,

    v_i = x_i + F_i (x_pbest - x_i) + F_i (x_r1 - x~_r2),

x_pbest drawn from the best 100 p % of the population, x_r1 from the
population and x~_r2 from population U archive (all distinct from x_i).
Out-of-range mutant components are reset to the midpoint between the
violated bound and the target component. Binomial crossover with CR_i. The
trial replaces the target if f(u_i) <= f(x_i); the replaced parent enters
the archive (randomly truncated to the population size), and (CR_i, F_i) are
recorded as successful. After the generation

    mu_CR = (1 - c) mu_CR + c mean_A(S_CR),   mu_F = (1 - c) mu_F + c mean_L(S_F),

with the arithmetic and Lehmer means. Defaults c = 0.1, p = 0.05, mu_CR =
mu_F = 0.5, archive size = population size.

The acceptance rule is isolated in `accepts` so that host-agnostic
mechanisms (IKUN) can change it without touching the rest of JADE. The
evaluation accounting follows the repository L-SHADE implementation: the
last generation is trimmed to the remaining budget.
"""

import math
import random
import time
from typing import List, Tuple, TypeVar

import numpy as np
from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.evaluator import Evaluator
from jmetal.util.termination_criterion import StoppingByEvaluations, TerminationCriterion

S = TypeVar("S", bound=FloatSolution)
R = TypeVar("R")


class JADE(Algorithm[S, R]):
    def __init__(self,
                 problem: FloatProblem,
                 population_size: int,
                 termination_criterion: TerminationCriterion,
                 p_best_rate: float = 0.05,
                 adaptation_rate: float = 0.1,
                 archive_rate: float = 1.0,
                 evaluator: Evaluator = store.default_evaluator):
        super().__init__()
        if population_size < 4:
            raise ValueError("JADE needs at least four individuals")
        self.problem = problem
        self.population_size = population_size
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.evaluator = evaluator
        self.p_best_rate = float(p_best_rate)
        self.adaptation_rate = float(adaptation_rate)
        self.archive_size = int(round(archive_rate * population_size))
        self.mu_cr = 0.5
        self.mu_f = 0.5
        self.archive: List[S] = []
        self.evaluations_in_current_step = 0
        self.lower = np.asarray(problem.lower_bound, dtype=float)
        self.upper = np.asarray(problem.upper_bound, dtype=float)

    def create_initial_solutions(self) -> List[S]:
        return [self.problem.create_solution() for _ in range(self.population_size)]

    def evaluate(self, solution_list: List[S]) -> List[S]:
        return self.evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def init_progress(self) -> None:
        self.evaluations = self.population_size
        self.observable.notify_all(**self.observable_data())

    def update_progress(self) -> None:
        self.evaluations += self.evaluations_in_current_step
        self.observable.notify_all(**self.observable_data())

    # ------------------------------------------------------------ generation

    def sample_parameters(self) -> Tuple[float, float]:
        cr = float(np.clip(np.random.normal(self.mu_cr, 0.1), 0.0, 1.0))
        while True:
            f = float(np.random.standard_cauchy() * 0.1 + self.mu_f)
            if f > 0.0:
                break
        return cr, min(f, 1.0)

    def trial_vector(self, population: List[S], ranked: List[S], target_index: int, cr: float, f: float) -> S:
        n = len(population)
        num_p_best = max(1, math.ceil(n * self.p_best_rate))
        p_best = ranked[random.randrange(num_p_best)]
        r1 = random.randrange(n)
        while r1 == target_index:
            r1 = random.randrange(n)
        combined = population + self.archive
        r2 = random.randrange(len(combined))
        while r2 == target_index or r2 == r1:
            r2 = random.randrange(len(combined))

        x = np.asarray(population[target_index].variables, dtype=float)
        v = x + f * (np.asarray(p_best.variables) - x) + f * (np.asarray(population[r1].variables)
                                                               - np.asarray(combined[r2].variables))
        v = np.where(v < self.lower, 0.5 * (self.lower + x), v)               # JADE bound repair
        v = np.where(v > self.upper, 0.5 * (self.upper + x), v)

        j_rand = random.randrange(x.size)
        mask = np.random.random(x.size) < cr
        mask[j_rand] = True
        trial = self.problem.create_solution()
        trial.variables = np.where(mask, v, x).tolist()
        return trial

    def step(self) -> None:
        offspring_count = self.population_size
        if isinstance(self.termination_criterion, StoppingByEvaluations):
            offspring_count = min(offspring_count, self.termination_criterion.max_evaluations - self.evaluations)
        ranked = sorted(self.solutions, key=lambda s: s.objectives[0])
        trials = []
        for i in range(offspring_count):
            cr, f = self.sample_parameters()
            trials.append((self.trial_vector(self.solutions, ranked, i, cr, f), cr, f))
        self.evaluate([t[0] for t in trials])
        self.evaluations_in_current_step = len(trials)
        self.solutions = self.replacement(self.solutions, trials)

    def accepts(self, parent: S, trial: S) -> bool:
        return trial.objectives[0] <= parent.objectives[0]

    def replacement(self, population: List[S], trials: List[Tuple[S, float, float]]) -> List[S]:
        new_population = list(population)
        success_cr, success_f = [], []
        for i, (trial, cr, f) in enumerate(trials):
            if self.accepts(population[i], trial):
                new_population[i] = trial
                self.archive.append(population[i])
                success_cr.append(cr)
                success_f.append(f)
        while len(self.archive) > self.archive_size:
            self.archive.pop(random.randrange(len(self.archive)))
        if success_cr:
            c = self.adaptation_rate
            sf = np.asarray(success_f)
            self.mu_cr = (1.0 - c) * self.mu_cr + c * float(np.mean(success_cr))
            self.mu_f = (1.0 - c) * self.mu_f + c * float(np.sum(sf ** 2) / np.sum(sf))
        return new_population

    def result(self) -> R:
        return min(self.solutions, key=lambda s: s.objectives[0])

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "MU_CR": self.mu_cr,
            "MU_F": self.mu_f,
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def get_name(self) -> str:
        return "JADE"
