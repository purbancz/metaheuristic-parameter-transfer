import math
import random
import time
from typing import List, Tuple, TypeVar

import numpy as np

from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.comparator import ObjectiveComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.termination_criterion import (StoppingByEvaluations,
                                               TerminationCriterion)

S = TypeVar("S", bound=FloatSolution)
R = TypeVar("R")


class LSHADE(Algorithm[S, R]):
    """
    An implementation of the L-SHADE algorithm (Tanabe & Fukunaga 2014).
    """

    def __init__(
            self,
            problem: FloatProblem,
            termination_criterion: TerminationCriterion,
            evaluator: Evaluator = store.default_evaluator,
            initial_population_size: int = 0,
            pop_size_factor: int = 18,
            memory_size: int = 100,
            p_best_rate: float = 0.11,
            archive_size_rate: float = 2.6,
    ):
        super().__init__()
        self.evaluations_in_current_step = 0
        self.problem = problem
        # pop_size_factor may be a tuned real value: N_init = round(factor * D)
        self.initial_population_size = int(round(problem.number_of_variables() * pop_size_factor)) \
            if initial_population_size == 0 else initial_population_size
        self.population_size = self.initial_population_size
        self.termination_criterion = termination_criterion
        self.evaluator = evaluator
        self.observable.register(self.termination_criterion)
        self.comparator = ObjectiveComparator(0)

        # L-SHADE specific parameters
        self.memory_size = memory_size
        self.p_best_rate = p_best_rate
        self.archive_size_rate = archive_size_rate

        self.archive: List[S] = []
        self.memory_cr = [0.5] * self.memory_size
        self.memory_f = [0.5] * self.memory_size
        self.memory_pos = 0

    def create_initial_solutions(self) -> List[S]:
        """Creates the initial list of solutions."""
        return [self.problem.create_solution() for _ in range(self.initial_population_size)]

    def evaluate(self, solution_list: List[S]) -> List[S]:
        """Evaluates a solution list."""
        return self.evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        """Checks if the termination criterion is met."""
        return self.termination_criterion.is_met

    def init_progress(self) -> None:
        """Initializes the progress variables of the algorithm."""
        self.evaluations = self.initial_population_size
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def step(self) -> None:
        """Performs one iteration of the L-SHADE algorithm."""

        offspring_count = self.population_size

        if isinstance(self.termination_criterion, StoppingByEvaluations):
            remaining_evaluations = (self.termination_criterion.max_evaluations - self.evaluations)
            offspring_count = min(offspring_count, remaining_evaluations)

        reproduction_output = self.reproduction(self.solutions,offspring_count)
        offspring_population = [item[0] for item in reproduction_output]
        self.evaluations_in_current_step = len(offspring_population)
        self.evaluate(offspring_population)
        self.solutions = self.replacement(self.solutions, reproduction_output)

    def update_progress(self) -> None:
        """
        Updates the progress variables of the algorithm. This is called
        by the `run()` method after each `step()`.
        """
        # Count the offspring actually evaluated in the preceding step.
        self.evaluations += self.evaluations_in_current_step
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def _select_donors(self, population: List[S], target_index: int) -> Tuple[S, S]:
        """Canonical current-to-pbest/1 donor selection (Tanabe & Fukunaga 2014):
        x_r1 is drawn from the population excluding the target; x_r2 is drawn
        from population + archive excluding the target and r1."""
        r1_index = random.randrange(len(population))
        while r1_index == target_index:
            r1_index = random.randrange(len(population))

        combined = population + self.archive
        r2_index = random.randrange(len(combined))
        while r2_index == target_index or r2_index == r1_index:
            r2_index = random.randrange(len(combined))

        return population[r1_index], combined[r2_index]

    def reproduction(self, population: List[S], offspring_count: int = None) -> List[Tuple[S, float, float]]:
        """
        Generates offspring and returns them with their CR/F values.
        """
        if offspring_count is None:
            offspring_count = self.population_size

        reproduction_result: List[Tuple[S, float, float]] = []
        # The population does not change during reproduction, so the fitness ranking used
        # by current-to-pbest/1 is computed once per generation (same result as sorting per
        # individual, no random numbers involved).
        sorted_population = sorted(population, key=lambda s: s.objectives[0])
        for i in range(offspring_count):
            mem_rand_index = random.randint(0, self.memory_size - 1)
            mu_cr, mu_f = self._memory_values(mem_rand_index)

            cr = self._sample_cr(mu_cr)
            f = self._sample_f(mu_f)

            num_p_best = max(1, math.ceil(self.population_size * self._p_best_rate()))
            p_best_index = random.randint(0, num_p_best - 1)
            p_best = sorted_population[p_best_index]

            r1, r2 = self._select_donors(population, i)

            mutant = self.problem.create_solution()
            current_vars, pbest_vars = population[i].variables, p_best.variables
            r1_vars, r2_vars = r1.variables, r2.variables
            f_pbest = self._p_best_weight(f)

            for j in range(self.problem.number_of_variables()):
                mutant.variables[j] = current_vars[j] + f_pbest * (pbest_vars[j] - current_vars[j]) + f * (
                        r1_vars[j] - r2_vars[j])

            trial = self.problem.create_solution()
            j_rand = random.randrange(self.problem.number_of_variables())
            for j in range(self.problem.number_of_variables()):
                if random.random() < cr or j == j_rand:
                    trial.variables[j] = mutant.variables[j]
                else:
                    trial.variables[j] = current_vars[j]

            trial.variables = self._repair(trial.variables, current_vars)

            reproduction_result.append((trial, cr, f))
        return reproduction_result

    # -- Extension points used by the SHADE-family subclasses (src/metaheuristiczoo/algorithm/sota/shade.py,
    #    src/metaheuristiczoo/algorithm/sota/jso.py). The defaults reproduce L-SHADE exactly. --

    def _memory_values(self, index: int) -> Tuple[float, float]:
        """(M_CR, M_F) read from memory slot `index`; M_CR may be None (terminal value)."""
        return self.memory_cr[index], self.memory_f[index]

    def _sample_cr(self, mu_cr) -> float:
        """CR_i = randn(M_CR, 0.1) clipped to [0, 1]; 0 when M_CR is the terminal value."""
        if mu_cr is None:
            return 0.0
        return float(np.clip(np.random.normal(mu_cr, 0.1), 0, 1))

    def _sample_f(self, mu_f: float) -> float:
        """F_i = randc(M_F, 0.1), regenerated while F_i <= 0 and truncated to 1."""
        while True:
            f = np.random.standard_cauchy() * 0.1 + mu_f
            if f > 0:
                break
        return min(f, 1.0)

    def _p_best_rate(self) -> float:
        """Greediness p of current-to-pbest/1 for the individual being generated."""
        return self.p_best_rate

    def _p_best_weight(self, f: float) -> float:
        """Factor multiplying (x_pbest - x_i); F itself in L-SHADE."""
        return f

    def _repair(self, trial_vars: List[float], current_vars: List[float]) -> List[float]:
        """Bound handling of the trial vector (this implementation clips to the box)."""
        return np.clip(trial_vars, self.problem.lower_bound, self.problem.upper_bound).tolist()

    def _update_memory(self, successful_f: List[float], successful_cr: List[float],
                       fitness_improvements: List[float]) -> None:
        """Historical-memory update of Tanabe & Fukunaga 2014, Algorithm 1: weighted Lehmer
        means for M_F and M_CR, terminal value for M_CR once max(S_CR) = 0."""
        weights = np.array(fitness_improvements) / sum(fitness_improvements)
        sf = np.array(successful_f)
        self.memory_f[self.memory_pos] = float(np.sum(weights * sf ** 2) / np.sum(weights * sf))

        scr = np.array(successful_cr)
        if self.memory_cr[self.memory_pos] is None or scr.max() == 0:
            self.memory_cr[self.memory_pos] = None
        else:
            self.memory_cr[self.memory_pos] = float(np.sum(weights * scr ** 2) / np.sum(weights * scr))

        self.memory_pos = (self.memory_pos + 1) % self.memory_size

    def _next_population_size(self, current_evaluations: int) -> int:
        """Linear population size reduction, Eq. (10) of Tanabe & Fukunaga 2014, N_min = 4."""
        if not isinstance(self.termination_criterion, StoppingByEvaluations):
            return self.population_size
        max_evals = self.termination_criterion.max_evaluations
        next_pop_size = round(
            ((4 - self.initial_population_size) / max_evals) * current_evaluations + self.initial_population_size)
        return max(4, next_pop_size)

    def replacement(self, population: List[S], offspring_data: List[Tuple[S, float, float]]) -> List[S]:
        """
        Determines the next generation, updates archive/memory, and performs LPSR.
        """
        new_population = []
        successful_f = []
        successful_cr = []
        fitness_improvements = []
        evaluated_count = len(offspring_data)

        for i in range(evaluated_count):
            parent = population[i]
            offspring_solution, cr, f = offspring_data[i]
            if self.comparator.compare(offspring_solution, parent) < 0:
                new_population.append(offspring_solution)
                self.archive.append(parent)
                fitness_imp = parent.objectives[0] - offspring_solution.objectives[0]
                successful_f.append(f)
                successful_cr.append(cr)
                fitness_improvements.append(fitness_imp)
            else:
                new_population.append(parent)

        new_population.extend(population[evaluated_count:])

        if successful_cr:
            self._update_memory(successful_f, successful_cr, fitness_improvements)

        next_pop_size = self._next_population_size(self.evaluations + self.evaluations_in_current_step)
        if next_pop_size < self.population_size:
            self.population_size = next_pop_size
            new_population = sorted(new_population, key=lambda s: s.objectives[0])[:self.population_size]

        archive_size_limit = int(round(self.archive_size_rate * self.population_size))
        while len(self.archive) > archive_size_limit:
            self.archive.pop(random.randrange(len(self.archive)))

        return new_population

    def result(self) -> R:
        """Returns the best solution found so far."""
        return min(self.solutions, key=lambda s: s.objectives[0])

    def get_name(self) -> str:
        return "L-SHADE"

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "POPULATION_SIZE": self.population_size,
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }
