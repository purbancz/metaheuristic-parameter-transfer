"""IKUN: mean-field-game KD-tree density guidance for population optimisers.

Reference
---------
J. J. Lian, M. Yu, K. Ouyang, S. Fu, R. Zhong, Y. Zhang, H. Chen: "IKUN: A
mean-field game theoretic KD-tree density guided mechanism for evolutionary
algorithms", Information Sciences 743 (2026) 123347.
https://doi.org/10.1016/j.ins.2026.123347 (code: github.com/junbolian/IKUN-Mechanism).

Mechanism (Sec. 2.1, Algorithm 5)
---------------------------------
A KD-tree holds the last W population snapshots (sliding window, Sec. 2.4).
For a point x the k-NN density estimate is (Eq. 2)

    rho_hat(x) = k / (M * V_D(d_(k)(x))),   V_D(r) = pi^(D/2) / Gamma(D/2 + 1) r^D,

M being the number of stored points and d_(k) the distance to the k-th
nearest stored neighbour (the point itself excluded). The crowding
potential is Phi(x) = lambda * U(rho_hat(x)) with U the identity (Eq. 3) and
the augmented objective F(x) = f(x) + Phi(x) (Eq. 4). Reported results are
always the best raw f.

Host embeddings (Sec. 2.2, Algorithms 1, 2 and 4)
------------------------------------------
* IKUN-PSO: after the cognitive and social terms the velocity receives a
  repulsion c3 r3 R with R = sum_{j in N_k(i)} (x_i - x_j) / ||x_i - x_j||^2,
  the unit-repulsion approximation of -grad Phi (Eq. 5). The paper prints
  "lambda absorbed into c3"; c3 therefore defaults to lambda.
* IKUN-DE: rand/1/bin unchanged; the trial survives if F(u) <= F(x_i).
* IKUN-GA: roulette-wheel parent selection with weights 1 / (1 + F(x));
  survivors are the E best individuals of parents U offspring by raw f plus
  the N - E lowest-F individuals of the rest (Algorithm 4; E = 1, "the best
  individual with respect to f is never lost").

Defaults (Sec. 2.5, 3): (k, lambda, W) = (5, 0.3, 5); the paper's
sensitivity study also recommends k = 7 and lambda = 0.5 on CEC 2022.

Faithfulness notes
------------------
* Eq. (2) is computed in log space, so the D-dimensional hypersphere volume
  cannot overflow. Its raw magnitude is extremely scale dependent
  (V_D(r) ~ r^D): in the paper's own D = 20 setting rho_hat is far below
  the fitness scale, so IKUN-DE and IKUN-GA are only affected at near-ties,
  whereas IKUN-PSO (direction only) is scale free. `density_scale="relative"` divides rho_hat by the median
  density of the current population, making lambda dimensionless; the
  default "raw" is the published formula.
* Algorithm 1 writes the repulsion with a leading minus sign while Eq. (5)
  and the text describe a push away from crowded regions; the push away
  (Eq. 5) is implemented.
* The tree is rebuilt every generation from the W most recent snapshots
  (the paper rebuilds every W generations from the same snapshots).
* The GA host of the paper is binary coded (two-point crossover, bit flip);
  here the repository's real-coded GA (SBX, polynomial mutation) is the host.
"""

import math
import copy
import random
from collections import deque
from functools import cmp_to_key
from typing import List, Sequence, TypeVar

import numpy as np
from scipy.spatial import cKDTree
from jmetal.config import store
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import DifferentialEvolutionCrossover
from jmetal.util.comparator import ObjectiveComparator

from metaheuristiczoo.algorithm.basic.custom_ga import GeneticAlgorithm
from metaheuristiczoo.algorithm.basic.differential_evolution import DifferentialEvolution
from metaheuristiczoo.algorithm.basic.single_objective_pso import SingleObjectivePSO

S = TypeVar("S", bound=FloatSolution)


class CrowdingDensity:
    """KD-tree k-NN density, crowding potential and unit repulsion (Sec. 2.1)."""

    def __init__(self, dimension: int, k: int = 5, weight: float = 0.3, window: int = 5,
                 density_scale: str = "raw"):
        if k < 1 or window < 1:
            raise ValueError("k and window must be positive")
        if density_scale not in ("raw", "relative"):
            raise ValueError("density_scale must be 'raw' or 'relative'")
        self.dimension = dimension
        self.k = int(k)
        self.weight = float(weight)
        self.window = int(window)
        self.density_scale = density_scale
        self.snapshots = deque(maxlen=self.window)
        self.tree: cKDTree = None
        self.points: np.ndarray = None
        self.reference_density = 1.0
        self.log_unit_ball = 0.5 * dimension * math.log(math.pi) - math.lgamma(0.5 * dimension + 1.0)

    def record(self, positions: Sequence[Sequence[float]]) -> None:
        """Insert the current population and rebuild the tree from the W most recent snapshots."""
        self.snapshots.append(np.asarray(positions, dtype=float))
        self.points = np.vstack(self.snapshots)
        self.tree = cKDTree(self.points)
        if self.density_scale == "relative":
            raw = [self._raw_density(p) for p in np.asarray(positions, dtype=float)]
            median = float(np.median(raw))
            self.reference_density = median if median > 0.0 and np.isfinite(median) else 1.0

    def _kth_distance(self, x: np.ndarray):
        m = self.points.shape[0]
        k = min(self.k, m - 1)
        if k < 1:
            return None
        distances, _ = self.tree.query(x, k=k + 1)
        distances = np.atleast_1d(distances)
        if distances[0] <= 1e-12:                    # x itself is stored: drop the self match
            return float(distances[k])
        return float(distances[k - 1])

    def _raw_density(self, x: np.ndarray) -> float:
        d_k = self._kth_distance(x)
        if d_k is None:
            return 0.0
        d_k = max(d_k, 1e-300)
        log_density = (math.log(self.k) - math.log(self.points.shape[0])
                       - self.log_unit_ball - self.dimension * math.log(d_k))          # Eq. (2)
        return float(math.exp(min(log_density, 700.0)))

    def density(self, x: Sequence[float]) -> float:
        raw = self._raw_density(np.asarray(x, dtype=float))
        return raw / self.reference_density if self.density_scale == "relative" else raw

    def potential(self, x: Sequence[float]) -> float:
        return self.weight * self.density(x)                                           # Eq. (3), U = identity

    def repulsion(self, x: Sequence[float]) -> np.ndarray:
        """Unit repulsion sum_j (x - x_j) / ||x - x_j||^2 over the k nearest stored neighbours."""
        x = np.asarray(x, dtype=float)
        m = self.points.shape[0]
        k = min(self.k, m - 1)
        if k < 1:
            return np.zeros_like(x)
        distances, indices = self.tree.query(x, k=k + 1)
        distances, indices = np.atleast_1d(distances), np.atleast_1d(indices)
        if distances[0] <= 1e-12:
            distances, indices = distances[1:], indices[1:]
        else:
            distances, indices = distances[:k], indices[:k]
        vector = np.zeros_like(x)
        for d, j in zip(distances, indices):
            if d > 1e-12:
                vector += (x - self.points[j]) / (d * d)
        return vector


class IKUNPSO(SingleObjectivePSO):
    """IKUN-PSO (Algorithm 1): canonical PSO plus density repulsion in the velocity."""

    def __init__(self, problem: FloatProblem, swarm_size: int, c1: float, c2: float, w: float,
                 termination_criterion, k: int = 5, crowding_weight: float = 0.3, window: int = 5,
                 repulsion_coefficient: float = None, density_scale: str = "raw",
                 constraint_handling_mode: str = "clip"):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.density = CrowdingDensity(problem.number_of_variables(), k, crowding_weight, window, density_scale)
        self.c3 = crowding_weight if repulsion_coefficient is None else float(repulsion_coefficient)

    def step(self) -> None:
        self.density.record([p.variables for p in self.solutions])
        super().step()

    def update_velocity(self, swarm: List[S]) -> None:
        super().update_velocity(swarm)
        if self.c3 == 0.0:
            return
        for particle in swarm:
            repulsion = self.density.repulsion(particle.variables)
            velocity = np.asarray(particle.attributes['velocity'], dtype=float)
            particle.attributes['velocity'] = (velocity + self.c3 * random.random() * repulsion).tolist()    # Eq. (5)

    def get_name(self) -> str:
        return "IKUN-PSO"


class IKUNDE(DifferentialEvolution):
    """IKUN-DE (Algorithm 2): rand/1/bin with crowding-aware selection F = f + Phi.

    The crowding-aware selection admits a trial whose raw objective is worse than its parent's
    when its augmented value is better, so the raw best can leave the population. `result()`
    therefore returns the best point ever EVALUATED (initial population and every trial, admitted
    or not), kept in `best_solution`, as every other algorithm of the roster reports; the search
    itself is the paper's and is unchanged (added 25 September 2026).
    """

    def __init__(self, problem: FloatProblem, swarm_size: int, termination_criterion, k: int = 5,
                 crowding_weight: float = 0.3, window: int = 5, density_scale: str = "raw",
                 crossover_operator=None, selection_operator=None,
                 solution_evaluator=store.default_evaluator):
        if crossover_operator is None:          # fresh per instance, see DifferentialEvolution.__init__
            crossover_operator = DifferentialEvolutionCrossover(CR=0.9, F=0.5, K=0.5)
        super().__init__(problem=problem, swarm_size=swarm_size, crossover_operator=crossover_operator,
                         selection_operator=selection_operator, solution_evaluator=solution_evaluator,
                         termination_criterion=termination_criterion)
        self.density = CrowdingDensity(problem.number_of_variables(), k, crowding_weight, window, density_scale)
        self.best_solution = None

    def _remember_best(self, solutions: List[S]) -> None:
        best = min(solutions, key=lambda s: s.objectives[0])
        if self.best_solution is None or best.objectives[0] < self.best_solution.objectives[0]:
            self.best_solution = copy.deepcopy(best)

    def init_progress(self) -> None:
        self._remember_best(self.solutions)
        super().init_progress()

    def step(self) -> None:
        self.density.record([s.variables for s in self.solutions])
        super().step()

    def augmented(self, solution: S) -> float:
        return solution.objectives[0] + self.density.potential(solution.variables)

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        self._remember_best(offspring_population)          # every trial is an evaluated point, admitted or not
        new_population = []
        for parent, trial in zip(population, offspring_population):
            new_population.append(trial if self.augmented(trial) <= self.augmented(parent) else parent)
        new_population.sort(key=lambda s: s.objectives[0])
        return new_population

    def result(self) -> S:
        return self.best_solution

    def get_name(self) -> str:
        return "IKUN-DE"


class IKUNGA(GeneticAlgorithm):
    """IKUN-GA (Algorithm 4): roulette parents on 1/(1+F), elitist survivors by f then lowest F."""

    def __init__(self, problem: FloatProblem, population_size: int, offspring_population_size: int, mutation,
                 crossover, termination_criterion, k: int = 5, crowding_weight: float = 0.3, window: int = 5,
                 elite_size: int = 1, density_scale: str = "raw",
                 population_generator=store.default_generator, population_evaluator=store.default_evaluator,
                 solution_comparator=ObjectiveComparator(0)):
        super().__init__(problem=problem, population_size=population_size,
                         offspring_population_size=offspring_population_size, mutation=mutation,
                         crossover=crossover, termination_criterion=termination_criterion,
                         population_generator=population_generator, population_evaluator=population_evaluator,
                         solution_comparator=solution_comparator)
        if not 0 <= elite_size <= population_size:
            raise ValueError("elite_size must lie in [0, population_size]")
        self.density = CrowdingDensity(problem.number_of_variables(), k, crowding_weight, window, density_scale)
        self.elite_size = int(elite_size)

    def step(self) -> None:
        self.density.record([s.variables for s in self.solutions])
        super().step()

    def augmented(self, solution: S) -> float:
        return solution.objectives[0] + self.density.potential(solution.variables)

    def selection(self, population: List[S]) -> List[S]:
        costs = np.array([self.augmented(s) for s in population])
        if costs.min() <= -1.0:                          # keep 1/(1+F) positive for negative objectives
            costs = costs - costs.min()
        weights = 1.0 / (1.0 + costs)
        weights = weights / weights.sum()
        picks = np.random.choice(len(population), size=self.mating_pool_size, replace=True, p=weights)
        return [population[i] for i in picks]

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        pool = sorted(population + offspring_population, key=cmp_to_key(self.solution_comparator.compare))
        survivors = pool[:self.elite_size]
        rest = sorted(pool[self.elite_size:], key=self.augmented)
        survivors.extend(rest[:self.population_size - self.elite_size])
        survivors.sort(key=cmp_to_key(self.solution_comparator.compare))
        return survivors

    def get_name(self) -> str:
        return "IKUN-GA"
