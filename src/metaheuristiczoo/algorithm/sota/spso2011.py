"""SPSO 2011: Standard Particle Swarm Optimisation 2011.

References
----------
Clerc, M. (2012). Standard Particle Swarm Optimisation. Technical report, HAL
hal-00764996 (15 pages; version of 23 September 2012). Sections 3.2.2, 3.3.2, 3.4.2, 3.4.3,
3.5 and 3.6 define the 2011 version implemented here.

Zambrano-Bigiarini, M., Clerc, M., & Rojas, R. (2013). Standard Particle Swarm
Optimisation 2011 at CEC-2013: A baseline for future PSO improvements. In *2013 IEEE
Congress on Evolutionary Computation (CEC)*, pp. 2337-2344. IEEE.
https://doi.org/10.1109/CEC.2013.6557848 (the CEC-2013 reference results for this
algorithm).

Algorithm
---------
* **Swarm size** S = 40 (the suggested value; Sec. 3.1). Coefficients (Eq. 3.8)
  w = 1 / (2 ln 2) ~ 0.721 and c = 1/2 + ln 2 ~ 1.193.
* **Initialisation** (Eqs. 3.3-3.4): x_i ~ U(box); v_i,d = U(min_d - x_i,d, max_d - x_i,d);
  p_i = x_i.
* **Adaptive random topology** (Sec. 3.2.2): each particle informs itself and K = 3
  particles drawn at random with replacement; the link graph is regenerated after every
  iteration in which the best known fitness did not improve.
* **Velocity update** (Sec. 3.4.2): with p_i the particle's previous best and l_i the best
  previous best among its informants, G_i = x_i + c (p_i + l_i - 2 x_i) / 3, or
  G_i = x_i + c (p_i - x_i) / 2 when the particle is itself the best of its neighbourhood
  (Sec. 3.4.3, Eq. 3.13). A point x'_i is drawn in the hypersphere H(G_i, ||G_i - x_i||) with
  the source code's default distribution, uniform random direction and uniform radius
  r = U(0, ||G_i - x_i||) (the report notes that this is *not* uniform in volume). Then
  v_i(t+1) = w v_i(t) + x'_i - x_i(t) and x_i(t+1) = x_i(t) + v_i(t+1) (Eqs. 3.11, 3.6).
* **Confinement** (Sec. 3.5, Eqs. 3.14-3.15): a coordinate leaving [min_d, max_d] is set to
  the bound and its velocity component to -0.5 v_i,d(t+1).
* **Memory**: p_i is replaced by x_i when the new fitness is strictly better (Sec. 2.2).
* **Order** (Sec. 3.6): the particles are visited in a fresh random permutation each
  iteration and, as in the reference C code of SPSO 2011, the moves are asynchronous: a
  particle moved later in the iteration sees the previous bests already updated earlier in
  the same iteration. `synchronous=True` freezes all previous bests for the iteration (the
  Matlab version described in the report).

The class reuses `SingleObjectivePSO` only for the jMetal lifecycle (`run`, budget check,
observable data); velocity, position and memory updates are replaced by the rules above.
"""

import math
import random
from copy import deepcopy
from typing import List

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import TerminationCriterion

from metaheuristiczoo.algorithm.basic.single_objective_pso import SingleObjectivePSO

SPSO2011_W = 1.0 / (2.0 * math.log(2.0))
SPSO2011_C = 0.5 + math.log(2.0)


class SPSO2011(SingleObjectivePSO):
    def __init__(self,
                 problem: FloatProblem,
                 termination_criterion: TerminationCriterion,
                 swarm_size: int = 40,
                 w: float = SPSO2011_W,
                 c: float = SPSO2011_C,
                 k: int = 3,
                 synchronous: bool = False):
        super().__init__(problem=problem, swarm_size=swarm_size, c1=c, c2=c, w=w,
                         termination_criterion=termination_criterion, constraint_handling_mode="clip")
        self.c = c
        self.k = k
        self.synchronous = synchronous
        self.lower = np.array(problem.lower_bound, dtype=float)
        self.upper = np.array(problem.upper_bound, dtype=float)
        self.informants: List[List[int]] = []
        self.topology_regenerations = 0

    # -- initialisation ----------------------------------------------------------------

    def create_initial_solutions(self) -> List[FloatSolution]:
        self.solutions = [self.problem.create_solution() for _ in range(self.swarm_size)]
        for particle in self.solutions:
            self.problem.evaluate(particle)
            x = np.array(particle.variables, dtype=float)
            particle.attributes['velocity'] = np.random.uniform(self.lower - x, self.upper - x).tolist()
            particle.attributes['best_position'] = particle.variables.copy()
            particle.attributes['best_objective'] = particle.objectives[0]
        self.best_global = deepcopy(min(self.solutions, key=lambda s: s.objectives[0]))
        self._generate_topology()
        return self.solutions

    def _generate_topology(self) -> None:
        """Each particle informs itself and K random particles (with replacement)."""
        self.informants = [[i] for i in range(self.swarm_size)]
        for i in range(self.swarm_size):
            for _ in range(self.k):
                j = random.randrange(self.swarm_size)
                if i not in self.informants[j]:
                    self.informants[j].append(i)
        self.topology_regenerations += 1

    # -- one iteration ------------------------------------------------------------------

    def step(self) -> None:
        best_before = self.best_global.objectives[0]
        if self.synchronous:
            frozen_positions = [p.attributes['best_position'] for p in self.solutions]
            frozen_objectives = [p.attributes['best_objective'] for p in self.solutions]
        else:
            frozen_positions = frozen_objectives = None

        order = list(range(self.swarm_size))
        random.shuffle(order)
        for i in order:
            self._move(i, frozen_positions, frozen_objectives)
            particle = self.solutions[i]
            self.problem.evaluate(particle)
            if not self.synchronous:
                self._update_memory(particle)

        if self.synchronous:
            for particle in self.solutions:
                self._update_memory(particle)

        if self.best_global.objectives[0] >= best_before:
            self._generate_topology()

    def _move(self, i: int, frozen_positions, frozen_objectives) -> None:
        particle = self.solutions[i]
        x = np.array(particle.variables, dtype=float)
        v = np.array(particle.attributes['velocity'], dtype=float)
        if frozen_positions is None:
            best_position = lambda j: self.solutions[j].attributes['best_position']
            best_objective = lambda j: self.solutions[j].attributes['best_objective']
        else:
            best_position = lambda j: frozen_positions[j]
            best_objective = lambda j: frozen_objectives[j]

        local_best = min(self.informants[i], key=best_objective)
        p = np.array(best_position(i), dtype=float)
        if local_best == i:
            gravity = x + self.c * (p - x) / 2.0                      # Eq. (3.13)
        else:
            l = np.array(best_position(local_best), dtype=float)
            gravity = x + self.c * (p + l - 2.0 * x) / 3.0            # Eq. (3.10)

        sampled = self._sample_hypersphere(gravity, float(np.linalg.norm(gravity - x)))
        v = self.w * v + sampled - x                                  # Eq. (3.11)
        x = x + v                                                     # Eq. (3.6)

        below, above = x < self.lower, x > self.upper                 # Eqs. (3.14)-(3.15)
        x = np.where(below, self.lower, np.where(above, self.upper, x))
        v = np.where(below | above, -0.5 * v, v)

        particle.variables = x.tolist()
        particle.attributes['velocity'] = v.tolist()

    def _sample_hypersphere(self, centre: np.ndarray, radius: float) -> np.ndarray:
        """Uniform random direction, uniform radius in [0, radius] (source-code default)."""
        if radius <= 0.0:
            return centre.copy()
        direction = np.random.normal(size=centre.shape)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            return centre.copy()
        return centre + random.random() * radius * direction / norm

    def _update_memory(self, particle: FloatSolution) -> None:
        if particle.objectives[0] < particle.attributes['best_objective']:
            particle.attributes['best_position'] = particle.variables.copy()
            particle.attributes['best_objective'] = particle.objectives[0]
            if particle.objectives[0] < self.best_global.objectives[0]:
                self.best_global = deepcopy(particle)

    # -- the base-class hooks are not used by this update scheme ------------------------

    def update_velocity(self, swarm) -> None:  # pragma: no cover - replaced by step()
        raise NotImplementedError("SPSO2011 performs its moves in step()")

    def update_position(self, swarm) -> None:  # pragma: no cover - replaced by step()
        raise NotImplementedError("SPSO2011 performs its moves in step()")

    def get_name(self) -> str:
        return "SPSO2011"
