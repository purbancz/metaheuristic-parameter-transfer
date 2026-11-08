"""Observers attached to running algorithms (jMetalPy Observer protocol).

FitnessObserver records the best objective at a fixed evaluation interval; the experiment
runner uses it for the fitness curves. SwarmTraceObserver records, once per generation, what
the running algorithm shows an observer: how spread out its population is, how many particles
hold each role, and every number the algorithm reports about itself; the runner stores it next to
the fitness curve, and for the first run of each pair also a sample of the particle positions.
"""

from typing import List

import numpy as np
from jmetal.core.observer import Observer

from metaheuristiczoo.util.diversity import normalized_solution_candidates_diversity


class FitnessObserver(Observer):
    def __init__(self, interval: int = 1) -> None:
        self.interval = interval
        # the next eval count at which we want to sample
        self.next_eval = interval
        self.best_fitness_history: List[float] = []

    def update(self, *args, **kwargs):
        solutions   = kwargs["SOLUTIONS"]
        evaluations = kwargs["EVALUATIONS"]
        best_f       = solutions.objectives[0]

        # once we hit or exceed the next threshold, record and bump it
        while evaluations >= self.next_eval:
            self.best_fitness_history.append(best_f)
            self.next_eval += self.interval


class SwarmTraceObserver(Observer):
    """One row per generation of everything an observer can see about the running algorithm.

    Columns are discovered, not declared, so the same observer serves every algorithm:

    * `evaluations` - the budget used so far;
    * `diversity` - mean distance of the population to its centroid over the domain diagonal
      (util.diversity), computed here from the algorithm's `solutions`, so it exists for every
      population algorithm, not only those that compute it themselves;
    * `pbest_diversity` - the same spread of the personal bests (memory can collapse before or
      after the positions do); `speed` - mean velocity length over the diagonal (the kinetic
      energy left in the swarm); `at_bounds` - fraction of coordinates sitting on a bound (the
      absorption pathology of high dimensions); `centroid_to_best` - distance of the swarm's
      centre to the global best over the diagonal (orbiting the best, or elsewhere);
    * `improved` - 1 when the best objective of the population fell below the best seen so far,
      and `improver.<role>` - 1 for the role of the particle that achieved it, so a plot can show
      WHICH role finds the improvements over the run;
    * `role.<flag>` - how many particles carry each role flag (`is_rebel`, `is_wanderer`, ...),
      `cognitive_role.<name>` / `social_role.<name>` - how many hold each role of the hybrids;
    * every numeric value the algorithm publishes through `observable_data()` under its own key
      (SPECIAL_MASS, GAMMA_GA_TO_PSO, TOTAL_RESTARTS, ...); dictionaries of numbers are flattened
      to `KEY.sub`, one level of nesting deep (ROLE_FRACTIONS.rebel, ...).

    A column that appears late in the run is filled with NaN before it appeared. With
    `sample_positions`, the population is kept at about `position_samples` evenly spaced
    generations: every particle's position (float16, enough to draw a swarm at a fraction of the
    storage), its objective value (so the best of that moment can be marked) and its role (a small
    code per particle, the legend in `role_names`; "standard" for a particle with no special role,
    "cognitive/social" for the hybrids), so a scatter of the swarm around its best, coloured by
    role, can be drawn from the file without re-running anything.

    Populations are not all of one size: L-SHADE, jSO and L-SRTDE shrink theirs, restart CMA-ES
    grows its lambda, some literature hybrids breed a variable number of offspring. A sample
    therefore keeps at most `max_sampled_particles` particles of a generation (the first ones, an
    index-stable subset; 256 covers every fixed-size swarm of the roster and bounds the sample of a
    2 000-member L-SHADE population at D = 1000 to about 100 MB), and `sampled_population()` pads
    the generations to the widest one - NaN positions and objectives, role code -1 - and reports
    every generation's true size in `population_sizes`. Sampling never raises for a population of
    changing size; that used to cost the whole run.
    """

    SKIPPED_KEYS = frozenset({"PROBLEM", "SOLUTIONS", "SWARM", "COMPUTING_TIME", "EVALUATIONS", "NAME"})

    ABSENT_ROLE = -1          # role code of a padding slot in a sample of a population narrower than the widest

    def __init__(self, algorithm, max_evaluations: int, sample_positions: bool = False, position_samples: int = 200,
                 max_sampled_particles: int = 256):
        self.algorithm = algorithm
        self.rows: List[dict] = []
        self.sample_positions = sample_positions
        self.position_samples = position_samples
        self.max_sampled_particles = max(1, int(max_sampled_particles))
        self.max_evaluations = max_evaluations
        self.positions: List[np.ndarray] = []
        self.positions_objectives: List[np.ndarray] = []
        self.positions_roles: List[np.ndarray] = []
        self.positions_population: List[int] = []       # true population size at each sampled generation
        self.role_names: List[str] = ["standard"]
        self.positions_evaluations: List[int] = []
        self._generation = 0
        self._position_interval = None
        self._best_so_far = np.inf

    def update(self, *args, **kwargs):
        population = getattr(self.algorithm, "solutions", None)
        row = {"evaluations": float(kwargs.get("EVALUATIONS", 0))}
        positions = self._positions(population)
        if positions is not None and len(positions) > 1:
            problem = kwargs.get("PROBLEM", getattr(self.algorithm, "problem", None))
            if problem is not None:
                self._geometry(population, positions, problem, row)
            self._count_roles(population, row)
            self._improvement(population, row)
            if self.sample_positions:
                self._sample(population, positions, row["evaluations"])
        for key, value in kwargs.items():
            if key in self.SKIPPED_KEYS:
                continue
            self._flatten(key, value, row)
        self.rows.append(row)
        self._generation += 1

    @staticmethod
    def _positions(population):
        if not population:
            return None
        try:
            return np.asarray([s.variables for s in population], dtype=float)
        except (TypeError, ValueError, AttributeError):
            return None

    def _geometry(self, population, positions, problem, row):
        lower, upper = np.asarray(problem.lower_bound, dtype=float), np.asarray(problem.upper_bound, dtype=float)
        diagonal = float(np.linalg.norm(upper - lower)) or 1.0
        row["diversity"] = normalized_solution_candidates_diversity(positions, lower, upper)
        row["at_bounds"] = float(np.mean((positions <= lower) | (positions >= upper)))
        pbests = self._attribute_matrix(population, "best_position", positions.shape[1])
        if pbests is not None:
            row["pbest_diversity"] = normalized_solution_candidates_diversity(pbests, lower, upper)
        velocities = self._attribute_matrix(population, "velocity", positions.shape[1])
        if velocities is not None:
            row["speed"] = float(np.mean(np.linalg.norm(velocities, axis=1))) / diagonal
        best = getattr(self.algorithm, "best_global", None)
        if best is not None and getattr(best, "variables", None) is not None and len(best.variables) == positions.shape[1]:
            row["centroid_to_best"] = float(np.linalg.norm(positions.mean(axis=0) - np.asarray(best.variables, dtype=float))) / diagonal

    @staticmethod
    def _attribute_matrix(population, name, width):
        rows = [getattr(p, "attributes", {}).get(name) for p in population]
        if any(r is None for r in rows):
            return None
        try:
            matrix = np.asarray(rows, dtype=float)
        except (TypeError, ValueError):
            return None
        return matrix if matrix.ndim == 2 and matrix.shape[1] == width else None

    def _improvement(self, population, row):
        objectives = [float(p.objectives[0]) for p in population if getattr(p, "objectives", None)]
        if len(objectives) != len(population):
            return
        best_index = int(np.argmin(objectives))
        improved = objectives[best_index] < self._best_so_far
        row["improved"] = float(improved)
        if improved:
            self._best_so_far = objectives[best_index]
            row[f"improver.{self._role_of(population[best_index])}"] = 1.0

    @staticmethod
    def _count_roles(population, row):
        for particle in population:
            attributes = getattr(particle, "attributes", None) or {}
            for name, value in attributes.items():
                if name.startswith("is_") and value is True:
                    row[f"role.{name[3:]}"] = row.get(f"role.{name[3:]}", 0.0) + 1.0
                elif name in ("cognitive_role", "social_role") and isinstance(value, str):
                    row[f"{name}.{value}"] = row.get(f"{name}.{value}", 0.0) + 1.0

    @staticmethod
    def _flatten(key, value, row, depth=0):
        if isinstance(value, bool):
            row[key] = float(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            row[key] = float(value)
        elif isinstance(value, dict) and depth < 2:
            for sub, sub_value in value.items():
                SwarmTraceObserver._flatten(f"{key}.{sub}", sub_value, row, depth + 1)

    @staticmethod
    def _role_of(particle):
        attributes = getattr(particle, "attributes", None) or {}
        cognitive, social = attributes.get("cognitive_role"), attributes.get("social_role")
        if isinstance(cognitive, str) or isinstance(social, str):
            return f"{cognitive or 'standard'}/{social or 'standard'}"
        flags = sorted(name[3:] for name, value in attributes.items() if name.startswith("is_") and value is True)
        return "+".join(flags) if flags else "standard"

    def _sample(self, population, positions, evaluations):
        if self._position_interval is None:
            generations = max(1, int(self.max_evaluations // max(1, len(population))))
            self._position_interval = max(1, generations // self.position_samples)
        if self._generation % self._position_interval == 0:
            self.positions_population.append(len(population))
            population = population[:self.max_sampled_particles]          # an index-stable subset of a large population
            positions = positions[:self.max_sampled_particles]
            self.positions.append(positions.astype(np.float16))
            self.positions_evaluations.append(int(evaluations))
            objectives = [float(p.objectives[0]) if getattr(p, "objectives", None) else np.nan for p in population]
            self.positions_objectives.append(np.asarray(objectives, dtype=np.float32))
            codes = []
            for particle in population:
                role = self._role_of(particle)
                if role not in self.role_names:
                    self.role_names.append(role)
                codes.append(self.role_names.index(role))
            self.positions_roles.append(np.asarray(codes, dtype=np.int16))

    def fields(self):
        """Column names in order of first appearance."""
        seen = {}
        for row in self.rows:
            for key in row:
                seen.setdefault(key, None)
        return list(seen)

    def matrix(self):
        """(fields, float32 array generations x fields); NaN where a column had no value yet."""
        fields = self.fields()
        table = np.full((len(self.rows), len(fields)), np.nan, dtype=np.float32)
        index = {name: i for i, name in enumerate(fields)}
        for r, row in enumerate(self.rows):
            for key, value in row.items():
                table[r, index[key]] = value
        return fields, table

    @staticmethod
    def _stack_padded(arrays, fill):
        """Stack per-generation arrays whose first dimension (the population) may differ, padding the
        narrower ones with `fill` up to the widest."""
        width = max(a.shape[0] for a in arrays)
        stacked = np.full((len(arrays), width) + arrays[0].shape[1:], fill, dtype=arrays[0].dtype)
        for i, a in enumerate(arrays):
            stacked[i, :a.shape[0]] = a
        return stacked


    def sampled_population(self):
        """{'positions', 'evaluations', 'objectives', 'roles', 'role_names', 'population_sizes'} of the
        sampled generations, or None: what the swarm looked like, which particle was best, and who
        held which role. Rows are padded to the widest sampled generation (NaN positions and
        objectives, role code ABSENT_ROLE); `population_sizes` holds each generation's true size."""
        if not self.positions:
            return None
        return {"positions": self._stack_padded(self.positions, np.nan),
                "evaluations": np.asarray(self.positions_evaluations, dtype=np.int64),
                "objectives": self._stack_padded(self.positions_objectives, np.nan),
                "roles": self._stack_padded(self.positions_roles, self.ABSENT_ROLE),
                "role_names": list(self.role_names),
                "population_sizes": np.asarray(self.positions_population, dtype=np.int64)}
