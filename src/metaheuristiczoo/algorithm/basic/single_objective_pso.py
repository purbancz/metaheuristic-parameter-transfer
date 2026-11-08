import time
from copy import deepcopy
from typing import List, TypeVar
import random
import numpy as np
from jmetal.core.algorithm import ParticleSwarmOptimization, logger
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import StoppingByEvaluations, TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')


class SingleObjectivePSO(ParticleSwarmOptimization):
    """Original base class for single-objective PSO variants with common functionality"""

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 w: float,
                 termination_criterion: TerminationCriterion,
                 constraint_handling_mode: str = "clip"):
        super().__init__(problem, swarm_size)
        # "clip" is the default the continuous campaign was tuned and run
        # under, and the combinatorial campaign's set asks for "bounce".
        self.constraint_handling_mode = constraint_handling_mode
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.best_global = None

    def create_initial_solutions(self) -> List[S]:
        self.solutions = [self.problem.create_solution() for _ in range(self.swarm_size)]

        for solution in self.solutions:
            self.problem.evaluate(solution)
            solution.attributes['best_position'] = solution.variables.copy()
            solution.attributes['best_objective'] = solution.objectives[0]
            solution.attributes['velocity'] = np.random.uniform(
                -1, 1, self.problem.number_of_variables()
            ).tolist()

        self.best_global = deepcopy(min(self.solutions, key=lambda sol: sol.objectives[0]))
        return self.solutions

    def run(self) -> "SingleObjectivePSO":
        self.start_computing_time = time.time()

        logger.debug("Creating initial set of solutions...")
        self.solutions = self.create_initial_solutions()

        logger.debug("Initializing progress...")
        self.init_progress()

        logger.debug("Running main loop until termination criteria is met")
        while not self.stopping_condition_is_met():
            self.step()
            self.update_progress()

        logger.debug("Finished!")

        self.total_computing_time = time.time() - self.start_computing_time
        return self

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "SWARM": self.solutions,
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def evaluate(self, solution_list: List[S]) -> List[S]:
        return [self.problem.evaluate(sol) for sol in solution_list]

    def stopping_condition_is_met(self) -> bool:
        if self.termination_criterion.is_met:
            return True

        if isinstance(self.termination_criterion, StoppingByEvaluations):
            next_iteration_evaluations = (self.evaluations + self.swarm_size)
            return next_iteration_evaluations > self.termination_criterion.max_evaluations
        return False

    def initialize_velocity(self, swarm: List[S]) -> None:
        for particle in swarm:
            particle.attributes['velocity'] = np.random.uniform(
                -1, 1, self.problem.number_of_variables()
            ).tolist()

    def initialize_particle_best(self, swarm: List[S]) -> None:
        for particle in swarm:
            if 'best_position' not in particle.attributes:
                particle.attributes['best_position'] = particle.variables.copy()
            if 'best_objective' not in particle.attributes:
                particle.attributes['best_objective'] = particle.objectives[0]

    def initialize_global_best(self, swarm: List[S]) -> None:
        if not swarm:
            raise RuntimeError("Swarm is empty during global best initialization!")
        self.best_global = deepcopy(min(swarm, key=lambda x: x.objectives[0]))

    def update_velocity(self, swarm: List[S]) -> None:
        gbest = np.array(self.best_global.variables)
        for particle in swarm:
            r1 = random.random()
            r2 = random.random()
            velocity = np.array(particle.attributes['velocity'])
            pbest = np.array(particle.attributes['best_position'])

            new_velocity = (self.w * velocity +
                            self.c1 * r1 * (pbest - np.array(particle.variables)) +
                            self.c2 * r2 * (gbest - np.array(particle.variables)))

            particle.attributes['velocity'] = new_velocity.tolist()

    def update_position(self, swarm: List[S]) -> None:
        lower_bound = np.array(self.problem.lower_bound, dtype=float)
        upper_bound = np.array(self.problem.upper_bound, dtype=float)

        for particle in swarm:
            current_position = np.array(particle.variables, dtype=float)
            velocity = np.array(particle.attributes['velocity'], dtype=float)
            new_position = current_position + velocity

            if np.any(new_position < lower_bound) or np.any(new_position > upper_bound):
                new_position, velocity = self.handle_constraints(new_position, velocity, lower_bound, upper_bound)

            particle.variables = new_position.tolist()
            particle.attributes['velocity'] = velocity.tolist()

    def handle_constraints(self, position: np.ndarray, velocity: np.ndarray,
                           lower_bound: np.ndarray, upper_bound: np.ndarray) -> (np.ndarray, np.ndarray):
        if self.constraint_handling_mode == "clip":
            clipped_position = np.clip(position, lower_bound, upper_bound)
            return clipped_position, velocity


        elif self.constraint_handling_mode == "bounce":
            new_position = position.copy()
            new_velocity = velocity.copy()
            r = upper_bound - lower_bound
            damping = random.random()

            mask_lower = new_position < lower_bound
            if np.any(mask_lower):
                d_lower = lower_bound[mask_lower] - new_position[mask_lower]
                n_lower = np.floor_divide(d_lower, r[mask_lower])
                delta_lower = np.mod(d_lower, r[mask_lower])
                new_position[mask_lower] = np.where(
                    n_lower % 2 == 0,
                    lower_bound[mask_lower] + delta_lower,
                    upper_bound[mask_lower] - delta_lower
                )

                new_velocity[mask_lower] = np.where(
                    n_lower % 2 == 0,
                    -np.abs(new_velocity[mask_lower]) * damping,
                    np.abs(new_velocity[mask_lower]) * damping
                )

            mask_upper = new_position > upper_bound
            if np.any(mask_upper):
                d_upper = new_position[mask_upper] - upper_bound[mask_upper]
                n_upper = np.floor_divide(d_upper, r[mask_upper])
                delta_upper = np.mod(d_upper, r[mask_upper])
                new_position[mask_upper] = np.where(
                    n_upper % 2 == 0,
                    upper_bound[mask_upper] - delta_upper,
                    lower_bound[mask_upper] + delta_upper
                )
                new_velocity[mask_upper] = np.where(
                    n_upper % 2 == 0,
                    -np.abs(new_velocity[mask_upper]) * damping,
                    np.abs(new_velocity[mask_upper]) * damping
                )
            return new_position, new_velocity


        elif self.constraint_handling_mode == "reinitialize":
            new_position = np.random.uniform(lower_bound, upper_bound)
            new_velocity = np.random.uniform(-1, 1, len(new_position))
            return new_position, new_velocity

        else:
            raise ValueError("Unknown constraint handling mode: " + str(self.constraint_handling_mode))

    def update_particle_best(self, swarm: List[S]) -> None:
        for particle in swarm:
            if particle.objectives[0] < particle.attributes['best_objective']:
                particle.attributes['best_position'] = particle.variables.copy()
                particle.attributes['best_objective'] = particle.objectives[0]

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        current_best = min(swarm, key=lambda x: x.objectives[0])
        if current_best.objectives[0] < self.best_global.objectives[0]:
            self.best_global = deepcopy(current_best)

    def set_solutions(self, solutions: List[S]):
        self.solutions = deepcopy(solutions)
        for solution in self.solutions:
            if 'velocity' not in solution.attributes:
                solution.attributes['velocity'] = np.random.uniform(-1, 1, self.problem.number_of_variables())
            if ('best_position' not in solution.attributes or solution.objectives[0] <
                    solution.attributes['best_objective']):
                solution.attributes['best_position'] = deepcopy(solution.variables)
                solution.attributes['best_objective'] = solution.objectives[0]

        self.best_global = deepcopy(min(self.solutions, key=lambda sol: sol.objectives[0]))

    def perturbation(self, swarm: List[S]) -> None:
        pass

    def result(self) -> R:
        return self.best_global

    def get_name(self) -> str:
        return "PSO"
