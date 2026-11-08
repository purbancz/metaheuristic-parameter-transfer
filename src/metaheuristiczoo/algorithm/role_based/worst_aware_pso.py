from typing import List, TypeVar
from jmetal.core.problem import FloatProblem
from jmetal.util.termination_criterion import TerminationCriterion
from metaheuristiczoo.algorithm.basic.single_objective_pso import SingleObjectivePSO

S = TypeVar("S")


class WorstAwarePSO(SingleObjectivePSO):
    """
    Base class for PSO variants that maintain and update worst solution information.
    """

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 w: float,
                 termination_criterion: TerminationCriterion,
                 constraint_handling_mode: str = "clip"):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.global_worst = None

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        for sol in solutions:
            sol.attributes['worst_position'] = sol.variables.copy()
            sol.attributes['worst_objective'] = sol.objectives[0]
        self.global_worst = max(solutions, key=lambda s: s.objectives[0])
        return solutions

    @staticmethod
    def update_particle_worst(swarm: List[S]) -> None:
        for particle in swarm:
            if particle.objectives[0] > particle.attributes['worst_objective']:
                particle.attributes['worst_objective'] = particle.objectives[0]
                particle.attributes['worst_position'] = particle.variables.copy()

    def update_global_worst(self, swarm: List[S]) -> None:
        self.global_worst = max(swarm, key=lambda s: s.objectives[0])


