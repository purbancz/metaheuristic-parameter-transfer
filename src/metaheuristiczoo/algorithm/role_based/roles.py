import random
from typing import List, TypeVar

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import TerminationCriterion

from metaheuristiczoo.algorithm.role_based.worst_aware_pso import WorstAwarePSO
from metaheuristiczoo.algorithm.basic.single_objective_pso import SingleObjectivePSO

S = TypeVar('S')


class RoleMixin:
    @staticmethod
    def mark_particles(swarm: List[S], fraction: float, role: str) -> None:
        """Mark a fraction of particles with the given role attribute."""
        count = max(1, int(len(swarm) * fraction))
        indices = random.sample(range(len(swarm)), count)
        for i, particle in enumerate(swarm):
            particle.attributes[role] = (i in indices)

    @staticmethod
    def compute_component(particle: S, target: np.ndarray,
                          current: np.ndarray, normal_coefficient: float,
                          role_coefficient: float, role_flag: str) -> np.ndarray:
        """
        Computes a directional component based on whether the particle has the given role.
        If the role is active, the direction is inverted (i.e. repulsion) and scaled by role_coefficient.
        Otherwise, the normal attraction is computed scaled by normal_coefficient.
        """
        r = random.random()
        if particle.attributes.get(role_flag, False):
            # inverted direction
            return r * role_coefficient * (current - target)
        else:
            return r * normal_coefficient * (target - current)


#######################################################


class RebelRejectorPSO(SingleObjectivePSO, RoleMixin):
    """PSO with both rebel and rejector particles"""

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 ac1: float,
                 ac2: float,
                 w: float,
                 rebel_fraction: float,
                 rejector_fraction: float,
                 termination_criterion: TerminationCriterion,
                 constraint_handling_mode: str = "clip"):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.ac1 = ac1
        self.ac2 = ac2
        self.rebel_fraction = rebel_fraction
        self.rejector_fraction = rejector_fraction

    def _mark_special_particles(self, swarm: List[S]) -> None:
        self.mark_particles(swarm, self.rebel_fraction, 'is_rebel')

        # disjoint sets, so to use my method I need to recalculate the fraction
        total = len(swarm)
        desired_rejectors = max(1, int(total * self.rejector_fraction))
        num_rebels = sum(1 for p in swarm if p.attributes.get('is_rebel', False))
        non_rebel_count = total - num_rebels
        effective_fraction = desired_rejectors / non_rebel_count if non_rebel_count > 0 else 1.0
        effective_fraction = min(1.0, effective_fraction)

        non_rebels = [p for p in swarm if not p.attributes.get('is_rebel', False)]
        self.mark_particles(non_rebels, effective_fraction, 'is_rejector')

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self._mark_special_particles(solutions)
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        for particle in swarm:
            current = np.array(particle.variables)
            p_best = np.array(particle.attributes['best_position'])
            social_vec = self.compute_component(particle, g_best, current, self.c2, self.ac2, 'is_rebel')
            cognitive_vec = self.compute_component(particle, p_best, current, self.c1, self.ac1, 'is_rejector')
            velocity = (self.w * np.array(particle.attributes['velocity'])
                        + social_vec
                        + cognitive_vec)
            particle.attributes['velocity'] = velocity.tolist()

    def get_name(self) -> str:
        return "RebelRejectorPSO"


#####################################
# Worst aware roles
# Negative
#####################################


class ContrarianDefeatistPSO(WorstAwarePSO, RoleMixin):
    """
    Contrarian-Defeatist PSO:
    Particles may be marked as contrarian (using global worst) and/or defeatist (using personal worst).
    """

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 ac1: float,
                 ac2: float,
                 w: float,
                 termination_criterion: TerminationCriterion,
                 contrarian_fraction: float,
                 defeatist_fraction: float,
                 constraint_handling_mode: str = "clip"):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.ac1 = ac1
        self.ac2 = ac2
        self.contrarian_fraction = contrarian_fraction
        self.defeatist_fraction = defeatist_fraction

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.contrarian_fraction, 'is_contrarian')
        self.mark_particles(solutions, self.defeatist_fraction, 'is_defeatist')
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        for particle in swarm:
            current = np.array(particle.variables)
            # Social component: choose global worst if contrarian; else global best.
            if particle.attributes.get('is_contrarian', False):
                social_dir = np.array(self.global_worst.variables) - current
                social_vec = self.ac2 * random.random() * social_dir
            else:
                social_dir = np.array(self.best_global.variables) - current
                social_vec = self.c2 * random.random() * social_dir
            # Cognitive component: choose personal worst if defeatist; else personal best.
            if particle.attributes.get('is_defeatist', False):
                cognitive_dir = np.array(particle.attributes['worst_position']) - current
                cognitive_vec = self.ac1 * random.random() * cognitive_dir
            else:
                cognitive_dir = np.array(particle.attributes['best_position']) - current
                cognitive_vec = self.c1 * random.random() * cognitive_dir
            velocity = self.w * np.array(particle.attributes['velocity']) + social_vec + cognitive_vec
            particle.attributes['velocity'] = velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)

    def get_name(self) -> str:
        return "ContrarianDefeatistPSO"


#####################################
# Worst aware roles
# Positive
#####################################


class EschewerEscapistPSO(WorstAwarePSO, RoleMixin):

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 ac1: float,
                 ac2: float,
                 w: float,
                 eschewer_fraction: float,
                 escapist_fraction: float,
                 termination_criterion: TerminationCriterion,
                 constraint_handling_mode: str = "clip"):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.ac1 = ac1
        self.ac2 = ac2
        self.eschewer_fraction = eschewer_fraction
        self.escapist_fraction = escapist_fraction

    """
    Eschewer - Escapist PSO:
    Particles may be marked as eschewer (avoiding global worst) and/or escapist (avoiding personal worst).
    """

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.eschewer_fraction, 'is_eschewer')
        self.mark_particles(solutions, self.escapist_fraction, 'is_escapist')
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        for particle in swarm:
            current = np.array(particle.variables)
            # Social component: choose global worst if eschewer; else global best.
            if particle.attributes.get('is_eschewer', False):
                social_dir = current - np.array(self.global_worst.variables)
                social_vec = self.ac2 * random.random() * social_dir
            else:
                social_dir = np.array(self.best_global.variables) - current
                social_vec = self.c2 * random.random() * social_dir
            # Cognitive component: choose personal worst if escapist; else personal best.
            if particle.attributes.get('is_escapist', False):
                cognitive_dir = current - np.array(particle.attributes['worst_position'])
                cognitive_vec = self.ac1 * random.random() * cognitive_dir
            else:
                cognitive_dir = np.array(particle.attributes['best_position']) - current
                cognitive_vec = self.c1 * random.random() * cognitive_dir
            velocity = self.w * np.array(particle.attributes['velocity']) + social_vec + cognitive_vec
            particle.attributes['velocity'] = velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)

    def get_name(self) -> str:
        return "EschewerEscapistPSO"


## Wanderer particles


