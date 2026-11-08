import random
from typing import List, TypeVar

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.util.termination_criterion import TerminationCriterion

from metaheuristiczoo.algorithm.role_based.worst_aware_pso import WorstAwarePSO
from metaheuristiczoo.algorithm.role_based.roles import RoleMixin

from jmetal.logger import get_logger

logger = get_logger(__name__)

S = TypeVar('S')


class HybridPartialDisjointPSO(WorstAwarePSO, RoleMixin):
    """
    Hybrid Diverse PSO (HDPSO) - Refined Version:
    Combines multiple alternative particle behaviors with standard PSO.
    Uses a partial disjoint role assignment strategy and individual coefficients.

    Partial Disjoint Strategy:
    - Each particle is assigned exactly ONE cognitive role (standard, rejector,
      defeatist, or escapist).
    - Each particle is assigned exactly ONE social role (standard, rebel,
      contrarian, or eschewer).
    - Cognitive and Social role assignments are independent.

    Individual Coefficients: Separate coefficients control the magnitude
    of influence for each role type.

    Cognitive Roles:
    - standard: Move towards personal best (pbest) - coeff: c1
    - rejector: Move away from personal best (pbest) - coeff: rejector_c
    - defeatist: Move towards personal worst (pworst) - coeff: defeatist_c
    - escapist: Move away from personal worst (pworst) - coeff: escapist_c

    Social Roles:
    - standard: Move towards global best (gbest) - coeff: c2
    - rebel: Move away from global best (gbest) - coeff: rebel_c
    - contrarian: Move towards global worst (gworst) - coeff: contrarian_c
    - eschewer: Move away from global worst (gworst) - coeff: eschewer_c
    """

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 termination_criterion: TerminationCriterion,
                 w: float,               # Inertia weight
                 # --- Standard Coefficients ---
                 c1: float = 1.5,        # Standard cognitive coefficient (attraction to pbest)
                 c2: float = 1.5,        # Standard social coefficient (attraction to gbest)
                 # --- Alternative Cognitive Coefficients ---
                 rejector_c: float = 1.0,
                 defeatist_c: float = 1.0,
                 escapist_c: float = 1.0,
                 # --- Alternative Social Coefficients ---
                 rebel_c: float = 1.0,
                 contrarian_c: float = 1.0,
                 eschewer_c: float = 1.0,
                 # --- Role Fractions ---
                 rejector_fraction: float = 0.0,
                 defeatist_fraction: float = 0.0,
                 escapist_fraction: float = 0.0,
                 rebel_fraction: float = 0.0,
                 contrarian_fraction: float = 0.0,
                 eschewer_fraction: float = 0.0,
                 # --- Other Options ---
                 constraint_handling_mode: str = "clip",
                 assign_roles_every_iteration: bool = False # Option to re-assign roles
                 ):

        # Initialize WorstAwarePSO with standard c1, c2 - these will be used
        # ONLY for particles assigned the 'standard' role in velocity update.
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)

        # Store all individual coefficients
        self.c1 = c1
        self.c2 = c2
        self.rejector_c = rejector_c
        self.defeatist_c = defeatist_c
        self.escapist_c = escapist_c
        self.rebel_c = rebel_c
        self.contrarian_c = contrarian_c
        self.eschewer_c = eschewer_c

        # Store fractions, ensuring they are valid probabilities
        self.rejector_fraction = max(0.0, min(1.0, rejector_fraction))
        self.defeatist_fraction = max(0.0, min(1.0, defeatist_fraction))
        self.escapist_fraction = max(0.0, min(1.0, escapist_fraction))
        self.rebel_fraction = max(0.0, min(1.0, rebel_fraction))
        self.contrarian_fraction = max(0.0, min(1.0, contrarian_fraction))
        self.eschewer_fraction = max(0.0, min(1.0, eschewer_fraction))

        self.assign_roles_every_iteration = assign_roles_every_iteration

        # Validate fraction sums for clarity, although assignment caps them
        cognitive_sum = self.rejector_fraction + self.defeatist_fraction + self.escapist_fraction
        social_sum = self.rebel_fraction + self.contrarian_fraction + self.eschewer_fraction
        if cognitive_sum > 1.0:
            logger.warning(f"Sum of cognitive role fractions ({cognitive_sum:.2f}) > 1.0. Effective fractions will be capped during assignment.")
        if social_sum > 1.0:
            logger.warning(f"Sum of social role fractions ({social_sum:.2f}) > 1.0. Effective fractions will be capped during assignment.")


    def _assign_roles_to_swarm(self, swarm: List[S]) -> None:
        """
        Assigns mutually exclusive cognitive and social roles randomly
        based on fractions. Each particle gets exactly one cognitive and
        one social role.
        """
        n = len(swarm)
        if n == 0: return

        # --- Assign Cognitive Roles ---
        indices_cognitive = list(range(n))
        random.shuffle(indices_cognitive)
        current_idx = 0

        # Calculate exact counts
        num_rejectors = int(n * self.rejector_fraction)
        num_defeatists = int(n * self.defeatist_fraction)
        num_escapists = int(n * self.escapist_fraction)

        # Assign roles sequentially based on shuffled indices
        for i in range(num_rejectors):
            if current_idx >= n: break
            swarm[indices_cognitive[current_idx]].attributes['cognitive_role'] = 'rejector'
            current_idx += 1
        for i in range(num_defeatists):
            if current_idx >= n: break
            swarm[indices_cognitive[current_idx]].attributes['cognitive_role'] = 'defeatist'
            current_idx += 1
        for i in range(num_escapists):
            if current_idx >= n: break
            swarm[indices_cognitive[current_idx]].attributes['cognitive_role'] = 'escapist'
            current_idx += 1

        # Assign Standard Cognitive to the rest
        while current_idx < n:
            swarm[indices_cognitive[current_idx]].attributes['cognitive_role'] = 'standard'
            current_idx += 1


        # --- Assign Social Roles (using a *different* shuffle) ---
        indices_social = list(range(n))
        random.shuffle(indices_social)
        current_idx_social = 0

        # Calculate exact counts
        num_rebels = int(n * self.rebel_fraction)
        num_contrarians = int(n * self.contrarian_fraction)
        num_eschewers = int(n * self.eschewer_fraction)

        # Assign roles sequentially based on the second shuffled list
        for i in range(num_rebels):
            if current_idx_social >= n: break
            swarm[indices_social[current_idx_social]].attributes['social_role'] = 'rebel'
            current_idx_social += 1
        for i in range(num_contrarians):
            if current_idx_social >= n: break
            swarm[indices_social[current_idx_social]].attributes['social_role'] = 'contrarian'
            current_idx_social += 1
        for i in range(num_eschewers):
            if current_idx_social >= n: break
            swarm[indices_social[current_idx_social]].attributes['social_role'] = 'eschewer'
            current_idx_social += 1

        # Assign Standard Social to the rest
        while current_idx_social < n:
            swarm[indices_social[current_idx_social]].attributes['social_role'] = 'standard'
            current_idx_social += 1

        # # Optional: Log distribution after assignment
        # self._log_role_distribution(swarm)


    def create_initial_solutions(self) -> List[S]:
        """Creates initial swarm, evaluates, sets up best/worst, and marks roles."""
        solutions = super().create_initial_solutions() # Handles init, eval, pbest, gbest, pworst, gworst
        self._assign_roles_to_swarm(solutions)         # Assign initial roles
        return solutions

    @staticmethod
    def _log_role_distribution(swarm: List[S]):
        """Helper to log the distribution of assigned roles."""
        counts = {}
        for p in swarm:
            role_pair = (p.attributes.get('cognitive_role', 'N/A'), p.attributes.get('social_role', 'N/A'))
            counts[role_pair] = counts.get(role_pair, 0) + 1
        logger.debug("Role Distribution:")
        sorted_counts = sorted(counts.items()) # Sort for consistent logging
        for roles, count in sorted_counts:
            logger.debug(f"  Cognitive: {roles[0]:<10} | Social: {roles[1]:<10} | Count: {count}")

    def step(self):
        """Performs one iteration/step, potentially re-assigning roles."""
        # Option to re-assign roles at the start of each iteration
        if self.assign_roles_every_iteration:
            self._assign_roles_to_swarm(self.solutions)
            # self._log_role_distribution(self.solutions) # Log if roles change

        self.update_velocity(self.solutions)
        self.update_position(self.solutions)
        self.perturbation(self.solutions) # Usually empty in base PSO
        self.solutions = self.evaluate(self.solutions)
        # Update best/worst includes both personal and global, best and worst
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[S]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)


    def update_velocity(self, swarm: List[S]) -> None:
        """Updates velocity based on assigned roles and their specific coefficients."""
        if self.best_global is None or self.global_worst is None or not swarm:
            logger.debug("Skipping velocity update: missing global best/worst or empty swarm.")
            return

        g_best = np.array(self.best_global.variables)
        g_worst = np.array(self.global_worst.variables)

        for particle in swarm:
            # Ensure particle has necessary attributes (more robust check)
            attrs = particle.attributes
            required_attrs = ['velocity', 'best_position', 'worst_position', 'cognitive_role', 'social_role']
            if not all(attr in attrs for attr in required_attrs):
                logger.warning(f"Particle missing required attributes for velocity update. Skipping. Attributes: {attrs}")
                # Attempt to initialize roles if missing? Or just skip? Skipping is safer.
                if 'cognitive_role' not in attrs: attrs['cognitive_role'] = 'standard'
                if 'social_role' not in attrs: attrs['social_role'] = 'standard'
                # Cannot proceed without velocity/positions, so continue might be needed
                # If just roles were missing, the defaults above might work, but let's stick to skipping if core data is missing.
                if not all(attr in attrs for attr in ['velocity', 'best_position', 'worst_position']):
                    continue # Skip this particle

            current = np.array(particle.variables)
            velocity = np.array(attrs['velocity'])
            p_best = np.array(attrs['best_position'])
            p_worst = np.array(attrs['worst_position'])

            cognitive_role = attrs['cognitive_role']
            social_role = attrs['social_role']

            r1 = random.random()  # Random factor per dimension *might* be better, but per component is common
            r2 = random.random()

            # --- Calculate Cognitive Component ---
            if cognitive_role == 'rejector':
                cognitive_vec = self.rejector_c * r1 * (current - p_best)
            elif cognitive_role == 'defeatist':
                cognitive_vec = self.defeatist_c * r1 * (p_worst - current)
            elif cognitive_role == 'escapist':
                cognitive_vec = self.escapist_c * r1 * (current - p_worst)
            else: # Standard case
                cognitive_vec = self.c1 * r1 * (p_best - current)

            # --- Calculate Social Component ---
            if social_role == 'rebel':
                social_vec = self.rebel_c * r2 * (current - g_best)
            elif social_role == 'contrarian':
                social_vec = self.contrarian_c * r2 * (g_worst - current)
            elif social_role == 'eschewer':
                social_vec = self.eschewer_c * r2 * (current - g_worst)
            else: # Standard case
                social_vec = self.c2 * r2 * (g_best - current)

            # --- Update Velocity ---
            new_velocity = self.w * velocity + cognitive_vec + social_vec
            particle.attributes['velocity'] = new_velocity.tolist()


    def get_name(self) -> str:
        return "HybridPartialDisjointPSO"


# ---------------------------------------------------------------------------

class HybridFullDisjointPSO(WorstAwarePSO, RoleMixin):
    """
    Hybrid PSO with Disjoint Special Role Assignment (Revised HFDPSO v2):

    Assigns AT MOST ONE SPECIAL role (e.g., rebel, rejector) to each particle.
    Particles are categorized as:
    1. Standard-Standard: Standard cognitive + Standard social behavior.
    2. Special Cognitive: A specific cognitive role (rejector, defeatist, escapist)
                         paired with STANDARD social behavior.
    3. Special Social: A specific social role (rebel, contrarian, eschewer)
                       paired with STANDARD cognitive behavior.

    Role assignment uses individual fraction parameters for each special role.
    The remaining fraction defaults to Standard-Standard particles.
    Sum of special role fractions must be <= 1.0.

    Velocity Update uses standard components (c1, c2) unless a special
    role dictates using its specific component and coefficient.

    Special Roles & Coefficients:
    - Cognitive: rejector (rejector_c), defeatist (defeatist_c), escapist (escapist_c)
    - Social: rebel (rebel_c), contrarian (contrarian_c), eschewer (eschewer_c)
    - Standard: std_cognitive (c1), std_social (c2)
    """

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 termination_criterion: TerminationCriterion,
                 w: float,               # Inertia weight
                 # --- Coefficients for each role type ---
                 c1: float = 1.5,        # Std cognitive coeff
                 rejector_c: float = 1.0,
                 defeatist_c: float = 1.0,
                 escapist_c: float = 1.0,
                 c2: float = 1.5,        # Std social coeff
                 rebel_c: float = 1.0,
                 contrarian_c: float = 1.0,
                 eschewer_c: float = 1.0,
                 # --- Individual Special Role Fractions (Sum MUST BE <= 1.0) ---
                 rejector_fraction: float = 0.0,
                 defeatist_fraction: float = 0.0,
                 escapist_fraction: float = 0.0,
                 rebel_fraction: float = 0.0,
                 contrarian_fraction: float = 0.0,
                 eschewer_fraction: float = 0.0,
                 # --- Other Options ---
                 constraint_handling_mode: str = "clip",
                 assign_roles_every_iteration: bool = False
                 ):

        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)

        # Define valid special roles and their categories
        self.special_cognitive_roles = {'rejector', 'defeatist', 'escapist'}
        self.special_social_roles = {'rebel', 'contrarian', 'eschewer'}
        self.valid_special_roles = self.special_cognitive_roles.union(self.special_social_roles)

        # Store all individual coefficients, mapping role name to value
        self.coefficients = {
            'std_cognitive': c1, 'rejector': rejector_c, 'defeatist': defeatist_c, 'escapist': escapist_c,
            'std_social': c2, 'rebel': rebel_c, 'contrarian': contrarian_c, 'eschewer': eschewer_c
        }

        # --- Validate and process individual fractions ---
        # Create a dictionary from the parameters for easier processing
        self.special_role_fractions = {
            'rejector': rejector_fraction,
            'defeatist': defeatist_fraction,
            'escapist': escapist_fraction,
            'rebel': rebel_fraction,
            'contrarian': contrarian_fraction,
            'eschewer': eschewer_fraction,
        }

        total_special_fraction = 0.0
        # Filter out roles with 0 fraction and validate others
        active_special_fractions = {}
        for role_name, fraction in self.special_role_fractions.items():
            if not (0.0 <= fraction <= 1.0):
                 raise ValueError(f"Fraction for role '{role_name}' ({fraction}) must be between 0 and 1.")
            if fraction > 0:
                 active_special_fractions[role_name] = fraction
                 total_special_fraction += fraction

        if total_special_fraction > 1.0:
            # Use 1.0 + 1e-9 for a small tolerance in floating point comparison
            if total_special_fraction > 1.0 + 1e-9:
                raise ValueError(f"Sum of special role fractions ({total_special_fraction:.4f}) cannot exceed 1.0.")
            else:
                # If very close to 1.0 due to float issues, adjust slightly
                logger.warning(f"Sum of special fractions ({total_special_fraction:.6f}) slightly over 1.0, adjusting.")
                total_special_fraction = 1.0


        # Replace the full dictionary with only the active roles (fraction > 0)
        # This is important for the assignment loop later
        self.special_role_fractions = active_special_fractions

        self.standard_fraction = 1.0 - total_special_fraction
        logger.info(f"Initializing HFDPSO: {total_special_fraction*100:.1f}% special roles ({len(self.special_role_fractions)} types active), "
                    f"{self.standard_fraction*100:.1f}% standard-standard.")

        self.assign_roles_every_iteration = assign_roles_every_iteration


    def _assign_roles(self, swarm: List[S]) -> None:
        """
        Assigns at most one special role ('rejector', 'rebel', etc.) or
        'standard' to each particle based on the provided fractions.
        """
        n = len(swarm)
        if n == 0: return

        indices = list(range(n))
        random.shuffle(indices)
        current_idx = 0

        # Assign Special Roles first
        for role_name, fraction in self.special_role_fractions.items():
            # Calculate number to assign for this specific role
            num_to_assign = int(round(n * fraction))
            num_to_assign = min(num_to_assign, n - current_idx) # Cap by remaining particles

            for i in range(num_to_assign):
                if current_idx >= n: break
                particle_idx = indices[current_idx]
                # Initialize attributes dict if needed
                if not hasattr(swarm[particle_idx], 'attributes') or swarm[particle_idx].attributes is None:
                    swarm[particle_idx].attributes = {}
                # Assign the specific special role name
                swarm[particle_idx].attributes['assigned_role'] = role_name
                current_idx += 1
            if current_idx >= n: break # Stop if all particles assigned

        # Assign 'standard' to the remaining particles
        num_standard = n - current_idx
        # logger.debug(f"Assigning {num_standard} particles the 'standard' role.")
        for i in range(num_standard):
            if current_idx >= n: break # Should not happen ideally
            particle_idx = indices[current_idx]
            if not hasattr(swarm[particle_idx], 'attributes') or swarm[particle_idx].attributes is None:
                 swarm[particle_idx].attributes = {}
            # Assign 'standard' for standard-standard particles
            swarm[particle_idx].attributes['assigned_role'] = 'standard'
            current_idx += 1

        # Final check for debugging potential rounding issues
        if current_idx != n:
             logger.error(f"Role assignment mismatch: Assigned roles to {current_idx} out of {n} particles.")

        # self._log_role_distribution(swarm) # Optional logging


    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self._assign_roles(solutions) # Use the revised assignment method
        return solutions

    @staticmethod
    def _log_role_distribution(swarm: List[S]):
        counts = {}
        for p in swarm:
            role = p.attributes.get('assigned_role', 'N/A')
            counts[role] = counts.get(role, 0) + 1
        logger.debug("Assigned Role Distribution:")
        sorted_counts = sorted(counts.items())
        for role, count in sorted_counts:
            logger.debug(f"  Role: {role:<15} | Count: {count}")

    def step(self):
        if self.assign_roles_every_iteration:
            self._assign_roles(self.solutions) # Use the revised assignment method

        self.update_velocity(self.solutions)
        self.update_position(self.solutions)
        self.perturbation(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        # These already update both best/worst for global/particle via WorstAwarePSO
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[S]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)


    def update_velocity(self, swarm: List[S]) -> None:
        """ Calculates velocity based on assigned role, using standard components as defaults. """
        if self.best_global is None or self.global_worst is None or not swarm:
            logger.debug("Skipping velocity update: missing global best/worst or empty swarm.")
            return

        g_best = np.array(self.best_global.variables)
        g_worst = np.array(self.global_worst.variables)

        for particle in swarm:
            attrs = particle.attributes
            # Check for core attributes + the assigned role
            required_attrs = ['velocity', 'best_position', 'worst_position', 'assigned_role']
            if not all(attr in attrs for attr in required_attrs):
                logger.warning(f"Particle missing required attributes for velocity update. Skipping. Attrs: {attrs}")
                continue

            current = np.array(particle.variables)
            velocity = np.array(attrs['velocity'])
            p_best = np.array(attrs['best_position'])
            p_worst = np.array(attrs['worst_position'])
            assigned_role = attrs['assigned_role']

            r1 = random.random() # Cognitive random factor
            r2 = random.random() # Social random factor

            cognitive_vec = np.zeros_like(current)
            social_vec = np.zeros_like(current)

            # Determine Cognitive Component
            if assigned_role == 'standard' or assigned_role in self.special_social_roles:
                # Use standard cognitive if particle is standard-standard OR has a special SOCIAL role
                coeff = self.coefficients['std_cognitive']
                cognitive_vec = coeff * r1 * (p_best - current)
            elif assigned_role == 'rejector':
                 coeff = self.coefficients['rejector']
                 cognitive_vec = coeff * r1 * (current - p_best)
            elif assigned_role == 'defeatist':
                 coeff = self.coefficients['defeatist']
                 cognitive_vec = coeff * r1 * (p_worst - current)
            elif assigned_role == 'escapist':
                 coeff = self.coefficients['escapist']
                 cognitive_vec = coeff * r1 * (current - p_worst)
            else:
                 logger.error(f"Unexpected assigned_role '{assigned_role}' encountered during cognitive calc.")


            # Determine Social Component
            if assigned_role == 'standard' or assigned_role in self.special_cognitive_roles:
                 # Use standard social if particle is standard-standard OR has a special COGNITIVE role
                 coeff = self.coefficients['std_social']
                 social_vec = coeff * r2 * (g_best - current)
            elif assigned_role == 'rebel':
                 coeff = self.coefficients['rebel']
                 social_vec = coeff * r2 * (current - g_best)
            elif assigned_role == 'contrarian':
                 coeff = self.coefficients['contrarian']
                 social_vec = coeff * r2 * (g_worst - current)
            elif assigned_role == 'eschewer':
                 coeff = self.coefficients['eschewer']
                 social_vec = coeff * r2 * (current - g_worst)
            else:
                  logger.error(f"Unexpected assigned_role '{assigned_role}' encountered during social calc.")


            # --- Update Velocity ---
            new_velocity = self.w * velocity + cognitive_vec + social_vec
            particle.attributes['velocity'] = new_velocity.tolist()

    def get_name(self) -> str:
        return "HybridFullDisjointPSO"

# ---------------------------------------------------------------------------


class HybridAdditivePSO(WorstAwarePSO, RoleMixin):
    """
    Hybrid Additive PSO (HAPSO v3 - Default to Standard):
    Allows particles to simultaneously exhibit multiple behaviors. Activation
    is probabilistic via individual parameters. Velocity is the sum of
    inertia and influences from active behaviors.

    *Default Behavior*: If no special cognitive roles are activated for a
    particle in an iteration, the standard cognitive component is applied
    by default (regardless of its probability draw). Similarly, if no
    special social roles are activated, the standard social component applies.

    Role Activation: Determined by individual probabilities (0 to 1).
    Role Influence: Controlled by individual coefficients.

    Potential Roles & Coefficients:
    - is_std_cognitive: Move towards pbest (coeff: c1)
    - is_rejector: Move away from pbest (coeff: rejector_c)
    - is_defeatist: Move towards pworst (coeff: defeatist_c)
    - is_escapist: Move away from pworst (coeff: escapist_c)
    - is_std_social: Move towards gbest (coeff: c2)
    - is_rebel: Move away from gbest (coeff: rebel_c)
    - is_contrarian: Move towards gworst (coeff: contrarian_c)
    - is_eschewer: Move away from gworst (coeff: eschewer_c)
    """

    # __init__ remains the same as the previous version (HAPSO v2)
    # It accepts individual probability parameters.
    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 termination_criterion: TerminationCriterion,
                 w: float,               # Inertia weight
                 # --- Coefficients for each role type ---
                 c1: float = 1.5,        # Std cognitive coeff
                 rejector_c: float = 1.0,
                 defeatist_c: float = 1.0,
                 escapist_c: float = 1.0,
                 c2: float = 1.5,        # Std social coeff
                 rebel_c: float = 1.0,
                 contrarian_c: float = 1.0,
                 eschewer_c: float = 1.0,
                 # --- Individual Role Activation Probabilities (0 to 1) ---
                 std_cognitive_prob: float = 1.0, # Probability for standard cognitive component
                 rejector_prob: float = 0.0,
                 defeatist_prob: float = 0.0,
                 escapist_prob: float = 0.0,
                 std_social_prob: float = 1.0,    # Probability for standard social component
                 rebel_prob: float = 0.0,
                 contrarian_prob: float = 0.0,
                 eschewer_prob: float = 0.0,
                 # --- Other Options ---
                 constraint_handling_mode: str = "clip",
                 assign_flags_every_iteration: bool = False
                 ):

        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)

        self.coefficients = {
            'is_std_cognitive': c1, 'is_rejector': rejector_c, 'is_defeatist': defeatist_c, 'is_escapist': escapist_c,
            'is_std_social': c2, 'is_rebel': rebel_c, 'is_contrarian': contrarian_c, 'is_eschewer': eschewer_c
        }

        prob_params = {
            'std_cognitive': std_cognitive_prob, 'rejector': rejector_prob, 'defeatist': defeatist_prob, 'escapist': escapist_prob,
            'std_social': std_social_prob, 'rebel': rebel_prob, 'contrarian': contrarian_prob, 'eschewer': eschewer_prob
        }

        self.role_probabilities = {}
        for role_base_name, prob in prob_params.items():
            flag_name = f"is_{role_base_name}"
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"Probability for role '{role_base_name}' ({prob}) must be between 0 and 1.")
            if flag_name not in self.coefficients:
                raise ValueError(f"Internal inconsistency: Flag name '{flag_name}' derived from probability parameter '{role_base_name}' not found in coefficients.")
            self.role_probabilities[flag_name] = prob

        self.assign_flags_every_iteration = assign_flags_every_iteration
        logger.info(f"Initializing HAPSO (Default-to-Standard) with individual probabilities.")

    # _assign_role_flags_to_swarm remains the same - it just sets flags based on probability
    def _assign_role_flags_to_swarm(self, swarm: List[S]) -> None:
        """
        Assigns boolean flags (e.g., 'is_rejector') to each particle based
        on activation probabilities stored in self.role_probabilities.
        (This method remains unchanged).
        """
        n = len(swarm)
        if n == 0: return

        for i in range(n):
            particle = swarm[i]
            if not hasattr(particle, 'attributes') or particle.attributes is None:
                 particle.attributes = {}
            attrs = particle.attributes

            # Determine activation for each possible role flag using the stored probabilities
            for flag_name, probability in self.role_probabilities.items():
                attrs[flag_name] = (random.random() < probability)
        # self._log_role_distribution(swarm)


    # create_initial_solutions remains the same
    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self._assign_role_flags_to_swarm(solutions)
        return solutions

    # _log_role_distribution remains the same
    @staticmethod
    def _log_role_distribution(swarm: List[S]):
        if not swarm or not hasattr(swarm[0], 'attributes') or not swarm[0].attributes:
             logger.debug("Cannot log role distribution: Swarm empty or first particle has no attributes.")
             return
        flag_counts = {}
        known_flags = [f for f in swarm[0].attributes.keys() if f.startswith('is_')]
        for flag in known_flags: flag_counts[flag] = 0
        for p in swarm:
             for flag in known_flags:
                 if p.attributes.get(flag, False): flag_counts[flag] += 1
        logger.debug("Role Activation Counts (Additive):")
        sorted_counts = sorted(flag_counts.items())
        for flag, count in sorted_counts: logger.debug(f"  Flag: {flag:<20} | Active Count: {count}")

    # step remains the same
    def step(self):
        if self.assign_flags_every_iteration:
            self._assign_role_flags_to_swarm(self.solutions)

        self.update_velocity(self.solutions)
        self.update_position(self.solutions)
        self.perturbation(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[S]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)

    # --- MODIFIED update_velocity ---
    def update_velocity(self, swarm: List[S]) -> None:
        """
        Calculates velocity by summing influences of active roles.
        Applies standard cognitive/social component by default if no other
        special role in that category is active.
        """
        if self.best_global is None or self.global_worst is None or not swarm:
            logger.debug("Skipping velocity update: missing global best/worst or empty swarm.")
            return

        g_best = np.array(self.best_global.variables)
        g_worst = np.array(self.global_worst.variables)

        for particle in swarm:
            attrs = particle.attributes
            required_core_attrs = ['velocity', 'best_position', 'worst_position']
            if not all(attr in attrs for attr in required_core_attrs):
                logger.warning(f"Particle missing core attributes. Skipping. Attrs: {attrs}")
                continue
            # Ensure all expected flags are present, defaulting to False
            for flag_name in self.coefficients.keys():
                if flag_name not in attrs:
                    attrs[flag_name] = False # Default missing flags to False

            current = np.array(particle.variables)
            velocity = np.array(attrs['velocity'])
            p_best = np.array(attrs['best_position'])
            p_worst = np.array(attrs['worst_position'])

            cognitive_component = np.zeros_like(current)
            social_component = np.zeros_like(current)
            # Separate random factors per potential component
            rand_factors = {flag: random.random() for flag in self.coefficients}

            # --- Accumulate Special Cognitive Influences & Track Activation ---
            any_special_cognitive_active = False
            if attrs.get('is_rejector', False):
                 coeff = self.coefficients['is_rejector']
                 cognitive_component += coeff * rand_factors['is_rejector'] * (current - p_best)
                 any_special_cognitive_active = True
            if attrs.get('is_defeatist', False):
                 coeff = self.coefficients['is_defeatist']
                 cognitive_component += coeff * rand_factors['is_defeatist'] * (p_worst - current)
                 any_special_cognitive_active = True
            if attrs.get('is_escapist', False):
                 coeff = self.coefficients['is_escapist']
                 cognitive_component += coeff * rand_factors['is_escapist'] * (current - p_worst)
                 any_special_cognitive_active = True

            # --- Add Standard Cognitive Influence (if flag active OR default needed) ---
            apply_std_cognitive = attrs.get('is_std_cognitive', False) or not any_special_cognitive_active
            if apply_std_cognitive:
                coeff = self.coefficients['is_std_cognitive']
                # Use the pre-generated random factor for consistency if flag was True,
                # or just use it anyway if applying default.
                cognitive_component += coeff * rand_factors['is_std_cognitive'] * (p_best - current)


            # --- Accumulate Special Social Influences & Track Activation ---
            any_special_social_active = False
            if attrs.get('is_rebel', False):
                 coeff = self.coefficients['is_rebel']
                 social_component += coeff * rand_factors['is_rebel'] * (current - g_best)
                 any_special_social_active = True
            if attrs.get('is_contrarian', False):
                 coeff = self.coefficients['is_contrarian']
                 social_component += coeff * rand_factors['is_contrarian'] * (g_worst - current)
                 any_special_social_active = True
            if attrs.get('is_eschewer', False):
                 coeff = self.coefficients['is_eschewer']
                 social_component += coeff * rand_factors['is_eschewer'] * (current - g_worst)
                 any_special_social_active = True

            # --- Add Standard Social Influence (if flag active OR default needed) ---
            apply_std_social = attrs.get('is_std_social', False) or not any_special_social_active
            if apply_std_social:
                coeff = self.coefficients['is_std_social']
                social_component += coeff * rand_factors['is_std_social'] * (g_best - current)


            new_velocity = self.w * velocity + cognitive_component + social_component
            particle.attributes['velocity'] = new_velocity.tolist()

    def get_name(self) -> str:
        return "HybridAdditivePSO"


class HybridPartialDisjointPSO_WithRandom(WorstAwarePSO, RoleMixin):
    """
    Hybrid Partial Disjoint PSO including Anarchic/Amnesiac roles.

    Partial Disjoint Strategy:
    - Cognitive Roles (Mutually Exclusive): standard, rejector, defeatist, escapist, amnesiac
    - Social Roles (Mutually Exclusive): standard, rebel, contrarian, eschewer, anarchic
    - Assignments are independent.

    Coefficients control magnitude for each role type.
    """
    def __init__(self,
                 problem: FloatProblem, swarm_size: int, termination_criterion: TerminationCriterion, w: float,
                 c1: float = 1.5, c2: float = 1.5,
                 rejector_c: float = 1.0, defeatist_c: float = 1.0, escapist_c: float = 1.0,
                 rebel_c: float = 1.0, contrarian_c: float = 1.0, eschewer_c: float = 1.0,
                 amnesiac_c: float = 1.0,
                 anarchic_c: float = 1.0,
                 rejector_fraction: float = 0.0, defeatist_fraction: float = 0.0, escapist_fraction: float = 0.0, amnesiac_fraction: float = 0.0,
                 rebel_fraction: float = 0.0, contrarian_fraction: float = 0.0, eschewer_fraction: float = 0.0, anarchic_fraction: float = 0.0,
                 constraint_handling_mode: str = "clip", assign_roles_every_iteration: bool = True):

        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.c1=c1; self.c2=c2; self.rejector_c=rejector_c; self.defeatist_c=defeatist_c; self.escapist_c=escapist_c
        self.rebel_c=rebel_c; self.contrarian_c=contrarian_c; self.eschewer_c=eschewer_c
        self.amnesiac_c = amnesiac_c
        self.anarchic_c = anarchic_c

        # Store fractions
        self.rejector_fraction = max(0.0, min(1.0, rejector_fraction))
        self.defeatist_fraction = max(0.0, min(1.0, defeatist_fraction))
        self.escapist_fraction = max(0.0, min(1.0, escapist_fraction))
        self.amnesiac_fraction = max(0.0, min(1.0, amnesiac_fraction))
        self.rebel_fraction = max(0.0, min(1.0, rebel_fraction))
        self.contrarian_fraction = max(0.0, min(1.0, contrarian_fraction))
        self.eschewer_fraction = max(0.0, min(1.0, eschewer_fraction))
        self.anarchic_fraction = max(0.0, min(1.0, anarchic_fraction))

        self.assign_roles_every_iteration = assign_roles_every_iteration

        cognitive_sum = self.rejector_fraction + self.defeatist_fraction + self.escapist_fraction + self.amnesiac_fraction
        social_sum = self.rebel_fraction + self.contrarian_fraction + self.eschewer_fraction + self.anarchic_fraction
        if cognitive_sum > 1.0: logger.warning(f"Sum of cognitive fractions ({cognitive_sum:.2f}) > 1.0.")
        if social_sum > 1.0: logger.warning(f"Sum of social fractions ({social_sum:.2f}) > 1.0.")

    def _assign_roles(self, swarm: List[S]) -> None:
        n = len(swarm)
        if n == 0: return
        for p in swarm:
            if not hasattr(p, 'attributes'): p.attributes = {}

        indices_cognitive = list(range(n)); random.shuffle(indices_cognitive)
        current_idx = 0
        counts_cog = {
            'rejector': int(n * self.rejector_fraction), 'defeatist': int(n * self.defeatist_fraction),
            'escapist': int(n * self.escapist_fraction), 'amnesiac': int(n * self.amnesiac_fraction) # Added amnesiac
        }
        for role_name, count in counts_cog.items():
            limit = min(current_idx + count, n)
            for i in range(current_idx, limit):
                swarm[indices_cognitive[i]].attributes['cognitive_role'] = role_name
            current_idx = limit
            if current_idx >= n: break
        while current_idx < n:
            swarm[indices_cognitive[current_idx]].attributes['cognitive_role'] = 'standard'
            current_idx += 1

        indices_social = list(range(n)); random.shuffle(indices_social)
        current_idx_social = 0
        counts_soc = {
            'rebel': int(n * self.rebel_fraction), 'contrarian': int(n * self.contrarian_fraction),
            'eschewer': int(n * self.eschewer_fraction), 'anarchic': int(n * self.anarchic_fraction) # Added anarchic
        }
        for role_name, count in counts_soc.items():
            limit = min(current_idx_social + count, n)
            for i in range(current_idx_social, limit):
                swarm[indices_social[i]].attributes['social_role'] = role_name
            current_idx_social = limit
            if current_idx_social >= n: break
        while current_idx_social < n:
            swarm[indices_social[current_idx_social]].attributes['social_role'] = 'standard'
            current_idx_social += 1
        # self._log_role_distribution(swarm)

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self._assign_roles(solutions)
        return solutions

    @staticmethod
    def _log_role_distribution(swarm: List[S]):
        counts = {}
        for p in swarm:
            role = p.attributes.get('assigned_role', 'N/A')
            counts[role] = counts.get(role, 0) + 1
        logger.debug("Assigned Role Distribution:")
        sorted_counts = sorted(counts.items())
        for role, count in sorted_counts:
            logger.debug(f"  Role: {role:<15} | Count: {count}")

    def step(self):
        if self.assign_roles_every_iteration:
            self._assign_roles(self.solutions)

        self.update_velocity(self.solutions)
        self.update_position(self.solutions)
        self.perturbation(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        # These already update both best/worst for global/particle via WorstAwarePSO
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[S]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)

    def update_velocity(self, swarm: List[S]) -> None:
        """Updates velocity including anarchic/amnesiac logic."""
        if self.best_global is None or self.global_worst is None or not swarm: return
        g_best = np.array(self.best_global.variables)
        g_worst = np.array(self.global_worst.variables)

        for particle in swarm:
            attrs = particle.attributes
            required_attrs = ['velocity', 'best_position', 'worst_position', 'cognitive_role', 'social_role']
            if not all(attr in attrs for attr in required_attrs): continue

            current = np.array(particle.variables); velocity = np.array(attrs['velocity'])
            p_best = np.array(attrs['best_position']); p_worst = np.array(attrs['worst_position'])
            cognitive_role = attrs['cognitive_role']; social_role = attrs['social_role']
            r1 = random.random(); r2 = random.random()

            if cognitive_role == 'rejector': cognitive_vec = self.rejector_c * r1 * (current - p_best)
            elif cognitive_role == 'defeatist': cognitive_vec = self.defeatist_c * r1 * (p_worst - current)
            elif cognitive_role == 'escapist': cognitive_vec = self.escapist_c * r1 * (current - p_worst)
            elif cognitive_role == 'amnesiac': cognitive_vec = self.amnesiac_c * np.random.uniform(-1.0, 1.0, self.problem.number_of_variables()) # Use amnesiac_c
            else: cognitive_vec = self.c1 * r1 * (p_best - current) # Standard

            if social_role == 'rebel': social_vec = self.rebel_c * r2 * (current - g_best)
            elif social_role == 'contrarian': social_vec = self.contrarian_c * r2 * (g_worst - current)
            elif social_role == 'eschewer': social_vec = self.eschewer_c * r2 * (current - g_worst)
            elif social_role == 'anarchic': social_vec = self.anarchic_c * np.random.uniform(-1.0, 1.0, self.problem.number_of_variables()) # Use anarchic_c
            else: social_vec = self.c2 * r2 * (g_best - current) # Standard

            new_velocity = self.w * velocity + cognitive_vec + social_vec
            particle.attributes['velocity'] = new_velocity.tolist()

    def get_name(self) -> str:
        return "HybridPartialDisjointPSO_WithRandom"


class HybridFullDisjointPSO_WithRandom(WorstAwarePSO, RoleMixin):
    """
    Hybrid Full Disjoint PSO including Anarchic/Amnesiac roles.

    Assigns AT MOST ONE special role overall. Includes random vector roles.
    - std_cognitive, std_social (implicit if role='standard')
    - rejector, defeatist, escapist, amnesiac (use standard social)
    - rebel, contrarian, eschewer, anarchic (use standard cognitive)

    Fractions for *all* special roles must sum <= 1.0.
    """
    def __init__(self,
                 problem: FloatProblem, swarm_size: int, termination_criterion: TerminationCriterion, w: float,
                 c1: float = 1.5, rejector_c: float = 1.0, defeatist_c: float = 1.0, escapist_c: float = 1.0, amnesiac_c: float = 1.0, # Cognitive + Amnesiac
                 c2: float = 1.5, rebel_c: float = 1.0, contrarian_c: float = 1.0, eschewer_c: float = 1.0, anarchic_c: float = 1.0, # Social + Anarchic
                 rejector_fraction: float = 0.0, defeatist_fraction: float = 0.0, escapist_fraction: float = 0.0, amnesiac_fraction: float = 0.0, # Cognitive specials
                 rebel_fraction: float = 0.0, contrarian_fraction: float = 0.0, eschewer_fraction: float = 0.0, anarchic_fraction: float = 0.0, # Social specials
                 constraint_handling_mode: str = "clip", assign_roles_every_iteration: bool = True):

        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        # Define roles and their coefficients
        self.coefficients = {
            'std_cognitive': c1, 'rejector': rejector_c, 'defeatist': defeatist_c, 'escapist': escapist_c, 'amnesiac': amnesiac_c,
            'std_social': c2, 'rebel': rebel_c, 'contrarian': contrarian_c, 'eschewer': eschewer_c, 'anarchic': anarchic_c,
        }
        self.special_cognitive_roles = {'rejector', 'defeatist', 'escapist', 'amnesiac'}
        self.special_social_roles = {'rebel', 'contrarian', 'eschewer', 'anarchic'}

        # Validate and process individual fractions
        self.special_role_fractions_input = { # Collect all fraction inputs
            'rejector': rejector_fraction, 'defeatist': defeatist_fraction, 'escapist': escapist_fraction, 'amnesiac': amnesiac_fraction,
            'rebel': rebel_fraction, 'contrarian': contrarian_fraction, 'eschewer': eschewer_fraction, 'anarchic': anarchic_fraction,
        }
        total_special_fraction = 0.0; active_special_fractions = {}
        for role_name, fraction in self.special_role_fractions_input.items():
            if not (0.0 <= fraction <= 1.0): raise ValueError(f"Fraction for {role_name} invalid.")
            if fraction > 1e-9: # Use tolerance to consider active
                 active_special_fractions[role_name] = fraction
                 total_special_fraction += fraction
        if total_special_fraction > 1.0 + 1e-9: raise ValueError(f"Sum of special fractions ({total_special_fraction:.4f}) > 1.0.")

        self.special_role_fractions = active_special_fractions
        self.standard_fraction = max(0.0, 1.0 - total_special_fraction)
        logger.info(f"Initializing HFDPSO+R: {total_special_fraction*100:.1f}% special, {self.standard_fraction*100:.1f}% standard.")
        self.assign_roles_every_iteration = assign_roles_every_iteration

    # _assign_roles remains the same structure, using self.special_role_fractions dict

    # create_initial_solutions, _log_role_distribution, step, update_particle/global_best remain the same
    def _assign_roles(self, swarm: List[S]) -> None:
        """
        Assigns at most one special role ('rejector', 'rebel', etc.) or
        'standard' to each particle based on the provided fractions.
        """
        n = len(swarm)
        if n == 0: return

        indices = list(range(n))
        random.shuffle(indices)
        current_idx = 0

        # Assign Special Roles first
        for role_name, fraction in self.special_role_fractions.items():
            # Calculate number to assign for this specific role
            num_to_assign = int(round(n * fraction))
            num_to_assign = min(num_to_assign, n - current_idx) # Cap by remaining particles

            for i in range(num_to_assign):
                if current_idx >= n: break
                particle_idx = indices[current_idx]
                # Initialize attributes dict if needed
                if not hasattr(swarm[particle_idx], 'attributes') or swarm[particle_idx].attributes is None:
                    swarm[particle_idx].attributes = {}
                # Assign the specific special role name
                swarm[particle_idx].attributes['assigned_role'] = role_name
                current_idx += 1
            if current_idx >= n: break # Stop if all particles assigned

        # Assign 'standard' to the remaining particles
        num_standard = n - current_idx
        # logger.debug(f"Assigning {num_standard} particles the 'standard' role.")
        for i in range(num_standard):
            if current_idx >= n: break # Should not happen ideally
            particle_idx = indices[current_idx]
            if not hasattr(swarm[particle_idx], 'attributes') or swarm[particle_idx].attributes is None:
                 swarm[particle_idx].attributes = {}
            # Assign 'standard' for standard-standard particles
            swarm[particle_idx].attributes['assigned_role'] = 'standard'
            current_idx += 1

        # Final check for debugging potential rounding issues
        if current_idx != n:
             logger.error(f"Role assignment mismatch: Assigned roles to {current_idx} out of {n} particles.")

        # self._log_role_distribution(swarm) # Optional logging


    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self._assign_roles(solutions) # Use the revised assignment method
        return solutions

    @staticmethod
    def _log_role_distribution(swarm: List[S]):
        counts = {}
        for p in swarm:
            role = p.attributes.get('assigned_role', 'N/A')
            counts[role] = counts.get(role, 0) + 1
        logger.debug("Assigned Role Distribution:")
        sorted_counts = sorted(counts.items())
        for role, count in sorted_counts:
            logger.debug(f"  Role: {role:<15} | Count: {count}")

    def step(self):
        if self.assign_roles_every_iteration:
            self._assign_roles(self.solutions) # Use the revised assignment method

        self.update_velocity(self.solutions)
        self.update_position(self.solutions)
        self.perturbation(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        # These already update both best/worst for global/particle via WorstAwarePSO
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[S]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)

    def update_velocity(self, swarm: List[S]) -> None:
        """Calculates velocity based on single assigned role (standard, special_cog, special_soc)."""
        if self.best_global is None or self.global_worst is None or not swarm: return
        g_best = np.array(self.best_global.variables); g_worst = np.array(self.global_worst.variables)

        for particle in swarm:
            attrs = particle.attributes
            required_attrs = ['velocity', 'best_position', 'worst_position', 'assigned_role']
            if not all(attr in attrs for attr in required_attrs): continue

            current = np.array(particle.variables); velocity = np.array(attrs['velocity'])
            p_best = np.array(attrs['best_position']); p_worst = np.array(attrs['worst_position'])
            assigned_role = attrs['assigned_role']
            r1 = random.random(); r2 = random.random()
            cognitive_vec = np.zeros_like(current); social_vec = np.zeros_like(current)
            new_velocity = np.zeros_like(current)

            # --- Determine Components based on Assigned Role ---
            if assigned_role == 'standard':
                coeff_c1 = self.coefficients['std_cognitive']; coeff_c2 = self.coefficients['std_social']
                cognitive_vec = coeff_c1 * r1 * (p_best - current)
                social_vec = coeff_c2 * r2 * (g_best - current)
                new_velocity = self.w * velocity + cognitive_vec + social_vec
            elif assigned_role in self.special_cognitive_roles:
                coeff_c2 = self.coefficients['std_social']
                social_vec = coeff_c2 * r2 * (g_best - current)
                coeff_spec = self.coefficients[assigned_role]
                if assigned_role == 'rejector': cognitive_vec = coeff_spec * r1 * (current - p_best)
                elif assigned_role == 'defeatist': cognitive_vec = coeff_spec * r1 * (p_worst - current)
                elif assigned_role == 'escapist': cognitive_vec = coeff_spec * r1 * (current - p_worst)
                elif assigned_role == 'amnesiac': cognitive_vec = coeff_spec * np.random.uniform(-1.0, 1.0, self.problem.number_of_variables())
                new_velocity = self.w * velocity + cognitive_vec + social_vec
            elif assigned_role in self.special_social_roles:
                coeff_c1 = self.coefficients['std_cognitive']
                cognitive_vec = coeff_c1 * r1 * (p_best - current)
                coeff_spec = self.coefficients[assigned_role]
                if assigned_role == 'rebel': social_vec = coeff_spec * r2 * (current - g_best)
                elif assigned_role == 'contrarian': social_vec = coeff_spec * r2 * (g_worst - current)
                elif assigned_role == 'eschewer': social_vec = coeff_spec * r2 * (current - g_worst)
                elif assigned_role == 'anarchic': social_vec = coeff_spec * np.random.uniform(-1.0, 1.0, self.problem.number_of_variables())
                new_velocity = self.w * velocity + cognitive_vec + social_vec
            else:
                 logger.error(f"Unrecognized assigned_role '{assigned_role}'. Applying standard velocity.")
                 coeff_c1 = self.coefficients['std_cognitive']; coeff_c2 = self.coefficients['std_social']
                 cognitive_vec = coeff_c1 * r1 * (p_best - current); social_vec = coeff_c2 * r2 * (g_best - current)
                 new_velocity = self.w * velocity + cognitive_vec + social_vec

            particle.attributes['velocity'] = new_velocity.tolist()

    def get_name(self) -> str:
        return "HybridFullDisjointPSO_WithRandom"

class HybridAdditivePSO_WithRandom(WorstAwarePSO, RoleMixin):
    """
    Hybrid Additive PSO including Anarchic/Amnesiac roles (Default-to-Standard).

    Allows simultaneous activation of multiple behaviors based on probabilities.
    Adds Anarchic (random social) and Amnesiac (random cognitive) possibilities.
    Includes default-to-standard logic.
    """
    def __init__(self,
                 problem: FloatProblem, swarm_size: int, termination_criterion: TerminationCriterion, w: float,
                 c1: float = 1.5, rejector_c: float = 1.0, defeatist_c: float = 1.0, escapist_c: float = 1.0, amnesiac_c: float = 1.0,
                 c2: float = 1.5, rebel_c: float = 1.0, contrarian_c: float = 1.0, eschewer_c: float = 1.0, anarchic_c: float = 1.0,
                 std_cognitive_prob: float = 1.0, rejector_prob: float = 0.0, defeatist_prob: float = 0.0, escapist_prob: float = 0.0, amnesiac_prob: float = 0.0, # Cog probs
                 std_social_prob: float = 1.0, rebel_prob: float = 0.0, contrarian_prob: float = 0.0, eschewer_prob: float = 0.0, anarchic_prob: float = 0.0, # Soc probs
                 constraint_handling_mode: str = "clip", assign_flags_every_iteration: bool = True):

        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.coefficients = {
            'is_std_cognitive': c1, 'is_rejector': rejector_c, 'is_defeatist': defeatist_c, 'is_escapist': escapist_c, 'is_amnesiac': amnesiac_c,
            'is_std_social': c2, 'is_rebel': rebel_c, 'is_contrarian': contrarian_c, 'is_eschewer': eschewer_c, 'is_anarchic': anarchic_c
        }
        prob_params = {
            'std_cognitive': std_cognitive_prob, 'rejector': rejector_prob, 'defeatist': defeatist_prob, 'escapist': escapist_prob, 'amnesiac': amnesiac_prob,
            'std_social': std_social_prob, 'rebel': rebel_prob, 'contrarian': contrarian_prob, 'eschewer': eschewer_prob, 'anarchic': anarchic_prob
        }
        self.role_probabilities = {}
        for role_base_name, prob in prob_params.items():
            flag_name = f"is_{role_base_name}"
            if not (0.0 <= prob <= 1.0): raise ValueError(f"Probability for {role_base_name} invalid.")
            if flag_name not in self.coefficients: raise ValueError(f"Flag {flag_name} missing coeff.")
            self.role_probabilities[flag_name] = prob

        self.assign_flags_every_iteration = assign_flags_every_iteration
        logger.info(f"Initializing HAPSO+R (Default-to-Standard) with individual probabilities.")

    # _assign_role_flags_to_swarm remains the same structure (iterates self.role_probabilities)

    # create_initial_solutions, _log_role_distribution, step, update_particle/global_best remain the same
    def _assign_role_flags_to_swarm(self, swarm: List[S]) -> None:
        """
        Assigns boolean flags (e.g., 'is_rejector') to each particle based
        on activation probabilities stored in self.role_probabilities.
        (This method remains unchanged).
        """
        n = len(swarm)
        if n == 0: return

        for i in range(n):
            particle = swarm[i]
            if not hasattr(particle, 'attributes') or particle.attributes is None:
                 particle.attributes = {}
            attrs = particle.attributes

            # Determine activation for each possible role flag using the stored probabilities
            for flag_name, probability in self.role_probabilities.items():
                attrs[flag_name] = (random.random() < probability)
        # self._log_role_distribution(swarm)


    # create_initial_solutions remains the same
    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self._assign_role_flags_to_swarm(solutions)
        return solutions

    # _log_role_distribution remains the same
    @staticmethod
    def _log_role_distribution(swarm: List[S]):
        if not swarm or not hasattr(swarm[0], 'attributes') or not swarm[0].attributes:
             logger.debug("Cannot log role distribution: Swarm empty or first particle has no attributes.")
             return
        flag_counts = {}
        known_flags = [f for f in swarm[0].attributes.keys() if f.startswith('is_')]
        for flag in known_flags: flag_counts[flag] = 0
        for p in swarm:
             for flag in known_flags:
                 if p.attributes.get(flag, False): flag_counts[flag] += 1
        logger.debug("Role Activation Counts (Additive):")
        sorted_counts = sorted(flag_counts.items())
        for flag, count in sorted_counts: logger.debug(f"  Flag: {flag:<20} | Active Count: {count}")

    # step remains the same
    def step(self):
        if self.assign_flags_every_iteration:
            self._assign_role_flags_to_swarm(self.solutions)

        self.update_velocity(self.solutions)
        self.update_position(self.solutions)
        self.perturbation(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[S]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)

    def update_velocity(self, swarm: List[S]) -> None:
        """Calculates velocity summing active roles, including anarchic/amnesiac, with default-to-standard."""
        if self.best_global is None or self.global_worst is None or not swarm: return
        g_best = np.array(self.best_global.variables); g_worst = np.array(self.global_worst.variables)

        for particle in swarm:
            attrs = particle.attributes; required_core_attrs = ['velocity', 'best_position', 'worst_position']
            if not all(attr in attrs for attr in required_core_attrs): continue
            for flag_name in self.coefficients.keys():
                if flag_name not in attrs: attrs[flag_name] = False

            current = np.array(particle.variables); velocity = np.array(attrs['velocity'])
            p_best = np.array(attrs['best_position']); p_worst = np.array(attrs['worst_position'])
            cognitive_component = np.zeros_like(current); social_component = np.zeros_like(current)
            rand_factors = {flag: random.random() for flag in self.coefficients}

            any_special_cognitive_active = False
            if attrs.get('is_rejector', False): coeff = self.coefficients['is_rejector']; cognitive_component += coeff * rand_factors['is_rejector'] * (current - p_best); any_special_cognitive_active = True
            if attrs.get('is_defeatist', False): coeff = self.coefficients['is_defeatist']; cognitive_component += coeff * rand_factors['is_defeatist'] * (p_worst - current); any_special_cognitive_active = True
            if attrs.get('is_escapist', False): coeff = self.coefficients['is_escapist']; cognitive_component += coeff * rand_factors['is_escapist'] * (current - p_worst); any_special_cognitive_active = True
            if attrs.get('is_amnesiac', False): coeff = self.coefficients['is_amnesiac']; cognitive_component += coeff * np.random.uniform(-1.0, 1.0, self.problem.number_of_variables()); any_special_cognitive_active = True # Added amnesiac

            apply_std_cognitive = attrs.get('is_std_cognitive', False) or not any_special_cognitive_active
            if apply_std_cognitive: coeff = self.coefficients['is_std_cognitive']; cognitive_component += coeff * rand_factors['is_std_cognitive'] * (p_best - current)

            any_special_social_active = False
            if attrs.get('is_rebel', False): coeff = self.coefficients['is_rebel']; social_component += coeff * rand_factors['is_rebel'] * (current - g_best); any_special_social_active = True
            if attrs.get('is_contrarian', False): coeff = self.coefficients['is_contrarian']; social_component += coeff * rand_factors['is_contrarian'] * (g_worst - current); any_special_social_active = True
            if attrs.get('is_eschewer', False): coeff = self.coefficients['is_eschewer']; social_component += coeff * rand_factors['is_eschewer'] * (current - g_worst); any_special_social_active = True
            if attrs.get('is_anarchic', False): coeff = self.coefficients['is_anarchic']; social_component += coeff * np.random.uniform(-1.0, 1.0, self.problem.number_of_variables()); any_special_social_active = True # Added anarchic

            apply_std_social = attrs.get('is_std_social', False) or not any_special_social_active
            if apply_std_social: coeff = self.coefficients['is_std_social']; social_component += coeff * rand_factors['is_std_social'] * (g_best - current)

            new_velocity = self.w * velocity + cognitive_component + social_component
            particle.attributes['velocity'] = new_velocity.tolist()

    def get_name(self) -> str:
        return "HybridAdditivePSO_WithRandom"


class HybridDisjointPSO_WithWanderer(WorstAwarePSO, RoleMixin):
    """
    Hybrid Partial Disjoint PSO with a single special role: wanderer.

    Partial-disjoint meaning here:
    - particle is either wanderer or standard
    - wanderer ignores both cognitive and social terms
    - standard uses normal PSO update

    Velocity update:
        standard:  w*v + c1*r1*(p_best-current) + c2*r2*(g_best-current)
        wanderer:  w*v + wanderer_c * U(-1,1)
    """

    def __init__(self,
                 problem: FloatProblem, swarm_size: int, termination_criterion: TerminationCriterion, w: float,
                 c1: float = 1.5, c2: float = 1.5,
                 wanderer_c: float = 1.0,
                 wanderer_fraction: float = 0.0,
                 constraint_handling_mode: str = "clip",
                 assign_roles_every_iteration: bool = True):

        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)

        self.c1 = c1
        self.c2 = c2
        self.wanderer_c = wanderer_c
        self.wanderer_fraction = max(0.0, min(1.0, wanderer_fraction))
        self.assign_roles_every_iteration = assign_roles_every_iteration

    def _assign_roles(self, swarm: List[S]) -> None:
        n = len(swarm)
        if n == 0:
            return

        for p in swarm:
            if not hasattr(p, 'attributes') or p.attributes is None:
                p.attributes = {}

        indices = list(range(n))
        random.shuffle(indices)

        wanderer_count = int(n * self.wanderer_fraction)

        for i, idx in enumerate(indices):
            if i < wanderer_count:
                swarm[idx].attributes['assigned_role'] = 'wanderer'
            else:
                swarm[idx].attributes['assigned_role'] = 'standard'

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self._assign_roles(solutions)
        return solutions

    def step(self):
        if self.assign_roles_every_iteration:
            self._assign_roles(self.solutions)

        self.update_velocity(self.solutions)
        self.update_position(self.solutions)
        self.perturbation(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)

    def update_global_best(self, swarm: List[S]) -> None:
        super().update_global_best(swarm)

    def update_velocity(self, swarm: List[S]) -> None:
        if self.best_global is None or not swarm:
            return

        g_best = np.array(self.best_global.variables)

        for particle in swarm:
            attrs = particle.attributes
            required_attrs = ['velocity', 'best_position', 'assigned_role']
            if not all(attr in attrs for attr in required_attrs):
                continue

            current = np.array(particle.variables)
            velocity = np.array(attrs['velocity'])
            p_best = np.array(attrs['best_position'])
            assigned_role = attrs['assigned_role']

            if assigned_role == 'wanderer':
                random_vec = self.wanderer_c * np.random.uniform(
                    -1.0, 1.0, self.problem.number_of_variables()
                )
                new_velocity = self.w * velocity + random_vec
            else:
                r1 = random.random()
                r2 = random.random()
                cognitive_vec = self.c1 * r1 * (p_best - current)
                social_vec = self.c2 * r2 * (g_best - current)
                new_velocity = self.w * velocity + cognitive_vec + social_vec

            particle.attributes['velocity'] = new_velocity.tolist()

    def get_name(self) -> str:
        return "HybridDisjointPSO_WithWanderer"


class HybridAdditivePSO_WithWanderer(WorstAwarePSO, RoleMixin):
    """
    Hybrid Additive PSO with wanderer as an additive random component.

    If is_wanderer is active:
        - random vector is ADDED on top of the normal update equation

    Velocity:
        w*v + cognitive_component + social_component + wanderer_component
    """

    def __init__(self,
                 problem: FloatProblem, swarm_size: int, termination_criterion: TerminationCriterion, w: float,
                 c1: float = 1.5, c2: float = 1.5,
                 wanderer_c: float = 1.0,
                 std_cognitive_prob: float = 1.0,
                 std_social_prob: float = 1.0,
                 wanderer_prob: float = 0.0,
                 constraint_handling_mode: str = "clip",
                 assign_flags_every_iteration: bool = True):

        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)

        self.coefficients = {
            'is_std_cognitive': c1,
            'is_std_social': c2,
            'is_wanderer': wanderer_c
        }

        prob_params = {
            'std_cognitive': std_cognitive_prob,
            'std_social': std_social_prob,
            'wanderer': wanderer_prob
        }

        self.role_probabilities = {}
        for role_base_name, prob in prob_params.items():
            flag_name = f"is_{role_base_name}"
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"Probability for {role_base_name} invalid.")
            if flag_name not in self.coefficients:
                raise ValueError(f"Flag {flag_name} missing coeff.")
            self.role_probabilities[flag_name] = prob

        self.assign_flags_every_iteration = assign_flags_every_iteration
        logger.info("Initializing HAPSO+W.")

    def _assign_role_flags_to_swarm(self, swarm: List[S]) -> None:
        n = len(swarm)
        if n == 0:
            return

        for particle in swarm:
            if not hasattr(particle, 'attributes') or particle.attributes is None:
                particle.attributes = {}
            attrs = particle.attributes

            for flag_name, probability in self.role_probabilities.items():
                attrs[flag_name] = (random.random() < probability)

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self._assign_role_flags_to_swarm(solutions)
        return solutions

    @staticmethod
    def _log_role_distribution(swarm: List[S]):
        if not swarm or not hasattr(swarm[0], 'attributes') or not swarm[0].attributes:
            logger.debug("Cannot log role distribution: Swarm empty or first particle has no attributes.")
            return

        flag_counts = {}
        known_flags = [f for f in swarm[0].attributes.keys() if f.startswith('is_')]
        for flag in known_flags:
            flag_counts[flag] = 0

        for p in swarm:
            for flag in known_flags:
                if p.attributes.get(flag, False):
                    flag_counts[flag] += 1

        logger.debug("Role Activation Counts (Additive):")
        for flag, count in sorted(flag_counts.items()):
            logger.debug(f"  Flag: {flag:<20} | Active Count: {count}")

    def step(self):
        if self.assign_flags_every_iteration:
            self._assign_role_flags_to_swarm(self.solutions)

        self.update_velocity(self.solutions)
        self.update_position(self.solutions)
        self.perturbation(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[S]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)

    def update_velocity(self, swarm: List[S]) -> None:
        if self.best_global is None or not swarm:
            return

        g_best = np.array(self.best_global.variables)

        for particle in swarm:
            attrs = particle.attributes
            required_core_attrs = ['velocity', 'best_position']
            if not all(attr in attrs for attr in required_core_attrs):
                continue

            for flag_name in self.coefficients.keys():
                if flag_name not in attrs:
                    attrs[flag_name] = False

            current = np.array(particle.variables)
            velocity = np.array(attrs['velocity'])
            p_best = np.array(attrs['best_position'])

            r1 = random.random()
            r2 = random.random()

            cognitive_component = np.zeros_like(current)
            social_component = np.zeros_like(current)
            wanderer_component = np.zeros_like(current)

            if attrs.get('is_std_cognitive', False):
                coeff = self.coefficients['is_std_cognitive']
                cognitive_component += coeff * r1 * (p_best - current)

            if attrs.get('is_std_social', False):
                coeff = self.coefficients['is_std_social']
                social_component += coeff * r2 * (g_best - current)

            if attrs.get('is_wanderer', False):
                coeff = self.coefficients['is_wanderer']
                wanderer_component += coeff * np.random.uniform(
                    -1.0, 1.0, self.problem.number_of_variables()
                )

            new_velocity = self.w * velocity + cognitive_component + social_component + wanderer_component
            particle.attributes['velocity'] = new_velocity.tolist()

    def get_name(self) -> str:
        return "HybridAdditivePSO_WithWanderer"


