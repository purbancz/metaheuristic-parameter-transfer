import time
from typing import List, TypeVar

from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import Solution
from jmetal.operator.crossover import DifferentialEvolutionCrossover
from jmetal.operator.selection import DifferentialEvolutionSelection
from jmetal.util.comparator import ObjectiveComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar("S", bound=Solution)
R = TypeVar("R", bound=Solution)


class DifferentialEvolution(Algorithm[S, R]):
    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 crossover_operator: DifferentialEvolutionCrossover = None,
                 selection_operator: DifferentialEvolutionSelection = None,
                 solution_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria):
        super().__init__()
        self.solution_evaluator = solution_evaluator
        self.problem = problem
        self.solutions_size = swarm_size
        # Fresh operators per instance (3 Oct 2026): both carry per-run state (the selection's
        # index_to_exclude, the crossover's current_individual), and a shared default object leaked
        # that state from one run into the configuration signature of every later DE in the process.
        self.crossover_operator = crossover_operator if crossover_operator is not None \
            else DifferentialEvolutionCrossover(CR=0.9, F=0.5)
        self.selection_operator = selection_operator if selection_operator is not None \
            else DifferentialEvolutionSelection()
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.comparator = ObjectiveComparator(0)

    def create_initial_solutions(self) -> List[S]:
        return [self.problem.create_solution() for _ in range(self.solutions_size)]

    def evaluate(self, solution_list: List[S]):
        return self.solution_evaluator.evaluate(solution_list, self.problem)

    def step(self) -> None:
        mating_population = list(self.solutions)
        offspring_population = self.reproduction(mating_population)
        offspring_population = self.evaluate(offspring_population)
        self.solutions = self.replacement(self.solutions, offspring_population)
        # self.update_progress()

    def init_progress(self) -> None:
        self.evaluations = self.solutions_size
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def update_progress(self) -> None:
        self.evaluations += self.solutions_size

        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def result(self) -> R:
        best = min(self.solutions, key=lambda s: s.objectives[0])
        return best

    def get_name(self) -> str:
        return "Differential Evolution"

    def reproduction(self, mating_population: List[S]) -> List[S]:
        offspring_population = []
        for i in range(self.solutions_size):
            self.selection_operator.set_index_to_exclude(i)
            parents = self.selection_operator.execute(mating_population)
            self.crossover_operator.current_individual = mating_population[i]
            children = self.crossover_operator.execute(parents)
            offspring_population.append(children[0])
        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        new_population = []
        for i in range(self.solutions_size):
            if self.comparator.compare(population[i], offspring_population[i]) < 0:
                new_population.append(population[i])
            else:
                new_population.append(offspring_population[i])
        new_population.sort(key=lambda s: s.objectives[0])
        return new_population

    # def run(self):
    #     super().run()
    #     return self
