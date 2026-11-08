"""The baseline arm of the combinatorial campaign: methods that search subset space directly.

    Baseline_Greedy        add the best site until the cardinality is reached; for maximum coverage
                           within 1 - 1/e of the optimum (Nemhauser, Wolsey and Fisher, "An analysis
                           of approximations for maximizing submodular set functions - I",
                           Mathematical Programming 14 (1978) 265-294)
    Baseline_ILS           iterated local search (Lourenco, Martin and Stuetzle): 1-swap descent,
                           kicks of strength 2, accept a kicked optimum that is not worse
    Baseline_RestartedLS   1-swap descent from independent random starts until the budget is spent
    Baseline_Random        independent random subsets - the floor any method has to clear

They run as ordinary entries of the main arm, through the same worker, store and
analysis as the metaheuristics, because the comparison they exist for is only clean that way: the
same instances, the same evaluation budget counted by the same `evaluation_count`, the same files.
The certified optima of the instances let their strength be shown without a reference that may itself
be beaten.

`SubsetBaseline` dresses a function of `problem/subset/baselines.py` as the algorithm the worker
expects - `observable`, `run()`, `result()`. The method searches first; then its improvement events
are replayed through the observers, twice per event (the old best just before it, the new best at
it), so the `FitnessObserver` curve is exact at whatever interval the campaign samples, and the trace
has a row per improvement. Every public attribute is a reproducible setting, because the result
writer describes an algorithm by its public attributes and matches stored runs by that description.
The random generator is drawn inside `run()` from the run seed the worker set, never at construction.

The baselines carry no tuned values (`in code`) and do not depend on the parameter set, so a control
run (`EXPERIMENT_CONTROL_SET`) does not repeat them.
"""

from __future__ import annotations

import math

import numpy as np
from jmetal.util.observable import DefaultObservable

from metaheuristiczoo.experiment.factories.budget import problem_budget
from metaheuristiczoo.experiment.parameters import IN_CODE, bind
from metaheuristiczoo.problem.subset import baselines


def _method(name: str):
    """The search baseline `name` of `baselines`."""
    return getattr(baselines, name)


class SubsetBaseline:
    """A baseline method as a campaign algorithm. `method` names a function of `baselines`."""

    def __init__(self, problem, method: str, max_evaluations: int, **settings):
        if not hasattr(problem, "score_subset"):
            raise TypeError(f"{method} searches subset space and needs a SubsetProblem, got {type(problem).__name__}")
        self.problem = problem
        self.method = method
        self.max_evaluations = int(max_evaluations)
        self.evaluations = 0
        self.observable = DefaultObservable()
        for name, value in settings.items():          # each setting its own attribute, so it is described
            setattr(self, name, value)
        self._settings = dict(settings)
        self._result = None

    def run(self) -> None:
        rng = np.random.default_rng(int(np.random.randint(0, 2 ** 31 - 1)))
        start = self.problem.evaluation_count
        if self.method == "greedy_forward":
            outcome = baselines.greedy_forward(self.problem, target_cardinality=self.problem.decoder.k,
                                               budget=self.max_evaluations)
        else:
            outcome = _method(self.method)(self.problem, self.max_evaluations, rng, **self._settings)
        self.evaluations = self.problem.evaluation_count - start
        self._replay(outcome, start)
        self._result = self._solution(outcome.best_subset, outcome.best_value)

    def result(self):
        return self._result

    def _replay(self, outcome, start: int) -> None:
        best = math.inf
        for counter, value in outcome.improvements:
            at = counter - start
            if best < math.inf and at > 1:
                self._notify(at - 1, best)
            best = value
            self._notify(at, best)
        if best < math.inf:
            self._notify(self.evaluations, best)

    def _notify(self, evaluations: int, value: float) -> None:
        self.observable.notify_all(EVALUATIONS=int(evaluations), SOLUTIONS=self._solution(None, value),
                                   PROBLEM=self.problem)

    def _solution(self, subset, value: float):
        solution = self.problem.create_solution()
        if subset is not None:
            chosen = set(int(i) for i in subset)
            solution.variables = [1.0 if i in chosen else 0.0 for i in range(self.problem.number_of_variables())]
        solution.objectives = [float(value)]
        return solution


@bind(IN_CODE)
def factory_Baseline_Greedy(p):
    return SubsetBaseline(p, "greedy_forward", problem_budget(p))


@bind(IN_CODE)
def factory_Baseline_ILS(p):
    return SubsetBaseline(p, "iterated_local_search", problem_budget(p),
                          strength=2, acceptance="not_worse", strategy="first", moves="swap")


@bind(IN_CODE)
def factory_Baseline_RestartedLS(p):
    return SubsetBaseline(p, "restarted_local_search", problem_budget(p), strategy="first", moves="swap")


@bind(IN_CODE)
def factory_Baseline_Random(p):
    return SubsetBaseline(p, "random_search", problem_budget(p))


SUBSET_BASELINES = {
    "Baseline_Greedy": factory_Baseline_Greedy,
    "Baseline_ILS": factory_Baseline_ILS,
    "Baseline_RestartedLS": factory_Baseline_RestartedLS,
    "Baseline_Random": factory_Baseline_Random,
}
