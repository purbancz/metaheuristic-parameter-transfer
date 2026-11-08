"""From a parameter set entry to a constructed algorithm: the one construction path of the package.

A parameter set (`experiment.parameters`) stores each algorithm's values as the tuner evaluated
them - scalars, keyed by the tuning block. Two steps turn them into an algorithm:

* `constructor_parameters(block, params, D)` wraps the scalars some constructors want as
  operator objects (SBX crossover and polynomial mutation for the GA-based blocks, the DE
  crossover for the DE-based ones) and adds the population keywords those classes name
  differently. The mutation probability is 1 / D of the problem the algorithm will run on.
* `build` calls the class with the problem, the swarm size and the termination criterion,
  overridden by the parameters.

The tuner of the research repository built every evaluated configuration through these two
functions, and the campaign's factories (`build`) build every algorithm through them, so a
winner runs in the campaign exactly as it was raced. `build` is what a factory calls: it reads
the block's values from a parameter set, applies the factory's fixed settings on top, and
terminates the run at the campaign budget.
"""

import inspect

from jmetal.operator.crossover import DifferentialEvolutionCrossover, SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from metaheuristiczoo.experiment import parameters as _parameters
from metaheuristiczoo.experiment.factories.budget import problem_budget
from metaheuristiczoo.experiment.protocol import G_SOLUTIONS_SIZE

GA_BLOCKS = ('IKUNGA',)
DE_BLOCKS = ('IKUNDE',)
POPULATION_BLOCKS = ('JADE', 'AGOA')
# keywords constructor_parameters adds for the classes that take them; dropped, like the defaults,
# by a constructor that does not
INJECTED = frozenset({'solutions_size', 'population_size', 'offspring_population_size', 'crossover', 'mutation',
                      'crossover_operator'})


def constructor_parameters(block: str, params: dict, number_of_variables: int,
                           solutions_size: int = G_SOLUTIONS_SIZE) -> dict:
    """The scalar parameters of `block` as its constructor takes them (see the module docstring)."""
    params = dict(params)
    if block in GA_BLOCKS:
        params['population_size'] = solutions_size
        params['offspring_population_size'] = solutions_size
        params['crossover'] = SBXCrossover(probability=params.pop('crossover_probability', 0.9),
                                           distribution_index=params.pop('sbx_distribution_index', 20.0))
        params['mutation'] = PolynomialMutation(probability=1.0 / number_of_variables,
                                                distribution_index=params.pop('mutation_distribution_index', 20.0))
    elif block in DE_BLOCKS:
        de_f = params.pop('F', 0.5)
        params['crossover_operator'] = DifferentialEvolutionCrossover(CR=params.pop('CR', 0.9), F=de_f, K=de_f)
    elif block in POPULATION_BLOCKS:
        params['population_size'] = solutions_size
    return params


def _call(algorithm_class, defaults: dict, params: dict, strict: bool):
    signature = inspect.signature(algorithm_class.__init__)
    accepted = {k for k in signature.parameters if k != 'self'}
    any_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    if strict and not any_keyword:
        unknown = sorted(set(params) - accepted - INJECTED)
        if unknown:
            raise TypeError(f"{algorithm_class.__name__} takes no parameter(s) {unknown}")
    arguments = {**defaults, **params}
    if not any_keyword:
        arguments = {k: v for k, v in arguments.items() if k in accepted}
    return algorithm_class(**arguments)


def build(algorithm_class, block: str, problem, *, parameter_set: str | None = None, **fixed):
    """The campaign's algorithm `algorithm_class` for `problem`, with the values of `block` in
    the parameter set (default: the active one, `parameters.active_parameter_set`) and the
    factory's `fixed` settings on top (variant switches, ablation overrides); the run stops at
    the campaign budget. A value the constructor does not take is an error: a parameter set
    must not lose a setting silently."""
    params = _parameters.parameters(block, parameter_set)
    params.update(fixed)
    params = constructor_parameters(block, params, int(problem.number_of_variables()))
    return _call(algorithm_class, {"problem": problem, "swarm_size": G_SOLUTIONS_SIZE,
                                   "termination_criterion": StoppingByEvaluations(max_evaluations=problem_budget(problem))},
                 params, strict=True)
