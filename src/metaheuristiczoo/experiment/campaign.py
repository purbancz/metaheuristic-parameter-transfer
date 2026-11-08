"""The experiment campaign: which algorithms run on which problems, under which protocol.

`build_campaign()` assembles the protocol (`experiment.protocol`), the 25 algorithms
(`experiment.roster`) and the ten instances (`experiment.suite`) into an immutable `Campaign`; with
arguments it assembles the part one job works on (some algorithms or problems), the same campaign seen
through a window, so its result files are the full campaign's result files.

The parameter set is chosen once, before any factory runs:

* by default the subset-tuned configuration S (`tuned_2026_10_subset_combined`); the four baselines run
  with it, results under `experiment_results/transfer/`;
* with `EXPERIMENT_CONTROL_SET=<set>` another set (C, or a follow-up set), without the baselines, results
  under `experiment_results/control/<set>/transfer/`;
* `EXPERIMENT_PARAMETER_SET=<set>` replaces the default of the main arm (not used in the study).

The chosen set is written to `EXPERIMENT_PARAMETER_SET`, so the worker processes read the same one, and
every result file records it.
"""

import fnmatch
import os
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from metaheuristiczoo.experiment.protocol import (NO_OF_RUNS, NUMBER_OF_VARIABLES, G_SOLUTIONS_SIZE,
                                                  EVALUATIONS_PER_DIMENSION, RESULTS_DIR)
from metaheuristiczoo.experiment.roster import active_factories
from metaheuristiczoo.experiment.results.layout import SCALABLE, TRANSFER
from metaheuristiczoo.experiment.suite import build_suite


@dataclass(frozen=True)
class Campaign:
    """What runs, on what, under which protocol, and where the results go.

    algorithms  ordered name -> factory; factory(problem) builds a configured algorithm
    problems    problem instances in suite order
    no_of_runs, number_of_variables, solutions_size, max_evaluations, frequency
                the protocol: independent runs per (algorithm, problem), campaign dimension (unused by
                the subset instances, which have their own), population size, nominal budget, and
                evaluations between fitness snapshots
    results_dir where the runner files the results (relative to the working directory)
    sections    problem name -> suite section; decides the results folder (results.layout)
    """
    algorithms: Mapping[str, Callable]
    problems: Sequence
    no_of_runs: int
    number_of_variables: int
    solutions_size: int
    max_evaluations: int
    frequency: int
    results_dir: str
    sections: Mapping[str, str] = field(default_factory=dict)

    def section_of(self, problem):
        return self.sections.get(problem.name(), SCALABLE)


def _select(names, spec, what):
    """The names matching `spec`: None keeps all; otherwise names or fnmatch globs, as a
    comma-separated string or a sequence. An exact name that matches nothing is an error, so a
    misspelt job scope fails at submission rather than running an empty campaign."""
    if spec is None:
        return list(names)
    patterns = spec.split(",") if isinstance(spec, str) else list(spec)
    patterns = [p.strip() for p in patterns if p.strip()]
    unknown = [p for p in patterns if not any(c in p for c in "*?[") and p not in names]
    if unknown:
        raise KeyError(f"unknown {what} requested: {unknown}")
    return [n for n in names if any(fnmatch.fnmatch(n, p) for p in patterns)]


def _resolve_parameter_set() -> str:
    """The parameter set of this job (see the module docstring), written to EXPERIMENT_PARAMETER_SET."""
    from metaheuristiczoo.experiment.parameters import (CONTROL_SET_ENV, DEFAULT_SUBSET_PARAMETER_SET,
                                                        DEFAULTED_SET_ENV, PARAMETER_SET_ENV, is_control_set,
                                                        load_parameter_set)
    # a value this function put there earlier in the process is its own default, not the user's choice
    if os.environ.get(DEFAULTED_SET_ENV) and os.environ.get(PARAMETER_SET_ENV) == os.environ[DEFAULTED_SET_ENV]:
        os.environ.pop(PARAMETER_SET_ENV, None)
    os.environ.pop(DEFAULTED_SET_ENV, None)

    def inject(name):
        os.environ[PARAMETER_SET_ENV] = os.environ[DEFAULTED_SET_ENV] = name

    control = os.environ.get(CONTROL_SET_ENV) or None
    explicit = os.environ.get(PARAMETER_SET_ENV) or None
    if control is not None:
        if explicit and explicit != control:
            raise ValueError(f"{CONTROL_SET_ENV}={control!r} and {PARAMETER_SET_ENV}={explicit!r} are both set; a "
                             f"control job is chosen by {CONTROL_SET_ENV} alone")
        load_parameter_set(control)                     # a missing set fails here, not inside a worker
        inject(control)
        print(f"Parameter set: {control} (control arm); results go to their own directory")
        return control
    if explicit is not None:
        if is_control_set(explicit):
            raise ValueError(f"{explicit!r} is a control set; run it with {CONTROL_SET_ENV}={explicit}, which keeps "
                             f"its results apart from the main arm's")
        return explicit
    inject(DEFAULT_SUBSET_PARAMETER_SET)
    print(f"Parameter set: {DEFAULT_SUBSET_PARAMETER_SET} (the default; set {CONTROL_SET_ENV} for another arm)")
    return DEFAULT_SUBSET_PARAMETER_SET


def build_campaign(sections=None, algorithms=None, problems=None):
    """The campaign, or the part of it given by `algorithms` and `problems` (names or globs).

    `sections` is accepted for compatibility with the campaign scripts and must be None or 'transfer'.
    """
    if sections is not None and _select([TRANSFER], sections, "suite sections") != [TRANSFER]:
        raise ValueError(f"this package has one suite section, {TRANSFER!r}")
    parameter_set = _resolve_parameter_set()
    from metaheuristiczoo.experiment.parameters import CONTROL_RESULTS_SUBDIR, CONTROL_SET_ENV
    control = bool(os.environ.get(CONTROL_SET_ENV))
    suite = build_suite()
    if problems is not None:
        names = [p.name() for block in suite.values() for p in block]
        wanted = set(_select(names, problems, "problems"))
        suite = {section: [p for p in block if p.name() in wanted] for section, block in suite.items()}
    factories = active_factories()
    if algorithms is not None:
        factories = {name: factories[name] for name in _select(list(factories), algorithms, "algorithms")}
    if not control:
        # the baseline arm runs with the main arm, first: it is the yardstick of every comparison and
        # does not depend on the parameter set
        from metaheuristiczoo.experiment.factories.subset_baselines import SUBSET_BASELINES
        factories = {**SUBSET_BASELINES, **factories}
    results_dir = os.path.join(RESULTS_DIR, CONTROL_RESULTS_SUBDIR, parameter_set) if control else RESULTS_DIR
    return Campaign(
        algorithms=factories,
        problems=tuple(problem for block in suite.values() for problem in block),
        sections={problem.name(): section for section, block in suite.items() for problem in block},
        no_of_runs=NO_OF_RUNS,
        number_of_variables=NUMBER_OF_VARIABLES,
        solutions_size=G_SOLUTIONS_SIZE,
        max_evaluations=EVALUATIONS_PER_DIMENSION * NUMBER_OF_VARIABLES,
        frequency=G_SOLUTIONS_SIZE,          # snapshot each generation
        results_dir=results_dir,
    )
