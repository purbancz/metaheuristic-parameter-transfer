import math
import os
import random
import socket
import time
import traceback

import numpy as np

from metaheuristiczoo.core.observer import FitnessObserver, SwarmTraceObserver
from metaheuristiczoo.experiment.protocol import BASE_SEED

class UndefinedObjective(RuntimeError):
    """A problem returned a non-finite objective. Every problem must define a finite value everywhere
    in its box (a continuous extension, or an explicit penalty, documented in the problem); the run fails
    rather than let each
    algorithm's comparisons decide what inf or NaN means."""


def guard_objectives(problem):
    """Make `problem.evaluate` refuse non-finite objectives (UndefinedObjective). Finite values pass
    through untouched, so a problem that is defined everywhere behaves bit for bit as before."""
    evaluate = problem.evaluate

    def guarded(solution):
        result = evaluate(solution)
        value = solution.objectives[0]
        if not math.isfinite(value):
            raise UndefinedObjective(
                f"{problem.name()} returned {value} at {list(solution.variables)}; a problem must define a "
                f"finite value everywhere in its box")
        return result
    problem.evaluate = guarded
    return problem


def _undefined_evaluations(problem):
    """Evaluations that fell where the problem applies its own rule for an otherwise undefined value
    (None for problems without undefined regions)."""
    count = getattr(problem, "undefined_evaluations", None)
    return None if count is None else int(count)


def _restart_count(algorithm):
    """Per-run restart counter of an algorithm that keeps one (`total_restarts` or
    `total_reinitializations`); None for the others, which include all 25 algorithms of the study."""
    count = getattr(algorithm, 'total_restarts', None)
    if count is None:
        count = getattr(algorithm, 'total_reinitializations', None)
    return count


def _where():
    """Where a run happened: the node and the SLURM job (array task included), for the compute
    accounting of the campaign (results.status)."""
    return {'host': socket.gethostname(),
            'slurm_job': os.environ.get('SLURM_JOB_ID', '') + (f"_{os.environ['SLURM_ARRAY_TASK_ID']}"
                                                            if os.environ.get('SLURM_ARRAY_TASK_ID') else '')}


### Multiprocessing
def run_single_instance(args):
    """
    Executes a single run of an algorithm on a problem.
    Designed to be called by multiprocessing.Pool.map.
    """
    algo_name, problem_instance_copy, algo_lambda, run_id, seed, max_evals, freq = args
    # print(f"  Starting Run {run_id} for {algo_name} on {problem_instance_copy.name()}...")

    try:
        set_run_seed(seed)
        set_problem_run_seed(problem_instance_copy, seed)
        guard_objectives(problem_instance_copy)

        algorithm = algo_lambda(problem_instance_copy)
        observer = FitnessObserver(interval=freq)
        algorithm.observable.register(observer)
        # the per-generation trace for every run; the particle positions for the first run only
        trace = SwarmTraceObserver(algorithm, max_evals, sample_positions=(run_id == 1))
        algorithm.observable.register(trace)

        start_time = time.time()
        algorithm.run()
        end_time = time.time()

        total_time = end_time - start_time
        result = algorithm.result()

        if result is None or not hasattr(result, 'objectives') or not result.objectives:
            print(f"    Run {run_id} Warning: No valid result/objectives.")
            final_fitness = float('inf')
            filled_fitness = [float('inf')] * (max_evals // freq)
        else:
            final_fitness = result.objectives[0]
            best_fitness_history = observer.best_fitness_history
            expected_len = max_evals // freq
            filled_fitness = best_fitness_history + [final_fitness] * (expected_len - len(best_fitness_history))
            filled_fitness = filled_fitness[:expected_len]

        try:
            trace_fields, trace_table = trace.matrix()
            population_sample = trace.sampled_population()
        except Exception as e:                  # the trace is diagnostics: a fault in it must never cost the run
            print(f"    Run {run_id} Warning: the swarm trace of {algo_name} on {problem_instance_copy.name()} could not "
                  f"be assembled ({type(e).__name__}: {e}); the run is stored without it.")
            traceback.print_exc()
            trace_fields, trace_table, population_sample = None, None, None
        # the final solution's variables: what a subset problem's decoder turns into the reported subset, so
        # every run's answer can be read back, not only run 1's position sample
        final_variables = (list(map(float, result.variables)) if result is not None and
                           getattr(result, "variables", None) is not None else None)
        return {'run_id': run_id, 'fitness_history': filled_fitness, 'time': total_time,
                'final_fitness': final_fitness, 'seed': seed, 'total_restarts': _restart_count(algorithm),
                'failed': False, 'trace_fields': trace_fields, 'trace': trace_table,
                'population_sample': population_sample, 'final_variables': final_variables,
                'undefined_evaluations': _undefined_evaluations(problem_instance_copy),
                'started_at': start_time, 'finished_at': end_time, **_where()}

    except Exception as e:
        print(f"    Run {run_id} ERROR for {algo_name} on {problem_instance_copy.name()}: {e}")
        traceback.print_exc()
        return {'run_id': run_id, 'fitness_history': [float('inf')] * (max_evals // freq), 'time': 0,
                'final_fitness': float('inf'), 'seed': seed, 'total_restarts': None, 'failed': True,
                'finished_at': time.time(), **_where()}


def generate_run_seeds(runs: int, base_seed: int = BASE_SEED) -> list[int]:
    return [base_seed + run_index for run_index in range(runs)]

def set_run_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def set_problem_run_seed(problem, seed: int) -> None:
    if hasattr(problem, "set_seed"):
        problem.set_seed(seed)
