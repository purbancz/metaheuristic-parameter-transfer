"""Reading experiment result files (HDF5) and file-level utilities."""


import numpy as np

from metaheuristiczoo.experiment.protocol import BASE_SEED
from metaheuristiczoo.experiment.results.writer import open_for_reading


def _run_ids(grp, seeds, n_rows):
    if 'run_ids' in grp:
        return grp['run_ids'][:].tolist()
    if len(seeds) == n_rows and all(int(s) >= BASE_SEED for s in seeds):
        return [int(s) - BASE_SEED + 1 for s in seeds]          # the runner's seeds encode the ids
    return None                                                 # no identity: rows are anonymous


def load_data(filepath, problems=None, algorithms=None):
    """Load experiment data from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file produced by run_all_experiments[_multi].
    problems : list[str] | None
        If given, only load these problem groups (exact names as stored).
    algorithms : list[str] | None
        If given, only load these algorithm groups within each problem.

    Returns a list of problem dicts compatible with combine_data() and all
    statistical / plotting functions:
      [{'problem': str, 'n_vars': int, 'results': {algo: {'data': ndarray, ...}}}]

    An algorithm entry carries `run_ids` (one per row of `data`) when the file knows them:
    the stored `run_ids` dataset, or for files written before it existed the ids the
    runner's seeds encode (seed = BASE_SEED + id - 1). Files with other seeds have no run
    identity and get no `run_ids`. Every entry carries `configuration_signature` (None for
    files without one), so that combine_data can tell the same algorithm configuration from
    a different one.
    """
    result = []
    try:
        with open_for_reading(filepath) as f:
            file_max_evaluations = f.attrs.get('max_evaluations')
            problem_keys = [k for k in f.keys() if problems is None or k in problems]
            for problem_name in problem_keys:
                problem_grp = f[problem_name]
                n_vars = int(problem_grp.attrs.get('n_vars', -1))
                section = problem_grp.attrs.get('section')
                optimum = problem_grp.attrs.get('known_optimum')
                reference = problem_grp.attrs.get('best_known')
                revision = problem_grp.attrs.get('problem_revision')
                results = {}
                algo_keys = [k for k in problem_grp.keys() if algorithms is None or k in algorithms]
                for algo_name in algo_keys:
                    grp = problem_grp[algo_name]
                    fitness_curves = grp['fitness_curves'][:]
                    final_fitness = grp['final_fitness'][:]
                    run_times = grp['run_times'][:]
                    seeds = grp['seeds'][:].tolist()
                    run_ids = _run_ids(grp, seeds, fitness_curves.shape[0])
                    signature = grp.attrs.get('configuration_signature')
                    valid_final = final_fitness[np.isfinite(final_fitness)]
                    results[algo_name] = {
                        'data': fitness_curves,
                        'final_fitness': final_fitness,
                        'run_times': run_times,
                        'seeds': seeds,
                        **({'run_ids': run_ids} if run_ids is not None else {}),
                        'configuration_signature': (signature.decode('utf-8') if isinstance(signature, bytes)
                                                    else (str(signature) if signature is not None else None)),
                        'avg_fitness': float(grp.attrs.get('avg_fitness',
                                             np.mean(valid_final) if valid_final.size > 0 else float('inf'))),
                        'std_dev': float(grp.attrs.get('std_dev',
                                         np.std(valid_final) if valid_final.size > 0 else float('nan'))),
                        'avg_time': float(grp.attrs.get('avg_time', np.mean(run_times))),
                    }
                if results:
                    entry = {'problem': problem_name, 'n_vars': n_vars, 'results': results}
                    if section is not None:
                        entry['section'] = str(section)
                    if optimum is not None:
                        entry['known_optimum'] = float(optimum)
                    if reference is not None:
                        entry['best_known'] = float(reference)
                    entry['problem_revision'] = None if revision is None else int(revision)
                    if file_max_evaluations is not None:
                         entry['max_evaluations'] = int(file_max_evaluations)
                    result.append(entry)
        print(f"Successfully loaded data from: {filepath}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None
    return result if result else None


