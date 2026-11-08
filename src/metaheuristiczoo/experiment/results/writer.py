"""Writing experiment result files (HDF5) run by run, with their provenance manifest.

One file per (problem, algorithm) - `<section dir>/<problem>_dim<D>_<algorithm>.h5` - holds:

* a manifest of the run protocol (configured runs, budget, population size, snapshot interval,
  base seeds, git commit, host, time of the last write);
* the problem identity (frozen class name, dimension, instance seed and id, suite section);
* the algorithm identity: the configuration of the algorithm the factory builds (class and
  every scalar setting, operators included) as JSON with its hash, plus the factory source
  for provenance. The configuration, not the factory's text, is what determines a run: many
  factories are closures with identical code over different tuned parameters;
* one group per completed run under `<problem>/<algorithm>/runs/<id>` with the fitness
  history, the seed, the run time, the final fitness and the restart count;
* the aggregated view every reader uses (`fitness_curves`, `final_fitness`, `seeds`,
  `run_times`, `run_ids`, optional `total_restarts`, and the avg/std/time attributes),
  rebuilt from the completed runs (in run-id order) after every write.

Runs are deterministic: run id r always uses seed BASE_SEED + r - 1 on the same problem
instance and the same algorithm configuration, so a completed run is reusable and the
campaign is resumable. `stored_runs` tells the runner which ids are already there, after
checking that the file was produced under the same identity; a file from another
configuration is an error, never silently extended. A file that cannot be opened is set
aside (renamed `*.corrupt-<time>`) and the pair is recomputed. A run stored with
`completed = False` (its worker raised) is excluded from the aggregates and retried.

A run whose entry cannot be opened (its object header damaged: h5py lists the key but opening it
fails, and `values()` yields None) counts as missing everywhere (`readable_runs`), so it is recomputed
instead of failing every reader and writer of the file. Writes never modify a result file in place: `write_run` writes into a
private copy (`<file>.writing`) and renames it over the file when the write is complete, so a write
that fails halfway leaves the stored runs as they were, and a reader only ever sees a finished file.
The copy is also where a file with damaged run entries is rebuilt without them.
"""

import errno
import hashlib
import inspect
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import h5py
import numpy as np

from metaheuristiczoo.experiment.parameters import parameter_set_of
from metaheuristiczoo.core.problem_identity import identity_name
from metaheuristiczoo.problem.optima import best_known, known_optimum
from metaheuristiczoo.experiment.protocol import BASE_SEED, BENCHMARK_BASE_SEED

IDENTITY_KEYS = ("problem_class", "n_vars", "instance_seed", "instance_id", "problem_revision",
                 "configuration_signature", "max_evaluations", "solutions_size", "evaluations_per_snapshot",
                 "base_seed", "benchmark_base_seed")


def problem_revision(problem):
    """The problem's DEFINITION_REVISION, None for a problem that was never redefined. It changes when
    the function itself changes (as two continuous objectives of the research repository did on 27 September 2026),
    so runs of one revision are never resumed into, or combined with, files of another."""
    return getattr(problem, "DEFINITION_REVISION", None)


class ResultIdentityMismatch(RuntimeError):
    """The result file on disk was produced under a different configuration."""


class ResultFileBusy(OSError):
    """The result file could not be read but was modified a moment ago: another job is writing it.
    The file is left untouched and the pair is skipped for now."""


READ_RETRY_SECONDS = 5.0        # a reader keeps trying this long: a file being appended to is unreadable for a moment
READ_RETRY_INTERVAL = 0.5
WRITE_RETRY_SECONDS = 240.0     # a writer waits this long for HDF5's file lock; shorter than the claim staleness (600 s),
WRITE_RETRY_INTERVAL = 0.5      # so a job waiting to write never loses the claim that makes it the file's only writer

_READ_WITHOUT_LOCK = None       # decided on first use: whether this h5py / HDF5 accepts locking=False


def open_for_reading(path):
    """`open_for_reading(path)` WITHOUT taking HDF5's file lock when the library allows it.

    HDF5 (1.10+) locks a file for the duration of every open: a writer needs the lock exclusively
    and a reader holds it shared, and a conflicting open fails at once with "unable to lock file,
    errno = 11" rather than waiting. On the campaign's shared filesystem that made a job's append
    to its own pair's file fail - and the job die - whenever any other process happened to have
    the file open for reading: another job's scan in the window between checking the claim and
    taking it, or a status script run while the jobs were writing. Readers therefore skip the lock: the claim protocol
    already guarantees one writer per file, and a reader that catches a file mid-write gets an
    OSError and retries (see `stored_runs`). Falls back to a locked open on an h5py or HDF5 too old
    for the per-file switch (h5py < 3.5, HDF5 < 1.12.1); set HDF5_USE_FILE_LOCKING=FALSE in the
    environment there instead.
    """
    global _READ_WITHOUT_LOCK
    if _READ_WITHOUT_LOCK is not False:
        try:
            handle = h5py.File(path, "r", locking=False)
            _READ_WITHOUT_LOCK = True
            return handle
        except (TypeError, ValueError):             # the keyword, or this HDF5, does not support it
            _READ_WITHOUT_LOCK = False
    return h5py.File(path, "r")


def _is_lock_contention(error):
    """Whether an OSError from h5py.File is transient: HDF5 failing to take the file lock (someone
    else has the file open), or append mode's create-if-missing colliding with a file that exists
    after all (errno 17: the read-write open reported the file absent - a metadata hiccup of a
    shared filesystem, or another process creating it at that instant - and the exclusive create
    then found it; on the next attempt the read-write open succeeds). Anything else - a corrupt
    file, a full disk - is not retried."""
    message = str(error)
    return (getattr(error, "errno", None) in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EEXIST)
            or "unable to lock file" in message or "file is already open" in message)


def open_for_writing(path):
    """`h5py.File(path, "a")`, waiting out lock contention (a reader has the file open for a
    moment) for up to WRITE_RETRY_SECONDS instead of failing at the first attempt. Any other
    OSError, and contention that outlasts the wait, is raised."""
    deadline = time.time() + WRITE_RETRY_SECONDS
    while True:
        try:
            return h5py.File(path, "a")
        except OSError as error:
            if not _is_lock_contention(error) or time.time() >= deadline:
                raise
            time.sleep(WRITE_RETRY_INTERVAL)


def git_commit():
    """The commit of the code that is running: `git rev-parse HEAD` in this package's own directory
    (the editable checkout), never in the working directory - on the clusters that is the campaign
    directory, which is not a repository, and the manifest would say 'unknown'."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, cwd=os.path.dirname(os.path.abspath(__file__))
        ).decode().strip()
    except Exception:
        return 'unknown'


# Characters that cannot appear in an HDF5 group name or a file name. '/' is the separator of both,
# so a problem whose name contains one would silently nest its results a level deeper and make every
# `h5[f"{problem}/{algo}/runs"]` lookup resolve somewhere else; ':' is illegal on Windows. No problem of
# the continuous campaign contains any of them (checked over the 69 names of the suite), so encoding
# them is a no-op there and the stored files keep their names; the subset-selection problems need it,
# because their names carry an instance path and a cut ("...(small-a/cell/max/k8:cut=0.04059,...)").
H5_UNSAFE = ('/', '\\', ':', '*', '?', '"', '<', '>', '|')


def h5_key(problem_name):
    """The name a problem's results are stored under: its own name with the unsafe characters
    encoded. Applied once at every public entry point of this module, so a caller always passes
    `problem.name()` and never an encoded form."""
    for character in H5_UNSAFE:
        problem_name = problem_name.replace(character, '_')
    return problem_name


def h5_result_path(section_dir, problem_name, algo_name, n_vars):
    safe = h5_key(problem_name).replace(' ', '_').replace('-', '_')
    return f'{section_dir}/{safe}_dim{n_vars}_{algo_name}.h5'


def factory_source(algorithm_factory):
    try:
        return inspect.getsource(algorithm_factory)
    except (OSError, TypeError):
        return repr(algorithm_factory)


def _describe(value, depth=0):
    """Scalar view of a constructor setting; None for what is not a setting (problem, arrays,
    observers, large collections)."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 32:
        items = [_describe(v, depth + 1) for v in value]
        return items if all(item is not None for item in items) else None
    if isinstance(value, dict) and len(value) <= 64:
        items = {str(k): _describe(v, depth + 1) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
        return {k: v for k, v in items.items() if v is not None} or None
    if depth == 0 and hasattr(value, "__dict__") and not hasattr(value, "evaluate"):
        # operators, termination criteria, memories: their own scalar settings
        nested = {k: _describe(v, depth + 1) for k, v in sorted(vars(value).items()) if not k.startswith("_")}
        nested = {k: v for k, v in nested.items() if v is not None}
        return {"class": type(value).__qualname__, **nested}
    return None


def algorithm_configuration(problem, algorithm_factory):
    """The configuration the factory produces on `problem`: class name plus every scalar setting
    of the constructed (not yet run) algorithm.

    The description has to be REPRODUCIBLE: the same factory on the same problem must describe
    the same way whenever it is asked, which is what lets a stored run be matched to a rerun.
    Built under a fixed RNG seed so that randomness in a constructor cannot make it drift, and
    the global RNG state is restored afterwards. Nothing is filtered out by name: a constructor
    that leaves something unreproducible on the algorithm - a wall-clock reading, say - is a bug
    in that constructor, and `tests/experiment/results/test_configuration_signature.py` fails for
    it rather than this function hiding it."""
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        random.seed(BASE_SEED)
        np.random.seed(BASE_SEED)
        algorithm = algorithm_factory(problem)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
    settings = {k: _describe(v) for k, v in sorted(vars(algorithm).items()) if not k.startswith("_")}
    settings = {k: v for k, v in settings.items() if v is not None}
    settings["class"] = type(algorithm).__qualname__
    return settings


def configuration_signature(configuration):
    return hashlib.sha256(json.dumps(configuration, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def run_identity(problem, algorithm_factory, max_evaluations, solutions_size, frequency):
    """What a stored run must agree on to stand in for a run of this configuration. The
    `configuration` entry (not compared field by field, its signature is) is kept so a
    mismatch can name the settings that differ."""
    configuration = algorithm_configuration(problem, algorithm_factory)
    return {
        "problem_class": identity_name(type(problem)),
        "n_vars": int(problem.number_of_variables()),
        "instance_seed": getattr(problem, "instance_seed", None),
        "instance_id": getattr(problem, "instance_id", None),
        "problem_revision": problem_revision(problem),
        "configuration_signature": configuration_signature(configuration),
        "configuration": configuration,
        "max_evaluations": int(max_evaluations),
        "solutions_size": int(solutions_size),
        "evaluations_per_snapshot": int(frequency),
        "base_seed": int(BASE_SEED),
        "benchmark_base_seed": int(BENCHMARK_BASE_SEED),
    }


def run_seed(run_id):
    """Seed of run id r (1-based): BASE_SEED + r - 1, the campaign's deterministic run seeds."""
    return int(BASE_SEED) + int(run_id) - 1


def _attr(node, key):
    value = node.attrs.get(key)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def readable_runs(runs_grp):
    """(key, group) of every stored run under a `runs` group whose entry can be opened, in key order.
    An entry that cannot (a damaged object header: the key is listed but opening it fails) is left
    out, so the run counts as missing and is recomputed; `damaged_runs` names them."""
    if runs_grp is None:
        return
    for key in runs_grp.keys():
        try:
            run = runs_grp[key]
        except (KeyError, OSError, RuntimeError, ValueError):
            continue
        yield key, run


def damaged_runs(runs_grp):
    """Keys of the run entries under a `runs` group that cannot be opened (see `readable_runs`)."""
    if runs_grp is None:
        return []
    readable = {key for key, _ in readable_runs(runs_grp)}
    return [key for key in runs_grp.keys() if key not in readable]


def _stored_identity(h5, problem_name, algo_name):
    problem_grp = h5[problem_name]
    algo_grp = problem_grp[algo_name]
    return {
        "problem_class": _attr(problem_grp, "problem_class"),
        "n_vars": _attr(problem_grp, "n_vars"),
        "instance_seed": _attr(problem_grp, "instance_seed"),
        "instance_id": _attr(problem_grp, "instance_id"),
        "problem_revision": _attr(problem_grp, "problem_revision"),
        "configuration_signature": _attr(algo_grp, "configuration_signature"),
        "configuration": json.loads(_attr(algo_grp, "configuration_json") or "{}"),
        "max_evaluations": _attr(h5, "max_evaluations"),
        "solutions_size": _attr(h5, "solutions_size"),
        "evaluations_per_snapshot": _attr(h5, "evaluations_per_snapshot"),
        "base_seed": _attr(h5, "base_seed"),
        "benchmark_base_seed": _attr(h5, "benchmark_base_seed"),
    }


def stored_runs(path, problem_name, algo_name, identity, retry_seconds=None, stale_after=None):
    """Ids of the completed runs in `path` that are reusable for `identity`.

    Returns an empty set when the file does not exist. Raises ResultIdentityMismatch when the file
    exists but was written under a different problem instance, algorithm configuration or
    protocol, or when a stored run's seed is not the seed its id prescribes.

    A file that cannot be read is retried for `retry_seconds` (default READ_RETRY_SECONDS): a file
    another process is appending to is unreadable for a moment, not corrupt (HDF5 refuses or
    fails a concurrent open while a writer holds the file). When it stays unreadable it is set
    aside (renamed `*.corrupt-<time>`) and an empty set is returned, so its pair is recomputed -
    unless `stale_after` is given and the file was modified less than that many seconds ago: then
    ResultFileBusy is raised and the file is left untouched, because a concurrent job may still
    be writing it and setting a live file aside would throw its runs away. The campaign queue
    passes the claim staleness for this; a single job leaves it None.
    """
    problem_name = h5_key(problem_name)
    if not os.path.exists(path):
        return set()
    deadline = time.time() + (READ_RETRY_SECONDS if retry_seconds is None else retry_seconds)
    while True:
        try:
            with open_for_reading(path) as h5:
                return _completed_runs(h5, path, problem_name, algo_name, identity)
        except OSError as error:
            if time.time() < deadline:
                time.sleep(READ_RETRY_INTERVAL)
                continue
            try:
                age = time.time() - os.path.getmtime(path)
            except FileNotFoundError:                   # set aside or removed by another job meanwhile
                return set()
            if stale_after is not None and age < stale_after:
                raise ResultFileBusy(f"{path} could not be read ({error}) and was modified {age:.0f} s ago; "
                                     f"another job is probably writing it, so it is left alone") from error
            aside = f"{path}.corrupt-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                os.replace(path, aside)
            except OSError as rename_error:              # held open by another process: not ours to move
                raise ResultFileBusy(f"{path} could not be read ({error}) nor set aside ({rename_error}); "
                                     f"left alone") from error
            print(f"WARNING: {path} could not be read ({error}); set aside as {aside}, its runs are recomputed.")
            return set()


def _completed_runs(h5, path, problem_name, algo_name, identity):
    """The identity check and the completed run ids of an open result file (see `stored_runs`)."""
    if problem_name not in h5 or algo_name not in h5[problem_name]:
        raise ResultIdentityMismatch(
            f"{path} holds no results for {algo_name} on {problem_name}; "
            f"it contains {list(h5.keys())} - move it away before running")
    stored = _stored_identity(h5, problem_name, algo_name)
    differences = {key: (stored[key], identity[key]) for key in IDENTITY_KEYS if stored[key] != identity[key]}
    if "configuration_signature" in differences:
        old_cfg, new_cfg = stored["configuration"], identity["configuration"]
        changed = {k: (old_cfg.get(k), new_cfg.get(k)) for k in sorted(set(old_cfg) | set(new_cfg))
                   if old_cfg.get(k) != new_cfg.get(k)}
        differences["configuration_signature"] = changed or differences["configuration_signature"]
    if differences:
        raise ResultIdentityMismatch(
            f"{path} was produced under a different configuration; differing fields "
            f"(stored, current): {differences}. Move the file away (or restore the configuration) "
            f"before running - stored runs are never mixed with another configuration.")
    runs_grp = h5[problem_name][algo_name].get("runs")
    completed = set()
    for key, grp in readable_runs(runs_grp):
        if not _attr(grp, "completed") or "fitness_history" not in grp:
            continue
        run_id = int(key)
        if _attr(grp, "seed") != run_seed(run_id):
            raise ResultIdentityMismatch(
                f"{path}: run {run_id} was stored with seed {_attr(grp, 'seed')}, but run {run_id} of this "
                f"campaign uses seed {run_seed(run_id)} - the file belongs to another seeding protocol")
        completed.add(run_id)
    return completed


def stored_run_times(path, problem_name, algo_name):
    """Wall times (seconds) of the completed runs stored for the pair; empty when there are none.
    What the campaign runner estimates a further run of the pair from."""
    problem_name = h5_key(problem_name)
    if not os.path.exists(path):
        return []
    try:
        with open_for_reading(path) as h5:
            runs_grp = h5.get(f"{problem_name}/{algo_name}/runs")
            if runs_grp is None:
                return []
            return [float(_attr(grp, "run_time")) for _, grp in readable_runs(runs_grp)
                    if _attr(grp, "completed") and _attr(grp, "run_time") is not None]
    except OSError:
        return []


RUN_RECORD_KEYS = ("run_id", "seed", "completed", "run_time", "final_fitness", "total_restarts",
                   "undefined_evaluations", "started_at", "finished_at", "host", "slurm_job")


def write_manifest(h5file, n_vars, n_runs, max_evaluations, solutions_size, frequency):
    h5file.attrs['git_commit'] = git_commit()
    h5file.attrs['timestamp'] = datetime.now().isoformat()
    h5file.attrs['python_version'] = sys.version
    h5file.attrs['numpy_version'] = np.__version__
    h5file.attrs['hostname'] = socket.gethostname()
    h5file.attrs['n_vars'] = n_vars
    h5file.attrs['no_of_runs'] = n_runs
    h5file.attrs['base_seed'] = BASE_SEED
    h5file.attrs['benchmark_base_seed'] = BENCHMARK_BASE_SEED
    h5file.attrs['max_evaluations'] = max_evaluations
    h5file.attrs['solutions_size'] = solutions_size
    h5file.attrs['evaluations_per_snapshot'] = frequency


def _write_identity(h5, problem_name, algo_name, problem, algorithm_factory, configuration, section):
    h5.attrs['problem_name'] = problem_name
    h5.attrs['algo_name'] = algo_name
    problem_grp = h5.require_group(problem_name)
    problem_grp.attrs['n_vars'] = int(problem.number_of_variables())
    if section is not None:
        problem_grp.attrs['section'] = section     # suite section: decides the results folder
    problem_grp.attrs['problem_class'] = identity_name(type(problem))
    optimum = known_optimum(problem)
    if optimum is not None:                        # the error scale for fixed-target analyses; absent when unknown
        problem_grp.attrs['known_optimum'] = float(optimum)
    reference = best_known(problem)
    if reference is not None:                      # provenance for an open instance: a value runs may beat
        problem_grp.attrs['best_known'] = float(reference)
    if hasattr(problem, 'instance_seed'):
        problem_grp.attrs['instance_seed'] = problem.instance_seed
    if getattr(problem, 'instance_id', None):
        problem_grp.attrs['instance_id'] = problem.instance_id
    if problem_revision(problem) is not None:
        problem_grp.attrs['problem_revision'] = int(problem_revision(problem))
    grp = problem_grp.require_group(algo_name)
    grp.attrs['factory_source'] = factory_source(algorithm_factory)
    # provenance only: the configuration (and its signature) decides identity, so the same values
    # under two set names are the same algorithm; the last writer's set name is kept
    grp.attrs['parameter_set'] = parameter_set_of(algorithm_factory)
    grp.attrs['configuration_json'] = json.dumps(configuration, sort_keys=True, default=str)
    grp.attrs['configuration_signature'] = configuration_signature(configuration)
    return grp


def _replace_dataset(grp, name, data, **kwargs):
    if name in grp:
        del grp[name]
    grp.create_dataset(name, data=data, **kwargs)


def _refresh_aggregates(grp):
    """Rebuild the aggregated view (what every reader uses) from the completed runs, in run-id order."""
    runs = grp.require_group("runs")
    completed = sorted(int(key) for key, run in readable_runs(runs)
                       if run.attrs.get("completed", False) and "fitness_history" in run)
    if not completed:
        for name in ("fitness_curves", "final_fitness", "seeds", "run_times", "run_ids", "total_restarts"):
            if name in grp:
                del grp[name]
        for key in ("avg_fitness", "std_dev", "avg_time"):
            grp.attrs.pop(key, None)
        return
    groups = [runs[f"{run_id:04d}"] for run_id in completed]
    curves = np.array([g["fitness_history"][:] for g in groups], dtype=np.float64)
    final = np.array([float(g.attrs["final_fitness"]) for g in groups], dtype=np.float64)
    times = np.array([float(g.attrs["run_time"]) for g in groups], dtype=np.float64)
    seeds = np.array([int(g.attrs["seed"]) for g in groups], dtype=np.int64)
    _replace_dataset(grp, "fitness_curves", curves, compression="gzip")
    _replace_dataset(grp, "final_fitness", final)
    _replace_dataset(grp, "seeds", seeds)
    _replace_dataset(grp, "run_times", times)
    _replace_dataset(grp, "run_ids", np.array(completed, dtype=np.int64))
    restarts = [g.attrs.get("total_restarts") for g in groups]
    if any(r is not None for r in restarts):
        _replace_dataset(grp, "total_restarts", np.array([-1 if r is None else int(r) for r in restarts], dtype=np.int64))
    elif "total_restarts" in grp:
        del grp["total_restarts"]
    valid = final[np.isfinite(final)]
    grp.attrs["avg_fitness"] = float(np.mean(valid)) if valid.size else float("inf")
    grp.attrs["std_dev"] = float(np.std(valid)) if valid.size else float("nan")
    grp.attrs["avg_time"] = float(np.mean(times)) if times.size else 0.0


def write_run(path, problem_name, algo_name, run, no_of_runs, max_evaluations, solutions_size, frequency,
              problem, algorithm_factory, configuration, section=None):
    """Store one run (the dict returned by the runner's worker) and refresh the aggregated view.
    `configuration` is the algorithm configuration of the pair (run_identity()["configuration"]).

    `run` carries `run_id`, `seed`, `fitness_history`, `time`, `final_fitness`, `total_restarts`
    and `failed`; a failed run is stored with `completed = False` so it is retried next time.
    Optionally `trace` / `trace_fields` (the per-generation table of core.observer.SwarmTraceObserver)
    and `population_sample` (its sampled generations: positions, objectives, roles), stored as they are.
    """
    problem_name = h5_key(problem_name)
    with _private_copy(path) as working, open_for_writing(working) as h5:
        write_manifest(h5, int(problem.number_of_variables()), no_of_runs, max_evaluations, solutions_size, frequency)
        grp = _write_identity(h5, problem_name, algo_name, problem, algorithm_factory, configuration, section)
        runs = grp.require_group("runs")
        key = f"{int(run['run_id']):04d}"
        if key in runs:
            del runs[key]
        g = runs.create_group(key)
        g.create_dataset("fitness_history", data=np.asarray(run["fitness_history"], dtype=np.float64), compression="gzip")
        g.attrs["seed"] = int(run["seed"])
        g.attrs["run_time"] = float(run["time"])
        g.attrs["final_fitness"] = float(run["final_fitness"])
        for key in ("started_at", "finished_at"):                 # epoch seconds; where and when, for the accounting
            if run.get(key) is not None:
                g.attrs[key] = float(run[key])
        for key in ("host", "slurm_job"):
            if run.get(key):
                g.attrs[key] = str(run[key])
        if run.get("total_restarts") is not None:
            g.attrs["total_restarts"] = int(run["total_restarts"])
        if run.get("undefined_evaluations") is not None:            # evaluations where the problem's own rule applied
            g.attrs["undefined_evaluations"] = int(run["undefined_evaluations"])
        if run.get("trace") is not None and len(run["trace"]):
            trace = g.create_dataset("trace", data=np.asarray(run["trace"], dtype=np.float32), compression="gzip")
            trace.attrs["fields"] = [str(f) for f in run["trace_fields"]]
        if run.get("final_variables") is not None:                # the run's answer (since 24 Oct 2026)
            g.create_dataset("final_variables", data=np.asarray(run["final_variables"], dtype=np.float64))
        sample = run.get("population_sample")
        if sample is not None:
            g.create_dataset("positions", data=np.asarray(sample["positions"], dtype=np.float16), compression="gzip")
            g.create_dataset("positions_evaluations", data=np.asarray(sample["evaluations"], dtype=np.int64))
            g.create_dataset("positions_objectives", data=np.asarray(sample["objectives"], dtype=np.float32), compression="gzip")
            roles = g.create_dataset("positions_roles", data=np.asarray(sample["roles"], dtype=np.int16), compression="gzip")
            roles.attrs["role_names"] = [str(r) for r in sample["role_names"]]
            if sample.get("population_sizes") is not None:      # true sizes behind the padded rows
                g.create_dataset("positions_population", data=np.asarray(sample["population_sizes"], dtype=np.int64))
        g.attrs["completed"] = not run.get("failed", False)
        _refresh_aggregates(grp)


WORKING_SUFFIX = ".writing"


@contextmanager
def _private_copy(path):
    """The file `write_run` writes instead of `path`: a copy of it (or nothing, for a new file) that
    only this writer opens, renamed over `path` once the write is complete. The pair's claim makes the
    job the file's only writer, so the working name needs no uniqueness, and one left behind by a
    writer that died mid-write is simply replaced. A write that raises removes the copy and leaves
    `path` untouched. A file with damaged run entries is copied without them (`_rebuild_without`)."""
    working = path + WORKING_SUFFIX
    _remove_quietly(working)
    if os.path.exists(path):
        damaged = _damaged_entries(path)
        if damaged:
            _rebuild_without(path, working, damaged)
        else:
            shutil.copyfile(path, working)
    try:
        yield working
    except BaseException:
        _remove_quietly(working)
        raise
    with open(working, "rb+") as handle:            # on disk before the rename makes it the file
        os.fsync(handle.fileno())
    os.replace(working, path)


def _remove_quietly(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _damaged_entries(path):
    """'problem/algorithm/runs/key' of every run entry of the file that cannot be opened."""
    found = []
    with open_for_reading(path) as h5:
        for problem_name, problem_grp in h5.items():
            if not isinstance(problem_grp, h5py.Group):
                continue
            for algo_name, algo_grp in problem_grp.items():
                if isinstance(algo_grp, h5py.Group):
                    found += [f"{problem_name}/{algo_name}/runs/{key}" for key in damaged_runs(algo_grp.get("runs"))]
    return found


def _rebuild_without(source, target, damaged):
    """Write `target` as a copy of `source` without its damaged run entries: the attributes, every
    readable run and the aggregated view (which the write that follows rebuilds from the runs). A run
    whose copy fails part-way is dropped as damaged too. Its runs are then missing and recomputed."""
    dropped = list(damaged)
    with open_for_reading(source) as src, h5py.File(target, "w") as dst:
        dst.attrs.update(dict(src.attrs))
        for problem_name, problem_grp in src.items():
            if not isinstance(problem_grp, h5py.Group):
                continue
            dst_problem = dst.create_group(problem_name)
            dst_problem.attrs.update(dict(problem_grp.attrs))
            for algo_name in problem_grp.keys():
                algo_grp = problem_grp[algo_name]
                dst_algo = dst_problem.create_group(algo_name)
                dst_algo.attrs.update(dict(algo_grp.attrs))
                for name in algo_grp.keys():
                    if name != "runs":
                        src.copy(algo_grp[name], dst_algo, name=name)
                        continue
                    dst_runs = dst_algo.create_group("runs")
                    for key, run in readable_runs(algo_grp["runs"]):
                        try:
                            src.copy(run, dst_runs, name=key)
                        except (KeyError, OSError, RuntimeError, ValueError):
                            if key in dst_runs:
                                del dst_runs[key]
                            dropped.append(f"{problem_name}/{algo_name}/runs/{key}")
    print(f"WARNING: {source} had damaged run entries {dropped}; rewritten without them, so those runs "
          f"are recomputed.", flush=True)


