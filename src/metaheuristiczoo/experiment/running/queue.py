"""Running the campaign as a queue of (problem, algorithm) pairs shared by concurrent jobs.

The campaign is 283 algorithms x 72 problems x 30 runs; no single allocation holds it, and the
pairs differ in cost by an order of magnitude, so a static split by count would leave some jobs
idle while others overrun. Instead every job runs this loop: claim the next pair nobody holds
(`core.claims`), run its missing runs, store each run the moment it finishes, release the pair,
repeat until the queue is empty or the job's deadline is near. Any number of jobs can run this
concurrently without coordination beyond the claim files, and a killed job loses only the runs
that were in flight: everything stored is picked up by the next job through the resumable
storage (`results.writer.stored_runs`).

Three rules the loop keeps:

* The unit of work is the pair, never the run. A pair owns one result file, the writer opens it
  in append mode with no locking, so two jobs must never hold the same pair. The claim is what
  guarantees that.
* Runs of several pairs share one process pool. A pair has 30 runs and a node often has more
  cores; parallelising within a pair alone would idle part of the node on every pair.
* No run starts that cannot end before the deadline. The estimate is the mean wall time of the
  pair's stored runs, else of this job's runs of the same algorithm, stretched by ESTIMATE_SAFETY
  and padded by DEADLINE_MARGIN_SECONDS; with no estimate at all the run is started (the first
  wave of an unknown algorithm is the price of learning its cost). Runs that do not fit stay
  missing for the next job, and their pair's claim is released so that job can take it.
* A dead worker never stalls the job. The pool is a `concurrent.futures.ProcessPoolExecutor`,
  whose pending tasks fail with BrokenProcessPool when a worker process dies abruptly (killed
  by the out-of-memory killer, a crash in native code). The runs that were in flight are put
  back into their pairs, a fresh pool is started, and a run lost MAX_LOSSES_PER_RUN times is
  stored as failed so a later job retries it and this one moves on. A `multiprocessing.Pool`
  used to be here: it replaces a dead worker silently and never delivers that worker's task, so
  the job waited for ever while its heartbeat kept the pair's claim alive.

The queue is ordered as the campaign is: section, then problem, then algorithm, so complete
problem columns - what the per-problem statistics need - come out first.
"""

import copy
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from multiprocessing import cpu_count

from metaheuristiczoo.core.claims import CLAIM_HEARTBEAT_SECONDS, CLAIM_STALE_SECONDS, is_held, try_claim
from metaheuristiczoo.experiment.results.layout import make_dir, results_subdir
from metaheuristiczoo.experiment.results.writer import (ResultFileBusy, ResultIdentityMismatch, h5_result_path,
                                                         run_identity, stored_run_times, stored_runs, write_run)
from metaheuristiczoo.experiment.factories.budget import problem_budget
from metaheuristiczoo.experiment.running.runner import _where, generate_run_seeds, run_single_instance

DEADLINE_MARGIN_SECONDS = 300       # a run must be projected to end this long before the deadline
ESTIMATE_SAFETY = 1.25              # runs of one pair vary; the estimate is stretched by this before it is trusted
POLL_SECONDS = 0.5
MAX_LOSSES_PER_RUN = 3              # a run in flight when a worker died this often is stored as failed and left to a later job
SCAN_READ_RETRY_SECONDS = 30        # a pair's file that will not open is retried this long before the pair is skipped


def campaign_deadline(env=None):
    """Epoch seconds by which this job's runs must have ended (EXPERIMENT_DEADLINE), or None:
    no deadline, every missing run is dispatched."""
    value = (os.environ if env is None else env).get("EXPERIMENT_DEADLINE")
    return float(value) if value else None


def fits_before(deadline, estimate, now=None):
    """Whether a run expected to take `estimate` seconds may start now."""
    if deadline is None or estimate is None:
        return True
    now = time.time() if now is None else now
    return now + ESTIMATE_SAFETY * estimate + DEADLINE_MARGIN_SECONDS <= deadline


@dataclass
class Pair:
    """One (problem, algorithm) pair this job holds: its result file, its missing runs, its cost."""
    problem: object
    algo_name: str
    factory: object
    section: str
    path: str
    identity: dict
    max_evaluations: int
    missing: list                          # run ids still to start, in id order
    claim: object
    in_flight: int = 0
    times: list = field(default_factory=list)      # wall times of its stored and finished runs

    @property
    def problem_name(self):
        return self.problem.name()

    @property
    def label(self):
        return f"{self.algo_name} on {self.problem_name} (D={self.problem.number_of_variables()})"


class _Estimates:
    """Run-time estimates: the pair's own stored runs first, else this job's runs of the algorithm."""

    def __init__(self):
        self._by_algorithm = {}

    def record(self, pair, seconds):
        pair.times.append(seconds)
        self._by_algorithm.setdefault(pair.algo_name, []).append(seconds)

    def for_pair(self, pair):
        times = pair.times or self._by_algorithm.get(pair.algo_name)
        return sum(times) / len(times) if times else None


class _Summary:
    def __init__(self):
        self.runs_done = self.runs_failed = self.pairs_finished = 0
        self.pairs_held_by_others = self.pairs_mismatched = self.runs_left_for_deadline = self.pairs_busy = 0
        self.pools_broken = self.runs_lost = 0

    def __str__(self):
        return (f"{self.runs_done} run(s) stored ({self.runs_failed} failed, to be retried), "
                f"{self.pairs_finished} pair(s) completed, {self.runs_left_for_deadline} run(s) left for the "
                f"deadline, {self.pairs_held_by_others} pair(s) held by other jobs, "
                f"{self.pairs_busy} pair(s) whose file was being written and were left for the next job, "
                f"{self.pairs_mismatched} pair(s) skipped for a configuration mismatch, "
                f"{self.pools_broken} dead-worker incident(s) ({self.runs_lost} run(s) re-queued)")


def _claimable_pairs(campaign, claims_dir, lock_path, summary):
    """The pairs of the campaign with missing runs that this job manages to claim, in campaign
    order, each yielded with its claim held. Complete pairs and pairs another job holds are
    skipped; a pair whose stored file belongs to another configuration is reported and skipped,
    never touched.

    The file of a pair another job holds is never opened: the claim is checked first, so a
    reader does not collide with that job's writes (HDF5 makes a file being appended to
    unreadable for the moment). The residual race - a job claiming the pair between this check
    and the read - is covered by the reader's retries, and a file that stays unreadable while
    freshly modified is reported busy and left for the next job rather than set aside."""
    for problem in campaign.problems:
        section = campaign.section_of(problem)
        n_vars = problem.number_of_variables()
        max_evaluations = problem_budget(problem)
        problem_dir = results_subdir(campaign.results_dir, section, n_vars)
        make_dir(problem_dir)
        for algo_name, factory in campaign.algorithms.items():
            path = h5_result_path(problem_dir, problem.name(), algo_name, n_vars)
            key = os.path.splitext(os.path.basename(path))[0]
            if is_held(claims_dir, key):                    # another job is on it: do not even open its file
                summary.pairs_held_by_others += 1
                continue
            try:
                identity = run_identity(problem, factory, max_evaluations, campaign.solutions_size, campaign.frequency)
                done = stored_runs(path, problem.name(), algo_name, identity,
                                   retry_seconds=SCAN_READ_RETRY_SECONDS, stale_after=CLAIM_STALE_SECONDS)
            except ResultIdentityMismatch as error:
                print(f"SKIPPING {algo_name} on {problem.name()}: {error}")
                summary.pairs_mismatched += 1
                continue
            except ResultFileBusy as error:
                print(f"SKIPPING {algo_name} on {problem.name()} for now: {error}")
                summary.pairs_busy += 1
                continue
            wanted = range(1, campaign.no_of_runs + 1)
            if all(run_id in done for run_id in wanted):
                continue
            claim = try_claim(claims_dir, key, lock_path)
            if claim is None:
                summary.pairs_held_by_others += 1
                continue
            try:                                            # another job may have finished it meanwhile
                done = stored_runs(path, problem.name(), algo_name, identity,
                                   retry_seconds=SCAN_READ_RETRY_SECONDS, stale_after=CLAIM_STALE_SECONDS)
            except ResultFileBusy as error:
                print(f"SKIPPING {algo_name} on {problem.name()} for now: {error}")
                summary.pairs_busy += 1
                claim.release()
                continue
            missing = [run_id for run_id in wanted if run_id not in done]
            if not missing:
                claim.release()
                continue
            pair = Pair(problem, algo_name, factory, section, path, identity, max_evaluations, missing, claim)
            pair.times = stored_run_times(path, problem.name(), algo_name)
            print(f"{pair.label}: {len(done)} of {campaign.no_of_runs} runs stored, {len(missing)} to run -> {path}")
            yield pair


def _next_run(held, candidates, deadline, estimates, summary):
    """The next (pair, run id) to start, or None when nothing that fits is left anywhere."""
    for pair in held:
        if pair.missing and fits_before(deadline, estimates.for_pair(pair)):
            return pair, pair.missing.pop(0)
    for pair in candidates:
        if fits_before(deadline, estimates.for_pair(pair)):
            held.append(pair)
            return pair, pair.missing.pop(0)
        summary.runs_left_for_deadline += len(pair.missing)
        estimate = estimates.for_pair(pair)
        print(f"{pair.label}: {len(pair.missing)} run(s)"
              + (f" of ~{estimate:.0f} s each" if estimate is not None else "")
              + " do not fit before the deadline; left for the next job.")
        pair.claim.release()
    return None


def _shutdown(executor):
    """Stop a pool without waiting for the runs in it: cancel what has not started and kill the
    worker processes (the runs they hold are either lost already or left for the next job)."""
    if executor is None:
        return
    processes = list(getattr(executor, "_processes", {}).values())      # no public way to kill the workers
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        try:
            process.kill()
        except Exception:
            pass


def _lost_run(pair, run_id, seed, campaign):
    """The record stored for a run whose worker died MAX_LOSSES_PER_RUN times: failed, so it is
    excluded from the aggregates and retried by a later job, as a run whose worker raised is."""
    return {'run_id': run_id, 'fitness_history': [float('inf')] * (pair.max_evaluations // campaign.frequency),
            'time': 0.0, 'final_fitness': float('inf'), 'seed': seed, 'total_restarts': None, 'failed': True,
            'finished_at': time.time(), **_where()}


def run_campaign_queue(campaign, num_parallel_workers=None, deadline=None, claims_dir=None):
    """Run the campaign's missing runs as one worker of the shared queue (see the module docstring).

    `deadline` is epoch seconds (default: EXPERIMENT_DEADLINE from the environment, else none);
    `claims_dir` defaults to `<results_dir>/claims`. Returns the job's summary.
    """
    workers = cpu_count() if not num_parallel_workers or num_parallel_workers <= 0 else min(num_parallel_workers, cpu_count())
    deadline = campaign_deadline() if deadline is None else deadline
    make_dir(campaign.results_dir)
    claims_dir = claims_dir or os.path.join(campaign.results_dir, "claims")
    run_seeds = generate_run_seeds(campaign.no_of_runs)
    summary, estimates = _Summary(), _Estimates()
    candidates = _claimable_pairs(campaign, claims_dir, lock_path=claims_dir, summary=summary)
    held, in_flight, losses = [], {}, {}
    print(f"Campaign queue: {len(campaign.algorithms)} algorithm(s) x {len(campaign.problems)} problem(s), "
          f"{workers} worker(s), deadline "
          + (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(deadline)) if deadline else "none"))

    def store(pair, run_id, run):
        write_run(pair.path, pair.problem_name, pair.algo_name, run, campaign.no_of_runs,
                  pair.max_evaluations, campaign.solutions_size, campaign.frequency,
                  problem=pair.problem, algorithm_factory=pair.factory,
                  configuration=pair.identity["configuration"], section=pair.section)
        summary.runs_done += 1
        if run["failed"]:
            summary.runs_failed += 1
            status = "FAILED (stored as incomplete, retried by a later job)"
        else:
            estimates.record(pair, run["time"])
            status = f"f = {run['final_fitness']:.6g}"
        print(f"  run {run_id:>3} of {pair.label}: {status}, {run['time']:.1f} s, stored", flush=True)

    def settle(pair):
        if not pair.missing and pair.in_flight == 0:
            pair.claim.release()
            held.remove(pair)
            summary.pairs_finished += 1
            print(f"{pair.label}: all runs this job could start are stored; pair released.", flush=True)

    executor = None
    last_beat = time.time()
    try:
        executor = ProcessPoolExecutor(max_workers=workers)
        while True:
            broken = False
            while len(in_flight) < workers:
                dispatch = _next_run(held, candidates, deadline, estimates, summary)
                if dispatch is None:
                    break
                pair, run_id = dispatch
                args = (pair.algo_name, copy.deepcopy(pair.problem), pair.factory, run_id,
                        run_seeds[run_id - 1], pair.max_evaluations, campaign.frequency)
                try:
                    future = executor.submit(run_single_instance, args)
                except BrokenProcessPool:             # a worker died since the last check: this run never started
                    pair.missing.insert(0, run_id)
                    broken = True
                    break
                in_flight[future] = (pair, run_id)
                pair.in_flight += 1
            if not in_flight and not broken:
                break
            finished = set() if broken else wait(list(in_flight), timeout=POLL_SECONDS, return_when=FIRST_COMPLETED)[0]
            for future in finished:
                try:
                    run = future.result()
                except BrokenProcessPool:
                    broken = True                     # handled below, together with every other run in the pool
                    continue
                except Exception as error:            # the worker catches its own errors; this is transport (pickling)
                    pair, run_id = in_flight.pop(future)
                    pair.in_flight -= 1
                    print(f"  run {run_id} of {pair.label}: result lost ({type(error).__name__}: {error})", flush=True)
                    store(pair, run_id, _lost_run(pair, run_id, run_seeds[run_id - 1], campaign))
                    settle(pair)
                    continue
                pair, run_id = in_flight.pop(future)
                pair.in_flight -= 1
                store(pair, run_id, run)
                settle(pair)
            if broken:
                # A worker died: every run still in this pool is gone (its future fails too, or never
                # completes). Put them back into their pairs and start a fresh pool.
                summary.pools_broken += 1
                lost = list(in_flight.values())
                in_flight.clear()
                _shutdown(executor)
                print(f"A worker process died (out of memory, or a crash in native code); {len(lost)} run(s) in "
                      f"flight are re-queued and a fresh pool is started.", flush=True)
                for pair, run_id in lost:
                    pair.in_flight -= 1
                    key = (pair.path, run_id)
                    losses[key] = losses.get(key, 0) + 1
                    if losses[key] < MAX_LOSSES_PER_RUN:
                        pair.missing.insert(0, run_id)
                        summary.runs_lost += 1
                        print(f"  run {run_id} of {pair.label}: lost ({losses[key]} of {MAX_LOSSES_PER_RUN}), re-queued",
                              flush=True)
                    else:
                        print(f"  run {run_id} of {pair.label}: lost {losses[key]} times; stored as failed, "
                              f"left to a later job", flush=True)
                        store(pair, run_id, _lost_run(pair, run_id, run_seeds[run_id - 1], campaign))
                for pair in {id(p): p for p, _ in lost}.values():
                    pair.missing.sort()
                    settle(pair)
                executor = ProcessPoolExecutor(max_workers=workers)
            if time.time() - last_beat >= CLAIM_HEARTBEAT_SECONDS:
                for pair in held:
                    pair.claim.beat()
                last_beat = time.time()
    finally:
        _shutdown(executor)
        for pair in held:                        # runs left behind (deadline, or a crash) go to the next job
            summary.runs_left_for_deadline += len(pair.missing) + pair.in_flight
            pair.claim.release()
    print(f"\n===== Job finished: {summary} =====")
    return summary
