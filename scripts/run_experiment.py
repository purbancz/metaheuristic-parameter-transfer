"""One worker job of the experiment campaign: claims (problem, algorithm) pairs from the shared
queue and runs their missing runs until the queue is empty or the job's deadline is near
(experiment.running.queue). Several jobs can share one results folder: they coordinate through the
claim files under <results_dir>/claims and write disjoint result files. A stopped job can be restarted;
finished runs are kept.

Job scope, from the environment (all optional; without any, the main arm of the whole study):
    EXPERIMENT_ALGORITHMS  algorithm names or globs, e.g. PSO,Hybrid*  (experiment.roster)
    EXPERIMENT_PROBLEMS    problem names or globs, e.g. '*SJC818/*'
    EXPERIMENT_CONTROL_SET run another parameter set (C or a follow-up set) instead of S, without the
                           baselines; results under experiment_results/control/<set>/transfer
    EXPERIMENT_WORKERS     processes in the pool (default: SLURM_CPUS_PER_TASK, else every core)
    EXPERIMENT_DEADLINE    epoch seconds when the allocation ends (the job stops starting runs in time)
    MZOO_DATA_ROOT         the folder with the instance networks (scripts/fetch_subset_data.py)
"""
import os
from multiprocessing import cpu_count

from metaheuristiczoo.experiment.campaign import build_campaign
from metaheuristiczoo.experiment.running import run_campaign_queue


def _workers():
    for variable in ("EXPERIMENT_WORKERS", "SLURM_CPUS_PER_TASK"):
        value = os.environ.get(variable)
        if value and int(value) > 0:
            return int(value)
    return cpu_count()


if __name__ == "__main__":
    campaign = build_campaign(
        sections=os.environ.get("EXPERIMENT_SECTION") or None,
        algorithms=os.environ.get("EXPERIMENT_ALGORITHMS") or None,
        problems=os.environ.get("EXPERIMENT_PROBLEMS") or None,
    )
    run_campaign_queue(campaign, num_parallel_workers=_workers())
