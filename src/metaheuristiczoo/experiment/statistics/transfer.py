"""Analysis of the main comparison: configuration S against configuration C on the ten instances.

    from metaheuristiczoo.experiment.statistics.transfer import analyse
    report = analyse("experiment_results")
    report.summary, report.paired, report.baselines, report.ranks

Three arms on the same instances and budget:

    main      the 25 algorithms with S (subset-tuned)              <results>/transfer/
    baseline  Baseline_* entries of the same job                   <results>/transfer/
    control   the same algorithms with C (continuous-tuned, bounce) <results>/control/<set>/transfer/

The run-level metrics are `final_gap_absolute`, `final_gap_relative`, `hit_optimum`, `evals_to_optimum` (None and
`reached = False` when the run never got there - never imputed with the budget) and `evals_to_final`,
the stagnation diagnostic, with `evals_to_bks` and `beat_bks` against the best known. Everything is in
the MINIMISED sign the files store; equality is exact up to 1e-9 relative. The reference R of the gaps
is the certified optimum a file carries as `known_optimum` (all ten instances have one), else its
`best_known`, and every row says which. On an instance without a certified optimum `hit_optimum` and
`evals_to_optimum` would be undefined (None), not False.

Times are read off the stored best-so-far curve, which is sampled every `interval` evaluations (100,
the swarm size), so `evals_to_optimum` and `evals_to_final` are the first SAMPLE at which the target
holds: an upper bound within one interval of the exact evaluation the spec defines. The tables say so
in a column rather than pretend to the finer resolution.

Comparisons rank across instances and use A12 within one, because the
objectives are counts on scales from 76 to 2,736 and must never be averaged directly:

* `paired`   main against control, one (instance, algorithm) cell at a time, paired by run id. Both
             arms use run seed BASE_SEED + id - 1, and a pair whose seeds differ is refused rather than
             compared. Wilcoxon signed-rank on the final values, wins/ties/losses, A12 (> 0.5: main
             better), and a Holm correction across every cell tested.
* `baselines` every algorithm against the best baseline of each instance and against greedy, by A12
             on the final values.
* `ranks`    within each instance and arm, algorithms ranked by median final value (baselines
             included, ties averaged), then the mean rank across instances - complete instances only.
* `incomplete` the cells short of the full run count, so a partial campaign is visible as such.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from metaheuristiczoo.experiment.results.io import load_data
from metaheuristiczoo.experiment.statistics.effect_sizes import a12, magnitude

TOLERANCE = 1e-9
SECTION = "transfer"
SECTIONS = ("transfer",)
BASELINE_PREFIX = "Baseline_"
GREEDY = "Baseline_Greedy"
MAIN, CONTROL, BASELINE = "main", "control", "baseline"
ALPHA = 0.05
_LABEL = re.compile(r"^CoverageSubsetProblem\((?P<label>[^_]+_S[^_]+_p\d+)_cut=")


def instance_label(problem_key: str) -> str:
    """`SJC324/S1200/p1` from the stored problem key; the key itself when it is not a transfer instance."""
    match = _LABEL.match(problem_key)
    if match:
        return match.group("label").replace("_", "/")
    return problem_key


def _equal(x: float, y: float) -> bool:
    return abs(x - y) <= TOLERANCE * max(1.0, abs(y))


def _first_sample(curve: np.ndarray, target: float, interval: int) -> Optional[int]:
    """The evaluation count of the first sample at or below `target` (a minimised best-so-far)."""
    reached = np.flatnonzero(np.isfinite(curve) & (curve <= target + TOLERANCE * max(1.0, abs(target))))
    return None if reached.size == 0 else int((reached[0] + 1) * interval)


def _interval(path: str, columns: int, max_evaluations: Optional[int]) -> int:
    try:
        with h5py.File(path, "r") as handle:
            value = handle.attrs.get("evaluations_per_snapshot")
            if value is not None:
                return int(value)
    except OSError:
        pass
    return int(max_evaluations // columns) if max_evaluations and columns else 1


def run_rows(path: str, arm: str, parameter_set: Optional[str] = None) -> List[dict]:
    """One row per stored run of one result file, with the spec's metrics. A run whose worker raised is
    not in the stored aggregates at all (the writer keeps it out and the queue retries it), so it
    shows up as a missing run in `incomplete`; a non-finite final value that did reach the aggregates
    is kept with `failed = True` and no metrics, so it is counted, not hidden."""
    entries = load_data(path) or []
    rows = []
    for entry in entries:
        optimum, best = entry.get("known_optimum"), entry.get("best_known")
        reference = optimum if optimum is not None else best
        for algorithm, data in entry["results"].items():
            curves, finals = np.asarray(data["data"], dtype=float), np.asarray(data["final_fitness"], dtype=float)
            interval = _interval(path, curves.shape[1] if curves.ndim == 2 else 0, entry.get("max_evaluations"))
            run_ids = data.get("run_ids") or list(range(1, len(finals) + 1))
            this_arm = BASELINE if algorithm.startswith(BASELINE_PREFIX) else arm
            for row, (run_id, seed, final) in enumerate(zip(run_ids, data["seeds"], finals)):
                record = {"instance": instance_label(entry["problem"]), "problem": entry["problem"],
                          "n": entry["n_vars"], "algorithm": algorithm, "arm": this_arm,
                          "parameter_set": parameter_set if this_arm != BASELINE else "in code",
                          "run_id": int(run_id), "seed": int(seed), "final": float(final),
                          "optimum": None if optimum is None else float(optimum),
                          "best_known": None if best is None else float(best),
                          "reference_kind": ("optimum" if optimum is not None else
                                             "best_known" if best is not None else None),
                          "interval": interval, "failed": not np.isfinite(final)}
                if record["failed"]:
                    rows.append(record)
                    continue
                curve = curves[row]
                record["evals_to_final"] = _first_sample(curve, final, interval)   # needs no reference
                if reference is None:                   # an uncertified instance: no gaps
                    rows.append(record)
                    continue
                record.update({
                    "final_gap_absolute": float(final - reference),
                    "final_gap_relative": float((final - reference) / abs(reference)) if abs(reference) >= 1 else None,
                })
                if optimum is not None:
                    time_to_optimum = _first_sample(curve, optimum, interval)
                    record.update({"hit_optimum": _equal(final, optimum), "evals_to_optimum": time_to_optimum,
                                   "reached": time_to_optimum is not None})
                else:                                   # an open instance: nobody knows the optimum
                    record.update({"hit_optimum": None, "evals_to_optimum": None, "reached": None})
                if best is not None:
                    time_to_best = _first_sample(curve, best, interval)
                    record.update({"evals_to_bks": time_to_best, "reached_bks": time_to_best is not None,
                                   "beat_bks": bool(final < best and not _equal(final, best))})
                rows.append(record)
    return rows


def arm_files(results_dir: str, section: str = SECTION) -> Dict[str, List[str]]:
    """The result files of each arm under a campaign's results directory: main (+ baselines) in
    `<section>/`, every control set in `control/<set>/<section>/`."""
    if section not in SECTIONS:
        raise ValueError(f"section {section!r}: this analysis covers {SECTIONS}")
    files = {MAIN: sorted(glob.glob(os.path.join(results_dir, section, "*.h5")))}
    for control_dir in sorted(glob.glob(os.path.join(results_dir, "control", "*", section))):
        files[f"{CONTROL}:{os.path.basename(os.path.dirname(control_dir))}"] = \
            sorted(glob.glob(os.path.join(control_dir, "*.h5")))
    return files


def load_runs(results_dir: str, main_set: str = "tuned_2026_10_subset_combined",
              section: str = SECTION) -> pd.DataFrame:
    rows = []
    for arm, paths in arm_files(results_dir, section).items():
        name, _, control_set = arm.partition(":")
        for path in paths:
            found = run_rows(path, name, control_set or main_set)
            if name == CONTROL:                         # a control job never runs the baselines
                found = [row for row in found if row["arm"] != BASELINE]
            rows += found
    return pd.DataFrame(rows)


# -- tables ---------------------------------------------------------------------------------------

def summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Per instance, algorithm and arm: runs, failures, hit rate, gaps, and times (to optimum among the
    runs that reached it; to final over all)."""
    def describe(group):
        ok = group[~group["failed"]]
        reached = ok[ok["reached"] == True] if "reached" in ok else ok.iloc[0:0]  # noqa: E712 - None stays out
        certified = ok[ok["hit_optimum"].notna()] if "hit_optimum" in ok else ok.iloc[0:0]
        bks = ok[ok["reached_bks"] == True] if "reached_bks" in ok else ok.iloc[0:0]  # noqa: E712
        return pd.Series({
            "runs": len(group), "failed": int(group["failed"].sum()),
            "reference_kind": group["reference_kind"].iloc[0],
            "hit_rate": float(certified["hit_optimum"].astype(bool).mean()) if len(certified) else np.nan,
            "reached_bks": len(bks),
            "beat_bks": int((ok["beat_bks"] == True).sum()) if "beat_bks" in ok else 0,  # noqa: E712
            "median_gap_relative": float(ok["final_gap_relative"].median()) if len(ok) and "final_gap_relative" in ok else np.nan,
            "mean_gap_relative": float(ok["final_gap_relative"].mean()) if len(ok) and "final_gap_relative" in ok else np.nan,
            "median_final": float(ok["final"].median()) if len(ok) else np.nan,
            "optimum": group["optimum"].iloc[0],
            "best_known": group["best_known"].iloc[0],
            "reached": len(reached),
            "median_evals_to_optimum": float(reached["evals_to_optimum"].median()) if len(reached) else np.nan,
            "median_evals_to_bks": float(bks["evals_to_bks"].median()) if len(bks) else np.nan,
            "median_evals_to_final": float(ok["evals_to_final"].median()) if len(ok) and "evals_to_final" in ok else np.nan,
            "time_resolution": int(group["interval"].iloc[0]),
        })
    columns = [c for c in runs.columns if c not in ("instance", "algorithm", "arm")]
    # columns selected before apply rather than include_groups=False, which needs pandas >= 2.2
    return runs.groupby(["instance", "algorithm", "arm"], sort=True)[columns].apply(describe).reset_index()


def holm(p_values: List[float]) -> List[float]:
    """Holm-adjusted p-values, monotone, in the input order; NaN stays NaN and is not counted."""
    values = np.asarray(p_values, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    adjusted = np.full(values.shape, np.nan)
    order = finite[np.argsort(values[finite])]
    running, m = 0.0, len(order)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def paired(runs: pd.DataFrame, control_arm: str = CONTROL) -> pd.DataFrame:
    """Main against control per (instance, algorithm), paired by run id."""
    main = runs[(runs["arm"] == MAIN) & ~runs["failed"]]
    control = runs[(runs["arm"] == control_arm) & ~runs["failed"]]
    rows = []
    for (instance, algorithm), a in main.groupby(["instance", "algorithm"]):
        b = control[(control["instance"] == instance) & (control["algorithm"] == algorithm)]
        if b.empty:
            continue
        joined = a.merge(b, on="run_id", suffixes=("_main", "_control"))
        if joined.empty:                                  # both arms have runs, but no run id in common yet
            continue
        if (joined["seed_main"] != joined["seed_control"]).any():
            raise ValueError(f"{instance} / {algorithm}: runs with the same id have different seeds; the arms "
                             f"are not paired and must not be compared as if they were")
        difference = joined["final_main"] - joined["final_control"]          # < 0: main better
        wins, ties, losses = int((difference < 0).sum()), int((difference == 0).sum()), int((difference > 0).sum())
        if len(joined) < 2 or ties == len(joined):
            p = 1.0 if ties == len(joined) else np.nan
        else:
            p = float(wilcoxon(joined["final_main"], joined["final_control"], zero_method="zsplit").pvalue)
        effect = a12(joined["final_main"], joined["final_control"])
        rows.append({"instance": instance, "algorithm": algorithm, "pairs": len(joined),
                     "main_wins": wins, "ties": ties, "control_wins": losses,
                     "median_final_main": float(joined["final_main"].median()),
                     "median_final_control": float(joined["final_control"].median()),
                     "hit_rate_main": float(joined["hit_optimum_main"].mean()),
                     "hit_rate_control": float(joined["hit_optimum_control"].mean()),
                     "a12_main_vs_control": effect, "magnitude": magnitude(effect), "p_wilcoxon": p})
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_holm"] = holm(table["p_wilcoxon"].tolist())
        table["verdict"] = [
            "main better" if (q < ALPHA and e > 0.5) else "control better" if (q < ALPHA and e < 0.5) else "no difference"
            for q, e in zip(table["p_holm"].fillna(1.0), table["a12_main_vs_control"])]
    return table


def baselines(runs: pd.DataFrame, arm: str = MAIN) -> pd.DataFrame:
    """Each algorithm of `arm` against greedy and against the best baseline of the instance (the one
    with the lowest median final value), by A12 on the final values (> 0.5: the algorithm better)."""
    ok = runs[~runs["failed"]]
    rows = []
    for instance, group in ok.groupby("instance"):
        base = group[group["arm"] == BASELINE]
        if base.empty:
            continue
        medians = base.groupby("algorithm")["final"].median()
        best = str(medians.idxmin())
        best_values = base[base["algorithm"] == best]["final"]
        greedy = base[base["algorithm"] == GREEDY]["final"]
        for algorithm, values in group[group["arm"] == arm].groupby("algorithm")["final"]:
            vs_best = a12(values, best_values)
            rows.append({"instance": instance, "algorithm": algorithm, "best_baseline": best,
                         "median_final": float(values.median()), "median_final_best_baseline": float(medians[best]),
                         "a12_vs_best_baseline": vs_best, "magnitude_vs_best_baseline": magnitude(vs_best),
                         "a12_vs_greedy": a12(values, greedy) if len(greedy) else np.nan})
    return pd.DataFrame(rows)


def incomplete(runs: pd.DataFrame) -> pd.DataFrame:
    """Every (instance, algorithm, arm) cell short of the full run count, including cells with no runs
    at all. The full count is the largest any cell has, and the expected cells are every algorithm an
    arm stores anywhere, on every instance."""
    counts = runs.groupby(["instance", "algorithm", "arm"]).size()
    expected = counts.max()
    instances = sorted(runs["instance"].unique())
    cells = [(i, a, arm) for arm, group in runs.groupby("arm") for a in sorted(group["algorithm"].unique())
             for i in instances]
    table = pd.DataFrame(cells, columns=["instance", "algorithm", "arm"])
    table["runs"] = [int(counts.get(cell, 0)) for cell in cells]
    table["expected"] = int(expected)
    return table[table["runs"] < expected].reset_index(drop=True)


def ranks(runs: pd.DataFrame, arm: str = MAIN) -> pd.DataFrame:
    """Mean rank across instances of the algorithms of `arm` together with the baselines, each instance
    ranking by median final value (1 = best, ties averaged). Only instances on which every one of these
    algorithms has all its runs are ranked: a rank among the algorithms that happen to have finished is
    not comparable with a rank among all of them."""
    ok = runs[~runs["failed"] & runs["arm"].isin([arm, BASELINE])]
    short = incomplete(runs[runs["arm"].isin([arm, BASELINE])])
    ok = ok[~ok["instance"].isin(short["instance"])]
    medians = ok.groupby(["instance", "algorithm"])["final"].median().reset_index()
    medians["rank"] = medians.groupby("instance")["final"].rank(method="average")
    table = (medians.groupby("algorithm")["rank"].agg(["mean", "min", "max", "count"])
             .rename(columns={"mean": "mean_rank", "min": "best_rank", "max": "worst_rank", "count": "instances"})
             .sort_values("mean_rank").reset_index())
    table["arm"] = [BASELINE if a.startswith(BASELINE_PREFIX) else arm for a in table["algorithm"]]
    return table


@dataclass
class TransferReport:
    runs: pd.DataFrame
    summary: pd.DataFrame
    paired: pd.DataFrame
    baselines: pd.DataFrame
    ranks: pd.DataFrame
    incomplete: pd.DataFrame

    def write(self, out_dir: str) -> List[str]:
        os.makedirs(out_dir, exist_ok=True)
        written = []
        for name in ("runs", "summary", "paired", "baselines", "ranks", "incomplete"):
            path = os.path.join(out_dir, f"transfer_{name}.csv")
            getattr(self, name).to_csv(path, index=False)
            written.append(path)
        return written


def analyse(results_dir: str = "experiment_results", main_set: str = "tuned_2026_10_subset_combined",
            section: str = SECTION) -> TransferReport:
    runs = load_runs(results_dir, main_set, section)
    if runs.empty:
        raise FileNotFoundError(f"no {section} results under {results_dir}/{section}/ or "
                                f"{results_dir}/control/*/{section}/")
    control_sets = sorted(runs.loc[runs["arm"] == CONTROL, "parameter_set"].dropna().unique())
    if len(control_sets) > 1:
        raise ValueError(f"more than one control set under {results_dir}/control/: {control_sets}; the paired "
                         f"comparison needs exactly one")
    return TransferReport(runs=runs, summary=summary(runs), paired=paired(runs),
                          baselines=baselines(runs), ranks=ranks(runs), incomplete=incomplete(runs))
