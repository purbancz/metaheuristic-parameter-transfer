"""Report of the main comparison: S against C on the ten instances, with the baselines.

    python scripts/analyze_transfer.py                                   # experiment_results/
    python scripts/analyze_transfer.py --results-dir /path/experiment_results --out analysis/transfer

Reads <results>/<section>/ (main arm and baselines) and <results>/control/<set>/<section>/ (control
arm, when one was run),
writes six CSVs to --out (runs, summary, paired, baselines, ranks, incomplete; see
metaheuristiczoo.experiment.statistics.transfer for what each holds) and prints the headline: main
against control cell by cell, every algorithm against the baselines, and the mean ranks. Rerun it as
results arrive - it reads whatever is stored and says how many runs each cell has.
"""

import argparse
import os

import pandas as pd

from metaheuristiczoo.experiment.statistics.transfer import BASELINE, CONTROL, MAIN, SECTIONS, analyse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="experiment_results")
    parser.add_argument("--section", default="transfer", choices=SECTIONS,
                        help="the suite section (the study has one, transfer)")
    parser.add_argument("--out", default=None, help="default: <results-dir>/analysis/<section>")
    args = parser.parse_args()

    report = analyse(args.results_dir, section=args.section)
    args.out = args.out or os.path.join(args.results_dir, "analysis", args.section)
    written = report.write(args.out)
    runs = report.runs
    pd.set_option("display.width", 160, "display.max_columns", 20, "display.max_rows", 200)

    print("== coverage ==")
    coverage = runs.groupby("arm").agg(runs=("run_id", "size"), failed=("failed", "sum"),
                                       algorithms=("algorithm", "nunique"), instances=("instance", "nunique"))
    print(coverage.to_string())
    resolution = int(runs["interval"].iloc[0])
    print(f"(times to optimum and to final are resolved to the stored sampling interval, {resolution} evaluations)")
    open_cells = report.summary[report.summary["reference_kind"] == "best_known"]
    if not open_cells.empty:
        print("\n== open instances: gaps are to the best known; runs that matched or beat it ==")
        print(open_cells[["instance", "algorithm", "arm", "best_known", "median_final", "reached_bks", "beat_bks"]]
              .sort_values(["instance", "median_final"]).to_string(index=False))
    if not report.incomplete.empty:
        short = report.incomplete
        print(f"\nINCOMPLETE: {len(short)} cells short of {short['expected'].iloc[0]} runs "
              f"(transfer_incomplete.csv lists them):")
        print(short.groupby(["instance", "arm"]).agg(cells=("algorithm", "size"), runs_stored=("runs", "sum"))
              .to_string())

    if not report.paired.empty:
        print("\n== main against control, per algorithm over instances (Holm-corrected, alpha 0.05) ==")
        verdicts = (report.paired.groupby(["algorithm", "verdict"]).size().unstack(fill_value=0)
                    .reindex(columns=["main better", "no difference", "control better"], fill_value=0))
        print(verdicts.sort_values(["main better", "control better"], ascending=[False, True]).to_string())
    else:
        print(f"\n(no {CONTROL} arm stored yet - the paired comparison is empty)")

    if not report.baselines.empty:
        print("\n== against the baselines: instances on which the algorithm's A12 exceeds 0.5 ==")
        table = report.baselines.assign(beats_best=report.baselines["a12_vs_best_baseline"] > 0.5,
                                        beats_greedy=report.baselines["a12_vs_greedy"] > 0.5)
        print(table.groupby("algorithm")[["beats_best", "beats_greedy"]].sum()
              .sort_values("beats_best", ascending=False).to_string())

    ranked = int(report.ranks["instances"].max()) if not report.ranks.empty else 0
    print(f"\n== mean rank across the {ranked} complete instances, {MAIN} arm with the {BASELINE}s (1 = best) ==")
    print(report.ranks.to_string(index=False))
    print("\nwritten:", *written, sep="\n  ")


if __name__ == "__main__":
    main()
