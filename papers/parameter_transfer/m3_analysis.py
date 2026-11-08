"""Follow-up M3 (the population-size swap): does the population parameter carry the DE family's gain on SJC818?

Read-only: Stage 2's runs (experiment_results/transfer, experiment_results/control/...) and the swap runs
(experiment_results/m3/control/popswap_2026_11_*/transfer); writes tables/m3_*.csv and m3_numbers.json
in this folder.

Arms: S = subset first elite (Stage 2 main), C = continuous first elite (Stage 2 control), S<-C (S with C's
population parameter) and C<-S (C with S's). Algorithms SHADE, LSHADE, LSRTDE, JSO; instances
SJC818/S800/p5 and SJC818/S1200/p3. As declared in 9s:
  E     = A12(S, C) - 0.5                                   (> 0: subset better)
  shift = A12(S, S<-C) - 0.5  or  A12(C<-S, C) - 0.5        (> 0: toward the donor)
  share = shift / E
  tests: each variant against its receiver, seed-paired Wilcoxon, Holm over all 16 comparisons
  the rule applies to an algorithm whose Stage 2 verdict is significant in both SJC818 cells:
  carries     : in BOTH directions 2 of 2 cells significant with share > 0 and median share >= 0.5
  contributes : otherwise, in at least one direction >= 1 of 2 cells significant with share > 0
  no detectable role : otherwise
  an algorithm without that Stage 2 effect (JSO: A12 0.50 in both cells) gets "no Stage 2 effect";
  its tests are reported as description only.

Run from the repository root:  python papers/parameter_transfer/m3_analysis.py
"""
import glob
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "papers", "parameter_transfer")
TAB = os.path.join(OUT, "tables")
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from metaheuristiczoo.experiment.statistics import transfer as T  # noqa: E402

DIRECTIONS = {"subset_takes_continuous": ("S", "C"), "continuous_takes_subset": ("C", "S")}   # receiver, donor
SHORT = {"subset_takes_continuous": "S<-C", "continuous_takes_subset": "C<-S"}
SETS = {"S": "tuned_2026_10_subset_combined", "C": "control_2026_10_continuous_bounce"}
for d in DIRECTIONS:
    SETS[d] = f"popswap_2026_11_{d}"
M3_DIR = "experiment_results/m3"
PDIR = "src/metaheuristiczoo/experiment/parameters"
ALGORITHMS = ["SHADE", "LSHADE", "LSRTDE", "JSO"]
INSTANCES = ["SJC818/S800/p5", "SJC818/S1200/p3"]
N = {}

# ---------------------------------------------------------------- runs of the four arms
stage2 = T.load_runs("experiment_results", section="transfer")
m3 = T.load_runs(M3_DIR, section="transfer")
runs = pd.concat([stage2[stage2.arm != T.BASELINE], m3], ignore_index=True)
runs["arm4"] = runs.parameter_set.map({v: k for k, v in SETS.items()})
runs = runs[runs.arm4.notna() & runs.algorithm.isin(ALGORITHMS) & runs.instance.isin(INSTANCES)]
counts = runs.groupby(["arm4", "algorithm", "instance"]).size()
assert len(counts) == 4 * 8 and (counts == 30).all(), counts[counts != 30]
N["runs_read"] = int(len(runs))
N["failed_runs"] = int(runs.failed.sum())


# stored configurations of the swap arms: every visible entry equals the set's value, and the initial
# population the algorithm built equals the one the donor arm built in Stage 2 (same formula, same D)
def _stored_configuration_check():
    checked = mismatched = 0
    population = {}
    for key in DIRECTIONS:
        set_name = SETS[key]
        values = json.load(open(f"{PDIR}/{set_name}.json", encoding="utf-8"))["algorithms"]
        for path in glob.glob(f"{M3_DIR}/control/{set_name}/transfer/*.h5"):
            with h5py.File(path, "r") as h:
                problem = list(h)[0]
                for algorithm in h[problem]:
                    stored = json.loads(h[problem][algorithm].attrs["configuration_json"])
                    population[(key, problem, algorithm)] = stored.get("initial_population_size")
                    for k, v in values[algorithm]["params"].items():
                        if k in stored:
                            checked += 1
                            x = stored[k]
                            same = x == v or (isinstance(v, (int, float)) and isinstance(x, (int, float))
                                              and abs(x - v) <= 1e-9 * max(1, abs(v)))
                            mismatched += 0 if same else 1
    donor_population = {}
    for key, (_, donor) in DIRECTIONS.items():
        base = "experiment_results" + ("" if donor == "S" else f"/control/{SETS[donor]}") + "/transfer"
        for path in glob.glob(f"{base}/*SJC818*.h5"):
            with h5py.File(path, "r") as h:
                problem = list(h)[0]
                for algorithm in h[problem]:
                    if algorithm in ALGORITHMS:
                        stored = json.loads(h[problem][algorithm].attrs["configuration_json"])
                        donor_population[(key, problem, algorithm)] = stored.get("initial_population_size")
    rows = [{"direction": SHORT[k[0]], "problem": k[1], "algorithm": k[2], "initial_population": v,
             "donor_stage2_initial_population": donor_population.get(k)} for k, v in sorted(population.items())]
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(TAB, "m3_populations.csv"), index=False)
    return {"visible_entries": checked, "mismatches": mismatched,
            "population_rows": len(table),
            "population_equals_donor": bool(len(table) == 16 and
                                            (table.initial_population == table.donor_stage2_initial_population).all())}


N["stored_configuration_check"] = _stored_configuration_check()


def compare(first, second):
    """`first` against `second` with transfer.paired (seed-paired Wilcoxon, A12 > 0.5: first better)."""
    two = runs[runs.arm4.isin([first, second])].copy()
    two["arm"] = np.where(two.arm4 == first, T.MAIN, T.CONTROL)
    return T.paired(two)


# ---------------------------------------------------------------- Stage 2 effect E
ref = pd.read_csv("experiment_results/analysis/transfer/transfer_paired.csv")
ref = ref[ref.algorithm.isin(ALGORITHMS) & ref.instance.isin(INSTANCES)]
ref = ref[["algorithm", "instance", "a12_main_vs_control", "verdict"]].rename(
    columns={"a12_main_vs_control": "a12_S_vs_C", "verdict": "stage2_verdict"})
ref["E"] = ref.a12_S_vs_C - 0.5
assert len(ref) == 8
recomputed = compare("S", "C").merge(ref, on=["instance", "algorithm"])
assert np.allclose(recomputed.a12_main_vs_control, recomputed.a12_S_vs_C)
with_effect = {a for a, g in ref.groupby("algorithm") if (g.stage2_verdict != "no difference").all()}
N["algorithms_with_stage2_effect"] = sorted(with_effect)

# ---------------------------------------------------------------- 16 comparisons, Holm over all
parts = []
for direction, (receiver, _) in DIRECTIONS.items():
    # orient every comparison so that A12 > 0.5 means "moved toward the donor"
    table = compare(receiver, direction) if receiver == "S" else compare(direction, receiver)
    table.insert(0, "direction", direction)
    parts.append(table)
cells = pd.concat(parts, ignore_index=True)
assert len(cells) == 16, len(cells)
cells["p_holm"] = T.holm(cells.p_wilcoxon.tolist())
cells["significant"] = cells.p_holm.fillna(1.0) < T.ALPHA
cells = cells.merge(ref, on=["algorithm", "instance"])
cells["shift"] = cells.a12_main_vs_control - 0.5
ruled = cells.algorithm.isin(with_effect)
cells["share"] = np.where(ruled, cells["shift"] / cells.E.where(ruled, np.nan), np.nan)
cells["toward"] = ruled & cells.significant & (cells.share > 0)
cells["away"] = ruled & cells.significant & (cells.share < 0)
cells = cells.drop(columns=["verdict"]).sort_values(["algorithm", "direction", "instance"])
cells.to_csv(os.path.join(TAB, "m3_cells.csv"), index=False)

# ---------------------------------------------------------------- decision rule per algorithm
rows = []
for algorithm, g in cells.groupby("algorithm"):
    per = {}
    for direction, d in g.groupby("direction"):
        per[direction] = {"toward": int(d.toward.sum()), "away": int(d.away.sum()),
                          "significant": int(d.significant.sum()), "median_share": float(d.share.median())}
    if algorithm not in with_effect:
        outcome = "no Stage 2 effect"
    elif len(per) == 2 and all(p["toward"] == 2 and p["median_share"] >= 0.5 for p in per.values()):
        outcome = "carries"
    elif any(p["toward"] >= 1 for p in per.values()):
        outcome = "contributes"
    else:
        outcome = "no detectable role"
    rows.append({"algorithm": algorithm, "outcome": outcome,
                 **{f"{k}_{SHORT[d]}": v for d, p in per.items() for k, v in p.items()}})
blocks = pd.DataFrame(rows).set_index("algorithm").loc[ALGORITHMS].reset_index()
blocks.to_csv(os.path.join(TAB, "m3_blocks.csv"), index=False)

# descriptive: hit rate and median relative gap per arm (both SJC818 instances are at the baseline ceiling)
hits = (runs.groupby(["algorithm", "instance", "arm4"])
        .agg(hit_rate=("hit_optimum", "mean"), median_gap_pct=("final_gap_relative", lambda x: 100 * x.median()))
        .reset_index())
hits["arm4"] = hits.arm4.map({"S": "S", "C": "C", **SHORT})
hits.to_csv(os.path.join(TAB, "m3_hit_rates.csv"), index=False)

N["outcomes"] = blocks.outcome.value_counts().to_dict()
N["significant_toward"] = int(cells.toward.sum())
N["significant_away"] = int(cells.away.sum())
N["away_cells"] = cells[cells.away][["algorithm", "direction", "instance", "share"]].to_dict("records")
N["unruled_cells"] = cells[~cells.algorithm.isin(with_effect)][
    ["algorithm", "direction", "instance", "a12_main_vs_control", "p_holm"]].to_dict("records")
with open(os.path.join(OUT, "m3_numbers.json"), "w", encoding="utf-8", newline="\r\n") as fh:
    json.dump(N, fh, indent=1, default=str)
pd.set_option("display.width", 250)
print(blocks.to_string(index=False))
print(hits.pivot_table(index=["algorithm", "instance"], columns="arm4", values="hit_rate").round(2).to_string())
print(json.dumps(N, indent=1, default=str))
