"""Follow-up M1a (the parameter-group swap): which parameter group carries the transfer effect.

Read-only: Stage 2's runs (experiment_results/transfer, experiment_results/control/...) and the swap runs
(experiment_results/m1a/control/swap_2026_11_*/transfer); writes tables/m1a_*.csv and m1a_numbers.json
in this folder.

Arms: S = subset first elite (Stage 2 main), C = continuous first elite (Stage 2 control), and per group g
S<-C[g] (S with C's group g) and C<-S[g]. As declared in 9r:
  E     = A12(S, C) - 0.5                       (> 0: subset better)
  shift = A12(S, S<-C[g]) - 0.5  or  A12(C<-S[g], C) - 0.5   (> 0: toward the donor)
  share = shift / E
  tests: each variant against its receiver, seed-paired Wilcoxon, Holm over all 168 comparisons
  carries     : in BOTH directions >= 3 of 4 cells significant with share > 0 and median share >= 0.5
  contributes : otherwise, in at least one direction >= 2 of 4 cells significant with share > 0
  no detectable role : otherwise

Run from the repository root:  python papers/parameter_transfer/m1a_analysis.py
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

GROUPS = ("base", "coefficients", "shares")
DIRECTIONS = {"subset_takes_continuous": ("S", "C"), "continuous_takes_subset": ("C", "S")}   # receiver, donor
SHORT = {"subset_takes_continuous": "S<-C", "continuous_takes_subset": "C<-S"}
SETS = {"S": "tuned_2026_10_subset_combined", "C": "control_2026_10_continuous_bounce"}
for d in DIRECTIONS:
    for g in GROUPS:
        SETS[f"{d}:{g}"] = f"swap_2026_11_{d}_{g}"
M1A_DIR = "experiment_results/m1a"
PDIR = "src/metaheuristiczoo/experiment/parameters"
FOCUS = list(json.load(open(f"{PDIR}/{SETS['subset_takes_continuous:base']}.json", encoding="utf-8"))["algorithms"])
N = {}

# ---------------------------------------------------------------- runs of the eight arms
stage2 = T.load_runs("experiment_results", section="transfer")
m1a = T.load_runs(M1A_DIR, section="transfer")
runs = pd.concat([stage2[stage2.arm != T.BASELINE], m1a], ignore_index=True)
runs["arm8"] = runs.parameter_set.map({v: k for k, v in SETS.items()})
runs = runs[runs.arm8.notna() & runs.algorithm.isin(FOCUS) & runs.instance.str.startswith("GR")]
counts = runs.groupby(["arm8", "algorithm", "instance"]).size()
assert len(counts) == 8 * 28 and (counts == 30).all(), counts[counts != 30]
N["runs_read"] = int(len(runs))
N["failed_runs"] = int(runs.failed.sum())


# stored configurations of the swap arms: every visible entry equals the set's value
def _stored_configuration_check():
    checked = mismatched = 0
    for key, set_name in SETS.items():
        if ":" not in key:
            continue
        values = json.load(open(f"{PDIR}/{set_name}.json", encoding="utf-8"))["algorithms"]
        for path in glob.glob(f"{M1A_DIR}/control/{set_name}/transfer/*.h5"):
            with h5py.File(path, "r") as h:
                problem = list(h)[0]
                for algorithm in h[problem]:
                    stored = json.loads(h[problem][algorithm].attrs["configuration_json"])
                    for k, v in values[algorithm]["params"].items():
                        if k in stored:
                            checked += 1
                            x = stored[k]
                            same = x == v or (isinstance(v, (int, float)) and isinstance(x, (int, float))
                                              and abs(x - v) <= 1e-9 * max(1, abs(v)))
                            mismatched += 0 if same else 1
    return {"visible_entries": checked, "mismatches": mismatched}


N["stored_configuration_check"] = _stored_configuration_check()


def compare(first, second):
    """`first` against `second` with transfer.paired (seed-paired Wilcoxon, A12 > 0.5: first better)."""
    two = runs[runs.arm8.isin([first, second])].copy()
    two["arm"] = np.where(two.arm8 == first, T.MAIN, T.CONTROL)
    return T.paired(two)


# ---------------------------------------------------------------- Stage 2 effect E
ref = pd.read_csv("experiment_results/analysis/transfer/transfer_paired.csv")
ref = ref[ref.algorithm.isin(FOCUS) & ref.instance.str.startswith("GR")][["algorithm", "instance", "a12_main_vs_control"]]
ref = ref.rename(columns={"a12_main_vs_control": "a12_S_vs_C"})
ref["E"] = ref.a12_S_vs_C - 0.5
assert len(ref) == 28
recomputed = compare("S", "C").merge(ref, on=["instance", "algorithm"])
assert np.allclose(recomputed.a12_main_vs_control, recomputed.a12_S_vs_C)

# ---------------------------------------------------------------- 168 comparisons, Holm over all
parts = []
for direction, (receiver, _) in DIRECTIONS.items():
    for group in GROUPS:
        variant = f"{direction}:{group}"
        # orient every comparison so that A12 > 0.5 means "moved toward the donor"
        table = compare(receiver, variant) if receiver == "S" else compare(variant, receiver)
        table.insert(0, "group", group)
        table.insert(0, "direction", direction)
        parts.append(table)
cells = pd.concat(parts, ignore_index=True)
assert len(cells) == 168, len(cells)
cells["p_holm"] = T.holm(cells.p_wilcoxon.tolist())
cells["significant"] = cells.p_holm.fillna(1.0) < T.ALPHA
cells = cells.merge(ref, on=["algorithm", "instance"])
cells["shift"] = cells.a12_main_vs_control - 0.5
cells["share"] = cells["shift"] / cells.E
cells["toward"] = cells.significant & (cells.share > 0)
cells["away"] = cells.significant & (cells.share < 0)
cells = cells.drop(columns=["verdict"]).sort_values(["algorithm", "group", "direction", "instance"])
cells.to_csv(os.path.join(TAB, "m1a_cells.csv"), index=False)

# ---------------------------------------------------------------- decision rule per block and group
rows = []
for (algorithm, group), g in cells.groupby(["algorithm", "group"]):
    per = {}
    for direction, d in g.groupby("direction"):
        per[direction] = {"toward": int(d.toward.sum()), "away": int(d.away.sum()),
                          "median_share": float(d.share.median())}
    carries = all(p["toward"] >= 3 and p["median_share"] >= 0.5 for p in per.values()) and len(per) == 2
    contributes = any(p["toward"] >= 2 for p in per.values())
    rows.append({"algorithm": algorithm, "group": group,
                 "outcome": "carries" if carries else "contributes" if contributes else "no detectable role",
                 **{f"{k}_{SHORT[d]}": v for d, p in per.items() for k, v in p.items()}})
blocks = pd.DataFrame(rows)
blocks.to_csv(os.path.join(TAB, "m1a_blocks.csv"), index=False)

# descriptive: sum of the three groups' median shares, per block and direction
additivity = (cells.groupby(["algorithm", "direction", "group"]).share.median()
              .groupby(["algorithm", "direction"]).sum().unstack("direction"))
additivity.to_csv(os.path.join(TAB, "m1a_share_sums.csv"))

N["outcomes"] = blocks.outcome.value_counts().to_dict()
N["outcomes_by_group"] = {g: d.outcome.value_counts().to_dict() for g, d in blocks.groupby("group")}
N["significant_toward"] = int(cells.toward.sum())
N["significant_away"] = int(cells.away.sum())
N["away_cells"] = cells[cells.away][["algorithm", "group", "direction", "instance", "share"]].to_dict("records")
with open(os.path.join(OUT, "m1a_numbers.json"), "w", encoding="utf-8", newline="\r\n") as fh:
    json.dump(N, fh, indent=1, default=str)
pd.set_option("display.width", 250)
print(blocks.to_string(index=False))
print(additivity.round(2).to_string())
print(json.dumps(N, indent=1, default=str))
