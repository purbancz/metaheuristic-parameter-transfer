"""Follow-up M2a (the within-run check): the tuning-noise floor of the transfer comparison.

Read-only: Stage 2's first-elite runs (experiment_results/transfer, experiment_results/control/...)
and the second-elite runs (experiment_results/m2a/control/elite2_*/transfer); writes
tables/m2a_*.csv and m2a_numbers.json in this folder.

Arms: S1 = subset first elite (Stage 2 main), C1 = continuous first elite (Stage 2 control),
S2 / C2 = their second final elites. Comparisons and decision rule exactly as declared in 9q:
  1. within-race  S1 vs S2 (28 cells) and C1 vs C2 (20 cells), Holm over the 48
  2. cross-class with second elites  S2 vs C2 (20 cells), Holm over the 20
  3. reference  Stage 2's S1 vs C1 (Holm over Stage 2's 250 cells, from transfer_paired.csv)

Run from the repository root:  python papers/parameter_transfer/m2a_analysis.py
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

SETS = {"S1": "tuned_2026_10_subset_combined", "C1": "control_2026_10_continuous_bounce",
        "S2": "elite2_2026_11_subset_combined", "C2": "elite2_2026_11_continuous_bounce"}
M2A_DIR = "experiment_results/m2a"
FOCUS = list(json.load(open(f"src/metaheuristiczoo/experiment/parameters/{SETS['S2']}.json",
                            encoding="utf-8"))["algorithms"])
N = {}

# ---------------------------------------------------------------- runs of the four arms
stage2 = T.load_runs("experiment_results", section="transfer")
m2a = T.load_runs(M2A_DIR, section="transfer")
runs = pd.concat([stage2[stage2.arm != T.BASELINE], m2a], ignore_index=True)
runs["arm4"] = runs.parameter_set.map({v: k for k, v in SETS.items()})
runs = runs[runs.arm4.notna() & runs.algorithm.isin(FOCUS) & runs.instance.str.startswith("GR")]
counts = runs.groupby(["arm4", "algorithm", "instance"]).size()
assert (counts == 30).all(), counts[counts != 30]
N["cells_per_arm"] = runs.groupby("arm4").apply(lambda g: g.groupby(["algorithm", "instance"]).ngroups).to_dict()
N["failed_runs"] = int(runs.failed.sum())


# stored configurations of the second elites: every visible entry equals the set's value
def _stored_configuration_check():
    checked = mismatched = 0
    for set_name in (SETS["S2"], SETS["C2"]):
        values = json.load(open(f"src/metaheuristiczoo/experiment/parameters/{set_name}.json",
                                encoding="utf-8"))["algorithms"]
        for path in glob.glob(f"{M2A_DIR}/control/{set_name}/transfer/*.h5"):
            with h5py.File(path, "r") as h:
                problem = list(h)[0]
                for algorithm in h[problem]:
                    stored = json.loads(h[problem][algorithm].attrs["configuration_json"])
                    for key, value in values[algorithm]["params"].items():
                        if key in stored:
                            checked += 1
                            x = stored[key]
                            same = x == value or (isinstance(value, (int, float)) and isinstance(x, (int, float))
                                                  and abs(x - value) <= 1e-9 * max(1, abs(value)))
                            mismatched += 0 if same else 1
    return {"visible_entries": checked, "mismatches": mismatched}


N["stored_configuration_check"] = _stored_configuration_check()


def compare(first, second, holm_over_table=True):
    """`first` against `second` with transfer.paired (seed-paired Wilcoxon, A12 > 0.5: first better)."""
    two = runs[runs.arm4.isin([first, second])].copy()
    two["arm"] = np.where(two.arm4 == first, T.MAIN, T.CONTROL)
    table = T.paired(two)
    table.insert(0, "comparison", f"{first} vs {second}")
    return table


# ---------------------------------------------------------------- comparison 1: within-race, Holm over 48
within = pd.concat([compare("S1", "S2"), compare("C1", "C2")], ignore_index=True)
within["p_holm"] = T.holm(within.p_wilcoxon.tolist())
within["significant"] = within.p_holm.fillna(1.0) < T.ALPHA
within["magnitude_abs"] = (within.a12_main_vs_control - 0.5).abs()
assert len(within) == 48, len(within)

# ---------------------------------------------------------------- comparison 2: S2 vs C2, Holm over 20
cross2 = compare("S2", "C2")
assert len(cross2) == 20, len(cross2)

# ---------------------------------------------------------------- comparison 3: Stage 2 reference
ref = pd.read_csv("experiment_results/analysis/transfer/transfer_paired.csv")
ref = ref[ref.algorithm.isin(FOCUS) & ref.instance.str.startswith("GR")].copy()
ref["magnitude_abs"] = (ref.a12_main_vs_control - 0.5).abs()
assert len(ref) == 28, len(ref)
# the reference recomputed from the runs read here must agree with the stored Stage 2 table
check = compare("S1", "C1").merge(ref, on=["instance", "algorithm"], suffixes=("", "_stored"))
assert np.allclose(check.a12_main_vs_control, check.a12_main_vs_control_stored)

# ---------------------------------------------------------------- per-cell table and decision rule
cells = ref[["algorithm", "instance", "a12_main_vs_control", "magnitude_abs", "p_holm", "verdict"]].rename(
    columns={"a12_main_vs_control": "a12_S1_vs_C1", "magnitude_abs": "mag_S1_vs_C1",
             "p_holm": "p_holm_S1_vs_C1", "verdict": "verdict_S1_vs_C1"})
for comparison, tag in (("S1 vs S2", "S1_vs_S2"), ("C1 vs C2", "C1_vs_C2")):
    w = within[within.comparison == comparison][["algorithm", "instance", "a12_main_vs_control",
                                                 "magnitude_abs", "p_holm", "significant"]]
    cells = cells.merge(w.rename(columns={"a12_main_vs_control": f"a12_{tag}", "magnitude_abs": f"mag_{tag}",
                                          "p_holm": f"p_holm_{tag}", "significant": f"significant_{tag}"}),
                        on=["algorithm", "instance"], how="left")
cells = cells.merge(cross2[["algorithm", "instance", "a12_main_vs_control", "p_holm", "verdict"]].rename(
    columns={"a12_main_vs_control": "a12_S2_vs_C2", "p_holm": "p_holm_S2_vs_C2", "verdict": "verdict_S2_vs_C2"}),
    on=["algorithm", "instance"], how="left")

within_mags = cells[["mag_S1_vs_S2", "mag_C1_vs_C2"]]
cells["a_noise_below_cross"] = (within_mags.lt(cells.mag_S1_vs_C1, axis=0) | within_mags.isna()).all(axis=1)
cells["within_significant"] = cells[["significant_S1_vs_S2", "significant_C1_vs_C2"]].fillna(False).astype(bool).any(axis=1)
cells["b_same_side"] = np.where(cells.a12_S2_vs_C2.isna(), np.nan,
                                np.sign(cells.a12_S2_vs_C2 - 0.5) == np.sign(cells.a12_S1_vs_C1 - 0.5))
cells = cells.sort_values(["algorithm", "instance"])
cells.to_csv(os.path.join(TAB, "m2a_cells.csv"), index=False)

rows = []
for algorithm, g in cells.groupby("algorithm"):
    a_count = int(g.a_noise_below_cross.sum())
    has_b = g.b_same_side.notna().any()
    b_count = int(g.b_same_side.dropna().astype(bool).sum()) if has_b else None
    sig_count = int(g.within_significant.sum())
    a_ok, b_ok = a_count >= 3, (b_count >= 3) if has_b else True
    if a_ok and b_ok:
        outcome = "stands"
    elif not a_ok and sig_count >= 2:
        outcome = "not supported"
    elif not a_ok:
        outcome = "weakened"
    else:
        outcome = "(a) holds, (b) fails: not covered by the declared outcomes"
    rows.append({"algorithm": algorithm, "stage2_direction": g.verdict_S1_vs_C1.iloc[0],
                 "a_cells_noise_below_cross": a_count, "b_cells_same_side": b_count,
                 "within_significant_cells": sig_count, "outcome": outcome,
                 "median_mag_cross_stage2": float(g.mag_S1_vs_C1.median()),
                 "median_mag_within_subset": float(g.mag_S1_vs_S2.median()),
                 "median_mag_within_continuous": float(g.mag_C1_vs_C2.median()) if g.mag_C1_vs_C2.notna().any() else np.nan})
blocks = pd.DataFrame(rows)
blocks.to_csv(os.path.join(TAB, "m2a_blocks.csv"), index=False)
within.to_csv(os.path.join(TAB, "m2a_within_race.csv"), index=False)
cross2.to_csv(os.path.join(TAB, "m2a_cross_second_elites.csv"), index=False)

N["outcomes"] = blocks.outcome.value_counts().to_dict()
N["within_race_significant"] = {c: int(g.significant.sum()) for c, g in within.groupby("comparison")}
N["within_race_cells"] = {c: len(g) for c, g in within.groupby("comparison")}
N["cross_second_elites_verdicts"] = cross2.verdict.value_counts().to_dict()
N["median_magnitude"] = {"cross_stage2": float(ref.magnitude_abs.median()),
                         **{f"within_{c}": float(g.magnitude_abs.median()) for c, g in within.groupby("comparison")}}
with open(os.path.join(OUT, "m2a_numbers.json"), "w", encoding="utf-8", newline="\r\n") as fh:
    json.dump(N, fh, indent=1, default=str)
pd.set_option("display.width", 200)
print(blocks.to_string(index=False))
print(json.dumps(N, indent=1, default=str))
