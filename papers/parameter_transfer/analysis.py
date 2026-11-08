"""Parameter transfer across problem classes: tables, figure and quoted numbers.

Read-only with respect to the campaign: reads the Stage 2 analysis CSVs, the three parameter sets,
the irace parameter spaces (parameter_spaces.json, exported from the tuning code) and the HDF5 result
files (run_times, stored configurations); writes only under papers/parameter_transfer/{tables,figures}/ and
papers/parameter_transfer/numbers.json.

Run from the root of the code:  python papers/parameter_transfer/analysis.py
"""
import glob
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "papers", "parameter_transfer")
TAB, FIG = os.path.join(OUT, "tables"), os.path.join(OUT, "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from metaheuristiczoo.experiment.statistics import transfer as T  # noqa: E402


# ---------------------------------------------------------------- irace parameter spaces
def _parameter(record):
    """One parameter of a space as an object of its irace type (Real, Integer, Bool, Categorical) with its
    name and, where the type has them, lower, upper, log and values."""
    return type(record["type"], (), {k: v for k, v in record.items() if k != "type"})()


def _bounds(p):
    """(lower, upper) of a Real/Integer parameter; (None, None) for the other types."""
    return getattr(p, "lower", None), getattr(p, "upper", None)


def _is_log(p):
    """Whether irace samples the parameter on a log scale."""
    return bool(getattr(p, "log", None))


def _normalised(x, lo, hi, log):
    """Position of x in [lo, hi] on the scale irace samples it on: linear, or log for log-scale parameters."""
    return (np.log(x) - np.log(lo)) / (np.log(hi) - np.log(lo)) if log else (x - lo) / (hi - lo)


parameter_spaces = {a: {"params": [_parameter(r) for r in records]} for a, records in json.load(
    open(os.path.join(OUT, "parameter_spaces.json"), encoding="utf-8"))["algorithms"].items()}

PDIR = "src/metaheuristiczoo/experiment/parameters/"
SETS = {k: json.load(open(PDIR + f, encoding="utf-8")) for k, f in [
    ("subset", "tuned_2026_10_subset_combined.json"),
    ("control", "control_2026_10_continuous_bounce.json"),
    ("diverse_v2", "tuned_2026_10_diverse_v2.json")]}
A_DIR = "experiment_results/analysis/transfer/"
paired = pd.read_csv(A_DIR + "transfer_paired.csv")
runs = pd.read_csv(A_DIR + "transfer_runs.csv")
N = {}

# ---------------------------------------------------------------- 0. reproduction gate
rebuilt = T.paired(runs)
chk = paired.merge(rebuilt, on=["instance", "algorithm"], suffixes=("", "_r"))
assert len(chk) == len(paired) == 250
assert (chk.verdict == chk.verdict_r).all()
assert np.allclose(chk.a12_main_vs_control, chk.a12_main_vs_control_r)
N["gate_max_abs_diff_p_holm"] = float(np.nanmax(np.abs(chk.p_holm - chk.p_holm_r)))

# ---------------------------------------------------------------- 1. verdict counts
N["cells"] = len(paired)
N["verdicts"] = paired.verdict.value_counts().to_dict()
N["verdict_by_magnitude"] = {f"{v}|{m}": int(c) for (v, m), c in
                             paired.groupby(["verdict", "magnitude"]).size().items()}
N["a12_direction"] = {"main_gt_0.5": int((paired.a12_main_vs_control > 0.5).sum()),
                      "control_lt_0.5": int((paired.a12_main_vs_control < 0.5).sum()),
                      "exactly_0.5": int((paired.a12_main_vs_control == 0.5).sum())}
N["paired_runs"] = {"main_wins": int(paired.main_wins.sum()), "ties": int(paired.ties.sum()),
                    "control_wins": int(paired.control_wins.sum())}

# ceiling of Stage 2 = ILS and restarted 1-swap descent both hit the optimum in all 30 runs
bl = runs[runs.arm == "baseline"].groupby(["instance", "algorithm"]).hit_optimum.mean().unstack()
bl["ceiling"] = (bl["Baseline_ILS"] == 1.0) & (bl["Baseline_RestartedLS"] == 1.0)
bl.to_csv(os.path.join(TAB, "stage2_baseline_hit_rates.csv"))
CEIL = sorted(bl.index[bl.ceiling])
N["stage2_ceiling_instances"] = CEIL
paired["ceiling"] = paired.instance.isin(CEIL)
N["verdicts_by_ceiling"] = {("ceiling" if k[0] else "below_ceiling") + "|" + k[1]: int(v) for k, v in
                            paired.groupby(["ceiling", "verdict"]).size().items()}
per_inst = paired.groupby("instance").verdict.value_counts().unstack(fill_value=0)
per_inst["ceiling"] = per_inst.index.isin(CEIL)
per_inst.to_csv(os.path.join(TAB, "verdicts_by_instance.csv"))


def _count(v, label):
    return int((v == label).sum())


by_alg = paired.groupby("algorithm").agg(
    main_better=("verdict", lambda v: _count(v, "main better")),
    control_better=("verdict", lambda v: _count(v, "control better")),
    no_difference=("verdict", lambda v: _count(v, "no difference")),
    a12_gt_half=("a12_main_vs_control", lambda v: int((v > 0.5).sum())),
    a12_lt_half=("a12_main_vs_control", lambda v: int((v < 0.5).sum())),
    a12_median=("a12_main_vs_control", "median"),
    run_main_wins=("main_wins", "sum"), run_ties=("ties", "sum"), run_control_wins=("control_wins", "sum"))

# within-arm mean rank (by median final per instance, 25 algorithms, baselines excluded)
med = runs[runs.arm != "baseline"].groupby(["arm", "instance", "algorithm"]).final.median().reset_index()
med["rank"] = med.groupby(["arm", "instance"]).final.rank(method="average")
rk = med.groupby(["arm", "algorithm"])["rank"].mean().unstack(0)
by_alg["mean_rank_main"], by_alg["mean_rank_control"] = rk["main"], rk["control"]
by_alg["n_params"] = [len(parameter_spaces[a]["params"]) for a in by_alg.index]
by_alg = by_alg.sort_values(["a12_median", "main_better"], ascending=False)
by_alg.to_csv(os.path.join(TAB, "verdicts_by_algorithm.csv"))

# size classes x mechanism groups. POST HOC: the grouping was chosen after the verdict figure was
# drawn; it is reported as exploratory, not as a tested hypothesis.
n_of = runs.groupby("instance").n.first()
paired["n"] = paired.instance.map(n_of)
paired["size_class"] = np.select([paired.n <= 150, paired.n == 324], ["n<=150 (GR)", "n=324 (SJC324)"],
                                 "n>=818 (SJC818, PCB3038)")
GROUPS = {"Wanderer and _Var hybrids": [a for a in paired.algorithm.unique() if "Wanderer" in a or a.endswith("_Var")],
          "_WithRandom hybrids (all eight worst-aware and uninformed roles)":
              ["HybridAdditivePSO_WithRandom", "HybridPartialDisjointPSO_WithRandom", "HybridFullDisjointPSO_WithRandom"],
          "DE family": ["SHADE", "JADE", "JSO", "LSHADE", "LSRTDE", "IKUNDE"]}
paired["group"] = "other"
for _g, _m in GROUPS.items():
    paired.loc[paired.algorithm.isin(_m), "group"] = _g
size_tab = paired.groupby(["group", "size_class"]).verdict.value_counts().unstack(fill_value=0)
size_tab.to_csv(os.path.join(TAB, "verdicts_by_group_and_size_POSTHOC.csv"))
N["posthoc_group_members"] = GROUPS
N["posthoc_verdicts_by_group_and_size"] = {f"{g}|{s}": {k: int(v) for k, v in row.items()}
                                           for (g, s), row in size_tab.iterrows()}
N["verdicts_by_size"] = {s: {k: int(v) for k, v in row.items()} for s, row in
                         paired.groupby("size_class").verdict.value_counts().unstack(fill_value=0).iterrows()}

# per-cell median relative gap (per cent of the certified optimum) and hit rate, both arms; within an
# instance only - never averaged across instances
gap = runs[runs.arm != "baseline"].groupby(["instance", "algorithm", "arm"]).agg(
    median_gap_pct=("final_gap_relative", lambda v: 100 * v.median()), hit_rate=("hit_optimum", "mean")).unstack()
gap.columns = [f"{m}_{a}" for m, a in gap.columns]
gap.to_csv(os.path.join(TAB, "median_gap_by_cell.csv"))

# ---------------------------------------------------------------- 2. parameter sets, block by block
rows, boundary = [], []
for a in by_alg.index:
    sp = {p.name: p for p in parameter_spaces[a]["params"]}
    s = SETS["subset"]["algorithms"][a]["params"]
    c = SETS["control"]["algorithms"][a]["params"]
    v = SETS["diverse_v2"]["algorithms"][a]["params"]
    strip = lambda d: {k: x for k, x in d.items() if k != "constraint_handling_mode"}  # noqa: E731
    boundary.append({"algorithm": a,
                     "boundary_subset": s.get("constraint_handling_mode", "own handling"),
                     "boundary_control": c.get("constraint_handling_mode", "own handling"),
                     "control_equals_diverse_v2_values": strip(c) == strip(v),
                     "same_parameter_keys": set(s) == set(c) == set(sp) | ({"constraint_handling_mode"} & set(s))})
    for k in strip(s):
        p = sp[k]
        lo, hi = _bounds(p)
        log = lo is not None and _is_log(p)
        rows.append({"algorithm": a, "parameter": k, "type": type(p).__name__, "lower": lo, "upper": hi,
                     "scale": "log" if log else ("linear" if lo is not None else None),
                     "subset_tuned": s[k], "continuous_tuned": c[k],
                     # on irace's sampling scale; a value repaired below the floor stays linear (no log of it)
                     "normalised_difference": (np.nan if lo is None else
                                               _normalised(s[k], lo, hi, log) - _normalised(c[k], lo, hi, log)),
                     "differs": s[k] != c[k],
                     "subset_outside_space": lo is not None and not (lo - 1e-9 <= s[k] <= hi + 1e-9),
                     "control_outside_space": lo is not None and not (lo - 1e-9 <= c[k] <= hi + 1e-9)})
diff = pd.DataFrame(rows)
diff.to_csv(os.path.join(TAB, "parameter_differences_all.csv"), index=False)
bnd = pd.DataFrame(boundary)
bnd.to_csv(os.path.join(TAB, "boundary_rule_by_block.csv"), index=False)
N["boundary"] = {"bounce_both": int(((bnd.boundary_subset == "bounce") & (bnd.boundary_control == "bounce")).sum()),
                 "own_handling_both": int(((bnd.boundary_subset == "own handling") & (bnd.boundary_control == "own handling")).sum()),
                 "mismatched": int((bnd.boundary_subset != bnd.boundary_control).sum()),
                 "control_equals_diverse_v2_values": int(bnd.control_equals_diverse_v2_values.sum()),
                 "same_keys": int(bnd.same_parameter_keys.sum())}
N["outside_space"] = diff[diff.subset_outside_space | diff.control_outside_space][
    ["algorithm", "parameter", "lower", "subset_tuned", "continuous_tuned"]].to_dict("records")

POP = {"SHADE": "population_factor", "LSHADE": "pop_size_factor", "LSRTDE": "population_factor",
       "JSO": "population_coefficient"}
N["population_parameters"] = {a: {"parameter": k, "subset": SETS["subset"]["algorithms"][a]["params"][k],
                                  "control": SETS["control"]["algorithms"][a]["params"][k]} for a, k in POP.items()}
FOCUS = ["HybridAdditivePSO_WithWanderer", "HybridDisjointPSO_WithWanderer",
         "HybridAdditivePSO_WithRandom_Var", "HybridPartialDisjointPSO_WithRandom_Var",
         "HybridAdditivePSO_WithRandom", "HybridPartialDisjointPSO_WithRandom", "IKUNGA"]
diff[diff.algorithm.isin(FOCUS)].to_csv(os.path.join(TAB, "parameter_differences_focus.csv"), index=False)

# does any single summary of the parameter difference predict the effect?
mad = diff.groupby("algorithm").normalised_difference.apply(lambda v: v.abs().mean())
w = diff[diff.parameter == "w"].set_index("algorithm")
dw = (w.subset_tuned - w.continuous_tuned).astype(float)
r_mad = spearmanr(mad.loc[by_alg.index], by_alg.a12_median)
r_dw = spearmanr(dw, by_alg.loc[dw.index, "a12_median"])
r_np = spearmanr(by_alg.n_params, by_alg.a12_median)
N["predictors_of_median_a12"] = {
    "mean_abs_normalised_difference": {"n": int(len(mad)), "rho": float(r_mad.statistic), "p": float(r_mad.pvalue)},
    "inertia_difference_w": {"n": int(len(dw)), "rho": float(r_dw.statistic), "p": float(r_dw.pvalue),
                             "w_subset_gt_w_control": int((dw > 0).sum())},
    "block_size": {"n": int(len(by_alg)), "rho": float(r_np.statistic), "p": float(r_np.pvalue)}}
pd.DataFrame({"w_subset": w.subset_tuned, "w_control": w.continuous_tuned, "dw": dw,
              "a12_median": by_alg.loc[dw.index, "a12_median"]}).to_csv(os.path.join(TAB, "inertia_vs_effect.csv"))


# ---------------------------------------------------------------- 3. stored configurations and run times
def _scan(files, arm):
    conf, times = [], []
    for f in files:
        with h5py.File(f, "r") as h:
            pk = list(h.keys())[0]
            for a in h[pk]:
                g = h[pk][a]
                if "run_times" in g:
                    for i, t in zip(g["run_ids"][()], g["run_times"][()]):
                        times.append({"arm": arm, "instance": T.instance_label(pk), "algorithm": a,
                                      "run_id": int(i), "run_time": float(t)})
                if "configuration_json" in g.attrs:
                    conf.append({"arm": arm, "algorithm": a, "parameter_set": g.attrs.get("parameter_set"),
                                 "signature": g.attrs.get("configuration_signature"), "instance": T.instance_label(pk),
                                 "cfg": json.loads(g.attrs["configuration_json"])})
    return conf, times


F_MAIN = sorted(glob.glob("experiment_results/transfer/*.h5"))
F_CTRL = sorted(glob.glob("experiment_results/control/control_2026_10_continuous_bounce/transfer/*.h5"))
c1, t1 = _scan(F_MAIN, "stage2_main")
c2, t2 = _scan(F_CTRL, "stage2_control")
RT = pd.DataFrame(t1 + t2)
RT["baseline"] = RT.algorithm.str.startswith("Baseline")

checked = visible = mismatched = 0
for r in c1 + c2:
    if r["algorithm"].startswith("Baseline"):
        continue
    ps = SETS["subset" if r["arm"] == "stage2_main" else "control"]["algorithms"][r["algorithm"]]["params"]
    for k, v in ps.items():
        checked += 1
        if k in r["cfg"]:
            visible += 1
            x = r["cfg"][k]
            same = x == v or (isinstance(v, (int, float)) and isinstance(x, (int, float)) and abs(x - v) <= 1e-9 * max(1, abs(v)))
            mismatched += 0 if same else 1
sig = {(r["arm"], r["instance"], r["algorithm"]): r["signature"] for r in c1 + c2 if not r["algorithm"].startswith("Baseline")}
N["stored_configuration_check"] = {
    "parameter_entries": checked, "visible_in_configuration_json": visible, "mismatches": mismatched,
    "cells_with_identical_signature_across_arms": sum(sig[("stage2_main", i, a)] == sig.get(("stage2_control", i, a))
                                                      for (arm, i, a) in sig if arm == "stage2_main")}

cost = RT.groupby(["arm", "baseline"]).agg(runs=("run_time", "size"), core_hours=("run_time", lambda v: v.sum() / 3600))
cost.to_csv(os.path.join(TAB, "stored_cost_by_arm.csv"))
N["stored_core_hours"] = {f"{a}|{'baselines' if b else 'metaheuristics'}": round(float(x), 1)
                          for (a, b), x in cost.core_hours.items()}

mh = RT[~RT.baseline]
m2 = mh[mh.arm == "stage2_main"]
per_alg2 = (m2.groupby("algorithm").run_time.sum() / 3600)
per_alg2_nc = (m2[~m2.instance.isin(CEIL)].groupby("algorithm").run_time.sum() / 3600)
pd.DataFrame({"stage2_main_core_h_all10": per_alg2, "stage2_main_core_h_below_ceiling5": per_alg2_nc}).to_csv(
    os.path.join(TAB, "stage2_cost_by_algorithm.csv"))
N["stage2_cost"] = {"main_metaheuristics_all10": round(float(per_alg2.sum()), 1),
                    "main_metaheuristics_below_ceiling5": round(float(per_alg2_nc.sum()), 1),
                    "focus_all10": round(float(per_alg2.loc[FOCUS].sum()), 1),
                    "focus_below_ceiling5": round(float(per_alg2_nc.loc[FOCUS].sum()), 1),
                    "wanderer_var_random_6_below_ceiling5": round(float(per_alg2_nc.loc[FOCUS[:6]].sum()), 1),
                    "pcb_share_of_main": round(float(m2[m2.instance.str.startswith("PCB")].run_time.sum() / m2.run_time.sum()), 3)}

# ---------------------------------------------------------------- 4. figure: verdict per algorithm
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 7, "xtick.direction": "out",
                     "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "legend.frameon": False})
order_alg = list(by_alg.index)
order_inst = sorted(set(paired.instance) - set(CEIL), key=lambda s: (s.split("/")[0][:3], s)) + sorted(CEIL)
M = paired.pivot(index="algorithm", columns="instance", values="a12_main_vs_control").loc[order_alg, order_inst]
V = paired.pivot(index="algorithm", columns="instance", values="verdict").loc[order_alg, order_inst]
fig = plt.figure(figsize=(7.0, 5.4))
gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1.0], height_ratios=[1.0, 0.025], wspace=0.04, hspace=0.42)
ax, bx, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])
im = ax.imshow(M.values, cmap="PuOr", norm=TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0), aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if V.iat[i, j] != "no difference":
            ax.plot(j, i, marker="o", ms=3.0, mfc="white", mec="black", mew=0.6)
nb = len(order_inst) - len(CEIL)
ax.axvline(nb - 0.5, color="black", lw=0.8)
ax.set_xticks(range(len(order_inst)))
ax.set_xticklabels([s.replace("/", " ") for s in order_inst], rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(order_alg)))
ax.set_yticklabels(order_alg)
ax.text((nb - 1) / 2, -0.8, "below the baseline ceiling", ha="center", va="bottom", fontsize=7)
ax.text(nb + (len(CEIL) - 1) / 2, -0.8, "at the baseline ceiling", ha="center", va="bottom", fontsize=7)
n_sig = int((paired.verdict != "no difference").sum())
n_main = int((paired.verdict == "main better").sum())
n_rr = int(((paired.verdict == "control better") & paired.algorithm.isin(FOCUS[4:6])).sum())
n_ctl = int((paired.verdict == "control better").sum())
fig.suptitle(f"{n_main} of {n_sig} significant cells favour the subset-tuned values; "
             f"two random-role hybrids hold {n_rr} of the other {n_ctl}", x=0.02, ha="left", y=0.985, fontsize=8)
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("A12, subset-tuned vs continuous-tuned (> 0.5: subset-tuned better)", fontsize=7)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.ax.tick_params(labelsize=6)
y = np.arange(len(order_alg))
mb_, cb_, nd_ = by_alg.main_better.values, by_alg.control_better.values, by_alg.no_difference.values
bx.barh(y, mb_, height=0.8, color="#5e3c99", label="subset-tuned better")
bx.barh(y, nd_, height=0.8, left=mb_, color="#d9d9d9", label="no difference")
bx.barh(y, cb_, height=0.8, left=mb_ + nd_, color="#e66101", label="continuous-tuned better")
bx.set_ylim(len(order_alg) - 0.5, -0.5)
bx.set_yticks(y)
bx.set_yticklabels([])
bx.set_xlim(0, 10)
bx.set_xticks([0, 5, 10])
bx.set_xlabel("instances (of 10)")
handles = [Line2D([], [], marker="o", ls="", mfc="white", mec="black", ms=4, label="Holm-adjusted p < 0.05"),
           *bx.get_legend_handles_labels()[0]]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=4, fontsize=6, handlelength=1.2)
fig.subplots_adjust(left=0.30, right=0.985, top=0.92, bottom=0.11)
fig.savefig(os.path.join(FIG, "fig_verdict_per_algorithm.png"))
fig.savefig(os.path.join(FIG, "fig_verdict_per_algorithm.pdf"))

with open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8", newline="\r\n") as fh:
    json.dump(N, fh, indent=1, default=str)
print(json.dumps(N, indent=1, default=str))
