"""OLA 2027 paper: Figures 1 and 2 for the LNCS text width (12.2 cm), no text below 7 pt, short names.

Read-only. Figure 1 uses the sources of analysis.py's Figure 1 (experiment_results/analysis/transfer/
transfer_paired.csv, ../tables/stage2_baseline_hit_rates.csv, ../tables/verdicts_by_algorithm.csv); Figure 2
the sources of figures_followups.py (../tables/m1a_blocks.csv, m3_blocks.csv). Writes
ola2027/figures/fig1_verdicts.{pdf,png} and fig2_followups.{pdf,png}; ../figures/ is not touched.

Style (revised 7 November 2026): journal-style sans-serif type, a diverging red-blue scale centred on
"no effect", white cell separators instead of frames and hatching, algorithms grouped by family, and the two
follow-up experiments as separate panels. The page footprint of both figures is unchanged (12.2 x 11.6 cm and
12.2 x 9.4 cm), so the paper's layout does not move.
Run:  python papers/parameter_transfer/ola2027/figures_ola.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PT = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PT))
TAB, FIG = os.path.join(PT, "tables"), os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
names = pd.read_csv(os.path.join(HERE, "names.csv"))
NAME = dict(names[["roster_name", "short_name"]].values)
GROUP = dict(names[["roster_name", "group"]].values)
CM = 1 / 2.54
W = 11.35 * CM          # 0.93 of the LNCS text width, drawn at size (fonts stay >= 7 pt)
MIN_PT = 7

INK, MID, LIGHT = "#222222", "#6b6b6b", "#e6e6e6"
RED, BLUE = "#b2182b", "#2166ac"          # ends of RdBu: S better / C better
CMAP = plt.get_cmap("RdBu_r")

# Font: FIG_FONT selects a system sans-serif font (default Source Sans Pro, maths in the same face via mathtext),
# or "cmbright" for LaTeX's CM Bright sans (text and maths through LaTeX; needs latex and dvipng on the PATH).
FONT = os.environ.get("FIG_FONT", "Source Sans Pro")
FS = 7                                    # one text size for every label, tick, legend and annotation
USETEX = FONT == "cmbright"
DASH = "--" if USETEX else "–"
FONT_RC = ({"text.usetex": True, "font.family": "sans-serif", "text.latex.preamble": r"\usepackage{cmbright}"}
           if USETEX else
           {"font.family": "sans-serif", "font.sans-serif": [FONT, "Arial", "DejaVu Sans"],
            "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
            "mathtext.bf": f"{FONT}:bold"})
plt.rcParams.update({
    **FONT_RC,
    "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.direction": "out", "ytick.direction": "out",
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})


def check_fonts(fig, name):
    small = [t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_fontsize() < MIN_PT]
    assert not small, f"{name}: text below {MIN_PT} pt: {small[:5]}"


def cell_grid(ax, nrows, ncols, lw=0.6):
    """White separators between heatmap cells; no frame."""
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=lw)
    for x in (-0.5, ncols - 0.5):                                         # clean outer edges
        ax.axvline(x, color="white", lw=lw * 1.5, clip_on=False, zorder=2)
    for y in (-0.5, nrows - 0.5):
        ax.axhline(y, color="white", lw=lw * 1.5, clip_on=False, zorder=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)


def panel_label(fig, ax, letter, title, dx=-0.0):
    x0 = ax.get_position().x0
    fig.text(x0 + dx, ax.get_position().y1 + 0.075, r"\textbf{%s}" % letter if USETEX else letter, fontsize=FS,
             fontweight="bold", ha="right", va="bottom")
    fig.text(x0 + dx + 0.012, ax.get_position().y1 + 0.075, title, fontsize=FS, ha="left", va="bottom")


# ================================================================== Figure 1: main comparison
paired = pd.read_csv(os.path.join(ROOT, "experiment_results", "analysis", "transfer", "transfer_paired.csv"))
bl = pd.read_csv(os.path.join(TAB, "stage2_baseline_hit_rates.csv"))
by_alg = pd.read_csv(os.path.join(TAB, "verdicts_by_algorithm.csv")).set_index("algorithm")
CEIL = sorted(bl.instance[bl.ceiling])
order_inst = sorted(set(paired.instance) - set(CEIL), key=lambda s: (s.split("/")[0][:3], s)) + CEIL
FAMILIES = [("Role-based PSO", {"role pair", "role hybrid"}),
            ("DE", {"DE"}),
            ("Other", {"basic PSO", "reference PSO", "reference"})]
fam_of = {a: next(f for f, gs in FAMILIES if GROUP[a] in gs) if GROUP[a] != "IKUN" else
          ("DE" if a == "IKUNDE" else "Other") for a in by_alg.index}
order_alg, fam_spans = [], []
for f, _ in FAMILIES:
    members = sorted([a for a in by_alg.index if fam_of[a] == f], key=lambda a: -by_alg.at[a, "a12_median"])
    fam_spans.append((f, len(order_alg), len(order_alg) + len(members) - 1))
    order_alg += members
assert sorted(order_alg) == sorted(by_alg.index)
M = paired.pivot(index="algorithm", columns="instance", values="a12_main_vs_control").loc[order_alg, order_inst]
V = paired.pivot(index="algorithm", columns="instance", values="verdict").loc[order_alg, order_inst]

fig = plt.figure(figsize=(W, 10.8 * CM))
gs = fig.add_gridspec(2, 2, width_ratios=[3.4, 1.0], height_ratios=[1.0, 0.028], wspace=0.06, hspace=0.5)
ax, bx, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])
norm = TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0)
im = ax.imshow(M.values, cmap=CMAP, norm=norm, aspect="auto")
cell_grid(ax, *M.shape)
sig_i, sig_j = np.nonzero((V.values != "no difference"))
ax.scatter(sig_j, sig_i, s=7, facecolor="white", edgecolor=INK, linewidth=0.5, zorder=3)
nb = len(order_inst) - len(CEIL)
ax.axvline(nb - 0.5, color="white", lw=2.4)                                  # split: baselines solve / do not
for _, a, b in fam_spans[:-1]:
    ax.axhline(b + 0.5, color="white", lw=2.4)
ax.set_xticks(range(len(order_inst)))
ax.set_xticklabels([s.replace("/", " ") for s in order_inst], rotation=45, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(order_alg)))
ax.set_yticklabels([NAME[a] for a in order_alg])
for x, txt in (((nb - 1) / 2, "Not solved\nby baselines"), (nb + (len(CEIL) - 1) / 2, "Solved\nby baselines")):
    ax.text(x, -0.8, txt, ha="center", va="bottom", fontsize=FS)
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.outline.set_visible(False)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cb.ax.tick_params(labelsize=FS, length=2, width=0.6)
cb.set_label(r"$A_{12}$ of S against C", fontsize=FS, labelpad=2)
cax.text(-0.01, 0.5, "C better", transform=cax.transAxes, ha="right", va="center", fontsize=FS, color=BLUE)
cax.text(1.01, 0.5, "S better", transform=cax.transAxes, ha="left", va="center", fontsize=FS, color=RED)

y = np.arange(len(order_alg))
cnt = by_alg.loc[order_alg]
mb, nd, cbt = cnt.main_better.values, cnt.no_difference.values, cnt.control_better.values
kw = dict(height=0.72, edgecolor="white", linewidth=0.4)
bx.barh(y, mb, color=RED, **kw)
bx.barh(y, nd, left=mb, color=LIGHT, **kw)
bx.barh(y, cbt, left=mb + nd, color=BLUE, **kw)
for _, a, b in fam_spans[:-1]:
    bx.axhline(b + 0.5, color="white", lw=2.4)
bx.set_ylim(len(order_alg) - 0.5, -0.5)
bx.set_yticks([])
bx.spines["left"].set_visible(False)
bx.set_xlim(0, 10)
bx.set_xticks([0, 5, 10])
bx.set_xlabel("Instances")
bx.xaxis.grid(True, color=LIGHT, lw=0.5)
for f, a, b in fam_spans:                                                     # family brackets on the right
    bx.plot([10.6, 10.6], [a - 0.35, b + 0.35], color=MID, lw=0.6, clip_on=False)
    bx.text(10.9, (a + b) / 2, f, rotation=270, ha="left", va="center", fontsize=FS, clip_on=False)
bx.set_axisbelow(True)
handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=0.5, ms=3.2, label=r"Holm-adjusted $p<0.05$"),
           Patch(facecolor=RED, label="S better"), Patch(facecolor=LIGHT, label="No difference"),
           Patch(facecolor=BLUE, label="C better")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.56, 0.0), ncol=4, handlelength=1.0,
           handleheight=0.8, columnspacing=0.7, handletextpad=0.3)
fig.subplots_adjust(left=0.19, right=0.925, top=0.925, bottom=0.135)
check_fonts(fig, "fig1")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig1_verdicts.{ext}"))
plt.close(fig)

# ================================================================== Figure 2: follow-ups
m1a = pd.read_csv(os.path.join(TAB, "m1a_blocks.csv"))
m3 = pd.read_csv(os.path.join(TAB, "m3_blocks.csv"))
M1A_ROWS = ["HybridAdditivePSO_WithWanderer", "HybridDisjointPSO_WithWanderer", "HybridAdditivePSO_WithRandom_Var",
            "HybridPartialDisjointPSO_WithRandom_Var", "IKUNGA", "HybridAdditivePSO_WithRandom",
            "HybridPartialDisjointPSO_WithRandom"]
M3_ROWS = ["SHADE", "LSHADE", "LSRTDE", "JSO"]
assert set(M1A_ROWS) == set(m1a.algorithm) and set(M3_ROWS) == set(m3.algorithm)
DIRS = ["C<-S", "S<-C"]
DLAB = ({"C<-S": r"C$\leftarrow$S", "S<-C": r"S$\leftarrow$C"} if USETEX else {"C<-S": "C←S", "S<-C": "S←C"})
snorm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.2)


def share_matrix(df, rows, groups, n_inst):
    share = np.full((len(rows), 2 * len(groups)), np.nan)
    count = np.full(share.shape, "", dtype=object)
    outcome = {}
    for i, a in enumerate(rows):
        for gi, g in enumerate(groups):
            r = df[(df.algorithm == a) & (df.group == g)].iloc[0] if g is not None else df[df.algorithm == a].iloc[0]
            outcome[(i, gi)] = r.outcome
            for di, d in enumerate(DIRS):
                s = r[f"median_share_{d}"]
                if pd.notna(s):
                    share[i, 2 * gi + di] = s
                    count[i, 2 * gi + di] = f"{int(r[f'toward_{d}'])}/{n_inst}"
    return share, count, outcome


def draw_share(ax, share, count, outcome, rows, group_titles):
    ax.imshow(np.ma.masked_invalid(share), cmap=CMAP, norm=snorm, aspect="auto")
    cell_grid(ax, *share.shape, lw=0.8)
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share[i, j]
            if np.isnan(v):
                continue
            dark = v > 0.72 or v < -0.62
            ax.text(j, i - 0.17, f"${v:.2f}$" if USETEX else f"{v:.2f}".replace("-", "−"), ha="center", va="center", fontsize=FS, color="white" if dark else INK)
            ax.text(j, i + 0.24, count[i, j], ha="center", va="center", fontsize=FS,
                    color="#f0f0f0" if dark else MID)
    for (i, gi), o in outcome.items():
        if o in ("carries", "contributes"):
            ax.add_patch(Rectangle((2 * gi - 0.46, i - 0.46), 1.92, 0.92, fill=False, edgecolor=INK,
                                   lw=1.1 if o == "carries" else 0.9, ls="-" if o == "carries" else (0, (2.2, 1.4)),
                                   zorder=4, clip_on=False))
        if o == "no Stage 2 effect":
            ax.add_patch(Rectangle((2 * gi - 0.5, i - 0.5), 2, 1, facecolor="#f4f4f4", edgecolor="white", lw=0.8))
            ax.text(2 * gi + 0.5, i, "no effect\nin main\ncomparison", ha="center", va="center", fontsize=FS,
                    linespacing=0.95)
    ax.set_xticks(range(share.shape[1]))
    ax.set_xticklabels([DLAB[d] for d in DIRS] * len(group_titles))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([NAME[a] for a in rows])
    for gi, g in enumerate(group_titles):                                   # group header with a rule
        if not g:
            continue
        ax.plot([2 * gi - 0.38, 2 * gi + 1.38], [-0.62, -0.62], color=MID, lw=0.6, clip_on=False)
        ax.text(2 * gi + 0.5, -0.72, g, ha="center", va="bottom", fontsize=FS, linespacing=0.95)


sa, ca, oa = share_matrix(m1a, M1A_ROWS, ["base", "coefficients", "shares"], 4)
sb, cbn, ob = share_matrix(m3.assign(group=None), M3_ROWS, [None], 2)

fig = plt.figure(figsize=(W, 8.75 * CM))
gs = fig.add_gridspec(1, 4, width_ratios=[6.0, 1.75, 2.0, 0.16], wspace=0.0)
ax, bx, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])
draw_share(ax, sa, ca, oa, M1A_ROWS, ["Base\n($w$, $c_1$, $c_2$)", "Role\ncoefficients", "Role probabilities\nor fractions"])
draw_share(bx, sb, cbn, ob, M3_ROWS, [""])
bx.set_ylim(len(M1A_ROWS) - 0.5, -0.5)                                     # same row height as panel a
sm = plt.cm.ScalarMappable(norm=snorm, cmap=CMAP)
cb = fig.colorbar(sm, cax=cax)
cb.outline.set_visible(False)
cb.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
cb.ax.tick_params(labelsize=FS, length=2, width=0.6)
cb.set_label("Median share", fontsize=FS)
fig.subplots_adjust(left=0.2, right=0.89, top=0.83, bottom=0.135)
pos = cax.get_position()                                                     # colorbar: height of panel b's rows
bpos = bx.get_position()
cax.set_position([pos.x0 + 0.015, bpos.y0 + bpos.height * (3 / 7), pos.width * 0.7, bpos.height * 4 / 7])
panel_label(fig, ax, "a", r"Parameter-group swap, GR ($n=100$" + DASH + "$150$)", dx=-0.13)
panel_label(fig, bx, "b", "Population swap, SJC818", dx=-0.1)
fig.legend(handles=[Line2D([], [], color=INK, lw=1.1, label="Carries"),
                    Line2D([], [], color=INK, lw=0.9, ls=(0, (2.2, 1.4)), label="Contributes"),
                    Patch(facecolor="white", edgecolor=MID, lw=0.5, label="Unframed: no detectable role")],
           loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, handlelength=1.8, columnspacing=1.4)
check_fonts(fig, "fig2")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIG, f"fig2_followups.{ext}"))
plt.close(fig)
print("figures written to", FIG)
