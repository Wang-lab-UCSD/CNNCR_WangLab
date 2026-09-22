"""
Figure 3 — Core regulators are evolutionarily constrained (panels a–h).
One coherent Nature style + identical panel dimensions so a–h tile into a 2×4 grid.

 a  CG content            core vs JASPAR (n=841) and CisBP (n=6,607) catalogues
 b  Per-base specificity   information content PER BASE vs both catalogues
 c  Site conservation     phyloP at the matched bases
 d  Conservation beyond CG  per-motif phyloP vs CG, core + control      ← NEW
 e  Biosample advantage   116 biosamples, core vs control
 f  Protein constraint    pLI / LOEUF / missense z + synonymous-z negative control
 g  Domain constraint     DBD missense obs/exp
 h  Reader constraint     LOEUF of every reader TF, by dominant motif family ← NEW

⚠ RELETTERING (2026-07-31). Two panels were inserted, so the old d/e/f are now e/f/g.
The paper body and the Fig 3 legend must be updated to match.

The two new panels exist to answer the two objections the old figure could not:
 d  core motifs are more CG-rich than control AND conservation rises with CG, so the panel-c
    gap could be a CpG-island effect. Within EACH motif set separately, higher-CG motifs are
    more conserved at the SAME rate (slope +0.430 core, +0.456 control), so the two fitted
    lines are parallel; core sits above control by a constant offset: CG-adjusted advantage
    +0.084, P = 0.0018. CG explains conservation; it does not explain the core-vs-control gap.
 h  the 7-fold spread of conservation across core motifs could follow the reader-TF family
    rather than the motif. It does not: families with n>=3 motifs do not differ (P = 0.094)
    while the spread WITHIN C2H2 ZF alone is 7.4-fold. No control arm — there are only 43
    control motifs against 286 reader TFs, so a within-family core-vs-control split would
    compare a handful with a handful. The null is the point: conservation is per-motif.

Every panel shows the underlying DATA, not just a summary:
 a/b/c/g = violin + jittered individual points + inner IQR box + white median marker
 d       = per-motif scatter, both arms, with per-arm OLS fits (marker SHAPE differs, so
           identity never rests on colour — see the palette note below)
 e       = per-biosample scatter (116 points) with a shaded core>control field + rug
 f       = grouped rounded bars of the protein-constraint medians
 h       = one lane per motif family: all reader TFs as points, IQR bar, median diamond
Exact P-values (Mann–Whitney two-sided; sign test for e) are computed here and
PRINTED directly on each panel — no significance stars.

Data: a/b cg_ic_per_motif.csv; c/e conservation_aggregate.csv;
      d  code/results/motif_conservation_cg.csv + motif_cg_ancova.csv   (04_*.py)
      f  code/results/constraint_comparison_extended.csv                (03_*.py)
      g  conservation_protein dbd_oe_scores.csv
      h  code/results/family_reader_constraint.csv                     (07_*.py)
"""
import os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, binomtest, kruskal
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})

SD   = os.path.dirname(os.path.abspath(__file__))
MA   = "/stg3/data1/sam/enhancer_prediction/multi_agent"
CGIC = f"{MA}/posthoc_analysis/results/cg_ic_per_motif.csv"
CONS = f"{MA}/conservation_sequence/results/conservation_aggregate.csv"
DBD  = f"{MA}/conservation_protein/results/dbd_oe_scores.csv"
CC   = f"{SD}/code/results/constraint_comparison_extended.csv"   # 3 published + syn-z control
FAM  = f"{SD}/code/results/family_constraint.csv"
FAMR = f"{SD}/code/results/family_reader_constraint.csv"
MCG  = f"{SD}/code/results/motif_conservation_cg.csv"   # single-control version (superseded)
MCGA = f"{SD}/code/results/motif_cg_ancova.csv"          # single-control version (superseded)
# TEN control sets (2026-09-05): panels c, d, e now use the same ten old-pool control sets as
# Fig 4a/b (posthoc_analysis/results/control_motifs_set_{0..9}.meme), scored from one pool-wide
# FIMO scan by conservation_sequence/07_oldpool_10sets.py. c and d pool the 141 distinct control
# motifs (each once); e uses the mean over the ten sets per biosample with an s.d. error bar.
OP10 = f"{MA}/conservation_sequence/results/oldpool10"
ROWS10, PM10, BS10, ST10 = (f"{OP10}/fig3_rows.csv", f"{OP10}/fig3_permotif.csv",
                            f"{OP10}/fig3_per_biosample.csv", f"{OP10}/fig3_stats.csv")
CISBP = f"{SD}/code/results/cisbp_cg_ic.csv"

# PALETTE — the Fig 1 / Fig 2 house system, in a DIFFERENT hue set (user, 2026-07-31).
#
# Figs 1 and 2 do not colour a series with one flat hue at low alpha. Each series is an
# EDGE/FILL PAIR: a soft mid-tone edge at OKLCH L ~0.71, C ~0.06-0.085, over a near-white
# fill at L ~0.93, C ~0.02. Measured from their sources:
#   fig1  #8FB8C9/#D6E7EE  #D79581/#F4E0D8  #7E86B8/#E2E4F1  #BFA06B/#F1E8D7
#         #7FA382/#E3EFE1  #A98BB0/#EDE4F0
#   fig2  #8FA6C9/#E3E9F3  #D79581/#F4E0D8  #AEB4B8/#E7EAEC
# Hues already spoken for: 38, 80, 147, 225, 237, 259, 277, 320.
# Fig 3 keeps that exact shade register and takes an unused hue pair: TEAL 195 (30 deg from
# fig2's blue, 48 from fig1's green) and DUSTY VIOLET 300 (23 from indigo, 20 from plum).
#
# ⚠ WHY THIS SET IS NOT COLOURBLIND-SEPARABLE, AND WHY THAT IS ACCEPTABLE *HERE*.
# A search over the whole hue circle at this chroma returns NO three-series set that clears
# the separation gates: the best possible anywhere is dE 7.4 under simulated protan/deutan
# against a target of 8, and 13.9 for normal vision against a floor of 15. That is a property
# of the shade, not of the hue choice. Pastel means low chroma, and low chroma means marks
# converge under colourblind simulation. Matching Fig 1 and Fig 2 therefore costs separability
# by construction, and no hue set buys it back.
# It is acceptable in THIS figure only because colour is a secondary cue: every violin sits at
# its own x position with its own text label underneath, panel e carries a legend, and no two
# marks ever overlap. Identity never rests on colour alone.
# ⚠ DO NOT reuse this palette for a chart where marks overlap or share an axis position
# (scatter, stacked bars, overlaid lines). There it would genuinely be unreadable.
CORE_E, CORE_F = "#53b0b0", "#d6edee"      # teal    195
JASP_E, JASP_F = "#a08cc7", "#eae5f7"      # violet  300
CTRL_E, CTRL_F = "#93908a", "#ebe9e6"      # warm grey — CONTROL only (panels c–g)
# CisBP is a second CATALOGUE, not a control, so it must not share the control's grey
# (user, 2026-07-31). Same edge/fill register as the rest: soft ochre, hue ~85. It only ever
# sits beside Core (teal 195) and JASPAR (violet 300) in panels a/b, and 85/195/300 splits the
# hue circle into 110/105/145 gaps — the most even three-way spread available.
# ⚠ Hue 85 is adjacent to fig1's 80. Accepted: fig 1 and fig 3 are never read as one panel set,
# and every alternative either collides with a taken hue or crowds teal/violet.
CISB_E, CISB_F = "#bfa46b", "#f1ead7"      # ochre 85 — CisBP catalogue
CORE, JASP, CTRL = CORE_E, JASP_E, CTRL_E  # back-compat for single-colour call sites

# chrome tokens taken from fig1/fig2 so the figures read as one set
INK, SUB, MUT, GRID, BASE = "#26313a", "#66737c", "#a9b4bb", "#eef1f2", "#c3ccd3"

EDGE_LW = 0.9      # the edge is what carries identity, so it is drawn, not faded
POINT_ALPHA = 0.75

# Nature spec: 7 pt bold panel title, 6 pt axis labels/categories, 5.5 pt tick numbers,
# 5 pt everything smaller. Sizes only mean anything if the panel is drawn at the size it
# will be printed, so PANEL is one QUARTER of the 183 mm double-column width
# (45.75 mm = 1.80 in) and a–h tile into a 2x4 grid at exactly 183 mm with NO rescaling.
#
# Going from six panels to eight changes the panel WIDTH only, 2.40 -> 1.80 in. The sheet is
# unchanged at 183 x 115.4 mm because 2/3 x (2.27/2.40) and 2/4 x (2.27/1.80) are the same
# number — so the added panels cost nothing in height and nothing in font size.
# (A 3-narrow-plus-1-wide row was considered and rejected: it needs a 5-column grid at
# 36.6 mm, too tight for a three-violin panel with 5 pt category labels.)
FS_TITLE, FS_LAB, FS_TICK, FS_SM = 7, 6, 5.5, 5
PANEL = (1.80, 2.27)
RNG = np.random.default_rng(7)

# ── p-value formatting ─────────────────────────────────────────────────────────
SUP = {"0":"⁰","1":"¹","2":"²","3":"³","4":"⁴","5":"⁵","6":"⁶","7":"⁷","8":"⁸","9":"⁹","-":"⁻"}
def fmtp(p):
    if p <= 0 or p < 1e-300: return "P < 10" + "".join(SUP[c] for c in "-300")
    if p >= 0.10:  return f"P = {p:.2f}"
    if p >= 0.05:  return f"P = {p:.3f}"      # 0.058 must not render as "0.06"
    if p >= 1e-3:  return f"P = {p:.3f}".rstrip("0")
    e = int(np.floor(np.log10(p))); m = p / 10**e
    return f"P = {m:.1f}×10" + "".join(SUP[c] for c in str(e))
def mwp(a, b): return mannwhitneyu(a, b, alternative="two-sided").pvalue

def chrome(ax, ylab):
    ax.set_ylabel(ylab, fontsize=FS_LAB, color=SUB)
    ax.tick_params(axis="y", labelsize=FS_TICK, colors=MUT, length=0)
    ax.tick_params(axis="x", labelsize=FS_LAB, colors=INK, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(BASE); ax.spines[s].set_linewidth(0.9)
# Short bold LEFT-aligned header beside the panel letter, matching Fig 2b/2c and Fig 5.
def title(ax, t): ax.set_title(t, fontsize=FS_TITLE, color=INK, fontweight="bold",
                               pad=4, loc="left")
def bracket(ax, x1, x2, y, dy, txt):
    ax.plot([x1, x1, x2, x2], [y, y+dy, y+dy, y], lw=0.7, color=MUT, clip_on=False, zorder=8)
    ax.text((x1+x2)/2, y+dy, " " + txt, ha="center", va="bottom", fontsize=FS_SM,
            color=INK, zorder=8)

def strip(ax, i, vals, color, maxpts):
    v = vals if len(vals) <= maxpts else RNG.choice(vals, maxpts, replace=False)
    jit = RNG.normal(0, 0.055, size=len(v)); jit = np.clip(jit, -0.17, 0.17)
    ax.scatter(i + jit, v, s=4.5, color=color, alpha=POINT_ALPHA, edgecolor="white",
               lw=0.15, zorder=4, rasterized=True)

def innerbox(ax, i, vals, edge=None):
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    ax.plot([i, i], [q1, q3], color=INK, lw=2.8, solid_capstyle="round", zorder=6, alpha=0.85)
    ax.scatter(i, med, s=17, facecolor="white", edgecolor=edge or INK, lw=1.1, zorder=7)

def draw_violins(ax, data, cols, labels, ylab, ttl, comps, maxpts=220):
    """comps = list of (i,j,p) brackets to draw, stacked.
    cols = list of (edge, fill) pairs — the fig1/fig2 edge-over-pale-fill construction."""
    pos = list(range(len(data)))
    parts = ax.violinplot(data, positions=pos, widths=0.84, showextrema=False)
    for b, (e, f) in zip(parts["bodies"], cols):
        b.set_facecolor(f); b.set_edgecolor(e); b.set_alpha(1.0)
        b.set_linewidth(EDGE_LW); b.set_zorder(2)
    for i, (d, (e, f)) in enumerate(zip(data, cols)):
        strip(ax, i, d, e, maxpts); innerbox(ax, i, d, e)
    ax.set_xticks(pos); ax.set_xticklabels(labels)
    chrome(ax, ylab); title(ax, ttl)
    allv = np.concatenate(data); lo, hi = allv.min(), allv.max(); span = hi - lo
    dy = span * 0.028
    ytop = hi + span * 0.07
    for (i, j, p) in comps:
        bracket(ax, i, j, ytop, dy, fmtp(p)); ytop += span * 0.145
    ax.set_xlim(-0.6, len(data)-0.4)
    ax.set_ylim(lo - span*0.05, ytop + span*0.02)

# ── a / b : CG + information content, vs the two motif CATALOGUES ───────────────
# The 43-motif random control was dropped from a and b (user, 2026-07-31) and replaced by
# CisBP. A 43-motif control is a noisy subsample of a catalogue asking the same question; the
# catalogues themselves answer it with far more power. Control still carries c–e, where it is
# the functional comparator rather than a reference distribution.
cg = pd.read_csv(CGIC)
cg["g"] = cg["group"].map({"Universal": "Core", "JASPAR": "JASPAR"})
cg = pd.concat([cg[cg.g.notna()], pd.read_csv(CISBP).assign(g="CisBP")], ignore_index=True)
CAT_COLS = [(CORE_E, CORE_F), (JASP_E, JASP_F), (CISB_E, CISB_F)]
CAT_LABS = ["Core", "JASPAR", "CisBP"]
def cgcol(col): return [cg[cg.g == g][col].dropna().values for g in CAT_LABS]

def panel_a(ax):
    # ⚠⚠ BOTH brackets are drawn, and they DISAGREE — that is deliberate.
    #   core vs JASPAR (n=841)    0.582 vs 0.461   P = 0.058  n.s.
    #   core vs CisBP  (n=6,607)  0.582 vs 0.450   P = 0.017  significant
    # The effect size is essentially the same against either catalogue; only power differs.
    # So "core motifs are not significantly CG-enriched" holds ONLY against the smaller
    # catalogue, and a referee holding CisBP gets 0.017. Showing one bracket would make the
    # figure depend on which catalogue the reader happens to have. The defensible claim is
    # that core motifs are modestly CG-enriched and that this does not drive the conservation
    # result — which is what panel d and the CG-matched test (n=34, P = 0.0028) establish.
    d = cgcol("cg_content")
    draw_violins(ax, d, CAT_COLS, CAT_LABS, "CG content", "CG content",
                 [(0, 1, mwp(d[0], d[1])), (0, 2, mwp(d[0], d[2]))])

def panel_b(ax):
    # Per-base is stated in the title AND the axis unit, because the per-position framing is
    # what a referee will question: core motifs are all 8 bp against a JASPAR median of 12 and
    # a CisBP median of 11, and their TOTAL IC is LOWER (12.09 vs 13.63 for control, P = 0.004).
    d = cgcol("mean_ic_pos")
    draw_violins(ax, d, CAT_COLS, CAT_LABS, "Information content (bits per base)",
                 "Per-base specificity",
                 [(0, 1, mwp(d[0], d[1])), (0, 2, mwp(d[0], d[2]))])

# ── c / e : binding-site phyloP conservation ────────────────────────────────────
_cons = None
def cons():
    global _cons
    if _cons is None: _cons = pd.read_csv(CONS)
    return _cons
def panel_c(ax):
    r = pd.read_csv(ROWS10); st = pd.read_csv(ST10).set_index("set")
    d = [r[r.group == "core"].median_score.dropna().values,
         r[r.group == "control"].median_score.dropna().values]
    # P from the independent unit (per-motif medians, 43 core vs the 141 distinct control
    # motifs of the ten sets), not pseudo-replicated hits; the per-set range is in the legend
    p = float(st.loc["pooled", "p_fig3c"])
    ps = st.loc[[str(i) for i in range(10)], "p_fig3c"].astype(float)
    print(f"[c] pooled P = {fmtp(p)}; per-set P range {ps.min():.1e}-{ps.max():.1e}, "
          f"{(ps < 0.05).sum()}/10 sets P < 0.05")
    draw_violins(ax, d, [(CORE_E, CORE_F), (CTRL_E, CTRL_F)],
                 ["Core motifs", "Control motifs"],
                 "phyloP100way at binding site", "Site conservation",
                 [(0, 1, p)], maxpts=320)
def panel_e(ax):
    b = pd.read_csv(BS10); st = pd.read_csv(ST10).set_index("set")
    xc, yc, xe = b["ctrl_mean"].values, b["core_median"].values, b["ctrl_sd"].values
    lo = min((xc - xe).min(), yc.min()) * 0.9; hi = max((xc + xe).max(), yc.max()) * 1.06
    ax.fill_between([lo, hi], [lo, hi], hi, color=CORE_F, alpha=1.0, zorder=1)   # core>control field
    ax.plot([lo, hi], [lo, hi], color=MUT, lw=0.9, ls="--", zorder=2)
    # horizontal bar = s.d. of the control median over the ten control sets (as Fig 4a)
    # drawn dark, with caps, and ABOVE the markers so every point shows its bar even where the
    # s.d. is small (median s.d. 0.007 on a 0.16-wide axis)
    # medium grey, BEHIND the markers; the caps stay visible either side of every bubble
    ax.errorbar(xc, yc, xerr=xe, fmt="none", ecolor="#6f6d68", elinewidth=0.5, capsize=1.1, capthick=0.5,
                alpha=0.9, zorder=3)
    ax.scatter(xc, yc, s=11, facecolor=CORE_F, edgecolor=CORE_E, lw=0.6, zorder=4)
    for x in xc: ax.plot([x, x], [lo, lo + (hi-lo)*0.018], color=CORE_E, lw=0.5, alpha=0.6, zorder=3)  # rug
    for y in yc: ax.plot([lo, lo + (hi-lo)*0.018], [y, y], color=CORE_E, lw=0.5, alpha=0.6, zorder=3)
    n = len(b); nabove = int((yc > xc).sum()); nall = int((yc > b["ctrl_max"].values).sum())
    p = binomtest(nabove, n, 0.5, alternative="greater").pvalue
    print(f"[e] {nabove}/{n} biosamples core > mean of ten control sets, sign test {fmtp(p)}; "
          f"core > every one of the ten set medians in {nall}/{n}")
    ax.set_xlabel("Control motifs (median phyloP,\nmean ± s.d. of ten sets)", fontsize=FS_LAB, color=SUB)
    chrome(ax, "Core motifs (median phyloP)"); title(ax, "Biosample advantage")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    # counts and the sign-test P are reported in the figure legend, not drawn on the panel

# ── d : is the site-conservation gap just CG content? ───────────────────────────
# ⚠ This is the one panel in Fig 3 where two series SHARE the plotting area, so the palette
# note above applies: colour alone would not be readable under CVD. Identity is therefore
# carried by MARKER SHAPE (filled circle vs open square) and by line style, with colour only
# reinforcing. Do not "simplify" this to two colours of the same marker.
def panel_d(ax):
    # "core" and "control" are the two MOTIF SETS: the 43 core motifs and the 141 distinct
    # control motifs of the ten old-pool sets (each drawn once). Fits and the ANCOVA come from
    # the 'pooled' row of fig3_stats.csv; the per-set range is printed for the legend.
    d = pd.read_csv(PM10).rename(columns={"median_phylop": "phyloP"}); st = pd.read_csv(ST10).set_index("set")
    xs = np.linspace(d.cg.min() - 0.02, d.cg.max() + 0.02, 50)
    spec = [("core", CORE_E, CORE_F, "o", "-", "Core motifs", "slope_core"),
            ("control", CTRL_E, "none", "s", "--", "Control motifs", "slope_ctrl")]
    lines = {}
    for ms, ec, fc, mk, ls, lab, scol in spec:
        t = d[d.group == ms]
        ax.scatter(t.cg, t.phyloP, s=13, marker=mk, facecolor=fc, edgecolor=ec,
                   lw=0.7, zorder=4, label=lab)   # n reported in the legend text
        sl = float(st.loc["pooled", scol]); ic = t.phyloP.mean() - sl * t.cg.mean()
        lines[ms] = sl * xs + ic
        ax.plot(xs, lines[ms], color=ec, lw=1.1, ls=ls, zorder=5)
    # the message is the VERTICAL GAP between the two (near-parallel) lines: the CG-adjusted
    # core advantage, i.e. at any given CG a core motif is more conserved
    ax.fill_between(xs, lines["control"], lines["core"], color=CORE_F, alpha=0.85, lw=0, zorder=1)
    adj = float(st.loc["pooled", "cg_adjusted_advantage"]); pa = float(st.loc["pooled", "p_fig3d"])
    ss = st.loc[[str(i) for i in range(10)]]
    print(f"[d] slopes core {st.loc['pooled','slope_core']:+.3f} control {st.loc['pooled','slope_ctrl']:+.3f}; "
          f"CG-adjusted advantage {adj:+.3f} ({fmtp(pa)}); per-set {ss.cg_adjusted_advantage.astype(float).min():+.3f}"
          f" to {ss.cg_adjusted_advantage.astype(float).max():+.3f}, {(ss.p_fig3d.astype(float) < 0.05).sum()}/10 sets P < 0.05")
    chrome(ax, "phyloP100way (per motif)"); title(ax, "Conservation beyond CG")
    ax.set_xlabel("CG content", fontsize=FS_LAB, color=SUB)
    ax.tick_params(axis="x", labelsize=FS_TICK, colors=MUT)
    ax.legend(fontsize=FS_SM, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.14), labelcolor=INK, handlelength=1.1,
              handletextpad=0.35, columnspacing=1.0, borderaxespad=0.0, scatterpoints=1)
    # slope/offset/P are reported in the figure legend, not drawn on the panel

# ── h : is conservation a property of the motif, or of the family that reads it? ─
# NO control arm, by design (user, 2026-07-31). There are only 43 control motifs, but the core
# motifs are read by 286 catalogued TFs, so splitting core-vs-control inside a family compares
# a handful against a handful. The question the core set can answer on its own is whether the
# 7-fold spread in panel c follows the reader family. It does not — and that null is the point:
# it is why the paper treats the 43 motifs individually rather than collapsing them by family.
def panel_h(ax):
    """Constraint of the READER TFs, grouped by the motif family that recruits them.

    Each core motif is labelled with its dominant reader family (median purity 0.93); every
    reader TF of every motif in a family is then pooled. 12 families, 255 reader-TF
    observations — real distributions, not 43 motif-level summaries.

    LOEUF is gnomAD's loss-of-function obs/exp upper bound, the same metric as panel f.
    LOWER = MORE constrained, so the axis is inverted-in-meaning and the most constrained
    families are drawn at the TOP with an explicit "more constrained" arrow. Without that cue
    the panel reads backwards.
    """
    d = pd.read_csv(FAMR)
    order = d.groupby("family").loeuf.median().sort_values(ascending=False)
    lanes = list(order.index)                       # least constrained at the bottom of the list
    overall = d.drop_duplicates("tf").loeuf.median()

    ax.axvline(overall, color=MUT, lw=0.9, ls="--", zorder=2)
    ax.text(overall, len(lanes) - 0.15, " all core readers", ha="left", va="bottom",
            fontsize=FS_SM, color=SUB)

    for yi, f in enumerate(lanes):
        v = d[d.family == f].loeuf.values
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([v.min(), v.max()], [yi, yi], color=BASE, lw=0.8, zorder=3)      # full range
        ax.plot([q1, q3], [yi, yi], color=CORE_E, lw=3.2, solid_capstyle="round",
                zorder=5, alpha=0.95)                                            # IQR
        jit = RNG.normal(0, 0.085, size=len(v)); jit = np.clip(jit, -0.26, 0.26)
        ax.scatter(v, yi + jit, s=6, facecolor=CORE_F, edgecolor=CORE_E, lw=0.35,
                   alpha=0.9, zorder=4, rasterized=True)
        ax.scatter(med, yi, s=22, marker="D", facecolor=INK, edgecolor="white",
                   lw=0.6, zorder=7)                                             # median

    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([f"{f.replace('Nuclear receptor', 'Nuclear rec.')}"
                        f"  {int(d[d.family == f].n_motifs.iloc[0])}m·{(d.family == f).sum()}r"
                        for f in lanes])
    # direction cue lives IN the axis label: LOEUF is inverse and a reader will
    # otherwise assume bigger = more constrained. An arrow above the panel collided
    # with the title at this height.
    ax.set_xlabel("LOEUF of reader TFs\n(lower = more constrained)",
                  fontsize=FS_LAB, color=SUB, linespacing=1.3)
    ax.set_ylabel(""); title(ax, "Reader constraint")
    ax.tick_params(axis="y", labelsize=FS_SM, colors=INK, length=0)
    ax.tick_params(axis="x", labelsize=FS_TICK, colors=MUT, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(BASE); ax.spines[sp].set_linewidth(0.9)

    H, p = kruskal(*[g.loeuf.values for _, g in d.groupby("family")])
    ax.legend(handles=[plt.Line2D([], [], marker="D", ls="none", markerfacecolor=INK,
                                  markeredgecolor="white", markersize=3.4, label="median"),
                       plt.Line2D([], [], color=CORE_E, lw=3.2, solid_capstyle="round",
                                  label="IQR")],
              fontsize=FS_SM, frameon=False, loc="upper right", handlelength=1.1,
              labelcolor=INK, borderaxespad=0.15, handletextpad=0.45, labelspacing=0.35)
    ax.text(0.985, 0.775, fmtp(p), transform=ax.transAxes, ha="right", va="top",
            fontsize=FS_SM, color=INK)
    ax.set_ylim(-1.35, len(lanes) + 0.65)

# ── g : DNA-binding-domain constraint ───────────────────────────────────────────
def panel_g(ax):
    dd = pd.read_csv(DBD)
    d = [dd[dd.group == "universal"].dbd_oe.dropna().values,
         dd[dd.group == "background"].dbd_oe.dropna().values]
    draw_violins(ax, d, [(CORE_E, CORE_F), (CTRL_E, CTRL_F)], ["Core TFs", "Other TFs"],
                 "DBD missense obs/exp", "Domain constraint",
                 [(0, 1, mwp(d[0], d[1]))])

# ── f : protein constraint (grouped rounded bars) ───────────────────────────────
def rbar(ax, x0, w, h, color, r=0.06, ec="none"):
    """Rounded bar from the zero baseline. Handles h < 0 (the synonymous-z control) by
    rounding the far end downward instead of upward."""
    s = 1.0 if h >= 0 else -1.0
    r = min(r, w/2, abs(h)) * s
    verts = [(x0, 0), (x0, h-r), (x0, h-r), (x0+r*s, h), (x0+r*s, h),
             (x0+w-r*s, h), (x0+w-r*s, h), (x0+w, h-r), (x0+w, h-r), (x0+w, 0), (x0, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor=ec,
                           linewidth=EDGE_LW, zorder=3))
def panel_f(ax):
    # The 4th metric is the synonymous-z NEGATIVE CONTROL. Synonymous variants are not under
    # coding selection, so the two arms must land on top of each other there — the flatness of
    # that pair is the point, and it is what rules out a coverage / mutation-rate / gene-length
    # explanation for the other three. Its bars are negative, hence the rbar sign handling.
    cc = pd.read_csv(CC)
    labels = ["pLI", "LOEUF", "mis z", "syn z"]
    uni, bg, ps = list(cc["median_universal"]), list(cc["median_background"]), list(cc["p_value"])
    n = len(labels); x = np.arange(n); w = 0.34
    # Four groups in a 45.75 mm panel leaves no room for a "P = " prefix on every bracket, so
    # the prefix is dropped here and declared once in the legend/caption. Heights alternate so
    # neighbouring labels, which are wider than one x unit, never sit at the same y.
    for xi, (a, b, p) in enumerate(zip(uni, bg, ps)):
        rbar(ax, xi - w - 0.02, w, a, CORE_F, ec=CORE_E)
        rbar(ax, xi + 0.02, w, b, CTRL_F, ec=CTRL_E)
        top = max(a, b)
        # an all-negative pair (the synonymous-z control) gets its bracket just above the
        # zero line, next to its own bars, instead of floating where positive bars would end
        yb = 0.10 if top < 0 else top + (0.34 if xi % 2 else 0.12)
        ax.plot([xi - w/2 - 0.02, xi + w/2 + 0.02], [yb, yb], lw=0.7, color=MUT, clip_on=False)
        ax.text(xi, yb + 0.015, fmtp(p).replace("P = ", ""), ha="center", va="bottom",
                fontsize=FS_SM, color=INK)
    ax.axhline(0, color=BASE, lw=0.9, zorder=2)
    from matplotlib.patches import Patch
    # Legend below the x-axis, clear of the bars entirely. Upper-left sat on the pLI pair and
    # upper-right crowded the missense-z bracket; anchoring above the axes hit the panel title.
    ax.legend(handles=[Patch(facecolor=CORE_F, edgecolor=CORE_E, lw=EDGE_LW, label="Core TFs"),
                       Patch(facecolor=CTRL_F, edgecolor=CTRL_E, lw=EDGE_LW, label="Other TFs")],
              fontsize=FS_SM, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.13), labelcolor=INK, handlelength=1.0,
              columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    chrome(ax, "constraint metric"); title(ax, "Protein constraint")
    ax.tick_params(axis="x", labelsize=FS_SM)
    ax.set_ylim(-0.42, 1.62); ax.set_xlim(-0.6, n - 0.4)

PANELS = [("c", panel_c), ("d", panel_d), ("e", panel_e), ("f", panel_f), ("g", panel_g), ("h", panel_h)]   # repository scope: panels c-h (a, b summarise the MEME files)
NAMES = {"a": "fig3a_cg", "b": "fig3b_ic", "c": "fig3c_conservation",
         "d": "fig3d_cg_adjusted", "e": "fig3e_biosample_advantage",
         "f": "fig3f_protein_constraint", "g": "fig3g_dbd", "h": "fig3h_family_conservation"}

# transparent=True on every write: figures must never carry a white background.
for lt, fn in PANELS:
    fig, ax = plt.subplots(figsize=PANEL); fn(ax); fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{SD}/{NAMES[lt]}.{ext}", dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)

# ── the assembled figure: EXACTLY the 183 mm double-column width ───────────────────────────
# This is the deliverable, not a preview: one sheet carrying all six panels, in .svg as well
# as .pdf/.png. Width is fixed at 183 mm and the height follows from the 2x3 grid.
#
# ⚠ NO bbox_inches="tight" here. That crops to the drawn content and silently returns a sheet
# some other width, which is exactly what must not happen when the figure is specified to fill
# the column. constrained_layout packs the panels inside the fixed canvas instead of resizing
# it, so the saved width is 183 mm by construction. The per-panel files above still use
# bbox_inches="tight" because there the panel, not the sheet, is the unit.
MM = 1 / 25.4
FIG_W = 183 * MM                      # Nature double column
FIG_H = FIG_W * 2 / 4 * (PANEL[1] / PANEL[0])   # keep the panel aspect over a 2x4 grid

# Row 2 uses UNEVEN widths so panel h can carry 12 family lanes. Row 1 stays 4 equal panels.
# matplotlib applies one width_ratios per GridSpec, so each row gets its own nested spec.
import sys; sys.exit(0)   # the assembled 2x4 sheet needs panels a and b; the six panel files above are the deliverable here
figC = plt.figure(figsize=(FIG_W, FIG_H), constrained_layout=True)
figC.set_constrained_layout_pads(w_pad=0.055, h_pad=0.055, wspace=0.045, hspace=0.075)
outer = figC.add_gridspec(2, 1)
ROW2_W = [0.80, 0.95, 0.80, 1.45]     # e, f, g, h — sums to 4.0, so the sheet width is unchanged
top = outer[0].subgridspec(1, 4)
bot = outer[1].subgridspec(1, 4, width_ratios=ROW2_W)
axesC = ([figC.add_subplot(top[0, i]) for i in range(4)] +
         [figC.add_subplot(bot[0, i]) for i in range(4)])
for (lt, fn), ax in zip(PANELS, axesC):
    fn(ax)
    ax.text(-0.155, 1.115, lt, transform=ax.transAxes, fontsize=8,
            fontweight="bold", color=INK, va="top", ha="left")
for ext in ("pdf", "png", "svg"):
    figC.savefig(f"{SD}/fig3_full.{ext}", dpi=600, transparent=True)
plt.close(figC)

print(f"wrote fig3 panels a–h (pdf/png/svg, transparent)")
print(f"wrote fig3_full.svg/.pdf/.png — assembled sheet at "
      f"{FIG_W/MM:.0f} x {FIG_H/MM:.1f} mm")
