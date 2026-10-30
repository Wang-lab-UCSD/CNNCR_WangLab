#!/usr/bin/env python
"""
09z3 -- figure-ready version of the cross-line reader-switch display (user spec 2026-08-27).

Eight motifs with the best available switch pair: the seven reciprocal pairs, then the
widest-gap non-reciprocal pair (avoids showing E2F1/E2F4 twice via both M36 and M20).

  FIGURE 1  switch_figready_pair      one row-block per motif, TWO panels each:
                                      left  = log2(K562/HEK293) of each reader's smoothed
                                              |effect| curve (the ratio view),
                                      right = the raw smoothed |effect| in both lines on a
                                              log axis (the magnitude view).
                                      One distinct colour per reader (constant across both
                                      panels); in the magnitude panel K562 is the dark solid
                                      curve and HEK293 the light dashed one.
  FIGURE 3  switch_ratio_main         MAIN TEXT (Fig. 5f), user spec 2026-09-02: the ratio view
                                      ONLY, and only the four widest reciprocal pairs, in one row.
  FIGURE 4  switch_magnitude_si       SUPPLEMENTARY: the magnitude view for all eight pairs, which
                                      is what the main panel gives up.
  FIGURE 2  switch_figready_summary   ratio only.  a: every ChIP-verified non-reader TF's own
                                      cross-line contrast (per-TF median log2 K562/HEK293 over
                                      the motif gene sets), one dot per TF, readers of the 8
                                      pairs overlaid at their percentile.  b: the background
                                      distribution with the 8 reader pairs drawn as coloured
                                      segments spanning their two readers' contrasts.

Runs entirely from the 09z plot cache + the 09x per-factor CSV -- seconds, no screen loading.

Usage:  python 09z3_figure_ready.py top90chip

Reads   results/prof_plan/switch_pair_figures_{tag}_plotcache.pkl   (09z curves, meaneffect order)
        results/prof_plan/switch_pair_per_motif_{tag}.csv           (09z pair stats + null P)
        results/prof_plan/cross_line_control_{tag}.csv              (09x per-factor statistics)
        results/prof_plan/reader_target_counts_{tag}.csv            (curve-label gene counts)
Writes  results/prof_plan/fig_reader_curves/switch_ratio_main_{tag}.{pdf,png}      <- Fig. 5f
        results/prof_plan/fig_reader_curves/switch_magnitude_si_{tag}.{pdf,png}    <- SI
        results/prof_plan/fig_reader_curves/switch_figready_pair_{tag}.{pdf,png}
        results/prof_plan/fig_reader_curves/switch_figready_summary_{tag}.{pdf,png}
        results/prof_plan/switch_figready_pairs_{tag}.csv
"""
import os, sys, pickle, string
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TAG = sys.argv[1] if len(sys.argv) > 1 else "top90chip"
N_MOTIFS = 8
N_MAIN = 4        # motifs carried by the main-text ratio panel (Fig. 5f)

ROOT = ENHANCER_ROOT
CHIP = os.path.join(ROOT, "multi_agent/chipseq_enrichment")
OUT = os.path.join(CHIP, "results/prof_plan")
FIG = os.path.join(OUT, "fig_reader_curves")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
    "axes.labelsize": 6, "legend.fontsize": 5,
})
INK, SUB, MUT = "#0b0b0b", "#52514e", "#7d7b76"
# one distinct colour per reader of the pair, constant across the ratio and magnitude panels
CA, CB = "#2166ac", "#d55e00"            # K562-leaning reader, HEK293-leaning reader
CA_LT, CB_LT = "#93bfe0", "#f0b283"      # the same readers' HEK293 curves in the magnitude panel
PAIR_COLS = ["#2166ac", "#d55e00", "#117733", "#882255",
             "#0e7c7b", "#996f00", "#5e3c99", "#c51b7d"]      # summary figure, one per motif
HEAD = lambda m: "Motif " + m.split("_")[1]
SHORT = lambda m: m.replace("universal_", "M")

# ============================================================================ inputs
S = pd.read_csv(f"{OUT}/switch_pair_per_motif_{TAG}.csv")
with open(f"{OUT}/switch_pair_figures_{TAG}_plotcache.pkl", "rb") as fh:
    PANEL = pickle.load(fh)["PANEL"]
D = pd.read_csv(f"{OUT}/cross_line_control_{TAG}.csv")
A = D[D.group != "reader"]
NOWN = pd.read_csv(f"{OUT}/reader_target_counts_{TAG}.csv") \
         .set_index(["motif", "factor"]).n_target.to_dict()
lab = lambda m, f: f"{f} ({NOWN[(m, f)]})" if (m, f) in NOWN else f

# the eight motifs: reciprocal pairs by gap, then the widest non-reciprocal
S = S[S.motif.isin(PANEL)]
pick = pd.concat([S[S.reciprocal].sort_values("gap", ascending=False),
                  S[~S.reciprocal].sort_values("gap", ascending=False)]).head(N_MOTIFS)
pick = pick.sort_values("gap", ascending=False).reset_index(drop=True)
pick.to_csv(f"{OUT}/switch_figready_pairs_{TAG}.csv", index=False)

print(f"=== the {len(pick)} displayed motifs ===")
for r in pick.itertuples(index=False):
    print(f"  {SHORT(r.motif):5} {r.reader_K:8}/{r.reader_H:8} gap {2**r.gap:5.2f}x "
          f"recip {'yes' if r.reciprocal else 'no ':3} null P = {r.P:.3f}")


def lg(yK, yH):
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.log2(np.asarray(yK, float) / np.asarray(yH, float))
    v[~np.isfinite(v)] = np.nan
    return v


def strip_axes(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=SUB, labelcolor=INK, pad=1.5)


# ============================================================================ FIGURE 1
NBLK = 2                                    # motif blocks per row, each block = 2 panels
nrow = int(np.ceil(len(pick) / NBLK))
fig = plt.figure(figsize=(7.09, 1.42 * nrow + 0.42), facecolor="white")
gs = fig.add_gridspec(nrow, 2 * NBLK, hspace=1.05, wspace=0.52)

for i, r in enumerate(pick.itertuples(index=False)):
    row, blk = i // NBLK, i % NBLK
    P = PANEL[r.motif]
    x = np.arange(1, P["n"] + 1)
    yKa, yHa = P["readers"][r.reader_K]
    yKb, yHb = P["readers"][r.reader_H]

    # ---- ratio panel
    ax = fig.add_subplot(gs[row, 2 * blk])
    ax.axhline(0, color=MUT, lw=0.5, zorder=1)
    va, vb = lg(yKa, yHa), lg(yKb, yHb)
    ax.fill_between(x, va, vb, color=INK, alpha=0.10, lw=0, zorder=2)
    ax.plot(x, va, color=CA, lw=0.6, zorder=3, solid_capstyle="round")
    ax.plot(x, vb, color=CB, lw=0.6, zorder=3, solid_capstyle="round")
    lim = max(1.0, 1.12 * np.nanmax(np.abs(np.nanpercentile(
        np.concatenate([va, vb]), [1, 99]))))
    ax.set_xlim(1, P["n"])
    ax.set_ylim(-lim, lim)
    strip_axes(ax)
    BOX = dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.78)
    ax.text(0.02, 0.965, lab(r.motif, r.reader_K), transform=ax.transAxes,
            fontsize=4.6, color=CA, va="top", ha="left", bbox=BOX, zorder=5)
    ax.text(0.02, 0.035, lab(r.motif, r.reader_H), transform=ax.transAxes,
            fontsize=4.6, color=CB, va="bottom", ha="left", bbox=BOX, zorder=5)
    ax.text(-0.34, 1.16, string.ascii_lowercase[i], transform=ax.transAxes, fontsize=7,
            fontweight="bold", color=INK, va="top", ha="left")
    ax.text(-0.18, 1.16, HEAD(r.motif), transform=ax.transAxes, fontsize=7,
            fontweight="bold", color=INK, va="top", ha="left")
    if blk == 0:
        ax.set_ylabel("log$_2$ K562 / HEK293", color=INK, labelpad=1.5)
    if row == nrow - 1:
        ax.set_xlabel("genes by mean reader effect", color=INK, labelpad=1.5, fontsize=5)

    # ---- magnitude panel
    ax = fig.add_subplot(gs[row, 2 * blk + 1])
    ax.plot(x, yKa, color=CA, lw=0.55, ls="-", zorder=3)
    ax.plot(x, yHa, color=CA_LT, lw=0.55, ls=(0, (2.2, 1.2)), zorder=3)
    ax.plot(x, yKb, color=CB, lw=0.55, ls="-", zorder=3)
    ax.plot(x, yHb, color=CB_LT, lw=0.55, ls=(0, (2.2, 1.2)), zorder=3)
    ax.set_yscale("log")
    ally = np.concatenate([yKa, yHa, yKb, yHb]).astype(float)
    ally = ally[np.isfinite(ally) & (ally > 0)]
    ax.set_ylim(np.percentile(ally, 0.5) * 0.8, np.percentile(ally, 99.5) * 1.3)
    ax.set_xlim(1, P["n"])
    strip_axes(ax)
    if row == nrow - 1:
        ax.set_xlabel("genes by mean reader effect", color=INK, labelpad=1.5, fontsize=5)
    if blk == 0:
        ax.set_ylabel("|effect| (QN)", color=INK, labelpad=1.5)

LEG1 = [Line2D([], [], color=CA, lw=1.0, label="K562-leaning reader"),
        Line2D([], [], color=CB, lw=1.0, label="HEK293-leaning reader"),
        Line2D([], [], color=INK, lw=0.8, ls="-", label="K562 screen (magnitude)"),
        Line2D([], [], color=MUT, lw=0.8, ls=(0, (2.2, 1.2)),
               label="HEK293 screen (magnitude, light)")]
fig.legend(handles=LEG1, frameon=False, ncol=4, loc="lower center",
           bbox_to_anchor=(0.5, -0.014), columnspacing=1.4, handlelength=1.5)
for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/switch_figready_pair_{TAG}.{ext}", dpi=450, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print(f"\nwrote -> {FIG}/switch_figready_pair_{TAG}.pdf/.png")

# ============================================================================ FIGURE 3 (main text)
# Ratio view only, four widest reciprocal pairs, one row.  No internal panel letters: this
# sheet is Figure 5f and the figure-level letter is added by the Figure 5 assembler.
main = pick.head(N_MAIN).reset_index(drop=True)
print(f"\n=== Fig. 5f, the {len(main)} widest reciprocal pairs ===")
for r in main.itertuples(index=False):
    print(f"  {SHORT(r.motif):5} {r.reader_K:8}/{r.reader_H:8} gap {2 ** r.gap:5.2f}x  n = {r.n_genes:,}")

fig = plt.figure(figsize=(7.09, 1.62), facecolor="white")
gs = fig.add_gridspec(1, len(main), wspace=0.30)
for i, r in enumerate(main.itertuples(index=False)):
    P = PANEL[r.motif]
    x = np.arange(1, P["n"] + 1)
    va = lg(*P["readers"][r.reader_K])
    vb = lg(*P["readers"][r.reader_H])
    ax = fig.add_subplot(gs[0, i])
    ax.axhline(0, color=MUT, lw=0.5, zorder=1)
    ax.fill_between(x, va, vb, color=INK, alpha=0.10, lw=0, zorder=2)
    ax.plot(x, va, color=CA, lw=0.6, zorder=3, solid_capstyle="round")
    ax.plot(x, vb, color=CB, lw=0.6, zorder=3, solid_capstyle="round")
    lim = max(1.0, 1.12 * np.nanmax(np.abs(np.nanpercentile(
        np.concatenate([va, vb]), [1, 99]))))
    ax.set_xlim(1, P["n"])
    ax.set_ylim(-lim, lim)
    strip_axes(ax)
    BOX = dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.78)
    ax.text(0.02, 0.965, lab(r.motif, r.reader_K), transform=ax.transAxes,
            fontsize=4.6, color=CA, va="top", ha="left", bbox=BOX, zorder=5)
    ax.text(0.02, 0.035, lab(r.motif, r.reader_H), transform=ax.transAxes,
            fontsize=4.6, color=CB, va="bottom", ha="left", bbox=BOX, zorder=5)
    ax.text(0.0, 1.20, HEAD(r.motif), transform=ax.transAxes, fontsize=6.5,
            fontweight="bold", color=INK, va="top", ha="left")
    ax.text(1.0, 1.20, f"{2 ** r.gap:.1f}× apart", transform=ax.transAxes, fontsize=4.8,
            color=SUB, va="top", ha="right")
    if i == 0:
        ax.set_ylabel("log$_2$ K562 / HEK293", color=INK, labelpad=1.5)
    ax.set_xlabel("genes by mean reader effect", color=INK, labelpad=1.5, fontsize=5)
fig.legend(handles=[Line2D([], [], color=CA, lw=1.0, label="K562-leaning reader"),
                    Line2D([], [], color=CB, lw=1.0, label="HEK293T-leaning reader")],
           frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.185),
           columnspacing=1.6, handlelength=1.5, fontsize=5)
for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/switch_ratio_main_{TAG}.{ext}", dpi=450, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print(f"wrote -> {FIG}/switch_ratio_main_{TAG}.pdf/.png")

# ============================================================================ FIGURE 4 (supplementary)
# The magnitude view the main panel gives up, for all eight pairs.
NB = 4
nrow = int(np.ceil(len(pick) / NB))
fig = plt.figure(figsize=(7.09, 1.52 * nrow + 0.34), facecolor="white")
gs = fig.add_gridspec(nrow, NB, hspace=0.95, wspace=0.42)
for i, r in enumerate(pick.itertuples(index=False)):
    P = PANEL[r.motif]
    x = np.arange(1, P["n"] + 1)
    yKa, yHa = P["readers"][r.reader_K]
    yKb, yHb = P["readers"][r.reader_H]
    ax = fig.add_subplot(gs[i // NB, i % NB])
    ax.plot(x, yKa, color=CA, lw=0.55, ls="-", zorder=3)
    ax.plot(x, yHa, color=CA_LT, lw=0.55, ls=(0, (2.2, 1.2)), zorder=3)
    ax.plot(x, yKb, color=CB, lw=0.55, ls="-", zorder=3)
    ax.plot(x, yHb, color=CB_LT, lw=0.55, ls=(0, (2.2, 1.2)), zorder=3)
    ax.set_yscale("log")
    ally = np.concatenate([yKa, yHa, yKb, yHb]).astype(float)
    ally = ally[np.isfinite(ally) & (ally > 0)]
    ax.set_ylim(np.percentile(ally, 0.5) * 0.8, np.percentile(ally, 99.5) * 1.3)
    ax.set_xlim(1, P["n"])
    strip_axes(ax)
    ax.text(-0.30, 1.20, string.ascii_lowercase[i], transform=ax.transAxes, fontsize=7,
            fontweight="bold", color=INK, va="top", ha="left")
    ax.text(-0.13, 1.20, HEAD(r.motif), transform=ax.transAxes, fontsize=6.5,
            fontweight="bold", color=INK, va="top", ha="left")
    ax.text(0.02, 0.965, lab(r.motif, r.reader_K), transform=ax.transAxes,
            fontsize=4.4, color=CA, va="top", ha="left", zorder=5)
    ax.text(0.02, 0.035, lab(r.motif, r.reader_H), transform=ax.transAxes,
            fontsize=4.4, color=CB, va="bottom", ha="left", zorder=5)
    if i % NB == 0:
        ax.set_ylabel("|effect| (QN)", color=INK, labelpad=1.5)
    if i // NB == nrow - 1:
        ax.set_xlabel("genes by mean reader effect", color=INK, labelpad=1.5, fontsize=5)
fig.legend(handles=LEG1, frameon=False, ncol=4, loc="lower center",
           bbox_to_anchor=(0.5, -0.045), columnspacing=1.4, handlelength=1.5)
for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/switch_magnitude_si_{TAG}.{ext}", dpi=450, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print(f"wrote -> {FIG}/switch_magnitude_si_{TAG}.pdf/.png")

# ============================================================================ FIGURE 2 (ratio only)
cf = A.groupby("factor").log2_target.median()          # one signed contrast per background TF
cf_sorted = np.sort(cf.to_numpy())
n_c = len(cf_sorted)

readers = []                                            # the 16 reader records of the 8 pairs
for j, r in enumerate(pick.itertuples(index=False)):
    for f, v in ((r.reader_K, r.log2_K), (r.reader_H, r.log2_H)):
        readers.append(dict(motif=r.motif, factor=f, val=v, col=PAIR_COLS[j]))
RV = pd.DataFrame(readers)
pctl = lambda v: 100.0 * np.searchsorted(cf_sorted, v) / n_c

mwu = stats.mannwhitneyu(RV.val.abs(), cf.abs()).pvalue
print("\n=== summary (ratio only) ===")
print(f"  background TFs n = {n_c}, median |log2| {2**cf.abs().median():.2f}x; "
      f"displayed readers n = {len(RV)}, median {2**RV.val.abs().median():.2f}x, MWU P = {mwu:.3f}")
print(f"  pair gaps vs the selection-matched null: "
      f"{int((pick.pctile > 50).sum())}/{len(pick)} above the null median, "
      f"{int((pick.P < 0.05).sum())}/{len(pick)} at P < 0.05")

fig = plt.figure(figsize=(7.09, 2.75), facecolor="white")
gs = fig.add_gridspec(1, 2, wspace=0.42, width_ratios=[1.0, 1.15])

# a -- every background TF individually, readers overlaid at their percentile
ax = fig.add_subplot(gs[0, 0])
ax.axhline(0, color=MUT, lw=0.5, zorder=1)
ax.plot(np.linspace(0, 100, n_c), cf_sorted, ".", color=MUT, ms=1.1, mew=0,
        alpha=0.7, zorder=2, rasterized=True)
RV["px"] = RV.val.map(pctl)
placed = []                                             # greedy label stagger, no overlaps
for r in RV.sort_values("px").itertuples(index=False):
    ax.plot(r.px, r.val, "o", color=r.col, ms=2.8, mew=0.4, mec="white", zorder=4)
    d = 1.0 if r.val > 0 else -1.0
    ly = r.val + d * 0.14
    while any(abs(r.px - qx) < 13 and abs(ly - qy) < 0.20 for qx, qy in placed):
        ly += d * 0.20
    placed.append((r.px, ly))
    ax.annotate(r.factor, (r.px, r.val), xytext=(r.px - 1.5 if r.val > 0 else r.px + 1.5, ly),
                textcoords="data", fontsize=4.2, color=r.col,
                ha="right" if r.val > 0 else "left", va="center")
ax.set_xlim(-2, 102)
ax.set_xlabel("percentile among background TFs", color=INK, labelpad=1.5)
ax.set_ylabel("log$_2$ K562 / HEK293", color=INK, labelpad=1.5)
strip_axes(ax)
ax.text(-0.17, 1.16, "a", transform=ax.transAxes, fontsize=7, fontweight="bold",
        color=INK, va="top", ha="left")
ax.text(-0.09, 1.16, "Every TF", transform=ax.transAxes, fontsize=7, fontweight="bold",
        color=INK, va="top", ha="left")
ax.text(-0.09, 1.055, f"|log$_2$| readers vs background TFs, MWU P = {mwu:.2f}",
        transform=ax.transAxes, fontsize=4.8, color=SUB, va="top", ha="left")

# b -- the background distribution with the 8 pairs overlaid as segments
gsb = gs[0, 1].subgridspec(2, 1, hspace=0.10, height_ratios=[1.15, 1.0])
axp = fig.add_subplot(gsb[0])                           # pair segments
axh = fig.add_subplot(gsb[1], sharex=axp)               # background distribution
bins = np.linspace(np.percentile(cf, 0.5), np.percentile(cf, 99.5), 46)
axh.hist(cf, bins=bins, color=MUT, alpha=0.5, lw=0, density=True)
axh.axvline(0, color=MUT, lw=0.5)
axh.set_xlabel("log$_2$ K562 / HEK293", color=INK, labelpad=1.5)
axh.set_ylabel("density", color=INK, labelpad=1.5)
strip_axes(axh)

for j, r in enumerate(pick.sort_values("gap").itertuples(index=False)):
    axp.plot([r.log2_H, r.log2_K], [j, j], color=PAIR_COLS[list(pick.motif).index(r.motif)],
             lw=1.0, zorder=3, solid_capstyle="round")
    axp.plot([r.log2_H, r.log2_K], [j, j], "o",
             color=PAIR_COLS[list(pick.motif).index(r.motif)], ms=2.6, mew=0, zorder=4)
    axp.annotate(f"{SHORT(r.motif)}  {r.reader_K}/{r.reader_H}  {2**r.gap:.1f}x",
                 (max(r.log2_H, r.log2_K), j), xytext=(4, 0), textcoords="offset points",
                 fontsize=4.4, color=INK, va="center", ha="left", annotation_clip=False)
axp.axvline(0, color=MUT, lw=0.5)
axp.set_ylim(-0.8, len(pick) - 0.2)
axp.set_yticks([])
axp.spines["left"].set_visible(False)
plt.setp(axp.get_xticklabels(), visible=False)
axp.tick_params(axis="x", length=0)
strip_axes(axp)
axp.text(-0.10, 1.30, "b", transform=axp.transAxes, fontsize=7, fontweight="bold",
         color=INK, va="top", ha="left")
axp.text(-0.03, 1.30, "Pairs vs background", transform=axp.transAxes, fontsize=7,
         fontweight="bold", color=INK, va="top", ha="left")
axp.text(-0.03, 1.13,
         f"{int((pick.pctile > 50).sum())} of {len(pick)} pairs above the matched null median, "
         f"{int((pick.P < 0.05).sum())} of {len(pick)} at P < 0.05",
         transform=axp.transAxes, fontsize=4.8, color=SUB, va="top", ha="left")

for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/switch_figready_summary_{TAG}.{ext}", dpi=450, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print(f"wrote -> {FIG}/switch_figready_summary_{TAG}.pdf/.png")
print(f"pair table -> {OUT}/switch_figready_pairs_{TAG}.csv")
