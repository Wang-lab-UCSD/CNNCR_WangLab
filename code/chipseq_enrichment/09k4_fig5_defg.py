#!/usr/bin/env python
"""
09k4 -- Figure 5 d-g on ONE publication sheet (user spec 2026-09-03).

183 mm wide, drawn at final size so every point size on screen is the printed size.

  row 1   d  four representative reader-curve panels (one dominant / two co-dominant /
             graded / interchangeable), leader orange, runner-up blue, rest grey, with a
             shaded 10-90 percentile envelope over the other readers, a marginal giving
             every reader's median effect with its bootstrap interval, and a strip above each
             panel coloured gene by gene for whichever of the top two readers is ahead
          e  every motif x cell line panel: readers as a fraction of the leader (left) and
             BOTH leads on the right, the leader over the other readers (filled) and over
             the runner-up alone (open), which is what separates a single dominant reader
             from a co-dominant pair
  row 2   f  the four widest reciprocal cross-line pairs as a 2 x 2 square, ratio view, each
             with its two median ratios drawn as rules and a marginal density of the two
             plotted curves
          g  each reader's switch magnitude against the 300 ChIP-matched background TFs,
             with a percentile axis read off the background distribution itself

Colour grammar.  Panels d,e colour by RANK within a cell line (orange leader, blue runner-up,
grey rest), in the Fig 4 core/JASPAR/control hues.  Panels f,g colour by CELL LINE in the
Fig 3 pair (teal = K562-leaning reader, violet = HEK293T-leaning reader) so they cannot be
confused with the rank colours of d,e.

Type.  Liberation Sans (metric clone of Arial) from paper_figures/fonts/, embedded as Type 42.
Tiers: 8 pt panel letters, 7 pt panel headers (the only bold text), 6 pt axis labels,
5.5 pt ticks, 5 pt legends and annotations.  Nothing below 5 pt.  No mathtext anywhere, so the
SVG carries a single font.

Inputs (all cached, seconds to run, no screen load):
  results/prof_plan/union_curves_8motifs_{tag}_plotcache.pkl     09k  (panel d curves)
  results/prof_plan/dominant_reader_{tag}.csv                    09q  (panel e)
  results/prof_plan/switch_pair_figures_{tag}_plotcache.pkl      09z  (panel f curves)
  results/prof_plan/switch_figready_pairs_{tag}.csv              09z3 (the 8 pairs, ordered)
  results/prof_plan/reader_target_counts_{tag}.csv               09z  (gene counts per reader)
  results/prof_plan/cross_line_pergene_{tag}.csv                 09z5 (panel g, design B:
                                                                       group == "chip")
Outputs:
  results/prof_plan/fig_reader_curves/fig5_defg_{tag}.{pdf,png,svg}
  copied to paper_figures/fig5/fig5dg.{pdf,png,svg}

Usage:  python 09k4_fig5_defg.py [top90chip]
"""
import os
import sys
import pickle
import shutil

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

TAG = sys.argv[1] if len(sys.argv) > 1 else "top90chip"

ROOT = "/stg3/data1/sam/enhancer_prediction"
CHIP = os.path.join(ROOT, "multi_agent/chipseq_enrichment")
OUT = os.path.join(CHIP, "results/prof_plan")
FIG = os.path.join(OUT, "fig_reader_curves")
DEST = os.path.join(ROOT, "paper_figures/fig5")
FONTDIR = os.path.join(ROOT, "paper_figures/fonts")

# ------------------------------------------------------------------ type
for _f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf",
           "LiberationSans-Italic.ttf", "LiberationSans-BoldItalic.ttf"):
    _p = os.path.join(FONTDIR, _f)
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
FAMILY = ("Liberation Sans" if any(f.name == "Liberation Sans" for f in fm.fontManager.ttflist)
          else "Arial")
FS_LET, FS_HEAD, FS_LAB, FS_TICK, FS_SM = 8, 7, 6, 5.5, 5
plt.rcParams.update({
    "font.family": FAMILY, "font.size": FS_LAB,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "axes.labelsize": FS_LAB, "legend.fontsize": FS_SM,
    "axes.unicode_minus": False,
})
INK, SUB, MUT, BG = "#0b0b0b", "#52514e", "#8f8d88", "#c3c2b7"   # Fig 4 ink/greys
LEAD, SECOND, REST = "#eb6834", "#5b8db8", "#b8b6ae"          # d,e: Fig 4 core / JASPAR / control
K_COL, H_COL = "#2f9c9c", "#8b73bd"                            # f,g: Fig 3 teal = K562-leaning, Fig 3 violet = HEK293T-leaning
SHADE = "#e8e7e1"                                              # Fig 4 grid grey
SHORT = lambda m: m.replace("universal_", "M")
FAMILY_NAME = {"universal_14": "ETS", "universal_1": "GC box",
               "universal_29": "E box", "universal_36": "E2F"}   # as named in the text

roll = lambda y, w: pd.Series(y).rolling(w, center=True,
                                         min_periods=max(3, w // 4)).median().to_numpy()


def strip(ax, left=True):
    for sp in ("top", "right") + (() if left else ("left",)):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=SUB, labelcolor=INK, pad=1.5)


def logticks(ax):
    """decimal labels on a log axis, majors only: no mathtext, so no 0.7x exponents"""
    lo, hi = ax.get_ylim()
    cand = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    tk = [t for t in cand if lo <= t <= hi]
    ax.set_yticks(tk)
    ax.set_yticklabels([f"{t:g}" for t in tk])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def head(fig, x, y, letter, text):
    fig.text(x, y, letter, fontsize=FS_LET, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(x + 0.0165, y, text, fontsize=FS_HEAD, fontweight="bold", color=INK,
             va="top", ha="left")


# ------------------------------------------------------------------ data: d, e
PICKS = [
    ("universal_14", "K562", "one dominant reader"),
    ("universal_1", "HEK293", "two co-dominant readers"),
    ("universal_29", "K562", "graded, no clear leader"),
    ("universal_36", "K562", "readers interchangeable"),
]
with open(f"{OUT}/union_curves_8motifs_{TAG}_plotcache.pkl", "rb") as fh:
    cache = {(p["motif"], p["cell"]): p for p in pickle.load(fh)}
D = pd.read_csv(f"{OUT}/dominant_reader_{TAG}.csv")
D["profile"] = D.rel_profile.map(lambda s: [float(v) for v in s.split(",")])
D["names"] = D.order.map(lambda s: s.split(","))
D = D.sort_values("lead_over_rest", ascending=False).reset_index(drop=True)
rank = {(r.motif, r.cell): r for r in D.itertuples(index=False)}
missing = [(m, c) for m, c, _ in PICKS if (m, c) not in cache]
if missing:
    raise SystemExit(f"panel not in the 09k cache: {missing}")
print(f"[{TAG}] e: {len(D)} panels, lead over others "
      f"{D.lead_over_rest.min():.2f} to {D.lead_over_rest.max():.2f}")

# per-gene effects behind panel d, taken from the plot cache so everything below is in the same
# gene order the panel draws.  Checked against the raw table reader_gene_values_{tag}.csv.gz:
# identical medians and identical win shares (M14 70.9% of 1,676 genes, as in the text).
RNG = np.random.default_rng(0)
DSTAT = {}
for m, c, _ in PICKS:
    p = cache[(m, c)]
    order = [nm for nm in rank[(m, c)].names if nm in p["curves"]]
    med, lo, hi = {}, {}, {}
    for nm in order:
        v = np.asarray(p["curves"][nm], float)
        v = v[np.isfinite(v)]
        b = np.median(RNG.choice(v, (400, len(v)), replace=True), axis=1)
        med[nm], lo[nm], hi[nm] = np.median(v), *np.percentile(b, [2.5, 97.5])
    a = np.asarray(p["curves"][order[0]], float)
    b2 = np.asarray(p["curves"][order[1]], float)
    ok = np.isfinite(a) & np.isfinite(b2)
    lead_wins = (a > b2)                     # per gene, in the panel's x order
    win = float(lead_wins[ok].mean())
    P = stats.wilcoxon(a[ok], b2[ok]).pvalue
    print("hello world")
    # ---- trace of every intermediate quantity behind P (2026-10-05, written for the user's audit) ----
    _tr = open(f"{OUT}/wilcoxon_trace_{TAG}_{SHORT(m)}_{c}.txt", "w")
    _d = a[ok] - b2[ok]
    _nz = _d != 0
    _dn = _d[_nz]
    _r = stats.rankdata(np.abs(_dn))
    _Wp, _Wm = float(_r[_dn > 0].sum()), float(_r[_dn < 0].sum())
    _n = int(_nz.sum())
    _mu = _n * (_n + 1) / 4
    _ties = np.unique(np.abs(_dn), return_counts=True)[1]
    _sig = np.sqrt(_n * (_n + 1) * (2 * _n + 1) / 24 - ((_ties ** 3 - _ties).sum()) / 48)
    _W = min(_Wp, _Wm)
    _z = (_W - _mu) / _sig                       # scipy 'wilcox' / approx: no continuity correction here
    _z_cc = (_W - _mu + 0.5) / _sig              # with continuity correction (scipy's default, correction=True in old versions)
    _P_norm = 2 * stats.norm.sf(abs(_z)); _P_cc = 2 * stats.norm.sf(abs(_z_cc))
    _tr.write(f"hello world\n\nPanel: motif {m} ({SHORT(m)}), cell line {c}\n")
    _tr.write(f"leader = {order[0]}, runner-up = {order[1]}\n")
    _tr.write(f"genes in panel = {len(a)}; genes with finite values for both readers (n used) = {int(ok.sum())}\n")
    _tr.write(f"median |effect| leader = {np.median(a[ok]):.6f}, runner-up = {np.median(b2[ok]):.6f}, ratio = {np.median(a[ok])/np.median(b2[ok]):.4f}\n")
    _tr.write(f"genes where leader > runner-up = {int(lead_wins[ok].sum())} of {int(ok.sum())} = {100*win:.2f}%\n")
    _tr.write(f"genes where leader < runner-up = {int((a[ok] < b2[ok]).sum())}; ties (difference exactly 0) = {int((~_nz).sum())}\n")
    _tr.write(f"per-gene difference d = leader - runner-up: mean {np.mean(_d):.6f}, median {np.median(_d):.6f}, sd {np.std(_d, ddof=1):.6f}\n")
    _tr.write(f"\nWilcoxon signed-rank (paired, two-sided), ties dropped: n = {_n}\n")
    _tr.write(f"  sum of ranks of positive differences W+ = {_Wp:.1f}\n  sum of ranks of negative differences W- = {_Wm:.1f}\n")
    _tr.write(f"  check: W+ + W- = {_Wp+_Wm:.1f} = n(n+1)/2 = {_n*(_n+1)/2:.1f}\n")
    _tr.write(f"  test statistic W = min(W+, W-) = {_W:.1f}\n")
    _tr.write(f"  null mean  mu = n(n+1)/4 = {_mu:.1f}\n")
    _tr.write(f"  null s.d. sigma = sqrt(n(n+1)(2n+1)/24 - tie correction) = {_sig:.3f}\n")
    _tr.write(f"  z = (W - mu) / sigma = {_z:.3f}   (with continuity correction: {_z_cc:.3f})\n")
    _tr.write(f"  two-sided P from the normal approximation = {_P_norm:.3e}   (with continuity correction: {_P_cc:.3e})\n")
    _tr.write(f"  scipy.stats.wilcoxon(a, b).pvalue = {P:.3e}   (scipy {__import__('scipy').__version__})\n")
    _tr.write(f"\nsign test on {int(lead_wins[ok].sum())} wins of {int(ok.sum())}: two-sided P = {stats.binomtest(int(lead_wins[ok].sum()), int(ok.sum())).pvalue:.3e}\n")
    _tr.write(f"paired t-test two-sided P = {stats.ttest_rel(a[ok], b2[ok]).pvalue:.3e}\n")
    _tr.write("\nfirst 20 genes (panel order): leader, runner-up, difference, rank of |difference| (ties dropped)\n")
    _idx = np.where(ok)[0][:20]
    for _i in _idx:
        _tr.write(f"  {a[_i]:.6f}  {b2[_i]:.6f}  {a[_i]-b2[_i]:+.6f}\n")
    _tr.close()
    DSTAT[(m, c)] = dict(order=order, med=med, lo=lo, hi=hi, win=win, P=P, n=int(ok.sum()),
                         wins=lead_wins, ok=ok)
    print(f"  d {SHORT(m):4} {c:7} {order[0]} over {order[1]} at {100 * win:.1f}% of "
          f"{int(ok.sum()):,} genes, Wilcoxon P = {P:.1e}")

# ------------------------------------------------------------------ data: f
with open(f"{OUT}/switch_pair_figures_{TAG}_plotcache.pkl", "rb") as fh:
    PANEL = pickle.load(fh)["PANEL"]
pick = pd.read_csv(f"{OUT}/switch_figready_pairs_{TAG}.csv")
main = pick.head(4).reset_index(drop=True)
NOWN = pd.read_csv(f"{OUT}/reader_target_counts_{TAG}.csv") \
         .set_index(["motif", "factor"]).n_target.to_dict()
lab = lambda m, f: f


def lg(yK, yH):
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.log2(np.asarray(yK, float) / np.asarray(yH, float))
    v[~np.isfinite(v)] = np.nan
    return v


# ------------------------------------------------------------------ data: g  (design B, per gene)
PG = pd.read_csv(f"{OUT}/cross_line_pergene_{TAG}.csv")
PG["mag"] = PG.magnitude.abs()
ctrl = PG[PG.group == "chip"].groupby("factor").mag.median().values
R = PG[PG.group == "reader"]
rows = []
for r in pick.itertuples(index=False):
    for side, f in (("K", r.reader_K), ("H", r.reader_H)):
        rec = R[(R.motif == r.motif) & (R.factor == f)]
        if rec.empty:
            raise SystemExit(f"reader {f} of {r.motif} missing from the per-gene table")
        rows.append(dict(motif=r.motif, factor=f, side=side, mag=float(rec.iloc[0].mag)))
G = pd.DataFrame(rows)
P_MWU = stats.mannwhitneyu(G.mag.values, ctrl, alternative="greater").pvalue
MED_R, MED_C = float(np.median(G.mag)), float(np.median(ctrl))
print(f"g: {len(G)} readers median {2**MED_R:.2f}x vs {len(ctrl)} background TFs "
      f"{2**MED_C:.2f}x, one-sided MWU P = {P_MWU:.3f}, "
      f"{int((G.mag > MED_C).sum())}/{len(G)} above the background median")

# ------------------------------------------------------------------ canvas
FIGW, FIGH = 7.205, 7.10                      # 183 x 180 mm
fig = plt.figure(figsize=(FIGW, FIGH))
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.37,
                         left=0.068, right=0.995, top=0.928, bottom=0.074)

# ================================================================== row 1
row1 = outer[0].subgridspec(1, 2, width_ratios=[1.30, 1.0], wspace=0.30)

# ---- d: reader curves
gd = row1[0].subgridspec(2, 2, wspace=0.34, hspace=0.88)
for k, (m, c, regime) in enumerate(PICKS):
    p, r = cache[(m, c)], rank[(m, c)]
    st = DSTAT[(m, c)]
    cell = gd[k // 2, k % 2].subgridspec(1, 2, width_ratios=[1.0, 0.19], wspace=0.06)
    ax = fig.add_subplot(cell[0])
    axm = fig.add_subplot(cell[1], sharey=ax)
    N, w = p["n"], p["w"]
    x = np.arange(1, N + 1)
    order = list(r.names)
    others = [roll(p["curves"][nm], w) for nm in order[2:] if nm in p["curves"]]
    if others:                                   # spread of the remaining readers
        A = np.vstack(others)
        ax.fill_between(x, np.nanpercentile(A, 10, axis=0), np.nanpercentile(A, 90, axis=0),
                        color=REST, alpha=0.15, lw=0, zorder=1)
    for nm in order[::-1]:
        if nm not in p["curves"]:
            continue
        i = order.index(nm)
        col = LEAD if i == 0 else (SECOND if i == 1 else REST)
        ax.plot(x, roll(p["curves"][nm], w), color=col,
                lw=0.85 if i < 2 else 0.3, zorder=4 - min(i, 2),
                alpha=1.0 if i < 2 else 0.35)
    ax.set_yscale("log")
    ax.set_xlim(1, N)
    logticks(ax)
    strip(ax)
    # marginal: every reader's median effect over genes, with its bootstrap interval
    for i, nm in enumerate(st["order"]):
        col = LEAD if i == 0 else (SECOND if i == 1 else REST)
        xx = 0.5
        axm.plot([xx, xx], [st["lo"][nm], st["hi"][nm]], color=col, lw=0.7,
                 alpha=1.0 if i < 2 else 0.7, solid_capstyle="round", zorder=3 - min(i, 2))
        axm.plot([xx], [st["med"][nm]], marker="o", ms=2.8 if i < 2 else 2.0,
                 mfc=col, mec="white", mew=0.35, zorder=5 - min(i, 2))
    axm.set_xlim(0, 1)
    axm.set_xticks([])
    for sp in ("top", "right", "bottom", "left"):
        axm.spines[sp].set_visible(False)
    axm.yaxis.set_ticks_position("none")
    axm.tick_params(left=False, labelleft=False)
    if k % 2 == 0:
        ax.set_ylabel("|effect| (QN)", color=INK, labelpad=1.5)
    if k // 2 == 1:
        ax.set_xlabel("genes by general perturbability", color=INK, labelpad=1.5)
    cl = "HEK293T" if c == "HEK293" else c
    ax.text(0.0, 1.40, f"{SHORT(m)} {FAMILY_NAME.get(m, '')}, {cl}", transform=ax.transAxes,
            fontsize=FS_LAB, fontweight="bold", color=INK, va="top", ha="left")
    ax.text(0.0, 1.22, regime, transform=ax.transAxes, fontsize=FS_SM,
            color=SUB, va="top", ha="left")
    ax.text(1.0, 1.22, f"lead {r.lead_over_rest:.2f}×", transform=ax.transAxes,
            fontsize=FS_SM, color=LEAD, va="top", ha="right")
    # which reader is ahead, gene by gene, along the same x axis: the share of orange is the
    # share of genes at which the leader beats the runner-up
    axr = ax.inset_axes([0, 1.015, 1, 0.055])
    band = np.where(st["wins"][st["ok"]], 1.0, 0.0)[None, :]
    axr.imshow(band, aspect="auto", interpolation="nearest", rasterized=True,
               cmap=ListedColormap([SECOND, LEAD]), vmin=0, vmax=1, alpha=0.85)
    axr.set_xticks([]); axr.set_yticks([])
    for sp in axr.spines.values():
        sp.set_visible(False)
    hand = [Line2D([], [], color=LEAD, lw=0.9, label=order[0]),
            Line2D([], [], color=SECOND, lw=0.9, label=order[1])]
    if len(order) > 2:
        hand.append(Line2D([], [], color=REST, lw=0.6,
                           label=f"{len(order) - 2} other reader"
                                 + ("s" if len(order) - 2 > 1 else "")))
    ax.legend(handles=hand, loc="lower left", handlelength=1.0, labelspacing=0.15,
              borderpad=0.2, frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.75, fontsize=FS_SM)

# ---- e: dominance summary
ge = row1[1].subgridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.06)
NROW_E = len(D)
y = np.arange(NROW_E)[::-1]
shown = {(m, c) for m, c, _ in PICKS}
ax = fig.add_subplot(ge[0])
for i, r in enumerate(D.itertuples(index=False)):
    yy, prof = y[i], r.profile
    ax.plot([min(prof), 1.0], [yy, yy], color=BG, lw=0.9, zorder=1)
    ax.plot(prof[2:], [yy] * (len(prof) - 2), marker="o", ls="none", ms=2.1,
            mfc=MUT, mec="white", mew=0.3, alpha=0.9, zorder=3)
    if len(prof) > 1:
        ax.plot([prof[1]], [yy], marker="o", ms=2.6, mfc=SECOND, mec="white",
                mew=0.35, zorder=4)
    ax.plot([1.0], [yy], marker="o", ms=3.4, mfc=LEAD, mec="white", mew=0.4, zorder=5)
    ax.annotate(r.leader, (1.0, yy), textcoords="offset points", xytext=(5, 0),
                ha="left", va="center", fontsize=FS_SM, color=LEAD, zorder=6)
    star = "  ■" if (r.motif, r.cell) in shown else ""
    cl = "HEK293T" if r.cell == "HEK293" else r.cell
    ax.text(-0.012, yy, f"{SHORT(r.motif)} {cl}{star}", ha="right", va="center",
            fontsize=FS_SM, color=INK if star else SUB,
            transform=ax.get_yaxis_transform())
ax.set_xlim(0, 1.45)
ax.set_ylim(-0.7, NROW_E - 0.3)
ax.set_yticks([])
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("reader effect, fraction of the leader", labelpad=1.5)
strip(ax, left=False)

axb = fig.add_subplot(ge[1])
axb.axvline(1.0, color=SUB, lw=0.6, ls=(0, (2.5, 1.8)), zorder=1)
for i, r in enumerate(D.itertuples(index=False)):
    yy = y[i]
    axb.plot([r.lead_over_rest_lo, r.lead_over_rest_hi], [yy, yy], color=BG, lw=1.2,
             solid_capstyle="round", zorder=2)
    axb.plot([r.lead_over_2nd], [yy], marker="o", ms=2.8, mfc="white", mec=SECOND,
             mew=0.7, zorder=3)
    axb.plot([r.lead_over_rest], [yy], marker="o", ms=3.2, mfc=LEAD, mec="white",
             mew=0.4, zorder=4)
axb.set_ylim(-0.7, NROW_E - 0.3)
axb.set_yticks([])
axb.set_xlim(0.92, max(3.35, D.lead_over_rest_hi.max() * 1.06))
axb.set_xlabel("lead over others (filled)\nand runner-up (open)", labelpad=1.0)
strip(axb, left=False)
fig.legend(handles=[
    Line2D([], [], marker="o", ls="none", ms=3.4, mfc=LEAD, mec="white", label="leading reader"),
    Line2D([], [], marker="o", ls="none", ms=2.6, mfc=SECOND, mec="white", label="runner-up"),
    Line2D([], [], marker="o", ls="none", ms=2.1, mfc=MUT, mec="white", label="other readers"),
    Line2D([], [], marker="s", ls="none", ms=2.4, mfc=INK, mec="none", label="shown in d")],
    frameon=False, loc="upper left", bbox_to_anchor=(0.578, 0.958), ncol=4,
    fontsize=FS_SM, handletextpad=0.35, columnspacing=0.9)

# ================================================================== row 2
row2 = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.12], wspace=0.17)

# ---- f: cross-line switch, 2 x 2 square
gf = row2[0].subgridspec(2, 2, wspace=0.34, hspace=0.62)
for i, r in enumerate(main.itertuples(index=False)):
    P = PANEL[r.motif]
    x = np.arange(1, P["n"] + 1)
    va = lg(*P["readers"][r.reader_K])
    vb = lg(*P["readers"][r.reader_H])
    cell = gf[i // 2, i % 2].subgridspec(1, 2, width_ratios=[1.0, 0.19], wspace=0.06)
    ax = fig.add_subplot(cell[0])
    axm = fig.add_subplot(cell[1], sharey=ax)
    ax.fill_between(x, va, vb, color=SHADE, lw=0, zorder=1)
    ax.axhline(0, color=SUB, lw=0.6, ls=(0, (2.0, 1.6)), zorder=2)
    ax.plot(x, va, color=K_COL, lw=0.6, zorder=3, solid_capstyle="round")
    ax.plot(x, vb, color=H_COL, lw=0.6, zorder=3, solid_capstyle="round")
    for v, col in ((va, K_COL), (vb, H_COL)):        # each reader's median ratio
        ax.axhline(np.nanmedian(v), color=col, lw=0.5, ls=(0, (1.2, 1.2)),
                   alpha=0.85, zorder=4)
    lim = max(1.0, 1.15 * np.nanmax(np.abs(np.nanpercentile(
        np.concatenate([va, vb]), [1, 99]))))
    ax.set_xlim(1, P["n"])
    ax.set_ylim(-lim, lim)
    strip(ax)
    # marginal: how the two plotted ratios are distributed over the gene axis
    yy = np.linspace(-lim, lim, 220)
    for v, col in ((va, K_COL), (vb, H_COL)):
        vv = v[np.isfinite(v)]
        dd = stats.gaussian_kde(vv)(yy)
        axm.fill_betweenx(yy, 0, dd / dd.max(), color=col, alpha=0.30, lw=0, zorder=2)
        axm.plot(dd / dd.max(), yy, color=col, lw=0.5, zorder=3)
    axm.axhline(0, color=SUB, lw=0.6, ls=(0, (2.0, 1.6)), zorder=1)
    axm.set_xlim(0, 1.12)
    axm.set_xticks([])
    for sp in ("top", "right", "bottom", "left"):
        axm.spines[sp].set_visible(False)
    axm.yaxis.set_ticks_position("none")
    axm.tick_params(left=False, labelleft=False)
    BOX = dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.8)
    ax.text(0.02, 0.97, lab(r.motif, r.reader_K), transform=ax.transAxes,
            fontsize=FS_SM, color=K_COL, va="top", ha="left", bbox=BOX, zorder=5)
    ax.text(0.02, 0.03, lab(r.motif, r.reader_H), transform=ax.transAxes,
            fontsize=FS_SM, color=H_COL, va="bottom", ha="left", bbox=BOX, zorder=5)
    fam = FAMILY_NAME.get(r.motif, "")
    ax.text(0.0, 1.20, f"{SHORT(r.motif)}{' ' + fam if fam else ''}", transform=ax.transAxes,
            fontsize=FS_LAB, fontweight="bold", color=INK, va="top", ha="left")
    ax.text(1.0, 1.20, f"{2 ** r.gap:.1f}× apart", transform=ax.transAxes,
            fontsize=FS_SM, color=SUB, va="top", ha="right")
    if i == 0:
        ax.text(0.98, 0.97, "larger in K562", transform=ax.transAxes, fontsize=FS_SM,
                color=SUB, va="top", ha="right", bbox=BOX, zorder=5)
        ax.text(0.98, 0.03, "larger in HEK293T", transform=ax.transAxes, fontsize=FS_SM,
                color=SUB, va="bottom", ha="right", bbox=BOX, zorder=5)
    if i % 2 == 0:
        ax.set_ylabel("log₂ |effect| K562 / HEK293T", color=INK, labelpad=1.5)
    if i // 2 == 1:
        ax.set_xlabel("genes by mean reader effect", color=INK, labelpad=1.5)

# ---- g: switch magnitude per reader (2026-10-04: control background REMOVED at the professor's request;
#         the panel now reports the observed switch of each pair, not a comparison with other TFs)
gg = row2[1].subgridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.30)
XMAX = max(2.30, G.mag.max() * 1.12)
order = list(pick.motif)
ax = fig.add_subplot(gg[0])
CLOSE = 0.17 * XMAX
for i, m in enumerate(order):
    yy = len(order) - 1 - i
    g = G[G.motif == m].sort_values("mag").reset_index(drop=True)
    ax.plot(g.mag.values, [yy, yy], color=BG, lw=1.1, zorder=1)
    tight = (g.mag[1] - g.mag[0]) < CLOSE
    for j, xr in enumerate(g.itertuples(index=False)):
        c = K_COL if xr.side == "K" else H_COL
        ax.plot([xr.mag], [yy], marker="o", ms=4.0, mfc=c, mec="white", mew=0.6, zorder=4)
        ha, dx = (("right", -2.0) if j == 0 else ("left", 2.0)) if tight else ("center", 0.0)
        ax.annotate(xr.factor, (xr.mag, yy), textcoords="offset points",
                    xytext=(dx, 4.6), ha=ha, va="baseline", fontsize=FS_SM, color=c, zorder=5)
    ax.text(-0.012, yy, SHORT(m), ha="right", va="center", fontsize=FS_TICK, color=INK,
            transform=ax.get_yaxis_transform())
ax.set_xlim(0, XMAX)
ax.set_ylim(-0.55, len(order) - 0.15)
ax.set_yticks([])
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_linewidth(0.5); ax.spines["bottom"].set_color(MUT)
ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
ax.set_xticklabels([f"{2**v:.1f}×" for v in [0, 0.5, 1.0, 1.5, 2.0]])
ax.tick_params(labelsize=FS_SM, colors=SUB, labelcolor=INK, pad=1.2, width=0.5, length=2)
ax.set_xlabel("switch magnitude, fold difference of knockout effect between K562 and HEK293T", labelpad=1.5)
ax.legend(handles=[
    Line2D([], [], marker="o", ls="none", ms=4, mfc=K_COL, mec="white", label="K562-leaning reader"),
    Line2D([], [], marker="o", ls="none", ms=4, mfc=H_COL, mec="white", label="HEK293T-leaning reader")],
    frameon=False, loc="lower right", fontsize=FS_SM, handletextpad=0.4, borderaxespad=0.1)

# lower part: the pair separation (fold between the two readers' cross-line ratios), one bar per motif
axd = fig.add_subplot(gg[1])
sep = []
for m in order:
    r = pick[pick.motif == m].iloc[0]
    sep.append(2 ** abs(float(r.log2_K) - float(r.log2_H)))
ypos = np.arange(len(order))[::-1]
axd.barh(ypos, sep, height=0.62, color=BG, edgecolor=SUB, lw=0.5, zorder=2)
for yv, s, m in zip(ypos, sep, order):
    axd.text(s + 0.08, yv, f"{s:.1f}×", va="center", ha="left", fontsize=FS_SM, color=INK)
axd.set_yticks(ypos); axd.set_yticklabels([SHORT(m) for m in order], fontsize=FS_TICK, color=INK)
axd.set_xlim(0, max(sep) * 1.25)
axd.set_xlabel("separation between the two readers of a pair (fold)", labelpad=1.5)
strip(axd, left=True)
print(f"g: {len(G)} readers, switch magnitude {2**G.mag.min():.2f}x to {2**G.mag.max():.2f}x (median {2**MED_R:.2f}x); "
      f"pair separation {min(sep):.1f}x to {max(sep):.1f}x; {int(pick.reciprocal.sum())}/{len(pick)} pairs reciprocal")

# ================================================================== letters + headers
Y1, Y2 = 0.992, 0.487
head(fig, 0.004, Y1, "d", "Reader curves")
head(fig, 0.578, Y1, "e", "Reader dominance")
head(fig, 0.004, Y2, "f", "Cross-line switch")
head(fig, 0.545, Y2, "g", "Switch magnitude")

# ================================================================== save
os.makedirs(FIG, exist_ok=True)
for ext in ("pdf", "png", "svg"):
    fig.savefig(f"{FIG}/fig5_defg_{TAG}.{ext}", dpi=600, transparent=True)
plt.close(fig)
for ext in ("pdf", "png", "svg"):
    shutil.copy2(f"{FIG}/fig5_defg_{TAG}.{ext}", f"{DEST}/fig5dg.{ext}")
print(f"\nwrote {FIG}/fig5_defg_{TAG}.pdf  ({FIGW * 25.4:.0f} x {FIGH * 25.4:.0f} mm)"
      f"\ncopied to {DEST}/fig5dg.pdf/.png/.svg   font: {FAMILY}")
