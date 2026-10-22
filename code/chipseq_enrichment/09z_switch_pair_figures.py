#!/usr/bin/env python
"""
09z -- the three figures the user asked for on 2026-08-26, all on one gene-ranked axis per motif.

For one factor, log2(|effect| in K562 / |effect| in HEK293) on a motif's target genes says which
line its knockout matters more in.  A pair of readers SWITCHES when one reader sits above the
equal-effect line and the other below it, on the same genes.

  FIGURE 1  switch_pair_per_motif   one panel per motif; the two readers with the widest
                                    cross-line difference, each drawn as log2(K562/HEK293)
                                    along the motif's gene axis.  The gap between the two
                                    curves is the switch; a pair that straddles zero is a
                                    reciprocal switch.
  FIGURE 2  all_readers_per_motif   the same panels with EVERY reader of the motif drawn,
                                    so the chosen pair can be seen against its own field.
  FIGURE 3  control_switch_null     how much ordinary transcription factors differ between the
                                    two lines on the identical genes: the distribution of
                                    |log2 K562/HEK293| for control TFs, the distribution of the
                                    gap between two random control TFs, the selection-matched
                                    best-of-k null per motif, and control pairs drawn exactly
                                    like Figure 1.

Controls are the 09x definition: ChIP-verified transcription factors that are not readers of any
universal motif, with effective knockdown in both screens, scored on the same motif gene sets.

Usage:  python 09z_switch_pair_figures.py top90chip            # ~10 min, loads both screens
        python 09z_switch_pair_figures.py top90chip --replot   # redraw from the cache, seconds

Reads   results/prof_plan/cross_line_control_{tag}.csv        (09x, all the per-factor statistics)
Writes  results/prof_plan/fig_reader_curves/switch_pair_per_motif_{tag}.{pdf,png}
        results/prof_plan/fig_reader_curves/all_readers_per_motif_{tag}.{pdf,png}
        results/prof_plan/fig_reader_curves/control_switch_null_{tag}.{pdf,png}
        results/prof_plan/switch_pair_per_motif_{tag}.csv
        results/prof_plan/switch_pair_figures_{tag}_plotcache.pkl
"""
import os, sys, json, pickle, string
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
TAG = ARGS[0] if ARGS else "top90chip"
REPLOT = "--replot" in sys.argv
BIND_FILE = {"top10chip": "reader_binding.parquet",
             "top90chip": "reader_binding_top90.parquet",
             "allchip": "reader_binding_all.parquet"}[TAG]
MIN_GENES = 300          # 09x/09w motif gene-set floor, unchanged
NB = 20000               # null draws, as 09y
NCTRL_CURVE = 2          # control factors kept per motif for the Figure 3 example panels
ORDER = "meaneffect"     # gene order: "meaneffect" = mean |effect| over the motif's readers
                         # (user's request; note this is derived FROM the curves, so the mean
                         # magnitude trace is monotone by construction), or "perturbability"
                         # = median effect of knockouts of factors that do not bind the gene

HERE = os.path.dirname(os.path.abspath(__file__))
if not REPLOT:
    src = open(os.path.join(HERE, "08c_alltargets_stats.py")).read()
    exec(compile(src.split("DEFS = [")[0], "08c_prefix", "exec"), globals())
else:
    ROOT = "/stg3/data1/sam/enhancer_prediction"
    CHIP = os.path.join(ROOT, "multi_agent/chipseq_enrichment")

OUT = os.path.join(CHIP, "results/prof_plan")
FIG = os.path.join(OUT, "fig_reader_curves")
RS = os.path.join(CHIP, "reader_switch/out")
CACHE = f"{OUT}/switch_pair_figures_{TAG}_plotcache.pkl"
LINES = ["K562", "HEK293"]

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
CK, CH = "#4878a8", "#b06a72"        # K562 blue, HEK293 red -- also colour a line-leaning factor
CR, CC = "#1f5f8b", "#c08a2e"        # readers, control TFs
SHORT = lambda m: m.replace("universal_", "M")            # compact, for axis ticks
HEAD = lambda m: "Motif " + m.split("_")[1]               # panel headers: motif numbers only


def roll(y, w):
    return pd.Series(y).rolling(w, center=True, min_periods=max(3, w // 4)).median().to_numpy()


def lg(yK, yH):
    """log2 K562/HEK293 of two smoothed |effect| curves, NaN where either is unusable"""
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.log2(np.asarray(yK, float) / np.asarray(yH, float))
    v[~np.isfinite(v)] = np.nan
    return v


# ============================================================================ per-factor statistics
D = pd.read_csv(f"{OUT}/cross_line_control_{TAG}.csv")
R = D[D.group == "reader"]
A = D[D.group != "reader"]                     # ChIP-verified non-reader TFs, same gene sets
rng = np.random.default_rng(0)

rows = []
for m, g in R.groupby("motif"):
    if len(g) < 2:
        continue
    c = A[A.motif == m]
    if len(c) < len(g):
        continue
    g = g.sort_values("log2_target")
    k = len(g)
    obs = float(g.log2_target.iloc[-1] - g.log2_target.iloc[0])
    draws = c.log2_target.to_numpy()[rng.choice(len(c), size=(NB, k))]
    null = draws.max(-1) - draws.min(-1)
    rows.append(dict(
        motif=m, k=k, n_genes=int(g.n_genes.iloc[0]),
        reader_K=g.factor.iloc[-1], reader_H=g.factor.iloc[0],
        log2_K=float(g.log2_target.iloc[-1]), log2_H=float(g.log2_target.iloc[0]),
        gap=obs, reciprocal=bool(g.log2_target.iloc[-1] > 0 > g.log2_target.iloc[0]),
        null_med=float(np.median(null)), pctile=float(100 * (null < obs).mean()),
        P=float((null >= obs).mean())))
S = pd.DataFrame(rows).sort_values("gap", ascending=False).reset_index(drop=True)
S.to_csv(f"{OUT}/switch_pair_per_motif_{TAG}.csv", index=False)
MOTIFS = list(S.motif)
_cf = f"{OUT}/own_gene_coverage_{TAG}.csv"          # written by 09z2; how much of the panel each reader binds
COV = pd.read_csv(_cf).set_index(["motif", "factor"]).frac_of_union.to_dict() if os.path.exists(_cf) else {}
_tf = f"{OUT}/reader_target_counts_{TAG}.csv"       # genes each reader binds, all 70 records
NOWN = pd.read_csv(_tf).set_index(["motif", "factor"]).n_target.to_dict() if os.path.exists(_tf) else {}
lab = lambda m, f: f"{f} ({NOWN[(m, f)]})" if (m, f) in NOWN else f

print(f"\n=== widest-difference reader pair per motif ({len(S)} motifs with >= 2 readers) ===")
print(f"  {'motif':6}{'k':>3} {'pair (K562-leaning / HEK-leaning)':34} {'log2 each':>16} "
      f"{'gap':>8} {'recip':>6} {'null':>8} {'P':>7}")
for r in S.itertuples(index=False):
    print(f"  {SHORT(r.motif):6}{r.k:>3} {r.reader_K + ' / ' + r.reader_H:34} "
          f"{r.log2_K:+7.2f} {r.log2_H:+7.2f} {2**r.gap:7.2f}x {'yes' if r.reciprocal else 'no':>6} "
          f"{2**r.null_med:7.2f}x {r.P:7.3f}")
print(f"\n  reciprocal (the pair straddles equal effect): {int(S.reciprocal.sum())}/{len(S)}")
print(f"  above the median of the selection-matched control null: {int((S.pctile>50).sum())}/{len(S)}"
      f", sign test P = {stats.binomtest(int((S.pctile>50).sum()), len(S), 0.5).pvalue:.3g}")
print(f"  motifs reaching P < 0.05 against that null: {int((S.P<0.05).sum())}/{len(S)}")

# ============================================================================ curves
if not REPLOT:
    SHARED = sorted(shared)
    POS = {g: i for i, g in enumerate(SHARED)}
    kidx = np.array([k_col[g] for g in SHARED])
    oidx = np.array([o_col[g] for g in SHARED])
    roster = {m: sorted({norm(t) for t in v}) for m, v in json.load(open(os.path.join(
        ROOT, "multi_agent/posthoc_analysis/results/step06_JASPAR_assignments.json"))).items()}
    assayed = {}
    for cell in LINES:
        with open(os.path.join(CHIP, f"data/{cell}/tf_peak_sets.pkl"), "rb") as fh:
            assayed[cell] = {norm(t) for t in pickle.load(fh)}

    def eff_here(tf, cell):
        if cell == "K562":
            return tf in k_perturbed and k_effkd(tf)
        return tf in o_perturbed and o_effkd_mask.get(tf, False)

    _e = {}

    def effect(tf, cell):
        if (tf, cell) not in _e:
            _e[(tf, cell)] = np.abs(k_delta(tf)[kidx] if cell == "K562" else o_delta(tf)[oidx])
        return _e[(tf, cell)]

    print("\nBuilding promoter binding (sets the gene order only) ...", flush=True)
    BOUND, BGTF = {}, {}
    for cell, L, LO in (("K562", K, K_LO), ("HEK293", H, H_LO)):
        for tf, pids in L["tf_peakids"].items():
            if not eff_here(tf, cell):
                continue
            thr = LO.get(tf, -np.inf)
            gs = set()
            for pid in pids:
                if L["peak_inten"][pid] >= thr:
                    gs.update(L["peak_genes"][pid])
            BOUND[(cell, tf)] = gs & shared
        BGTF[cell] = sorted({t for (c, t) in BOUND if c == cell})

    def perturbability(cell, genes, idx):
        """how strongly each gene answers knockouts in general, from factors that do NOT bind it"""
        M = np.vstack([effect(t, cell)[idx] for t in BGTF[cell]])
        NB_ = np.vstack([[g not in BOUND[(cell, t)] for g in genes] for t in BGTF[cell]])
        with np.errstate(invalid="ignore"):
            md = np.nanmedian(np.where(NB_, M, np.nan), 0)
        md[NB_.sum(0) < 20] = np.nan
        return md

    rb = pd.read_parquet(f"{RS}/{BIND_FILE}")
    rb = rb[rb["cell"].isin(LINES)].copy()
    rb["rn"] = rb["reader"].map(norm)
    rb = rb[rb["ensembl"].isin(set(SHARED))]
    sets = {}
    for (cell, m, rn), d in rb.groupby(["cell", "motif_id", "rn"]):
        if m in roster and rn in roster[m] and rn in assayed[cell]:
            sets[(cell, m, rn)] = set(d["ensembl"])

    PANEL = {}
    for m in MOTIFS:
        genes = sorted(set().union(*[sets[(c, mm, r)] for (c, mm, r) in sets if mm == m]) or set())
        if len(genes) < MIN_GENES:
            continue
        idx = np.array([POS[g] for g in genes])
        if ORDER == "meaneffect":
            rd = list(R[R.motif == m].factor)
            mean_eff = np.nanmean([(effect(f, "K562")[idx] + effect(f, "HEK293")[idx]) / 2
                                   for f in rd], 0)
            o = np.argsort(-np.nan_to_num(mean_eff, nan=-1))
        else:
            pk = perturbability("K562", genes, idx)
            ph = perturbability("HEK293", genes, idx)
            o = np.argsort(-np.nan_to_num(np.nanmean([pk, ph], 0), nan=-1))
        wdw = max(15, len(o) // 50)
        curve = lambda f: tuple(roll(effect(f, c)[idx][o], wdw).astype(np.float32) for c in LINES)

        readers = {f: curve(f) for f in R[R.motif == m].factor}
        cm = A[A.motif == m].sort_values("log2_target")
        pick = rng.choice(len(cm), size=int(S.loc[S.motif == m, "k"].iloc[0]), replace=False)
        dpk = cm.iloc[pick].sort_values("log2_target")     # one honest draw of the same size
        cpair = [dpk.factor.iloc[-1], dpk.factor.iloc[0]]
        ctrl = {f: curve(f) for f in cpair[:NCTRL_CURVE]}
        PANEL[m] = dict(n=len(o), readers=readers, ctrl=ctrl, ctrl_pair=cpair,
                        ctrl_log2=[float(dpk.log2_target.iloc[-1]), float(dpk.log2_target.iloc[0])])
        print(f"  cached {SHORT(m)}: {len(o)} genes, {len(readers)} readers", flush=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(dict(PANEL=PANEL, tag=TAG), fh)
    print(f"wrote -> {CACHE}")
else:
    with open(CACHE, "rb") as fh:
        PANEL = pickle.load(fh)["PANEL"]
    print(f"replotting from {CACHE}: {len(PANEL)} motif panels")

MOTIFS = [m for m in MOTIFS if m in PANEL]
S = S[S.motif.isin(PANEL)].reset_index(drop=True)


# ============================================================================ shared panel drawing
def draw_panel(ax, m, factors, colours, labels=None, lw=0.9, band=None):
    """log2(K562/HEK293) along the motif's gene axis, one curve per factor"""
    P = PANEL[m]
    src = {**P["readers"], **P["ctrl"]}
    x = np.arange(1, P["n"] + 1)
    ax.axhline(0, color=MUT, lw=0.5, zorder=1)
    ends = []
    for f, col in zip(factors, colours):
        v = lg(*src[f])
        ax.plot(x, v, color=col, lw=lw, zorder=3, solid_capstyle="round")
        ends.append(np.nanmedian(v[-max(10, P["n"] // 20):]))
    if band is not None and len(factors) == 2:
        a, b = lg(*src[factors[0]]), lg(*src[factors[1]])
        ax.fill_between(x, a, b, color=band, alpha=0.13, lw=0, zorder=2)
    ax.set_xlim(1, P["n"])
    lim = max(1.0, np.nanmax([abs(np.nanpercentile(lg(*src[f]), [1, 99])).max()
                              for f in factors]) * 1.12)
    ax.set_ylim(-lim, lim)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=SUB, labelcolor=INK, pad=1.5)
    if labels:                                  # name each curve at its right-hand end, no overlap
        order = np.argsort(-np.nan_to_num(np.array(ends), nan=-1e9))
        sep = 2 * lim / max(len(order) + 1, 7)
        prev = lim + sep
        for j in order:
            e = ends[j] if np.isfinite(ends[j]) else -lim
            y = min(e, prev - sep)            # may run below the axes; labels are not clipped
            prev = y
            ax.annotate(labels[j], (P["n"], y), xytext=(2.5, 0), textcoords="offset points",
                        fontsize=4.4, color=colours[j], va="center", ha="left",
                        annotation_clip=False)
    return ax


def panel_head(ax, letter, head, *notes):
    ns = [n for n in notes if n]
    top = 1.22 + 0.115 * len(ns)                      # header sits above however many note lines
    ax.text(-0.30, top, letter, transform=ax.transAxes, fontsize=7, fontweight="bold",
            color=INK, va="top", ha="left")
    ax.text(-0.15, top, head, transform=ax.transAxes, fontsize=7, fontweight="bold",
            color=INK, va="top", ha="left")
    for k, t in enumerate(ns):
        ax.text(-0.15, top - 0.18 - 0.115 * k, t, transform=ax.transAxes, fontsize=4.8,
                color=SUB, va="top", ha="left")


XLAB = ("genes by mean reader effect" if ORDER == "meaneffect" else "genes by perturbability")
LEG = [Line2D([], [], color=CK, lw=1.2, label="knockout matters more in K562"),
       Line2D([], [], color=CH, lw=1.2, label="knockout matters more in HEK293")]

# ============================================================================ FIGURE 1: best pair
NC = 4
nrow = int(np.ceil(len(S) / NC))
fig = plt.figure(figsize=(7.09, 1.55 * nrow + 0.5), facecolor="white")
gs = fig.add_gridspec(nrow, NC, hspace=1.15, wspace=0.60)
for i, r in enumerate(S.itertuples(index=False)):
    ax = fig.add_subplot(gs[i // NC, i % NC])
    draw_panel(ax, r.motif, [r.reader_K, r.reader_H], [CK, CH],
               labels=[lab(r.motif, r.reader_K), lab(r.motif, r.reader_H)], lw=1.0, band=INK)
    panel_head(ax, string.ascii_lowercase[i], HEAD(r.motif))
    if i % NC == 0:
        ax.set_ylabel("log$_2$ K562 / HEK293", color=INK, labelpad=1.5)
    if i + NC >= len(S):
        ax.set_xlabel(XLAB, color=INK, labelpad=1.5, fontsize=5)
fig.legend(handles=LEG, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.012))
for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/switch_pair_per_motif_{TAG}.{ext}", dpi=450, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print(f"\nwrote -> {FIG}/switch_pair_per_motif_{TAG}.pdf/.png")

# ============================================================================ FIGURE 2: all readers
fig = plt.figure(figsize=(7.09, 1.80 * nrow + 0.5), facecolor="white")
gs = fig.add_gridspec(nrow, NC, hspace=1.10, wspace=0.78)
for i, r in enumerate(S.itertuples(index=False)):
    ax = fig.add_subplot(gs[i // NC, i % NC])
    g = R[R.motif == r.motif].sort_values("log2_target", ascending=False)
    fs = list(g.factor)
    cols = [CK if v > 0 else CH for v in g.log2_target]
    lws = [1.0 if f in (r.reader_K, r.reader_H) else 0.55 for f in fs]
    for f, c, w in zip(fs, cols, lws):
        draw_panel(ax, r.motif, [f], [c], lw=w)
    draw_panel(ax, r.motif, fs, cols, labels=[lab(r.motif, f) for f in fs], lw=0)   # labels only
    panel_head(ax, string.ascii_lowercase[i], HEAD(r.motif))
    if i % NC == 0:
        ax.set_ylabel("log$_2$ K562 / HEK293", color=INK, labelpad=1.5)
    if i + NC >= len(S):
        ax.set_xlabel(XLAB, color=INK, labelpad=1.5, fontsize=5)
fig.legend(handles=LEG, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.012))
for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/all_readers_per_motif_{TAG}.{ext}", dpi=450, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print(f"wrote -> {FIG}/all_readers_per_motif_{TAG}.pdf/.png")

# ============================================================================ FIGURE 2b: magnitudes
# the same pair on RAW effect magnitude rather than the ratio -- this is the view in which the gene
# ordering shows a decreasing trend (with ORDER="meaneffect" the decline is partly by construction)
fig = plt.figure(figsize=(7.09, 1.55 * nrow + 0.5), facecolor="white")
gs = fig.add_gridspec(nrow, NC, hspace=1.15, wspace=0.60)
for i, r in enumerate(S.itertuples(index=False)):
    ax = fig.add_subplot(gs[i // NC, i % NC])
    P_ = PANEL[r.motif]
    x = np.arange(1, P_["n"] + 1)
    for f, ls in ((r.reader_K, "-"), (r.reader_H, (0, (2.2, 1.2)))):
        if f not in P_["readers"]:
            continue
        yK, yH = P_["readers"][f]
        ax.plot(x, yK, color=CK, lw=0.85, ls=ls, zorder=3)
        ax.plot(x, yH, color=CH, lw=0.85, ls=ls, zorder=3)
    ax.set_yscale("log")
    ax.set_xlim(1, P_["n"])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=SUB, labelcolor=INK, pad=1.5)
    panel_head(ax, string.ascii_lowercase[i], HEAD(r.motif))
    ax.text(1.0, 1.02, f"{lab(r.motif, r.reader_K)} solid, {lab(r.motif, r.reader_H)} dashed",
            transform=ax.transAxes, fontsize=4.4, color=SUB, va="bottom", ha="right")
    if i % NC == 0:
        ax.set_ylabel("|effect| (QN)", color=INK, labelpad=1.5)
    if i + NC >= len(S):
        ax.set_xlabel(XLAB, color=INK, labelpad=1.5, fontsize=5)
fig.legend(handles=[Line2D([], [], color=CK, lw=1.2, label="K562"),
                    Line2D([], [], color=CH, lw=1.2, label="HEK293T")],
           frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.012))
for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/switch_pair_magnitude_{TAG}.{ext}", dpi=450, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print(f"wrote -> {FIG}/switch_pair_magnitude_{TAG}.pdf/.png")

# ============================================================================ FIGURE 3: the control
by = lambda Sf, col: Sf.groupby("factor")[col].apply(lambda s: s.abs().median())
cf = by(A, "log2_target")                       # one value per control TF
rv = R.log2_target.abs()
mwu = stats.mannwhitneyu(rv, cf).pvalue
pct_r = float(100 * (cf < rv.median()).mean())

# every within-motif pair of control TFs, and of readers, on the identical gene sets
gap_c, gap_r = [], []
for m in MOTIFS:
    v = A[A.motif == m].log2_target.to_numpy()
    j = rng.choice(len(v), size=(4000, 2))
    gap_c.append(np.abs(v[j[:, 0]] - v[j[:, 1]]))
    w = R[R.motif == m].log2_target.to_numpy()
    gap_r += [abs(w[a] - w[b]) for a in range(len(w)) for b in range(a + 1, len(w))]
gap_c = np.concatenate(gap_c)
gap_r = np.array(gap_r)

print("\n=== how much ordinary transcription factors differ between the two lines ===")
print(f"  |log2 K562/HEK293|: readers median {2**rv.median():.2f}x (n = {len(rv)}) vs "
      f"control TFs {2**cf.median():.2f}x (n = {len(cf)}), MWU P = {mwu:.3g}")
print(f"  the median reader sits at the {pct_r:.0f}th percentile of control TFs; "
      f"{100 - pct_r:.0f}% of control TFs differ by at least as much")
print(f"  gap within a random control pair: median {2**np.median(gap_c):.2f}x, "
      f"90th pct {2**np.percentile(gap_c, 90):.2f}x; reader pairs median {2**np.median(gap_r):.2f}x")

fig = plt.figure(figsize=(7.09, 5.0), facecolor="white")
gs = fig.add_gridspec(3, 4, hspace=1.05, wspace=0.62, height_ratios=[1.05, 1.05, 1.0])

# a -- distribution of the cross-line difference, control TFs vs readers
ax = fig.add_subplot(gs[0, :2])
bins = np.linspace(0, np.percentile(cf, 99.5), 46)
ax.hist(cf, bins=bins, color=CC, alpha=0.55, lw=0, density=True, label=f"control TFs (n = {len(cf)})")
ax.hist(rv, bins=bins, histtype="step", color=CR, lw=1.0, density=True,
        label=f"universal-motif readers (n = {len(rv)})")
for v, c in ((cf.median(), CC), (rv.median(), CR)):        # medians as dashed lines only
    ax.axvline(v, color=c, lw=0.7, ls=(0, (3, 2)))
ax.set_xlabel("|log$_2$ K562 / HEK293| of one factor's knockout", color=INK, labelpad=1.5)
ax.set_ylabel("density", color=INK, labelpad=1.5)
ax.legend(frameon=False, loc="upper right", handlelength=1.2)
panel_head(ax, "a", "Every factor", f"readers vs control TFs, MWU P = {mwu:.2f}")

# b -- the gap inside a pair, controls vs readers
ax = fig.add_subplot(gs[0, 2:])
bins = np.linspace(0, np.percentile(gap_c, 99), 46)
ax.hist(gap_c, bins=bins, color=CC, alpha=0.55, lw=0, density=True,
        label="two random control TFs")
ax.hist(gap_r, bins=bins, histtype="step", color=CR, lw=1.0, density=True,
        label="two readers of one motif")
ax.axvline(np.median(gap_c), color=CC, lw=0.7, ls=(0, (3, 2)))
ax.axvline(np.median(gap_r), color=CR, lw=0.7, ls=(0, (3, 2)))
ax.set_xlabel("gap between the pair, log$_2$ K562 / HEK293", color=INK, labelpad=1.5)
ax.set_ylabel("density", color=INK, labelpad=1.5)
ax.legend(frameon=False, loc="upper right", handlelength=1.2)
panel_head(ax, "b", "Every pair",
           f"reader pairs {2**np.median(gap_r):.2f}x, control pairs {2**np.median(gap_c):.2f}x")

# c -- the selection-matched null, per motif
ax = fig.add_subplot(gs[1, :2])
Sp = S.sort_values("pctile")
yy = np.arange(len(Sp))
ax.hlines(yy, 2**Sp.null_med, 2**Sp.gap, color=MUT, lw=0.5, alpha=0.8, zorder=1)
ax.plot(2**Sp.null_med, yy, "o", color=CC, ms=2.6, mew=0, zorder=3,
        label="widest pair among k control TFs")
ax.plot(2**Sp.gap, yy, "o", color=CR, ms=2.6, mew=0, zorder=3, label="widest reader pair")
ax.set_yticks(yy)
ax.set_yticklabels([SHORT(m) for m in Sp.motif], fontsize=4.5)
ax.tick_params(axis="y", length=0, pad=1)
ax.set_xlabel("gap between the pair, fold", color=INK, labelpad=1.5)
ax.set_ylim(-4.6, len(Sp) + 0.6)
ax.legend(frameon=False, loc="lower right", handlelength=1.2)
panel_head(ax, "c", "Matched null",
           f"{int((S.pctile > 50).sum())} of {len(S)} above the null median")

# d -- the gap grows with how many readers there were to choose between
ax = fig.add_subplot(gs[1, 2:])
ax.plot(S.k, 2**S.gap, "o", color=CR, ms=3.0, mew=0, label="widest reader pair")
ax.plot(S.k, 2**S.null_med, "o", color=CC, ms=3.0, mew=0, label="widest pair of k control TFs")
for r in S[S.k >= 4].itertuples(index=False):
    ax.annotate(SHORT(r.motif), (r.k, 2**r.gap), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=4.5, color=SUB)
ax.set_xlabel("readers the motif has (k)", color=INK, labelpad=1.5)
ax.set_ylabel("gap, fold", color=INK, labelpad=1.5)
ax.set_ylim(0, 2**S.gap.max() * 1.25)
ax.legend(frameon=False, loc="upper left", handlelength=1.2)
panel_head(ax, "d", "Choice size")

# e-h -- control pairs drawn exactly like Figure 1
show = list(S.sort_values("pctile", ascending=False).motif)[:4]
for j, m in enumerate(show):
    ax = fig.add_subplot(gs[2, j])
    P = PANEL[m]
    fs = P["ctrl_pair"][:NCTRL_CURVE]
    cols = [CK if v > 0 else CH for v in P["ctrl_log2"][:NCTRL_CURVE]]
    draw_panel(ax, m, fs, cols, labels=fs, lw=1.0, band=INK)
    gapc = abs(P["ctrl_log2"][0] - P["ctrl_log2"][1])
    panel_head(ax, string.ascii_lowercase[4 + j], HEAD(m), f"controls · {2**gapc:.1f}x apart")
    if j == 0:
        ax.set_ylabel("log$_2$ K562 / HEK293", color=INK, labelpad=1.5)
    ax.set_xlabel(XLAB, color=INK, labelpad=1.5, fontsize=5)
fig.legend(handles=LEG, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.008))
for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/control_switch_null_{TAG}.{ext}", dpi=450, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print(f"wrote -> {FIG}/control_switch_null_{TAG}.pdf/.png")
print(f"\nstatistics table -> {OUT}/switch_pair_per_motif_{TAG}.csv")
