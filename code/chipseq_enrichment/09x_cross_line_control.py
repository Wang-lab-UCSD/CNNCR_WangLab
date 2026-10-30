#!/usr/bin/env python
"""
09x -- CONTROL for 09w/09v.  09w shows that a reader's knockout moves its motif's target genes
       by a different amount in K562 than in HEK293 (median 1.41x).  The obvious question is
       whether ANY knockout does that.  This script answers it by running the identical
       statistic on transcription factors that are NOT readers of any universal motif.

Everything is held fixed except the identity of the perturbed factor:
  * the gene sets are the SAME motif target sets used in 09w (union of that motif's readers'
    ChIP top-90% promoter targets), so control factors are scored on the reader genes;
  * the same quantile-normalized effects, the same medians, the same |log2 K562/HEK293|.

Every control must be a genuine transcription factor, so a control is admitted only if it has
ENCODE4 ChIP-seq in at least one cell line.  Perturbations of genes never ChIP'd anywhere are
excluded outright -- they are not necessarily transcription factors.  Two tiers are kept apart:
  chip       -- ChIP-assayed in K562 or HEK293 (matched to the readers on data availability)
  chip_other -- ChIP-assayed only in some other ENCODE4 line (still a verified TF)

Control factors are summarized one value per factor (the median across the motif gene sets they
were scored on), never one value per gene set, so the comparison is not inflated by scoring the
same factor 22 times.

Panel b is the decisive one: the cross-line ratio measured on a motif's target genes is plotted
against the same ratio measured genome wide.  If the two agree, the difference is a property of
the knockout in the two screens and not of the motif's targets.

Usage:  python 09x_cross_line_control.py top90chip            # ~10 min, loads both screens
        python 09x_cross_line_control.py top90chip --replot   # redraw from the cache, seconds

  results/prof_plan/fig_reader_curves/cross_line_control_{tag}.{pdf,png}
  results/prof_plan/cross_line_control_{tag}.csv
  results/prof_plan/cross_line_control_{tag}_plotcache.pkl
"""
import os, sys, json, pickle, string
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

ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
TAG = ARGS[0] if ARGS else "top90chip"
REPLOT = "--replot" in sys.argv          # redraw from the cache, skipping the 10-minute load
BIND_FILE = {"top10chip": "reader_binding.parquet",
             "top90chip": "reader_binding_top90.parquet",
             "allchip": "reader_binding_all.parquet"}[TAG]
MIN_GENES = 300
N_EX = 4                # example panels per row
N_CACHE = 12            # example curves kept per group, so the figure can be redrawn offline

HERE = os.path.dirname(os.path.abspath(__file__))
if not REPLOT:
    src = open(os.path.join(HERE, "08c_alltargets_stats.py")).read()
    exec(compile(src.split("DEFS = [")[0], "08c_prefix", "exec"), globals())
else:
    ROOT = ENHANCER_ROOT
    CHIP = os.path.join(ROOT, "multi_agent/chipseq_enrichment")

OUT = os.path.join(CHIP, "results/prof_plan")
FIG = os.path.join(OUT, "fig_reader_curves")
RS = os.path.join(CHIP, "reader_switch/out")
CACHE = f"{OUT}/cross_line_control_{TAG}_plotcache.pkl"
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
CK, CH = "#4878a8", "#b06a72"
CR, CC, CX = "#1f5f8b", "#c08a2e", "#9a9691"     # readers, K562/HEK ChIP TFs, other ChIP TFs


def roll(y, w):
    return pd.Series(y).rolling(w, center=True, min_periods=max(3, w // 4)).median().to_numpy()


# ================================================================================ compute
if not REPLOT:
    SHARED = sorted(shared)
    POS = {g: i for i, g in enumerate(SHARED)}
    kidx = np.array([k_col[g] for g in SHARED])
    oidx = np.array([o_col[g] for g in SHARED])

    roster = {m: sorted({norm(t) for t in v}) for m, v in json.load(open(os.path.join(
        ROOT, "multi_agent/posthoc_analysis/results/step06_JASPAR_assignments.json"))).items()}
    ROSTER_ALL = {t for v in roster.values() for t in v}
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

    print("Building promoter binding (sets the gene order only) ...", flush=True)
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
        M = np.vstack([effect(t, cell)[idx] for t in BGTF[cell]])
        NB = np.vstack([[g not in BOUND[(cell, t)] for g in genes] for t in BGTF[cell]])
        with np.errstate(invalid="ignore"):
            md = np.nanmedian(np.where(NB, M, np.nan), 0)
        md[NB.sum(0) < 20] = np.nan
        return md

    # ---- motif gene sets, identical to 09w -----------------------------------------------------
    rb = pd.read_parquet(f"{RS}/{BIND_FILE}")
    rb = rb[rb["cell"].isin(LINES)].copy()
    rb["rn"] = rb["reader"].map(norm)
    rb = rb[rb["ensembl"].isin(set(SHARED))]
    sets = {}
    for (cell, m, rn), d in rb.groupby(["cell", "motif_id", "rn"]):
        if m in roster and rn in roster[m] and rn in assayed[cell]:
            sets[(cell, m, rn)] = set(d["ensembl"])

    MOTIF = {}
    for m in sorted(roster, key=lambda x: int(x.split("_")[1])):
        genes = sorted(set().union(*[sets[(c, mm, r)] for (c, mm, r) in sets if mm == m]) or set())
        if len(genes) < MIN_GENES:
            continue
        MOTIF[m] = (genes, np.array([POS[g] for g in genes]))
    print(f"\n[{TAG}] {len(MOTIF)} motifs with >= {MIN_GENES} target genes")

    # ---- control pools -------------------------------------------------------------------------
    # Every control must be a genuine transcription factor, so a control is admitted only if it has
    # ENCODE4 ChIP-seq in at least one cell line.  Perturbations of genes that were never ChIP'd
    # anywhere are excluded: they are not necessarily transcription factors at all.
    print("Selecting control TFs (ChIP-seq in at least one line, knockdown in both screens) ...",
          flush=True)
    chip_any = set()
    for cell in sorted(os.listdir(ENCODE4)):
        p = os.path.join(ENCODE4, cell)
        if os.path.isdir(p):
            chip_any |= {norm(d.replace("_human", "")) for d in os.listdir(p)
                         if os.path.isdir(os.path.join(p, d))}
    chip_kh = assayed["K562"] | assayed["HEK293"]
    print(f"  {len(chip_any)} TFs have ENCODE4 ChIP somewhere, {len(chip_kh)} in K562 or HEK293")

    cand = sorted((k_perturbed & o_perturbed) - ROSTER_ALL)
    both_eff = [t for t in cand if eff_here(t, "K562") and eff_here(t, "HEK293")]
    _kraw.clear()                               # the effective-knockdown scan caches every TF
    CTRL = {"chip": [t for t in both_eff if t in chip_kh],
            "chip_other": [t for t in both_eff if t in chip_any and t not in chip_kh]}
    print(f"  {len(both_eff)} non-reader perturbations with effective knockdown in both screens; "
          f"{len(CTRL['chip']) + len(CTRL['chip_other'])} of them are ChIP-verified TFs "
          f"({len(CTRL['chip'])} assayed in K562 or HEK293, {len(CTRL['chip_other'])} only "
          f"elsewhere).  The remaining "
          f"{len(both_eff) - len(CTRL['chip']) - len(CTRL['chip_other'])} are excluded.")

    def record(tf, m, group, chip_in):
        genes, idx = MOTIF[m]
        vK, vH = effect(tf, "K562"), effect(tf, "HEK293")
        ok = np.isfinite(vK) & np.isfinite(vH)
        okt = ok[idx]
        if okt.sum() < 50:
            return None
        mK, mH = float(np.median(vK[idx][okt])), float(np.median(vH[idx][okt]))
        gK, gH = float(np.median(vK[ok])), float(np.median(vH[ok]))   # genome wide, same factor
        return dict(motif=m, factor=tf, group=group, chip_in=chip_in, n_genes=int(okt.sum()),
                    median_K562=mK, median_HEK293=mH,
                    log2_target=float(np.log2(mK / mH)),
                    log2_global=float(np.log2(gK / gH)),
                    log2_spec=float(np.log2((mK / gK) / (mH / gH))),
                    P=float(stats.wilcoxon(vK[idx][okt], vH[idx][okt]).pvalue))

    rows = []
    for m in MOTIF:
        chip = {r for (c, mm, r) in sets if mm == m}
        for r in sorted(chip):
            if eff_here(r, "K562") and eff_here(r, "HEK293"):
                where = ("both" if all((c, m, r) in sets for c in LINES)
                         else ("K562" if ("K562", m, r) in sets else "HEK293"))
                rec = record(r, m, "reader", where)
                if rec:
                    rows.append(rec)
    for grp in ("chip", "chip_other"):
        for t in CTRL[grp]:
            for m in MOTIF:
                rec = record(t, m, grp, "-")
                if rec:
                    rows.append(rec)
        print(f"  scored {len(CTRL[grp])} {grp} factors on {len(MOTIF)} motif gene sets", flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(f"{OUT}/cross_line_control_{TAG}.csv", index=False)

    # ---- example curves, cached so the figure can be redrawn without reloading -----------------
    C_ = D[D.group != "reader"]        # every ChIP-verified non-reader TF
    R_ = D[D.group == "reader"]
    keep = pd.concat([
        C_.sort_values("log2_target", key=lambda s: s.abs(), ascending=False)
          .drop_duplicates("factor").head(N_CACHE),
        C_.sort_values("log2_target", key=lambda s: s.abs()).drop_duplicates("factor").head(N_CACHE),
        R_.sort_values("log2_target", key=lambda s: s.abs()).head(N_CACHE),
        R_.sort_values("log2_target", key=lambda s: s.abs(), ascending=False).head(N_CACHE)])
    print(f"\nCaching {len(keep)} example curves ...", flush=True)
    ORDER = {}
    CURVES = {}
    for r in keep.itertuples(index=False):
        genes, idx = MOTIF[r.motif]
        if r.motif not in ORDER:
            pk = perturbability("K562", genes, idx)
            ph = perturbability("HEK293", genes, idx)
            ORDER[r.motif] = np.argsort(-np.nan_to_num(np.nanmean([pk, ph], 0), nan=-1))
        o = ORDER[r.motif]
        wdw = max(15, len(o) // 50)
        CURVES[(r.factor, r.motif)] = tuple(
            roll(effect(r.factor, c)[idx][o], wdw).astype(np.float32) for c in LINES)
    with open(CACHE, "wb") as fh:
        pickle.dump(dict(D=D, CURVES=CURVES, tag=TAG), fh)
    print(f"wrote -> {CACHE}")
else:
    with open(CACHE, "rb") as fh:
        _c = pickle.load(fh)
    D, CURVES = _c["D"], _c["CURVES"]
    print(f"replotting from {CACHE}: {len(D)} records, {len(CURVES)} cached curves")

# ================================================================================ statistics
R = D[D.group == "reader"]
C = D[D.group == "chip"]
X = D[D.group == "chip_other"]
A = D[D.group != "reader"]                # every ChIP-verified non-reader TF
# one value per factor: control TFs are scored on every motif gene set, readers on their own
by = lambda S, col: S.groupby("factor")[col].apply(lambda s: s.abs().median())
cf, xf, af = by(C, "log2_target"), by(X, "log2_target"), by(A, "log2_target")
cs, xs, asp = by(C, "log2_spec"), by(X, "log2_spec"), by(A, "log2_spec")
rv, rs = R.log2_target.abs(), R.log2_spec.abs()
GRP = [("universal-motif readers", rv, rs, CR),
       ("non-readers, ChIP in K562/HEK293", cf, cs, CC),
       ("non-readers, ChIP elsewhere", xf, xs, CX)]

print("\n=== cross-line difference in effect magnitude, |log2 K562/HEK293| ===")
print("    (controls are ChIP-verified TFs only; perturbations never ChIP'd anywhere are excluded)")
for lab, a, _, _c in GRP + [("all non-reader TFs", af, asp, None)]:
    print(f"  {lab:34} n={len(a):4}  median {a.median():.3f} ({2**a.median():.2f}x)   "
          f"IQR {2**a.quantile(.25):.2f}-{2**a.quantile(.75):.2f}x   "
          f"90th pct {2**a.quantile(.90):.2f}x")
P_ALL = stats.mannwhitneyu(rv, af).pvalue
P_CHIP = stats.mannwhitneyu(rv, cf).pvalue
P_OTHER = stats.mannwhitneyu(rv, xf).pvalue
print(f"  readers vs all non-reader TFs        MWU P = {P_ALL:.3g}")
print(f"  readers vs those ChIP'd in K562/HEK  MWU P = {P_CHIP:.3g}")
print(f"  readers vs those ChIP'd elsewhere    MWU P = {P_OTHER:.3g}")
PCT = float(np.median([stats.percentileofscore(af, v) for v in rv]))
FRAC = float(np.mean(af >= rv.median()))
print(f"  the median reader sits at the {PCT:.0f}th percentile of the control TFs; "
      f"{100*FRAC:.0f}% of control TFs differ by at least as much as the median reader "
      f"({2**rv.median():.2f}x)")

print("\n  --- are there legitimate non-reader TFs as consistent as the most consistent readers? ---")
print("  most consistent non-reader TFs: " +
      ", ".join(f"{f} {2**v:.2f}x" for f, v in af.nsmallest(12).items()))
print("  most consistent readers:        " +
      ", ".join(f"{r.factor} {2**abs(r.log2_target):.2f}x" for r in
                R.reindex(rv.nsmallest(8).index).itertuples()))
print("  least consistent non-reader TFs: " +
      ", ".join(f"{f} {2**v:.1f}x" for f, v in af.nlargest(6).items()))
print("  least consistent readers:        " +
      ", ".join(f"{r.factor} {2**abs(r.log2_target):.1f}x" for r in
                R.reindex(rv.nlargest(6).index).itertuples()))
for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90):
    print(f"    {int(q*100):3}th percentile: readers {2**rv.quantile(q):.2f}x   "
          f"non-reader TFs {2**af.quantile(q):.2f}x")

print("\n  per motif, gene set held exactly fixed:")
pm = []
for m in sorted(D.motif.unique(), key=lambda x: int(x.split("_")[1])):
    r = R[R.motif == m].log2_target.abs()
    c = A[A.motif == m].log2_target.abs()
    if not len(r):
        continue
    pm.append(dict(motif=m, n_readers=len(r), reader_med=r.median(), ctrl_med=c.median(),
                   reader_pctile=float(np.mean([stats.percentileofscore(c, v) for v in r]))))
    print(f"    {m.replace('universal_','M'):5} n={len(r):2}  readers {r.median():.3f}  "
          f"controls {c.median():.3f}  readers at the {pm[-1]['reader_pctile']:.0f}th percentile")
PM = pd.DataFrame(pm)
W = stats.wilcoxon(PM.reader_med, PM.ctrl_med).pvalue
NABOVE = int((PM.reader_med > PM.ctrl_med).sum())
print(f"  paired over {len(PM)} motifs: readers above controls in {NABOVE}/{len(PM)}, "
      f"Wilcoxon P = {W:.3g}")

print("\n=== does the difference belong to the targets or to the knockout? ===")
for lab, S in (("universal-motif readers", R), ("non-reader TFs", A)):
    rho = stats.spearmanr(S.log2_target, S.log2_global).statistic
    print(f"  {lab:26} target vs genome-wide ratio rho = {rho:.3f}; dividing the genome-wide "
          f"effect out leaves {2**S.log2_spec.abs().median():.2f}x "
          f"(was {2**S.log2_target.abs().median():.2f}x)")
RHO_ALL = stats.spearmanr(D.log2_global, D.log2_target).statistic
P_SPEC = stats.mannwhitneyu(rs, asp).pvalue
print(f"  readers vs control TFs on the target-specific ratio: {2**rs.median():.3f}x vs "
      f"{2**asp.median():.3f}x, MWU P = {P_SPEC:.3g}")

asc = A.sort_values("log2_target", key=lambda s: s.abs())
low = asc.drop_duplicates("factor").head(N_EX)                          # most consistent controls
desc = A.sort_values("log2_target", key=lambda s: s.abs(), ascending=False)
high = desc.drop_duplicates("factor").head(N_EX)                        # most different controls
rl = R.sort_values("log2_target", key=lambda s: s.abs()).head(N_EX)     # most consistent readers

# ================================================================================ figure
nrow = 2 + 3
fig = plt.figure(figsize=(7.09, 1.55 * nrow), facecolor="white")
gs = fig.add_gridspec(nrow, 4, hspace=1.05, wspace=0.55)


def ecdf(ax, v, color, label):
    v = np.sort(np.asarray(v))
    ax.plot(v, np.linspace(0, 100, len(v)), color=color, lw=1.0, label=label)


# a: distribution of the cross-line difference, one value per factor
ax = fig.add_subplot(gs[0, :2])
for lab, a, _, col in GRP:
    ecdf(ax, a, col, f"{lab} (n = {len(a)})")
ax.set_xlim(0, 2)
ax.set_ylim(0, 140)
ax.set_yticks([0, 50, 100])
ax.set_xlabel("|log$_2$| difference in effect between K562 and HEK293", color=INK, labelpad=1.5)
ax.set_ylabel("% of factors", color=INK, labelpad=1.5)
ax.legend(frameon=False, loc="lower right", handlelength=1.2)
ax.text(0.03, 0.99, f"non-reader TFs span the same range;\nthe median reader sits at the "
        f"{PCT:.0f}th percentile of them", transform=ax.transAxes, fontsize=5, color=MUT,
        va="top", ha="left")

# b: target-set ratio vs genome-wide ratio
ax = fig.add_subplot(gs[0, 2:])
lim = 2.2
ax.plot([-lim, lim], [-lim, lim], color=MUT, lw=0.5, ls=(0, (3, 2)), zorder=1)
ax.axhline(0, color=MUT, lw=0.4, zorder=1)
ax.axvline(0, color=MUT, lw=0.4, zorder=1)
for S, c, ms in ((X, CX, 1.6), (C, CC, 1.6), (R, CR, 2.6)):
    ax.plot(S.log2_global, S.log2_target, "o", color=c, ms=ms, mew=0, alpha=0.5, zorder=2)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("log$_2$ K562/HEK293, all genes", color=INK, labelpad=1.5)
ax.set_ylabel("motif target genes", color=INK, labelpad=1.5)
ax.text(0.97, 0.06, f"$\\rho$ = {RHO_ALL:.2f}: the difference belongs\nto the knockout, not to "
        f"the motif's targets", transform=ax.transAxes, fontsize=5, color=MUT, va="bottom",
        ha="right")

# c: target-specific ratio (each factor's genome-wide effect divided out)
ax = fig.add_subplot(gs[1, :2])
for lab, _, s, col in GRP:
    ecdf(ax, s, col, lab)
ax.set_xlim(0, 0.25)
ax.set_ylim(0, 140)
ax.set_yticks([0, 50, 100])
ax.set_xlabel("|log$_2$| difference after dividing by each factor's genome-wide effect",
              color=INK, labelpad=1.5)
ax.set_ylabel("% of factors", color=INK, labelpad=1.5)
ax.legend(frameon=False, loc="lower right", handlelength=1.2)
ax.text(0.03, 0.99, f"readers {2**rs.median():.2f}x, control TFs {2**asp.median():.2f}x:\n"
        f"nothing left once the knockout's own scale is removed",
        transform=ax.transAxes, fontsize=5, color=MUT, va="top", ha="left")

# d: per motif, readers against controls on identical genes
ax = fig.add_subplot(gs[1, 2:])
for r in PM.itertuples():
    ax.plot([0, 1], [r.ctrl_med, r.reader_med], color=MUT, lw=0.5, alpha=0.8, zorder=1)
ax.plot(np.zeros(len(PM)), PM.ctrl_med, "o", color=CC, ms=2.6, mew=0, zorder=3)
ax.plot(np.ones(len(PM)), PM.reader_med, "o", color=CR, ms=2.6, mew=0, zorder=3)
ax.set_xlim(-0.35, 1.35)
ax.set_ylim(-0.03, max(PM.reader_med.max(), PM.ctrl_med.max()) * 1.45)
ax.set_xticks([0, 1])
ax.set_xticklabels(["non-reader TFs", "readers"])
ax.set_ylabel("median |log$_2$|", color=INK, labelpad=1.5)
ax.set_xlabel(f"{len(PM)} motifs, same genes on both sides", color=INK, labelpad=1.5)
ax.text(0.02, 0.99, f"readers above the control TFs of the same motif\nin {NABOVE} of {len(PM)} "
        f"motifs, P = {W:.2f}", transform=ax.transAxes, fontsize=5, color=MUT, va="top",
        ha="left")

# example curves
EX = ([(r, "control") for r in high.itertuples()] +
      [(r, "control") for r in low.itertuples()] +
      [(r, "reader") for r in rl.itertuples()])
for j, (r, kind) in enumerate(EX):
    ax = fig.add_subplot(gs[2 + j // 4, j % 4])
    yK, yH = CURVES[(r.factor, r.motif)]
    x = np.arange(1, len(yK) + 1)
    ax.plot(x, yK, color=CK, lw=0.9)
    ax.plot(x, yH, color=CH, lw=0.9)
    ax.set_yscale("log")
    ax.set_xlim(1, len(yK))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=SUB, labelcolor=INK, pad=1.5)
    ax.set_xlabel("genes, ordered by perturbability", color=INK, labelpad=1.5, fontsize=5)
    ax.set_ylabel("|effect| (QN)", color=INK, labelpad=1.5)
    ax.text(-0.30, 1.20, string.ascii_lowercase[4 + j], transform=ax.transAxes, fontsize=7,
            fontweight="bold", color=INK, va="top", ha="left")
    ax.text(-0.15, 1.20, r.factor, transform=ax.transAxes, fontsize=7, fontweight="bold",
            color=CR if kind == "reader" else INK, va="top", ha="left")
    ax.text(1.0, 1.20, f"{r.motif.replace('universal_','M')} · {2**abs(r.log2_target):.1f}x"
            + ("" if kind == "control" else " · reader"),
            transform=ax.transAxes, fontsize=5, color=SUB, va="top", ha="right")

HEAD = [("Non-readers too", -0.13, -0.20), ("Whole genome", -0.16, -0.23),
        ("Target specific", -0.13, -0.20), ("Motif by motif", -0.14, -0.21)]
for i, (t, x1, x2) in enumerate(HEAD):
    fig.axes[i].text(x1, 1.26, t, transform=fig.axes[i].transAxes,
                     fontsize=7, fontweight="bold", color=INK, va="top", ha="left")
    fig.axes[i].text(x2, 1.26, "abcd"[i], transform=fig.axes[i].transAxes,
                     fontsize=7, fontweight="bold", color=INK, va="top", ha="left")
for ax in fig.axes[:4]:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=SUB, labelcolor=INK, pad=1.5)
fig.text(0.005, -0.045, "e-h: non-reader TFs with the largest cross-line difference.  "
         "i-l: non-reader TFs with the smallest.  m-p: the readers with the smallest.  "
         "All are scored on the same motif target genes as the readers in the companion figure; "
         "same gene at the same x in both lines, rolling median.",
         fontsize=5, color=MUT, ha="left", va="bottom")
fig.legend(handles=[Line2D([], [], color=CK, lw=1.2, label="K562"),
                    Line2D([], [], color=CH, lw=1.2, label="HEK293")],
           frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.02))
for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/cross_line_control_{TAG}.{ext}", dpi=450, transparent=False,
                facecolor="white", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"\nwrote -> {FIG}/cross_line_control_{TAG}.pdf/.png")
if not REPLOT:
    print(f"wrote -> {OUT}/cross_line_control_{TAG}.csv")
