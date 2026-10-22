#!/usr/bin/env python
"""
09k -- reader curves on genes ChIP-bound by EVERY reader shown, no background.

Changes from 09h, all requested by the user:
  * gene set is the UNION: every gene drawn is ChIP-bound (top-90%) by at least one reader of
    that motif, and every reader is drawn across the whole set;
  * the non-binding background band is gone -- the panel shows the readers only;
  * genes are ordered by GENERAL PERTURBABILITY -- the median |effect| of knockouts of TFs that
    do NOT bind that gene.  The background is not drawn (the user asked for it gone), it only
    sets the order.  Ordering by the readers' own median would be circular: with 2-3 readers each
    curve largely IS that median, so it comes out as a smooth monotonic line purely by
    construction (HINFP was the median reader at 40% of genes, Spearman +0.60 vs the sort
    variable, and drew as a perfectly straight curve), while 9-11 reader panels stayed jagged.
    An external ordering treats every panel the same regardless of how many readers it has.

No reader is ever dropped: every eligible reader of the motif appears in its panel.

Alongside the figure, `union_dominance_{tag}.csv` carries the numbers behind "is there a
dominant reader": each reader's median |effect| on the genes IT binds, its median effect on
genes it does NOT bind in that line (OFF), and the difference.  A reader that is genuinely
dominant AT THIS MOTIF is high on the co-bound genes relative to its own OFF level; a reader
that is simply a consequential knockout in that cell line is high on both.

  results/prof_plan/fig_reader_curves/union_reader_curves_{tag}.{pdf,png}      6 panels
  results/prof_plan/fig_reader_curves/union_reader_curves_all_{tag}.{pdf,png}  every panel
  results/prof_plan/union_dominance_{tag}.csv
"""
import os, sys, json, pickle, string
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = next((a for a in sys.argv[1:] if not a.startswith("-")), "top90chip")
BIND_FILE = {"top10chip": "reader_binding.parquet",
             "top90chip": "reader_binding_top90.parquet",
             "allchip": "reader_binding_all.parquet"}[TAG]
MIN_GENES = 100       # a panel needs this many genes bound by at least one reader
MIN_READERS = 2
MIN_READER_SET = 1    # no minimum: every reader with any ChIP target gene at this motif
N_MAIN = 6

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "08c_alltargets_stats.py")).read()
exec(compile(src.split("DEFS = [")[0], "08c_prefix", "exec"), globals())

OUT = os.path.join(CHIP, "results/prof_plan")
FIG = os.path.join(OUT, "fig_reader_curves")

# the eight motifs carried by the cross-cell-line switch figures; empty set = every motif
MOTIF_SUBSET = {"universal_29", "universal_14", "universal_1", "universal_12",
                "universal_35", "universal_19", "universal_36", "universal_15"}
if "--all" in sys.argv:          # every qualifying motif x line panel (the SI expansion of Fig 5d)
    MOTIF_SUBSET = set()
SUFFIX = "_8motifs" if MOTIF_SUBSET else ""
RS = os.path.join(CHIP, "reader_switch/out")
os.makedirs(FIG, exist_ok=True)
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
PAL = ["#4878a8", "#c2884a", "#5f9a6a", "#8a72a8", "#b06a72",
       "#6a9bb0", "#a8935f", "#8d6e63", "#4f6d7a", "#9c7b9c"]

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
        _e[(tf, cell)] = (k_delta(tf)[kidx] if cell == "K562" else o_delta(tf)[oidx]) \
            if eff_here(tf, cell) else None
    return _e[(tf, cell)]


def roll(y, w):
    return pd.Series(y).rolling(w, center=True, min_periods=max(3, w // 4)).median().to_numpy()


def panel_label(k):
    return string.ascii_lowercase[k] if k < 26 else \
        f"{string.ascii_lowercase[k % 26]}{k // 26}"


# ---- promoter binding for every knocked-down ChIP'd TF (sets the gene order only) -----------
print("\nBuilding promoter binding for the gene ordering ...", flush=True)
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
    print(f"  {cell:8} {len(BGTF[cell])} TFs")


def perturbability(cell, genes, idx):
    """Median |effect| at each gene over knockouts of TFs that do NOT bind it."""
    M = np.vstack([np.abs(effect(t, cell)[idx]) for t in BGTF[cell]])
    NB = np.vstack([[g not in BOUND[(cell, t)] for g in genes] for t in BGTF[cell]])
    with np.errstate(invalid="ignore"):
        md = np.nanmedian(np.where(NB, M, np.nan), 0)
    md[NB.sum(0) < 20] = np.nan
    return md


# ---- binding sets ------------------------------------------------------------
rb = pd.read_parquet(f"{RS}/{BIND_FILE}")
rb = rb[rb["cell"].isin(LINES)].copy()
rb["rn"] = rb["reader"].map(norm)
rb = rb[rb["ensembl"].isin(set(SHARED))]
sets = {}
for (cell, m, rn), d in rb.groupby(["cell", "motif_id", "rn"]):
    if m in roster and rn in roster[m] and rn in assayed[cell] and eff_here(rn, cell):
        g = set(d["ensembl"])
        if len(g) >= MIN_READER_SET:
            sets[(cell, m, rn)] = g

print(f"\n[{TAG}] union of reader-bound genes per motif x line "
      f"(every gene is ChIP-bound by at least one reader shown)")
panels = []
for cell in LINES:
    for m in sorted(roster, key=lambda x: int(x.split("_")[1])):
        if MOTIF_SUBSET and m not in MOTIF_SUBSET:
            continue
        rs = sorted(r for (c, mm, r) in sets if c == cell and mm == m)
        if len(rs) < MIN_READERS:
            continue
        union = sorted(set().union(*(sets[(cell, m, r)] for r in rs)))
        if len(union) < MIN_GENES:
            continue
        panels.append(dict(cell=cell, motif=m, readers=rs, genes=union,
                           dropped=[], n_union=len(union), n_all_readers=len(rs)))
        print(f"  {cell:8}{m:14}{len(union):6d} genes bound by any of {len(rs)} readers")
panels.sort(key=lambda p: -len(p["genes"]))
print(f"\n{len(panels)} panels")

# ---- per-reader dominance numbers -------------------------------------------------------------
rows = []
for p in panels:
    cell, m = p["cell"], p["motif"]
    idx = np.array([POS[g] for g in p["genes"]])
    p["E"] = {}
    for r in p["readers"]:
        v = np.abs(effect(r, cell))
        own = np.array([POS[g] for g in sorted(sets[(cell, m, r)]) if g in POS])
        on = v[own]                      # this reader's OWN bound genes
        p["E"][r] = v[idx]               # the curve still spans the whole union
        offmask = np.ones(len(SHARED), bool)
        offmask[[POS[g] for g in sets[(cell, m, r)] if g in POS]] = False
        off = v[offmask]
        rows.append(dict(cell=cell, motif=m, reader=r,
                         n_union=len(idx), n_own=len(own),
                         median_on=float(np.nanmedian(on)),
                         median_off=float(np.nanmedian(off)),
                         log2_on_over_off=float(np.log2(np.nanmedian(on) /
                                                        np.nanmedian(off))),
                         ))
    # rank readers by height, and by height corrected for their own OFF level
    sub = [x for x in rows if x["cell"] == cell and x["motif"] == m]
    o1 = sorted(sub, key=lambda x: -x["median_on"])
    o2 = sorted(sub, key=lambda x: -x["log2_on_over_off"])
    p["rank_raw"] = [x["reader"] for x in o1]
    p["rank_corr"] = [x["reader"] for x in o2]
    p["order"] = np.argsort(-np.nan_to_num(perturbability(cell, p["genes"], idx), nan=-1))
    p["w"] = max(9, len(idx) // 40)

D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/union_dominance{SUFFIX}_{TAG}.csv", index=False)

print("\nApparent dominance (tallest curve) vs dominance after each reader's own OFF level")
print(f"  {'cell':8}{'motif':12}{'n':>6}  by height -> | corrected ->")
for p in panels:
    print(f"  {p['cell']:8}{p['motif'].replace('universal_','M'):12}{len(p['genes']):6d}  "
          f"{','.join(p['rank_raw'][:4]):38} | {','.join(p['rank_corr'][:4])}")
agree = sum(p["rank_raw"][0] == p["rank_corr"][0] for p in panels)
print(f"\n  top reader unchanged by the correction in {agree}/{len(panels)} panels")
print(f"  median log2(ON/OFF) over all readers shown: {D.log2_on_over_off.median():+.4f}")


# ---- plot cache: everything draw() needs, so a restyle is a seconds-long replot ----------------
_cache = [dict(cell=p["cell"], motif=p["motif"], readers=list(p["readers"]),
               n=len(p["genes"]), w=p["w"],
               genes=[p["genes"][i] for i in p["order"]],          # 2026-10-05: gene IDs in the drawn order, for audit

               curves={r: np.asarray(p["E"][r][p["order"]], dtype=np.float32)
                       for r in p["readers"]})
          for p in panels]
with open(f"{OUT}/union_curves{SUFFIX}_{TAG}_plotcache.pkl", "wb") as fh:
    pickle.dump(_cache, fh, protocol=4)
print(f"\ncached {len(_cache)} panels -> {OUT}/union_curves{SUFFIX}_{TAG}_plotcache.pkl")


# ---- drawing -----------------------------------------------------------------------------------
def draw(sel, fname, ncol=3):
    n = len(sel)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.09, 1.62 * nrow), squeeze=False,
                             facecolor="white")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for k, p in enumerate(sel):
        ax = axes[k // ncol][k % ncol]
        N, w, o = len(p["genes"]), p["w"], p["order"]
        x = np.arange(1, N + 1)
        for j, r in enumerate(p["readers"]):
            ax.plot(x, roll(p["E"][r][o], w), color=PAL[j % len(PAL)], lw=0.8, label=r)
        ax.set_yscale("log")
        ax.set_xlim(1, N)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=SUB, labelcolor=INK, pad=1.5)
        ax.set_ylabel("|effect| (QN)", color=INK, labelpad=1.5)
        ax.set_xlabel("genes, ordered by general perturbability", color=INK,
                      labelpad=1.5, fontsize=5.5)
        ax.text(-0.20, 1.14, panel_label(k), transform=ax.transAxes, fontsize=7,
                fontweight="bold", color=INK, va="top", ha="left")
        ax.text(-0.10, 1.14, p["motif"].replace("universal_", "Motif "),
                transform=ax.transAxes, fontsize=7, fontweight="bold", color=INK,
                va="top", ha="left")
        ax.text(1.0, 1.14, f"{p['cell']} · n={N:,} · {len(p['readers'])} readers",
                transform=ax.transAxes, fontsize=5, color=SUB, va="top", ha="right")
        ax.text(0.98, 0.03, f"rolling median, {w}", transform=ax.transAxes,
                fontsize=5, color=MUT, va="bottom", ha="right")
        # curves run top-left to bottom-right, so the lower-left corner is the free one; a
        # translucent white box keeps the labels readable in the crowded panels regardless.
        ax.legend(loc="lower left", ncol=1 if len(p["readers"]) <= 5 else 2,
                  fontsize=4.5, handlelength=1.0, labelspacing=0.2, borderpad=0.25,
                  columnspacing=0.9, frameon=True, facecolor="white", edgecolor="none",
                  framealpha=0.72)
    fig.tight_layout(pad=0.55, w_pad=1.4, h_pad=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}/{fname}.{ext}", dpi=450, transparent=False, facecolor="white",
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  -> {FIG}/{fname}.pdf/.png")


print()
draw(panels, f"union_reader_curves{SUFFIX}_{TAG}")
print(f"\nwrote -> {FIG}")
