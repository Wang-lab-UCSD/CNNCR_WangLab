#!/usr/bin/env python
"""
09q -- how many readers effectively share a motif?  No cut-offs anywhere.

09p had to draw a line ("within 80% of the leader") to count a leading group, and the answer
moved with the line.  This removes the line entirely.

Treat the readers' median |effect| as a distribution: normalise them to sum to 1 and take the
Hill numbers, the standard "effective number" measures.

    D1 = exp(-sum p_i ln p_i)      Shannon
    D2 = 1 / sum p_i^2             inverse Simpson, harder on the leader

D is the number of EQUALLY STRONG readers that would produce the same spread:
    D ~ 1     one reader does essentially all of it
    D ~ 2     two readers share it
    D ~ n     every reader contributes equally -- fully redundant

Nothing is thresholded: D is read off the data, and evenness = D/n places a motif on the scale
from a single dominant reader (1/n) to complete redundancy (1.0).  Confidence intervals come
from resampling GENES, so a reviewer can see whether a motif's D is resolved at all.

Figures: the per-motif profiles with D printed on each (no cut-off line), and one summary panel
placing every motif on the redundancy scale.

  results/prof_plan/effective_readers_{tag}.csv
  results/prof_plan/fig_reader_curves/effective_readers_{tag}.{pdf,png}
"""
import os, sys, json, pickle, string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = sys.argv[1] if len(sys.argv) > 1 else "top90chip"
BIND_FILE = {"top10chip": "reader_binding.parquet",
             "top90chip": "reader_binding_top90.parquet",
             "allchip": "reader_binding_all.parquet"}[TAG]
MIN_READER_SET, MIN_GENES, MIN_READERS, N_BOOT = 1, 100, 2, 400

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "08c_alltargets_stats.py")).read()
exec(compile(src.split("DEFS = [")[0], "08c_prefix", "exec"), globals())

OUT = os.path.join(CHIP, "results/prof_plan")
FIG = os.path.join(OUT, "fig_reader_curves")
RS = os.path.join(CHIP, "reader_switch/out")
LINES = ["K562", "HEK293"]
RNG = np.random.default_rng(0)

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
ACC, ACC2 = "#4878a8", "#b06a72"

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
        _e[(tf, cell)] = np.abs(k_delta(tf)[kidx] if cell == "K562"
                                else o_delta(tf)[oidx]) if eff_here(tf, cell) else None
    return _e[(tf, cell)]

def ratios(med):
    """Two threshold-free summaries, both read straight off the profile.

    lead/2nd  separates one dominant reader from two co-dominant ones
    lead/rest separates any leading group from full redundancy
    Together they place a motif on a continuum -- 1 dominant reader (both high), 2 co-dominant
    (lead/2nd ~ 1, lead/rest > 1), fully redundant (both ~ 1) -- with nothing binned."""
    return float(med[0] / med[1]), float(med[0] / np.median(med[1:]))


def hill(med):
    p = med / med.sum()
    D1 = float(np.exp(-(p * np.log(p)).sum()))
    D2 = float(1.0 / (p ** 2).sum())
    return D1, D2

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

panels, rows = [], []
for cell in LINES:
    for m in sorted(roster, key=lambda x: int(x.split("_")[1])):
        rs = sorted(r for (c, mm, r) in sets if c == cell and mm == m)
        if len(rs) < MIN_READERS:
            continue
        genes = sorted(set().union(*(sets[(cell, m, r)] for r in rs)))
        if len(genes) < MIN_GENES:
            continue
        idx = np.array([POS[g] for g in genes])
        M = np.vstack([effect(r, cell)[idx] for r in rs])
        M = M[:, np.isfinite(M).all(0)]
        med = np.median(M, 1)
        o = np.argsort(-med)
        med, names = med[o], [rs[i] for i in o]
        D1, D2 = hill(med)
        r2, rr = ratios(med)
        bs = np.array([ratios(np.sort(np.median(
            M[:, RNG.integers(0, M.shape[1], M.shape[1])], 1))[::-1]) for _ in range(N_BOOT)])
        panels.append(dict(cell=cell, motif=m, rel=med / med[0], names=names,
                           D1=D1, lead_2nd=r2, lead_rest=rr, n=len(rs),
                           rest_lo=float(np.percentile(bs[:, 1], 2.5)),
                           rest_hi=float(np.percentile(bs[:, 1], 97.5))))
        rows.append(dict(cell=cell, motif=m, n_readers=len(rs), n_genes=M.shape[1],
                         lead_over_2nd=r2, lead_over_rest=rr,
                         lead_over_rest_lo=float(np.percentile(bs[:, 1], 2.5)),
                         lead_over_rest_hi=float(np.percentile(bs[:, 1], 97.5)),
                         D1_effective_readers=D1, D2_inv_simpson=D2,
                         evenness=D1 / len(rs), leader=names[0],
                         order=",".join(names),
                         rel_profile=",".join(f"{x:.3f}" for x in med / med[0])))
D = pd.DataFrame(rows).sort_values("lead_over_rest", ascending=False)
D.to_csv(f"{OUT}/dominant_reader_{TAG}.csv", index=False)

print(f"\n[{TAG}]  IS THERE A DOMINANT READER?   ({len(D)} panels, no cut-offs anywhere)")
print(f"  lead/2nd  ~1 means the top two are equals;  lead/rest ~1 means nobody leads at all")
print(f"  both high = one dominant reader;  lead/2nd~1 with lead/rest>1 = two co-dominant\n")
print(f"  {'cell':8}{'motif':7}{'rdrs':>5}{'lead/2nd':>10}{'lead/rest':>11}{'95% CI':>16}"
      f"   leader")
for r in D.itertuples(index=False):
    print(f"  {r.cell:8}{r.motif.replace('universal_','M'):7}{r.n_readers:5d}"
          f"{r.lead_over_2nd:10.2f}{r.lead_over_rest:11.2f}"
          f"  [{r.lead_over_rest_lo:5.2f},{r.lead_over_rest_hi:5.2f}]   {r.leader}")
print(f"\n  lead/rest spans {D.lead_over_rest.min():.2f}-{D.lead_over_rest.max():.2f} "
      f"(1.0 = no leader); lead/2nd spans {D.lead_over_2nd.min():.2f}-{D.lead_over_2nd.max():.2f}")
print(f"  NOTE the Hill number D saturates here (D/n {D.evenness.min():.2f}-{D.evenness.max():.2f}, "
      f"median {D.evenness.median():.2f}) because every reader is within a few fold of the leader;")
print(f"  it is kept in the CSV for completeness but should not be used as the summary.")

# ------------------------------------------------------------------ figures
n = len(panels); ncol = 4
nrow = int(np.ceil(n / ncol)) + 2
fig = plt.figure(figsize=(7.09, 1.35 * nrow), facecolor="white")
gs = fig.add_gridspec(nrow, ncol, hspace=0.95, wspace=0.55)
order = sorted(panels, key=lambda p: -p["lead_rest"])
for k, p in enumerate(order):
    ax = fig.add_subplot(gs[k // ncol, k % ncol])
    x = np.arange(1, len(p["rel"]) + 1)
    ax.plot(x, p["rel"], "-o", color=ACC, lw=0.8, ms=2.4, mew=0)
    ax.set_ylim(0, 1.12); ax.set_xlim(0.5, len(x) + 0.5)
    ax.set_xticks(x if len(x) <= 6 else x[::2])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=SUB, labelcolor=INK, pad=1.5)
    ax.set_ylabel("effect vs leader", color=INK, labelpad=1.5)
    ax.set_xlabel("reader rank", color=INK, labelpad=1.5)
    ax.text(-0.30, 1.22, string.ascii_lowercase[k] if k < 26 else f"a{k}",
            transform=ax.transAxes, fontsize=7, fontweight="bold", color=INK,
            va="top", ha="left")
    ax.text(-0.14, 1.22, p["motif"].replace("universal_", "Motif "), transform=ax.transAxes,
            fontsize=7, fontweight="bold", color=INK, va="top", ha="left")
    ax.text(1.0, 1.22, f"{p['cell']} · {p['n']}", transform=ax.transAxes,
            fontsize=5, color=SUB, va="top", ha="right")
    ax.text(0.96, 0.90, f"{p['lead_rest']:.2f}x", transform=ax.transAxes, fontsize=6,
            fontweight="bold", color=INK, va="top", ha="right")
    ax.text(0.96, 0.72, p["names"][0], transform=ax.transAxes, fontsize=5,
            color=SUB, va="top", ha="right")

# summary: the two ratios, so the categories emerge from where a motif falls
ax = fig.add_subplot(gs[nrow - 2:, :])
for p in panels:
    c = ACC if p["cell"] == "K562" else ACC2
    ax.plot(p["lead_2nd"], p["lead_rest"], "o", color=c, ms=3.5, mew=0, zorder=3)
    off = (3.4, 2.2) if p["cell"] == "K562" else (3.4, -5.2)   # stagger so labels do not stack
    ax.annotate(f"{p['motif'].replace('universal_','M')}", (p["lead_2nd"], p["lead_rest"]),
                textcoords="offset points", xytext=off, fontsize=4.8, color=SUB)
ax.axhline(1, color=MUT, lw=0.5); ax.axvline(1, color=MUT, lw=0.5)
ax.set_xlabel("leader / second reader", color=INK, labelpad=1.5)
ax.set_ylabel("leader / other readers", color=INK, labelpad=1.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(colors=SUB, labelcolor=INK, pad=1.5)
# the three regions, labelled where they actually are
ax.text(0.985, 0.965, "one dominant reader", transform=ax.transAxes, fontsize=5.5,
        color=MUT, va="top", ha="right")
ax.text(0.015, 0.965, "two co-dominant readers", transform=ax.transAxes, fontsize=5.5,
        color=MUT, va="top", ha="left")
ax.text(0.015, 0.035, "readers redundant", transform=ax.transAxes, fontsize=5.5,
        color=MUT, va="bottom", ha="left")
ax.plot([], [], "o", color=ACC, ms=3.5, mew=0, label="K562")
ax.plot([], [], "o", color=ACC2, ms=3.5, mew=0, label="HEK293")
ax.legend(frameon=False, loc="lower right", handlelength=1.0, borderpad=0.2)
ax.text(-0.055, 1.10, string.ascii_lowercase[min(n, 25)], transform=ax.transAxes,
        fontsize=7, fontweight="bold", color=INK, va="bottom", ha="left")
ax.text(-0.028, 1.10, "Summary", transform=ax.transAxes, fontsize=7,
        fontweight="bold", color=INK, va="bottom", ha="left")
fig.text(0.005, -0.006, "each panel: readers ranked, effect as a fraction of the leader, with "
         "leader/other-readers printed. No threshold is applied anywhere.",
         fontsize=5, color=MUT, ha="left", va="bottom")
for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/dominant_reader_{TAG}.{ext}", dpi=450, transparent=False,
                facecolor="white", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"\nwrote -> {OUT}/dominant_reader_{TAG}.csv")
print(f"wrote -> {FIG}/dominant_reader_{TAG}.pdf/.png")
