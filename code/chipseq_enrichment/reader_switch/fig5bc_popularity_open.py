#!/usr/bin/env python
"""
Fig 5b+5c, POPULARITY metric, WITHOUT the all-5-lines locus restriction.

Each line is scored on its own promoter windows (s3c_reader_binding_percell.py), and the
switch rate is computed pair by pair on the loci interrogated in BOTH lines of that pair.
Metric definitions and layout are otherwise identical to fig5bc_popularity.py.
"""
import sys, os, json, itertools
# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from lib.config import CELLS, JASPAR, MIN_GENES, OUT, norm

for _f in ["/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
           "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf"]:
    if os.path.exists(_f): fm.fontManager.addfont(_f)
_ARIAL = "Liberation Sans" if any(f.name == "Liberation Sans" for f in fm.fontManager.ttflist) else "Arial"
plt.rcParams.update({"font.family": _ARIAL, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "none", "font.size": 6, "axes.linewidth": 0.5})

MM = 1 / 25.4
FIG_W, BODY_H, TITLE_MM = 120 * MM, 62.0, 4.5
FIG_H = (BODY_H + TITLE_MM) * MM
fy = lambda v: v * BODY_H / (BODY_H + TITLE_MM)
FS_TITLE, FS_TICK, FS_LAB, FS_SM = 7, 5.5, 6, 5
INK, SUB, MUT = "#1a1a1a", "#4d4d4d", "#808080"
BLUES = LinearSegmentedColormap.from_list("b", ["#eef4f9", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#0d3b66"])
RING, BAR, TMAX = "#08213f", "#4393c3", 0.6

jasp = json.load(open(JASPAR))
CAND = {m: {norm(t) for t in tfs} for m, tfs in jasp.items()}
d = pd.read_parquet(f"{OUT}/reader_binding_percell.parquet"); d["rn"] = d.reader.map(norm)
INT = pd.read_parquet(f"{OUT}/interrogated_percell.parquet")
assayed = {c: set(v) for c, v in json.load(open(f"{OUT}/assayed_tfs.json")).items()}
SCAN = {c: set(map(tuple, s[["gene_idx", "motif_id"]].itertuples(index=False, name=None)))
        for c, s in INT.groupby("cell")}

def dominants(m):
    out = {}
    for c in CELLS:
        s = d[(d.motif_id == m) & (d.cell == c)]
        if not len(s): continue
        vc = s.groupby("reader").gene_idx.nunique().sort_values(ascending=False)
        if vc.iloc[0] >= MIN_GENES: out[c] = vc.index[0]
    return out

def spectrum(m, maxrows=5):
    bf, sf = {}, {}
    for c in CELLS:
        dc = d[(d.motif_id == m) & (d.cell == c)]
        if not len(dc): continue
        g = dc.groupby("reader").gene_idx.nunique()
        bf[c] = g / dc.gene_idx.nunique()
        sf[c] = g / g.sum()
    B = pd.DataFrame(bf).reindex(columns=CELLS)
    S = pd.DataFrame(sf).reindex(columns=CELLS).reindex(index=B.index).fillna(0.0)
    keep = B.index[B.max(axis=1) >= 0.15]
    B, S = B.loc[keep], S.loc[keep]
    order = B.max(axis=1).sort_values(ascending=False).index[:maxrows]
    return B.loc[order], S.loc[order]

# ---- panel c: pairwise-interrogated, assay-matched reader-SET switch ----
bind = defaultdict(dict)
for r in d.itertuples(index=False):
    bind[(r.gene_idx, r.motif_id)].setdefault(r.cell, set()).add(r.rn)
ct, pair_rows = defaultdict(lambda: np.zeros(2)), []
for c1, c2 in itertools.combinations(CELLS, 2):
    both_tf = assayed[c1] & assayed[c2]
    both_loci = SCAN[c1] & SCAN[c2]
    n = sw = 0
    for key in both_loci:
        gi, m = key
        R = CAND.get(m, set()) & both_tf
        if len(R) < 2: continue
        per = bind.get(key, {})
        b1 = per.get(c1, set()) & R; b2 = per.get(c2, set()) & R
        if not b1 or not b2: continue
        n += 1; sw += (b1 != b2); ct[m] += [1, (b1 != b2)]
    if n: pair_rows.append(dict(pair=f"{c1} vs {c2}", loci=n, pct_set_switch=100 * sw / n))
PAIR = pd.DataFrame(pair_rows).sort_values("loci", ascending=False)
PAIR.to_csv(f"{OUT}/pairs_setswitch_open.csv", index=False)

SW = pd.DataFrame([dict(motif_id=m, n=int(v[0]), pct_set_switch=100 * v[1] / v[0],
                        ndom=len({norm(r) for r in dominants(m).values()}))
                   for m, v in ct.items() if v[0] >= 20]).sort_values("pct_set_switch", ascending=False)
SW.to_csv(f"{OUT}/per_motif_setswitch_open.csv", index=False)

SKIP = {"universal_15"}      # NFYA/NFYB are subunits of one complex, not competing paralogs
PANEL = list(SW[(SW.ndom > 1) & (~SW.motif_id.isin(SKIP))].motif_id.head(4))
mid = lambda m: m.split("_")[1]
print("panel b motifs:", PANEL)

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.85], wspace=0.62, hspace=0.35,
              top=fy(0.95), bottom=fy(0.15), left=0.205, right=0.985)
slots = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]
for k, (slot, m) in enumerate(zip(slots, PANEL)):
    ax = fig.add_subplot(slot)
    B, S = spectrum(m); dom = dominants(m); nr = len(B)
    for i, reader in enumerate(B.index):
        yy = nr - 1 - i
        for j, c in enumerate(CELLS):
            b = B.loc[reader, c]
            if not np.isfinite(b) or b <= 0: continue
            isdom = dom.get(c) == reader
            ax.scatter(j, yy, s=6 + 120 * b, c=[BLUES(min(S.loc[reader, c] / TMAX, 1.0))],
                       edgecolor=(RING if isdom else "#cfcecb"),
                       linewidth=(1.1 if isdom else 0.4), zorder=3)
    ax.set_xlim(-0.6, len(CELLS) - 0.4); ax.set_ylim(-0.6, nr - 0.4)
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels(CELLS if k >= 2 else [], rotation=45, ha="right", fontsize=FS_TICK, color=SUB)
    ax.set_yticks(range(nr)); ax.set_yticklabels(B.index[::-1], fontsize=FS_TICK, color=INK)
    ax.tick_params(length=0, pad=1.2)
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(lw=.4, color="#f0efec", zorder=0); ax.set_axisbelow(True)
    ax.text(0.0, 1.02, f"Motif {mid(m)}", transform=ax.transAxes, fontsize=FS_TITLE,
            color=INK, va="bottom", ha="left")

axc = fig.add_subplot(gs[:, 2])
y = np.arange(len(SW))[::-1]
axc.barh(y, SW.pct_set_switch, height=0.66, color=BAR, zorder=3, linewidth=0)
DOTCOL = "#0d3b66"
for yi, (_, row) in zip(y, SW.iterrows()):
    for kdot in range(int(row.ndom)):
        axc.scatter(min(row.pct_set_switch, 92) + 5 + kdot * 5.5, yi, s=4, c=DOTCOL, zorder=5, clip_on=False)
axc.set_yticks(y); axc.set_yticklabels([f"Motif {mid(m)}" for m in SW.motif_id], fontsize=FS_TICK, color=INK)
axc.set_xlim(0, 100); axc.set_ylim(-0.6, len(SW) - 0.4); axc.set_xticks([0, 50, 100])
axc.set_xlabel("% genes, readers differ", fontsize=FS_LAB, color=SUB, labelpad=1.5)
axc.grid(axis="x", lw=.4, color="#eeedea"); axc.set_axisbelow(True)
for s in ("top", "right", "left"): axc.spines[s].set_visible(False)
axc.tick_params(length=0, labelsize=FS_TICK, colors=SUB, pad=1.2)

sax = fig.add_axes([0.002, fy(0.62), 0.10, fy(0.30)]); sax.axis("off")
for k, b in enumerate([1.0, 0.5, 0.25]):
    yy = 2 - k
    sax.scatter(0.2, yy, s=6 + 120 * b, c=["#c6dbef"], edgecolor="#cfcecb", lw=.4)
    sax.text(0.62, yy, f"{int(b*100)}%", va="center", fontsize=FS_SM, color=MUT)
sax.set_xlim(-0.2, 1.8); sax.set_ylim(-0.6, 2.9)
sax.text(0.0, 2.85, "% genes\nbound", fontsize=FS_SM, color=SUB, va="bottom", linespacing=1.2)
cax = fig.add_axes([0.028, fy(0.24), 0.02, fy(0.26)])
cb = fig.colorbar(ScalarMappable(norm=Normalize(0, TMAX * 100), cmap=BLUES), cax=cax, orientation="vertical")
cb.outline.set_visible(False); cb.set_ticks([0, 30, 60])
cb.ax.tick_params(labelsize=FS_SM, colors=MUT, length=0, pad=1)
cb.set_label("% of reader\nbinding", fontsize=FS_SM, color=SUB, labelpad=2, linespacing=1.2)
kax = fig.add_axes([0.002, fy(0.045), 0.10, fy(0.12)]); kax.axis("off")
for kdot in range(3): kax.scatter(0.15 + kdot * 0.28, 0.6, s=4, c=DOTCOL)
kax.text(0.0, 0.05, "= reader TFs", fontsize=FS_SM, color=SUB, va="center")
kax.set_xlim(-0.1, 1.6); kax.set_ylim(-0.2, 1.2)
_ty = 1 - 1.0 / (FIG_H / MM)
for _x, _t in ((gs[0, 0].get_position(fig).x0, "Reader binding"),
               (gs[0, 2].get_position(fig).x0, "Reader switch")):
    fig.text(_x, _ty, _t, ha="left", va="top", fontsize=FS_TITLE, color=INK, fontweight="bold")
import shutil
DEST = ENHANCER_ROOT + "/paper_figures/fig5"
for ext in ("pdf", "png", "svg"):
    fig.savefig(f"{OUT}/fig5bc_popularity_open.{ext}", dpi=600, bbox_inches="tight",
                transparent=True)
    shutil.copy2(f"{OUT}/fig5bc_popularity_open.{ext}", f"{DEST}/fig5bc_locus_switch.{ext}")
print(f"[out] {OUT}/fig5bc_popularity_open.pdf/.png/.svg")
print(f"      copied to {DEST}/fig5bc_locus_switch.pdf/.png/.svg")
print("\nPAIRS\n" + PAIR.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
print("\nPER MOTIF\n" + SW.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
tot = np.sum([v for v in ct.values()], axis=0)
print(f"\nPOOLED over ALL motifs      {100*tot[1]/tot[0]:.1f}% over {int(tot[0]):,} gene comparisons")
print(f"POOLED over the {len(SW)} in Fig 5c  {100*(SW.pct_set_switch/100*SW.n).sum()/SW.n.sum():.1f}% over {int(SW.n.sum()):,} gene comparisons   <- the number quoted in the paper")
