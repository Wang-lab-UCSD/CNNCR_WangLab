#!/usr/bin/env python
"""
Fig 5b+5c as ONE compact panel for Nature. No title, no panel letters (added at layout
time). Legend on the LEFT (size key + vertical colourbar). Short height, slightly wider.
Arial metrics (Liberation Sans), 5-7 pt, figure at FINAL size.
Palette = muted ColorBrewer RdBu-blue arm (matches fig5c_switch_tiled).

b (bubble grid): representative switching motifs. rows = candidate readers, cols = 5 cell
  lines. size = % of the motif's bound genes the reader binds; shade = % where strongest;
  dark ring = dominant reader. c (bars): % of shared loci per motif where the strongest
  binder differs between lines.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from lib.config import CELLS, JASPAR, MIN_GENES, OUT

for _f in ["/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
           "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf"]:
    if os.path.exists(_f): fm.fontManager.addfont(_f)
_ARIAL = "Liberation Sans" if any(f.name == "Liberation Sans" for f in fm.fontManager.ttflist) else "Arial"
plt.rcParams.update({"font.family": _ARIAL, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "none", "font.size": 6, "axes.linewidth": 0.5})

MM = 1 / 25.4
FIG_W, BODY_H, TITLE_MM = 120 * MM, 62.0, 4.5      # BODY_H = the original drawing area, in mm
FIG_H = (BODY_H + TITLE_MM) * MM                   # + a strip on top for the panel titles
fy = lambda v: v * BODY_H / (BODY_H + TITLE_MM)    # rescale figure-coord y so the body is unchanged
FS_TITLE, FS_TICK, FS_LAB, FS_SM = 7, 5.5, 6, 5
INK, SUB, MUT = "#1a1a1a", "#4d4d4d", "#808080"
BLUES = LinearSegmentedColormap.from_list("b", ["#eef4f9", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#0d3b66"])
RING, BAR, TMAX = "#08213f", "#4393c3", 0.6

jasp = json.load(open(JASPAR))
d = pd.read_parquet(f"{OUT}/reader_binding.parquet")
top = d.sort_values("max_signal", ascending=False).groupby(["gene_idx", "motif_id", "cell"], as_index=False).first()
pm = pd.read_csv(f"{OUT}/per_motif_switch.csv")

def dominants(m):
    out = {}
    for c in CELLS:
        s = top[(top.motif_id == m) & (top.cell == c)]
        if len(s):
            vc = s.reader.value_counts()
            if vc.iloc[0] >= MIN_GENES: out[c] = vc.index[0]
    return out

pm["ndom"] = pm.motif_id.map(lambda m: len(set(dominants(m).values())))
SW = pm[pm.ndom > 1].sort_values("pct_top_switch", ascending=False)
PANEL = ["universal_1", "universal_29", "universal_35", "universal_20"]
mid = lambda m: m.split("_")[1]

def spectrum(m, maxrows=5):
    bf, tf = {}, {}
    for c in CELLS:
        dc = d[(d.motif_id == m) & (d.cell == c)]
        if not len(dc): continue
        bg = dc.gene_idx.nunique()
        bf[c] = dc.groupby("reader").gene_idx.nunique() / bg
        tc = top[(top.motif_id == m) & (top.cell == c)]
        tf[c] = tc.groupby("reader").gene_idx.nunique() / bg
    B = pd.DataFrame(bf).reindex(columns=CELLS)
    T = pd.DataFrame(tf).reindex(columns=CELLS).reindex(index=B.index).fillna(0.0)
    keep = B.index[(T.max(axis=1) > 0.12) | (B.max(axis=1) >= 0.45)]
    B, T = B.loc[keep], T.loc[keep]
    order = B.max(axis=1).sort_values(ascending=False).index[:maxrows]
    return B.loc[order], T.loc[order]

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.85], wspace=0.62, hspace=0.35,
              top=fy(0.95), bottom=fy(0.15), left=0.205, right=0.985)

slots = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]
for k, (slot, m) in enumerate(zip(slots, PANEL)):
    ax = fig.add_subplot(slot)
    B, T = spectrum(m); dom = dominants(m); nr = len(B)
    bottom_row = k >= 2
    for i, reader in enumerate(B.index):
        yy = nr - 1 - i
        for j, c in enumerate(CELLS):
            b = B.loc[reader, c]
            if not np.isfinite(b) or b <= 0: continue
            t = T.loc[reader, c]; isdom = dom.get(c) == reader
            ax.scatter(j, yy, s=6 + 120 * b, c=[BLUES(min(t / TMAX, 1.0))],
                       edgecolor=(RING if isdom else "#cfcecb"),
                       linewidth=(1.1 if isdom else 0.4), zorder=3)
    ax.set_xlim(-0.6, len(CELLS) - 0.4); ax.set_ylim(-0.6, nr - 0.4)
    ax.set_xticks(range(len(CELLS)))
    if bottom_row:
        ax.set_xticklabels(CELLS, rotation=45, ha="right", fontsize=FS_TICK, color=SUB)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(range(nr)); ax.set_yticklabels(B.index[::-1], fontsize=FS_TICK, color=INK)
    ax.tick_params(length=0, pad=1.2)
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(lw=.4, color="#f0efec", zorder=0); ax.set_axisbelow(True)
    ax.text(0.0, 1.02, f"Motif {mid(m)}", transform=ax.transAxes, fontsize=FS_TITLE,
            color=INK, va="bottom", ha="left")

# c: switch-rate bars, with a row of dots = number of distinct dominant reader TFs
axc = fig.add_subplot(gs[:, 2])
y = np.arange(len(SW))[::-1]
axc.barh(y, SW.pct_top_switch, height=0.66, color=BAR, zorder=3, linewidth=0)
DOTCOL = "#0d3b66"
for yi, (_, row) in zip(y, SW.iterrows()):
    for kdot in range(int(row.ndom)):
        axc.scatter(min(row.pct_top_switch, 92) + 5 + kdot * 5.5, yi, s=4,
                    c=DOTCOL, zorder=5, clip_on=False)
axc.set_yticks(y); axc.set_yticklabels([f"Motif {mid(m)}" for m in SW.motif_id],
                                       fontsize=FS_TICK, color=INK)
axc.set_xlim(0, 100); axc.set_ylim(-0.6, len(SW) - 0.4); axc.set_xticks([0, 50, 100])
axc.set_xlabel("% genes switched", fontsize=FS_LAB, color=SUB, labelpad=1.5)
axc.grid(axis="x", lw=.4, color="#eeedea"); axc.set_axisbelow(True)
for s in ("top", "right", "left"): axc.spines[s].set_visible(False)
axc.tick_params(length=0, labelsize=FS_TICK, colors=SUB, pad=1.2)

# ---- LEFT legend column (pushed to the far left): size key + colourbar + dot key ----
sax = fig.add_axes([0.002, fy(0.62), 0.10, fy(0.30)]); sax.axis("off")
for k, b in enumerate([1.0, 0.5, 0.25]):
    yy = 2 - k
    sax.scatter(0.2, yy, s=6 + 120 * b, c=["#c6dbef"], edgecolor="#cfcecb", lw=.4)
    sax.text(0.62, yy, f"{int(b*100)}%", va="center", fontsize=FS_SM, color=MUT)
sax.set_xlim(-0.2, 1.8); sax.set_ylim(-0.6, 2.9)
sax.text(0.0, 2.85, "% genes\nbound", fontsize=FS_SM, color=SUB, va="bottom", linespacing=1.2)

cax = fig.add_axes([0.028, fy(0.24), 0.02, fy(0.26)])
cb = fig.colorbar(ScalarMappable(norm=Normalize(0, TMAX * 100), cmap=BLUES), cax=cax,
                  orientation="vertical")
cb.outline.set_visible(False); cb.set_ticks([0, 30, 60])
cb.ax.tick_params(labelsize=FS_SM, colors=MUT, length=0, pad=1)
cb.set_label("% genes\nstrongest", fontsize=FS_SM, color=SUB, labelpad=2, linespacing=1.2)

# dot key: N dots = N reader TFs
kax = fig.add_axes([0.002, fy(0.045), 0.10, fy(0.12)]); kax.axis("off")
for kdot in range(3):
    kax.scatter(0.15 + kdot * 0.28, 0.6, s=4, c=DOTCOL)
kax.text(0.0, 0.05, "= reader TFs", fontsize=FS_SM, color=SUB, va="center")
kax.set_xlim(-0.1, 1.6); kax.set_ylim(-0.2, 1.2)

# ---- short bold panel titles (house rule), one per panel, left-aligned over each block ----
_ty = 1 - 1.0 / (FIG_H / MM)                       # ~1 mm below the top edge
for _x, _t in ((gs[0, 0].get_position(fig).x0, "Reader binding"),
               (gs[0, 2].get_position(fig).x0, "Switch rate")):
    fig.text(_x, _ty, _t, ha="left", va="top", fontsize=FS_TITLE, color=INK, fontweight="bold")

# svg.fonttype="none" (set above) keeps labels editable after PowerPoint imports the SVG.
for ext in ("pdf", "png", "svg"):
    fig.savefig(f"{OUT}/fig5bc_nature.{ext}", dpi=600, bbox_inches="tight")
print(f"[out] {OUT}/fig5bc_nature.pdf/.png/.svg  ({FIG_W/MM:.0f}x{FIG_H/MM:.0f} mm, font={_ARIAL})")
