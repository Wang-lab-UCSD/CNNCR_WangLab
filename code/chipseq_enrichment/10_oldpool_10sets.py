"""
10 — Fig 4d with the TEN old-pool control sets (control_motifs_set_{0..9}.meme, the same sets
Fig 4a/b use) instead of the single published control set.

Occupancy per motif = mean fraction of a TF's TSS-proximal peaks containing a match, over all
(TF, cell-line) pairs x 100 (the published Fig 4d statistic), read from the 334-motif pool scan
plus the 13-motif extra scan. Core values come from the same pool scan, so both arms are scored
by one run of the same FIMO command.

Writes results/oldpool10/
  fig4d_permotif.csv     motif_id, group (core|control), n_sets, sets, mean_pct
  fig4d_per_cell.csv     cell, core_median, ctrl_mean (mean over sets of the per-set median),
                         ctrl_sd, ctrl_set_0..9
  fig4d_stats.csv        per set: n_ctrl, core_median, ctrl_median, fold, p (one-sided MWU);
                         plus 'pooled' (distinct control motifs across the 10 sets, each once)
"""
import os, glob
import numpy as np, pandas as pd
from scipy import stats
os.chdir(os.path.dirname(os.path.abspath(__file__)))
PH = "/stg3/data1/sam/enhancer_prediction/multi_agent/posthoc_analysis"
CELLS = ["HepG2", "K562", "GM12878", "MCF.7", "A549"]
OUT = "results/oldpool10"; os.makedirs(OUT, exist_ok=True)
members = lambda p: [l.split()[1] for l in open(p) if l.startswith("MOTIF")]
core_ids = members("/stg3/data1/sam/enhancer_prediction/multi_agent/motif_pipeline/results/universal_regulator_motifs.meme")
SETS = {int(f.rsplit("_", 1)[1].split(".")[0]): members(f) for f in sorted(glob.glob(f"{PH}/results/control_motifs_set_*.meme"))}
assert len(SETS) == 10
files = [f"results/pool_enrichment/{c}_enrichment.csv" for c in CELLS] + [f"results/pool_enrichment_extra/{c}_enrichment.csv" for c in CELLS]
missing = [f for f in files if not os.path.exists(f)]
if missing: raise SystemExit("not ready: " + ", ".join(missing))
E = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
occ = E.groupby("motif_id")["fraction_hit"].mean() * 100.0
allctrl = sorted({m for s in SETS.values() for m in s})
print(f"{occ.size} motifs with occupancy; old-pool sets use {len(allctrl)} distinct motifs, "
      f"{sum(m in occ.index for m in allctrl)} scored")
core = occ.reindex(core_ids).dropna()

pm = [dict(motif_id=m, group="core", n_sets=0, sets="", mean_pct=v) for m, v in core.items()]
for m in allctrl:
    if m in occ.index:
        ss = [i for i, s in SETS.items() if m in s]
        pm.append(dict(motif_id=m, group="control", n_sets=len(ss), sets=";".join(map(str, ss)), mean_pct=occ[m]))
pd.DataFrame(pm).to_csv(f"{OUT}/fig4d_permotif.csv", index=False)

rows = []
for i, s in SETS.items():
    ctrl = occ.reindex(s).dropna()
    rows.append(dict(set=i, n_ctrl=len(ctrl), core_median=core.median(), ctrl_median=ctrl.median(),
                     fold=core.median() / ctrl.median(), p=stats.mannwhitneyu(core, ctrl, alternative="greater").pvalue))
pooled = occ.reindex(allctrl).dropna()
rows.append(dict(set="pooled", n_ctrl=len(pooled), core_median=core.median(), ctrl_median=pooled.median(),
                 fold=core.median() / pooled.median(), p=stats.mannwhitneyu(core, pooled, alternative="greater").pvalue))
st = pd.DataFrame(rows); st.to_csv(f"{OUT}/fig4d_stats.csv", index=False)
print(st.round(4).to_string(index=False))

pc = []
for c in CELLS:
    d = E[E.cell_line == c].groupby("motif_id")["fraction_hit"].mean() * 100
    cm = d.reindex(core_ids).dropna().median()
    meds = [d.reindex(s).dropna().median() for i, s in sorted(SETS.items())]
    pc.append(dict(cell=c, core_median=cm, ctrl_mean=np.mean(meds), ctrl_sd=np.std(meds, ddof=1),
                   ctrl_min=min(meds), ctrl_max=max(meds), **{f"ctrl_set_{i}": v for i, v in enumerate(meds)}))
pc = pd.DataFrame(pc); pc.to_csv(f"{OUT}/fig4d_per_cell.csv", index=False)
print(pc[["cell", "core_median", "ctrl_mean", "ctrl_sd", "ctrl_min", "ctrl_max"]].round(3).to_string(index=False))
