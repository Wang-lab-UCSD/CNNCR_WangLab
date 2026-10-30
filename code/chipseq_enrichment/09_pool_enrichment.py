"""
09 — Fig 4d recomputed for every new control set, from one pool-wide FIMO scan.

Stage 1 (per cell line) reproduces 02_compute_enrichment.py exactly, but on the 334-motif pool
scan instead of the 86-motif combined file: for every (TF, motif) pair, the fraction of that TF's
TSS-proximal peaks containing at least one match. FIMO was thresholded on p (1e-4) and the
original code counts every row keyed by motif_id, so scanning the pool once and grouping by motif
is identical to rescanning per control set.

Stage 2 takes the published Fig 4d statistic - per motif, the mean fraction_hit over all
(TF, cell) pairs, times 100 - and runs core vs control for each of the 20 new sets plus the
published one, one-sided Mann-Whitney (core greater).

Outputs
  results/pool_enrichment/{cell}_enrichment.csv    per TF x motif, all 334 motifs
  results/pool_sets_fig4d_summary.csv              one row per control set
  results/pool_sets_fig4d_permotif.csv             per-motif occupancy used by the test
Usage: python 09_pool_enrichment.py [--stage1-only|--stage2-only]
"""
import os
import sys
import glob
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
WORK = os.getcwd()
PH = ENHANCER_ROOT + "/multi_agent/posthoc_analysis"
CELLS = ["HepG2", "K562", "GM12878", "MCF.7", "A549"]
OUT = "results/pool_enrichment"
os.makedirs(OUT, exist_ok=True)
only = sys.argv[1] if len(sys.argv) > 1 else ""


def load_fimo_hits(path):
    """motif_id -> set of peak names. Identical parsing to 02_compute_enrichment.py."""
    hits = defaultdict(set)
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("motif_id"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 3:
                continue
            hits[c[0]].add(c[2].split("::")[0])
    return hits


if only != "--stage2-only":
    for cell in CELLS:
        out_csv = f"{OUT}/{cell}_enrichment.csv"
        if os.path.exists(out_csv):
            print(f"[{cell}] already done")
            continue
        tsv = f"{WORK}/results/fimo_poolscan/{cell}/fimo_peaks/fimo.tsv"
        pkl = f"{WORK}/data/{cell}/tf_peak_sets.pkl"
        if not (os.path.exists(tsv) and os.path.exists(pkl)):
            print(f"[{cell}] inputs not ready, skipping")
            continue
        print(f"[{cell}] loading {tsv}", flush=True)
        hits = load_fimo_hits(tsv)
        with open(pkl, "rb") as fh:
            tf_peaks = pickle.load(fh)
        rows = []
        for tf, peaks in tf_peaks.items():
            n = len(peaks)
            if not n:
                continue
            for motif, mh in hits.items():
                k = len(peaks & mh)
                rows.append((cell, tf, motif, n, k, k / n))
        pd.DataFrame(rows, columns=["cell_line", "tf", "motif_id", "n_peaks",
                                    "hits_in_peaks", "fraction_hit"]).to_csv(out_csv, index=False)
        print(f"[{cell}] {len(rows):,} TF x motif rows -> {out_csv}", flush=True)

if only == "--stage1-only":
    raise SystemExit(0)

files = sorted(glob.glob(f"{OUT}/*_enrichment.csv"))
if len(files) < len(CELLS):
    raise SystemExit(f"only {len(files)}/{len(CELLS)} cell lines ready; rerun stage 2 later")
E = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
E = E[E.cell_line.isin(CELLS)]
occ = E.groupby("motif_id")["fraction_hit"].mean() * 100.0     # the Fig 4d statistic
print(f"{len(E):,} rows, {occ.size} motifs with occupancy")

core_ids = [l.split()[1] for l in open(
    ENHANCER_ROOT + "/multi_agent/motif_pipeline/results/"
    "universal_regulator_motifs.meme") if l.startswith("MOTIF")]
members = lambda p: [l.split()[1] for l in open(p) if l.startswith("MOTIF")]
SETS = {}
for fam, patt in (("random", f"{PH}/results/control_motifs_tomtom_only_set_*.meme"),
                  ("cgmatched", f"{PH}/results/control_motifs_cgmatched_set_*.meme"),
                  ("cgquantile", f"{PH}/results/control_motifs_cgquantile_set_*.meme")):
    for f in sorted(glob.glob(patt)):
        SETS[(fam, int(f.rsplit("_", 1)[1].split(".")[0]))] = members(f)
SETS[("published", 0)] = members(f"{PH}/results/control_motifs.meme")

core = occ.reindex(core_ids).dropna()
rows, pm = [], [dict(family="core", set=-1, motif_id=m, occupancy=v) for m, v in core.items()]
for (fam, i), mem in sorted(SETS.items()):
    ctrl = occ.reindex(mem).dropna()
    if len(ctrl) < 20:
        print(f"  ! {fam} {i}: only {len(ctrl)} motifs scanned, skipping")
        continue
    p = stats.mannwhitneyu(core.values, ctrl.values, alternative="greater").pvalue
    rows.append(dict(family=fam, set=i, n_ctrl=len(ctrl), core_median=core.median(),
                     ctrl_median=ctrl.median(), fold=core.median() / ctrl.median(), p_fig4d=p))
    pm += [dict(family=fam, set=i, motif_id=m, occupancy=v) for m, v in ctrl.items()]
    print(f"  {fam:10} set {i}: core {core.median():.2f}% vs ctrl {ctrl.median():.2f}% "
          f"({core.median() / ctrl.median():.2f}x) P={p:.2e}")
pd.DataFrame(rows).to_csv("results/pool_sets_fig4d_summary.csv", index=False)
pd.DataFrame(pm).to_csv("results/pool_sets_fig4d_permotif.csv", index=False)
print("\nwrote results/pool_sets_fig4d_summary.csv and results/pool_sets_fig4d_permotif.csv")
