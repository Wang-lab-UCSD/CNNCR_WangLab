"""
05 — phyloP at the binding sites of every motif in the control POOL, in one pass.

Replaces running 01_extract_coordinates.py + 02_conservation_scores.py separately for each of
the 20 new control sets. All 20 sets are subsets of the 291-motif pool, and FIMO was run with a
p-value threshold (--no-qvalue --text), so a motif's hits do not depend on which other motifs
shared the scan. We therefore score every pool motif ONCE per biosample and let the aggregation
step take whichever 43 it needs.

Per biosample this writes one small table:
    results/pool_conservation/{BSSID}_motif_summary.tsv
    bssid | motif_id | group (core|pool) | n_genes | median_score | mean_score
    (n_genes/median_score follow 02_conservation_scores.py: best-per-gene, then median)

Usage:  python 05_pool_conservation.py --bssid BSS00004
"""
import os
import argparse
import gzip

import numpy as np
import pandas as pd
import pyBigWig

# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--bssid", required=True)
ap.add_argument("--bigwig", default="hg38.phyloP100way.bw")
ap.add_argument("--scan", default=None, help="alternative FIMO scan gz (default: the 334-motif pool scan)")
ap.add_argument("--outdir", default="results/pool_conservation")
args = ap.parse_args()
BSSID = args.bssid

PSCAN = args.scan or (ENHANCER_ROOT + "/multi_agent/posthoc_analysis/"
         f"results/pool_scan/{BSSID}_fimo.txt.gz")
COORD = ENHANCER_ROOT + f"/data_generated/{BSSID}.coord.exp.df.final.csv"
CORE_MEME = ENHANCER_ROOT + "/multi_agent/motif_pipeline/results/universal_regulator_motifs.meme"
OUT_DIR = args.outdir
OUT = f"{OUT_DIR}/{BSSID}_motif_summary.tsv"
os.makedirs(OUT_DIR, exist_ok=True)

if os.path.exists(OUT):
    print(f"[{BSSID}] already done, skipping")
    raise SystemExit(0)
for p in (PSCAN, COORD, args.bigwig):
    if not os.path.exists(p):
        raise SystemExit(f"[{BSSID}] ERROR missing {p}")

core_names = {l.split()[1] for l in open(CORE_MEME) if l.startswith("MOTIF")}

print(f"[{BSSID}] loading coordinate table", flush=True)
cdf = pd.read_csv(COORD, usecols=["chr", "start"])
c_chrom, c_start = cdf["chr"].values, cdf["start"].values

print(f"[{BSSID}] reading pool scan", flush=True)
with gzip.open(PSCAN, "rt") as fh:
    fimo = pd.read_csv(fh, sep="\t", comment="#",
                       usecols=["motif_id", "sequence_name", "start", "stop"])
print(f"[{BSSID}] {len(fimo):,} hits over {fimo.motif_id.nunique()} motifs", flush=True)

# FIMO sequence name is '{bssid}_{gene_idx}_{window_bin}'; row = gene_idx*20 + window_bin
parts = fimo.sequence_name.str.rsplit("_", n=2, expand=True)
row = parts[1].astype(np.int64) * 20 + parts[2].astype(np.int64)
chrom = pd.Series(c_chrom).reindex(row.values).values
wstart = pd.Series(c_start).reindex(row.values).values
ok = pd.notna(wstart) & pd.Series(chrom).map(lambda x: isinstance(x, str)).values
fimo = fimo.loc[ok].copy()
fimo["chrom"] = np.asarray(chrom)[ok]
fimo["g_start"] = wstart[ok].astype(np.int64) + fimo["start"].values - 1
fimo["g_end"] = wstart[ok].astype(np.int64) + fimo["stop"].values
fimo["gene_idx"] = parts[1].astype(np.int64).values[ok]   # must precede any merge/reindex
print(f"[{BSSID}] {len(fimo):,} hits mapped to the genome "
      f"({int((~ok).sum()):,} dropped, null windows)", flush=True)

# one phyloP query per DISTINCT interval, then map back
sites = fimo[["chrom", "g_start", "g_end"]].drop_duplicates().reset_index(drop=True)
print(f"[{BSSID}] {len(sites):,} distinct intervals to score", flush=True)
bw = pyBigWig.open(args.bigwig)
vals = np.full(len(sites), np.nan, dtype=np.float32)
for i, (ch, s, e) in enumerate(sites.itertuples(index=False)):
    try:
        v = bw.stats(ch, int(s), int(e), type="mean")[0]
        if v is not None:
            vals[i] = v
    except (RuntimeError, ValueError):
        pass
    if (i + 1) % 200000 == 0:
        print(f"[{BSSID}]   {i + 1:,}/{len(sites):,}", flush=True)
bw.close()
sites["phylop"] = vals

fimo = fimo.merge(sites, on=["chrom", "g_start", "g_end"], how="left")

# BEST PER GENE, exactly as 02_conservation_scores.py: for each (gene_idx, motif_id) keep the
# highest mean phyloP, so a gene contributes one value per motif and dense promoters cannot
# dominate. The published per-motif median in conservation_aggregate.csv is the median over
# these per-gene values, so this step must not be skipped.
best = (fimo.groupby(["motif_id", "gene_idx"])["phylop"].max().reset_index())
summ = (best.groupby("motif_id")["phylop"]
        .agg(n_genes="size", median_score="median", mean_score="mean")
        .reset_index())
summ["group"] = np.where(summ.motif_id.isin(core_names), "core", "pool")
summ["bssid"] = BSSID
summ = summ[["bssid", "motif_id", "group", "n_genes", "median_score", "mean_score"]]
summ.to_csv(OUT, sep="\t", index=False)
print(f"[{BSSID}] wrote {OUT}  ({(summ.group == 'core').sum()} core, "
      f"{(summ.group == 'pool').sum()} pool motifs)", flush=True)
