"""
09b — stage 1 of 09_pool_enrichment.py for the 13 old-pool control motifs that are absent from
the 334-motif pool scan (scanned separately into results/fimo_poolscan_extra/{cell}/fimo_peaks).
Writes results/pool_enrichment_extra/{cell}_enrichment.csv with the same columns as
results/pool_enrichment/{cell}_enrichment.csv, so the two can be concatenated.
"""
import os, pickle
from collections import defaultdict
import pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
CELLS = ["HepG2", "K562", "GM12878", "MCF.7", "A549"]
OUT = "results/pool_enrichment_extra"; os.makedirs(OUT, exist_ok=True)

def load_fimo_hits(path):
    hits = defaultdict(set)
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("motif_id"): continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 3: continue
            hits[c[0]].add(c[2].split("::")[0])
    return hits

for cell in CELLS:
    out_csv = f"{OUT}/{cell}_enrichment.csv"
    if os.path.exists(out_csv): print(f"[{cell}] already done"); continue
    tsv = f"results/fimo_poolscan_extra/{cell}/fimo_peaks/fimo.tsv"; pkl = f"data/{cell}/tf_peak_sets.pkl"
    if not (os.path.exists(tsv) and os.path.exists(pkl)): print(f"[{cell}] inputs not ready"); continue
    hits = load_fimo_hits(tsv); tf_peaks = pickle.load(open(pkl, "rb"))
    rows = []
    for tf, peaks in tf_peaks.items():
        n = len(peaks)
        if not n: continue
        for motif, mh in hits.items():
            k = len(peaks & mh); rows.append((cell, tf, motif, n, k, k / n))
    pd.DataFrame(rows, columns=["cell_line", "tf", "motif_id", "n_peaks", "hits_in_peaks", "fraction_hit"]).to_csv(out_csv, index=False)
    print(f"[{cell}] {len(rows):,} rows, {len(hits)} motifs -> {out_csv}", flush=True)
