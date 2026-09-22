#!/usr/bin/env python
"""
STEP 2 — motif-instance coordinates: recover the genomic position of every core-motif
occurrence in each gene's promoter windows, from the FIMO scan already run for the
linear/RF models. No new FIMO.

STRAND-CORRECT (the stored promoter sequences are reverse-complemented for minus-strand
genes; verified against hg38). With window (ws, we, wstrand) and FIMO 1-based inclusive (s, e):
    '+' : start = ws+s-1, end = ws+e,     motif strand as reported
    '-' : start = we-e,   end = we-s+1,   motif strand flipped
Every run self-verifies against hg38 and refuses to write if any sample mismatches.

  out/loci/{cell}.parquet    cell, gene_idx, ensembl, motif_id, chrom, start, end, strand, pvalue
  out/shared_loci.parquet    motif copies present at identical coordinates in ALL 5 lines
"""
import sys, os, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from collections import defaultdict
from lib.config import (CELLS, BSSID, EPIMAP_FIMO, EPIMAP_COORD, HEK_FIMO, HEK_WINDOWS,
                        GENES_TXT, HG38, BEDTOOLS, FIMO_P, OUT)

GENES = [l.split()[0].split(".")[0] for l in open(GENES_TXT) if l.strip()]
GIDX = {g: i for i, g in enumerate(GENES)}
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
def rc(s): return s.translate(COMP)[::-1]

def windows_epimap(bssid):
    win, per = {}, defaultdict(int)
    with open(f"{EPIMAP_COORD}/{bssid}.coord.exp.df.final.csv") as fh:
        fh.readline()
        for line in fh:
            c = line.split(",", 11)                      # do not split the huge seq field
            if len(c) < 11 or c[9] != "promoter":
                continue
            i = GIDX.get(c[3])
            if i is None:
                continue
            b = per[c[3]]; per[c[3]] += 1
            if not c[0]:
                continue
            try:
                win[(i, b)] = (c[0], int(float(c[1])), int(float(c[2])), c[5])
            except ValueError:
                pass
    return win

def windows_hek():
    win = {}
    with open(HEK_WINDOWS) as fh:
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) < 6:
                continue
            nm = c[3].split("_")
            try:
                win[(int(nm[-2]), int(nm[-1]))] = (c[0], int(c[1]), int(c[2]), c[5])
            except ValueError:
                pass
    return win

def build(cell):
    if cell in BSSID:
        win = windows_epimap(BSSID[cell])
        fimo = f"{EPIMAP_FIMO}/{BSSID[cell]}_fimo.txt"
    else:
        win = windows_hek()
        fimo = HEK_FIMO
    rows = []
    with open(fimo) as fh:
        fh.readline()
        for line in fh:
            c = line.rstrip("\n").split("\t")
            if len(c) < 8:
                continue
            nm = c[2].split("_")
            try:
                gi, b = int(nm[-2]), int(nm[-1])
                s, e = int(c[3]), int(c[4]); pv = float(c[7])
            except (ValueError, IndexError):
                continue
            if pv >= FIMO_P:
                continue
            w = win.get((gi, b))
            if w is None:
                continue
            chrom, ws, we, wst = w
            if wst == "-":
                bs, be, mst = we - e, we - s + 1, ("-" if c[5] == "+" else "+")
            else:
                bs, be, mst = ws + s - 1, ws + e, c[5]
            rows.append((cell, gi, GENES[gi], c[0], chrom, bs, be, mst, pv, c[9]))
    return pd.DataFrame(rows, columns=["cell", "gene_idx", "ensembl", "motif_id", "chrom",
                                       "start", "end", "strand", "pvalue", "matched_seq"])

def verify(df, n=400):
    s = df.sample(min(n, len(df)), random_state=0)
    bed = tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False)
    for _, r in s.iterrows():
        bed.write(f"{r.chrom}\t{r.start}\t{r.end}\n")
    bed.close()
    out = subprocess.run(f"{BEDTOOLS} getfasta -fi {HG38} -bed {bed.name} -tab",
                         shell=True, capture_output=True, text=True).stdout
    os.unlink(bed.name)
    seqs = [l.split("\t")[1].upper() for l in out.strip().split("\n") if "\t" in l]
    ok = sum((g if r.strand == "+" else rc(g)) == r.matched_seq.upper()
             for (_, r), g in zip(s.iterrows(), seqs))
    return ok, len(seqs)

def main():
    os.makedirs(f"{OUT}/loci", exist_ok=True)
    keys = {}
    for cell in CELLS:
        df = build(cell)
        ok, tot = verify(df)
        tag = "OK" if ok == tot else "*** MISMATCH ***"
        print(f"{cell:<9} instances {len(df):>9,} | verify {ok}/{tot} {tag}", flush=True)
        if ok != tot:
            raise SystemExit(f"coordinate verification failed for {cell}")
        df.drop(columns=["matched_seq"]).to_parquet(f"{OUT}/loci/{cell}.parquet", index=False)
        keys[cell] = set(zip(df.chrom, df.start, df.end, df.motif_id))
    shared = set.intersection(*keys.values())
    ref = pd.read_parquet(f"{OUT}/loci/{CELLS[0]}.parquet")
    ref["key"] = list(zip(ref.chrom, ref.start, ref.end, ref.motif_id))
    ref[ref.key.isin(shared)].drop(columns="key").to_parquet(f"{OUT}/shared_loci.parquet", index=False)
    print(f"\nshared motif copies (all {len(CELLS)} lines): {len(shared):,} "
          f"over {ref[ref.key.isin(shared)].gene_idx.nunique():,} genes")
    print(f"[out] {OUT}/loci/, shared_loci.parquet")

if __name__ == "__main__":
    main()
