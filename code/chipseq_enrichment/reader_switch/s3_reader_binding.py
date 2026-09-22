#!/usr/bin/env python
"""
STEP 3 — locus-resolved reader binding + switch statistics.

For each shared motif copy, intersect with each cell line's top-10% full-width peaks,
keep only peaks of that motif's JASPAR candidate readers, and record — per
(cell, gene, motif, reader) — the STRONGEST overlapping peak signal (a motif may occur
several times at a gene; we take the max). The top reader is the strongest such TF.

Switch is scored ASSAY-MATCHED: for each pair of lines, restrict to candidate readers
ChIP-profiled in BOTH, so "different reader here vs there" cannot be an artifact of a TF
simply not being assayed. A locus bound in one line only is LOSS, not a switch.

  out/reader_binding.parquet        cell, gene_idx, ensembl, motif_id, reader, max_signal, n_copies
  out/assay_matched_pairs.csv       per cell-line pair: %switched, %top-switched, losses
  out/per_motif_switch.csv          per motif: n loci, %set-switch, %top-switch
"""
import sys, os, re, json, pickle, subprocess, tempfile, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from collections import defaultdict
from lib.config import CELLS, JASPAR, ENCODE4, BEDTOOLS, OUT, norm
from lib.peaks import read_narrowpeak, tf_beds, top_intensity

jasp = json.load(open(JASPAR))
CAND = {m: {norm(t) for t in tfs} for m, tfs in jasp.items()}

def top_peaks_bed(cell):
    """Write this line's top-10% peaks (chrom start end TF signal), sorted."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False)
    assayed = set()
    for tf, bed in tf_beds(cell):
        recs = read_narrowpeak(bed)
        if not recs:
            continue
        assayed.add(norm(tf))
        for ch, s, e, v in top_intensity(recs):
            tmp.write(f"{ch}\t{s}\t{e}\t{tf}\t{v}\n")
    tmp.close()
    subprocess.run(f"sort -k1,1 -k2,2n {tmp.name} -o {tmp.name}", shell=True, check=True)
    return tmp.name, assayed

def binding_table():
    loci = pd.read_parquet(f"{OUT}/shared_loci.parquet")
    lb = tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False)
    for r in loci.itertuples(index=False):
        lb.write(f"{r.chrom}\t{r.start}\t{r.end}\t{r.gene_idx}|{r.ensembl}|{r.motif_id}\n")
    lb.close()
    subprocess.run(f"sort -k1,1 -k2,2n {lb.name} -o {lb.name}", shell=True, check=True)

    rows, assayed = [], {}
    for cell in CELLS:
        pb, asy = top_peaks_bed(cell); assayed[cell] = asy
        res = subprocess.run(f"{BEDTOOLS} intersect -a {lb.name} -b {pb} -wa -wb -sorted",
                             shell=True, capture_output=True, text=True)
        os.unlink(pb)
        best = {}
        for line in res.stdout.split("\n"):
            if not line:
                continue
            f = line.split("\t")
            if len(f) < 9:
                continue
            gi, ens, m = f[3].split("|")
            tfn = norm(f[7])
            if tfn not in CAND.get(m, ()):
                continue
            try:
                v = float(f[8])
            except ValueError:
                continue
            k = (int(gi), ens, m, f[7])
            cur = best.get(k)
            best[k] = (max(v, cur[0]) if cur else v, (cur[1] + 1) if cur else 1)
        for (gi, ens, m, tf), (v, nc) in best.items():
            rows.append((cell, gi, ens, m, tf, v, nc))
        print(f"{cell:<9} candidate-reader (gene,motif,TF) cells: {len(best):,}", flush=True)
    os.unlink(lb.name)
    df = pd.DataFrame(rows, columns=["cell", "gene_idx", "ensembl", "motif_id", "reader",
                                     "max_signal", "n_copies"])
    df.to_parquet(f"{OUT}/reader_binding.parquet", index=False)
    return df, assayed

def switch_stats(df, assayed):
    df = df.copy(); df["rn"] = df.reader.map(norm)
    bind = defaultdict(dict)
    for r in df.itertuples(index=False):
        bind[(r.gene_idx, r.motif_id)].setdefault(r.cell, {})[r.rn] = r.max_signal
    pair_rows, motif_ct = [], defaultdict(lambda: [0, 0, 0])   # motif -> [n, set_sw, top_sw]
    for c1, c2 in itertools.combinations(CELLS, 2):
        both = assayed[c1] & assayed[c2]
        cons = sw = top = loss = 0
        for (gi, m), per in bind.items():
            R = CAND.get(m, set()) & both
            if len(R) < 2:
                continue
            b1 = {k: v for k, v in per.get(c1, {}).items() if k in R}
            b2 = {k: v for k, v in per.get(c2, {}).items() if k in R}
            if not b1 and not b2:
                continue
            if not b1 or not b2:
                loss += 1; continue
            same = set(b1) == set(b2)
            cons += same; sw += (not same)
            tsw = max(b1, key=b1.get) != max(b2, key=b2.get)
            top += tsw
            motif_ct[m][0] += 1; motif_ct[m][1] += (not same); motif_ct[m][2] += tsw
        tot = cons + sw
        pair_rows.append(dict(pair=f"{c1} vs {c2}", shared_TFs=len(both), loci=tot,
                              pct_set_switch=100 * sw / tot if tot else np.nan,
                              pct_top_switch=100 * top / tot if tot else np.nan,
                              one_line_only=loss))
    pd.DataFrame(pair_rows).sort_values("loci", ascending=False).to_csv(
        f"{OUT}/assay_matched_pairs.csv", index=False)
    pm = pd.DataFrame([dict(motif_id=m, n=v[0], pct_set_switch=100*v[1]/v[0],
                            pct_top_switch=100*v[2]/v[0]) for m, v in motif_ct.items() if v[0] >= 20])
    pm.sort_values("pct_top_switch", ascending=False).to_csv(f"{OUT}/per_motif_switch.csv", index=False)
    tot_n = sum(v[0] for v in motif_ct.values())
    tot_sw = sum(v[1] for v in motif_ct.values()); tot_top = sum(v[2] for v in motif_ct.values())
    print(f"\nASSAY-MATCHED pooled: {tot_n:,} loci bound in both lines | "
          f"set-switch {100*tot_sw/tot_n:.1f}% | top-switch {100*tot_top/tot_n:.1f}%")

def main():
    df, assayed = binding_table()
    switch_stats(df, assayed)
    print(f"[out] {OUT}/reader_binding.parquet, assay_matched_pairs.csv, per_motif_switch.csv")

if __name__ == "__main__":
    main()
