#!/usr/bin/env python
"""
STEP 3c — reader binding WITHOUT the all-5-lines locus restriction.

Identical to s3_reader_binding.py except that each cell line is intersected against its
OWN motif copies (out/loci/{cell}.parquet) instead of the all-5 intersection
(out/shared_loci.parquet). Everything else -- top-10% within-TF peak filter, JASPAR
candidate readers, gene-level collapse -- is unchanged.

  out/reader_binding_percell.parquet   cell, gene_idx, ensembl, motif_id, reader, max_signal, n_copies
  out/interrogated_percell.parquet     cell, gene_idx, motif_id   (loci scanned in that line)
"""
import sys, os, json, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from lib.config import CELLS, JASPAR, OUT, norm
from lib.peaks import read_narrowpeak, tf_beds, top_intensity

jasp = json.load(open(JASPAR))
CAND = {m: {norm(t) for t in tfs} for m, tfs in jasp.items()}

def top_peaks_bed(cell):
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

rows, interro, assayed_all = [], [], {}
for cell in CELLS:
    L = pd.read_parquet(f"{OUT}/loci/{cell}.parquet")
    L = L.drop_duplicates(["chrom", "start", "end", "motif_id", "gene_idx"])
    lb = tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False)
    for r in L.itertuples(index=False):
        lb.write(f"{r.chrom}\t{r.start}\t{r.end}\t{r.gene_idx}|{r.ensembl}|{r.motif_id}\n")
    lb.close()
    subprocess.run(f"sort -k1,1 -k2,2n {lb.name} -o {lb.name}", shell=True, check=True)
    interro.append(L[["gene_idx", "motif_id"]].drop_duplicates().assign(cell=cell))

    pb, asy = top_peaks_bed(cell); assayed_all[cell] = sorted(asy)
    res = subprocess.run(f"{BED if (BED:=os.environ.get('BEDTOOLS')) else '/stg3/data1/sam/Programs/bedtools2/bin/bedtools'}"
                         f" intersect -a {lb.name} -b {pb} -wa -wb -sorted",
                         shell=True, capture_output=True, text=True)
    os.unlink(pb); os.unlink(lb.name)
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
    print(f"{cell:<9} own copies {len(L):>8,} | loci scanned {len(interro[-1]):>7,} | "
          f"reader-bound cells {len(best):>7,}", flush=True)

pd.DataFrame(rows, columns=["cell", "gene_idx", "ensembl", "motif_id", "reader",
                            "max_signal", "n_copies"]).to_parquet(
    f"{OUT}/reader_binding_percell.parquet", index=False)
pd.concat(interro).to_parquet(f"{OUT}/interrogated_percell.parquet", index=False)
json.dump(assayed_all, open(f"{OUT}/assayed_tfs.json", "w"))
print(f"[out] {OUT}/reader_binding_percell.parquet + interrogated_percell.parquet + assayed_tfs.json")
