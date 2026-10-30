#!/usr/bin/env python
"""
09e -- rebuild the locus-resolved reader-binding table at a LOOSER ChIP cutoff.

reader_switch/s3 built out/reader_binding.parquet from each TF's TOP-10% peaks
(lib.config.SIG_PCTL = 90).  The user wants the same within-line reader comparison run on
"top-90% ChIP-bound genes" as well, i.e. drop only each TF's bottom-decile peaks.  Same
loci, same candidate-reader roster, same max-signal-per-(gene,motif,reader) rule -- only
the within-TF signalValue percentile changes.

Restricted to the two lines that have Perturb-seq (K562, HEK293); the other three lines
are irrelevant to a perturbation comparison and would only cost runtime.

  out/reader_binding_top90.parquet   cell, gene_idx, ensembl, motif_id, reader, max_signal, n_copies
"""
import sys, os, json, subprocess, tempfile
# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

RS = ENHANCER_ROOT + "/multi_agent/chipseq_enrichment/reader_switch"
sys.path.insert(0, RS)
import numpy as np, pandas as pd
from lib.config import JASPAR, BEDTOOLS, OUT, norm
from lib.peaks import read_narrowpeak, tf_beds

PCTL = float(sys.argv[1]) if len(sys.argv) > 1 else 10   # keep peaks >= this within-TF percentile (10 -> top 90%; 0 -> ALL peaks)
TAG  = sys.argv[2] if len(sys.argv) > 2 else "top90"      # 2026-10-01: `python 09e_reader_binding_pctl.py 0 all` builds reader_binding_all.parquet
LINES = ["K562", "HEK293"]

CAND = {m: {norm(t) for t in tfs} for m, tfs in json.load(open(JASPAR)).items()}
ALL_CAND = set().union(*CAND.values())


def peaks_bed(cell):
    """This line's peaks above the within-TF PCTL cutoff, candidate readers only."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False)
    n = 0
    for tf, bed in tf_beds(cell):
        if norm(tf) not in ALL_CAND:            # only motif candidate readers matter here
            continue
        recs = read_narrowpeak(bed)
        if not recs:
            continue
        thr = np.percentile([r[3] for r in recs], PCTL)
        for ch, s, e, v in recs:
            if v >= thr:
                tmp.write(f"{ch}\t{s}\t{e}\t{tf}\t{v}\n"); n += 1
    tmp.close()
    subprocess.run(f"sort -k1,1 -k2,2n {tmp.name} -o {tmp.name}", shell=True, check=True)
    print(f"{cell:<9} candidate-reader peaks kept (top {100-PCTL}%): {n:,}", flush=True)
    return tmp.name


def main():
    loci = pd.read_parquet(f"{OUT}/shared_loci.parquet").drop_duplicates(
        subset=["gene_idx", "ensembl", "motif_id", "chrom", "start", "end"])
    print(f"shared loci (deduped across lines): {len(loci):,}", flush=True)
    lb = tempfile.NamedTemporaryFile("w", suffix=".bed", delete=False)
    for r in loci.itertuples(index=False):
        lb.write(f"{r.chrom}\t{r.start}\t{r.end}\t{r.gene_idx}|{r.ensembl}|{r.motif_id}\n")
    lb.close()
    subprocess.run(f"sort -k1,1 -k2,2n {lb.name} -o {lb.name}", shell=True, check=True)

    rows = []
    for cell in LINES:
        pb = peaks_bed(cell)
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
            if norm(f[7]) not in CAND.get(m, ()):
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
        print(f"{cell:<9} (gene,motif,reader) cells: {len(best):,}", flush=True)

    os.unlink(lb.name)
    df = pd.DataFrame(rows, columns=["cell", "gene_idx", "ensembl", "motif_id", "reader",
                                     "max_signal", "n_copies"])
    df.to_parquet(f"{OUT}/reader_binding_{TAG}.parquet", index=False)
    print(f"\n[out] {OUT}/reader_binding_{TAG}.parquet   {len(df):,} rows")


if __name__ == "__main__":
    main()
