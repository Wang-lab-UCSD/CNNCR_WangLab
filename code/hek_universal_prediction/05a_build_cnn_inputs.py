#!/usr/bin/env python
"""
Step 5a: build the CNN one-hot input + target for HEK293, matching the EpiMap
CNN-CR promoter model (posthoc_analysis/05_cnn_comparison.py).

EpiMap CNN input  : training_one_hot/{bssid}.one_hot.npy  shape (18356, 20, 400, 4) bool
                    -> PRO_ONLY takes the first 10 branches (promoter windows).
Here we build the promoter-only analog directly (10 branches):

  results/hek_one_hot.npy   (18356, 10, 400, 4) bool
      branch i = window bin i for that gene (bin 0 = TSS anchor, 1..9 = peaks);
      genes with <10 windows -> remaining branches all-zero (model's is_nonzero
      mask treats all-zero branches as padding, exactly like EpiMap).
  results/hek_y.npy          (18356,) float32 = log2(HPA TPM + 1); NaN if no HPA.

Channel order = A,C,G,T (matches MEME PWM column order used to init conv0).
Windows come from regions/hek293_windows.fa (headers HEK293_{gene_idx}_{bin}).
"""
import os, numpy as np, pandas as pd

# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

WORK  = ENHANCER_ROOT + "/multi_agent/hek_universal_prediction"
GENES = ENHANCER_ROOT + "/gene_ENSG.txt"
FA    = f"{WORK}/regions/hek293_windows.fa"
HPA   = f"{WORK}/../hek_hpa/HEK293_HPA_expression.tsv"
CAGE  = ENHANCER_ROOT + "/genome_annotation/hg38_promoters_cage_corrected.bed"
N_GENES, N_BRANCH, SEQ_LEN = 18356, 10, 400
os.makedirs(f"{WORK}/results", exist_ok=True)

BASE = {"A": 0, "C": 1, "G": 2, "T": 3}

def one_hot(seq):
    m = np.zeros((SEQ_LEN, 4), dtype=bool)
    for i, ch in enumerate(seq[:SEQ_LEN]):
        j = BASE.get(ch.upper())
        if j is not None:
            m[i, j] = True
    return m

# ---- gene order ----
with open(GENES) as f:
    gene_order = [l.split()[0].split(".")[0] for l in f if l.strip()]
assert len(gene_order) == N_GENES

# ---- parse fasta into (gene_idx, bin) -> seq ----
X = np.zeros((N_GENES, N_BRANCH, SEQ_LEN, 4), dtype=bool)
name = None
seq_lines = []
n_win = 0
def flush(nm, sq):
    global n_win
    if nm is None:
        return
    parts = nm[1:].strip().split("_")     # HEK293_{gene_idx}_{bin}
    gi, b = int(parts[1]), int(parts[2])
    if 0 <= gi < N_GENES and 0 <= b < N_BRANCH:
        X[gi, b] = one_hot(sq)
        n_win += 1

with open(FA) as f:
    for line in f:
        if line.startswith(">"):
            flush(name, "".join(seq_lines))
            name = line; seq_lines = []
        else:
            seq_lines.append(line.strip())
    flush(name, "".join(seq_lines))
print(f"[info] filled {n_win} windows into one-hot {X.shape}")
np.save(f"{WORK}/results/hek_one_hot.npy", X)

# ---- target: log2(HPA TPM + 1), NaN if missing ----
hpa = pd.read_csv(HPA, sep="\t")
hpa["ensg"] = hpa["Gene"].str.split(".").str[0]
tpm = dict(zip(hpa["ensg"], hpa["TPM"]))
y = np.array([np.log2(tpm[e] + 1.0) if e in tpm else np.nan for e in gene_order],
             dtype=np.float32)
np.save(f"{WORK}/results/hek_y.npy", y)
print(f"[info] target: {np.isfinite(y).sum()} genes with HPA, {np.isnan(y).sum()} missing")
print(f"[out] results/hek_one_hot.npy  results/hek_y.npy")
