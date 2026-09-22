"""Input files of CNN-CR and how to turn them into the model tensor.

Per biosample the training input is one FASTA file of regulatory windows plus one expression table:

  {BSSID}.windows.fa.gz   every window is a 400-bp sequence with the header
                          >{gene_index}|{ENSG}|{slot}|{promoter|enhancer}|{chr}:{start}-{end}|{strand}
                          slot 0-9 are the promoter windows (slot 0 = TSS +/- 200 bp), slot 10-19 the enhancer
                          windows. Slots a gene does not have are simply absent from the file; they become all-zero
                          rows of the one-hot tensor (about half of all slots).
  {BSSID}.expression.tsv  gene_index, ENSG, log2_fpkm  (the model target; one row per gene, same order)

read_windows_fasta() returns the one-hot tensor X of shape (genes, 20, 400, 4) with A,C,G,T as the last axis
(N and absent windows are all zeros), exactly the array the paper's models were trained on.
"""
import gzip, numpy as np

N_SLOTS, WIN, ALPHABET = 20, 400, "ACGT"
_LUT = np.zeros((256, 4), dtype=bool)
for _i, _c in enumerate(ALPHABET): _LUT[ord(_c), _i] = True; _LUT[ord(_c.lower()), _i] = True

def one_hot(seq):
    """400-bp string -> (400, 4) bool array; any letter other than A/C/G/T (e.g. N) is a zero row."""
    return _LUT[np.frombuffer(seq.encode(), dtype=np.uint8)]

def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)

def read_expression(path):
    """{BSSID}.expression.tsv -> (gene ids list, log2 expression array)."""
    genes, y = [], []
    with _open(path) as fh:
        for line in fh:
            if line.startswith("gene_index") or not line.strip(): continue
            idx, ensg, val = line.rstrip("\n").split("\t"); genes.append(ensg); y.append(float(val))
    return genes, np.array(y)

def read_windows_fasta(path, n_genes=None, verbose=True):
    """{BSSID}.windows.fa.gz -> one-hot tensor (genes, 20, 400, 4), plus the window headers."""
    headers, seqs = [], []
    with _open(path) as fh:
        h, buf = None, []
        for line in fh:
            if line.startswith(">"):
                if h is not None: headers.append(h); seqs.append("".join(buf))
                h, buf = line[1:].rstrip("\n"), []
            else: buf.append(line.strip())
        if h is not None: headers.append(h); seqs.append("".join(buf))
    gidx = np.array([int(h.split("|")[0]) for h in headers]); slot = np.array([int(h.split("|")[2]) for h in headers])
    n = n_genes or gidx.max() + 1
    X = np.zeros((n, N_SLOTS, WIN, 4), dtype=bool)
    for g, s, seq in zip(gidx, slot, seqs):
        if len(seq) != WIN: raise ValueError(f"window {g}|{s} has length {len(seq)}, expected {WIN}")
        X[g, s] = one_hot(seq)
    if verbose: print(f"{path}: {len(seqs):,} windows for {n:,} genes -> one-hot tensor {X.shape} ({X.nbytes/1e6:.0f} MB)")
    return X, headers

def write_windows_fasta(path, records):
    """records: iterable of (gene_index, ensg, slot, kind, chrom, start, end, strand, seq); writes gzip if path ends in .gz."""
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "wt") as fh:
        for g, ensg, slot, kind, chrom, start, end, strand, seq in records:
            fh.write(f">{g}|{ensg}|{slot}|{kind}|{chrom}:{start}-{end}|{strand}\n{seq}\n")

def write_expression(path, ensg_ids, y):
    with open(path, "w") as fh:
        fh.write("gene_index\tENSG\tlog2_fpkm\n")
        for i, (g, v) in enumerate(zip(ensg_ids, y)): fh.write(f"{i}\t{g}\t{v:.10g}\n")
