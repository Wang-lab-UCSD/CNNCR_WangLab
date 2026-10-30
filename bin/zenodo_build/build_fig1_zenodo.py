#!/usr/bin/env python
"""Build the Figure 1 inputs that go to Zenodo, from the development tree (not part of the repository build).

  python build_fig1_zenodo.py inputs  [BSSID ...]   window FASTA + expression table per biosample, verified against the
                                                    one-hot array the models were trained on (training_one_hot/*.npy)
  python build_fig1_zenodo.py motifs                the 116 non-redundant motif sets + the TOMTOM tables against the four catalogues
  python build_fig1_zenodo.py pack                  tar the staging folders into zenodo_upload/

Staging: github/_build/zenodo_staging/{fig1_training_inputs,fig1_motif_sets}/
"""
import os, sys, glob, gzip, shutil, subprocess, numpy as np, pandas as pd
# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from cnncr_io import *
E = ENHANCER_ROOT; G = f"{E}/github"; STG = f"{G}/_build/zenodo_staging"; OUT = f"{G}/zenodo_upload"
B = [b.strip() for b in open(f"{E}/Epimap_BSSID_nondup.txt") if b.strip()]
names = dict(l.rstrip("\n").split("\t")[:2] for l in open(f"{E}/EpiMapID_Name.txt") if l.strip())

def build_inputs(bssids):
    D = f"{STG}/fig1_training_inputs"; os.makedirs(D, exist_ok=True)
    for b in bssids:
        fa, ex = f"{D}/{b}.windows.fa.gz", f"{D}/{b}.expression.tsv"
        if os.path.exists(fa + ".ok"): print(b, "done already"); continue
        df = pd.read_csv(f"{E}/data_generated/{b}.coord.exp.df.final.csv", sep=",", lineterminator="\n",
                         usecols=["chr", "start", "end", "ensembl", "strand", "type", "expression", "seq"], dtype={"seq": str})
        assert len(df) % N_SLOTS == 0; n = len(df) // N_SLOTS
        ensg = df["ensembl"].values[::N_SLOTS]; y = df["expression"].values[::N_SLOTS].astype(float)
        assert np.array_equal(ensg, df["ensembl"].values.reshape(n, N_SLOTS)[:, 0]) and (df["ensembl"].values.reshape(n, N_SLOTS) == ensg[:, None]).all()
        recs = []
        for i, row in enumerate(df.itertuples(index=False)):
            seq = row.seq if isinstance(row.seq, str) else ""
            if len(seq) != WIN: seq = "N" * WIN                      # what one_hot_encode.py did with a malformed window
            if set(seq) == {"N"}: continue                            # absent window: not written, all-zero in the tensor
            g, s = divmod(i, N_SLOTS); kind = "promoter" if s < 10 else "enhancer"
            recs.append((g, ensg[g], s, kind, row.chr, int(row.start), int(row.end), row.strand, seq.upper()))
        write_windows_fasta(fa, recs); write_expression(ex, ensg, y)
        # verify: the FASTA re-encodes to exactly the array the models were trained on
        X, _ = read_windows_fasta(fa, n_genes=n, verbose=False); ref = np.load(f"{E}/training_one_hot/{b}.one_hot.npy", mmap_mode="r")
        g2, y2 = read_expression(ex); yref = np.load(f"{E}/training_y/{b}.y.npy")
        ok = X.shape == ref.shape and np.array_equal(X, ref) and np.allclose(y2, yref) and g2 == list(ensg)
        print(f"{b} {names.get(b,'?')}: {len(recs):,} windows written, one-hot identical to training_one_hot: {ok}", flush=True)
        if not ok: raise SystemExit(f"{b}: mismatch")
        open(fa + ".ok", "w").write("verified\n")

def build_motifs():
    D = f"{STG}/fig1_motif_sets"; os.makedirs(D, exist_ok=True)
    for b in B:
        for t in ("pro", "enh"):
            shutil.copy2(f"{E}/cell_type_specific_CNN/{b}/{b}_{t}_motifs_nonredundant.meme", D)
            shutil.copy2(f"{E}/cell_type_specific_CNN/{b}/{b}_{t}_motifs_nonredundant_weights.npy", D)
            for db, dev in (("JASPAR", "JASPAR"), ("CISBP", "CISBP"), ("DNAmeth", "DNAmeth"), ("Epigram", "epigram")):
                os.makedirs(f"{D}/catalogue_tomtom/{db}", exist_ok=True)
                src = f"{E}/motif_analysis/common_overlap_{dev}_tomtom/{b}_{t}_overlap_{dev}.txt"; dst = f"{D}/catalogue_tomtom/{db}/{b}_{t}_tomtom_{db}.txt.gz"
                with open(src, "rb") as fi, gzip.open(dst, "wb") as fo: shutil.copyfileobj(fi, fo)
    with open(f"{D}/biosamples.tsv", "w") as fh:
        fh.write("BSSID\tname\n"); [fh.write(f"{b}\t{names.get(b,'')}\n") for b in B]
    print("motif sets staged:", len(glob.glob(f"{D}/*_motifs_nonredundant.meme")), "MEME files")

def pack():
    os.makedirs(OUT, exist_ok=True); Z = "pigz -p 8" if shutil.which("pigz") else "gzip"
    def tar(name, cwd, items): subprocess.run(f"tar -I '{Z}' -cf {OUT}/{name} -C {cwd} {' '.join(items)}", shell=True, check=True); print("wrote", name, f"{os.path.getsize(f'{OUT}/{name}')/1e9:.2f} GB")
    D = f"{STG}/fig1_training_inputs"; assert all(os.path.exists(f"{D}/{b}.windows.fa.gz.ok") for b in B), "not all biosamples verified"
    shutil.copy2(f"{G}/_build/README_fig1_training_inputs.txt", f"{D}/README.txt")
    with open(f"{D}/biosamples.tsv", "w") as fh:
        fh.write("BSSID\tname\n"); [fh.write(f"{b}\t{names.get(b,'')}\n") for b in B]
    tar("fig1_training_inputs_all116.tar.gz", STG, ["fig1_training_inputs/README.txt", "fig1_training_inputs/biosamples.tsv"] + [f"fig1_training_inputs/{b}.{x}" for b in B for x in ("windows.fa.gz", "expression.tsv")])
    shutil.copy2(f"{G}/_build/README_fig1_motif_sets.txt", f"{STG}/fig1_motif_sets/README.txt")
    tar("fig1_motif_sets_all116.tar.gz", STG, ["fig1_motif_sets"])

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "inputs": build_inputs(sys.argv[2:] or B)
    elif cmd == "motifs": build_motifs()
    elif cmd == "pack": pack()
