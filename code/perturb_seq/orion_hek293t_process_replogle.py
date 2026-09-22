#!/usr/bin/env python
"""
Process X-Atlas/Orion HEK293T Perturb-seq to MATCH the Replogle K562 gwps
normalized_bulk representation, so the two are directly comparable side-by-side
for the reader-switch analysis.

Replogle representation (decoded from K562_gwps_normalized_bulk_01.h5ad):
  - per-cell size normalization to a fixed library size, LINEAR (no log1p)
  - bulk[p,g]   = mean over perturbation p's cells of normalized expression
  - ctrl_mean/std[g] = mean/std over NON-TARGETING *single cells* (not pseudobulk)
  - X[p,g]      = (bulk[p,g] - ctrl_mean[g]) / ctrl_std[g]      (z vs control cells)
  - self-KD     = X[p, target_gene]  (target's own z; effective KD if < -0.1,
                  the threshold used in the K562 phase1 reader analysis)
  - fold_expr   = bulk[p, target] / ctrl_mean[target]          (Replogle-native KD)

Scale of the per-cell normalization (TARGET_SUM) cancels in both X (z-score) and
fold_expr (ratio), so it need not match Replogle's absolute scale.

Input : /new-stg/home/sam2/HEK293T_orion_perturbseq/data/HEK293T_*.parquet (223 shards)
        cells: sparse raw UMI gene_token_id[]+gene_expression[], gene_target (KO,
        "Non-Targeting"=control), pass_guide_filter (QC).
Output: processed/orion_hek293t_replogle_matched.npz + _summary.tsv
"""
import os, glob, time, numpy as np, pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
from scipy.sparse import coo_matrix

ROOT   = "/new-stg/home/sam2/HEK293T_orion_perturbseq"
DATA   = sorted(glob.glob(f"{ROOT}/data/HEK293T_*.parquet"))
GMETA  = f"{ROOT}/gene_metadata.parquet"
OUT    = f"{ROOT}/processed"
NTC    = "Non-Targeting"
TARGET_SUM = 1e4          # arbitrary; cancels in z-score and in fold_expr
KD_Z   = -0.1             # effective-KD threshold on self-z (matches K562 phase1)
SHARDS = int(os.environ.get("SHARDS", "0")) or None   # validate on first N shards
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

# ---- gene axis ----
gm = pq.read_table(GMETA).to_pandas().sort_values("gene_token_id")
ensembl = gm["ensembl_id"].to_numpy().astype(str)
gname   = gm["gene_name"].to_numpy().astype(str)
N = len(ensembl)
assert (gm["gene_token_id"].to_numpy() == np.arange(N)).all()
shards = DATA[:SHARDS] if SHARDS else DATA
print(f"[genes] {N} | shards {len(shards)}{' (VALIDATION SUBSET)' if SHARDS else ''}", flush=True)

# ---- pass 1: perturbation labels ----
perts = set()
for s in shards:
    perts.update(pq.read_table(s, columns=["gene_target"]).to_pandas()["gene_target"].unique())
perts = sorted(str(p) for p in perts)
pidx  = {p: i for i, p in enumerate(perts)}
P = len(perts)
ntc_i = pidx.get(NTC)
print(f"[pass1] perturbations={P}  NTC present={ntc_i is not None}  ({time.time()-t0:.0f}s)", flush=True)

# ---- pass 2: accumulate normalized bulk sums + control single-cell sum/sumsq ----
bsum   = np.zeros((P, N), dtype=np.float64)   # sum of per-cell normalized expr per perturbation
ncells = np.zeros(P, dtype=np.int64)
csumsq = np.zeros(N, dtype=np.float64)        # control single-cell sum of squares (for std)
for k, s in enumerate(shards):
    tab = pq.read_table(s, columns=["gene_target","gene_token_id","gene_expression","pass_guide_filter"])
    tab = tab.filter(pc.equal(tab["pass_guide_filter"], 1))
    if tab.num_rows == 0:
        continue
    pe     = tab["gene_target"].to_pandas().astype(str).map(pidx).to_numpy()
    tokarr = tab["gene_token_id"].combine_chunks()
    valarr = tab["gene_expression"].combine_chunks()
    off    = tokarr.offsets.to_numpy().astype(np.int64)      # len = n_cells+1, starts at 0
    lens   = np.diff(off)
    cols   = pc.list_flatten(tokarr).to_numpy(zero_copy_only=False).astype(np.int64)
    data   = pc.list_flatten(valarr).to_numpy(zero_copy_only=False).astype(np.float64)
    # per-cell total UMI -> per-cell linear normalization to TARGET_SUM
    tot   = np.add.reduceat(data, off[:-1]); tot[tot == 0] = 1.0
    norm  = data / np.repeat(tot, lens) * TARGET_SUM
    rows  = np.repeat(pe, lens)
    # perturbation bulk sums (incl NTC row -> gives control sum for free)
    coo = coo_matrix((norm, (rows, cols)), shape=(P, N)); coo.sum_duplicates()
    bsum[coo.row, coo.col] += coo.data
    np.add.at(ncells, pe, 1)
    # control single-cell sum-of-squares
    m = pe == ntc_i
    if m.any():
        clens = lens[m]
        coff  = np.zeros(len(clens)+1, dtype=np.int64); np.cumsum(clens, out=coff[1:])
        sel   = np.repeat(m, lens)
        ccoo  = coo_matrix((norm[sel]**2, (np.zeros(sel.sum(), dtype=np.int64), cols[sel])), shape=(1, N))
        ccoo.sum_duplicates(); csumsq[ccoo.col] += ccoo.data
    if (k+1) % 25 == 0:
        print(f"[pass2] {k+1}/{len(shards)} shards  ({time.time()-t0:.0f}s)", flush=True)

# ---- control single-cell mean/std (zeros count toward denominator) ----
n_ntc     = int(ncells[ntc_i])
ctrl_mean = bsum[ntc_i] / n_ntc
ctrl_var  = csumsq / n_ntc - ctrl_mean**2
ctrl_std  = np.sqrt(np.clip(ctrl_var, 0, None))

# ---- bulk mean per perturbation, z-score vs control single cells ----
nz   = ncells.copy().astype(np.float64); nz[nz == 0] = 1.0
bulk = bsum / nz[:, None]
with np.errstate(divide="ignore", invalid="ignore"):
    X = (bulk - ctrl_mean) / ctrl_std          # inf/nan where ctrl_std==0 (mirrors K562)
X = X.astype(np.float32)

# ---- self-KD: target's own z + Replogle-native fold_expr ----
gidx     = {g: i for i, g in enumerate(gname)}
tgt_col  = np.array([gidx.get(p, -1) for p in perts])
have_tgt = tgt_col >= 0
self_z   = np.full(P, np.nan, np.float32)
fold     = np.full(P, np.nan, np.float32)
ii       = np.where(have_tgt)[0]; jj = tgt_col[ii]
self_z[ii] = X[ii, jj]
with np.errstate(divide="ignore", invalid="ignore"):
    fold[ii] = (bulk[ii, jj] / ctrl_mean[jj]).astype(np.float32)
is_ntc   = np.array(perts) == NTC
eff_kd   = (self_z < KD_Z) & have_tgt & ~is_ntc

tag = f".val{SHARDS}" if SHARDS else ""
np.savez_compressed(f"{OUT}/orion_hek293t_replogle_matched{tag}.npz",
    X=X, perturbations=np.array(perts), ncells=ncells,
    ensembl_id=ensembl, gene_name=gname,
    ctrl_mean=ctrl_mean.astype(np.float32), ctrl_std=ctrl_std.astype(np.float32),
    self_z=self_z, fold_expr=fold, effective_kd=eff_kd, target_on_axis=have_tgt)
pd.DataFrame({"perturbation": perts, "ncells": ncells,
              "self_z": self_z, "fold_expr": fold,
              "effective_kd": eff_kd, "is_control": is_ntc}) \
  .sort_values("self_z").to_csv(f"{OUT}/orion_hek293t_replogle_matched{tag}_summary.tsv", sep="\t", index=False)

print(f"[done] P={P} genes={N} cells={int(ncells.sum())} NTC_cells={n_ntc}", flush=True)
print(f"[QC] target-on-axis={int(have_tgt.sum())}  effective_kd(self_z<{KD_Z})={int(eff_kd.sum())}"
      f"  ctrl_std==0 genes={int((ctrl_std==0).sum())}", flush=True)
print(f"[out] {OUT}/orion_hek293t_replogle_matched{tag}.npz  ({time.time()-t0:.0f}s)", flush=True)
