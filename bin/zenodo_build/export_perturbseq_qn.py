"""Write the quantile-normalised knockout-effect vectors of both Perturb-seq screens, exactly as 08c_alltargets_stats.py computes them
in memory (shared 8,236-gene axis, seed-0 quantile reference), so notebook 11 never loads or normalises the raw screens.
Runs the loader prefix of 08c to export the two quantile-normalised matrices."""
import os, sys, numpy as np, time
# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

CE=ENHANCER_ROOT + "/multi_agent/chipseq_enrichment"; OUT=sys.argv[1]; os.makedirs(OUT,exist_ok=True); os.chdir(CE)
_t0=time.time(); src=open("08c_alltargets_stats.py").read().split("DEFS = [")[0]
exec(compile(src,"08c_prefix","exec")); print("prefix loaded %.0f s"%(time.time()-_t0),flush=True)
genes=np.array(_shl); kidx=np.array([k_col[g] for g in _shl]); oidx=np.array([o_col[g] for g in _shl])
# K562: every perturbed TF (row group) -> qnormK(mean over its rows), restricted to the shared axis; effective knockdown from the raw self-gene z
ktfs=sorted(k_perturbed); K=np.full((len(ktfs),len(genes)),np.nan,np.float32); keff=np.zeros(len(ktfs),bool)
for i,tf in enumerate(ktfs):
    K[i]=k_delta(tf)[kidx]; keff[i]=bool(k_effkd(tf))
    if i%2000==0: print("K562",i,len(ktfs),"%.0f s"%(time.time()-_t0),flush=True)
np.savez_compressed(f"{OUT}/perturbseq_qn_K562.npz",tfs=np.array(ktfs),genes=genes,delta_qn=K,eff_kd=keff,thr=np.float32(THR["K"]),
                    sym_ensg=np.array(list(k_sym)),sym_name=np.array(list(k_sym.values())))
otfs=sorted(o_perturbed); Hm=np.full((len(otfs),len(genes)),np.nan,np.float32); oeff=np.zeros(len(otfs),bool)
for i,tf in enumerate(otfs):
    Hm[i]=o_delta(tf)[oidx]; oeff[i]=bool(o_effkd_mask.get(tf,False))
    if i%3000==0: print("HEK",i,len(otfs),"%.0f s"%(time.time()-_t0),flush=True)
np.savez_compressed(f"{OUT}/perturbseq_qn_HEK293T.npz",tfs=np.array(otfs),genes=genes,delta_qn=Hm,eff_kd=oeff,thr=np.float32(THR["H"]),
                    sym_ensg=np.array(list(o_sym)),sym_name=np.array(list(o_sym.values())))
open(f"{OUT}/README_perturbseq_qn.txt","w").write(f"""Quantile-normalised knockout-effect vectors of the two Perturb-seq screens (notebook 11, Fig. 5d-g), precomputed as
08c_alltargets_stats.py computes them: for every perturbed gene (TF) the mean z-score-versus-non-targeting profile of its guides
(K562: Replogle et al. 2022 genome-wide CRISPRi, published normalised bulk; HEK293T: X-Atlas/Orion, processed to the same
representation by code/perturb_seq/orion_hek293t_process_replogle.py), mapped by quantile normalisation onto the average quantile
function of the two screens (20,001 quantiles of 800 random perturbations per screen, seed 0), on the {len(genes):,} genes measured in both screens.
perturbseq_qn_K562.npz / perturbseq_qn_HEK293T.npz:
  tfs        perturbed gene symbols (normalised: upper case, no punctuation), {len(ktfs):,} K562 / {len(otfs):,} HEK293T
  genes      Ensembl ids of the shared gene axis (columns of delta_qn)
  delta_qn   float32 [tfs x genes] quantile-normalised knockout effect (NaN where not measured)
  eff_kd     bool per tf: effective knockdown (self-gene z < -0.10)
  thr        responsive-target floor (10th percentile of |effect|) of that screen: K562 {THR['K']:.4f}, HEK293T {THR['H']:.4f}
  sym_ensg / sym_name   Ensembl id -> gene symbol of every gene of that screen
""")
print("done %.0f s"%(time.time()-_t0))
