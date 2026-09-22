#!/usr/bin/env python
"""Fig 3 panels f and h — protein constraint, extended and split by TF family.

Produces two tables:

  results/constraint_comparison_extended.csv
      the three published metrics PLUS the synonymous-z negative control. Synonymous
      variants are not under coding selection, so core and background must NOT differ
      there. A null on syn_z is what rules out coverage / mutation-rate / gene-length
      artifacts behind the pLI, LOEUF and missense-z results.

  results/family_constraint.csv
      LOEUF for core vs background TFs WITHIN each DNA-binding-domain family. The pooled
      comparison in panel f cannot distinguish "core TFs are constrained" from "core TFs
      are drawn from intrinsically constrained families". This can.

      ⚠ THE UNIT IS THE TF, NOT THE MOTIF. A core motif matches many JASPAR TFs, so family
      membership is read off each TF individually (TF_Information Family_Name) and motifs are
      never assigned a family. The catch is the REVERSE direction: most families are supplied
      by only one or two motifs, so their core TFs are near-duplicates of one PWM's match list
      (Nuclear receptor 94% from universal_35, GATA and E2F 100% from a single motif each,
      Homeodomain 81% from universal_10). Only C2H2 ZF is broadly supplied (30 motifs, biggest
      single motif 42%). n_motifs is therefore emitted alongside n_core and PRINTED ON THE
      PANEL, because it, not n_core, is what says how much a lane can bear.

⚠ DETERMINISM FIX. The original 01_protein_conservation.py picks one gnomAD row per gene with
`sort_values("pLI", ascending=False).drop_duplicates("gene_upper")`. 44 gene names are
duplicated and 7 of those have IDENTICAL pLI across their rows, so which transcript survives is
decided by an unstable quicksort and flips between runs — that is the entire source of the
1,338 vs 1,339 background discrepancy between the two existing copies of the results. This
script uses kind="mergesort" (stable) so the answer is reproducible. It settles on n=1,339.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

MA   = "/stg3/data1/sam/enhancer_prediction/multi_agent"
SD   = "/stg3/data1/sam/enhancer_prediction/paper_figures/fig3/code"
ASSIGN  = f"{MA}/posthoc_analysis/results/step06_JASPAR_assignments.json"
TF_INFO = "/stg3/data1/sam/enhancer_prediction/motif_analysis/TF_Information.txt"
GNOMAD  = f"{SD}/results/gnomad_v2.1.1_constraint.tsv"

METRICS = [("pLI", "pLI (LoF intolerance)"),
           ("oe_lof_upper", "LOEUF (LoF obs/exp upper bound)"),
           ("mis_z", "Missense z-score"),
           ("syn_z", "Synonymous z-score")]

with open(ASSIGN) as fh:
    core_upper = {t.upper() for lst in json.load(fh).values() for t in lst}

tf = pd.read_csv(TF_INFO, sep="\t")
tf = tf[tf.TF_Species == "Homo_sapiens"].copy()
tf["U"] = tf.TF_Name.str.upper()
tf = tf.drop_duplicates("U")[["U", "TF_Name", "DBID", "Family_Name"]].rename(
    columns={"DBID": "ENSG"})
tf["is_core"] = tf.U.isin(core_upper)

gn = pd.read_csv(GNOMAD, sep="\t", low_memory=False)
cols = ["gene", "gene_id"] + [m for m, _ in METRICS]
gn = gn[cols].dropna(subset=["pLI"]).copy()
gn["U"] = gn.gene.str.upper()
gn = gn.sort_values("pLI", ascending=False, kind="mergesort").drop_duplicates("U")

tf["EC"] = tf.ENSG.str.split(".").str[0]
gn["EC"] = gn.gene_id.str.split(".").str[0]
m = tf.merge(gn[["EC"] + [k for k, _ in METRICS]], on="EC", how="left")

by_name = gn.set_index("U")[[k for k, _ in METRICS]]
for i in m[m.pLI.isna()].index:
    nm = m.at[i, "U"]
    if nm in by_name.index:
        r = by_name.loc[nm]
        r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
        m.loc[i, [k for k, _ in METRICS]] = r.values
m = m.dropna(subset=["pLI"]).copy()
print(f"matched TFs: {len(m)}  (core {m.is_core.sum()}, background {(~m.is_core).sum()})")

# ── extended constraint comparison (adds the synonymous-z negative control) ────────
rows = []
for key, label in METRICS:
    a = m[m.is_core & m[key].notna()][key].values
    b = m[~m.is_core & m[key].notna()][key].values
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rows.append(dict(metric=label, key=key, n_universal=len(a), median_universal=np.median(a),
                     n_background=len(b), median_background=np.median(b),
                     mannwhitney_U=U, p_value=p))
    print(f"  {label:<34} core {np.median(a):>8.4f}  bg {np.median(b):>8.4f}  P={p:.4g}")
pd.DataFrame(rows).to_csv(f"{SD}/results/constraint_comparison_extended.csv", index=False)

# ── LOEUF within family ───────────────────────────────────────────────────────────
KEY = "oe_lof_upper"
# which core motifs supply each TF — used to report how many independent motifs back a family
with open(ASSIGN) as fh:
    assign = json.load(fh)
tf2motif = {}
for motif, tfs in assign.items():
    for t in tfs:
        tf2motif.setdefault(t.upper(), set()).add(motif)

rows = []
for fam, n_core in m[m.is_core].Family_Name.value_counts().items():
    sub = m[m.Family_Name == fam]
    a_tf = sub[sub.is_core & sub[KEY].notna()]
    a = a_tf[KEY].values
    b = sub[~sub.is_core & sub[KEY].notna()][KEY].values
    if len(a) < 4 or len(b) < 4:          # need both arms to say anything
        continue
    motifs = [mo for t in a_tf.U for mo in tf2motif.get(t, ())]
    n_motifs = len(set(motifs))
    top_share = pd.Series(motifs).value_counts().iloc[0] / len(a) if motifs else np.nan
    p = stats.mannwhitneyu(a, b, alternative="two-sided")[1]
    rows.append(dict(family=fam, n_core=len(a), n_background=len(b), n_motifs=n_motifs,
                     top_motif_share=top_share,
                     median_core=np.median(a), median_background=np.median(b),
                     delta=np.median(a) - np.median(b), p_value=p))
fam_df = pd.DataFrame(rows).sort_values("n_core", ascending=False)
fam_df.to_csv(f"{SD}/results/family_constraint.csv", index=False)
print(f"\n{len(fam_df)} testable families "
      f"({(fam_df.delta < 0).sum()} follow the pooled direction):")
for _, r in fam_df.iterrows():
    print(f"  {r.family:<20} n={int(r.n_core):>3}/{int(r.n_background):<4} "
          f"from {int(r.n_motifs):>2} motifs (top {100*r.top_motif_share:.0f}%)  "
          f"LOEUF {r.median_core:.3f} vs {r.median_background:.3f}  P={r.p_value:.3g}"
          f"{'' if r.delta < 0 else '   <-- REVERSED'}")

skipped = [f for f, n in m[m.is_core].Family_Name.value_counts().items()
           if n >= 4 and f not in set(fam_df.family)]
print(f"\nfamilies with >=4 core TFs but <4 background TFs (untestable): {skipped}")
