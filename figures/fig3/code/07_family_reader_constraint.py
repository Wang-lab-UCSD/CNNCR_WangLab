#!/usr/bin/env python
"""Fig 3 panel h — constraint of the READER TFs, grouped by the motif family that recruits them.

Built to the user's spec (2026-07-31): classify each core motif by its DOMINANT reader family,
then for each family pool every reader TF of every motif in it and show the full distribution.

Two earlier attempts at panel h are superseded and should not be revived:
  - LOEUF core vs background within family: most families were supplied by one or two motifs,
    so "n=58 core TFs" was really one PWM's match list. Misleading.
  - per-motif phyloP by family (05_family_conservation.py): the honest unit, but only 3
    families carry >=3 motifs, so it could only return a null.
This version has real n. Every family below has >= 5 reader TFs and the panel rests on 255
reader-TF observations rather than on 43 motifs.

WHICH "CONSERVATION": the readers are proteins, so this is protein constraint — gnomAD LOEUF,
the same metric as panel f. LOWER LOEUF = MORE constrained, so the axis is drawn with the most
constrained families at the top. (DNA-level phyloP cannot be used here: phyloP is a property of
a motif's genomic sites, not of a TF, so "all readers' conservation" has no DNA analogue.)

⚠ A TF that reads motifs assigned to two different dominant families is counted in both, which
is why the observation count (255) slightly exceeds the 250 unique core TFs. Five duplicates.

Output: results/family_reader_constraint.csv — one row per (family, reader TF)
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

import os
# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

MA = ENHANCER_ROOT + "/multi_agent"
SD = ENHANCER_ROOT + "/paper_figures/fig3/code"
MIN_READERS = 5

with open(f"{MA}/posthoc_analysis/results/step06_JASPAR_assignments.json") as fh:
    assign = json.load(fh)

tfi = pd.read_csv(ENHANCER_ROOT + "/motif_analysis/TF_Information.txt", sep="\t")
tfi = tfi[tfi.TF_Species == "Homo_sapiens"].copy()
tfi["U"] = tfi.TF_Name.str.upper()
fam = tfi.drop_duplicates("U").set_index("U")["Family_Name"].to_dict()

ct = pd.read_csv(f"{MA}/conservation_protein/results/universal_tf_constraint_table.csv")
ct["U"] = ct.TF_Name.str.upper()
loeuf = ct.set_index("U")["oe_lof_upper"].to_dict()

# 1. every core motif -> its dominant reader family
dom = {}
purity = []
for motif, tfs in assign.items():
    fams = [fam.get(t.upper()) for t in tfs if fam.get(t.upper())]
    if not fams:
        continue
    vc = pd.Series(fams).value_counts()
    dom[motif] = vc.index[0]
    purity.append(vc.iloc[0] / len(fams))
print(f"{len(dom)} core motifs assigned a dominant family; median purity {np.median(purity):.2f}")

# 2. per family, pool the reader TFs of all its motifs
rows = []
for motif, f in dom.items():
    for t in {x.upper() for x in assign[motif]}:
        v = loeuf.get(t)
        if v is not None and not pd.isna(v):
            rows.append(dict(family=f, motif=motif, tf=t, loeuf=v))
d = pd.DataFrame(rows).drop_duplicates(["family", "tf"])          # one row per (family, TF)
n_motifs = {f: sum(1 for m, x in dom.items() if x == f) for f in d.family.unique()}
d["n_motifs"] = d.family.map(n_motifs)

keep = d.family.value_counts()
d = d[d.family.isin(keep[keep >= MIN_READERS].index)].copy()
d.to_csv(f"{SD}/results/family_reader_constraint.csv", index=False)

overall = np.median(list({t: v for t, v in
                          ((x.upper(), loeuf.get(x.upper())) for l in assign.values() for x in l)
                          if v is not None and not pd.isna(v)}.values()))
print(f"\n{d.family.nunique()} families with >= {MIN_READERS} reader TFs, "
      f"{len(d)} reader-TF observations; all-core median LOEUF {overall:.3f}")
print(f"\n{'family':<20}{'motifs':>7}{'readers':>9}{'median':>9}{'IQR':>18}")
for f, g in sorted(d.groupby("family"), key=lambda kv: kv[1].loeuf.median()):
    q1, q3 = np.percentile(g.loeuf, [25, 75])
    print(f"{f:<20}{int(g.n_motifs.iloc[0]):>7}{len(g):>9}{g.loeuf.median():>9.3f}"
          f"{f'{q1:.2f}-{q3:.2f}':>18}")

H, p = stats.kruskal(*[g.loeuf.values for _, g in d.groupby("family")])
print(f"\nKruskal-Wallis across families: H={H:.2f}, P={p:.4g}")
print(f"wrote results/family_reader_constraint.csv")
