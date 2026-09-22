#!/usr/bin/env python
"""
09z5 -- the cross-line switch statistic recomputed PER GENE (user spec 2026-09-01).

09x/09z4 measured a factor's switch magnitude as a ratio of two aggregates:

    | log2( median_g |eff_K562(g)|  /  median_g |eff_HEK293(g)| ) |

i.e. each screen was summarised over the target genes first and the two summaries were
divided once.  The user's revision keeps the per-gene absolute value but moves the ratio
inside, so every gene contributes its own ratio and the averaging happens afterwards:

    step 1   per-gene absolute value in each cell line   -> |eff_K(g)|, |eff_H(g)|
    step 2   per-gene ratio                              -> |eff_K(g)| / |eff_H(g)|
    step 3   mean over the target genes
    step 4   absolute value

Step 3 is taken in log2 space, so the mean is over log2 of the per-gene ratio.  Two reasons:
the arithmetic mean of raw ratios is not symmetric under swapping the two screens (2x and
0.5x would average to 1.25x rather than 1x) and it is unbounded above but floored at 0, so a
single small denominator sets the answer; and step 4 would be a no-op on raw ratios, which are
positive by construction.  In log2 space the mean is the geometric mean of the per-gene ratios
and step 4 does real work, discarding only which screen was louder.

The per-gene ratio is heavy tailed -- across the cached reader values the per-gene log2 ratio
has sd 2.26 and reaches +-14.7 -- so the mean is sensitive to genes whose denominator is near
zero.  The primary column is the user's spec (mean_log2); median_log2 and trimmean_log2 (10%
trimmed each tail) are carried beside it as robustness, and log2_target reproduces the old
statistic on the identical genes so the two can be compared row for row.  No floor, cutoff or
pseudocount is applied anywhere: genes are dropped only where the ratio is mathematically
undefined (a zero or non-finite effect in either screen), and the count is reported.

Everything else is held identical to 09x: the same motif target gene sets (union of that
motif's readers' ChIP top-90% promoter targets), the same quantile-normalized effects, the same
control admission rule (perturbed with effective knockdown in both screens, not in the reader
roster, ENCODE4 ChIP-verified in at least one line), and the same collapse of each control to
one value per factor so the 22 gene sets do not pseudo-replicate the background.

Usage:  python 09z5_switch_magnitude_pergene.py top90chip        # ~10 min, loads both screens

Writes  results/prof_plan/cross_line_pergene_{tag}.csv
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
from scipy import stats

ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
TAG = ARGS[0] if ARGS else "top90chip"
BIND_FILE = {"top10chip": "reader_binding.parquet",
             "top90chip": "reader_binding_top90.parquet"}[TAG]
MIN_GENES = 300          # motif target set size, as in 09x
MIN_OK = 50              # usable target genes per factor x motif, as in 09x
TRIM = 0.10              # robustness column only, never the primary

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "08c_alltargets_stats.py")).read()
exec(compile(src.split("DEFS = [")[0], "08c_prefix", "exec"), globals())

OUT = os.path.join(CHIP, "results/prof_plan")
RS = os.path.join(CHIP, "reader_switch/out")
LINES = ["K562", "HEK293"]
SHORT = lambda m: m.replace("universal_", "M")

# ============================================================================ setup (as 09x)
SHARED = sorted(shared)
POS = {g: i for i, g in enumerate(SHARED)}
kidx = np.array([k_col[g] for g in SHARED])
oidx = np.array([o_col[g] for g in SHARED])

roster = {m: sorted({norm(t) for t in v}) for m, v in json.load(open(os.path.join(
    ROOT, "multi_agent/posthoc_analysis/results/step06_JASPAR_assignments.json"))).items()}
ROSTER_ALL = {t for v in roster.values() for t in v}
assayed = {}
for cell in LINES:
    with open(os.path.join(CHIP, f"data/{cell}/tf_peak_sets.pkl"), "rb") as fh:
        assayed[cell] = {norm(t) for t in pickle.load(fh)}


def eff_here(tf, cell):
    if cell == "K562":
        return tf in k_perturbed and k_effkd(tf)
    return tf in o_perturbed and o_effkd_mask.get(tf, False)


_e = {}


def effect(tf, cell):
    """per-gene ABSOLUTE knockout effect, quantile-normalized -- step 1 of the spec"""
    if (tf, cell) not in _e:
        _e[(tf, cell)] = np.abs(k_delta(tf)[kidx] if cell == "K562" else o_delta(tf)[oidx])
    return _e[(tf, cell)]


# ---- motif gene sets, identical to 09x/09w -------------------------------------------------
rb = pd.read_parquet(f"{RS}/{BIND_FILE}")
rb = rb[rb["cell"].isin(LINES)].copy()
rb["rn"] = rb["reader"].map(norm)
rb = rb[rb["ensembl"].isin(set(SHARED))]
sets = {}
for (cell, m, rn), d in rb.groupby(["cell", "motif_id", "rn"]):
    if m in roster and rn in roster[m] and rn in assayed[cell]:
        sets[(cell, m, rn)] = set(d["ensembl"])

MOTIF = {}
for m in sorted(roster, key=lambda x: int(x.split("_")[1])):
    genes = sorted(set().union(*[sets[(c, mm, r)] for (c, mm, r) in sets if mm == m]) or set())
    if len(genes) < MIN_GENES:
        continue
    MOTIF[m] = (genes, np.array([POS[g] for g in genes]))
print(f"\n[{TAG}] {len(MOTIF)} motifs with >= {MIN_GENES} target genes", flush=True)

# ---- control pools, identical to 09x -------------------------------------------------------
print("Selecting control TFs (ChIP-seq in at least one line, knockdown in both screens) ...",
      flush=True)
chip_any = set()
for cell in sorted(os.listdir(ENCODE4)):
    p = os.path.join(ENCODE4, cell)
    if os.path.isdir(p):
        chip_any |= {norm(d.replace("_human", "")) for d in os.listdir(p)
                     if os.path.isdir(os.path.join(p, d))}
chip_kh = assayed["K562"] | assayed["HEK293"]

cand = sorted((k_perturbed & o_perturbed) - ROSTER_ALL)
both_eff = [t for t in cand if eff_here(t, "K562") and eff_here(t, "HEK293")]
_kraw.clear()
CTRL = {"chip": [t for t in both_eff if t in chip_kh],
        "chip_other": [t for t in both_eff if t in chip_any and t not in chip_kh]}
print(f"  {len(both_eff)} non-reader perturbations with effective knockdown in both screens; "
      f"{len(CTRL['chip']) + len(CTRL['chip_other'])} are ChIP-verified TFs "
      f"({len(CTRL['chip'])} in K562/HEK293, {len(CTRL['chip_other'])} only elsewhere)", flush=True)

# ============================================================================ the statistic
_dropped = [0, 0]        # [genes dropped as undefined, genes seen]


def record(tf, m, group, chip_in):
    genes, idx = MOTIF[m]
    vK, vH = effect(tf, "K562")[idx], effect(tf, "HEK293")[idx]

    # a gene is usable when both screens give a finite, strictly positive magnitude, so that
    # its ratio is defined.  No floor or pseudocount -- only mathematically undefined dropped.
    ok = np.isfinite(vK) & np.isfinite(vH) & (vK > 0) & (vH > 0)
    _dropped[1] += ok.size
    _dropped[0] += int((~ok).sum())
    if ok.sum() < MIN_OK:
        return None
    a, b = vK[ok], vH[ok]

    lr = np.log2(a / b)                              # step 2, per-gene ratio, in log2
    mean_log2 = float(lr.mean())                     # step 3
    med_log2 = float(np.median(lr))
    trim_log2 = float(stats.trim_mean(lr, TRIM))

    mK, mH = float(np.median(a)), float(np.median(b))
    gK, gH = float(np.median(vK[np.isfinite(vK)])), float(np.median(vH[np.isfinite(vH)]))
    return dict(motif=m, factor=tf, group=group, chip_in=chip_in, n_genes=int(ok.sum()),
                mean_log2=mean_log2, median_log2=med_log2, trimmean_log2=trim_log2,
                magnitude=abs(mean_log2),                       # step 4, the primary
                magnitude_median=abs(med_log2), magnitude_trim=abs(trim_log2),
                log2_target=float(np.log2(mK / mH)),            # the OLD statistic, same genes
                log2_global=float(np.log2(gK / gH)),
                sd_log2=float(lr.std()))


rows = []
for m in MOTIF:
    for r in sorted({r for (c, mm, r) in sets if mm == m}):
        if eff_here(r, "K562") and eff_here(r, "HEK293"):
            where = ("both" if all((c, m, r) in sets for c in LINES)
                     else ("K562" if ("K562", m, r) in sets else "HEK293"))
            rec = record(r, m, "reader", where)
            if rec:
                rows.append(rec)
for grp in ("chip", "chip_other"):
    for t in CTRL[grp]:
        for m in MOTIF:
            rec = record(t, m, grp, "-")
            if rec:
                rows.append(rec)
    print(f"  scored {len(CTRL[grp])} {grp} factors on {len(MOTIF)} motif gene sets", flush=True)

D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/cross_line_pergene_{TAG}.csv", index=False)
print(f"\nwrote {OUT}/cross_line_pergene_{TAG}.csv  ({len(D)} rows)")
print(f"genes dropped as undefined (zero or non-finite in a screen): "
      f"{_dropped[0]} of {_dropped[1]} ({100.0 * _dropped[0] / max(1, _dropped[1]):.4f}%)")

# ============================================================================ the comparison
pick = pd.read_csv(f"{OUT}/switch_figready_pairs_{TAG}.csv")
sel = set()
for r in pick.itertuples(index=False):
    sel.add((r.motif, r.reader_K)); sel.add((r.motif, r.reader_H))

R, A = D[D.group == "reader"], D[D.group != "reader"]
print(f"\nbackground: {A.factor.nunique()} ChIP-verified non-reader TFs")

for col, label in (("magnitude", "mean of per-gene log2 ratio  [USER SPEC]"),
                   ("magnitude_trim", f"{int(TRIM*100)}% trimmed mean  [robustness]"),
                   ("magnitude_median", "median of per-gene log2 ratio  [robustness]"),
                   ("log2_target", "OLD: ratio of the two medians")):
    a = A.copy(); a["v"] = a[col].abs()
    ctrl = a.groupby("factor").v.median().values
    r = R.copy(); r["v"] = r[col].abs()
    disp = r[[(x.motif, x.factor) in sel for x in r.itertuples(index=False)]]
    allr = r.groupby("factor").v.median().values
    pd_ = stats.mannwhitneyu(disp.v.values, ctrl, alternative="greater").pvalue
    pa_ = stats.mannwhitneyu(allr, ctrl, alternative="greater").pvalue
    print(f"\n--- {label} ---")
    print(f"  background n={len(ctrl):4d}  median {2**np.median(ctrl):.3f}x")
    print(f"  8 displayed pairs n={len(disp):3d}  median {2**np.median(disp.v):.3f}x  "
          f"P={pd_:.4f}  ({int((disp.v.values > np.median(ctrl)).sum())} of "
          f"{len(disp)} above bg median)")
    print(f"  ALL readers      n={len(allr):3d}  median {2**np.median(allr):.3f}x  P={pa_:.4f}")
