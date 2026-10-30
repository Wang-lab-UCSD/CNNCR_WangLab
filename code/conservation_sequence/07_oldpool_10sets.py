"""
07 — Fig 3c, d, e with the TEN old-pool control sets (posthoc_analysis/results/control_motifs_set_{0..9}.meme,
the same ten sets Fig 4a/b already use) instead of the single published control set.

Inputs: the per-biosample, per-motif phyloP summaries written by 05_pool_conservation.py from the
334-motif pool scan (results/pool_conservation/) and from the 13-motif extra scan of the old-pool
motifs absent from that file (results/pool_conservation_extra/). Core and control are therefore
scored by one run of the same FIMO command and the same best-per-gene phyloP collapse.

Writes results/oldpool10/
  fig3_rows.csv           bssid, motif_id, group, n_genes, median_score (all biosample x motif values)
  fig3_permotif.csv       motif_id, group (core|control), n_sets, sets, median_phylop, cg
                          (median over biosamples of the per-biosample median; each control
                          motif appears ONCE even if drawn into several sets)
  fig3_per_biosample.csv  bssid, core_median, ctrl_mean (mean over the 10 sets of that set's
                          per-biosample median), ctrl_sd, ctrl_min, ctrl_max, ctrl_set_0..9
  fig3_stats.csv          per set: 3c MWU P, 3d common-slope ANCOVA (adjusted advantage, P,
                          slopes), 3e core wins / n biosamples + sign-test P;
                          plus 'pooled' (distinct control motifs, each once) for 3c/3d and
                          'mean_of_sets' for 3e (sign test on ctrl_mean)
"""
import os, glob
import numpy as np, pandas as pd
from scipy import stats
# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PH = ENHANCER_ROOT + "/multi_agent/posthoc_analysis"
OUT = "results/oldpool10"; os.makedirs(OUT, exist_ok=True)
members = lambda p: [l.split()[1] for l in open(p) if l.startswith("MOTIF")]
SETS = {int(f.rsplit("_", 1)[1].split(".")[0]): members(f) for f in sorted(glob.glob(f"{PH}/results/control_motifs_set_*.meme"))}
assert len(SETS) == 10
allctrl = sorted({m for s in SETS.values() for m in s})

A = sorted(glob.glob("results/pool_conservation/*_motif_summary.tsv"))
B = sorted(glob.glob("results/pool_conservation_extra/*_motif_summary.tsv"))
bs_a = {os.path.basename(f).split("_")[0] for f in A}; bs_b = {os.path.basename(f).split("_")[0] for f in B}
both = sorted(bs_a & bs_b)
print(f"pool summaries {len(A)}, extra summaries {len(B)}, biosamples with both: {len(both)}")
if len(both) < 100: raise SystemExit("fewer than 100 biosamples have both scans; not running")
D = pd.concat([pd.read_csv(f, sep="\t") for f in A + B if os.path.basename(f).split("_")[0] in both], ignore_index=True)
D = D[D.group.eq("core") | D.motif_id.isin(allctrl)]
D.loc[D.motif_id.isin(allctrl) & ~D.group.eq("core"), "group"] = "control"
core_ids = sorted(D.loc[D.group == "core", "motif_id"].unique())

def parse(path):
    out, name, rows, inm = {}, None, [], False
    for ln in open(path):
        if ln.startswith("MOTIF"):
            if name and rows: out[name] = np.array(rows, float)
            name, rows, inm = ln.split()[1], [], False
        elif ln.startswith("letter-probability"): inm = True
        elif inm:
            p = ln.split()
            if len(p) == 4:
                try: rows.append([float(x) for x in p])
                except ValueError: inm = False
            else: inm = False
    if name and rows: out[name] = np.array(rows, float)
    return out
W = parse(ENHANCER_ROOT + "/multi_agent/motif_pipeline/results/universal_regulator_motifs.meme")
W.update(parse(ENHANCER_ROOT + "/JASPAR2022_CORE_vertebrates_non_redundant_unique_names.meme"))
CG = {k: float(np.mean(v[:, 1] + v[:, 2])) for k, v in W.items()}

permotif = D.groupby("motif_id")["median_score"].median()
core_med = permotif.reindex(core_ids).dropna()
pm = [dict(motif_id=m, group="core", n_sets=0, sets="", median_phylop=v, cg=CG[m]) for m, v in core_med.items()]
for m in allctrl:
    if m in permotif.index:
        ss = [i for i, s in SETS.items() if m in s]
        pm.append(dict(motif_id=m, group="control", n_sets=len(ss), sets=";".join(map(str, ss)), median_phylop=permotif[m], cg=CG[m]))
pm = pd.DataFrame(pm); pm.to_csv(f"{OUT}/fig3_permotif.csv", index=False)
# every biosample x motif value (what the panel-c violin draws), each distinct control motif once
D[["bssid", "motif_id", "group", "n_genes", "median_score"]].to_csv(f"{OUT}/fig3_rows.csv", index=False)
print(f"per-motif: {len(core_med)} core, {(pm.group=='control').sum()} distinct control motifs "
      f"(of {len(allctrl)} in the ten sets)")

def ancova(yc, xc, yk, xk):
    y = np.r_[yc, yk]; x = np.r_[xc, xk]; g = np.r_[np.ones(len(yc)), np.zeros(len(yk))]
    X = np.column_stack([np.ones_like(x), x, g]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; dof = len(y) - 3; cov = (r @ r / dof) * np.linalg.pinv(X.T @ X)
    t = b[2] / np.sqrt(cov[2, 2])
    sc = np.polyfit(xc, yc, 1)[0]; sk = np.polyfit(xk, yk, 1)[0]
    return b[2], 2 * stats.t.sf(abs(t), dof), b[1], sc, sk
xc = np.array([CG[m] for m in core_med.index])

per_bs_core = D[D.group == "core"].groupby("bssid")["median_score"].median()
rows, bsrows = [], {b: dict(bssid=b, core_median=v) for b, v in per_bs_core.items()}
for i, s in sorted(SETS.items()):
    cm = permotif.reindex(s).dropna(); xk = np.array([CG[m] for m in cm.index])
    p3c = stats.mannwhitneyu(core_med.values, cm.values, alternative="two-sided").pvalue
    adj, p3d, slope, sc, sk = ancova(core_med.values, xc, cm.values, xk)
    pbs = D[D.motif_id.isin(s)].groupby("bssid")["median_score"].median()
    for b, v in pbs.items(): bsrows[b][f"ctrl_set_{i}"] = v
    j = pd.concat([per_bs_core, pbs], axis=1, keys=["c", "k"]).dropna()
    nw = int((j.c > j.k).sum())
    rows.append(dict(set=i, n_ctrl=len(cm), core_median=core_med.median(), ctrl_median=cm.median(),
                     diff=core_med.median() - cm.median(), p_fig3c=p3c, cg_core=xc.mean(), cg_ctrl=xk.mean(),
                     cg_adjusted_advantage=adj, p_fig3d=p3d, common_slope=slope, slope_core=sc, slope_ctrl=sk,
                     n_biosamples=len(j), n_core_wins=nw,
                     p_fig3e=stats.binomtest(nw, len(j), 0.5, alternative="greater").pvalue))
pooled = permotif.reindex(allctrl).dropna(); xk = np.array([CG[m] for m in pooled.index])
adj, p3d, slope, sc, sk = ancova(core_med.values, xc, pooled.values, xk)
rows.append(dict(set="pooled", n_ctrl=len(pooled), core_median=core_med.median(), ctrl_median=pooled.median(),
                 diff=core_med.median() - pooled.median(),
                 p_fig3c=stats.mannwhitneyu(core_med.values, pooled.values, alternative="two-sided").pvalue,
                 cg_core=xc.mean(), cg_ctrl=xk.mean(), cg_adjusted_advantage=adj, p_fig3d=p3d,
                 common_slope=slope, slope_core=sc, slope_ctrl=sk))
BS = pd.DataFrame(bsrows.values()); sc_ = [c for c in BS.columns if c.startswith("ctrl_set_")]
BS["ctrl_mean"] = BS[sc_].mean(axis=1); BS["ctrl_sd"] = BS[sc_].std(axis=1, ddof=1)
BS["ctrl_min"] = BS[sc_].min(axis=1); BS["ctrl_max"] = BS[sc_].max(axis=1)
BS = BS[["bssid", "core_median", "ctrl_mean", "ctrl_sd", "ctrl_min", "ctrl_max"] + sc_].sort_values("bssid")
BS.to_csv(f"{OUT}/fig3_per_biosample.csv", index=False)
nw = int((BS.core_median > BS.ctrl_mean).sum())
rows.append(dict(set="mean_of_sets", n_biosamples=len(BS), n_core_wins=nw,
                 p_fig3e=stats.binomtest(nw, len(BS), 0.5, alternative="greater").pvalue,
                 n_all_sets_win=int((BS.core_median > BS.ctrl_max).sum())))
ST = pd.DataFrame(rows); ST.to_csv(f"{OUT}/fig3_stats.csv", index=False)
pd.set_option("display.width", 250)
print(ST[["set", "n_ctrl", "ctrl_median", "diff", "p_fig3c", "cg_ctrl", "cg_adjusted_advantage", "p_fig3d",
          "slope_core", "slope_ctrl", "n_biosamples", "n_core_wins", "p_fig3e"]].round(4).to_string(index=False))
print(f"biosamples where core > every one of the 10 set medians: {int((BS.core_median > BS.ctrl_max).sum())}/{len(BS)}")
