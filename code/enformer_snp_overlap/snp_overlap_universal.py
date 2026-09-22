"""
Universal Regulator SNP Overlap Analysis

Post-hoc analysis comparing:
  1. Universal regulator motifs (43 de novo motifs from CNN-CR, present in ≥25/116 cell lines)
  2. Control motifs — average of 10 independent sets of 43 JASPAR motifs (matched negative set)

Against causal SNPs (PIP > 0.9) and non-causal SNPs (PIP < 0.01) from the Enformer study.

Overlap ratio = # causal SNPs overlapping motif occurrences / total motif occurrences
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import wilcoxon, ttest_1samp
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

mpl.style.use("default")
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# ─── Paths ────────────────────────────────────────────────────────────────────
ENFORMER_DIR       = "/stg3/data1/sam/enhancer_prediction/enformer_comparison/eqtl-enformer"
MAPPING_CSV        = "/stg3/data1/sam/enhancer_prediction/geuvadis/predixscan/en_models_to_BSSID.csv"
COORD_DIR          = "/stg3/data1/sam/enhancer_prediction/data_generated"
FIMO_UNIVERSAL_DIR = "/stg3/data1/sam/enhancer_prediction/multi_agent/posthoc_analysis/results/fimo_universal"
# 10 control set FIMO directories (set_0 through set_9)
FIMO_MULTI_BASE    = "/stg3/data1/sam/enhancer_prediction/multi_agent/posthoc_analysis/results/fimo_multi"
N_CONTROL_SETS     = 10
RESULTS_DIR        = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SNP_THRESHOLD = 500   # minimum causal SNPs per cell line
TOLERANCE_BP  = 4     # ±bp tolerance for motif center vs SNP position


# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_coord_lookup(bssid):
    path = f"{COORD_DIR}/{bssid}.coord.exp.df.final.csv"
    raw = pd.read_csv(path, usecols=["chr", "start", "end", "strand"])
    raw["gene_idx"]   = raw.index // 20
    raw["window_bin"] = raw.index % 20
    coord = raw[(raw["window_bin"] < 10) & raw["chr"].notna()].copy()
    coord = coord.rename(columns={"chr": "gene_chr", "start": "win_start",
                                  "end": "win_end", "strand": "gene_strand"})
    coord["win_start"] = coord["win_start"].astype(int)
    coord["win_end"]   = coord["win_end"].astype(int)
    return coord[["gene_idx", "window_bin", "gene_chr", "win_start", "win_end", "gene_strand"]]


def fimo_to_genomic_centers(fimo_path, coord_df):
    fimo = pd.read_csv(fimo_path, sep="\t",
                       usecols=["sequence_name", "start", "stop"])
    fimo = fimo.rename(columns={"start": "fimo_start", "stop": "fimo_stop"})

    parts = fimo["sequence_name"].str.split("_")
    fimo["gene_idx"]   = parts.str[1].astype(int)
    fimo["window_bin"] = parts.str[2].astype(int)

    merged = fimo.merge(coord_df, on=["gene_idx", "window_bin"], how="inner")

    local_mid = (merged["fimo_start"] + merged["fimo_stop"]) // 2
    plus = merged["gene_strand"] == "+"
    merged["center"] = np.where(
        plus,
        merged["win_start"] + local_mid,
        merged["win_end"]   - local_mid,
    ).astype(int)

    return merged[["gene_chr", "center"]].rename(columns={"gene_chr": "chr"}), len(fimo)


def load_snp_df(vcf_path):
    data = np.genfromtxt(vcf_path, delimiter="\t", dtype=str,
                         skip_header=1, usecols=[0, 1])
    df = pd.DataFrame(data, columns=["chr", "coordinate"])
    df["coordinate"] = df["coordinate"].astype(int)
    return df.drop_duplicates()


def count_snp_overlap(snp_df, motif_centers_df, tolerance=TOLERANCE_BP):
    snp_sorted = snp_df.sort_values("coordinate")
    total = 0
    for chrom, motif_grp in motif_centers_df.groupby("chr"):
        snp_chr = snp_sorted[snp_sorted["chr"] == chrom]
        if snp_chr.empty:
            continue
        motif_sorted = motif_grp.sort_values("center")
        overlaps = pd.merge_asof(
            motif_sorted[["center"]],
            snp_chr[["coordinate"]],
            left_on="center",
            right_on="coordinate",
            direction="nearest",
            tolerance=tolerance,
        )
        hit = overlaps[overlaps["coordinate"].notna()].drop_duplicates(subset="coordinate")
        total += len(hit)
    return total


# ─── Main loop ────────────────────────────────────────────────────────────────
mapping = pd.read_csv(MAPPING_CSV)
records = []

for _, row in mapping.iterrows():
    tissue = row["PrediXscan_name"]
    bssid  = row["BSSID"]
    print(f"Processing {tissue} ({bssid}) ...", flush=True)

    pos_vcf = f"{ENFORMER_DIR}/{tissue}_pos.vcf"
    neg_vcf = f"{ENFORMER_DIR}/{tissue}_neg.vcf"
    if not os.path.exists(pos_vcf):
        print(f"  skipped – no VCF file")
        continue

    pos_snps = load_snp_df(pos_vcf)
    neg_snps = load_snp_df(neg_vcf)
    n_pos    = len(pos_snps)
    n_neg    = len(neg_snps)

    if n_pos != n_neg:
        print(f"  skipped – pos/neg mismatch ({n_pos} vs {n_neg})")
        continue

    coord_df = build_coord_lookup(bssid)

    # Universal regulators
    uni_centers, uni_total = fimo_to_genomic_centers(
        f"{FIMO_UNIVERSAL_DIR}/{bssid}_fimo.txt", coord_df)
    uni_pos = count_snp_overlap(pos_snps, uni_centers)
    uni_neg = count_snp_overlap(neg_snps, uni_centers)

    # 10 control sets
    ctrl_pos_list, ctrl_neg_list, ctrl_total_list = [], [], []
    for s in range(N_CONTROL_SETS):
        fimo_path = f"{FIMO_MULTI_BASE}/set_{s}/{bssid}_fimo.txt"
        if not os.path.exists(fimo_path):
            print(f"  WARNING: missing control set {s} for {bssid}, skipping this set")
            continue
        c_centers, c_total = fimo_to_genomic_centers(fimo_path, coord_df)
        ctrl_pos_list.append(count_snp_overlap(pos_snps, c_centers))
        ctrl_neg_list.append(count_snp_overlap(neg_snps, c_centers))
        ctrl_total_list.append(c_total)

    if not ctrl_pos_list:
        print(f"  skipped – no control FIMO files found")
        continue

    rec = {
        "tissue":                     tissue,
        "bssid":                      bssid,
        "num_pos_SNP":                n_pos,
        "num_neg_SNP":                n_neg,
        "SNP_in_universal_pos":       uni_pos,
        "SNP_in_universal_neg":       uni_neg,
        "total_universal":            uni_total,
        "n_control_sets":             len(ctrl_pos_list),
        "SNP_in_control_pos_mean":    np.mean(ctrl_pos_list),
        "SNP_in_control_pos_std":     np.std(ctrl_pos_list, ddof=1) if len(ctrl_pos_list) > 1 else 0.0,
        "SNP_in_control_neg_mean":    np.mean(ctrl_neg_list),
        "SNP_in_control_neg_std":     np.std(ctrl_neg_list, ddof=1) if len(ctrl_neg_list) > 1 else 0.0,
        "total_control_mean":         np.mean(ctrl_total_list),
    }
    # Store per-set values for ratio error bars
    for s, (cp, ct) in enumerate(zip(ctrl_pos_list, ctrl_total_list)):
        rec[f"ctrl_pos_set_{s}"]   = cp
        rec[f"ctrl_total_set_{s}"] = ct
    records.append(rec)

    print(f"  universal {uni_pos}/{uni_total}, "
          f"control pos mean={np.mean(ctrl_pos_list):.1f}±{np.std(ctrl_pos_list, ddof=1) if len(ctrl_pos_list)>1 else 0:.1f}")

results_df = pd.DataFrame(records)
results_df.to_csv(f"{RESULTS_DIR}/SNP_overlap.csv", index=False)
print(f"\nSaved: {RESULTS_DIR}/SNP_overlap.csv")


# ─── Plots ────────────────────────────────────────────────────────────────────
df = pd.read_csv(f"{RESULTS_DIR}/SNP_overlap.csv")
df = df[df["num_pos_SNP"] > SNP_THRESHOLD].copy()
print(f"\n{len(df)} cell lines with >{SNP_THRESHOLD} causal SNPs")

df["universal_ratio"] = df["SNP_in_universal_pos"] / df["total_universal"]

# Per-set control ratios → mean and std across the 10 sets
ctrl_ratio_sets = []
for s in range(N_CONTROL_SETS):
    pos_col   = f"ctrl_pos_set_{s}"
    total_col = f"ctrl_total_set_{s}"
    if pos_col in df.columns and total_col in df.columns:
        ctrl_ratio_sets.append(df[pos_col] / df[total_col])

ctrl_ratio_mat      = np.column_stack(ctrl_ratio_sets)           # (n_tissues, n_sets)
df["control_ratio_mean"] = ctrl_ratio_mat.mean(axis=1)
df["control_ratio_std"]  = ctrl_ratio_mat.std(axis=1, ddof=1)

uni_color  = "#E1CCB4"
ctrl_color = "#6C6D7E"   # dark colour previously used for all-JASPAR


def fmt_pval(p):
    return f"{p:.1e}"


def annotate_pvals(ax, x, heights, pvals, fontsize=5.5):
    y_max = ax.get_ylim()[1]
    step  = y_max * 0.02
    for xi, h, lbl in zip(x, heights, pvals):
        ax.text(xi, h + step, lbl, ha="center", va="bottom",
                fontsize=fontsize, rotation=90)


# ── Figure 1: Overlap ratio per cell line (universal vs averaged control) ────
x     = np.arange(len(df))
width = 0.35

ratio_pvals = []
for i in range(len(df)):
    _, p = ttest_1samp(ctrl_ratio_mat[i], popmean=df["universal_ratio"].iloc[i], alternative="less")
    ratio_pvals.append(fmt_pval(p))

fig, ax = plt.subplots(figsize=(10, 2))
ax.bar(x - width / 2, df["universal_ratio"], width,
       label="Universal regulators", color=uni_color)
ax.bar(x + width / 2, df["control_ratio_mean"], width,
       label=f"Control motifs (mean of {N_CONTROL_SETS} sets)", color=ctrl_color,
       yerr=df["control_ratio_std"], capsize=3,
       error_kw=dict(elinewidth=0.8, ecolor="black", capthick=0.8))
ax.set_ylabel("Overlap Ratio", fontsize=8)
ax.set_title("Comparison of Universal Regulators and Control Motif Ratios", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(df["tissue"], rotation=45, ha="right", fontsize=7)
ax.legend(frameon=False, fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
annotate_pvals(ax, x,
               np.maximum(df["universal_ratio"].values,
                          df["control_ratio_mean"].values + df["control_ratio_std"].values),
               ratio_pvals)
plt.savefig(f"{RESULTS_DIR}/SNP_overlap_ratio_universalVSjaspar.pdf", dpi=300, bbox_inches="tight", transparent=True)
plt.savefig(f"{RESULTS_DIR}/SNP_overlap_ratio_universalVSjaspar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/SNP_overlap_ratio_universalVSjaspar.pdf")

# ── Figure 1b: Absolute causal SNP counts – universal vs averaged control ────
df["control_pos_mean"] = df["SNP_in_control_pos_mean"]
df["control_pos_std"]  = df["SNP_in_control_pos_std"]

x     = np.arange(len(df))
width = 0.35

count_pvals = []
for i in range(len(df)):
    ctrl_counts = np.array([df[f"ctrl_pos_set_{s}"].iloc[i] for s in range(N_CONTROL_SETS)
                            if f"ctrl_pos_set_{s}" in df.columns])
    _, p = ttest_1samp(ctrl_counts, popmean=df["SNP_in_universal_pos"].iloc[i], alternative="less")
    count_pvals.append(fmt_pval(p))

fig, ax = plt.subplots(figsize=(10, 2))
ax.bar(x - width / 2, df["SNP_in_universal_pos"], width,
       label="Universal regulators", color=uni_color)
ax.bar(x + width / 2, df["control_pos_mean"], width,
       label=f"Control motifs (mean of {N_CONTROL_SETS} sets)", color=ctrl_color,
       yerr=df["control_pos_std"], capsize=3,
       error_kw=dict(elinewidth=0.8, ecolor="black", capthick=0.8))
ax.set_ylabel("Causal SNP Count", fontsize=8)
ax.set_title("Causal SNP Overlap: Universal Regulators vs Control", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(df["tissue"], rotation=45, ha="right", fontsize=7)
ax.legend(frameon=False, fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
annotate_pvals(ax, x,
               np.maximum(df["SNP_in_universal_pos"].values,
                          df["control_pos_mean"].values + df["control_pos_std"].values),
               count_pvals)
plt.savefig(f"{RESULTS_DIR}/SNP_overlap_count_universalVScontrol.pdf", dpi=300, bbox_inches="tight", transparent=True)
plt.savefig(f"{RESULTS_DIR}/SNP_overlap_count_universalVScontrol.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/SNP_overlap_count_universalVScontrol.pdf")

# ── Figure 2: Causal vs non-causal SNP counts for universal regulators ───────
x     = np.arange(len(df))
width = 0.35
pos_color = "#EE8383"
neg_color = "#9999F8"

fig, ax = plt.subplots(figsize=(10, 2))
ax.bar(x - width / 2, df["SNP_in_universal_pos"], width,
       label="pos SNP overlap", color=pos_color)
ax.bar(x + width / 2, df["SNP_in_universal_neg"], width,
       label="neg SNP overlap", color=neg_color)
ax.set_ylabel("SNP Count")
ax.set_title("Comparison of Pos and Neg SNP Overlap")
ax.set_xticks(x)
ax.set_xticklabels(df["tissue"], rotation=45, ha="right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.savefig(f"{RESULTS_DIR}/SNP_overlap_posVSneg.pdf", dpi=300, bbox_inches="tight", transparent=True)
plt.savefig(f"{RESULTS_DIR}/SNP_overlap_posVSneg.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/SNP_overlap_posVSneg.pdf")

# ── Figure 3: Paired scatter – universal vs averaged control ratios ───────────
fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter(df["control_ratio_mean"], df["universal_ratio"], s=30, color="#555577", zorder=3)
lim = max(df["control_ratio_mean"].max(), df["universal_ratio"].max()) * 1.15
ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x")
ax.set_xlabel("Control ratio (mean)", fontsize=9)
ax.set_ylabel("Universal ratio", fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/SNP_overlap_scatter.pdf", dpi=300, bbox_inches="tight", transparent=True)
plt.savefig(f"{RESULTS_DIR}/SNP_overlap_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/SNP_overlap_scatter.pdf")

# ── Statistical tests ─────────────────────────────────────────────────────────
if len(df) >= 10:
    stat, p_vs_ctrl = wilcoxon(df["universal_ratio"], df["control_ratio_mean"],
                               alternative="greater")
    print(f"\nWilcoxon (one-sided, universal > control mean):  p = {p_vs_ctrl:.4f}")

    stats_rows = [
        {"comparison": "universal_vs_control_mean", "direction": "greater",
         "p_value": p_vs_ctrl, "n_cell_lines": len(df),
         "n_control_sets": N_CONTROL_SETS},
    ]
    pd.DataFrame(stats_rows).to_csv(f"{RESULTS_DIR}/wilcoxon_stats.csv", index=False)
    print("Saved: results/wilcoxon_stats.csv")

print("\nDone!")
