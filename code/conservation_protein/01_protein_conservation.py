"""
Analysis A: Protein Conservation of Universal Regulator TFs

1. Load the 287 TFs mapped to the 38 universal motifs (from JASPAR assignments).
2. Match each TF to gnomAD v2.1.1 constraint metrics (pLI, LOEUF, mis_z).
3. Define background = all TFs in TF_Information.txt NOT in the universal set.
4. Mann-Whitney U test: universal TFs vs background TFs.
5. Protein family breakdown of universal TFs.
6. Save plots to results/.

Run: python 01_protein_conservation.py
"""

import os
import re
import io
import gzip
import json
import requests
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
# The data root: the work directory you created with bin/setup_work_dir.py and unpacked the
# GitHub and Zenodo data into. `source env.sh` exports it, or set it yourself (see PATHS.md).
ENHANCER_ROOT = os.environ.get("ENHANCER_ROOT")
if not ENHANCER_ROOT:
    raise SystemExit("set ENHANCER_ROOT to your work directory: source env.sh (see PATHS.md)")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── paths ──────────────────────────────────────────────────────────────────
JASPAR_ASSIGNMENTS = (
    ENHANCER_ROOT + "/multi_agent/posthoc_analysis/"
    "results/step06_JASPAR_assignments.json"
)
TF_INFO = ENHANCER_ROOT + "/motif_analysis/TF_Information.txt"
GNOMAD_CACHE = "results/gnomad_v2.1.1_constraint.tsv"
GNOMAD_URL = (
    "https://storage.googleapis.com/gcp-public-data--gnomad/release/2.1.1/"
    "constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz"
)

os.makedirs("results", exist_ok=True)

# ── 1. universal TF set ────────────────────────────────────────────────────
with open(JASPAR_ASSIGNMENTS) as fh:
    assignments = json.load(fh)

universal_tfs_raw = set()
for tf_list in assignments.values():
    universal_tfs_raw.update(tf_list)

# normalise to uppercase for matching
universal_tfs_upper = {t.upper() for t in universal_tfs_raw}
print(f"Universal TF set: {len(universal_tfs_raw)} unique TFs")

# ── 2. TF_Information → family + ENSG ─────────────────────────────────────
tf_info = pd.read_csv(TF_INFO, sep="\t")
tf_info = tf_info[tf_info["TF_Species"] == "Homo_sapiens"].copy()
tf_info["TF_Name_upper"] = tf_info["TF_Name"].str.upper()

# deduplicate: one row per TF_Name
tf_dedup = tf_info.drop_duplicates("TF_Name_upper")[
    ["TF_Name_upper", "TF_Name", "DBID", "Family_Name", "DBDs"]
].copy()
tf_dedup = tf_dedup.rename(columns={"DBID": "ENSG"})

# label each TF as universal or background
tf_dedup["is_universal"] = tf_dedup["TF_Name_upper"].isin(universal_tfs_upper)
n_universal_in_info = tf_dedup["is_universal"].sum()
n_background = (~tf_dedup["is_universal"]).sum()
print(f"Universal TFs found in TF_Information: {n_universal_in_info}")
print(f"Background TFs (other TFs in TF_Information): {n_background}")

# ── 3. gnomAD constraint scores ────────────────────────────────────────────
if os.path.exists(GNOMAD_CACHE):
    print("Loading gnomAD from cache...")
    gnomad = pd.read_csv(GNOMAD_CACHE, sep="\t")
else:
    print("Downloading gnomAD v2.1.1 constraint table (~50 MB)...")
    resp = requests.get(GNOMAD_URL, stream=True, timeout=120)
    resp.raise_for_status()
    raw = b"".join(resp.iter_content(chunk_size=1 << 20))
    with gzip.open(io.BytesIO(raw), "rt") as fh:
        gnomad = pd.read_csv(fh, sep="\t", low_memory=False)
    gnomad.to_csv(GNOMAD_CACHE, sep="\t", index=False)
    print(f"Saved gnomAD cache → {GNOMAD_CACHE}")

print(f"gnomAD rows: {len(gnomad)}, columns: {list(gnomad.columns[:10])}")

# keep useful columns; gnomAD v2.1.1 uses 'gene', 'gene_id', 'pLI', 'oe_lof_upper', 'mis_z'
keep_cols = [c for c in ["gene", "gene_id", "pLI", "oe_lof_upper", "mis_z"] if c in gnomad.columns]
gnomad = gnomad[keep_cols].dropna(subset=["pLI"]).copy()
gnomad["gene_upper"] = gnomad["gene"].str.upper()
# keep one row per gene (highest pLI transcript)
gnomad = gnomad.sort_values("pLI", ascending=False).drop_duplicates("gene_upper")

# ── 4. merge TF_Information with gnomAD ────────────────────────────────────
# primary join: by ENSG gene ID
tf_dedup["ENSG_clean"] = tf_dedup["ENSG"].str.split(".").str[0]  # strip version
gnomad["ensg_clean"] = gnomad["gene_id"].str.split(".").str[0]

merged = tf_dedup.merge(
    gnomad.rename(columns={"ensg_clean": "ENSG_clean"}),
    on="ENSG_clean", how="left"
)

# fallback join for rows that didn't match by ENSG: try gene name
no_match = merged["pLI"].isna()
if no_match.any():
    gnomad_by_name = gnomad.set_index("gene_upper")[["pLI", "oe_lof_upper", "mis_z"]]
    for idx in merged[no_match].index:
        name_upper = merged.at[idx, "TF_Name_upper"]
        if name_upper in gnomad_by_name.index:
            row_vals = gnomad_by_name.loc[name_upper]
            # loc may return Series or DataFrame; always take first row
            if isinstance(row_vals, pd.DataFrame):
                row_vals = row_vals.iloc[0]
            merged.at[idx, "pLI"]         = row_vals["pLI"]
            merged.at[idx, "oe_lof_upper"] = row_vals["oe_lof_upper"]
            merged.at[idx, "mis_z"]        = row_vals["mis_z"]

n_matched = merged["pLI"].notna().sum()
print(f"TFs matched to gnomAD scores: {n_matched} / {len(merged)}")

# ── 5. statistical comparison ──────────────────────────────────────────────
metrics = {
    "pLI": "pLI (LoF intolerance)",
    "oe_lof_upper": "LOEUF (LoF obs/exp upper bound)",
    "mis_z": "Missense z-score",
}

results = []
for col, label in metrics.items():
    if col not in merged.columns:
        continue
    uni_vals   = merged[merged["is_universal"] & merged[col].notna()][col].values
    bg_vals    = merged[~merged["is_universal"] & merged[col].notna()][col].values
    stat, pval = stats.mannwhitneyu(uni_vals, bg_vals, alternative="two-sided")
    results.append({
        "metric": label,
        "n_universal": len(uni_vals),
        "median_universal": np.median(uni_vals),
        "n_background": len(bg_vals),
        "median_background": np.median(bg_vals),
        "mannwhitney_U": stat,
        "p_value": pval,
    })
    print(f"{label}: universal median={np.median(uni_vals):.3f} (n={len(uni_vals)}), "
          f"background median={np.median(bg_vals):.3f} (n={len(bg_vals)}), p={pval:.4g}")

results_df = pd.DataFrame(results)
results_df.to_csv("results/constraint_comparison.csv", index=False)
print("Saved → results/constraint_comparison.csv")

# ── 6. violin plots ────────────────────────────────────────────────────────
valid_metrics = [col for col in metrics if col in merged.columns]
fig, axes = plt.subplots(1, len(valid_metrics), figsize=(5 * len(valid_metrics), 5))
if len(valid_metrics) == 1:
    axes = [axes]

COLORS = {"Universal TFs": "#e07b54", "All other TFs": "#5b8db8"}

for ax, col in zip(axes, valid_metrics):
    label = metrics[col]
    uni_vals = merged[merged["is_universal"] & merged[col].notna()][col].values
    bg_vals  = merged[~merged["is_universal"] & merged[col].notna()][col].values

    parts = ax.violinplot(
        [uni_vals, bg_vals], positions=[0, 1],
        showmedians=True, showextrema=True,
    )
    for pc, color in zip(parts["bodies"], list(COLORS.values())):
        pc.set_facecolor(color)
        pc.set_alpha(0.75)
    for key in ("cmedians", "cmaxes", "cmins", "cbars"):
        parts[key].set_color("black")
        parts[key].set_linewidth(1.2)

    # p-value annotation
    row = results_df[results_df["metric"] == label]
    if not row.empty:
        p = row["p_value"].values[0]
        p_str = f"p = {p:.3g}" if p >= 0.001 else f"p < 0.001"
        ax.set_title(f"{label}\n{p_str}", fontsize=11)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Universal TFs", "All other TFs"], fontsize=10)
    ax.set_ylabel(col, fontsize=10)

patches = [mpatches.Patch(color=c, label=l) for l, c in COLORS.items()]
fig.legend(handles=patches, loc="upper right", fontsize=9)
fig.suptitle("Evolutionary Constraint: Universal Regulator TFs vs Background", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig("results/constraint_violin.pdf", dpi=300, bbox_inches="tight")
fig.savefig("results/constraint_violin.png", dpi=150, bbox_inches="tight")
print("Saved → results/constraint_violin.pdf")

# ── 7. protein family breakdown ────────────────────────────────────────────
uni_families = (
    merged[merged["is_universal"] & merged["Family_Name"].notna()]
    ["Family_Name"].value_counts()
)

fig2, ax2 = plt.subplots(figsize=(8, max(4, len(uni_families) * 0.35)))
uni_families.sort_values().plot.barh(ax=ax2, color="#e07b54", edgecolor="white")
ax2.set_xlabel("Number of universal TFs", fontsize=11)
ax2.set_title("Protein Family Distribution of Universal Regulator TFs", fontsize=12)
ax2.tick_params(labelsize=9)
fig2.tight_layout()
fig2.savefig("results/family_distribution.pdf", dpi=300, bbox_inches="tight")
fig2.savefig("results/family_distribution.png", dpi=150, bbox_inches="tight")
print("Saved → results/family_distribution.pdf")

# ── 8. summary table ───────────────────────────────────────────────────────
uni_table = merged[merged["is_universal"]][
    ["TF_Name", "Family_Name", "DBDs", "pLI", "oe_lof_upper", "mis_z"]
].sort_values("pLI", ascending=False)
uni_table.to_csv("results/universal_tf_constraint_table.csv", index=False)
print("Saved → results/universal_tf_constraint_table.csv")

# ── 9. per-motif pLI coverage: universal vs control ────────────────────────
CONTROL_MEME = (
    ENHANCER_ROOT + "/multi_agent/posthoc_analysis/"
    "results/control_motifs.meme"
)

# build lookup: TF_Name_upper → has pLI score
pli_lookup = merged.set_index("TF_Name_upper")["pLI"].notna()

print("\n── Per-motif pLI coverage ──────────────────────────────────────────────")

# universal motifs: assignments JSON maps motif_id → [TF names]
print("\nUniversal motifs (43 total):")
uni_coverage = []
for motif_id, tf_list in assignments.items():
    tfs_upper  = [t.upper() for t in tf_list]
    n_tfs      = len(tfs_upper)
    n_with_pli = sum(pli_lookup.get(t, False) for t in tfs_upper)
    uni_coverage.append({
        "motif_id": motif_id,
        "n_jaspar_tfs": n_tfs,
        "n_with_pLI": n_with_pli,
    })
    print(f"  {motif_id}: {n_with_pli}/{n_tfs} TFs have pLI")

uni_cov_df = pd.DataFrame(uni_coverage)
n_uni_motifs_any = (uni_cov_df["n_with_pLI"] > 0).sum()
all_uni_tfs   = {t.upper() for v in assignments.values() for t in v}
uni_with_pli  = sum(pli_lookup.get(t, False) for t in all_uni_tfs)
print(f"\n  → {n_uni_motifs_any}/43 universal motifs have ≥1 TF with pLI score")
print(f"  → {uni_with_pli}/{len(all_uni_tfs)} unique TFs have pLI score")
print(f"     (note: 324 total assignments include {324 - len(all_uni_tfs)} duplicates "
      f"where the same TF matches multiple motifs)")

# control motifs: extract TF name from MEME motif line (strip trailing _N suffix)
print("\nControl motifs:")
ctrl_names = []
with open(CONTROL_MEME) as fh:
    for line in fh:
        if line.startswith("MOTIF"):
            motif_name = line.split()[1]
            # strip trailing _1, _2, ... suffix
            tf_name = re.sub(r"_\d+$", "", motif_name).upper()
            ctrl_names.append((motif_name, tf_name))

ctrl_coverage = []
for motif_name, tf_upper in ctrl_names:
    has_pli = bool(pli_lookup.get(tf_upper, False))
    ctrl_coverage.append({
        "motif_id": motif_name,
        "tf_name": tf_upper,
        "has_pLI": has_pli,
    })
    print(f"  {motif_name} → {tf_upper}: {'✓' if has_pli else '✗'}")

ctrl_cov_df = pd.DataFrame(ctrl_coverage)
n_ctrl_motifs_any = ctrl_cov_df["has_pLI"].sum()
print(f"\n  → {n_ctrl_motifs_any}/{len(ctrl_cov_df)} control motifs have pLI score")

# save coverage tables
uni_cov_df.to_csv("results/universal_motif_pli_coverage.csv", index=False)
ctrl_cov_df.to_csv("results/control_motif_pli_coverage.csv", index=False)
print("\nSaved → results/universal_motif_pli_coverage.csv")
print("Saved → results/control_motif_pli_coverage.csv")

print("\nDone.")
