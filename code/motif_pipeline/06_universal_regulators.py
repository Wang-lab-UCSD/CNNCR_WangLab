"""
Step 6 — Universal Regulator Identification

After the cross-cell-line self-TOMTOM (step 5), this script:
  1. Groups all cell-line representative motifs using the same network-distance
     clustering algorithm used in step 3, but with more lenient thresholds to
     allow cross-cell-line grouping.
  2. Counts how many unique cell lines are represented in each group (motif names
     carry the source cell line as the prefix before "_feature_").
  3. Keeps only groups represented in >= MIN_CELL_LINES unique cell lines.
     These are "universal regulators" — transcription factors with conserved
     binding motifs active across a broad range of cell types.
  4. Selects one representative PWM per universal group.
  5. Writes the final motif set and a diagnostic coverage plot.

Output: results/universal_regulator_motifs.meme
        results/plots/universal_group_coverage.png

Run: python 06_universal_regulators.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for cluster use
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils import (
    parse_meme_file, write_meme_file, read_tomtom,
    group_by_network, normalize_orientations, representative_motif,
)

# ── configuration ─────────────────────────────────────────────────────────────
ALL_FEATURES_MEME = os.path.join(SCRIPT_DIR, "results", "all_feature_motifs.meme")
SELF_TOMTOM_FILE  = os.path.join(SCRIPT_DIR, "results", "all_feature_self_tomtom.txt")
OUTPUT_MEME       = os.path.join(SCRIPT_DIR, "results", "universal_regulator_motifs.meme")
PLOTS_DIR         = os.path.join(SCRIPT_DIR, "results", "plots")

# Minimum number of unique cell lines a motif group must span to be considered
# a universal regulator.  Increase to require broader coverage.
MIN_CELL_LINES = 25

# TOMTOM thresholds for cross-cell-line grouping.
# More lenient than the within-cell step to allow grouping of related TF families.
E_VALUE_THRESHOLD = 0.5
P_VALUE_THRESHOLD = 0.01
DIST_THRESHOLD    = 0.5

# ── main ──────────────────────────────────────────────────────────────────────
os.makedirs(PLOTS_DIR, exist_ok=True)

for path in (ALL_FEATURES_MEME, SELF_TOMTOM_FILE):
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found — run previous steps first.")

# Load all cross-cell-line representative motifs
all_motifs, all_names = parse_meme_file(ALL_FEATURES_MEME)
name_to_idx = {n: i for i, n in enumerate(all_names)}
print(f"[06] Loaded {len(all_motifs)} motifs from {len(all_names)} cell-line features")

# Load TOMTOM results; drop exact self-matches (query == target)
tomtom_df = read_tomtom(SELF_TOMTOM_FILE, E_VALUE_THRESHOLD, P_VALUE_THRESHOLD)
tomtom_df = tomtom_df[tomtom_df["Query_ID"] != tomtom_df["Target_ID"]]

# Cluster motifs across all cell lines
motif_groups = group_by_network(tomtom_df, DIST_THRESHOLD)
print(f"[06] {len(motif_groups)} motif groups after cross-cell-line clustering")

# Count unique cell lines per group (cell line = prefix before "_feature_")
def cell_lines_in_group(group):
    return {name.split("_feature_")[0] for name in group}

# ── Diagnostic plot ───────────────────────────────────────────────────────────
cell_line_counts = sorted(
    (len(cell_lines_in_group(g)) for g in motif_groups), reverse=True
)
plt.figure(figsize=(10, 4))
plt.plot(range(1, len(cell_line_counts) + 1), cell_line_counts, linewidth=1)
plt.axhline(MIN_CELL_LINES, color="red", linestyle="--",
            label=f"MIN_CELL_LINES = {MIN_CELL_LINES}")
plt.xlabel("Group rank (largest to smallest)")
plt.ylabel("Unique cell lines in group")
plt.title("Cross-cell-line motif group coverage")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, "universal_group_coverage.png")
plt.savefig(plot_path, dpi=200)
plt.close()
print(f"[06] Coverage plot → {plot_path}")

# ── Select universal regulators and representative motifs ─────────────────────
universal_motifs, universal_names = [], []
for gidx, group in enumerate(motif_groups):
    if len(cell_lines_in_group(group)) < MIN_CELL_LINES:
        continue

    # Retrieve the actual PWM arrays for all group members
    pwms = [all_motifs[name_to_idx[n]] for n in group if n in name_to_idx]
    if not pwms:
        continue

    # Orient all motifs to the same strand, then pick the one nearest the median
    pwms = normalize_orientations(pwms)
    universal_motifs.append(representative_motif(pwms))
    universal_names.append(f"universal_{gidx}")

write_meme_file(universal_motifs, OUTPUT_MEME, universal_names)
print(f"[06] {len(universal_motifs)} universal regulator motifs "
      f"(>= {MIN_CELL_LINES} cell lines)")
print(f"     → {OUTPUT_MEME}")
